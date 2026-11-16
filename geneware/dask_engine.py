"""Dask-based distributed chunk processing for GeneWeaver.

Design
------
The genome is already split into :class:`~geneware.chunking.GenomeChunk`
objects. Each chunk is an independent alignment task, so Dask distributes
them across worker processes. Results are collected as they complete,
statistics are merged, and candidates are deduplicated with the same
:func:`geneware.aligner.deduplicate` used by the sequential path — so
boundary-overlap duplicates are removed identically.

Honesty rules
-------------
* ``backend`` reported is always what actually ran: ``'dask-processes'``
  (separate worker processes) or ``'dask-threads'`` (in-process threads).
  There is no distributed-GPU claim: GPU workers are only used when the
  GPU backend is genuinely available *and* the caller selects them.
* Worker exceptions are captured per-task and surfaced in
  :attr:`DaskScanResult.errors`; a failed worker never fabricates results.
* ``dask_status`` reports real scheduler measurements (workers, cores,
  memory) or an honest unavailable reason.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .aligner import (
    DEFAULT_PAM_PATTERNS,
    DEFAULT_STRANDS,
    AlignError,
    Candidate,
    ScanStats,
    deduplicate,
    scan_chunk,
    validate_guide,
    validate_overlap,
    _normalise_patterns,
    _validate_scan_args,
)
from .chunking import chunk_records, total_chunks
from .fasta import FastaRecord


class DaskUnavailableError(RuntimeError):
    """Raised when Dask/distributed cannot be used."""


def _default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


@dataclass
class DaskScanResult:
    """Outcome of a Dask-distributed scan."""

    candidates: list[Candidate]
    stats: ScanStats
    backend: str                     # 'dask-processes' | 'dask-threads'
    chunks_total: int
    chunks_completed: int
    chunks_pending: int = 0          # tasks not yet finished (0 at end)
    tasks_failed: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "chunks_total": self.chunks_total,
            "chunks_completed": self.chunks_completed,
            "chunks_pending": self.chunks_pending,
            "tasks_failed": self.tasks_failed,
            "errors": self.errors,
            "stats": self.stats.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
        }


def _scan_chunk_task(
    chunk: Any,
    guide: str,
    pam_patterns: tuple[str, ...],
    strands: tuple[str, ...],
    max_mismatches: int,
    require_pam: bool,
) -> tuple[list[Candidate], ScanStats]:
    """Top-level (picklable) worker function: scan one chunk on a CPU."""
    stats = ScanStats()
    cands = list(
        scan_chunk(
            chunk,
            guide,
            pam_patterns=pam_patterns,
            strands=strands,
            max_mismatches=max_mismatches,
            require_pam=require_pam,
            stats=stats,
        )
    )
    return cands, stats


def _merge_stats(a: ScanStats, b: ScanStats) -> ScanStats:
    a.bases_processed += b.bases_processed
    a.windows_evaluated += b.windows_evaluated
    a.candidates_found += b.candidates_found
    a.chunks_scanned += b.chunks_scanned
    return a


def _make_client(n_workers: int | None, processes: bool):
    """Create a dask.distributed Client; threads fallback on failure.

    Returns ``(client, backend, n_workers)`` so callers can size their
    in-flight task window from the *actual* worker count.
    """
    try:
        from distributed import Client, LocalCluster
    except Exception as exc:
        raise DaskUnavailableError(
            f"dask/distributed not importable: {type(exc).__name__}: {exc}"
        ) from exc

    workers = n_workers or _default_workers()
    backend = "dask-processes" if processes else "dask-threads"
    try:
        if processes:
            cluster = LocalCluster(
                n_workers=workers, threads_per_worker=1,
                processes=True, dashboard_address=None,
            )
            return Client(cluster), backend, workers
        return (Client(n_workers=workers, threads_per_worker=1,
                       processes=False), backend, workers)
    except Exception:
        if processes:
            # Process-based clusters can fail on restricted Windows setups;
            # fall back to threads honestly and report the real backend.
            return (Client(n_workers=workers, threads_per_worker=1,
                           processes=False), "dask-threads", workers)
        raise


def scan_records_dask(
    records: list[FastaRecord],
    guide: str,
    *,
    chunk_size: int = 1_000_000,
    overlap: int = 64,
    pam_patterns: Iterable[str] = DEFAULT_PAM_PATTERNS,
    strands: Iterable[str] = DEFAULT_STRANDS,
    max_mismatches: int = 0,
    require_pam: bool = True,
    n_workers: int | None = None,
    processes: bool = True,
    client: Any | None = None,
    progress: Any | None = None,
) -> DaskScanResult:
    """Scan records by distributing chunks across Dask workers.

    ``progress(done, total)`` is called as futures complete.
    ``client`` may be supplied to reuse an existing cluster.
    """
    guide = validate_guide(guide)
    records = list(records)
    validate_overlap(records, guide, chunk_size, overlap, pam_patterns)
    pams = tuple(_normalise_patterns(pam_patterns))
    strand_tuple = tuple(strands)
    _validate_scan_args(pams, strand_tuple)
    if max_mismatches < 0:
        raise AlignError(f"max_mismatches must be >= 0, got {max_mismatches}")

    n_total = total_chunks(records, chunk_size, overlap)

    own_client = client is None
    if own_client:
        client, backend, workers = _make_client(n_workers, processes)
    else:
        backend = "dask-external"
        workers = n_workers or _default_workers()

    stats = ScanStats()
    all_candidates: list[Candidate] = []
    done = 0
    failed = 0
    errors: list[str] = []

    try:
        from dask.distributed import as_completed

        # Rolling submission window: at most ~4 tasks per worker are kept
        # in flight, so peak memory stays bounded (each pending future
        # pins one chunk's sequence) instead of materialising every chunk
        # future up front. Chunks are still produced lazily.
        window = max(8, workers * 4)
        chunk_iter = iter(chunk_records(records, chunk_size, overlap))
        exhausted = False
        futures: list[Any] = []

        def _submit() -> bool:
            nonlocal exhausted
            if exhausted:
                return False
            try:
                ch = next(chunk_iter)
            except StopIteration:
                exhausted = True
                return False
            futures.append(client.submit(
                _scan_chunk_task, ch, guide, pams, strand_tuple,
                max_mismatches, require_pam,
                pure=False,
            ))
            return True

        while True:
            while len(futures) < window and _submit():
                pass
            if not futures:
                break
            # as_completed yields each future as it finishes; topping the
            # window back up inside the loop keeps workers saturated.
            for fut in as_completed(list(futures)):
                futures.remove(fut)
                try:
                    cands, cstats = fut.result()
                except Exception as exc:  # worker failure: record, continue
                    failed += 1
                    errors.append(f"{type(exc).__name__}: {exc}")
                else:
                    all_candidates.extend(cands)
                    _merge_stats(stats, cstats)
                done += 1
                if progress is not None:
                    progress(done, max(n_total, 1))
                while len(futures) < window and _submit():
                    pass
    finally:
        if own_client:
            client.close()

    unique = deduplicate(all_candidates)
    unique.sort(key=lambda c: (c.chromosome, c.start, c.end, c.strand))
    stats.candidates_found = len(unique)
    return DaskScanResult(
        candidates=unique,
        stats=stats,
        backend=backend,
        chunks_total=n_total,
        chunks_completed=done - failed,
        chunks_pending=max(0, n_total - done),
        tasks_failed=failed,
        errors=errors,
    )


def dask_status(client: Any | None = None) -> dict[str, object]:
    """Real scheduler status: workers, cores, memory — or honest failure."""
    own = client is None
    try:
        if own:
            client, _, _ = _make_client(None, processes=False)
        info = client.scheduler_info()
        workers = info.get("workers", {})
        n_workers = len(workers)
        cores = sum(
            (w.get("nthreads") or w.get("nthreads", 0)) for w in workers.values()
        ) if isinstance(workers, dict) else 0
        mem = sum(
            (w.get("memory") or 0) for w in workers.values()
        ) if isinstance(workers, dict) else 0
        return {
            "available": True,
            "workers": n_workers,
            "total_threads": cores,
            "total_memory_bytes": mem,
            "scheduler_address": str(getattr(client, "scheduler.address", "")),
        }
    except Exception as exc:
        return {
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
        }
    finally:
        if own and client is not None:
            try:
                client.close()
            except Exception:
                pass
