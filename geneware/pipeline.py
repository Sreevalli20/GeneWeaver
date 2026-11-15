"""Unified analysis pipeline for GeneWeaver.

One entry point — :func:`run_analysis` — that executes the full
parse -> scan -> score -> report-data flow on a chosen backend:

* ``'cpu'``   — sequential :func:`geneware.aligner.scan_records`
* ``'gpu'``   — :func:`geneware.gpu.scan_gpu` (requires working CUDA;
                raises rather than silently falling back)
* ``'dask'``  — :func:`geneware.dask_engine.scan_records_dask`

and :func:`run_benchmark` which measures any of those backends with the
same real ``perf_counter``/``tracemalloc`` instrumentation as the Week 1
CPU benchmark.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .aligner import (
    DEFAULT_PAM_PATTERNS,
    DEFAULT_STRANDS,
    Candidate,
    ScanStats,
    scan_records,
)
from .benchmark import BenchmarkResult
from .chunking import total_chunks
from .fasta import FastaParseError, FastaRecord, compute_stats, iter_fasta
from .scoring import ScoredCandidate, score_candidates, score_summary

BACKENDS = ("cpu", "gpu", "dask")


class AnalysisError(RuntimeError):
    """Raised for invalid analysis configuration."""


@dataclass
class AnalysisResult:
    """Complete outcome of one analysis run (all real data)."""

    backend: str
    guide: str
    pam_patterns: tuple[str, ...]
    strands: tuple[str, ...]
    max_mismatches: int
    require_pam: bool
    fasta_path: str
    records: list[FastaRecord]
    sequence_stats: dict[str, Any]
    candidates: list[Candidate]
    scored: list[ScoredCandidate]
    stats: ScanStats
    score_summary: dict[str, Any]
    elapsed_seconds: float
    extra: dict[str, Any] = field(default_factory=dict)  # dask/gpu details

    def analysis_dict(self) -> dict[str, Any]:
        """JSON-serialisable analysis payload for reports/exports."""
        return {
            "backend": self.backend,
            "guide": self.guide,
            "pam_patterns": list(self.pam_patterns),
            "strands": list(self.strands),
            "max_mismatches": self.max_mismatches,
            "require_pam": self.require_pam,
            "fasta_path": self.fasta_path,
            "elapsed_seconds": self.elapsed_seconds,
            "sequence_stats": self.sequence_stats,
            "scan_stats": self.stats.to_dict(),
            "score_summary": self.score_summary,
            "score_model": "geneware-heuristic-v1 (experimental, not validated)",
            "extra": self.extra,
            "candidates": [s.to_dict() for s in self.scored],
        }


def _scan_for_backend(
    backend: str,
    records: list[FastaRecord],
    guide: str,
    *,
    chunk_size: int,
    overlap: int,
    pam_patterns: Iterable[str],
    strands: Iterable[str],
    max_mismatches: int,
    require_pam: bool,
    n_workers: int | None = None,
    processes: bool = True,
    client: Any | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[Candidate], ScanStats, dict[str, Any]]:
    """Dispatch to the requested backend. Returns (cands, stats, extra)."""
    kwargs = dict(
        chunk_size=chunk_size, overlap=overlap,
        pam_patterns=pam_patterns, strands=strands,
        max_mismatches=max_mismatches, require_pam=require_pam,
    )
    n_chunks = total_chunks(records, chunk_size, overlap)
    if backend == "cpu":
        cands, stats = scan_records(records, guide, progress=progress, **kwargs)
        return cands, stats, {
            "workers": 1, "mode": "sequential",
            "chunks_total": n_chunks, "chunks_completed": stats.chunks_scanned,
        }
    if backend == "gpu":
        from .gpu import detect_gpu, scan_gpu
        status = detect_gpu()
        if not status.available:
            raise AnalysisError(
                "GPU backend requested but unavailable: " + status.reason
            )
        cands, stats = scan_gpu(records, guide, progress=progress, **kwargs)
        return cands, stats, {
            "gpu": status.to_dict(),
            "chunks_total": n_chunks, "chunks_completed": stats.chunks_scanned,
        }
    if backend == "dask":
        from .dask_engine import scan_records_dask
        result = scan_records_dask(
            records, guide, n_workers=n_workers, processes=processes,
            client=client, progress=progress, **kwargs,
        )
        extra = {
            "dask_backend": result.backend,
            "chunks_total": result.chunks_total,
            "chunks_completed": result.chunks_completed,
            "chunks_pending": result.chunks_pending,
            "tasks_failed": result.tasks_failed,
            "errors": result.errors,
        }
        return result.candidates, result.stats, extra
    raise AnalysisError(f"unknown backend {backend!r}; choose cpu|gpu|dask")


def run_analysis(
    fasta_path: str | Path,
    guide: str,
    *,
    backend: str = "cpu",
    chunk_size: int = 1_000_000,
    overlap: int = 64,
    pam_patterns: Iterable[str] = DEFAULT_PAM_PATTERNS,
    strands: Iterable[str] = DEFAULT_STRANDS,
    max_mismatches: int = 0,
    require_pam: bool = True,
    n_workers: int | None = None,
    processes: bool = True,
    client: Any | None = None,
    records: list[FastaRecord] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> AnalysisResult:
    """Full analysis: parse FASTA, scan on backend, score, summarise."""
    if backend not in BACKENDS:
        raise AnalysisError(f"unknown backend {backend!r}; choose cpu|gpu|dask")
    fasta_path = Path(fasta_path)
    pams = tuple("".join(p.split()).upper() for p in pam_patterns)
    strand_tuple = tuple(strands)

    if records is None:
        records = list(iter_fasta(fasta_path))
    seq_stats = compute_stats(records).to_dict()

    # Measure real peak Python memory for the scan when tracemalloc is
    # not already running (run_benchmark starts its own trace, in which
    # case the benchmark-level peak is the authoritative measurement).
    tracing_here = False
    if not tracemalloc.is_tracing():
        try:
            tracemalloc.start()
            tracing_here = True
        except Exception:
            pass

    t0 = time.perf_counter()
    try:
        cands, stats, extra = _scan_for_backend(
            backend, records, guide,
            chunk_size=chunk_size, overlap=overlap,
            pam_patterns=pams, strands=strand_tuple,
            max_mismatches=max_mismatches, require_pam=require_pam,
            n_workers=n_workers, processes=processes, client=client,
            progress=progress,
        )
    except MemoryError as exc:
        raise AnalysisError(
            f"out of memory during '{backend}' scan: {exc}. "
            "Try a smaller --chunk-size or fewer/smaller input records."
        ) from exc
    finally:
        mem_peak = None
        if tracing_here:
            _, mem_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
    elapsed = time.perf_counter() - t0
    extra["memory_peak_bytes"] = mem_peak

    scored = score_candidates(cands)
    return AnalysisResult(
        backend=backend,
        guide=guide,
        pam_patterns=pams,
        strands=strand_tuple,
        max_mismatches=max_mismatches,
        require_pam=require_pam,
        fasta_path=str(fasta_path),
        records=records,
        sequence_stats=seq_stats,
        candidates=cands,
        scored=scored,
        stats=stats,
        score_summary=score_summary(scored),
        elapsed_seconds=elapsed,
        extra=extra,
    )


def run_benchmark(
    fasta_path: str | Path,
    guide: str,
    *,
    backend: str = "cpu",
    chunk_size: int = 1_000_000,
    overlap: int = 64,
    pam_patterns: Iterable[str] = DEFAULT_PAM_PATTERNS,
    strands: Iterable[str] = DEFAULT_STRANDS,
    max_mismatches: int = 0,
    require_pam: bool = True,
    n_workers: int | None = None,
    processes: bool = True,
    client: Any | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[BenchmarkResult, AnalysisResult]:
    """Timed end-to-end benchmark for any backend.

    Returns ``(benchmark, analysis)``. All timings are real
    ``perf_counter`` measurements; memory is real ``tracemalloc`` peak.
    """
    fasta_path = Path(fasta_path)
    notes: list[str] = []

    t0 = time.perf_counter_ns()
    records = list(iter_fasta(fasta_path))
    t_parse = time.perf_counter_ns() - t0

    tracemalloc_started = False
    try:
        tracemalloc.start()
        tracemalloc_started = True
    except Exception:
        notes.append("tracemalloc unavailable; memory_peak_bytes is None")

    t1 = time.perf_counter_ns()
    analysis = run_analysis(
        fasta_path, guide, backend=backend,
        chunk_size=chunk_size, overlap=overlap,
        pam_patterns=pam_patterns, strands=strands,
        max_mismatches=max_mismatches, require_pam=require_pam,
        n_workers=n_workers, processes=processes, client=client,
        records=records, progress=progress,
    )
    # analysis.elapsed_seconds is the scan-only wall time.
    t_align = int(analysis.elapsed_seconds * 1e9)
    t_total = time.perf_counter_ns() - t0

    _, peak = tracemalloc.get_traced_memory() if tracemalloc_started else (0, None)
    if tracemalloc_started:
        tracemalloc.stop()

    stats = analysis.stats
    result = BenchmarkResult(
        fasta_path=str(fasta_path),
        guide=guide,
        backend=backend,
        parse_ns=t_parse,
        align_ns=t_align,
        total_ns=t_total,
        records_parsed=len(records),
        bases_processed=stats.bases_processed,
        windows_evaluated=stats.windows_evaluated,
        candidates_found=stats.candidates_found,
        chunks_scanned=stats.chunks_scanned,
        memory_peak_bytes=peak,
        notes=notes + [f"backend: {backend}"] + [
            f"{k}={v}" for k, v in analysis.extra.items()
            if k in ("dask_backend", "tasks_failed")
        ],
    )
    return result, analysis
