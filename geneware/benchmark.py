"""Genuine performance benchmarking for GeneWeaver (Week 1).

Every number reported here is measured at runtime with
``time.perf_counter_ns()`` and ``tracemalloc``. Nothing is estimated,
hard-coded, or extrapolated. Parsing time, alignment time, and total
time are measured separately.

Memory note: ``tracemalloc`` tracks Python-level allocations only; it
does not capture memory inside C extensions (e.g. NumPy buffers) and
is not available on every platform. ``memory_peak_bytes`` is therefore
reported as ``None`` when tracemalloc cannot run.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .aligner import DEFAULT_PAM_PATTERNS, DEFAULT_STRANDS, Candidate, ScanStats, scan_records
from .fasta import FastaParseError, FastaRecord, iter_fasta


@dataclass
class BenchmarkResult:
    """Measured results of one end-to-end run."""

    fasta_path: str
    guide: str
    parse_ns: int
    align_ns: int
    total_ns: int
    records_parsed: int
    bases_processed: int
    windows_evaluated: int
    candidates_found: int
    chunks_scanned: int
    memory_peak_bytes: int | None
    notes: list[str] = field(default_factory=list)

    @property
    def total_seconds(self) -> float:
        return self.total_ns / 1e9

    @property
    def throughput_bases_per_sec(self) -> float:
        """Genomic bases scanned per second of *alignment* time."""
        if self.align_ns <= 0:
            return 0.0
        return self.bases_processed / (self.align_ns / 1e9)

    @property
    def throughput_windows_per_sec(self) -> float:
        if self.align_ns <= 0:
            return 0.0
        return self.windows_evaluated / (self.align_ns / 1e9)

    def to_dict(self) -> dict[str, object]:
        return {
            "fasta_path": self.fasta_path,
            "guide": self.guide,
            "parse_seconds": self.parse_ns / 1e9,
            "align_seconds": self.align_ns / 1e9,
            "total_seconds": self.total_seconds,
            "records_parsed": self.records_parsed,
            "bases_processed": self.bases_processed,
            "windows_evaluated": self.windows_evaluated,
            "candidates_found": self.candidates_found,
            "chunks_scanned": self.chunks_scanned,
            "throughput_bases_per_sec": self.throughput_bases_per_sec,
            "throughput_windows_per_sec": self.throughput_windows_per_sec,
            "memory_peak_bytes": self.memory_peak_bytes,
            "notes": self.notes,
        }


def _format_bytes(n: int | None) -> str:
    if n is None:
        return "n/a"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n} B"


def format_result(r: BenchmarkResult) -> str:
    """Human-readable summary of a benchmark result."""
    lines = [
        "GeneWeaver benchmark (CPU baseline)",
        f"  input          : {r.fasta_path}",
        f"  guide          : {r.guide}",
        f"  parse time     : {r.parse_ns / 1e9:.6f} s",
        f"  align time     : {r.align_ns / 1e9:.6f} s",
        f"  total time     : {r.total_seconds:.6f} s",
        f"  records        : {r.records_parsed}",
        f"  bases scanned  : {r.bases_processed:,}",
        f"  windows scored : {r.windows_evaluated:,}",
        f"  candidates     : {r.candidates_found:,}",
        f"  chunks scanned : {r.chunks_scanned:,}",
        f"  throughput     : {r.throughput_bases_per_sec:,.0f} bases/s "
        f"({r.throughput_windows_per_sec:,.0f} windows/s)",
        f"  peak memory    : {_format_bytes(r.memory_peak_bytes)}",
    ]
    if r.notes:
        lines.append("  notes          :")
        lines += [f"    - {n}" for n in r.notes]
    return "\n".join(lines)


def run_benchmark(
    fasta_path: str | Path,
    guide: str,
    *,
    chunk_size: int = 1_000_000,
    overlap: int = 64,
    pam_patterns: tuple[str, ...] = DEFAULT_PAM_PATTERNS,
    strands: tuple[str, ...] = DEFAULT_STRANDS,
    max_mismatches: int = 0,
    require_pam: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[BenchmarkResult, list[Candidate], ScanStats]:
    """Run a full parse -> chunk -> align pipeline with real timing.

    Returns ``(result, candidates, scan_stats)``. ``candidates`` are the
    deduplicated hits; ``scan_stats`` carries the raw counters.
    """
    fasta_path = Path(fasta_path)
    notes: list[str] = []

    # --- parse phase (timed separately) ---
    t0 = time.perf_counter_ns()
    records: list[FastaRecord] = []
    try:
        for rec in iter_fasta(fasta_path):
            records.append(rec)
    except FastaParseError:
        raise
    t_parse = time.perf_counter_ns() - t0

    # --- alignment phase (timed separately) ---
    tracemalloc_started = False
    try:
        tracemalloc.start()
        tracemalloc_started = True
    except Exception:
        notes.append("tracemalloc unavailable; memory_peak_bytes is None")

    t1 = time.perf_counter_ns()
    candidates, scan_stats = scan_records(
        records,
        guide,
        chunk_size=chunk_size,
        overlap=overlap,
        pam_patterns=pam_patterns,
        strands=strands,
        max_mismatches=max_mismatches,
        require_pam=require_pam,
        progress=progress,
    )
    t_align = time.perf_counter_ns() - t1
    t_total = time.perf_counter_ns() - t0

    _, peak = tracemalloc.get_traced_memory() if tracemalloc_started else (0, None)
    if tracemalloc_started:
        tracemalloc.stop()

    result = BenchmarkResult(
        fasta_path=str(fasta_path),
        guide=guide,
        parse_ns=t_parse,
        align_ns=t_align,
        total_ns=t_total,
        records_parsed=len(records),
        bases_processed=scan_stats.bases_processed,
        windows_evaluated=scan_stats.windows_evaluated,
        candidates_found=scan_stats.candidates_found,
        chunks_scanned=scan_stats.chunks_scanned,
        memory_peak_bytes=peak,
        notes=notes,
    )
    return result, candidates, scan_stats
