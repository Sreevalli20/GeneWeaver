"""Day-4 tests: fast CPU engine parity, overlap validation, large-input
handling, stats correctness, dask batching, memory-error surfacing.

All sequences are small/medium synthetic TEST FIXTURES.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from geneware.aligner import (
    AlignError,
    ScanStats,
    scan_chunk,
    scan_records,
)
from geneware.chunking import GenomeChunk
from geneware.fasta import FastaRecord, compute_stats, parse_fasta

GUIDE = "ACGTACGTACGTACGTACGT"  # 20 nt
PAM = "AGG"


def _chunk(seq: str, chrom: str = "chrX", start: int = 0) -> GenomeChunk:
    return GenomeChunk(chromosome=chrom, chunk_index=0, start=start,
                       end=start + len(seq), sequence=seq)


def _key(c):
    return (c.chromosome, c.start, c.end, c.strand, c.mismatches,
            c.mismatch_positions, c.pam, c.pam_compatible,
            c.matched_sequence, c.genomic_sequence, c.ambiguous_positions)


def _scan_both_engines(seq: str, guide: str = GUIDE, **kw):
    """Scan the same chunk with numba engine and forced-python engine."""
    kw.setdefault("max_mismatches", 2)
    kw.setdefault("pam_patterns", ("NGG",))
    kw.setdefault("strands", ("+", "-"))
    with_numba = list(scan_chunk(_chunk(seq), guide, **kw))
    import os
    os.environ["GENEWEARE_ALIGNER_ENGINE"] = "python"
    try:
        py = list(scan_chunk(_chunk(seq), guide, **kw))
    finally:
        del os.environ["GENEWEARE_ALIGNER_ENGINE"]
    return with_numba, py


# ---------- numba/python parity ----------

def test_numba_and_python_engines_identical_on_mixed_sequence():
    from geneware.aligner import revcomp
    rng = random.Random(20261115)
    seq = "".join(rng.choice("ACGT") for _ in range(600))
    # plant a forward hit (guide+PAM) and a reverse-strand hit
    # (revcomp(guide) followed by revcomp(PAM) on the plus strand)
    seq = seq[:100] + GUIDE + PAM + seq[123:400]
    # minus-strand PAM sits 5' of the window on plus coords: the 3 bases
    # before the revcomp(guide) window must revcomp to a NGG-compatible PAM
    seq = seq[:427] + revcomp(PAM) + revcomp(GUIDE) + seq[450:]
    a, b = _scan_both_engines(seq)
    assert sorted(map(_key, a)) == sorted(map(_key, b))
    assert len(a) > 0


def test_numba_python_parity_with_ambiguity_codes():
    seq = ("ACGT" * 30 + "NRYWSKMBDHV" + GUIDE + PAM + "ACGT" * 20
           + GUIDE + "CCT" + "TGCA" * 15)
    a, b = _scan_both_engines(seq, max_mismatches=4)
    assert sorted(map(_key, a)) == sorted(map(_key, b))


def test_scan_stats_equal_across_engines():
    seq = "".join(random.Random(7).choice("ACGT") for _ in range(2000))
    seq = seq[:50] + GUIDE + PAM + seq[80:]
    st1, st2 = ScanStats(), ScanStats()
    list(scan_chunk(_chunk(seq), GUIDE, stats=st1))
    import os
    os.environ["GENEWEARE_ALIGNER_ENGINE"] = "python"
    try:
        list(scan_chunk(_chunk(seq), GUIDE, stats=st2))
    finally:
        del os.environ["GENEWEARE_ALIGNER_ENGINE"]
    assert st1.to_dict() == st2.to_dict()


def test_engine_env_python_disables_numba():
    import os
    os.environ["GENEWEARE_ALIGNER_ENGINE"] = "python"
    try:
        from geneware.aligner import _get_numba_kernel
        assert _get_numba_kernel() is None
    finally:
        del os.environ["GENEWEARE_ALIGNER_ENGINE"]


# ---------- overlap / chunking validation ----------

def _rec(seq: str, rid: str = "chrV") -> FastaRecord:
    return FastaRecord(record_id=rid, description="t", sequence=seq)


def test_overlap_too_small_raises_for_multichunk_record():
    seq = "ACGT" * 400  # 1600 bases > chunk_size
    with pytest.raises(AlignError, match="overlap"):
        scan_records([_rec(seq)], GUIDE, chunk_size=500, overlap=10)


def test_overlap_too_small_ok_when_single_chunk():
    seq = "ACGT" * 50  # 200 bases <= chunk_size
    cands, stats = scan_records([_rec(seq)], GUIDE, chunk_size=500,
                                overlap=0)
    assert stats.chunks_scanned == 1


def test_overlap_must_be_below_chunk_size():
    with pytest.raises(AlignError, match="overlap"):
        scan_records([_rec("A" * 100)], GUIDE, chunk_size=100, overlap=100)


def test_boundary_hit_found_with_valid_overlap():
    # guide+PAM straddles the chunk boundary; must be found exactly once.
    W = len(GUIDE) + 3
    chunk_size, overlap = 300, 64
    pad = chunk_size - W + 5  # hit starts 5 bases before boundary
    seq = "A" * pad + GUIDE + PAM + "C" * (chunk_size * 2)
    cands, stats = scan_records([_rec(seq)], GUIDE, chunk_size=chunk_size,
                                overlap=overlap)
    hits = [c for c in cands if c.strand == "+" and c.pam == PAM]
    assert len(hits) >= 1
    keys = [(c.start, c.end, c.strand) for c in hits]
    assert len(keys) == len(set(keys))


# ---------- large-input handling ----------

def test_large_multi_record_fasta(tmp_path):
    rng = random.Random(99)
    lines = []
    total = 0
    for r in range(3):
        lines.append(f">chrL{r} large record {r}")
        seq = "".join(rng.choice("ACGT") for _ in range(120_000))
        for j in range(0, len(seq), 60):
            lines.append(seq[j:j + 60])
        # plant a guide+PAM in each record
        lines[-1] = lines[-1][:30] + GUIDE + PAM + lines[-1][53:]
        total += len(seq)
    p = tmp_path / "large.fasta"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    records = parse_fasta(p)
    assert sum(r.length for r in records) == total
    progress_calls = []
    cands, stats = scan_records(
        records, GUIDE, chunk_size=100_000, overlap=64,
        progress=lambda d, t: progress_calls.append((d, t)))
    # bases_processed counts chunk bases (with overlap), >= total
    assert stats.bases_processed >= total
    assert stats.chunks_scanned == len(progress_calls) == \
        progress_calls[-1][1] > 1
    keys = [(c.chromosome, c.start, c.end, c.strand) for c in cands]
    assert len(keys) == len(set(keys)), "duplicate candidates reported"
    # each planted hit found exactly once per record
    for r in range(3):
        rec_hits = [c for c in cands if c.chromosome == f"chrL{r}"
                    and c.strand == "+" and c.pam == PAM]
        assert len(rec_hits) >= 1


def test_compute_stats_matches_naive_counting(tmp_path):
    seq = ("ACGTN" * 1000) + ("RYSWKMBDHV" * 100)
    p = tmp_path / "s.fasta"
    p.write_text(f">chrS\n{seq}\n", encoding="utf-8")
    records = parse_fasta(p)
    stats = compute_stats(records)
    assert stats.total_bases == len(seq)
    assert stats.base_counts["A"] == seq.count("A")
    assert stats.ambiguous_count == sum(
        1 for b in seq if b not in "ACGT")
    # GC% only counts G and C in numerator
    assert stats.gc_percent == round(
        100.0 * (seq.count("G") + seq.count("C")) / len(seq), 4)


def test_scan_chunk_stats_parameter_still_works():
    st = ScanStats()
    list(scan_chunk(_chunk("A" * 60), GUIDE, stats=st))
    assert st.bases_processed == 60
    assert st.windows_evaluated > 0


# ---------- dask batching ----------

def test_dask_scan_small_chunk_count():
    from geneware.dask_engine import scan_records_dask
    example = Path(__file__).resolve().parent.parent / "data" / "example.fasta"
    recs = parse_fasta(example)
    r = scan_records_dask(recs, GUIDE, chunk_size=120, overlap=30,
                          n_workers=1, processes=False)
    assert r.chunks_completed == r.chunks_total
    assert r.chunks_pending == 0
    assert r.tasks_failed == 0
    keys = [(c.chromosome, c.start, c.end, c.strand) for c in r.candidates]
    assert len(keys) == len(set(keys))


def test_dask_result_pending_field():
    from geneware.dask_engine import DaskScanResult
    import inspect
    sig = inspect.signature(DaskScanResult)
    assert "chunks_pending" in sig.parameters


# ---------- pipeline memory / errors ----------

def test_run_analysis_reports_memory_and_chunks(tmp_path):
    from geneware.pipeline import run_analysis
    seq = "".join(random.Random(3).choice("ACGT") for _ in range(50_000))
    seq = seq[:1000] + GUIDE + PAM + seq[1023:]
    p = tmp_path / "m.fasta"
    p.write_text(f">chrM\n{seq}\n", encoding="utf-8")
    r = run_analysis(p, GUIDE, backend="cpu", chunk_size=20_000,
                     overlap=64)
    assert r.extra["chunks_total"] == r.stats.chunks_scanned
    assert r.extra["chunks_completed"] == r.stats.chunks_scanned
    # tracemalloc measures real Python allocations on CPython
    assert r.extra["memory_peak_bytes"] is None or \
        r.extra["memory_peak_bytes"] > 0


def test_memory_error_becomes_analysis_error(tmp_path, monkeypatch):
    from geneware import pipeline
    p = tmp_path / "e.fasta"
    p.write_text(">chrE\n" + "ACGT" * 100 + "\n", encoding="utf-8")

    def _boom(*a, **kw):
        raise MemoryError("simulated")

    monkeypatch.setattr(pipeline, "_scan_for_backend", _boom)
    with pytest.raises(pipeline.AnalysisError, match="out of memory"):
        pipeline.run_analysis(p, GUIDE, backend="cpu")


def test_benchmark_result_backend_field(tmp_path):
    from geneware.pipeline import run_benchmark
    p = tmp_path / "b.fasta"
    p.write_text(">chrB\n" + "ACGT" * 200 + GUIDE + PAM + "TGCA" * 50 + "\n",
                 encoding="utf-8")
    bench, analysis = run_benchmark(p, GUIDE, backend="cpu",
                                    chunk_size=5_000, overlap=64)
    assert bench.backend == "cpu"
    assert bench.memory_peak_bytes is not None
    assert bench.bases_processed > 0
