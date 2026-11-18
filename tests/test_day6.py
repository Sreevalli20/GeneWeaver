"""Day 6 tests: auto backend selection, GPU VRAM fields, Dask failure handling."""

from pathlib import Path

import pytest

from geneware.gpu import detect_gpu
from geneware.pipeline import _resolve_auto_backend, run_analysis

EXAMPLE = Path(__file__).resolve().parent.parent / "data" / "example.fasta"
GUIDE = "ACGTACGTACGTACGTACGT"


def test_resolve_auto_backend_returns_known_backend():
    assert _resolve_auto_backend() in ("cpu", "gpu", "dask")


def test_auto_backend_matches_resolved():
    resolved = _resolve_auto_backend()
    result = run_analysis(EXAMPLE, GUIDE, backend="auto",
                          chunk_size=100_000, n_workers=1, processes=False)
    # result.backend is the concrete backend that actually ran
    assert result.backend == resolved
    assert result.stats.bases_processed > 0


def test_gpu_status_vram_fields():
    st = detect_gpu().to_dict()
    assert "vram_total_mb" in st and "vram_free_mb" in st
    if st["available"] and st["vram_free_mb"] is not None:
        assert 0 < st["vram_free_mb"] <= st["vram_total_mb"] + 1


def test_dask_worker_failures_are_captured():
    """A failing worker task must be recorded, not crash or fabricate."""
    import geneware.dask_engine as de
    from geneware.fasta import parse_fasta

    records = parse_fasta(EXAMPLE)

    def _boom(*a, **k):
        raise RuntimeError("simulated worker failure")

    orig = de._scan_chunk_task
    de._scan_chunk_task = _boom
    try:
        r = de.scan_records_dask(records, GUIDE, chunk_size=100_000,
                                 overlap=30, n_workers=1, processes=False)
    finally:
        de._scan_chunk_task = orig
    assert r.tasks_failed > 0
    assert r.errors and "simulated worker failure" in r.errors[0]
    assert r.candidates == []
    assert r.chunks_completed == 0
    assert r.chunks_pending == 0
    assert r.chunks_total == r.tasks_failed


def test_dask_pending_and_completed_fields():
    import geneware.dask_engine as de
    from geneware.fasta import parse_fasta
    records = parse_fasta(EXAMPLE)
    r = de.scan_records_dask(records, GUIDE, chunk_size=50_000,
                             overlap=30, n_workers=1, processes=False)
    assert r.chunks_completed + r.tasks_failed == r.chunks_total
    assert r.chunks_pending == 0
    # dedup: no duplicate (chr,start,end,strand) keys
    keys = [(c.chromosome, c.start, c.end, c.strand) for c in r.candidates]
    assert len(keys) == len(set(keys))
