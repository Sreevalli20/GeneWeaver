"""Tests for the unified pipeline (cpu backend, scoring, benchmark)."""

from pathlib import Path

import pytest

from geneware.pipeline import AnalysisError, run_analysis, run_benchmark

EXAMPLE = Path(__file__).resolve().parent.parent / "data" / "example.fasta"
GUIDE = "ACGTACGTACGTACGTACGT"


def test_run_analysis_cpu():
    r = run_analysis(EXAMPLE, GUIDE, backend="cpu",
                     chunk_size=200, overlap=30)
    assert r.backend == "cpu"
    assert r.stats.bases_processed > 0
    assert r.score_summary["count"] == len(r.candidates)
    assert all(0 <= s.score <= 100 for s in r.scored)


def test_run_analysis_scores_ranked():
    r = run_analysis(EXAMPLE, GUIDE, backend="cpu",
                     chunk_size=200, overlap=30)
    scores = [s.score for s in r.scored]
    assert scores == sorted(scores, reverse=True)


def test_unknown_backend_rejected():
    with pytest.raises(AnalysisError):
        run_analysis(EXAMPLE, GUIDE, backend="warp-drive")


def test_gpu_backend_rejected_when_unavailable():
    from geneware.gpu import gpu_available
    if gpu_available():
        pytest.skip("GPU available")
    with pytest.raises(AnalysisError):
        run_analysis(EXAMPLE, GUIDE, backend="gpu")


def test_run_benchmark_cpu():
    bench, analysis = run_benchmark(EXAMPLE, GUIDE, backend="cpu",
                                    chunk_size=200, overlap=30)
    assert bench.bases_processed > 0
    assert bench.align_ns >= 0
    assert bench.throughput_bases_per_sec > 0
    assert analysis.backend == "cpu"
    assert "backend: cpu" in bench.notes


def test_analysis_dict_serialisable():
    import json
    r = run_analysis(EXAMPLE, GUIDE, backend="cpu",
                     chunk_size=500, overlap=30)
    json.dumps(r.analysis_dict())  # must not raise
