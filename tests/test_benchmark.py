"""Tests for geneware.benchmark - verifies measured (not fabricated) output.

Test fixtures only - small synthetic sequences.
"""

from __future__ import annotations

from geneware.benchmark import format_result, run_benchmark

GUIDE = "ACGTACGTAC"

FASTA = """>chrB1 bench fixture
TTTTAAAACCCCGGGG
ACGTACGTACAGG
CCCCGGGGTTTTAAAA
"""


def _write(tmp_path):
    p = tmp_path / "bench.fasta"
    p.write_text(FASTA, encoding="utf-8")
    return p


def test_benchmark_returns_real_measurements(tmp_path):
    p = _write(tmp_path)
    result, candidates, stats = run_benchmark(str(p), GUIDE)

    # timings are real nanosecond counters and must be positive
    assert result.parse_ns > 0
    assert result.align_ns > 0
    assert result.total_ns >= result.parse_ns + result.align_ns - 1_000_000
    assert result.total_seconds > 0

    # counters reflect the actual fixture
    assert result.records_parsed == 1
    assert result.bases_processed == len("TTTTAAAACCCCGGGGACGTACGTACAGGCCCCGGGGTTTTAAAA")
    assert result.windows_evaluated > 0
    assert result.candidates_found == len(candidates)
    assert result.chunks_scanned == 1

    # throughput derived from measured values, not hard-coded
    assert result.throughput_bases_per_sec > 0
    assert result.throughput_windows_per_sec > 0

    # memory measured via tracemalloc on CPython
    assert result.memory_peak_bytes is not None
    assert result.memory_peak_bytes > 0

    # the designed hit: guide at offset 16, followed by AGG
    hits = [c for c in candidates if c.strand == "+" and c.start == 16]
    assert len(hits) == 1
    assert hits[0].pam == "AGG"


def test_benchmark_format_result_contains_measurements(tmp_path):
    p = _write(tmp_path)
    result, _, _ = run_benchmark(str(p), GUIDE)
    text = format_result(result)
    for token in ("parse time", "align time", "total time",
                  "bases scanned", "windows scored", "throughput",
                  "peak memory", "candidates"):
        assert token in text


def test_benchmark_json_serialisable(tmp_path):
    import json
    p = _write(tmp_path)
    result, _, _ = run_benchmark(str(p), GUIDE)
    d = result.to_dict()
    json.dumps(d)  # must not raise
    assert d["records_parsed"] == 1
    assert isinstance(d["throughput_bases_per_sec"], float)


def test_benchmark_progress_callback(tmp_path):
    p = _write(tmp_path)
    calls = []
    run_benchmark(str(p), GUIDE,
                  progress=lambda done, total: calls.append((done, total)))
    assert calls and calls[-1][0] == calls[-1][1]  # finished all chunks
