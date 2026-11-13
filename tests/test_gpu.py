"""Tests for GPU detection and fallback behaviour.

The GPU backend is only exercised when genuinely available; tests verify
the honest detection contract and the CPU fallback path.
"""

import pytest

from geneware.gpu import (
    GPU_REQUIREMENTS,
    GpuStatus,
    GpuUnavailableError,
    detect_gpu,
    gpu_available,
    scan_gpu,
    scan_with_fallback,
)


def test_detect_gpu_returns_status_object():
    st = detect_gpu(force=True)
    assert isinstance(st, GpuStatus)
    assert isinstance(st.available, bool)
    assert st.backend in ("numba-cuda", "cupy", "none")
    if not st.available:
        assert st.reason  # honest reason required
        assert st.requirements == GPU_REQUIREMENTS


def test_gpu_available_matches_detect():
    assert gpu_available() == detect_gpu().available


def test_scan_gpu_raises_when_unavailable():
    if gpu_available():
        pytest.skip("GPU is available; cannot test unavailable path")
    from geneware.fasta import parse_fasta
    from pathlib import Path
    records = parse_fasta(
        Path(__file__).resolve().parent.parent / "data" / "example.fasta")
    with pytest.raises(GpuUnavailableError):
        scan_gpu(records, "ACGTACGTACGTACGTACGT")


def test_scan_with_fallback_returns_truthful_backend():
    from geneware.fasta import parse_fasta
    from pathlib import Path
    records = parse_fasta(
        Path(__file__).resolve().parent.parent / "data" / "example.fasta")
    cands, stats, backend = scan_with_fallback(
        records, "ACGTACGTACGTACGTACGT", prefer_gpu=False,
        require_pam=False)
    assert backend == "cpu"
    assert stats.bases_processed > 0
    # fallback must still find the planted guide occurrences
    assert len(cands) > 0


def test_gpu_status_dict_serialisable():
    d = detect_gpu().to_dict()
    assert "available" in d and "requirements" in d
