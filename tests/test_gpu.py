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


# Pure-ACGT sequence with known guide hits on both strands (no ambiguous
# bases, so the conservative GPU/CPU difference does not apply).
PURE_SEQ = (
    "ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT"
    "GGGGAAAACCCCGGGGTTTTAAAACCCCGGGGAAAACCCCGGGGTTTTAAAACCCCGGGGTTTT"
)


def _pure_records():
    from geneware.fasta import FastaRecord
    return [FastaRecord(record_id="chr_t", description="pure", sequence=PURE_SEQ)]


def _key(c):
    return (c.chromosome, c.start, c.end, c.strand, c.mismatches,
            c.pam, c.pam_compatible, c.mismatch_positions,
            c.matched_sequence, c.genomic_sequence, c.ambiguous_positions)


@pytest.mark.skipif(not gpu_available(), reason="no usable GPU backend")
def test_gpu_scan_matches_cpu_exactly():
    """On unambiguous sequence the GPU path must equal the CPU path."""
    from geneware.aligner import scan_records
    records = _pure_records()
    kw = dict(chunk_size=60, overlap=30, max_mismatches=2,
              pam_patterns=("NGG",), strands=("+", "-"), require_pam=True)
    cpu, _ = scan_records(records, "TTTTAAAACCCC", **kw)
    gpu, stats = scan_gpu(records, "TTTTAAAACCCC", **kw)
    assert stats.bases_processed > 0
    assert sorted(map(_key, cpu)) == sorted(map(_key, gpu))
    assert len(gpu) > 0  # sanity: the fixture really has hits


@pytest.mark.skipif(not gpu_available(), reason="no usable GPU backend")
def test_scan_with_fallback_uses_gpu_when_available():
    records = _pure_records()
    cands, stats, backend = scan_with_fallback(
        records, "TTTTAAAACCCC", prefer_gpu=True,
        chunk_size=100, overlap=30, max_mismatches=1,
        require_pam=True)
    assert backend == "gpu"
    assert stats.bases_processed > 0


def test_ensure_cuda_home_is_idempotent_and_safe():
    """Provisioning must not crash and must respect existing CUDA_HOME."""
    import os
    from geneware.gpu import _ensure_cuda_home, _provision_cuda_home
    before = os.environ.get("CUDA_HOME")
    home = _provision_cuda_home()
    if home is not None:
        assert (home / "nvvm" / "bin").is_dir()
    _ensure_cuda_home()  # second call must be a no-op
    _ensure_cuda_home()
    # env may have been set by provisioning; that's the intended behaviour
    assert os.environ.get("CUDA_HOME") in (None, before, str(home))
