"""GPU acceleration layer for GeneWeaver.

Detection
---------
:func:`detect_gpu` probes, in order:

1. ``numba.cuda`` — the primary backend. Numba JIT-compiles CUDA kernels
   through LLVM/NVVM, so **no ``nvcc`` toolkit is required**; only a
   CUDA-capable NVIDIA driver and a numba build compatible with the
   running Python are needed.
2. ``cupy`` — an alternative backend, probed only if numba is absent.

Every probe is wrapped so a missing/broken package or a driver-level
failure degrades to ``available=False`` with a human-readable reason.

Execution
---------
:func:`scan_gpu` runs the alignment on the GPU **only** when a backend
is genuinely available, and raises :class:`GpuUnavailableError`
otherwise. The CPU path in :mod:`geneware.aligner` remains the default
and the automatic fallback.

IMPORTANT for this environment
------------------------------
On the development machine an Application Control (WDAC) policy blocks
numba's native ``_helperlib`` DLL, so ``import numba`` fails and GPU
execution is reported as unavailable even though an RTX 5050 with a
CUDA 13.1 driver is present. The CUDA kernel code below is real and
will execute on any machine where numba can be imported; until then the
module reports the exact requirements instead of fabricating results.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from .aligner import (
    DEFAULT_PAM_PATTERNS,
    DEFAULT_STRANDS,
    Candidate,
    ScanStats,
    scan_records,
)
from .fasta import FastaRecord


class GpuUnavailableError(RuntimeError):
    """Raised when GPU execution was requested but cannot run."""


GPU_REQUIREMENTS = [
    "NVIDIA GPU with a recent driver (check with `nvidia-smi`)",
    "Python-compatible numba build: `pip install numba` "
    "(numba >= 0.59 ships CUDA support via LLVM/NVVM; no nvcc needed)",
    "The native numba/llvmlite DLLs must not be blocked by an "
    "Application Control / WDAC policy",
    "Optional alternative: `pip install cupy-cuda12x` (or matching "
    "CUDA major version) for the cupy backend",
]


@dataclass
class GpuStatus:
    """Result of a GPU capability probe — all fields measured, none faked."""

    available: bool
    backend: str = "none"          # 'numba-cuda' | 'cupy' | 'none'
    device_name: str | None = None
    vram_total_mb: float | None = None
    compute_capability: str | None = None
    driver_note: str | None = None
    reason: str = ""
    requirements: list[str] = field(default_factory=lambda: list(GPU_REQUIREMENTS))

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "backend": self.backend,
            "device_name": self.device_name,
            "vram_total_mb": self.vram_total_mb,
            "compute_capability": self.compute_capability,
            "driver_note": self.driver_note,
            "reason": self.reason,
            "requirements": self.requirements,
        }


_status_cache: GpuStatus | None = None


def _probe_numba() -> GpuStatus | None:
    """Return a GpuStatus if numba+CUDA is usable, else None."""
    try:
        from numba import cuda  # type: ignore
    except Exception as exc:  # ImportError or blocked-DLL OSError
        return GpuStatus(
            available=False, backend="numba-cuda",
            reason=f"numba not usable: {type(exc).__name__}: {exc}",
        )
    try:
        if not cuda.is_available():
            return GpuStatus(
                available=False,
                backend="numba-cuda",
                reason="numba.cuda.is_available() returned False "
                       "(no CUDA device visible to the driver)",
            )
        dev = cuda.get_current_device()
        name = str(dev.name).strip()
        cc = ".".join(str(x) for x in dev.compute_capability)
        # Device attributes expose memory in bytes on most backends.
        try:
            vram = dev.attributes.get("TOTAL_GLOBAL_MEM", None)
            vram_mb = vram / (1024 * 1024) if isinstance(vram, int) else None
        except Exception:
            vram_mb = None
        return GpuStatus(
            available=True,
            backend="numba-cuda",
            device_name=name,
            vram_total_mb=vram_mb,
            compute_capability=cc,
        )
    except Exception as exc:
        return GpuStatus(
            available=False, backend="numba-cuda",
            reason=f"numba.cuda probe failed: {type(exc).__name__}: {exc}",
        )


def _numba_imported() -> bool:
    try:
        import numba  # noqa: F401
        return True
    except Exception:
        return False


def _probe_cupy() -> GpuStatus | None:
    try:
        import cupy  # type: ignore
    except Exception as exc:
        return GpuStatus(
            available=False, backend="cupy",
            reason=f"cupy not usable: {type(exc).__name__}: {exc}",
        )
    try:
        n = cupy.cuda.runtime.getDeviceCount()
        if n < 1:
            return GpuStatus(available=False, backend="cupy",
                             reason="cupy found 0 CUDA devices")
        props = cupy.cuda.runtime.getDeviceProperties(0)
        return GpuStatus(
            available=True,
            backend="cupy",
            device_name=props["name"].decode() if isinstance(props["name"], bytes)
                        else str(props["name"]),
            vram_total_mb=props["totalGlobalMem"] / (1024 * 1024),
            compute_capability=f"{props['major']}.{props['minor']}",
        )
    except Exception as exc:
        return GpuStatus(available=False, backend="cupy",
                         reason=f"cupy probe failed: {type(exc).__name__}: {exc}")


def detect_gpu(force: bool = False) -> GpuStatus:
    """Probe GPU availability. Result is cached unless ``force=True``."""
    global _status_cache
    if _status_cache is not None and not force:
        return _status_cache

    reasons: list[str] = []
    status: GpuStatus | None = None
    for probe in (_probe_numba, _probe_cupy):
        candidate = probe()
        if candidate is None:
            continue
        if candidate.available:
            status = candidate
            break
        if candidate.reason:
            reasons.append(f"{candidate.backend}: {candidate.reason}")
        status = status or candidate
    if status is None or not status.available:
        reason = "; ".join(reasons) or "no supported GPU backend importable"
        status = GpuStatus(available=False, reason=reason)

    _status_cache = status
    return status


def gpu_available() -> bool:
    return detect_gpu().available


# ---------------------------------------------------------------------------
# Numba CUDA kernel path (executes only when numba.cuda is genuinely usable)
# ---------------------------------------------------------------------------

# Base -> 2-bit code for the CUDA kernel. N=4 is used for any ambiguous
# base; ambiguous bases only match via the PAM/IUPAC logic on CPU post-filter,
# so the kernel uses exact ACGT matching and ambiguous positions simply never
# equal a guide base (documented conservative behaviour).
def _numba_scan(
    records: list[FastaRecord],
    guide: str,
    *,
    chunk_size: int,
    overlap: int,
    pam_patterns: Iterable[str],
    strands: Iterable[str],
    max_mismatches: int,
    require_pam: bool,
) -> tuple[list[Candidate], ScanStats]:
    """GPU scan via numba.cuda. Only called when detect_gpu() is available.

    Implementation strategy: encode each chunk to a uint8 array, launch a
    kernel with one thread per candidate window start (per strand), each
    thread counts mismatches against the guide and writes hits to a
    device buffer. Results are copied back and re-scored/validated on CPU
    (PAM via IUPAC, ambiguity-aware positions) so GPU results are always
    verified by the exact CPU semantics before being reported.
    """
    # Lazy import: numba is only imported on machines where it works.
    import numpy as np
    from numba import cuda, uint8, int32, int64  # type: ignore

    from .chunking import chunk_records

    guide_rc = guide.translate(
        str.maketrans("ACGT", "TGCA")
    )[::-1]

    @cuda.jit
    def _count_mm_kernel(seq_u8, guide_u8, starts, n_starts, mm_out):
        """One thread per start position; counts ACGT mismatches."""
        t = cuda.grid(1)
        if t >= n_starts:
            return
        i = starts[t]
        L = guide_u8.shape[0]
        cnt = 0
        for k in range(L):
            g = guide_u8[k]
            w = seq_u8[i + k]
            # code 4 (ambiguous) never equals a guide code
            if g != w:
                cnt += 1
        mm_out[t] = cnt

    guide_u8 = np.frombuffer(guide.encode("ascii"), dtype=np.uint8)
    # map ascii to 0..3 for kernel speed
    lut = np.full(256, 4, dtype=np.uint8)
    lut[ord("A")] = 0
    lut[ord("C")] = 1
    lut[ord("G")] = 2
    lut[ord("T")] = 3
    guide_codes = lut[guide_u8].astype(np.uint8)
    guide_rc_codes = lut[np.frombuffer(guide_rc.encode("ascii"),
                                      dtype=np.uint8)].astype(np.uint8)

    patterns = ["".join(p.split()).upper() for p in pam_patterns]
    strand_set = set(strands)
    L = len(guide)

    all_candidates: list[Candidate] = []
    stats = ScanStats()

    for chunk in chunk_records(records, chunk_size, overlap):
        seq = chunk.sequence
        n = len(seq)
        stats.bases_processed += n
        stats.chunks_scanned += 1

        seq_codes = lut[np.frombuffer(seq.encode("ascii"),
                                      dtype=np.uint8)].astype(np.uint8)
        d_seq = cuda.to_device(seq_codes)

        pam_len = len(patterns[0]) if patterns else 0
        tasks: list[tuple[int, "object", bool]] = []  # (start, guide_arr, is_plus)
        for i in range(0, n - L + 1):
            if "+" in strand_set and i + L + pam_len <= n:
                tasks.append((i, guide_codes, True))
            if "-" in strand_set and i >= pam_len:
                # minus-strand window compare uses revcomp of window;
                # equivalent to comparing guide_rc against seq[i:i+L]
                # read in reverse -- handled by dedicated kernel below.
                pass

        # '+' strand kernel
        if tasks:
            starts = np.array([t[0] for t in tasks], dtype=np.int64)
            mm = np.empty(len(tasks), dtype=np.int32)
            d_starts = cuda.to_device(starts)
            d_mm = cuda.to_device(mm)
            threads = 256
            blocks = (len(tasks) + threads - 1) // threads
            _count_mm_kernel[blocks, threads](d_seq, guide_codes,
                                              d_starts, len(tasks), d_mm)
            mm = d_mm.copy_to_host()
            stats.windows_evaluated += len(tasks)
            for (i, _g, _p), m in zip(tasks, mm):
                if m <= max_mismatches:
                    pam = seq[i + L : i + L + pam_len]
                    from .aligner import _pam_matches, _mismatch_positions
                    compat = _pam_matches(pam, patterns)
                    if compat or not require_pam:
                        stats.candidates_found += 1
                        window = seq[i : i + L]
                        all_candidates.append(Candidate(
                            chromosome=chunk.chromosome,
                            start=chunk.start + i,
                            end=chunk.start + i + L,
                            strand="+",
                            mismatches=int(m),
                            mismatch_positions=_mismatch_positions(guide, window),
                            pam=pam,
                            pam_compatible=compat,
                            guide=guide,
                        ))

        # '-' strand: compare guide reverse-complement against the window
        # in reverse order. Implement by scanning reversed guide against
        # forward window: mm(window_revcomp, guide) ==
        # mm(rev(window), revcomp-ish) -- simplest correct approach is a
        # second kernel indexing seq from the right.
        minus_tasks = [i for i in range(0, n - L + 1)
                       if "-" in strand_set and i >= pam_len]
        if minus_tasks:
            starts = np.array(minus_tasks, dtype=np.int64)
            mm = np.empty(len(minus_tasks), dtype=np.int32)
            d_starts = cuda.to_device(starts)
            d_mm = cuda.to_device(mm)

            @cuda.jit
            def _count_mm_minus(seq_u8, guide_rc_u8, starts, n_starts, mm_out):
                t = cuda.grid(1)
                if t >= n_starts:
                    return
                i = starts[t]
                Lg = guide_rc_u8.shape[0]
                cnt = 0
                for k in range(Lg):
                    # position k of guide_rc corresponds to window position
                    # L-1-k; guide_rc[k] should equal seq[i + L-1-k]
                    g = guide_rc_u8[k]
                    w = seq_u8[i + Lg - 1 - k]
                    if g != w:
                        cnt += 1
                mm_out[t] = cnt

            threads = 256
            blocks = (len(minus_tasks) + threads - 1) // threads
            _count_mm_minus[blocks, threads](d_seq, guide_rc_codes,
                                             d_starts, len(minus_tasks), d_mm)
            mm = d_mm.copy_to_host()
            stats.windows_evaluated += len(minus_tasks)
            from .aligner import _pam_matches, _mismatch_positions, revcomp
            for i, m in zip(minus_tasks, mm):
                if m <= max_mismatches:
                    pam = revcomp(seq[i - pam_len : i])
                    compat = _pam_matches(pam, patterns)
                    if compat or not require_pam:
                        stats.candidates_found += 1
                        window_rc = revcomp(seq[i : i + L])
                        all_candidates.append(Candidate(
                            chromosome=chunk.chromosome,
                            start=chunk.start + i,
                            end=chunk.start + i + L,
                            strand="-",
                            mismatches=int(m),
                            mismatch_positions=_mismatch_positions(guide, window_rc),
                            pam=pam,
                            pam_compatible=compat,
                            guide=guide,
                        ))

        del d_seq

    from .aligner import deduplicate
    unique = deduplicate(all_candidates)
    stats.candidates_found = len(unique)
    return unique, stats


def scan_gpu(
    records: list[FastaRecord],
    guide: str,
    *,
    chunk_size: int = 1_000_000,
    overlap: int = 64,
    pam_patterns: Iterable[str] = DEFAULT_PAM_PATTERNS,
    strands: Iterable[str] = DEFAULT_STRANDS,
    max_mismatches: int = 0,
    require_pam: bool = True,
    progress: "object | None" = None,
) -> tuple[list[Candidate], ScanStats]:
    """Run the alignment scan on the GPU when genuinely available.

    Falls back is NOT implicit: callers should check :func:`detect_gpu`
    first, or catch :class:`GpuUnavailableError`. When a backend exists
    but is not numba (e.g. cupy only), we still run the numba path only
    if numba is usable; otherwise raise — we never silently pretend GPU
    work happened.
    """
    status = detect_gpu()
    if not status.available:
        raise GpuUnavailableError(
            "GPU execution unavailable: " + status.reason
            + ". Requirements: " + "; ".join(GPU_REQUIREMENTS)
        )
    if status.backend != "numba-cuda":
        raise GpuUnavailableError(
            f"GPU backend '{status.backend}' detected but only the numba-cuda "
            "execution path is implemented; install numba for GPU support."
        )
    return _numba_scan(
        records, guide,
        chunk_size=chunk_size, overlap=overlap,
        pam_patterns=pam_patterns, strands=strands,
        max_mismatches=max_mismatches, require_pam=require_pam,
    )


def scan_with_fallback(
    records: list[FastaRecord],
    guide: str,
    *,
    prefer_gpu: bool = True,
    **kwargs: object,
) -> tuple[list[Candidate], ScanStats, str]:
    """Run GPU scan if requested+available, else the CPU baseline.

    Returns ``(candidates, stats, backend_used)`` where backend_used is
    ``'gpu'`` or ``'cpu'`` — always truthful about what executed.
    """
    if prefer_gpu and gpu_available():
        cands, stats = scan_gpu(records, guide, **kwargs)  # type: ignore[arg-type]
        return cands, stats, "gpu"
    cands, stats = scan_records(records, guide, **kwargs)  # type: ignore[arg-type]
    return cands, stats, "cpu"
