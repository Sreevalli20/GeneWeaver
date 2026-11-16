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
On the development machine numba 0.68.1 imports and the NVIDIA driver is
visible (RTX 5050, CUDA 13.1 driver), but ``cuda.is_available()`` was
False because the CUDA *libraries* (``nvvm.dll``, ``nvrtc``, ``cudart``,
``libdevice``) were absent — no CUDA toolkit is installed. The
``nvidia-cuda-nvcc-cu12`` / ``nvidia-cuda-nvrtc-cu12`` /
``nvidia-cuda-runtime-cu12`` pip wheels do ship them, so
:func:`_ensure_cuda_home` links those DLLs into a CUDA_HOME-layout
directory under the user cache before numba is probed. On machines
without those wheels (or with a blocked DLL policy) the module still
reports GPU as unavailable with the exact requirements — never faked.
"""

from __future__ import annotations

import os
import shutil
import site
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from .aligner import (
    DEFAULT_PAM_PATTERNS,
    DEFAULT_STRANDS,
    AlignError,
    Candidate,
    ScanStats,
    _ambiguous_positions,
    _mismatch_positions,
    _normalise_patterns,
    _pam_matches,
    _validate_scan_args,
    deduplicate,
    revcomp,
    scan_records,
    validate_guide,
    validate_overlap,
)
from .fasta import FastaRecord


class GpuUnavailableError(RuntimeError):
    """Raised when GPU execution was requested but cannot run."""


GPU_REQUIREMENTS = [
    "NVIDIA GPU with a recent driver (check with `nvidia-smi`)",
    "Python-compatible numba build: `pip install numba` "
    "(numba >= 0.59 ships CUDA support via LLVM/NVVM; no nvcc needed)",
    "CUDA libraries (nvvm/nvrtc/cudart/libdevice): either a CUDA toolkit "
    "install, CUDA_HOME pointing at one, or the pip wheels "
    "`nvidia-cuda-nvcc-cu12`, `nvidia-cuda-nvrtc-cu12`, "
    "`nvidia-cuda-runtime-cu12` (auto-detected and linked by geneware)",
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


def _iter_site_packages() -> list[Path]:
    dirs: list[str] = []
    try:
        dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        dirs.append(site.getusersitepackages())
    except Exception:
        pass
    return [Path(d) for d in dict.fromkeys(dirs) if d and Path(d).is_dir()]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _provision_cuda_home() -> Path | None:
    """Assemble a CUDA_HOME tree from ``nvidia-*-cu12`` pip wheels.

    Numba locates ``nvvm.dll``/``nvrtc``/``cudart``/``libdevice`` via the
    ``CUDA_HOME``/``CUDA_PATH`` env vars (or a conda env / system CUDA
    toolkit). On pip-only installs the ``nvidia-cuda-nvcc-cu12``,
    ``nvidia-cuda-nvrtc-cu12`` and ``nvidia-cuda-runtime-cu12`` wheels
    ship exactly those libraries, but under per-package directories that
    numba does not search. This function links them into a single
    CUDA-layout directory under the user cache so numba can find them.

    Returns the staging directory, or ``None`` when the wheels are not
    installed (nothing is created then).
    """
    nvidia_roots = [sp / "nvidia" for sp in _iter_site_packages()]
    nvvm_dlls = sorted(
        p for root in nvidia_roots
        for p in (root / "cuda_nvcc" / "nvvm" / "bin").glob("nvvm*.dll")
        if p.is_file()
    )
    if not nvvm_dlls:
        return None

    base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / ".cache"))
    home = base / "geneware" / "cuda-home"

    for dll in nvvm_dlls:
        _link_or_copy(dll, home / "nvvm" / "bin" / dll.name)
    libdevice_dir = home / "nvvm" / "libdevice"
    libdevice_dir.mkdir(parents=True, exist_ok=True)
    for bc in sorted(
        p for root in nvidia_roots
        for p in (root / "cuda_nvcc" / "nvvm" / "libdevice").glob("libdevice*.bc")
        if p.is_file()
    ):
        _link_or_copy(bc, libdevice_dir / bc.name)
    for root in nvidia_roots:
        for dll in (root / "cuda_runtime" / "bin").glob("*.dll"):
            _link_or_copy(dll, home / "bin" / dll.name)
        for dll in (root / "cuda_nvrtc" / "bin").glob("*.dll"):
            _link_or_copy(dll, home / "bin" / dll.name)

    return home


def _ensure_cuda_home() -> None:
    """Point ``CUDA_HOME`` at provisioned nvidia-wheel libraries if needed.

    Must run *before* numba's CUDA path cache is populated (i.e. before the
    first ``cuda.is_available()`` call). Respects a user-supplied
    ``CUDA_HOME``/``CUDA_PATH`` and clears numba's cached search result if
    numba was already imported earlier in the process.
    """
    if os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH"):
        return
    home = _provision_cuda_home()
    if home is None:
        return
    os.environ["CUDA_HOME"] = str(home)
    try:
        from numba.cuda import cuda_paths  # type: ignore
        if hasattr(cuda_paths.get_cuda_paths, "_cached_result"):
            del cuda_paths.get_cuda_paths._cached_result
    except Exception:
        pass


def _probe_numba() -> GpuStatus | None:
    """Return a GpuStatus if numba+CUDA is usable, else None."""
    _ensure_cuda_home()
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
        raw_name = dev.name
        name = (raw_name.decode() if isinstance(raw_name, bytes)
                else str(raw_name)).strip()
        cc = ".".join(str(x) for x in dev.compute_capability)
        vram_mb = None
        try:
            # cuMemGetInfo needs an active context; select_device creates one.
            cuda.select_device(0)
            from ctypes import byref, c_size_t
            from numba.cuda.cudadrv.driver import driver as _drv  # type: ignore
            _free, _total = c_size_t(), c_size_t()
            _drv.cuMemGetInfo(byref(_free), byref(_total))
            vram_mb = _total.value / (1024 * 1024)
        except Exception:
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
_GPU_KERNEL = None


def _get_count_kernel():
    """Lazily create the shared mismatch-count CUDA kernel (once).

    One thread per window start counts ACGT-code mismatches against the
    guide; used for BOTH strands by passing the forward guide codes for
    '+' and the reverse-complement guide codes for '-'. Compiled once
    per process — the previous per-chunk ``@cuda.jit`` definitions were
    recompiled for every chunk.
    """
    global _GPU_KERNEL
    if _GPU_KERNEL is not None:
        return _GPU_KERNEL
    from numba import cuda  # type: ignore

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

    _GPU_KERNEL = _count_mm_kernel
    return _GPU_KERNEL


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
    progress: "object | None" = None,
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
    from numba import cuda  # type: ignore
    from numba.core.errors import NumbaPerformanceWarning  # type: ignore

    # Kernel-launch occupancy warnings are noise for batch scans.
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

    from .chunking import chunk_records, total_chunks

    guide = validate_guide(guide)
    records = list(records)
    validate_overlap(records, guide, chunk_size, overlap, pam_patterns)
    patterns = _normalise_patterns(pam_patterns)
    strand_set, pam_len = _validate_scan_args(patterns, strands)
    if max_mismatches < 0:
        raise AlignError(f"max_mismatches must be >= 0, got {max_mismatches}")
    L = len(guide)
    guide_rc = revcomp(guide)

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
    d_guide_plus = cuda.to_device(guide_codes)
    d_guide_minus = cuda.to_device(guide_rc_codes)
    kernel = _get_count_kernel()
    threads = 256

    all_candidates: list[Candidate] = []
    stats = ScanStats()
    n_total = total_chunks(records, chunk_size, overlap)
    done_chunks = 0

    for chunk in chunk_records(records, chunk_size, overlap):
        seq = chunk.sequence
        n = len(seq)
        stats.bases_processed += n
        stats.chunks_scanned += 1

        seq_codes = lut[np.frombuffer(seq.encode("ascii"),
                                      dtype=np.uint8)].astype(np.uint8)
        d_seq = cuda.to_device(seq_codes)

        # Window starts are contiguous ranges, so np.arange replaces the
        # previous O(n) Python task-list construction entirely.
        #   '+' : i in [0, n-L-pam_len]          (protospacer + PAM fit)
        #   '-' : i in [pam_len, n-L]            (PAM before protospacer)
        # Both ranges have the same length nwin = n-L-pam_len+1.
        nwin = max(0, n - L - pam_len + 1)

        def _launch(guide_dev, i_lo: int, i_hi: int):
            """Run the count kernel; return (starts, mm_counts) on host."""
            nw = i_hi - i_lo + 1
            if nw <= 0:
                return None, None
            starts = np.arange(i_lo, i_hi + 1, dtype=np.int64)
            mm = np.empty(nw, dtype=np.int32)
            d_starts = cuda.to_device(starts)
            d_mm = cuda.to_device(mm)
            blocks = (nw + threads - 1) // threads
            kernel[blocks, threads](d_seq, guide_dev, d_starts, nw, d_mm)
            return starts, d_mm.copy_to_host()

        if "+" in strand_set:
            stats.windows_evaluated += nwin
            starts, mm = _launch(d_guide_plus, 0, nwin - 1)
            if starts is not None:
                # Only surviving windows are re-verified on CPU with the
                # exact IUPAC semantics (positions + strand PAM string).
                for j in np.flatnonzero(mm <= max_mismatches):
                    i = int(starts[j])
                    genomic_window = seq[i : i + L]
                    matched = genomic_window
                    mm_pos = _mismatch_positions(guide, matched)
                    if len(mm_pos) > max_mismatches:
                        continue
                    amb_pos = _ambiguous_positions(guide, matched, mm_pos)
                    pam = seq[i + L : i + L + pam_len]
                    compat = _pam_matches(pam, patterns)
                    if compat or not require_pam:
                        stats.candidates_found += 1
                        all_candidates.append(Candidate(
                            chromosome=chunk.chromosome,
                            start=chunk.start + i,
                            end=chunk.start + i + L,
                            strand="+",
                            mismatches=len(mm_pos),
                            mismatch_positions=mm_pos,
                            pam=pam,
                            pam_compatible=compat,
                            guide=guide,
                            matched_sequence=matched,
                            genomic_sequence=genomic_window,
                            ambiguous_positions=amb_pos,
                        ))

        if "-" in strand_set:
            stats.windows_evaluated += nwin
            starts, mm = _launch(d_guide_minus, pam_len,
                                 pam_len + nwin - 1)
            if starts is not None:
                for j in np.flatnonzero(mm <= max_mismatches):
                    i = int(starts[j])
                    genomic_window = seq[i : i + L]
                    matched = revcomp(genomic_window)
                    mm_pos = _mismatch_positions(guide, matched)
                    if len(mm_pos) > max_mismatches:
                        continue
                    amb_pos = _ambiguous_positions(guide, matched, mm_pos)
                    pam = revcomp(seq[i - pam_len : i])
                    compat = _pam_matches(pam, patterns)
                    if compat or not require_pam:
                        stats.candidates_found += 1
                        all_candidates.append(Candidate(
                            chromosome=chunk.chromosome,
                            start=chunk.start + i,
                            end=chunk.start + i + L,
                            strand="-",
                            mismatches=len(mm_pos),
                            mismatch_positions=mm_pos,
                            pam=pam,
                            pam_compatible=compat,
                            guide=guide,
                            matched_sequence=matched,
                            genomic_sequence=genomic_window,
                            ambiguous_positions=amb_pos,
                        ))

        del d_seq
        done_chunks += 1
        if progress is not None:
            progress(done_chunks, max(n_total, 1))

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
        progress=progress,
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
