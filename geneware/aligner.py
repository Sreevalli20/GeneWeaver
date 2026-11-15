"""CPU protospacer/PAM search engine for GeneWeaver.

This is a deliberately simple, fully documented baseline. It is NOT a
scientifically validated off-target prediction model.

Matching semantics
------------------
* A *window* is an exact-length slice of ``len(guide)`` bases.
* For strand ``'+'`` the window is compared directly to the guide.
* For strand ``'-'`` the window's reverse complement is compared to the
  guide (the guide is itself reverse-complemented once up front).
* Base comparison is IUPAC-aware: a position is a *match* when the IUPAC
  expansion sets of the guide base and the window base intersect;
  otherwise it is a *mismatch*. Positions are reported 0-based relative
  to the protospacer's 5'->3' orientation.
* Coordinates are 0-based, half-open ``[start, end)`` on the plus strand
  of the source record, matching :mod:`geneware.chunking`.

PAM model
---------
* PAM patterns are IUPAC strings (default ``["NGG"]``), matched against
  the 3 bases immediately 3' of the protospacer *on the candidate
  strand* (standard Cas9 NGG geometry, no bulge).
* ``'+'`` candidate at ``[i, i+L)``: PAM = ``seq[i+L : i+L+3]``.
* ``'-'`` candidate at ``[i, i+L)``: PAM on the minus strand is
  ``revcomp(seq[i-3 : i])``.
* A window is only evaluated when the full protospacer *and* PAM region
  fit inside the sequence being scanned; edge windows that cannot be
  fully assessed are skipped, not silently treated as matches.
* ``require_pam=True`` (default) keeps only PAM-compatible candidates.

Performance (Day 4)
-------------------
Sequences are validated to the 15-letter IUPAC DNA alphabet, so every
base is encoded as a 4-bit code and the IUPAC intersection rule becomes
a 15x15 lookup table. Two execution paths share identical semantics:

* **numba JIT** (default when ``numba`` imports cleanly): per-window
  mismatch counting is compiled to machine code with early exit at
  ``max_mismatches + 1``; only surviving windows are re-scored on CPU
  for exact mismatch positions and PAM strings.
* **pure Python fallback**: the same encoded-table loop without numba —
  still substantially faster than the Week-1 frozenset-per-position
  implementation. Force it with ``GENEWEARE_ALIGNER_ENGINE=python``
  (useful for A/B benchmarking).

For a minus-strand window at ``[i, i+L)`` the mismatch count equals the
forward comparison of ``seq[i:i+L]`` against ``revcomp(guide)`` — this
follows from the symmetry of IUPAC expansion intersection under
complementation — so no per-window ``revcomp`` is ever needed.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from .chunking import GenomeChunk, chunk_records, iter_chunks, total_chunks
from .fasta import FastaRecord, IUPAC_DNA, IUPAC_EXPANSION

DEFAULT_PAM_PATTERNS = ("NGG",)
DEFAULT_STRANDS = ("+", "-")
MIN_GUIDE_LEN = 10
MAX_GUIDE_LEN = 30

_COMPLEMENT = str.maketrans("ACGTRYSWKMBDHVNacgtryswkmbdhvn", "TGCAYRSWMKVHDBNtgcayrswmkvhdbn")


class AlignError(ValueError):
    """Raised for invalid guide/alignment configuration."""


def revcomp(seq: str) -> str:
    """Reverse-complement a DNA sequence using the IUPAC alphabet."""
    return seq.translate(_COMPLEMENT)[::-1]


def _expand(base: str) -> frozenset[str]:
    return frozenset(IUPAC_EXPANSION[base.upper()])


def bases_compatible(a: str, b: str) -> bool:
    """True if IUPAC expansions of ``a`` and ``b`` intersect."""
    return bool(_expand(a) & _expand(b))


def validate_guide(
    guide: str,
    *,
    min_length: int = MIN_GUIDE_LEN,
    max_length: int = MAX_GUIDE_LEN,
) -> str:
    """Validate and normalise a guide RNA spacer sequence.

    Returns the uppercase, whitespace-free guide. Raises
    :class:`AlignError` on empty input, invalid characters, or length
    outside ``[min_length, max_length]``.
    """
    compact = "".join(guide.split()).upper().replace("U", "T")
    if not compact:
        raise AlignError("guide sequence is empty")
    bad = sorted({c for c in compact if c not in IUPAC_DNA})
    if bad:
        raise AlignError(
            f"guide contains invalid nucleotide character(s) {bad!r} "
            "(expected IUPAC DNA alphabet)"
        )
    if not (min_length <= len(compact) <= max_length):
        raise AlignError(
            f"guide length {len(compact)} outside allowed range "
            f"[{min_length}, {max_length}]"
        )
    return compact


def _pam_matches(pam_seq: str, patterns: Iterable[str]) -> bool:
    """True if ``pam_seq`` is compatible with any IUPAC pattern."""
    for pat in patterns:
        pat = "".join(pat.split()).upper()
        if len(pat) != len(pam_seq):
            continue
        if all(bases_compatible(g, p) for g, p in zip(pam_seq, pat, strict=True)):
            return True
    return False


# ---------------------------------------------------------------------------
# Encoded-base fast path
# ---------------------------------------------------------------------------

# Canonical order of the 15 IUPAC DNA bases (uppercase; sequences are
# normalised to uppercase before scanning).
_IUPAC_ORDER = "ACGTRYSWKMBDHVN"
_N_BASES = len(_IUPAC_ORDER)
_BASE_CODE = {b: i for i, b in enumerate(_IUPAC_ORDER)}

# _COMPAT_PY[g][w] is True iff the IUPAC expansions of base g and base w
# intersect — exactly the rule used by ``bases_compatible``.
_EXP_SETS = [frozenset(IUPAC_EXPANSION[b]) for b in _IUPAC_ORDER]
_COMPAT_PY = tuple(
    tuple(bool(_EXP_SETS[g] & _EXP_SETS[w]) for w in range(_N_BASES))
    for g in range(_N_BASES)
)

# bytes.translate table mapping an ASCII byte to its IUPAC code. Unknown
# bytes map to the 'N' code (index _N_BASES - 1); validated sequences
# never contain them.
_TRANS = bytearray([_N_BASES - 1] * 256)
for _i, _b in enumerate(_IUPAC_ORDER):
    _TRANS[ord(_b)] = _i
    _TRANS[ord(_b.lower())] = _i
_TRANS = bytes(_TRANS)


def _encode_codes(seq: str) -> bytes:
    """Encode a validated sequence to ``bytes`` of IUPAC base codes."""
    return seq.encode("ascii").translate(_TRANS)


def _aligner_engine() -> str:
    """Engine selector: ``auto`` (numba if usable), ``python``, ``numba``."""
    return os.environ.get("GENEWEARE_ALIGNER_ENGINE", "auto").strip().lower()


_NUMBA_KERNEL = None
_NUMBA_KERNEL_FAILED = False
_COMPAT_NP = None
_LUT_NP = None


def _get_numba_kernel():
    """Return the JIT-compiled scan kernel, or ``None`` when unusable.

    The kernel counts mismatches for every window start in
    ``[i_lo, i_hi]`` against ``gcodes`` using the IUPAC compatibility
    table, with early exit once ``max_mm + 1`` mismatches are seen.
    Hits are written as ``(start, mismatch_count)`` rows into ``out``;
    the hit count is stored in ``count[0]``.
    """
    global _NUMBA_KERNEL, _NUMBA_KERNEL_FAILED
    if _aligner_engine() == "python":
        return None
    if _NUMBA_KERNEL is not None or _NUMBA_KERNEL_FAILED:
        return _NUMBA_KERNEL
    try:
        from numba import njit
    except Exception:
        _NUMBA_KERNEL_FAILED = True
        return None

    @njit(nogil=True)
    def _scan_strand(seq, gcodes, compat, max_mm, i_lo, i_hi, out, count):
        L = gcodes.shape[0]
        cnt = 0
        for i in range(i_lo, i_hi + 1):
            mm = 0
            for k in range(L):
                if compat[gcodes[k], seq[i + k]] == 0:
                    mm += 1
                    if mm > max_mm:
                        break
            if mm <= max_mm:
                out[cnt, 0] = i
                out[cnt, 1] = mm
                cnt += 1
        count[0] = cnt

    _NUMBA_KERNEL = _scan_strand
    return _NUMBA_KERNEL


def _get_compat_np():
    """NumPy uint8 view of ``_COMPAT_PY`` (built once, cached)."""
    global _COMPAT_NP
    if _COMPAT_NP is None:
        import numpy as np
        _COMPAT_NP = np.array(_COMPAT_PY, dtype=np.uint8)
    return _COMPAT_NP


def _encode_np(seq: str):
    """Encode a sequence to a uint8 NumPy array of IUPAC codes."""
    global _LUT_NP
    import numpy as np
    if _LUT_NP is None:
        lut = np.full(256, _N_BASES - 1, dtype=np.uint8)
        for i, b in enumerate(_IUPAC_ORDER):
            lut[ord(b)] = i
            lut[ord(b.lower())] = i
        _LUT_NP = lut
    return _LUT_NP[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


@dataclass(frozen=True)
class Candidate:
    """A single protospacer candidate with full provenance."""

    chromosome: str
    start: int            # 0-based inclusive, plus-strand coordinates
    end: int              # 0-based exclusive
    strand: str           # '+' or '-'
    mismatches: int
    mismatch_positions: tuple[int, ...]
    pam: str              # PAM as read on the candidate strand (5'->3')
    pam_compatible: bool
    guide: str            # the guide used (5'->3')

    def to_dict(self) -> dict[str, object]:
        return {
            "chromosome": self.chromosome,
            "start": self.start,
            "end": self.end,
            "strand": self.strand,
            "mismatches": self.mismatches,
            "mismatch_positions": list(self.mismatch_positions),
            "pam": self.pam,
            "pam_compatible": self.pam_compatible,
            "guide": self.guide,
        }


@dataclass
class ScanStats:
    """Counters collected during a scan (all real measurements)."""

    bases_processed: int = 0
    windows_evaluated: int = 0
    candidates_found: int = 0
    chunks_scanned: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "bases_processed": self.bases_processed,
            "windows_evaluated": self.windows_evaluated,
            "candidates_found": self.candidates_found,
            "chunks_scanned": self.chunks_scanned,
        }


def _mismatch_positions(guide: str, window: str) -> tuple[int, ...]:
    return tuple(
        i for i, (g, w) in enumerate(zip(guide, window, strict=True))
        if not bases_compatible(g, w)
    )


def _normalise_patterns(pam_patterns: Iterable[str]) -> list[str]:
    return ["".join(p.split()).upper() for p in pam_patterns]


def _validate_scan_args(
    patterns: list[str], strands: Iterable[str]
) -> tuple[set[str], int]:
    """Validate PAM/strand configuration; return (strand_set, pam_len)."""
    strand_set = set(strands)
    if not strand_set <= {"+", "-"}:
        raise AlignError(f"invalid strand(s) in {strand_set!r}; use '+' and/or '-'")
    pam_lens = {len(p) for p in patterns}
    if len(pam_lens) > 1:
        raise AlignError(
            f"all PAM patterns must have equal length, got lengths {sorted(pam_lens)}"
        )
    return strand_set, (pam_lens.pop() if pam_lens else 0)


def validate_overlap(
    records: Iterable[FastaRecord],
    guide: str,
    chunk_size: int,
    overlap: int,
    pam_patterns: Iterable[str] = DEFAULT_PAM_PATTERNS,
) -> None:
    """Raise :class:`AlignError` when chunking could silently miss hits.

    A protospacer+PAM window has length ``len(guide) + pam_len``. Any
    record longer than ``chunk_size`` is split into overlapping chunks;
    if ``overlap`` is smaller than that window length, a match crossing
    a chunk boundary can fall outside *every* chunk and be silently
    lost. Records that fit in a single chunk are unaffected.
    """
    if chunk_size < 1:
        raise AlignError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise AlignError(
            f"overlap must satisfy 0 <= overlap < chunk_size "
            f"(got overlap={overlap}, chunk_size={chunk_size})"
        )
    records = list(records)
    if not any(len(r.sequence) > chunk_size for r in records):
        return
    patterns = _normalise_patterns(pam_patterns)
    pam_lens = {len(p) for p in patterns}
    pam_len = max(pam_lens) if pam_lens else 0
    need = len(guide) + pam_len
    if overlap < need:
        raise AlignError(
            f"overlap {overlap} < guide length + PAM length {need}; "
            "matches crossing a chunk boundary could be missed. "
            f"Use --overlap >= {need}."
        )


def _eval_hit(
    chunk: GenomeChunk,
    i: int,
    strand: str,
    guide: str,
    seq: str,
    patterns: list[str],
    pam_len: int,
    max_mismatches: int,
    require_pam: bool,
) -> Candidate | None:
    """Build a fully verified Candidate for window start ``i``.

    Recomputes exact IUPAC mismatch positions and the strand-oriented
    PAM string — the fast paths only pre-filter on mismatch *count*.
    """
    L = len(guide)
    if strand == "+":
        window = seq[i : i + L]
        pam = seq[i + L : i + L + pam_len]
    else:
        window = revcomp(seq[i : i + L])
        pam = revcomp(seq[i - pam_len : i])
    mm_pos = _mismatch_positions(guide, window)
    if len(mm_pos) > max_mismatches:
        return None
    compat = _pam_matches(pam, patterns)
    if not compat and require_pam:
        return None
    return Candidate(
        chromosome=chunk.chromosome,
        start=chunk.start + i,
        end=chunk.start + i + L,
        strand=strand,
        mismatches=len(mm_pos),
        mismatch_positions=mm_pos,
        pam=pam,
        pam_compatible=compat,
        guide=guide,
    )


def _iter_chunk_python(
    chunk: GenomeChunk,
    guide: str,
    patterns: list[str],
    strand_set: set[str],
    pam_len: int,
    max_mismatches: int,
    require_pam: bool,
    stats: ScanStats,
) -> Iterator[Candidate]:
    """Encoded pure-Python scan (fallback when numba is unusable)."""
    seq = chunk.sequence
    n = len(seq)
    L = len(guide)
    nwin = max(0, n - L - pam_len + 1)
    seq_codes = _encode_codes(seq)
    guide_p = [_BASE_CODE[b] for b in guide]
    guide_m = [_BASE_CODE[b] for b in revcomp(guide)]

    if "+" in strand_set:
        stats.windows_evaluated += nwin
        for i in range(nwin):
            mm = 0
            for k in range(L):
                if not _COMPAT_PY[guide_p[k]][seq_codes[i + k]]:
                    mm += 1
                    if mm > max_mismatches:
                        break
            if mm <= max_mismatches:
                c = _eval_hit(chunk, i, "+", guide, seq, patterns, pam_len,
                              max_mismatches, require_pam)
                if c is not None:
                    stats.candidates_found += 1
                    yield c

    if "-" in strand_set:
        stats.windows_evaluated += nwin
        # Minus-strand count == forward compare vs revcomp(guide).
        for i in range(pam_len, pam_len + nwin):
            mm = 0
            for k in range(L):
                if not _COMPAT_PY[guide_m[k]][seq_codes[i + k]]:
                    mm += 1
                    if mm > max_mismatches:
                        break
            if mm <= max_mismatches:
                c = _eval_hit(chunk, i, "-", guide, seq, patterns, pam_len,
                              max_mismatches, require_pam)
                if c is not None:
                    stats.candidates_found += 1
                    yield c


def _iter_chunk_numba(
    chunk: GenomeChunk,
    guide: str,
    patterns: list[str],
    strand_set: set[str],
    pam_len: int,
    max_mismatches: int,
    require_pam: bool,
    stats: ScanStats,
    kernel,
) -> Iterator[Candidate]:
    """Numba-JIT scan: kernel pre-filters, CPU verifies survivors."""
    import numpy as np

    seq = chunk.sequence
    n = len(seq)
    L = len(guide)
    nwin = max(0, n - L - pam_len + 1)
    seq_np = _encode_np(seq)
    compat = _get_compat_np()
    guide_p_np = _encode_np(guide)
    guide_m_np = _encode_np(revcomp(guide))
    count = np.zeros(1, dtype=np.int64)

    def _run(gcodes, i_lo: int, i_hi: int):
        """Return (out_array, hit_count); hit_count 0 when range empty."""
        if i_hi < i_lo:
            return None, 0
        nw = i_hi - i_lo + 1
        out = np.empty((nw, 2), dtype=np.int64)
        kernel(seq_np, gcodes, compat, max_mismatches,
               i_lo, i_hi, out, count)
        return out, int(count[0])

    if "+" in strand_set:
        stats.windows_evaluated += nwin
        out, cnt = _run(guide_p_np, 0, nwin - 1)
        if out is not None:
            for j in range(cnt):
                i = int(out[j, 0])
                c = _eval_hit(chunk, i, "+", guide, seq, patterns, pam_len,
                              max_mismatches, require_pam)
                if c is not None:
                    stats.candidates_found += 1
                    yield c

    if "-" in strand_set:
        stats.windows_evaluated += nwin
        out, cnt = _run(guide_m_np, pam_len, pam_len + nwin - 1)
        if out is not None:
            for j in range(cnt):
                i = int(out[j, 0])
                c = _eval_hit(chunk, i, "-", guide, seq, patterns, pam_len,
                              max_mismatches, require_pam)
                if c is not None:
                    stats.candidates_found += 1
                    yield c


def scan_chunk(
    chunk: GenomeChunk,
    guide: str,
    *,
    pam_patterns: Iterable[str] = DEFAULT_PAM_PATTERNS,
    strands: Iterable[str] = DEFAULT_STRANDS,
    max_mismatches: int = 0,
    require_pam: bool = True,
    stats: ScanStats | None = None,
) -> Iterator[Candidate]:
    """Scan one :class:`GenomeChunk` for guide candidates.

    ``guide`` must already be validated/normalised. Coordinates in the
    yielded candidates are absolute (chunk offset already applied).
    Uses the numba-JIT engine when available (see module docstring);
    results are identical either way.
    """
    if stats is None:
        stats = ScanStats()
    patterns = _normalise_patterns(pam_patterns)
    strand_set, pam_len = _validate_scan_args(patterns, strands)

    seq = chunk.sequence
    n = len(seq)
    stats.bases_processed += n
    stats.chunks_scanned += 1

    kernel = _get_numba_kernel() if _aligner_engine() != "python" else None
    if kernel is not None:
        yield from _iter_chunk_numba(
            chunk, guide, patterns, strand_set, pam_len,
            max_mismatches, require_pam, stats, kernel,
        )
    else:
        yield from _iter_chunk_python(
            chunk, guide, patterns, strand_set, pam_len,
            max_mismatches, require_pam, stats,
        )


def deduplicate(candidates: Iterable[Candidate]) -> list[Candidate]:
    """Remove duplicate candidates arising from chunk-boundary overlap.

    Uniqueness key: (chromosome, start, end, strand). The PAM is a pure
    function of those coordinates, so it cannot differ for duplicates.
    Order of first occurrence is preserved.
    """
    seen: set[tuple[str, int, int, str]] = set()
    out: list[Candidate] = []
    for c in candidates:
        key = (c.chromosome, c.start, c.end, c.strand)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def scan_records(
    records: Iterable[FastaRecord],
    guide: str,
    *,
    chunk_size: int = 1_000_000,
    overlap: int = 64,
    pam_patterns: Iterable[str] = DEFAULT_PAM_PATTERNS,
    strands: Iterable[str] = DEFAULT_STRANDS,
    max_mismatches: int = 0,
    require_pam: bool = True,
    progress: "callable[[int, int], None] | None" = None,
) -> tuple[list[Candidate], ScanStats]:
    """Scan FASTA records end-to-end with chunking and deduplication.

    ``overlap`` must be >= ``len(guide) + pam_len`` whenever any record
    exceeds ``chunk_size``; :func:`validate_overlap` enforces this so
    boundary-spanning matches can never be silently missed.
    ``progress(done_chunks, total_chunks)`` is called after each chunk.
    """
    guide = validate_guide(guide)
    records = list(records)
    validate_overlap(records, guide, chunk_size, overlap, pam_patterns)
    n_chunks = total_chunks(records, chunk_size, overlap)

    stats = ScanStats()
    all_candidates: list[Candidate] = []
    done = 0

    for chunk in chunk_records(records, chunk_size, overlap):
        for cand in scan_chunk(
            chunk,
            guide,
            pam_patterns=pam_patterns,
            strands=strands,
            max_mismatches=max_mismatches,
            require_pam=require_pam,
            stats=stats,
        ):
            all_candidates.append(cand)
        done += 1
        if progress is not None:
            progress(done, n_chunks)

    unique = deduplicate(all_candidates)
    # Recount after dedup so stats reflect reported candidates.
    stats.candidates_found = len(unique)
    return unique, stats
