"""CPU baseline CRISPR-protospacer search for GeneWeaver (Week 1).

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

Performance note
----------------
The inner loop is pure Python. It is correct and adequate for small
loci and tests; throughput on chromosome-scale data is a known Week 1
limitation to be addressed with Numba/CUDA in Week 2+.
"""

from __future__ import annotations

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
    """
    if stats is None:
        stats = ScanStats()
    patterns = ["".join(p.split()).upper() for p in pam_patterns]
    strand_set = set(strands)
    if not strand_set <= {"+", "-"}:
        raise AlignError(f"invalid strand(s) in {strand_set!r}; use '+' and/or '-'")
    pam_lens = {len(p) for p in patterns}
    if len(pam_lens) > 1:
        raise AlignError(
            f"all PAM patterns must have equal length, got lengths {sorted(pam_lens)}"
        )
    pam_len = pam_lens.pop() if pam_lens else 0

    seq = chunk.sequence
    L = len(guide)
    n = len(seq)
    stats.bases_processed += n
    stats.chunks_scanned += 1

    for i in range(0, n - L + 1):
        # '+' strand: protospacer seq[i:i+L], PAM seq[i+L:i+L+pam_len]
        if "+" in strand_set and i + L + pam_len <= n:
            window = seq[i : i + L]
            stats.windows_evaluated += 1
            mm = _mismatch_positions(guide, window)
            if len(mm) <= max_mismatches:
                pam = seq[i + L : i + L + pam_len]
                compat = _pam_matches(pam, patterns)
                if compat or not require_pam:
                    stats.candidates_found += 1
                    yield Candidate(
                        chromosome=chunk.chromosome,
                        start=chunk.start + i,
                        end=chunk.start + i + L,
                        strand="+",
                        mismatches=len(mm),
                        mismatch_positions=mm,
                        pam=pam,
                        pam_compatible=compat,
                        guide=guide,
                    )

        # '-' strand: protospacer revcomp(seq[i:i+L]) vs guide,
        # PAM = revcomp(seq[i-pam_len:i]); needs i >= pam_len.
        if "-" in strand_set and i >= pam_len:
            window_rc = revcomp(seq[i : i + L])
            stats.windows_evaluated += 1
            mm = _mismatch_positions(guide, window_rc)
            if len(mm) <= max_mismatches:
                pam = revcomp(seq[i - pam_len : i])
                compat = _pam_matches(pam, patterns)
                if compat or not require_pam:
                    stats.candidates_found += 1
                    yield Candidate(
                        chromosome=chunk.chromosome,
                        start=chunk.start + i,
                        end=chunk.start + i + L,
                        strand="-",
                        mismatches=len(mm),
                        mismatch_positions=mm,
                        pam=pam,
                        pam_compatible=compat,
                        guide=guide,
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

    ``overlap`` should be >= ``len(guide) + pam_len`` so that any
    protospacer+PAM window crossing a chunk boundary is fully contained
    in at least one chunk; the default 64 covers a 30-nt guide + 3-nt PAM.
    ``progress(done_chunks, total_chunks)`` is called after each chunk.
    """
    guide = validate_guide(guide)
    records = list(records)
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
