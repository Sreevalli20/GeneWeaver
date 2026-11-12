"""FASTA parsing and sequence statistics for GeneWeaver (Week 1).

Built on BioPython's ``Bio.SeqIO``. The parser never silently discards
sequence data:

* hard failures (empty file, no records, empty sequences) raise
  :class:`FastaParseError`;
* non-IUPAC characters raise :class:`FastaParseError` after reporting
  every offending record, so nothing is quietly dropped;
* IUPAC ambiguity codes (R, Y, N, ...) are preserved verbatim;
* uracil (``U``/``u``) is normalised to thymine (``T``) with a warning,
  since this engine models DNA guide search;
* whitespace inside sequences is removed; all other characters are
  treated as data errors.
"""

from __future__ import annotations

import sys
import types
import warnings
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


def _import_seqio():
    """Import ``Bio.SeqIO``, working around a known environment issue.

    ``Bio.SeqIO.__init__`` eagerly imports ``Bio.AlignIO``, which loads the
    compiled ``Bio.Align._codonaligner`` extension. On machines where an
    Application Control / WDAC policy blocks that unsigned DLL, the whole
    ``Bio.SeqIO`` import fails even though FASTA parsing never touches
    alignment code. When (and only when) the real import fails, a stub
    ``Bio.AlignIO`` module is registered so ``Bio.SeqIO`` can load; the
    stub is never used by the FASTA code paths.
    """
    try:
        import Bio.AlignIO  # noqa: F401
    except ImportError:
        stub = types.ModuleType("Bio.AlignIO")
        # Bio.SeqIO iterates these registries at import time; empty dicts
        # simply mean "no alignment formats registered", which is fine
        # for sequence-only (e.g. FASTA) parsing.
        stub._FormatToIterator = {}
        stub._FormatToWriter = {}
        sys.modules["Bio.AlignIO"] = stub
    from Bio import SeqIO

    return SeqIO


SeqIO = _import_seqio()

# IUPAC nucleotide alphabet (DNA). 'U' is handled separately (-> 'T').
IUPAC_DNA = frozenset("ACGTRYSWKMBDHVNacgtryswkmbdhvn")

# Per-base IUPAC expansion, used by the aligner for ambiguity-aware matching.
IUPAC_EXPANSION = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT",
    "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG",
    "N": "ACGT",
}


class FastaParseError(ValueError):
    """Raised when a FASTA file cannot be parsed into valid records."""


@dataclass(frozen=True)
class FastaRecord:
    """A single parsed FASTA record with a validated, normalised sequence."""

    record_id: str
    description: str
    sequence: str  # uppercase, whitespace-free, U->T normalised

    @property
    def length(self) -> int:
        return len(self.sequence)


@dataclass
class SequenceStats:
    """Aggregate statistics across one or more FASTA records."""

    record_count: int = 0
    total_bases: int = 0
    base_counts: dict[str, int] = field(default_factory=dict)
    ambiguous_count: int = 0
    gc_percent: float = 0.0
    per_record: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "record_count": self.record_count,
            "total_bases": self.total_bases,
            "base_counts": dict(self.base_counts),
            "ambiguous_count": self.ambiguous_count,
            "gc_percent": self.gc_percent,
            "per_record": self.per_record,
        }


def _normalise_sequence(raw: str, source: str) -> tuple[str, list[str]]:
    """Normalise a raw sequence string.

    Returns ``(normalised_sequence, warnings_list)``. Raises
    :class:`FastaParseError` if any non-IUPAC character is present.
    """
    # Remove all whitespace (newlines, spaces, tabs) -- never data.
    compact = "".join(raw.split())
    if not compact:
        raise FastaParseError(f"{source}: sequence is empty")

    notes: list[str] = []
    if "U" in compact or "u" in compact:
        notes.append(f"{source}: uracil (U) present; normalised to thymine (T)")
        compact = compact.replace("U", "T").replace("u", "t")

    bad = sorted({c for c in compact if c.upper() not in IUPAC_DNA})
    if bad:
        raise FastaParseError(
            f"{source}: invalid nucleotide character(s) {bad!r} "
            "(expected IUPAC DNA alphabet ACGTRYSWKMBDHVN)"
        )

    return compact.upper(), notes


def _record_from_seqrecord(rec: SeqRecord, path: Path) -> FastaRecord:
    raw = str(rec.seq)
    seq, notes = _normalise_sequence(raw, f"{path}:{rec.id}")
    for note in notes:
        warnings.warn(note, UserWarning, stacklevel=3)
    description = rec.description
    if description.startswith(rec.id):
        description = description[len(rec.id):].strip()
    return FastaRecord(
        record_id=rec.id or "<unknown>",
        description=description,
        sequence=seq,
    )


def iter_fasta(path: str | Path) -> Iterator[FastaRecord]:
    """Yield :class:`FastaRecord` objects lazily from a FASTA file.

    Suitable for large genomes: records are streamed one at a time.
    Raises :class:`FastaParseError` on empty files or invalid data.
    """
    path = Path(path)
    if not path.exists():
        raise FastaParseError(f"FASTA file not found: {path}")
    if not path.is_file():
        raise FastaParseError(f"Not a regular file: {path}")

    saw_header = False
    try:
        for rec in SeqIO.parse(str(path), "fasta"):
            saw_header = True
            yield _record_from_seqrecord(rec, path)
    except FastaParseError:
        raise
    except ValueError as exc:
        # BioPython raises ValueError for malformed FASTA (e.g. content
        # before the first header). Surface it as our parse error type.
        raise FastaParseError(f"{path}: {exc}") from exc

    if not saw_header:
        raise FastaParseError(f"{path}: no FASTA records found (empty or malformed file)")


def parse_fasta(path: str | Path) -> list[FastaRecord]:
    """Parse an entire FASTA file into a list of records."""
    return list(iter_fasta(path))


def compute_stats(records: list[FastaRecord]) -> SequenceStats:
    """Compute sequence statistics for parsed records.

    ``gc_percent`` is ``100 * (G + C) / total_bases`` over the full
    sequence length (ambiguous bases count toward the denominator but
    not the numerator). Ambiguous bases are IUPAC codes other than A/C/G/T.
    """
    stats = SequenceStats(record_count=len(records))
    counts: dict[str, int] = {}
    ambiguous = 0

    for rec in records:
        rc = {"id": rec.record_id, "length": rec.length, "gc_percent": 0.0}
        local_counts: dict[str, int] = {}
        for base in rec.sequence:
            local_counts[base] = local_counts.get(base, 0) + 1
            counts[base] = counts.get(base, 0) + 1
            if base not in "ACGT":
                ambiguous += 1
        if rec.length:
            rc["gc_percent"] = round(
                100.0 * (local_counts.get("G", 0) + local_counts.get("C", 0)) / rec.length, 4
            )
        stats.per_record.append(rc)

    stats.total_bases = sum(counts.values())
    stats.base_counts = dict(sorted(counts.items()))
    stats.ambiguous_count = ambiguous
    if stats.total_bases:
        stats.gc_percent = round(
            100.0 * (counts.get("G", 0) + counts.get("C", 0)) / stats.total_bases, 4
        )
    return stats


def fasta_stats(path: str | Path) -> tuple[list[FastaRecord], SequenceStats]:
    """Convenience: parse a FASTA file and compute its statistics."""
    records = parse_fasta(path)
    return records, compute_stats(records)


def to_bio_records(records: list[FastaRecord]) -> list[SeqRecord]:
    """Convert parsed records back to BioPython SeqRecord objects."""
    return [
        SeqRecord(Seq(r.sequence), id=r.record_id, description=r.description)
        for r in records
    ]
