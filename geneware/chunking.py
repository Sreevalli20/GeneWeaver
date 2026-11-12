"""Memory-conscious genome chunking for GeneWeaver (Week 1).

Splits large sequences into overlapping windows while preserving
original genomic coordinates (0-based, half-open ``[start, end)`` on the
plus strand of each chromosome/record). Coordinates always refer to the
*original* sequence, never to chunk-local offsets.

Chunk overlap exists so that protospacer+PAM windows crossing a chunk
boundary are not missed. Candidates found in more than one chunk are
deduplicated downstream by the aligner (see ``aligner.scan_chunks``).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from .fasta import FastaRecord


@dataclass(frozen=True)
class GenomeChunk:
    """A contiguous subsequence of a record with original coordinates."""

    chromosome: str          # record_id of the source FASTA record
    chunk_index: int         # 0-based ordinal within the record
    start: int               # 0-based inclusive offset in the source sequence
    end: int                 # 0-based exclusive offset in the source sequence
    sequence: str            # the chunk's bases (uppercase, normalised)

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, object]:
        return {
            "chromosome": self.chromosome,
            "chunk_index": self.chunk_index,
            "start": self.start,
            "end": self.end,
            "length": self.length,
        }


def iter_chunks(
    record: FastaRecord,
    chunk_size: int,
    overlap: int = 0,
) -> Iterator[GenomeChunk]:
    """Yield :class:`GenomeChunk` objects covering ``record``.

    Parameters
    ----------
    record :
        A parsed FASTA record (``record_id`` is used as chromosome id).
    chunk_size :
        Maximum bases per chunk. Must be >= 1.
    overlap :
        Bases shared between consecutive chunks. Must satisfy
        ``0 <= overlap < chunk_size``. Overlap guarantees windows that
        span a boundary are evaluated in full in at least one chunk.

    Guarantees
    ----------
    * Coordinates are absolute positions in the source record.
    * Consecutive chunks overlap by exactly ``overlap`` bases, except the
      final chunk which may be shorter (never empty).
    * A record shorter than ``chunk_size`` yields exactly one chunk
      covering ``[0, length)``.
    * Sequences are sliced lazily; only one chunk string exists at a time
      during iteration, though the source record's sequence is already in
      memory (BioPython parsing is per-record, not per-base streamed).
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    seq = record.sequence
    n = len(seq)
    if n == 0:
        return

    step = chunk_size - overlap
    idx = 0
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        yield GenomeChunk(
            chromosome=record.record_id,
            chunk_index=idx,
            start=start,
            end=end,
            sequence=seq[start:end],
        )
        idx += 1
        if end == n:
            break
        start += step


def chunk_records(
    records: Iterable[FastaRecord],
    chunk_size: int,
    overlap: int = 0,
) -> Iterator[GenomeChunk]:
    """Flatten :func:`iter_chunks` over multiple records."""
    for rec in records:
        yield from iter_chunks(rec, chunk_size, overlap)


def total_chunks(records: Iterable[FastaRecord], chunk_size: int, overlap: int) -> int:
    """Count chunks without materialising sequences (for progress bars)."""
    total = 0
    for rec in records:
        n = len(rec.sequence)
        if n == 0:
            continue
        if n <= chunk_size:
            total += 1
        else:
            step = chunk_size - overlap
            total += 1 + (n - chunk_size + step - 1) // step
    return total
