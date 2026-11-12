"""Tests for geneware.chunking.

Test fixtures only - small synthetic sequences.
"""

from __future__ import annotations

import pytest

from geneware.chunking import GenomeChunk, iter_chunks, total_chunks
from geneware.fasta import FastaRecord


def rec(seq: str, rid: str = "chrT") -> FastaRecord:
    return FastaRecord(record_id=rid, description="test", sequence=seq)


def test_chunk_coordinates_and_slices():
    seq = "".join(chr(65 + (i % 26)) for i in range(25))
    r = rec(seq)
    chunks = list(iter_chunks(r, chunk_size=10, overlap=2))
    # step = 8 -> starts 0, 8, 16
    assert [(c.start, c.end) for c in chunks] == [(0, 10), (8, 18), (16, 25)]
    for c in chunks:
        assert c.sequence == seq[c.start : c.end]
        assert c.chromosome == "chrT"
    assert [c.chunk_index for c in chunks] == [0, 1, 2]


def test_short_sequence_single_chunk():
    r = rec("ACGT", rid="chrShort")
    chunks = list(iter_chunks(r, chunk_size=100, overlap=10))
    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end) == (0, 4)
    assert chunks[0].chromosome == "chrShort"


def test_exact_multiple_no_extra_chunk():
    # len 20, chunk 10, overlap 0 -> exactly 2 chunks, no empty tail
    r = rec("A" * 20)
    chunks = list(iter_chunks(r, chunk_size=10, overlap=0))
    assert [(c.start, c.end) for c in chunks] == [(0, 10), (10, 20)]


def test_full_coverage_no_gaps():
    seq = "C" * 37
    r = rec(seq)
    chunks = list(iter_chunks(r, chunk_size=10, overlap=3))
    # every position covered; first starts 0, last ends at 37
    assert chunks[0].start == 0
    assert chunks[-1].end == 37
    covered = set()
    for c in chunks:
        covered.update(range(c.start, c.end))
    assert covered == set(range(37))


def test_overlap_equals_step_semantics():
    seq = "G" * 30
    r = rec(seq)
    chunks = list(iter_chunks(r, chunk_size=10, overlap=4))
    # step 6: starts 0,6,12,18,24 -> ends 10,16,22,28,30
    assert [(c.start, c.end) for c in chunks] == [
        (0, 10), (6, 16), (12, 22), (18, 28), (24, 30)
    ]


def test_invalid_parameters():
    r = rec("A" * 50)
    with pytest.raises(ValueError, match="chunk_size"):
        list(iter_chunks(r, chunk_size=0))
    with pytest.raises(ValueError, match="overlap"):
        list(iter_chunks(r, chunk_size=10, overlap=-1))
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        list(iter_chunks(r, chunk_size=10, overlap=10))
    with pytest.raises(ValueError, match="smaller than chunk_size"):
        list(iter_chunks(r, chunk_size=10, overlap=15))


def test_empty_sequence_yields_nothing():
    r = rec("")
    assert list(iter_chunks(r, chunk_size=10)) == []


def test_total_chunks_matches_iteration():
    for n in (1, 10, 11, 25, 100):
        for cs, ov in ((10, 0), (10, 3), (7, 2)):
            r = rec("A" * n)
            assert total_chunks([r], cs, ov) == len(list(iter_chunks(r, cs, ov)))


def test_genome_chunk_is_frozen():
    c = GenomeChunk("chr", 0, 0, 5, "ACGTA")
    with pytest.raises(Exception):
        c.start = 5  # type: ignore[misc]
