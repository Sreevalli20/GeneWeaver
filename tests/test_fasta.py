"""Tests for geneware.fasta.

All sequences here are small synthetic TEST FIXTURES, not real genomic
data.
"""

from __future__ import annotations

import pytest

from geneware.fasta import (
    FastaParseError,
    compute_stats,
    fasta_stats,
    iter_fasta,
    parse_fasta,
)


def write_tmp(tmp_path, text: str):
    p = tmp_path / "test.fasta"
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_single_record(tmp_path):
    p = write_tmp(tmp_path, ">chr1 some description here\nACGTACGTAC\n")
    records = parse_fasta(p)
    assert len(records) == 1
    rec = records[0]
    assert rec.record_id == "chr1"
    assert "some description" in rec.description
    assert rec.sequence == "ACGTACGTAC"
    assert rec.length == 10


def test_parse_multiple_records_preserves_ids(tmp_path):
    p = write_tmp(
        tmp_path,
        ">chr1 first\nAAAA\n>chr2 second\nCCCCGGGG\n>chr3\nTTTT\n",
    )
    records = parse_fasta(p)
    assert [r.record_id for r in records] == ["chr1", "chr2", "chr3"]
    assert records[1].sequence == "CCCCGGGG"


def test_multiline_sequence_joined(tmp_path):
    p = write_tmp(tmp_path, ">x\nACGT\nACGT\n  ACGT\n")
    assert parse_fasta(p)[0].sequence == "ACGTACGTACGT"


def test_empty_file_raises(tmp_path):
    p = write_tmp(tmp_path, "")
    with pytest.raises(FastaParseError, match="no FASTA records"):
        parse_fasta(p)


def test_whitespace_only_file_raises(tmp_path):
    p = write_tmp(tmp_path, "   \n\n")
    with pytest.raises(FastaParseError):
        parse_fasta(p)


def test_header_without_sequence_raises(tmp_path):
    p = write_tmp(tmp_path, ">chr1 no bases here\n")
    with pytest.raises(FastaParseError, match="empty"):
        parse_fasta(p)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FastaParseError, match="not found"):
        parse_fasta(tmp_path / "does_not_exist.fasta")


def test_invalid_character_rejected(tmp_path):
    p = write_tmp(tmp_path, ">chr1\nACGTXACGT\n")
    with pytest.raises(FastaParseError, match="invalid nucleotide"):
        parse_fasta(p)


def test_ambiguous_bases_preserved(tmp_path):
    p = write_tmp(tmp_path, ">chr1\nACGTNRYMKWSBDHV\n")
    rec = parse_fasta(p)[0]
    assert rec.sequence == "ACGTNRYMKWSBDHV"
    stats = compute_stats([rec])
    assert stats.ambiguous_count == 11  # all except ACGT


def test_uracil_normalised_to_thymine(tmp_path):
    p = write_tmp(tmp_path, ">chr1\nACGUACGT\n")
    with pytest.warns(UserWarning, match="uracil"):
        rec = parse_fasta(p)[0]
    assert rec.sequence == "ACGTACGT"


def test_sequence_stats_known_values(tmp_path):
    # A C G T N -> total 5, GC = 2/5 = 40%
    p = write_tmp(tmp_path, ">chr1\nACGTN\n")
    records, stats = fasta_stats(p)
    assert stats.record_count == 1
    assert stats.total_bases == 5
    assert stats.base_counts == {"A": 1, "C": 1, "G": 1, "N": 1, "T": 1}
    assert stats.gc_percent == 40.0
    assert stats.ambiguous_count == 1
    assert stats.per_record[0]["id"] == "chr1"
    assert stats.per_record[0]["length"] == 5


def test_gc_percent_all_gc(tmp_path):
    p = write_tmp(tmp_path, ">chr1\nGCGCGC\n")
    _, stats = fasta_stats(p)
    assert stats.gc_percent == 100.0


def test_iter_fasta_is_lazy(tmp_path):
    p = write_tmp(tmp_path, ">a\nAAAA\n>b\nCCCC\n")
    it = iter_fasta(p)
    first = next(it)
    assert first.record_id == "a"
    assert next(it).record_id == "b"
