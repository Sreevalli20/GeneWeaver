"""Tests for geneware.aligner (CPU baseline).

All sequences are small synthetic TEST FIXTURES, not real genomic data.

Guide under test: ACGTACGTAC (10 nt, IUPAC-clean).
Its reverse complement: GTACGTACGT.
"""

from __future__ import annotations

import pytest

from geneware.aligner import (
    AlignError,
    Candidate,
    ScanStats,
    bases_compatible,
    deduplicate,
    revcomp,
    scan_chunk,
    scan_records,
    validate_guide,
)
from geneware.chunking import GenomeChunk
from geneware.fasta import FastaRecord

GUIDE = "ACGTACGTAC"
RC_GUIDE = "GTACGTACGT"
PAM_PLUS = "AGG"   # compatible with NGG
PAM_MINUS_CTX = "CCT"  # revcomp -> AGG on minus strand


def chunk(seq: str, chrom: str = "chrT", start: int = 0) -> GenomeChunk:
    return GenomeChunk(chromosome=chrom, chunk_index=0, start=start,
                       end=start + len(seq), sequence=seq)


# ---------- guide validation ----------

def test_validate_guide_ok():
    assert validate_guide("acgtacgtac") == "ACGTACGTAC"
    assert validate_guide("  ACGT ACGTAC  ") == "ACGTACGTAC"
    assert validate_guide("ACGTACGTAC") == "ACGTACGTAC"


def test_validate_guide_rejects_bad_chars():
    with pytest.raises(AlignError, match="invalid"):
        validate_guide("ACGTACGTAX")


def test_validate_guide_rejects_bad_length():
    with pytest.raises(AlignError, match="length"):
        validate_guide("ACGT")          # too short (<10)
    with pytest.raises(AlignError, match="length"):
        validate_guide("A" * 31)        # too long (>30)
    # custom bounds
    assert validate_guide("ACGT", min_length=4, max_length=4) == "ACGT"


def test_validate_guide_empty():
    with pytest.raises(AlignError, match="empty"):
        validate_guide("   ")


# ---------- helpers ----------

def test_revcomp():
    assert revcomp("ACGTACGTAC") == "GTACGTACGT"
    assert revcomp("AAAACCCCGGGGTTTT") == "AAAACCCCGGGGTTTT"
    assert revcomp(revcomp("ACGTN")) == "ACGTN"


def test_bases_compatible():
    assert bases_compatible("A", "A")
    assert bases_compatible("A", "R")   # R = A/G intersects A
    assert not bases_compatible("A", "C")
    assert bases_compatible("N", "G")
    assert not bases_compatible("A", "T")


# ---------- plus strand ----------

def test_plus_strand_exact_hit_coordinates():
    seq = "T" * 20 + GUIDE + PAM_PLUS + "G" * 20
    cands = list(scan_chunk(chunk(seq), GUIDE))
    assert len(cands) == 1
    c = cands[0]
    assert c.chromosome == "chrT"
    assert (c.start, c.end) == (20, 30)
    assert c.strand == "+"
    assert c.mismatches == 0
    assert c.mismatch_positions == ()
    assert c.pam == "AGG"
    assert c.pam_compatible is True


def test_plus_strand_custom_start_offset():
    seq = "C" * 10 + GUIDE + PAM_PLUS + "C" * 10
    c = chunk(seq, start=500)
    cands = list(scan_chunk(c, GUIDE))
    assert len(cands) == 1
    assert cands[0].start == 510
    assert cands[0].end == 520


def test_mismatch_counting_and_positions():
    # two mismatches vs guide at positions 1 and 8
    variant = "A" + "C" + "GTACGT" + "A" + "C" + "GT" + "T" + "C"
    # build carefully: guide ACGTACGTAC, want mismatches at idx1 (C->T) and idx8 (A->T)
    variant = list(GUIDE)
    variant[1] = "T"
    variant[8] = "T"
    variant = "".join(variant)
    assert variant[1] == "T" and variant[8] == "T"
    seq = "A" * 15 + variant + PAM_PLUS + "A" * 15
    cands = list(scan_chunk(chunk(seq), GUIDE, max_mismatches=2))
    hits = [c for c in cands if c.start == 15]
    assert len(hits) == 1
    c = hits[0]
    assert c.mismatches == 2
    assert c.mismatch_positions == (1, 8)
    assert c.pam == "AGG"


def test_max_mismatches_threshold_filters():
    variant = list(GUIDE)
    variant[0] = "T"
    variant = "".join(variant)
    seq = "A" * 15 + variant + PAM_PLUS + "A" * 15
    assert list(scan_chunk(chunk(seq), GUIDE, max_mismatches=0)) == []
    cands = list(scan_chunk(chunk(seq), GUIDE, max_mismatches=1))
    assert len(cands) == 1 and cands[0].mismatches == 1


# ---------- minus strand ----------

def test_minus_strand_hit():
    seq = "G" * 10 + PAM_MINUS_CTX + RC_GUIDE + "T" * 10
    # protospacer on plus coords [13, 23); minus-strand PAM = revcomp(CCT) = AGG
    cands = [c for c in scan_chunk(chunk(seq), GUIDE) if c.strand == "-"]
    assert len(cands) == 1
    c = cands[0]
    assert (c.start, c.end) == (13, 23)
    assert c.mismatches == 0
    assert c.pam == "AGG"
    assert c.pam_compatible is True


def test_strand_selection():
    seq = "T" * 20 + GUIDE + PAM_PLUS + "G" * 20
    plus_only = list(scan_chunk(chunk(seq), GUIDE, strands=("+",)))
    minus_only = list(scan_chunk(chunk(seq), GUIDE, strands=("-",)))
    assert all(c.strand == "+" for c in plus_only)
    # minus-only scan of a pure plus-strand fixture may still find
    # incidental minus hits; assert it does not report the + hit coords
    assert not any(
        c.strand == "+" and c.start == 20 for c in minus_only
    )


def test_invalid_strand_rejected():
    with pytest.raises(AlignError, match="strand"):
        list(scan_chunk(chunk("A" * 50), GUIDE, strands=("x",)))


# ---------- PAM handling ----------

def test_incompatible_pam_filtered_by_default():
    seq = "T" * 20 + GUIDE + "TTT" + "G" * 20  # TTT not NGG-compatible? N matches T! use CCC? 
    # TTT: pattern N G G -> pos2 G vs T mismatch, pos3 G vs T -> incompatible
    cands = list(scan_chunk(chunk(seq), GUIDE))
    assert all(not (c.start == 20 and c.strand == "+") for c in cands)


def test_pam_mismatch_recorded_when_allowed():
    seq = "T" * 20 + GUIDE + "TTT" + "G" * 20
    cands = [
        c for c in scan_chunk(chunk(seq), GUIDE, require_pam=False)
        if c.start == 20 and c.strand == "+"
    ]
    assert len(cands) == 1
    assert cands[0].pam == "TTT"
    assert cands[0].pam_compatible is False


def test_alternative_pam_pattern():
    # guide followed by AAG; pattern NAG should match, NGG should not
    seq = "T" * 20 + GUIDE + "AAG" + "G" * 20
    assert list(scan_chunk(chunk(seq), GUIDE, pam_patterns=("NGG",))) == [] or \
        all(not (c.start == 20 and c.strand == "+") for c in
            scan_chunk(chunk(seq), GUIDE, pam_patterns=("NGG",)))
    hits = [
        c for c in scan_chunk(chunk(seq), GUIDE, pam_patterns=("NAG",))
        if c.start == 20 and c.strand == "+"
    ]
    assert len(hits) == 1 and hits[0].pam == "AAG" and hits[0].pam_compatible


def test_ambiguous_base_compatibility():
    # genomic N at a guide-A position still counts as compatible (IUPAC)
    variant = list(GUIDE)
    variant[0] = "N"  # guide[0] = 'A'; N expands to ACGT -> compatible
    variant = "".join(variant)
    seq = "T" * 20 + variant + PAM_PLUS + "G" * 20
    hits = [c for c in scan_chunk(chunk(seq), GUIDE) if c.start == 20]
    assert len(hits) == 1 and hits[0].mismatches == 0


# ---------- deduplication ----------

def test_deduplicate_identical_keys():
    c1 = Candidate("chr", 10, 20, "+", 0, (), "AGG", True, GUIDE)
    c2 = Candidate("chr", 10, 20, "+", 0, (), "AGG", True, GUIDE)
    c3 = Candidate("chr", 10, 20, "-", 0, (), "AGG", True, GUIDE)
    c4 = Candidate("chr", 11, 21, "+", 0, (), "AGG", True, GUIDE)
    out = deduplicate([c1, c2, c3, c4])
    assert out == [c1, c3, c4]


def _scan_records_hits(seq: str, chunk_size: int, overlap: int,
                       guide: str = GUIDE, **kw):
    rec = FastaRecord(record_id="chrD", description="dup test", sequence=seq)
    cands, stats = scan_records([rec], guide, chunk_size=chunk_size,
                                overlap=overlap, **kw)
    return cands, stats


def test_chunk_overlap_deduplicates_boundary_hit():
    # hit at [6,19) fits inside TWO overlapping chunks (chunk 20 / overlap 14
    # -> step 6; window needs 13 nt incl. PAM < 14 overlap) -> would be
    # reported twice without deduplication.
    seq = "T" * 6 + GUIDE + PAM_PLUS + "G" * 40
    cands, stats = _scan_records_hits(seq, chunk_size=20, overlap=14)
    plus_hits = [c for c in cands if c.strand == "+" and c.start == 6]
    assert len(plus_hits) == 1, f"expected dedup to 1 hit, got {len(plus_hits)}"
    assert plus_hits[0].end == 16
    assert plus_hits[0].pam == "AGG"


def test_scan_records_stats_are_real():
    seq = "A" * 30 + GUIDE + PAM_PLUS + "C" * 30   # 73 bases
    cands, stats = _scan_records_hits(seq, chunk_size=1000, overlap=64)
    assert stats.bases_processed == 73
    assert stats.windows_evaluated > 0
    assert stats.candidates_found == len(cands)
    assert stats.chunks_scanned == 1
    assert isinstance(stats, ScanStats)


def test_scan_records_multi_record():
    r1 = FastaRecord("chrA", "", "T" * 20 + GUIDE + PAM_PLUS + "G" * 20)
    r2 = FastaRecord("chrB", "", "A" * 50)
    cands, stats = scan_records([r1, r2], GUIDE, chunk_size=1000, overlap=64)
    assert {c.chromosome for c in cands} == {"chrA"}
    assert stats.chunks_scanned == 2


def test_edge_windows_require_full_pam():
    # guide at very start of sequence: '+' hit needs 3 PAM bases after it,
    # so put PAM immediately; hit at seq end without PAM room must not appear.
    seq = GUIDE + PAM_PLUS + "A" * 10
    cands = [c for c in scan_chunk(chunk(seq), GUIDE) if c.strand == "+"]
    assert len(cands) == 1 and cands[0].start == 0 and cands[0].pam == "AGG"

    tail = "A" * 10 + GUIDE  # no room for PAM after position 10
    cands2 = [c for c in scan_chunk(chunk(tail), GUIDE) if c.strand == "+"]
    assert all(c.start != 10 for c in cands2)


def test_scan_chunk_stats_parameter():
    st = ScanStats()
    list(scan_chunk(chunk("A" * 40), GUIDE, stats=st))
    assert st.bases_processed == 40
    assert st.windows_evaluated > 0
