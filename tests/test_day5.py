"""Day-5 tests: candidate sequence provenance, match typing, PAM
validation, scoring breakdown, views, and report contents."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geneware.aligner import (
    AlignError,
    Candidate,
    revcomp,
    scan_chunk,
    scan_records,
)
from geneware.alignment_view import alignment_segments, candidate_view
from geneware.chunking import GenomeChunk
from geneware.fasta import FastaRecord
from geneware.pipeline import run_analysis
from geneware.reports import write_csv, write_html, write_json, write_pdf
from geneware.scoring import (
    AMBIGUITY_PENALTY,
    score_breakdown,
    score_candidate,
    score_candidates,
)

GUIDE = "ACGTACGTAC"
PAM = "AGG"
RC_GUIDE = revcomp(GUIDE)
RC_PAM = revcomp(PAM)  # CCT on the plus strand before a minus-strand hit


def _chunk(seq: str, chrom: str = "chrD5", start: int = 0) -> GenomeChunk:
    return GenomeChunk(chromosome=chrom, chunk_index=0, start=start,
                       end=start + len(seq), sequence=seq)


def test_forward_candidate_contains_real_sequences_and_coordinates():
    seq = "T" * 12 + GUIDE + PAM + "G" * 12
    c = [c for c in scan_chunk(_chunk(seq), GUIDE)
         if c.strand == "+" and c.start == 12][0]
    assert c.matched_sequence == GUIDE
    assert c.genomic_sequence == GUIDE
    assert c.pam == PAM
    assert (c.pam_start, c.pam_end) == (22, 25)
    assert c.match_type == "exact"
    assert c.to_dict()["matched_sequence"] == GUIDE


def test_reverse_candidate_sequences_are_strand_correct():
    seq = "G" * 8 + RC_PAM + RC_GUIDE + "T" * 8
    start = 8 + len(RC_PAM)
    c = [c for c in scan_chunk(_chunk(seq), GUIDE)
         if c.strand == "-" and c.start == start][0]
    assert c.genomic_sequence == RC_GUIDE
    assert c.matched_sequence == GUIDE
    assert c.pam == PAM
    assert (c.pam_start, c.pam_end) == (start - 3, start)
    assert c.match_type == "exact"


def test_iupac_compatible_position_is_ambiguous_not_mismatch():
    variant = "N" + GUIDE[1:]
    seq = "A" * 10 + variant + PAM + "C" * 10
    c = [c for c in scan_chunk(_chunk(seq), GUIDE)
         if c.strand == "+" and c.start == 10][0]
    assert c.mismatches == 0
    assert c.mismatch_positions == ()
    assert c.ambiguous_positions == (0,)
    assert c.match_type == "near"
    assert score_candidate(c) == pytest.approx(100.0 - AMBIGUITY_PENALTY)


def test_pam_patterns_are_validated():
    seq = "A" * 30
    with pytest.raises(AlignError, match="invalid nucleotide"):
        list(scan_chunk(_chunk(seq), GUIDE, pam_patterns=("NGX",)))
    with pytest.raises(AlignError, match="at least one PAM"):
        list(scan_chunk(_chunk(seq), GUIDE, pam_patterns=()))
    with pytest.raises(AlignError, match="PAM pattern"):
        list(scan_chunk(_chunk(seq), GUIDE, pam_patterns=("",)))


def test_negative_max_mismatches_rejected():
    with pytest.raises(AlignError, match="max_mismatches"):
        list(scan_chunk(_chunk("A" * 40), GUIDE, max_mismatches=-1))


def test_pipeline_normalizes_guide_for_results_and_reports(tmp_path):
    seq = "A" * 10 + GUIDE + PAM + "C" * 10
    p = tmp_path / "norm.fasta"
    p.write_text(f">chrNorm\n{seq}\n", encoding="utf-8")
    result = run_analysis(p, f"  {GUIDE.lower()}  ", backend="cpu",
                          chunk_size=200, overlap=30)
    assert result.guide == GUIDE
    assert all(c.guide == GUIDE for c in result.candidates)
    assert result.analysis_dict()["fasta_filename"] == "norm.fasta"


def test_score_breakdown_matches_score_and_is_deterministic():
    c = Candidate("chr", 0, 10, "+", 1, (8,), "AGG", True, GUIDE,
                  matched_sequence=GUIDE[:-1] + "T",
                  genomic_sequence=GUIDE[:-1] + "T")
    breakdown = score_breakdown(c)
    assert breakdown["score"] == pytest.approx(score_candidate(c))
    assert score_candidate(c) == score_candidate(c)
    ranked = score_candidates([c, Candidate(
        "chr", 20, 30, "+", 0, (), "AGG", True, GUIDE,
        matched_sequence=GUIDE, genomic_sequence=GUIDE)])
    assert ranked[0].candidate.mismatches == 0
    assert ranked[1].candidate.mismatches == 1


def test_candidate_view_recovers_legacy_candidate_from_records():
    rec = FastaRecord("chrV", "view", RC_PAM + RC_GUIDE + "A" * 5)
    start = len(RC_PAM)
    legacy = Candidate("chrV", start, start + len(GUIDE), "-", 0, (),
                       PAM, True, GUIDE)
    view = candidate_view(legacy, [rec])
    assert view.genomic_sequence == RC_GUIDE
    assert view.matched_sequence == GUIDE
    assert view.pam_plus_sequence == RC_PAM
    assert view.pam_sequence == PAM
    kinds = [s["kind"] for s in alignment_segments(view)]
    assert kinds == ["match"] * len(GUIDE)


def test_reports_contain_actual_alignment_fields(tmp_path):
    seq = "A" * 8 + GUIDE + PAM + "C" * 8
    p = tmp_path / "day5.fasta"
    p.write_text(f">chrReport\n{seq}\n", encoding="utf-8")
    analysis = run_analysis(p, GUIDE, backend="cpu",
                            chunk_size=200, overlap=30)
    assert analysis.scored, "fixture should produce at least one candidate"

    csv_path = write_csv(analysis, tmp_path / "r.csv")
    rows = list(csv_path.read_text(encoding="utf-8").splitlines())
    assert "matched_sequence" in rows[0]
    assert GUIDE in rows[1]

    json_path = write_json(analysis, tmp_path / "r.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["candidates"][0]["matched_sequence"] == GUIDE
    assert payload["processing_method"]
    assert payload["total_seconds"] >= payload["elapsed_seconds"] - 1e-9

    html_path = write_html(analysis, tmp_path / "r.html")
    html_text = html_path.read_text(encoding="utf-8")
    assert "Candidate details and alignments" in html_text
    assert "guide     " in html_text or "guide" in html_text
    assert "not a validated" in html_text

    pdf_path = write_pdf(analysis, tmp_path / "r.pdf")
    assert pdf_path.read_bytes().startswith(b"%PDF")
