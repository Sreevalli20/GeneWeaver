"""Tests for the experimental scoring module."""

import pytest

from geneware.aligner import Candidate, validate_guide
from geneware.scoring import (
    MISMATCH_PENALTY,
    PAM_PENALTY,
    SEED_PENALTY,
    score_candidate,
    score_candidates,
    score_formula_text,
    score_summary,
)


def _cand(guide="ACGTACGTACGT", mm=0, mm_pos=(), pam="NGG", pam_ok=True,
          strand="+", start=0, chrom="chr1"):
    return Candidate(
        chromosome=chrom, start=start, end=start + len(guide),
        strand=strand, mismatches=mm, mismatch_positions=tuple(mm_pos),
        pam=pam, pam_compatible=pam_ok, guide=guide,
    )


def test_perfect_match_scores_100():
    assert score_candidate(_cand()) == 100.0


def test_each_mismatch_costs_base_penalty():
    c = _cand(mm=2, mm_pos=(0, 1))  # positions 0,1 are distal (5')
    expected = 100.0 - 2 * MISMATCH_PENALTY
    assert score_candidate(c) == pytest.approx(expected)


def test_seed_region_mismatch_costs_extra():
    guide = "ACGTACGTACGT"  # len 12, seed = last 8 -> positions 4..11
    distal = _cand(guide=guide, mm=1, mm_pos=(0,))
    seed = _cand(guide=guide, mm=1, mm_pos=(11,))
    assert score_candidate(seed) == pytest.approx(
        score_candidate(distal) - SEED_PENALTY)


def test_pam_incompatible_penalty():
    ok = _cand(pam="NGG", pam_ok=True)
    bad = _cand(pam="NAA", pam_ok=False)
    assert score_candidate(bad) == pytest.approx(
        score_candidate(ok) - PAM_PENALTY)


def test_score_clamped_at_zero():
    c = _cand(mm=20, mm_pos=tuple(range(20)), pam_ok=False)
    assert score_candidate(c) == 0.0


def test_ranking_orders_best_first():
    cands = [
        _cand(mm=3, mm_pos=(0, 5, 10), start=30),
        _cand(mm=0, start=10),
        _cand(mm=1, mm_pos=(11,), start=20),
    ]
    scored = score_candidates(cands)
    assert scored[0].candidate.mismatches == 0
    assert scored[-1].candidate.mismatches == 3
    assert all(s.score for s in scored)


def test_summary_counts():
    scored = score_candidates([_cand(), _cand(mm=1, mm_pos=(0,), start=5)])
    s = score_summary(scored)
    assert s["count"] == 2
    assert s["perfect_matches"] == 1
    assert s["max_score"] == 100.0
    assert score_summary([])["count"] == 0


def test_formula_text_explains_model():
    text = score_formula_text()
    assert "experimental" in text.lower()
    assert "not" in text.lower()


def test_scored_to_dict_has_score():
    d = score_candidates([_cand()])[0].to_dict()
    assert d["score"] == 100.0
    assert "chromosome" in d
