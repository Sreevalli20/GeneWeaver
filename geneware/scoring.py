"""Experimental off-target ranking score for GeneWeaver.

IMPORTANT
---------
This is an *internally designed, experimental ranking score*. It is NOT
a medically validated biological prediction and must not be presented as
one. It exists to order candidate protospacer matches by a simple,
fully documented heuristic so users can triage results.

Scoring model
-------------
Each candidate starts at 100 points and is penalised:

* ``MISMATCH_PENALTY`` (10 points) per mismatched protospacer position.
* ``SEED_PENALTY`` (5 points) *additional* per mismatch located in the
  "seed" region — the last ``SEED_LEN`` (8) positions of the protospacer
  in guide orientation, i.e. the positions immediately adjacent to the
  PAM. Mismatches near the PAM are empirically more disruptive for
  Cas9 binding, so they cost more.
* ``PAM_PENALTY`` (15 points) when the PAM is incompatible with every
  configured pattern (only possible when ``require_pam=False``).

The final score is clamped to ``[0, 100]``. Higher is better. Ties are
broken by fewer mismatches, then by genomic coordinate, for stable
ordering.

The constants are module-level so the formula can be tuned or audited
in one place. The score is computed purely from fields already present
on :class:`geneware.aligner.Candidate`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .aligner import Candidate

MISMATCH_PENALTY = 10.0
SEED_PENALTY = 5.0
SEED_LEN = 8
PAM_PENALTY = 15.0
MAX_SCORE = 100.0


@dataclass(frozen=True)
class ScoredCandidate:
    """A candidate plus its experimental score."""

    candidate: Candidate
    score: float

    def to_dict(self) -> dict[str, object]:
        d = self.candidate.to_dict()
        d["score"] = self.score
        d["score_model"] = SCORE_MODEL_NAME
        return d


SCORE_MODEL_NAME = "geneware-heuristic-v1"


def score_candidate(cand: Candidate) -> float:
    """Compute the experimental score for one candidate (0-100)."""
    L = len(cand.guide)
    seed_start = max(0, L - SEED_LEN)
    seed_mm = sum(1 for p in cand.mismatch_positions if p >= seed_start)
    score = MAX_SCORE
    score -= MISMATCH_PENALTY * cand.mismatches
    score -= SEED_PENALTY * seed_mm
    if not cand.pam_compatible:
        score -= PAM_PENALTY
    return max(0.0, score)


def score_candidates(candidates: list[Candidate]) -> list[ScoredCandidate]:
    """Score and rank candidates (best first, deterministic ties)."""
    scored = [ScoredCandidate(c, score_candidate(c)) for c in candidates]
    scored.sort(
        key=lambda sc: (
            -sc.score,
            sc.candidate.mismatches,
            sc.candidate.chromosome,
            sc.candidate.start,
            sc.candidate.end,
            sc.candidate.strand,
        )
    )
    return scored


def score_formula_text() -> str:
    """Human-readable explanation of the scoring model (for UI/reports)."""
    return (
        "Experimental ranking score (geneware-heuristic-v1), NOT a "
        "validated biological prediction.\n"
        f"  start at {MAX_SCORE:.0f}\n"
        f"  - {MISMATCH_PENALTY:.0f} per mismatch\n"
        f"  - {SEED_PENALTY:.0f} extra per mismatch in the last "
        f"{SEED_LEN} protospacer positions (PAM-proximal seed region)\n"
        f"  - {PAM_PENALTY:.0f} if PAM incompatible with all patterns\n"
        "  clamped to [0, 100]; higher is better."
    )


def score_summary(scored: list[ScoredCandidate]) -> dict[str, object]:
    """Aggregate stats over scored results (all real measurements)."""
    if not scored:
        return {"count": 0, "mean_score": None, "min_score": None,
                "max_score": None, "perfect_matches": 0}
    scores = [s.score for s in scored]
    return {
        "count": len(scored),
        "mean_score": round(sum(scores) / len(scores), 4),
        "min_score": min(scores),
        "max_score": max(scores),
        "perfect_matches": sum(1 for s in scored if s.score >= MAX_SCORE),
    }
