"""Shared candidate-alignment views for GeneWeaver UI and reports.

The aligner stores the verified protospacer sequences on each
:class:`~geneware.aligner.Candidate`. This module only resolves those
stored values (or re-derives them from the parsed records for legacy
hand-built candidates) so reports never invent illustrative sequence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .aligner import Candidate, _ambiguous_positions, bases_compatible, revcomp
from .fasta import FastaRecord


@dataclass(frozen=True)
class CandidateView:
    """Resolved sequence context for one candidate."""

    candidate: Candidate
    guide: str
    matched_sequence: str       # guide/candidate orientation, 5'->3'
    genomic_sequence: str       # plus-strand protospacer sequence
    pam_sequence: str           # PAM on the candidate strand
    pam_plus_sequence: str      # PAM footprint on the plus strand
    mismatch_positions: tuple[int, ...]
    ambiguous_positions: tuple[int, ...]

    @property
    def match_type(self) -> str:
        return self.candidate.match_type


def _record_map(records: Iterable[FastaRecord] | None) -> dict[str, FastaRecord]:
    if not records:
        return {}
    return {r.record_id: r for r in records}


def candidate_view(
    candidate: Candidate,
    records: Iterable[FastaRecord] | None = None,
) -> CandidateView:
    """Resolve real sequence context for ``candidate``.

    Stored candidate fields are authoritative. For older/hand-built
    candidates with empty sequence fields, the plus-strand protospacer is
    recovered from ``records`` by chromosome and coordinates when possible.
    """
    guide = candidate.guide
    L = len(guide)
    rec = _record_map(records).get(candidate.chromosome)
    plus_window = ""
    if rec is not None:
        plus_window = rec.sequence[candidate.start : candidate.end]
        if len(plus_window) != L:
            plus_window = ""

    genomic = candidate.genomic_sequence
    if len(genomic) != L:
        genomic = plus_window

    if candidate.matched_sequence and len(candidate.matched_sequence) == L:
        matched = candidate.matched_sequence
    elif genomic:
        matched = genomic if candidate.strand == "+" else revcomp(genomic)
    else:
        matched = ""

    if not candidate.ambiguous_positions and matched:
        amb = _ambiguous_positions(guide, matched, candidate.mismatch_positions)
    else:
        amb = candidate.ambiguous_positions

    pam_len = len(candidate.pam)
    pam_plus = ""
    if rec is not None and pam_len:
        if candidate.strand == "+":
            pam_plus = rec.sequence[candidate.end : candidate.end + pam_len]
        else:
            pam_plus = rec.sequence[
                max(0, candidate.start - pam_len) : candidate.start
            ]

    return CandidateView(
        candidate=candidate,
        guide=guide,
        matched_sequence=matched,
        genomic_sequence=genomic,
        pam_sequence=candidate.pam,
        pam_plus_sequence=pam_plus,
        mismatch_positions=candidate.mismatch_positions,
        ambiguous_positions=amb,
    )


def alignment_segments(view: CandidateView) -> list[dict[str, object]]:
    """Per-position guide/match classification for visualisation."""
    mm = set(view.mismatch_positions)
    amb = set(view.ambiguous_positions)
    out: list[dict[str, object]] = []
    for i, (g, m) in enumerate(zip(view.guide, view.matched_sequence, strict=False)):
        if i in mm:
            kind = "mismatch"
        elif i in amb:
            kind = "ambiguous"
        elif bases_compatible(g, m):
            kind = "match"
        else:
            kind = "mismatch"
        out.append({"position": i, "guide": g, "matched": m, "kind": kind})
    return out
