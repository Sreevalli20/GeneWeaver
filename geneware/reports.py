"""Report and export generation for GeneWeaver.

Exports are built exclusively from real :class:`AnalysisResult` data:

* ``write_csv``      — ranked candidate alignment table (UTF-8 CSV)
* ``write_json``     — full analysis summary including settings
* ``write_html``     — self-contained human-readable HTML report
                       (inline CSS, no external resources)
* ``write_pdf``      — printable PDF report via ReportLab

Every report embeds the analysis settings (FASTA filename, guide, PAM,
mismatch threshold, strands, backend/method), real timings, sequence
statistics, candidate details, and the experimental score explanation.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

from .alignment_view import alignment_segments, candidate_view
from .scoring import score_breakdown, score_formula_text

if TYPE_CHECKING:
    from .pipeline import AnalysisResult

CSV_FIELDS = [
    "fasta_filename", "guide", "pam_patterns", "strands", "max_mismatches",
    "require_pam", "backend", "processing_method", "sequences_processed",
    "candidate_count", "matches_found", "scan_seconds", "preprocess_seconds",
    "total_seconds", "score_model",
    "rank", "score", "score_breakdown", "match_type",
    "chromosome", "start", "end", "pam_start", "pam_end", "strand",
    "mismatches", "mismatch_positions", "ambiguous_positions",
    "pam", "pam_compatible", "matched_sequence", "genomic_sequence",
]


def _esc(s: object) -> str:
    return html.escape(str(s))


def _processing_method(analysis: "AnalysisResult") -> str:
    parts = [analysis.backend]
    extra = analysis.extra or {}
    for key in ("mode", "dask_backend"):
        value = extra.get(key)
        if value and str(value) not in parts:
            parts.append(str(value))
    gpu = extra.get("gpu")
    if isinstance(gpu, dict) and gpu.get("backend") and gpu["backend"] not in parts:
        parts.append(str(gpu["backend"]))
    return "/".join(parts)


def _sequences_processed(analysis: "AnalysisResult") -> int:
    stats = analysis.sequence_stats or {}
    value = stats.get("record_count")
    return int(value) if value is not None else len(analysis.records)


def _candidate_count(analysis: "AnalysisResult") -> int:
    return len(analysis.scored)


def _matches_found(analysis: "AnalysisResult") -> int:
    return int(analysis.stats.candidates_found)


def _seconds(analysis: "AnalysisResult", key: str, default: float) -> float:
    value = (analysis.extra or {}).get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _csv_metadata(analysis: "AnalysisResult") -> dict[str, object]:
    extra = analysis.extra or {}
    return {
        "fasta_filename": Path(analysis.fasta_path).name,
        "guide": analysis.guide,
        "pam_patterns": ",".join(analysis.pam_patterns),
        "strands": ",".join(analysis.strands),
        "max_mismatches": analysis.max_mismatches,
        "require_pam": analysis.require_pam,
        "backend": analysis.backend,
        "processing_method": _processing_method(analysis),
        "sequences_processed": _sequences_processed(analysis),
        "candidate_count": _candidate_count(analysis),
        "matches_found": _matches_found(analysis),
        "scan_seconds": analysis.elapsed_seconds,
        "preprocess_seconds": _seconds(analysis, "preprocess_seconds", 0.0),
        "total_seconds": _seconds(
            analysis, "total_seconds", analysis.elapsed_seconds
        ),
        "score_model": "geneware-heuristic-v1 (experimental, not validated)",
    }


def write_csv(analysis: "AnalysisResult", path: str | Path) -> Path:
    """Write ranked candidates to CSV. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _csv_metadata(analysis)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rank, sc in enumerate(analysis.scored, start=1):
            c = sc.candidate
            row = dict(metadata)
            row.update({
                "rank": rank,
                "score": f"{sc.score:.4f}",
                "score_breakdown": json.dumps(
                    score_breakdown(c), sort_keys=True
                ),
                "match_type": c.match_type,
                "chromosome": c.chromosome,
                "start": c.start,
                "end": c.end,
                "pam_start": c.pam_start,
                "pam_end": c.pam_end,
                "strand": c.strand,
                "mismatches": c.mismatches,
                "mismatch_positions": ";".join(map(str, c.mismatch_positions)),
                "ambiguous_positions": ";".join(map(str, c.ambiguous_positions)),
                "pam": c.pam,
                "pam_compatible": c.pam_compatible,
                "matched_sequence": c.matched_sequence,
                "genomic_sequence": c.genomic_sequence,
            })
            writer.writerow(row)
    return path


def write_json(analysis: "AnalysisResult", path: str | Path) -> Path:
    """Write the full analysis summary (settings + results) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = analysis.analysis_dict()
    payload["processing_method"] = _processing_method(analysis)
    payload["sequences_processed"] = _sequences_processed(analysis)
    payload["candidate_count"] = _candidate_count(analysis)
    payload["matches_found"] = _matches_found(analysis)
    payload["preprocess_seconds"] = _seconds(
        analysis, "preprocess_seconds", 0.0
    )
    payload["total_seconds"] = _seconds(
        analysis, "total_seconds", analysis.elapsed_seconds
    )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _fmt_bytes(n: object) -> str:
    if not isinstance(n, int):
        return "n/a"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n} B"


def _alignment_html(view) -> str:
    if not view.matched_sequence:
        return "<p><em>Sequence context unavailable for this candidate.</em></p>"
    seg = alignment_segments(view)
    guide_html = "".join(
        f'<span class="base">{_esc(s["guide"])}</span>' for s in seg
    )
    match_html = "".join(
        f'<span class="base {s["kind"]}">{_esc(s["matched"])}</span>'
        for s in seg
    )
    marker = "".join(
        "^" if s["kind"] == "mismatch" else "~" if s["kind"] == "ambiguous" else " "
        for s in seg
    )
    return (
        "<div class='alignment'><div class='aln-row'><span class='lbl'>guide   </span>"
        f"<code>{guide_html}</code></div>"
        "<div class='aln-row'><span class='lbl'>candidate</span>"
        f"<code>{match_html}</code></div>"
        f"<div class='aln-row marker'><code>{_esc(marker)}</code></div>"
        f"<div class='aln-row'><span class='lbl'>PAM      </span>"
        f"<code>{_esc(view.pam_sequence)}</code> "
        f"<span class='note-small'>(candidate strand; plus-strand footprint "
        f"{_esc(view.pam_plus_sequence) or 'n/a'})</span></div></div>"
    )


def _candidate_detail_html(analysis: "AnalysisResult", rank: int,
                           sc) -> str:
    c = sc.candidate
    view = candidate_view(c, analysis.records)
    breakdown = score_breakdown(c)
    bd = ", ".join(
        f"{k}={v}" for k, v in breakdown.items() if k != "score"
    )
    return f"""
<details class="cand">
<summary>#{rank} { _esc(c.chromosome) }:{c.start}-{c.end} {c.strand}
 score={sc.score:.1f} type={c.match_type}</summary>
<table class="meta">
<tr><td>Guide (5'→3')</td><td><code>{_esc(view.guide)}</code></td></tr>
<tr><td>Matched sequence</td><td><code>{_esc(view.matched_sequence) or 'n/a'}</code></td></tr>
<tr><td>Plus-strand genomic sequence</td><td><code>{_esc(view.genomic_sequence) or 'n/a'}</code></td></tr>
<tr><td>PAM (candidate strand)</td><td><code>{_esc(view.pam_sequence)}</code></td></tr>
<tr><td>PAM plus-strand footprint</td><td><code>{_esc(view.pam_plus_sequence) or 'n/a'}</code></td></tr>
<tr><td>Coordinates</td><td>{c.start}..{c.end} (0-based half-open); PAM {c.pam_start}..{c.pam_end}</td></tr>
<tr><td>Strand</td><td>{c.strand}</td></tr>
<tr><td>Mismatch count</td><td>{c.mismatches}</td></tr>
<tr><td>Mismatch positions</td><td>{_esc(';'.join(map(str, c.mismatch_positions)) or '-')}</td></tr>
<tr><td>Ambiguous positions</td><td>{_esc(';'.join(map(str, c.ambiguous_positions)) or '-')}</td></tr>
<tr><td>PAM compatible</td><td>{'yes' if c.pam_compatible else 'no'}</td></tr>
<tr><td>Experimental score</td><td>{sc.score:.4f}</td></tr>
<tr><td>Score breakdown</td><td><code>{_esc(bd)}</code></td></tr>
</table>
{_alignment_html(view)}
</details>"""


def write_html(analysis: "AnalysisResult", path: str | Path) -> Path:
    """Write a self-contained HTML report (no external assets)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    a = analysis
    rows = []
    details = []
    for rank, sc in enumerate(a.scored, start=1):
        c = sc.candidate
        view = candidate_view(c, a.records)
        rows.append(
            "<tr>"
            f"<td>{rank}</td><td>{sc.score:.1f}</td>"
            f"<td>{_esc(c.match_type)}</td>"
            f"<td>{_esc(c.chromosome)}</td>"
            f"<td>{c.start}</td><td>{c.end}</td>"
            f"<td>{_esc(str(c.pam_start))}..{c.pam_end}</td>"
            f"<td>{c.strand}</td><td>{c.mismatches}</td>"
            f"<td>{_esc(';'.join(map(str, c.mismatch_positions)) or '-')}</td>"
            f"<td>{_esc(';'.join(map(str, c.ambiguous_positions)) or '-')}</td>"
            f"<td>{_esc(c.pam)}</td>"
            f"<td>{'yes' if c.pam_compatible else 'no'}</td>"
            f"<td><code>{_esc(view.matched_sequence) or 'n/a'}</code></td>"
            f"<td><code>{_esc(view.genomic_sequence) or 'n/a'}</code></td>"
            "</tr>"
        )
        details.append(_candidate_detail_html(a, rank, sc))
    table = (
        "<table><thead><tr>"
        "<th>#</th><th>Score</th><th>Type</th><th>Chr</th><th>Start</th>"
        "<th>End</th><th>PAM coords</th><th>Strand</th><th>MM</th>"
        "<th>MM pos</th><th>Ambig pos</th><th>PAM</th><th>PAM ok</th>"
        "<th>Matched</th><th>Genomic</th>"
        "</tr></thead><tbody>" + "".join(rows)
        + ("</tbody></table>" if rows else
           "<tr><td colspan=\"15\"><em>No candidates found.</em></td></tr>"
           "</tbody></table>")
    )

    sstats = a.sequence_stats
    scan = a.stats.to_dict()
    summ = a.score_summary
    extra_rows = "".join(
        f"<tr><td>{_esc(k)}</td><td><code>{_esc(v)}</code></td></tr>"
        for k, v in sorted(a.extra.items())
    )
    preprocess = _seconds(a, "preprocess_seconds", 0.0)
    total = _seconds(a, "total_seconds", a.elapsed_seconds)

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>GeneWeaver Analysis Report</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 2rem; color: #1c2430; }}
h1 {{ color: #1f4e79; }} h2 {{ color: #2e75b6; margin-top: 1.6rem; }}
table {{ border-collapse: collapse; width: 100%; margin: .6rem 0; }}
th, td {{ border: 1px solid #b9c6d8; padding: .3rem .5rem; font-size: .85rem; }}
th {{ background: #dce6f2; text-align: left; }}
tr:nth-child(even) td {{ background: #f4f7fb; }}
.meta td:first-child {{ font-weight: 600; width: 16rem; }}
.note {{ background: #fff4e5; border: 1px solid #e0a458; padding: .6rem .8rem;
        border-radius: 4px; }}
code {{ background: #eef1f5; padding: 0 .25rem; overflow-wrap: anywhere; }}
.cand {{ margin: .5rem 0; border: 1px solid #c7d3e3; padding: .5rem .7rem;
        border-radius: 4px; }}
.alignment {{ margin-top: .5rem; font-family: Consolas, monospace; }}
.aln-row {{ white-space: pre-wrap; }}
.lbl {{ color: #51606f; }}
.match {{ color: #1e7b34; font-weight: bold; }}
.mismatch {{ color: #b42318; font-weight: bold; text-decoration: underline; }}
.ambiguous {{ color: #b54708; font-weight: bold; }}
.marker {{ color: #b42318; }}
.note-small {{ color: #667085; font-family: Segoe UI, Arial, sans-serif; }}
</style></head><body>
<h1>GeneWeaver — CRISPR Alignment Report</h1>
<p class="note"><strong>Experimental tool.</strong> The score column uses an
internally designed heuristic ranking score; it is <em>not</em> a validated
biological off-target prediction and does not prove that a CRISPR mutation
will occur.</p>

<h2>Analysis settings</h2>
<table class="meta">
<tr><td>FASTA filename</td><td><code>{_esc(Path(a.fasta_path).name)}</code></td></tr>
<tr><td>FASTA path</td><td><code>{_esc(a.fasta_path)}</code></td></tr>
<tr><td>Guide (5'→3')</td><td><code>{_esc(a.guide)}</code></td></tr>
<tr><td>PAM patterns</td><td><code>{_esc(', '.join(a.pam_patterns))}</code></td></tr>
<tr><td>Strands</td><td><code>{_esc(','.join(a.strands))}</code></td></tr>
<tr><td>Max mismatches</td><td>{a.max_mismatches}</td></tr>
<tr><td>Require PAM</td><td>{a.require_pam}</td></tr>
<tr><td>Backend</td><td><code>{_esc(a.backend)}</code></td></tr>
<tr><td>Processing method</td><td><code>{_esc(_processing_method(a))}</code></td></tr>
<tr><td>Preprocess / scan / total time</td><td>{preprocess:.6f} s / {a.elapsed_seconds:.6f} s / {total:.6f} s</td></tr>
<tr><td>Sequences processed</td><td>{_sequences_processed(a):,}</td></tr>
<tr><td>Candidate matches exported</td><td>{len(a.scored):,}</td></tr>
<tr><td>Matches found before export</td><td>{_matches_found(a):,}</td></tr>
</table>

<h2>Score model</h2>
<p><code>{_esc(score_formula_text())}</code></p>
<table class="meta">
<tr><td>Count</td><td>{summ.get('count')}</td></tr>
<tr><td>Mean score</td><td>{summ.get('mean_score')}</td></tr>
<tr><td>Min / Max score</td><td>{summ.get('min_score')} / {summ.get('max_score')}</td></tr>
<tr><td>Perfect scores (100)</td><td>{summ.get('perfect_matches')}</td></tr>
</table>

<h2>Sequence statistics</h2>
<table class="meta">
<tr><td>Records</td><td>{sstats.get('record_count')}</td></tr>
<tr><td>Total bases</td><td>{sstats.get('total_bases'):,}</td></tr>
<tr><td>GC %</td><td>{sstats.get('gc_percent')}%</td></tr>
<tr><td>Ambiguous bases</td><td>{sstats.get('ambiguous_count'):,}</td></tr>
</table>

<h2>Scan statistics</h2>
<table class="meta">
<tr><td>Bases processed</td><td>{scan.get('bases_processed'):,}</td></tr>
<tr><td>Windows evaluated</td><td>{scan.get('windows_evaluated'):,}</td></tr>
<tr><td>Chunks scanned</td><td>{scan.get('chunks_scanned'):,}</td></tr>
<tr><td>Candidates (deduplicated)</td><td>{scan.get('candidates_found'):,}</td></tr>
</table>

<h2>Backend details</h2>
<table class="meta">{extra_rows or '<tr><td colspan="2"><em>none</em></td></tr>'}</table>

<h2>Candidates ({len(a.scored)})</h2>
{table}

<h2>Candidate details and alignments</h2>
{''.join(details) if details else '<p><em>No candidates found.</em></p>'}

<p><small>Generated by GeneWeaver. Coordinates are 0-based half-open
[start, end) on the plus strand of each FASTA record; PAM coordinates are
the plus-strand footprint. Score model: geneware-heuristic-v1
(experimental, not a validated biological prediction).</small></p>
</body></html>"""

    path.write_text(doc, encoding="utf-8")
    return path


def write_pdf(analysis: "AnalysisResult", path: str | Path) -> Path:
    """Write a PDF report using ReportLab (real candidate data only)."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import landscape, letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            "PDF export requires ReportLab: pip install reportlab"
        ) from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    small = styles["BodyText"].clone("SmallBody")
    small.fontSize = 8
    small.leading = 10

    def P(text: object) -> Paragraph:
        return Paragraph(str(text).replace("&", "&amp;").replace("<", "&lt;")
                         .replace(">", "&gt;").replace("\n", "<br/>"), small)

    story = [
        Paragraph("GeneWeaver — CRISPR Alignment Report", styles["Title"]),
        Paragraph(
            "<b>Experimental tool.</b> Scores are an internally designed "
            "heuristic ranking score, not a validated biological "
            "off-target prediction.", styles["BodyText"]
        ),
        Spacer(1, 8),
        Paragraph("Analysis settings", styles["Heading2"]),
    ]

    preprocess = _seconds(analysis, "preprocess_seconds", 0.0)
    total = _seconds(analysis, "total_seconds", analysis.elapsed_seconds)
    meta = [
        ["FASTA filename", Path(analysis.fasta_path).name],
        ["FASTA path", analysis.fasta_path],
        ["Guide (5'->3')", analysis.guide],
        ["PAM patterns", ", ".join(analysis.pam_patterns)],
        ["Strands", ",".join(analysis.strands)],
        ["Max mismatches", str(analysis.max_mismatches)],
        ["Require PAM", str(analysis.require_pam)],
        ["Backend", analysis.backend],
        ["Processing method", _processing_method(analysis)],
        ["Preprocess seconds", f"{preprocess:.6f}"],
        ["Scan seconds", f"{analysis.elapsed_seconds:.6f}"],
        ["Total seconds", f"{total:.6f}"],
        ["Sequences processed", f"{_sequences_processed(analysis):,}"],
        ["Candidate matches", f"{len(analysis.scored):,}"],
        ["Matches found", f"{_matches_found(analysis):,}"],
    ]
    meta_table = Table([[P(k), P(v)] for k, v in meta], colWidths=[130, 340])
    meta_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story += [meta_table, Spacer(1, 10), Paragraph("Score model", styles["Heading2"]),
              Paragraph(score_formula_text().replace("\n", "<br/>"), small)]

    cand_data = [[
        "#", "Score", "Type", "Chr", "Start", "End", "Strand", "MM",
        "MM pos", "Ambig", "PAM", "PAM ok", "Matched", "Genomic",
    ]]
    for rank, sc in enumerate(analysis.scored, start=1):
        c = sc.candidate
        view = candidate_view(c, analysis.records)
        cand_data.append([
            str(rank), f"{sc.score:.1f}", c.match_type, c.chromosome,
            str(c.start), str(c.end), c.strand, str(c.mismatches),
            ";".join(map(str, c.mismatch_positions)) or "-",
            ";".join(map(str, c.ambiguous_positions)) or "-",
            c.pam, "yes" if c.pam_compatible else "no",
            view.matched_sequence or "n/a", view.genomic_sequence or "n/a",
        ])
    if len(cand_data) == 1:
        cand_data.append(["-", "-", "-", "-", "-", "-", "-", "-", "-", "-",
                          "-", "-", "No candidates found.", "-"])

    cand_table = Table(
        cand_data, repeatRows=1,
        colWidths=[18, 36, 32, 48, 30, 30, 26, 18, 42, 20, 32, 22,
                   110, 110],
    )
    cand_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dce6f2")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 6.5),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f4f7fb")]),
    ]))
    story += [
        Paragraph("Candidates", styles["Heading2"]),
        cand_table,
        Spacer(1, 10),
        Paragraph(
            "Generated by GeneWeaver. Coordinates are 0-based half-open "
            "[start, end) on the plus strand; score model "
            "geneware-heuristic-v1 (experimental, not validated).",
            small,
        ),
    ]
    doc = SimpleDocTemplate(
        str(path), pagesize=landscape(letter), leftMargin=30,
        rightMargin=30, topMargin=30, bottomMargin=30,
    )
    doc.build(story)
    return path


def write_report_bundle(
    analysis: "AnalysisResult",
    out_dir: str | Path,
    stem: str = "geneware_report",
) -> dict[str, Path]:
    """Write CSV + JSON + HTML + PDF into ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "csv": write_csv(analysis, out_dir / f"{stem}.csv"),
        "json": write_json(analysis, out_dir / f"{stem}.json"),
        "html": write_html(analysis, out_dir / f"{stem}.html"),
        "pdf": write_pdf(analysis, out_dir / f"{stem}.pdf"),
    }
