"""Report and export generation for GeneWeaver.

Exports are built exclusively from real :class:`AnalysisResult` data:

* ``write_csv``      — candidate alignment table (UTF-8 CSV)
* ``write_json``     — full analysis summary including settings
* ``write_html``     — self-contained human-readable HTML report
                       (inline CSS, no external resources)

Every report embeds the analysis settings (guide, PAM, mismatch
threshold, strands, backend) so results are reproducible.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import AnalysisResult

CSV_FIELDS = [
    "rank", "score", "chromosome", "start", "end", "strand",
    "mismatches", "mismatch_positions", "pam", "pam_compatible",
    "guide",
]


def _esc(s: object) -> str:
    return html.escape(str(s))


def write_csv(analysis: "AnalysisResult", path: str | Path) -> Path:
    """Write ranked candidates to CSV. Returns the written path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rank, sc in enumerate(analysis.scored, start=1):
            c = sc.candidate
            writer.writerow({
                "rank": rank,
                "score": sc.score,
                "chromosome": c.chromosome,
                "start": c.start,
                "end": c.end,
                "strand": c.strand,
                "mismatches": c.mismatches,
                "mismatch_positions": ";".join(map(str, c.mismatch_positions)),
                "pam": c.pam,
                "pam_compatible": c.pam_compatible,
                "guide": c.guide,
            })
    return path


def write_json(analysis: "AnalysisResult", path: str | Path) -> Path:
    """Write the full analysis summary (settings + results) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = analysis.analysis_dict()
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


def write_html(analysis: "AnalysisResult", path: str | Path) -> Path:
    """Write a self-contained HTML report (no external assets)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    a = analysis
    rows = []
    for rank, sc in enumerate(a.scored, start=1):
        c = sc.candidate
        rows.append(
            "<tr>"
            f"<td>{rank}</td><td>{sc.score:.1f}</td>"
            f"<td>{_esc(c.chromosome)}</td>"
            f"<td>{c.start}</td><td>{c.end}</td>"
            f"<td>{c.strand}</td><td>{c.mismatches}</td>"
            f"<td>{_esc(';'.join(map(str, c.mismatch_positions)) or '-')}</td>"
            f"<td>{_esc(c.pam)}</td>"
            f"<td>{'yes' if c.pam_compatible else 'no'}</td>"
            "</tr>"
        )
    table = (
        "<table><thead><tr>"
        "<th>#</th><th>Score</th><th>Chr</th><th>Start</th><th>End</th>"
        "<th>Strand</th><th>MM</th><th>MM pos</th><th>PAM</th><th>PAM ok</th>"
        "</tr></thead><tbody>" + "".join(rows)
        + ("</tbody></table>" if rows else
           "<tr><td colspan=\"10\"><em>No candidates found.</em></td></tr>"
           "</tbody></table>")
    )

    sstats = a.sequence_stats
    scan = a.stats.to_dict()
    summ = a.score_summary
    extra_rows = "".join(
        f"<tr><td>{_esc(k)}</td><td><code>{_esc(v)}</code></td></tr>"
        for k, v in sorted(a.extra.items())
    )

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>GeneWeaver Analysis Report</title>
<style>
body {{ font-family: Segoe UI, Arial, sans-serif; margin: 2rem; color: #1c2430; }}
h1 {{ color: #1f4e79; }} h2 {{ color: #2e75b6; margin-top: 1.6rem; }}
table {{ border-collapse: collapse; width: 100%; margin: .6rem 0; }}
th, td {{ border: 1px solid #b9c6d8; padding: .3rem .5rem; font-size: .9rem; }}
th {{ background: #dce6f2; text-align: left; }}
tr:nth-child(even) td {{ background: #f4f7fb; }}
.meta td:first-child {{ font-weight: 600; width: 14rem; }}
.note {{ background: #fff4e5; border: 1px solid #e0a458; padding: .6rem .8rem;
        border-radius: 4px; }}
code {{ background: #eef1f5; padding: 0 .25rem; }}
</style></head><body>
<h1>GeneWeaver — CRISPR Alignment Report</h1>
<p class="note"><strong>Experimental tool.</strong> The score column uses an
internally designed heuristic ranking score; it is <em>not</em> a validated
biological off-target prediction.</p>

<h2>Analysis settings</h2>
<table class="meta">
<tr><td>FASTA file</td><td><code>{_esc(a.fasta_path)}</code></td></tr>
<tr><td>Guide (5'→3')</td><td><code>{_esc(a.guide)}</code></td></tr>
<tr><td>PAM patterns</td><td><code>{_esc(', '.join(a.pam_patterns))}</code></td></tr>
<tr><td>Strands</td><td><code>{_esc(','.join(a.strands))}</code></td></tr>
<tr><td>Max mismatches</td><td>{a.max_mismatches}</td></tr>
<tr><td>Require PAM</td><td>{a.require_pam}</td></tr>
<tr><td>Backend</td><td><code>{_esc(a.backend)}</code></td></tr>
<tr><td>Scan wall time</td><td>{a.elapsed_seconds:.6f} s</td></tr>
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

<h2>Score summary</h2>
<table class="meta">
<tr><td>Count</td><td>{summ.get('count')}</td></tr>
<tr><td>Mean score</td><td>{summ.get('mean_score')}</td></tr>
<tr><td>Min / Max score</td><td>{summ.get('min_score')} / {summ.get('max_score')}</td></tr>
<tr><td>Perfect scores (100)</td><td>{summ.get('perfect_matches')}</td></tr>
</table>

<h2>Backend details</h2>
<table class="meta">{extra_rows or '<tr><td colspan="2"><em>none</em></td></tr>'}</table>

<h2>Candidates ({len(a.scored)})</h2>
{table}

<p><small>Generated by GeneWeaver. Coordinates are 0-based half-open
[start, end) on the plus strand of each FASTA record. Score model:
geneware-heuristic-v1 (experimental, not a validated biological
prediction).</small></p>
</body></html>"""

    path.write_text(doc, encoding="utf-8")
    return path


def write_report_bundle(
    analysis: "AnalysisResult",
    out_dir: str | Path,
    stem: str = "geneware_report",
) -> dict[str, Path]:
    """Write CSV + JSON + HTML into ``out_dir``; return path mapping."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return {
        "csv": write_csv(analysis, out_dir / f"{stem}.csv"),
        "json": write_json(analysis, out_dir / f"{stem}.json"),
        "html": write_html(analysis, out_dir / f"{stem}.html"),
    }
