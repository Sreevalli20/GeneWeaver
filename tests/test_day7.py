"""Day 7 tests: full end-to-end workflow and dashboard integration.

Covers FASTA -> stats -> scan -> scoring -> export for a synthetic
FASTA with a known guide+PAM site, plus dashboard state checks and CLI
integration. All expectations are computed by hand, not copied from the
implementation output.
"""

import csv
import json
from pathlib import Path

import pytest
from textual.widgets import Button, Input

from geneware.app import GeneWeaverApp
from geneware.cli import main as cli_main
from geneware.pipeline import run_analysis
from geneware.reports import write_csv, write_html, write_json, write_pdf

GUIDE = "ACGTACGTACGTACGTACGT"  # 20 nt, revcomp-palindrome


def _make_fasta(path: Path) -> Path:
    """chrT: 50 A + guide + AGG PAM + 50 C  -> exactly one + strand hit."""
    seq = "A" * 50 + GUIDE + "AGG" + "C" * 50
    path.write_text(f">chrT synthetic test contig\n{seq}\n", encoding="utf-8")
    return path


@pytest.fixture()
def fasta(tmp_path):
    return _make_fasta(tmp_path / "chrT.fasta")


def test_end_to_end_analysis_values(fasta):
    result = run_analysis(str(fasta), GUIDE, backend="cpu",
                          chunk_size=10_000, overlap=64)

    # Sequence statistics (hand-computed: 123 bases, 62 GC).
    assert result.sequence_stats["record_count"] == 1
    assert result.sequence_stats["total_bases"] == 123
    assert result.sequence_stats["gc_percent"] == pytest.approx(
        100.0 * 62 / 123, abs=1e-3)

    # Exactly one candidate: the engineered + strand site at [50, 70).
    # The '-' strand window at the same locus has PAM revcomp(AAA)=TTT,
    # incompatible with NGG, so require_pam filters it out.
    assert len(result.candidates) == 1
    assert len(result.scored) == 1
    c = result.candidates[0]
    assert c.chromosome == "chrT"
    assert (c.start, c.end) == (50, 70)
    assert c.strand == "+"
    assert c.mismatches == 0
    assert c.mismatch_positions == ()
    assert c.pam == "AGG"
    assert c.pam_compatible is True
    assert c.matched_sequence == GUIDE
    assert c.match_type == "exact"

    # Experimental score: perfect match -> 100.
    sc = result.scored[0]
    assert sc.score == pytest.approx(100.0)
    assert result.score_summary["count"] == 1
    assert result.score_summary["perfect_matches"] == 1
    assert result.stats.candidates_found == 1
    assert result.stats.bases_processed == 123
    assert result.backend == "cpu"


def test_end_to_end_report_contents(fasta, tmp_path):
    result = run_analysis(str(fasta), GUIDE, backend="cpu",
                          chunk_size=10_000, overlap=64)
    out = tmp_path / "reports"

    csv_path = write_csv(result, out / "r.csv")
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert len(rows) == 1
    row = rows[0]
    assert row["chromosome"] == "chrT"
    assert row["start"] == "50" and row["end"] == "70"
    assert row["strand"] == "+" and row["pam"] == "AGG"
    assert row["pam_compatible"] == "True"
    assert row["matched_sequence"] == GUIDE
    assert row["score"] == "100.0000"
    assert row["fasta_filename"] == "chrT.fasta"
    assert "cpu" in row["processing_method"]

    json_path = write_json(result, out / "r.json")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 1
    assert payload["guide"] == GUIDE
    assert payload["sequences_processed"] == 1
    assert payload["matches_found"] == 1
    assert payload["candidates"][0]["start"] == 50
    assert "heuristic" in payload["score_model"]

    html_path = write_html(result, out / "r.html")
    doc = html_path.read_text(encoding="utf-8")
    assert "GeneWeaver" in doc and GUIDE in doc
    assert "chrT" in doc and "Experimental" in doc

    pdf_path = write_pdf(result, out / "r.pdf")
    assert pdf_path.stat().st_size > 500


def test_analysis_rejects_bad_inputs(tmp_path):
    bad = tmp_path / "bad.fasta"
    bad.write_text(">x\nACGTXYZ\n", encoding="utf-8")
    with pytest.raises(Exception):
        run_analysis(str(tmp_path / "missing.fasta"), GUIDE, backend="cpu")
    with pytest.raises(Exception):
        run_analysis(str(bad), GUIDE, backend="cpu")
    fasta = _make_fasta(tmp_path / "ok.fasta")
    with pytest.raises(Exception):
        run_analysis(str(fasta), "NOT_A_GUIDE!!", backend="cpu")
    with pytest.raises(Exception):
        run_analysis(str(fasta), GUIDE, backend="cpu", max_mismatches=-1)
    with pytest.raises(Exception):
        run_analysis(str(fasta), GUIDE, backend="bogus")


def test_cli_analyze_end_to_end(fasta, capsys):
    rc = cli_main(["analyze", str(fasta), "--guide", GUIDE,
                   "--backend", "cpu", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["chromosome"] == "chrT"


@pytest.mark.asyncio
async def test_dashboard_full_workflow(tmp_path):
    """Load genome -> analyze -> results -> overview all share state."""
    fasta = _make_fasta(tmp_path / "chrT.fasta")
    app = GeneWeaverApp()
    async with app.run_test() as pilot:
        app.query_one("#genome-path", Input).value = str(fasta)
        app.query_one("#genome-load", Button).press()
        for _ in range(60):
            await pilot.pause(0.25)
            if app.state.fasta_path:
                break
        assert app.state.fasta_path == str(fasta)
        assert app.state.sequence_stats["total_bases"] == 123

        # Overview shows the loaded file and real stats.
        overview = str(app.query_one("#home-status").render())
        assert "chrT.fasta" in overview and "123" in overview

        app.query_one("#guide-input", Input).value = GUIDE
        app.query_one("#analyze-btn", Button).press()
        for _ in range(120):
            await pilot.pause(0.25)
            if app.state.analysis is not None:
                break
        analysis = app.state.analysis
        assert analysis is not None and analysis.backend == "cpu"
        assert len(analysis.scored) == 1

        # Analyzer status reports completion with real numbers.
        status = str(app.query_one("#analyze-status").render())
        assert "complete" in status and "1" in status

        # Results table carries all 13 columns incl. Guide.
        table = app.query_one("#results-table")
        assert table.row_count == 1
        row = table.get_row_at(0)
        assert row[2] == "chrT"          # chromosome
        assert row[3] == "50"            # start
        assert row[12] == GUIDE          # guide column

        # Overview now reports the finished analysis.
        await pilot.pause(0.2)
        overview = str(app.query_one("#home-status").render())
        assert "cpu" in overview and "chrT.fasta" in overview

        # Cancel with nothing running is a safe no-op.
        app.query_one("#analyze-cancel", Button).press()
        await pilot.pause(0.2)
        assert app.state.analysis is not None


@pytest.mark.asyncio
async def test_dashboard_invalid_guide_error(tmp_path):
    fasta = _make_fasta(tmp_path / "chrT.fasta")
    app = GeneWeaverApp()
    async with app.run_test() as pilot:
        app.query_one("#genome-path", Input).value = str(fasta)
        app.query_one("#genome-load", Button).press()
        for _ in range(60):
            await pilot.pause(0.25)
            if app.state.fasta_path:
                break
        app.query_one("#guide-input", Input).value = "XYZ!"
        app.query_one("#analyze-btn", Button).press()
        await pilot.pause(0.5)
        assert app.state.analysis is None  # rejected before scanning
