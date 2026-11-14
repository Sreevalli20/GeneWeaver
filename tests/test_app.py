"""Smoke tests for the Textual dashboard (pilot-based, no real scan)."""

import pytest
from textual.widgets import Button, Input

from geneware.app import GeneWeaverApp


@pytest.mark.asyncio
async def test_app_mounts_all_tabs():
    app = GeneWeaverApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        # All ten tabs must exist in the DOM.
        for tid in ("tab-home", "tab-genome", "tab-analyzer", "tab-results",
                    "tab-scoring", "tab-performance", "tab-dask",
                    "tab-benchmarks", "tab-reports", "tab-settings"):
            assert app.query_one(f"#{tid}") is not None


@pytest.mark.asyncio
async def test_genome_load_example():
    app = GeneWeaverApp()
    async with app.run_test() as pilot:
        # The Genome tab is inactive by default; press the button widget
        # directly rather than coordinate-clicking hidden tab content.
        app.query_one("#genome-example", Button).press()
        await pilot.pause(0.5)
        # wait for worker thread
        for _ in range(40):
            await pilot.pause(0.25)
            if app.state.fasta_path:
                break
        assert app.state.fasta_path is not None
        assert app.state.sequence_stats["record_count"] == 2


@pytest.mark.asyncio
async def test_analyze_requires_genome():
    app = GeneWeaverApp()
    async with app.run_test() as pilot:
        app.query_one("#analyze-btn", Button).press()
        await pilot.pause(0.3)
        assert app.state.analysis is None  # nothing ran


@pytest.mark.asyncio
async def test_full_cpu_analysis_flow():
    """Load example genome, set guide, run CPU analysis end-to-end."""
    app = GeneWeaverApp()
    async with app.run_test() as pilot:
        app.query_one("#genome-example", Button).press()
        app.query_one("#guide-input", Input).value = "CCCCGGGGNNNNAAAACCCC"
        for _ in range(60):
            await pilot.pause(0.25)
            if app.state.fasta_path:
                break
        assert app.state.fasta_path is not None
        app.query_one("#analyze-btn", Button).press()
        for _ in range(80):
            await pilot.pause(0.25)
            if app.state.analysis is not None:
                break
        assert app.state.analysis is not None
        assert app.state.analysis.backend == "cpu"
        assert app.state.analysis.stats.bases_processed > 0
        # Results metrics line reports real measurements.
        metrics = str(app.query_one("#results-metrics").render())
        assert "bases" in metrics and "matches found" in metrics
        # Results table is populated.
        table = app.query_one("#results-table")
        assert table.row_count > 0
        # Filtering to an impossible chromosome yields the empty state.
        app.query_one("#f-chrom", Input).value = "no_such_chrom"
        app.query_one("#filter-btn", Button).press()
        await pilot.pause(0.3)
        count = str(app.query_one("#results-count").render())
        assert "0 of" in count or "No matches" in count
        assert table.row_count == 0


@pytest.mark.asyncio
async def test_results_column_sort():
    """Header-click sorting reorders the results table."""
    from textual.widgets import DataTable
    from geneware.aligner import Candidate
    from geneware.scoring import score_candidates

    app = GeneWeaverApp()
    async with app.run_test() as pilot:
        cands = [
            Candidate("chr1", 10, 22, "+", 2, (1, 5), "NGG", True, "A" * 12),
            Candidate("chr1", 30, 42, "-", 0, (), "AGG", True, "A" * 12),
            Candidate("chr2", 5, 17, "+", 1, (3,), "CGG", True, "A" * 12),
        ]
        scored = score_candidates(cands)
        # Fake a completed analysis so metrics/count render correctly.
        from geneware.pipeline import AnalysisResult
        from geneware.aligner import ScanStats
        app.state.analysis = AnalysisResult(
            backend="cpu", guide="A" * 12, pam_patterns=("NGG",),
            strands=("+", "-"), max_mismatches=2, require_pam=True,
            fasta_path="x", records=[], sequence_stats={},
            candidates=cands, scored=scored, stats=ScanStats(),
            score_summary={}, elapsed_seconds=0.0)
        app._fill_results(app.state.analysis)
        table = app.query_one("#results-table")
        assert table.row_count == 3
        # Sort by 'mm' ascending via the header-selected handler.
        event = DataTable.HeaderSelected(table, "mm", 5, "MM")
        app.sort_results(event)
        await pilot.pause(0.1)
        first_mm = str(table.get_row_at(0)[5])
        assert first_mm == "0"
