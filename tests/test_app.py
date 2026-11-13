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
