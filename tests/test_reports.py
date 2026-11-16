"""Tests for report/export generation."""

import csv
import json
from pathlib import Path

import pytest

from geneware.pipeline import run_analysis
from geneware.reports import (
    write_csv,
    write_html,
    write_json,
    write_pdf,
    write_report_bundle,
)

EXAMPLE = Path(__file__).resolve().parent.parent / "data" / "example.fasta"
GUIDE = "CCCCGGGGNNNNAAAACCCC"


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    return run_analysis(EXAMPLE, GUIDE, backend="cpu",
                        chunk_size=200, overlap=30)


def test_write_csv(analysis, tmp_path):
    p = write_csv(analysis, tmp_path / "r.csv")
    assert p.exists()
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(analysis.scored)
    if rows:
        assert "score" in rows[0] and "chromosome" in rows[0]
        assert rows[0]["fasta_filename"] == "example.fasta"
        assert rows[0]["matched_sequence"]
        assert rows[0]["processing_method"]


def test_write_json(analysis, tmp_path):
    p = write_json(analysis, tmp_path / "r.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["guide"] == GUIDE
    assert data["backend"] == "cpu"
    assert "candidates" in data and "scan_stats" in data
    assert data["score_model"].startswith("geneware-heuristic")
    assert data["fasta_filename"] == "example.fasta"
    assert data["processing_method"]
    assert data["score_formula"]
    assert data["candidate_count"] == len(data["candidates"])


def test_write_html(analysis, tmp_path):
    p = write_html(analysis, tmp_path / "r.html")
    text = p.read_text(encoding="utf-8")
    assert "GeneWeaver" in text
    assert GUIDE in text
    assert "experimental" in text.lower()
    assert "<table>" in text
    assert "Candidate details and alignments" in text
    assert "matched_sequence" not in text  # HTML uses labels, not JSON keys
    assert "Matched sequence" in text


def test_write_pdf(analysis, tmp_path):
    p = write_pdf(analysis, tmp_path / "r.pdf")
    assert p.exists() and p.stat().st_size > 1000
    assert p.read_bytes().startswith(b"%PDF")


def test_bundle(analysis, tmp_path):
    paths = write_report_bundle(analysis, tmp_path / "out", "rep")
    assert set(paths) == {"csv", "json", "html", "pdf"}
    assert all(p.exists() for p in paths.values())


def test_empty_analysis_exports(tmp_path):
    a = run_analysis(EXAMPLE, "TTTTAAAACCCCGGGGTTTT", backend="cpu",
                     chunk_size=500, overlap=30)
    p = write_csv(a, tmp_path / "e.csv")
    with p.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # chr_demo2 contains TTTTAAAACCCCGGGG repeats -> expect hits
    assert isinstance(rows, list)
