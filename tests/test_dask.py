"""Tests for the Dask distributed engine (CPU workers only)."""

from pathlib import Path

import pytest

from geneware.dask_engine import (
    DaskScanResult,
    dask_status,
    scan_records_dask,
)
from geneware.fasta import parse_fasta

EXAMPLE = Path(__file__).resolve().parent.parent / "data" / "example.fasta"
GUIDE = "ACGTACGTACGTACGTACGT"


@pytest.fixture(scope="module")
def records():
    return parse_fasta(EXAMPLE)


def test_dask_status_shape():
    st = dask_status()
    assert isinstance(st, dict)
    assert "available" in st


def test_dask_scan_matches_sequential(records):
    from geneware.aligner import scan_records
    seq_c, seq_s = scan_records(records, GUIDE, chunk_size=100, overlap=30)
    dask_r = scan_records_dask(
        records, GUIDE, chunk_size=100, overlap=30,
        n_workers=1, processes=False)
    assert isinstance(dask_r, DaskScanResult)
    assert dask_r.backend == "dask-threads"
    keys_seq = {(c.chromosome, c.start, c.end, c.strand) for c in seq_c}
    keys_dask = {(c.chromosome, c.start, c.end, c.strand)
                 for c in dask_r.candidates}
    assert keys_seq == keys_dask
    assert dask_r.stats.bases_processed == seq_s.bases_processed
    assert dask_r.tasks_failed == 0


def test_dask_dedup_across_chunks(records):
    r = scan_records_dask(records, GUIDE, chunk_size=64, overlap=30,
                          n_workers=1, processes=False)
    keys = [(c.chromosome, c.start, c.end, c.strand)
            for c in r.candidates]
    assert len(keys) == len(set(keys))


def test_dask_result_to_dict(records):
    r = scan_records_dask(records, GUIDE, chunk_size=200, overlap=30,
                          n_workers=1, processes=False)
    d = r.to_dict()
    assert d["chunks_total"] >= 1
    assert "stats" in d and "candidates" in d
