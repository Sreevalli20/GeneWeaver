# GeneWeaver — GPU-Accelerated CRISPR Alignment Engine

GeneWeaver is a Python application for CRISPR guide-RNA off-target style
protospacer search. It parses real FASTA genomes, scans both DNA strands
for guide matches with configurable PAM and mismatch tolerance, ranks
hits with an experimental scoring model, and can execute on CPU, GPU
(Numba CUDA — verified on the RTX 5050 in this repo), or a
Dask-distributed CPU cluster — with real measured benchmarks and
exportable reports.

> **Honesty note:** the off-target *score* is an internally designed
> heuristic for ranking candidates. It is **not** a medically validated
> biological prediction. GPU metrics are only ever displayed when a GPU
> backend genuinely executes; otherwise the app reports GPU as
> unavailable with the exact requirements.

## Environment audit

* **Python:** 3.14.6 (`C:\Python314\python.exe`) — verified, unchanged.
* **GPU:** NVIDIA GeForce RTX 5050 Laptop (8 GB), driver 592.82,
  CUDA runtime 13.1 per `nvidia-smi`. Compute capability 12.0 (sm_120).
* **GPU status on this machine:** **available**. numba 0.68.1
  JIT-compiles CUDA via LLVM/NVVM (no `nvcc` needed). The CUDA *libraries*
  (nvvm/nvrtc/cudart/libdevice) are not part of a toolkit install, so
  geneware auto-links them from the `nvidia-cuda-*-cu12` pip wheels into
  a CUDA_HOME-layout directory under the user cache before probing
  (`geneware/gpu.py:_ensure_cuda_home`). `cupy` is not installed; the
  numba backend is used. Run `python -m geneware.cli gpu-status` to probe.

### Dependency compatibility (Python 3.14.6, Windows x64)

| Package      | Status | Notes |
|--------------|--------|-------|
| biopython    | 1.88   | `Bio.SeqIO` needs the documented AlignIO shim (WDAC blocks `Bio.Align._codonaligner`); FASTA parsing unaffected |
| numpy        | 2.5.4  | upgraded from 1.26.4 (no cp314 wheel) |
| textual      | 8.2.8  | terminal dashboard |
| dask         | 2024.12.1 | distributed CPU chunk processing |
| distributed  | 2024.12.1 | LocalCluster backend |
| numba        | 0.68.1 | GPU backend — works on this machine |
| nvidia-cuda-nvcc-cu12 / -nvrtc-cu12 / -runtime-cu12 | 12.8.x | provide nvvm/nvrtc/cudart/libdevice DLLs for numba (auto-linked) |
| cupy         | not installed | optional alternative GPU backend |
| pytest       | 8.3.3  | test runner |

## Project layout

```
geneware/
  __init__.py      package marker + version
  __main__.py      `python -m geneware` entry point
  fasta.py         FASTA parsing + validation + statistics (BioPython)
  chunking.py      memory-conscious genome chunker with coordinates
  aligner.py       CPU protospacer/PAM scanner (IUPAC-aware, both strands)
  scoring.py       experimental off-target ranking score (heuristic-v1)
  gpu.py           GPU detection (numba.cuda / cupy) + CUDA kernel path
  dask_engine.py   Dask-distributed chunk processing + worker status
  pipeline.py      unified backend dispatcher + timed benchmarks
  benchmark.py     real timing/throughput/memory measurements
  reports.py       CSV / JSON / HTML report export
  app.py           Textual dashboard — 10 functional tabs
  cli.py           argparse CLI (parse/scan/bench/analyze/report/gpu-status)
tests/             pytest suite (85 tests)
data/example.fasta small synthetic demo genome (test fixture)
requirements.txt   pinned dependencies (+ optional GPU notes)
pytest.ini         pytest-asyncio configuration
```

## Implemented functionality

### Genome file management (`fasta.py`, Genome tab)
* BioPython `SeqIO` streaming parser; multi-record FASTA support.
* Record IDs, descriptions, lengths, per-record and aggregate GC%,
  base counts, ambiguous-base counts — all real measurements.
* IUPAC DNA validation; invalid characters raise `FastaParseError`
  (never silently dropped); uracil normalised to thymine with warning.
* Lazy per-record iteration for large files; clear error messages.

### CRISPR off-target analysis (`aligner.py`, `pipeline.py`)
* Guide validation (IUPAC alphabet, length bounds, U→T).
* Both-strand search: `+` direct, `-` via reverse complement.
* IUPAC-aware base compatibility; configurable PAM patterns
  (default `NGG`, Cas9 geometry, PAM-proximal to protospacer).
* Configurable max mismatches; mismatch positions recorded.
* 0-based half-open `[start, end)` coordinates on the plus strand.
* Chunk-overlap deduplication (key: chr/start/end/strand).

### Biological scoring (`scoring.py`, Scoring tab)
* Experimental heuristic `geneware-heuristic-v1`, clearly labelled as
  **not** a validated biological prediction:
  `score = 100 − 10·mismatches − 5·(mismatches in last 8 protospacer
  positions) − 15·(PAM incompatible)`, clamped to [0, 100].
* Seed-region (PAM-proximal) mismatches cost extra. Full formula shown
  in the UI and every exported report.

### GPU acceleration (`gpu.py`) — verified working on this laptop
* Real detection probe: `numba.cuda` first, `cupy` second; reports
  device name, VRAM (`cuda.mem_get_info`), and compute capability
  (RTX 5050, CC 12.0) when present.
* `_ensure_cuda_home()` auto-provisions the CUDA libraries numba needs
  from the `nvidia-*-cu12` pip wheels (or respects a user-set
  `CUDA_HOME`/`CUDA_PATH` / CUDA toolkit).
* Real CUDA implementation via `@cuda.jit` kernels (one thread per
  window start, per strand) with device buffers. Every GPU hit is
  re-verified on CPU with the exact IUPAC/PAM semantics before being
  reported, so coordinates, mismatch positions, and scores are always
  consistent with the CPU path.
* `scan_with_fallback` returns a truthful `'gpu'`/`'cpu'` backend tag.
* **Conservative difference:** the kernels compare exact ACGT codes, so
  windows containing ambiguous (IUPAC) bases can be filtered out even
  when the CPU IUPAC-aware matcher would accept them. On pure-ACGT
  sequence, GPU and CPU results are identical (verified by test).
* On machines without numba/CUDA libraries the UI and `gpu-status`
  report GPU as unavailable with the exact requirements.

### Dask processing (`dask_engine.py`, Dask tab)
* Genome split into `GenomeChunk`s; each chunk is an independent task
  submitted to a `distributed.LocalCluster` (processes or threads).
* `as_completed` progress tracking; per-task error capture (failed
  workers are counted and listed, never faked).
* Results merged with the same `deduplicate` as sequential scan.
* `dask_status()` reports real worker/thread/memory measurements.
* Backend honestly reported as `dask-processes`, `dask-threads`, or
  `dask-external`. No distributed-GPU claims.

### Dashboard (`app.py`) — 10 tabs, every control functional
1. **Home** — live status (genome/GPU/Dask/last analysis), quick start.
2. **Genome** — load FASTA or example, record table, real stats.
3. **Analyzer** — guide, PAM, strand, max mismatches, require-PAM,
   backend select, workers, progress bar, analyze.
4. **Results** — metrics line (sequences, bases, windows, matches,
   mismatches, backend, elapsed), sortable DataTable (click headers to
   sort, click again to reverse), min-score/chromosome/strand/PAM/coordinate
   search filters, quick CSV/JSON/HTML export, explicit empty-state
   message when nothing matches.
5. **Scoring** — formula explanation + last-run score summary.
6. **Performance** — benchmark runner for cpu/gpu/dask backends.
7. **Dask** — real scheduler status + one-click Dask analysis.
8. **Benchmarks** — comparison table of every recorded run.
9. **Reports** — write CSV/JSON/HTML/bundle to the configured dir.
10. **Settings** — chunk size, overlap, workers, processes, out dir.

### Benchmarking (`benchmark.py`, `pipeline.py`)
* `perf_counter_ns` timing split parse/align/total; `tracemalloc` peak
  Python memory (reported `n/a` when unavailable); bases/s and
  windows/s throughput; backend recorded in notes.
* CPU-vs-GPU comparison only appears when both backends actually ran —
  never fabricated.

### Reports (`reports.py`)
* CSV (ranked candidates), JSON (full settings + stats + candidates),
  self-contained HTML (inline CSS, no external assets). Every report
  embeds guide, PAM, mismatch threshold, strands, backend, timings.

## Installation

```bat
cd C:\Users\Sreevalli\Downloads\geneware
python -m pip install -r requirements.txt
```

## Usage

### Terminal dashboard

```bat
python -m geneware
```

Workflow: **Genome** tab → Load Genome (or Load Example) → **Analyzer**
tab → enter guide, PAM, mismatches, backend → **Analyze** → review in
**Results** → export in **Reports**.

### CLI

```bat
python -m geneware.cli parse data\example.fasta
python -m geneware.cli scan data\example.fasta --guide CCCCggggnnnnAAAACCCC
python -m geneware.cli analyze data\example.fasta --guide CCCCggggnnnnAAAACCCC --backend cpu
python -m geneware.cli analyze data\example.fasta --guide CCCCggggnnnnAAAACCCC --backend dask --threads
python -m geneware.cli bench data\example.fasta --guide CCCCggggnnnnAAAACCCC
python -m geneware.cli report data\example.fasta --guide CCCCggggnnnnAAAACCCC --out-dir reports
python -m geneware.cli gpu-status
```

Common options: `--chunk-size`, `--overlap`, `--pam`, `--strand`,
`--max-mismatches`, `--allow-pam-mismatch`, `--backend cpu|gpu|dask`,
`--workers`, `--threads`, `--json`.

## Testing

```bat
python -m pytest tests -q
```

85 tests cover: FASTA parsing edge cases, chunk boundaries, both
strands, PAM matching, mismatch positions, deduplication, scoring
formula/ranking, GPU detection contract + CPU fallback + real GPU
execution parity with CPU (skipped automatically when no GPU backend
is usable), Dask scan equality with sequential results, report
generation, pipeline backends, and Textual app smoke tests (tab
mounting, genome load, full CPU analysis, results metrics, filtering
empty state).

## Coordinate & matching conventions

* Coordinates: 0-based, half-open `[start, end)` on the plus strand of
  each FASTA record (`record_id` = chromosome).
* `-` strand candidates match the reverse complement of `[start, end)`.
* PAM = 3 bases immediately 3' of the protospacer **on the candidate
  strand**; edge windows that can't fit protospacer+PAM are skipped.
* IUPAC-aware matching throughout; ambiguous bases preserved.

## Known limitations

* **GPU vs CPU on ambiguous bases:** the CUDA kernels use exact ACGT
  codes, so windows containing IUPAC ambiguity codes (N, R, ...) are
  conservatively filtered even when the CPU IUPAC-aware matcher would
  accept them. On unambiguous sequence the two backends return
  identical results (covered by `test_gpu_scan_matches_cpu_exactly`).
* **GPU requires the CUDA libraries:** numba needs nvvm/nvrtc/cudart/
  libdevice. This is handled automatically when the
  `nvidia-cuda-*-cu12` pip wheels (or a CUDA toolkit / `CUDA_HOME`) are
  present; otherwise `gpu-status` reports the exact reason.
* **Pure-Python CPU inner loop** — adequate for small/medium loci;
  chromosome-scale scans will be slow (measured honestly by `bench`).
  Numba CPU JIT (`@njit`, no CUDA needed) is a planned Week-3
  optimisation.
* Dask workers are CPU-only here; distributed GPU is not implemented.
* `tracemalloc` measures Python allocations only; GPU VRAM is only
  reported when a GPU backend actually runs.
* `Bio.AlignIO` and compiled BioPython modules unusable under WDAC
  (shim covers FASTA needs).
* Chunk overlap must be ≥ guide length + PAM length (default 64).
