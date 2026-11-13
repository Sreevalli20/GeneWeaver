# GeneWeaver — GPU-Accelerated CRISPR Alignment Engine

GeneWeaver is a Python application for CRISPR guide-RNA off-target style
protospacer search. It parses real FASTA genomes, scans both DNA strands
for guide matches with configurable PAM and mismatch tolerance, ranks
hits with an experimental scoring model, and can execute on CPU, GPU
(Numba CUDA, when the environment permits), or a Dask-distributed CPU
cluster — with real measured benchmarks and exportable reports.

> **Honesty note:** the off-target *score* is an internally designed
> heuristic for ranking candidates. It is **not** a medically validated
> biological prediction. GPU metrics are only ever displayed when a GPU
> backend genuinely executes; otherwise the app reports GPU as
> unavailable with the exact requirements.

## Environment audit

* **Python:** 3.14.6 (`C:\Python314\python.exe`) — verified, unchanged.
* **GPU:** NVIDIA GeForce RTX 5050 Laptop (8 GB), driver 592.82,
  CUDA runtime 13.1 per `nvidia-smi`. `nvcc` is **not** installed —
  and none is needed for the numba backend (it JIT-compiles via
  LLVM/NVVM).
* **GPU status on this machine:** **unavailable**. numba 0.68.1 (the
  first release with cp314 wheels) installs, but Windows Application
  Control / WDAC blocks its native `_helperlib` DLL at import
  (`ImportError: DLL load failed ... Application Control policy has
  blocked this file`) — the same policy that blocks BioPython's
  `Bio.Align._codonaligner`. `cupy` is not installed. The GPU code path
  is fully implemented and activates automatically on any machine where
  numba (or cupy) can actually run; run
  `python -m geneware.cli gpu-status` to probe.

### Dependency compatibility (Python 3.14.6, Windows x64)

| Package      | Status | Notes |
|--------------|--------|-------|
| biopython    | 1.88   | `Bio.SeqIO` needs the documented AlignIO shim (WDAC blocks `Bio.Align._codonaligner`); FASTA parsing unaffected |
| numpy        | 2.5.4  | upgraded from 1.26.4 (no cp314 wheel) |
| textual      | 8.2.8  | terminal dashboard |
| dask         | 2024.12.1 | distributed CPU chunk processing |
| distributed  | 2024.12.1 | LocalCluster backend |
| numba        | 0.68.1 installed, **blocked by WDAC** | GPU path present but cannot execute here |
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
tests/             pytest suite (82 tests)
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

### GPU acceleration (`gpu.py`)
* Real detection probe: `numba.cuda` first, `cupy` second; reports
  device name, VRAM, compute capability when present.
* Real CUDA implementation via `@cuda.jit` kernels (one thread per
  window start, per strand) with device buffers, plus CPU-side
  IUPAC/PAM verification of every GPU hit before reporting.
* `scan_with_fallback` returns a truthful `'gpu'`/`'cpu'` backend tag.
* On this machine GPU is **unavailable** (WDAC blocks numba DLLs);
  the UI and `gpu-status` say so explicitly with requirements.

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
4. **Results** — sortable DataTable (click headers), score-ranked,
   min-score/chromosome/strand filters, quick CSV/JSON/HTML export.
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

82 tests cover: FASTA parsing edge cases, chunk boundaries, both
strands, PAM matching, mismatch positions, deduplication, scoring
formula/ranking, GPU detection contract + fallback, Dask scan equality
with sequential results, report generation, pipeline backends, and
Textual app smoke tests (tab mounting, genome load, full CPU analysis).
GPU-execution tests skip automatically when no GPU backend is usable.

## Coordinate & matching conventions

* Coordinates: 0-based, half-open `[start, end)` on the plus strand of
  each FASTA record (`record_id` = chromosome).
* `-` strand candidates match the reverse complement of `[start, end)`.
* PAM = 3 bases immediately 3' of the protospacer **on the candidate
  strand**; edge windows that can't fit protospacer+PAM are skipped.
* IUPAC-aware matching throughout; ambiguous bases preserved.

## Known limitations

* **GPU execution unavailable on this machine** — WDAC blocks numba's
  native DLLs (and would likely block cupy/torch DLLs too). To enable:
  install numba on a machine without the blocking policy
  (`pip install numba`; no nvcc needed), or `pip install
  cupy-cuda12x`. Then `gpu-status` reports the RTX 5050 and the GPU
  backend runs automatically.
* **Pure-Python CPU inner loop** — adequate for small/medium loci;
  chromosome-scale scans will be slow (measured honestly by `bench`).
  Numba CPU JIT (`@njit`, no CUDA needed) is a planned Week-3
  optimisation that also requires unblocked DLLs.
* Dask workers are CPU-only here; distributed GPU is not implemented.
* `tracemalloc` measures Python allocations only; GPU VRAM metrics are
  only reported when a GPU backend actually runs.
* `Bio.AlignIO` and compiled BioPython modules unusable under WDAC
  (shim covers FASTA needs).
* Chunk overlap must be ≥ guide length + PAM length (default 64).

## Enabling GPU on an unrestricted machine

```bat
pip install numba==0.68.1
python -m geneware.cli gpu-status   # should report available: true
python -m geneware.cli analyze genome.fasta --guide <guide> --backend gpu
```

The CUDA kernels use one thread per window start per strand, 2-bit base
encoding, device-side mismatch counting, and CPU-side IUPAC/PAM
verification of every hit. Ambiguous (non-ACGT) bases never match on
the GPU path — a documented conservative difference from the CPU
IUPAC-aware matcher.
