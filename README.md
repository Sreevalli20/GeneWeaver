# GeneWeaver — CRISPR Guide Search Engine

GeneWeaver is a Python application for CRISPR guide-RNA off-target style
protospacer search. It parses real FASTA genomes, scans both DNA strands
for guide matches with configurable PAM and mismatch tolerance, ranks
hits with an experimental scoring model, and exports CSV/JSON/HTML/PDF
reports. The default and verified backend is **CPU** (Numba-JIT engine
with a pure-Python fallback). Optional GPU (Numba CUDA) and Dask
backends exist but only run when genuinely detected — otherwise the app
says so plainly.

> **Honesty note:** the off-target *score* is an internally designed
> heuristic for ranking candidates. It is **not** a medically validated
> biological prediction. GPU metrics are only ever displayed when a GPU
> backend genuinely executes; otherwise the app reports GPU as
> unavailable with the exact requirements.

## Environment audit

* **Python:** 3.14.6 — verified, unchanged.
* **GPU:** NVIDIA GeForce RTX 5050 Laptop is present, but **GPU compute
  is currently unavailable on this machine**: an Application Control /
  WDAC policy blocks the unsigned `numba.cuda` native DLL
  (`mviewbuf`). `python -m geneware.cli gpu-status` reports this
  honestly. The CPU path is unaffected and is the default.
* **CPU engine:** numba 0.68.1 JIT-compiles the scan kernel (no CUDA
  involved). If numba ever fails to import, the encoded pure-Python
  fallback is used automatically (~5x slower, identical results).

### Dependency compatibility (Python 3.14.6, Windows x64)

| Package      | Status | Notes |
|--------------|--------|-------|
| biopython    | 1.88   | `Bio.SeqIO` needs the documented AlignIO shim (WDAC blocks `Bio.Align._codonaligner`); FASTA parsing unaffected |
| numpy        | 2.5.4  | upgraded from 1.26.4 (no cp314 wheel) |
| textual      | 8.2.8  | terminal dashboard |
| reportlab    | 4.2.5  | PDF report export |
| dask         | 2024.12.1 | distributed CPU chunk processing |
| distributed  | 2024.12.1 | LocalCluster backend |
| numba        | 0.68.1 | CPU JIT engine works; `numba.cuda` DLL blocked by WDAC policy |
| nvidia-cuda-nvcc-cu12 / -nvrtc-cu12 / -runtime-cu12 | 12.8.x | CUDA libs present, but unusable while `numba.cuda`'s DLLs are policy-blocked |
| cupy         | not installed | optional alternative GPU backend |
| pytest       | 8.3.3  | test runner |

## Project layout

```
geneware/
  __init__.py      package marker + version
  __main__.py      `python -m geneware` entry point
  fasta.py         FASTA parsing + validation + statistics (BioPython)
  chunking.py      memory-conscious genome chunker with coordinates
  aligner.py       CPU protospacer/PAM scanner — numba-JIT engine with
                   encoded IUPAC table + pure-Python fallback
  scoring.py       experimental off-target ranking score (heuristic-v1)
  alignment_view.py shared real-sequence alignment views for UI/reports
  gpu.py           GPU detection (numba.cuda / cupy) + CUDA kernel path
  dask_engine.py   Dask-distributed chunk processing + worker status
  pipeline.py      unified backend dispatcher + timed benchmarks
  benchmark.py     real timing/throughput/memory measurements
  reports.py       CSV / JSON / HTML / PDF report export
  app.py           Textual dashboard — 10 functional tabs
  cli.py           argparse CLI (parse/scan/bench/analyze/report/gpu-status)
tests/             pytest suite (122 tests)
data/example.fasta small synthetic demo genome (test fixture)
data/bench_20mb.fasta 20 Mbp synthetic benchmark genome (3 records,
                   90 planted guide+PAM hits)
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
* PAM footprint coordinates (`pam_start`/`pam_end`) are reported in
  plus-strand coordinates for both candidate strands.
* Every candidate stores the real `matched_sequence` (guide/candidate
  orientation), `genomic_sequence` (plus strand), and
  `ambiguous_positions` for compatible IUPAC ambiguity.
* `match_type` is `exact` only for an unambiguous zero-mismatch hit;
  all other hits are labelled `near`.
* Chunk-overlap deduplication (key: chr/start/end/strand) and
  deterministic candidate ordering for repeatable reports.
* PAM patterns are validated as non-empty, equal-length IUPAC strings;
  invalid characters and empty pattern lists raise `AlignError`.
* **Day-4 CPU engine:** bases are encoded to 4-bit IUPAC codes and the
  per-window scan runs as table lookups. When `numba` imports, an
  `@njit(nogil=True)` kernel counts mismatches with early exit at
  `max_mismatches+1`; only surviving windows are re-scored on CPU for
  exact IUPAC positions and PAM strings, so results are identical to
  the pure-Python path (verified by parity tests). Force the fallback
  with `GENEWEARE_ALIGNER_ENGINE=python` (useful for A/B benchmarks).
* **Overlap safety:** `validate_overlap` raises a clear `AlignError`
  when `overlap < len(guide)+pam_len` on multi-chunk records, so
  boundary-spanning matches can never be silently missed.

### Biological scoring (`scoring.py`, Scoring tab)
* Experimental heuristic `geneware-heuristic-v1`, clearly labelled as
  **not** a validated biological prediction:
  `score = 100 − 10·mismatches − 5·(mismatches in last 8 protospacer
  positions) − 15·(PAM incompatible) − 2·(compatible IUPAC-ambiguous
  positions)`, clamped to [0, 100].
* Seed-region (PAM-proximal) mismatches cost extra; compatible ambiguity
  lowers confidence without being counted as a mismatch. `score_breakdown`
  exposes every penalty component for the selected candidate and reports.

### GPU acceleration (`gpu.py`) — implemented, currently unavailable here
* Real detection probe: `numba.cuda` first, `cupy` second; reports
  device name, total and free VRAM (`cuMemGetInfo`), and compute
  capability (RTX 5050, CC 12.0) when present.
* `_ensure_cuda_home()` auto-provisions the CUDA libraries numba needs
  from the `nvidia-*-cu12` pip wheels (or respects a user-set
  `CUDA_HOME`/`CUDA_PATH` / CUDA toolkit).
* Real CUDA implementation via a single `@cuda.jit` kernel (one thread
  per window start, shared by both strands via forward/revcomp guide
  codes) compiled once per process. Window starts are `np.arange`
  ranges and only kernel-surviving windows are re-verified on CPU with
  the exact IUPAC/PAM semantics, so coordinates, mismatch positions,
  and scores are always consistent with the CPU path.
* `scan_with_fallback` returns a truthful `'gpu'`/`'cpu'` backend tag.
* **Status on this machine:** `gpu-status` reports `available: false`
  because the Application Control policy blocks numba.cuda's native
  DLLs (`mviewbuf`). GPU execution was verified in an earlier session
  before the policy applied; it will resume automatically if the DLLs
  become loadable — no code change needed.
* **Conservative difference:** the kernels compare exact ACGT codes, so
  windows containing ambiguous (IUPAC) bases can be filtered out even
  when the CPU IUPAC-aware matcher would accept them. On pure-ACGT
  sequence, GPU and CPU results are identical (verified by test).
* On machines without numba/CUDA libraries the UI and `gpu-status`
  report GPU as unavailable with the exact requirements.

### Dask processing (`dask_engine.py`, Dask tab)
* Genome split into `GenomeChunk`s; each chunk is an independent task
  submitted to a `distributed.LocalCluster` (processes or threads).
* **Day-4:** futures are submitted through a rolling window (~4 per
  worker) instead of all at once, bounding peak memory; chunks are
  produced lazily. `as_completed` progress tracking; per-task error
  capture (failed workers are counted and listed, never faked);
  `chunks_pending` is reported alongside completed/total.
* Results merged with the same `deduplicate` as sequential scan.
* `dask_status()` reports real worker/thread/memory measurements.
* Backend honestly reported as `dask-processes`, `dask-threads`, or
  `dask-external`. No distributed-GPU claims.

### Dashboard (`app.py`) — 10 sections, every control functional
1. **Overview** — live state from real app data only: loaded FASTA
   filename, record/base/GC/ambiguity statistics, last analysis
   (backend, candidate and match counts, elapsed time, processing
   method, guide), latest benchmark (backend, total seconds,
   throughput), and backend availability. Clear empty-state text when
   nothing has been loaded or run yet.
2. **Genome Files** — load FASTA or the bundled example, per-record
   table (ID, length, GC%, description), real aggregate statistics,
   clear parse-error messages.
3. **CRISPR Analysis** — guide spacer, PAM patterns, strand, max
   mismatches, require-PAM toggle, backend select (CPU/GPU/Dask/auto),
   worker count, worker-process toggle, Analyze and Cancel buttons.
   Live status line reports the running backend, completed/total
   chunks, and elapsed seconds; the progress bar tracks real chunk
   completion. Cancellation is cooperative — the scan stops at the
   next chunk boundary and the app stays usable.
4. **Alignment Results** — guide/PAM/settings summary, metrics line
   (sequences, bases, windows, matches, mismatches, method, elapsed),
   sortable DataTable with score/type/coordinates/strand/MM
   positions/ambiguity/PAM/matched sequence/guide, min-score,
   max-mismatch, chromosome, strand, PAM-compatibility, match-type and
   free-text search filters, click-row candidate detail with
   highlighted guide-vs-candidate alignment and score breakdown, and
   working quick CSV/JSON/HTML/PDF export of the current filtered
   view.
5. **Biological Scoring** — score formula explanation plus last-run
   score summary (count, mean, min, max, perfect matches).
6. **Processing Performance** — measured benchmark runner for
   cpu/gpu/dask backends, last-run panel (method, status, chunks
   done/total, elapsed, throughput, peak Python memory, candidates),
   and a table of every recorded benchmark.
7. **CPU/GPU Status** — real probes: CPU model/core counts/platform/
   Python version, GPU detection (backend, device, compute capability,
   VRAM, or the exact unavailability reason plus requirements),
   aligner engine selection, numba and dask availability, and what
   `auto` resolves to. Manual refresh re-probes live.
8. **Dask Processing** — real scheduler status (workers, threads,
   memory) plus one-click Dask analysis writing to the same results
   pipeline.
9. **Reports and Exports** — write CSV/JSON/HTML/PDF or a full bundle
   of the last analysis (or current filtered view) to the configured
   output directory.
10. **Settings** — chunk size, overlap, Dask workers, worker-process
    toggle, report output directory; validated on save.

### Benchmarking (`benchmark.py`, `pipeline.py`)
* `perf_counter_ns` timing split parse/align/total; `tracemalloc` peak
  Python memory (reported `n/a` when unavailable); bases/s and
  windows/s throughput; backend recorded in notes and in
  `BenchmarkResult.backend`.
* `run_analysis` also measures real peak Python memory per run
  (`extra["memory_peak_bytes"]`) and reports `chunks_total` /
  `chunks_completed` / `chunks_pending` for every backend.
* CPU-vs-GPU comparison only appears when both backends actually ran —
  never fabricated. CLI: `bench --backend cpu|gpu|dask|auto` or
  `bench --compare` (runs the identical workload on CPU, Dask, then
  GPU — each skipped with a printed reason when unavailable).
* `--backend auto` (CLI and dashboard) resolves to the best genuinely
  available backend at run time: GPU → Dask parallel CPU → sequential
  CPU; `result.backend`/`processing_method` always report what actually
  ran. On this machine `auto` resolves to CPU because GPU is
  policy-blocked and Dask is only chosen when it is faster/available.

#### Measured on this machine (20 Mbp synthetic genome, guide 20 nt,
max_mismatches=0, both strands, chunk_size=1 Mb, overlap=64)

| Backend | Align time | Throughput | Candidates |
|---------|-----------|------------|------------|
| CPU numba-JIT (default) | 4.33 s | 4.62 M bases/s | 90 |
| CPU pure-Python fallback | ~27 s | ~0.7 M bases/s | 90 |
| GPU numba-cuda (RTX 5050) | not runnable — DLL policy blocks numba.cuda | — | — |
| Dask processes/threads | optional; slower than sequential at this size | — | 90 |

The CPU row was re-measured for this release (`bench --backend cpu`);
the pure-Python figure is from the earlier A/B run. GPU and Dask rows
are omitted rather than guessed — run `bench --compare` on a machine
where those backends are usable to get fresh numbers.

### Reports (`reports.py`)
* CSV (metadata columns plus ranked candidate details), JSON (full
  settings, timings, stats, score formula, and candidates), self-contained
  HTML (inline CSS, candidate summary plus expandable per-candidate
  metadata and highlighted alignments), and PDF via ReportLab.
* Every report embeds FASTA filename, guide, PAM configuration, strand
  and mismatch settings, processing backend/method, preprocess/scan/total
  times, sequence/candidate counts, score model, and score formula.

## Installation

```bat
cd C:\Users\Sreevalli\Downloads\geneware
python -m pip install -r requirements.txt
```

## Usage

### Browser dashboard (default)

```bat
python -m geneware
```

Launches the Streamlit web app (`geneware/webapp.py`) in your browser
at `http://localhost:8501`. Sections: Home Dashboard, Genome Upload,
CRISPR Analysis, Analysis Results, Performance, Reports. Upload your
own `.fasta`/`.fa` or load the bundled example, enter a guide + PAM +
mismatch tolerance, run the real CPU/Dask/GPU backend, browse ranked
candidates with search/filters, view measured timings, and download
CSV/JSON/HTML reports. GPU is only offered when `detect_gpu()` genuinely
reports it available.

### Terminal dashboard (optional)

```bat
python -m geneware --textual
```

Workflow: **Genome Files** → Load Genome (or Load Example) →
**CRISPR Analysis** → enter guide, PAM, mismatches, backend →
**Analyze** (live chunk progress; **Cancel** stops cooperatively at
the next chunk) → review in **Alignment Results** → measure in
**Processing Performance** → check hardware in **CPU/GPU Status** →
export in **Reports and Exports**.

### CLI

```bat
python -m geneware.cli parse data\example.fasta
python -m geneware.cli scan data\example.fasta --guide CCCCggggnnnnAAAACCCC
python -m geneware.cli analyze data\example.fasta --guide CCCCggggnnnnAAAACCCC --backend cpu
python -m geneware.cli analyze data\example.fasta --guide CCCCggggnnnnAAAACCCC --backend dask --threads
python -m geneware.cli bench data\example.fasta --guide CCCCggggnnnnAAAACCCC
python -m geneware.cli bench data\bench_20mb.fasta --guide GAGTCCGAGCAGAAGAAGAA --compare
python -m geneware.cli bench data\bench_20mb.fasta --guide GAGTCCGAGCAGAAGAAGAA --backend dask --workers 4
python -m geneware.cli report data\example.fasta --guide CCCCggggnnnnAAAACCCC --out-dir reports
python -m geneware.cli gpu-status
```

Common options: `--chunk-size`, `--overlap`, `--pam`, `--strand`,
`--max-mismatches`, `--allow-pam-mismatch`, `--backend cpu|gpu|dask|auto`,
`--workers`, `--threads`, `--compare`, `--json`.
`GENEWEARE_ALIGNER_ENGINE=python` forces the pure-Python CPU engine
(for A/B benchmarking against the numba-JIT default).

## Testing

```bat
python -m pytest tests -q
```

122 tests cover: FASTA parsing edge cases, chunk boundaries, both
strands, PAM matching/validation, mismatch and ambiguity positions,
matched/genomic sequence provenance, exact-vs-near typing, deduplication,
scoring formula/breakdown/ranking, GPU detection contract + CPU fallback
and real GPU execution parity with CPU (skipped automatically when no GPU
backend is usable), numba-vs-Python engine parity, overlap validation,
multi-record/large-FASTA scans, compute_stats correctness, dask scan
equality with sequential results + pending/completed tracking, CSV/JSON/
HTML/PDF report generation and contents, pipeline backends +
memory-error surfacing, benchmark backend field, and Textual app smoke
tests (section mounting, genome load, full CPU analysis, results metrics,
filtering, sorting, candidate detail, quick export, and empty state),
auto-backend resolution, GPU VRAM field reporting, Dask worker-failure
capture / pending-completed accounting, and Day-7 end-to-end tests:
a synthetic FASTA with a known guide+PAM site verified through analysis
stats, candidate coordinates/mismatches/PAM/score, all four report
formats, CLI `analyze --json`, invalid-input rejection, and a full
dashboard workflow (load -> overview -> analyze -> results -> cancel
no-op).

## Coordinate & matching conventions

* Coordinates: 0-based, half-open `[start, end)` on the plus strand of
  each FASTA record (`record_id` = chromosome).
* `-` strand candidates match the reverse complement of `[start, end)`;
  `matched_sequence` is always in guide/candidate orientation while
  `genomic_sequence` remains plus-strand.
* PAM = 3 bases immediately 3' of the protospacer **on the candidate
  strand**; `pam_start`/`pam_end` give its plus-strand footprint.
  Edge windows that can't fit protospacer+PAM are skipped.
* `match_type=exact` requires zero mismatches and no compatible IUPAC
  ambiguity; otherwise the candidate is `near`.
* IUPAC-aware matching throughout; ambiguous bases preserved and reported
  in `ambiguous_positions`.

## Known limitations

* **GPU is currently blocked on this machine:** an Application Control /
  WDAC policy prevents `numba.cuda`'s native DLLs (`mviewbuf`) from
  loading, so `gpu-status` reports GPU unavailable and every backend
  selection falls back to CPU. The GPU code path is implemented and was
  benchmarked in an earlier session, but it cannot run while the policy
  stands. CPU analysis is fully functional and is the default.
* **BioPython native extensions are policy-blocked:** `Bio.AlignIO`'s
  `_codonaligner` and `Bio.SeqIO.TwoBitIO`'s `_twoBitIO` DLLs are
  blocked; `fasta.py` stubs them so FASTA parsing (pure Python) is
  unaffected. TwoBit/binary formats are unavailable.
* **GPU vs CPU on ambiguous bases:** the CUDA kernels use exact ACGT
  codes, so windows containing IUPAC ambiguity codes (N, R, ...) are
  conservatively filtered even when the CPU IUPAC-aware matcher would
  accept them. On unambiguous sequence the two backends return
  identical results (covered by `test_gpu_scan_matches_cpu_exactly`).
* **GPU requires the CUDA libraries:** numba needs nvvm/nvrtc/cudart/
  libdevice. This is handled automatically when the
  `nvidia-cuda-*-cu12` pip wheels (or a CUDA toolkit / `CUDA_HOME`) are
  present; otherwise `gpu-status` reports the exact reason.
* **CPU engine needs numba for full speed:** without numba the
  encoded pure-Python fallback is correct but ~5x slower (measured
  below). Numba JIT-compiles on first scan (~1-2 s once per process).
* **Dask overhead on small chunk counts:** each worker process
  re-JIT-compiles the kernel and pickles chunk sequences; on the
  20 Mbp benchmark (23 chunks) sequential numba was faster than
  dask-processes. Dask pays off on larger inputs / more workers.
* **Dask threads are GIL-limited:** the njit kernel releases the GIL
  (`nogil=True`), but thread workers were still slower than processes
  in measurements — prefer `--workers N` (processes) for real speedup.
* **Terminal exports write files:** the dashboard/CLI write CSV/JSON/
  HTML/PDF reports to the configured output directory; there is no
  browser download manager.
* **`exact` is conservative:** any compatible IUPAC ambiguity makes a
  zero-mismatch candidate `near`, because the base identity is not fully
  resolved.
* Whole-record sequences are held in memory (BioPython `SeqIO`
  semantics); chunking bounds *scan* memory but not *parse* memory.
  Very large genomes may need a streaming parser (not implemented).
* Dask workers are CPU-only here; distributed GPU is not implemented.
* `tracemalloc` measures Python allocations only; GPU VRAM is only
  reported when a GPU backend actually runs.
* `Bio.AlignIO` and compiled BioPython modules unusable under WDAC
  (shim covers FASTA needs).
* Chunk overlap must be ≥ guide length + PAM length (default 64).
