# GeneWeaver — GPU-Accelerated CRISPR Alignment Engine

**Week 1 foundation: repository audit + CPU baseline.**

GeneWeaver is an engineering project for Infotact Solutions: a Python
application for CRISPR guide-RNA off-target style protospacer search,
architected to move from a CPU baseline to GPU acceleration in later
weeks. This repository currently implements the **Week 1 CPU-only
foundation**. No GPU/CUDA code exists yet and none is claimed.

## Environment audit (Week 1)

* **Python:** 3.14.6 (`C:\Python314\python.exe`) — verified.
* **GPU:** NVIDIA GeForce RTX 5050 (8 GB), driver 592.82, CUDA runtime
  13.1 reported by `nvidia-smi`. `nvcc` is **not** installed, so no CUDA
  compilation is possible yet. GPU work is Week 2+.
* **Pre-existing state:** the workspace was empty; this is a fresh
  codebase, not a modification of prior work.

### Dependency compatibility findings

| Package   | Status on Python 3.14.6 | Notes |
|-----------|------------------------|-------|
| biopython | 1.88 installed | `Bio.SeqIO` import needs a shim — see below |
| numpy     | **upgraded 1.26.4 → 2.5.4** | 1.26.4 has no cp314 wheel; its DLL is blocked/incompatible and `import numpy` fails. 2.5.4 has a cp314 wheel and works. |
| numba     | not installed | No cp314-compatible release verified yet; deferred to Week 2 GPU work. |
| CUDA      | driver present, no toolkit | `nvcc` absent; required before any CUDA kernel work. |
| dask      | 2024.12.1 already installed | not used in Week 1 |
| textual   | 8.2.8 installed | dashboard framework |
| pytest    | 8.3.3 already installed | test runner |

### Known environment limitation: Application Control policy

Windows Application Control (WDAC/Smart App Control) on this machine
blocks `Bio.Align._codonaligner`, a compiled BioPython extension, with
*"DLL load failed … Application Control policy has blocked this file."*
`Bio.SeqIO` eagerly imports `Bio.AlignIO`, which loads that DLL, so a
plain `from Bio import SeqIO` fails. `geneware.fasta` contains a
**documented shim**: if `Bio.AlignIO` cannot be imported, a stub module
is registered so `Bio.SeqIO` loads for sequence formats only. FASTA
parsing is unaffected; alignment-format IO through `Bio.AlignIO` is
unavailable in this environment. This is an environment policy issue,
not a code defect.

## Project layout

```
geneware/
  __init__.py      package marker + version
  __main__.py      `python -m geneware` entry point
  fasta.py         FASTA parsing + validation + statistics (BioPython)
  chunking.py      memory-conscious genome chunker with coordinates
  aligner.py       CPU protospacer/PAM baseline scanner
  benchmark.py     real timing/throughput/memory measurements
  app.py           Textual terminal dashboard
  cli.py           argparse CLI (parse / scan / bench)
tests/             pytest suite (49 tests)
data/example.fasta small synthetic demo genome (test fixture)
requirements.txt   pinned Week 1 dependencies
```

## Implemented functionality (Week 1)

* **FASTA processing** (`geneware.fasta`): BioPython `SeqIO`-based
  streaming parser; multi-record support; ID/description preservation;
  IUPAC DNA validation; empty/malformed input errors; ambiguous-base
  preservation; U→T normalisation with warning; per-base counts, GC%,
  ambiguous count, record stats. Invalid characters are never silently
  discarded — they raise `FastaParseError`.
* **Genome chunking** (`geneware.chunking`): configurable chunk size and
  overlap; absolute 0-based half-open coordinates preserved; chromosome
  IDs carried through; short sequences handled; overlap sized so
  boundary-spanning windows are found in full.
* **CPU alignment baseline** (`geneware.aligner`): guide validation
  (IUPAC alphabet, configurable length bounds); configurable IUPAC PAM
  patterns (default `NGG`); forward and reverse-complement strand
  search; exact-length window comparison with IUPAC-aware base
  compatibility; mismatch positions recorded; chromosome/coordinate/
  strand/PAM provenance on every candidate; configurable mismatch
  threshold; candidate deduplication across chunk boundaries.
  *This is a baseline scanner, not a validated off-target model.*
* **Benchmarking** (`geneware.benchmark`): `time.perf_counter_ns()`
  timing split into parse / align / total; bases processed; windows
  evaluated; candidates; chunks; throughput (bases/s, windows/s);
  peak Python memory via `tracemalloc` (reported as unavailable if the
  facility cannot start). No numbers are fabricated.
* **Textual dashboard** (`geneware.app`): branding header; genome load
  with real validation stats; guide/PAM/strand/mismatch controls; scan
  with live progress; results table; benchmark panel with measured
  figures; log of errors and warnings. Every button performs its real
  operation. No GPU metrics are displayed.
* **CLI** (`geneware.cli`): `parse`, `scan`, `bench` subcommands.

## Installation

```bat
cd C:\Users\Sreevalli\Downloads\geneware
python -m pip install -r requirements.txt
```

(Only needed on a fresh environment; packages are already installed.)

## Usage

### Terminal dashboard

```bat
python -m geneware
:: or
python -m geneware.app
```

Controls: enter a FASTA path (or *Load Example Genome*), set the guide,
PAM patterns, strand, and max mismatches, then *Run Scan* / *Run
Benchmark*.

### CLI

```bat
python -m geneware.cli parse data\example.fasta
python -m geneware.cli scan data\example.fasta --guide ACGTACGTACGTACGTACGT --max-mismatches 2
python -m geneware.cli bench data\example.fasta --guide ACGTACGTACGTACGTACGT
```

Common scan options: `--chunk-size`, `--overlap`, `--pam` (comma-separated
IUPAC patterns), `--strand both|+|-`, `--max-mismatches`,
`--allow-pam-mismatch`.

## Testing

```bat
python -m pytest tests -q
```

49 tests cover: valid/multi-record/invalid/empty FASTA, ambiguous
nucleotides, uracil normalisation, stats correctness, chunk boundaries
and coordinate preservation, plus/minus-strand matching, mismatch
counting, PAM compatibility, custom PAM patterns, chunk-overlap
deduplication, and benchmark output sanity. All test sequences are
small synthetic fixtures, not real genomic data.

## Coordinate & matching conventions

* Coordinates are **0-based, half-open `[start, end)`** on the plus
  strand of each FASTA record (`record_id` = chromosome identifier).
* A candidate's protospacer occupies `[start, end)`; for `-` strand
  candidates the guide matches the **reverse complement** of that
  window.
* PAM is the 3 bases immediately 3' of the protospacer **on the
  candidate strand** (Cas9 NGG geometry, no bulge).
* Base comparison is IUPAC-aware: positions match when the IUPAC
  expansion sets intersect.
* Windows are only evaluated when protospacer + PAM fit entirely inside
  the scanned chunk; edge windows that cannot be fully assessed are
  skipped.

## Known limitations (Week 1)

* Pure-Python inner loop: throughput on chromosome-scale FASTA will be
  modest (measured, not estimated — see `bench` output). Numba/CUDA
  acceleration is planned for Week 2+.
* `Bio.AlignIO` and other compiled BioPython modules are unusable under
  this machine's Application Control policy (shim covers FASTA needs).
* `tracemalloc` measures Python-level allocations only.
* No GPU code, no CUDA kernels, no validated off-target scoring model.
* Chunk overlap must be >= guide length + PAM length to guarantee
  boundary coverage (default 64 covers up to 30-nt guides + 3-nt PAM).

## Planned for Week 2

* Numba/CUDA-accelerated seed-and-extend or k-mer indexing alignment.
* NumPy-vectorised mismatch counting.
* CUDA toolkit installation + first genuine GPU kernel with verified
  execution before any GPU claims are made.
* Larger real-world locus benchmarks.
