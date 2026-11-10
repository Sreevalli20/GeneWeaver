# 🧬 GeneWeaver
### GPU-Accelerated CRISPR Alignment Engine

**Domain:** Bioinformatics & High-Performance Computing (HPC)  
**Project:** Infotact Solutions – Advanced Python Engineering (Second Month)  
**Repository:** GeneWeaver

---

## 📌 Project Overview

GeneWeaver is a Python-based, GPU-accelerated genomic sequence analysis engine designed to identify potential CRISPR off-target sites in DNA sequences.

In CRISPR gene editing, guide RNA directs the Cas9 system towards a specific DNA target. However, similar sequences at other genomic locations can potentially cause unintended off-target interactions.

GeneWeaver aims to address this challenge by providing a computational pipeline that processes genomic FASTA files, compares target sequences against reference DNA, identifies potential off-target matches, and ranks candidate locations based on sequence similarity and biological scoring criteria.

The system combines Python-based bioinformatics processing, NVIDIA GPU acceleration, distributed task scheduling, and a real-time terminal dashboard.

## 🎯 Project Objectives

- Develop an efficient genomic FASTA processing pipeline.
- Implement DNA sequence alignment and mismatch detection.
- Accelerate suitable sequence-comparison workloads using NVIDIA CUDA.
- Use parallel processing to handle large genomic datasets.
- Identify and rank potential CRISPR off-target candidates.
- Monitor processing progress and system resources through a live terminal dashboard.
- Benchmark CPU and GPU performance.
- Produce structured and reproducible analysis reports.

## 🧠 Problem Statement

Searching large genomic sequences for unintended CRISPR off-target matches can require significant computational resources.

Traditional sequential Python implementations may become inefficient when processing large datasets.

GeneWeaver explores high-performance Python techniques to improve genomic sequence analysis through:

- GPU parallel computation.
- Efficient memory management.
- Genome chunking.
- Distributed task scheduling.
- Optimized sequence alignment algorithms.

The goal is to create a scalable computational screening engine that makes genomic sequence analysis more efficient.

## 🏗️ System Architecture

```text
                 GENOME INPUT
                      |
                      v
              FASTA FILE UPLOAD
                      |
                      v
              GENOMIC PARSER
                 BioPython
                      |
                      v
             SEQUENCE VALIDATION
                      |
                      v
               GENOME CHUNKING
                      |
                      v
             PROCESSING SCHEDULER
                    Dask
                      |
             +--------+--------+
             |                 |
             v                 v
        CPU BASELINE       GPU ENGINE
                           Numba CUDA
             |                 |
             +--------+--------+
                      |
                      v
              SEQUENCE ALIGNMENT
                      |
                      v
              OFF-TARGET DETECTION
                      |
                      v
               BIOLOGICAL SCORING
                      |
                      v
              RESULTS & REPORTING
                      |
                      v
             TEXTUAL LIVE DASHBOARD
```

## ⚙️ Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.14.6 | Core application development |
| BioPython | FASTA parsing and biological sequence processing |
| Numba | JIT compilation and GPU acceleration |
| CUDA | Parallel computation on NVIDIA GPUs |
| Dask | Parallel and distributed task scheduling |
| Textual | Interactive terminal user interface |
| NumPy | Numerical and array-based computation |
| pytest | Testing and validation |
| Git & GitHub | Version control and collaboration |

**Compatibility note:** Dependency versions and GPU support will be verified against Python 3.14.6 before implementation. Unsupported packages will not be silently treated as compatible.

## 🔬 Core Modules

### 1. Genomic Parser

Responsible for reading and preparing genomic data.

Planned capabilities:

- FASTA file ingestion.
- DNA sequence validation.
- Sequence normalization.
- Chromosome identification.
- Genome chunking.
- Sequence statistics.
- Input error handling.

### 2. CUDA Kernel Engine

Responsible for accelerating genomic sequence comparisons.

Planned capabilities:

- CPU-based alignment baseline.
- CUDA kernel implementation using Numba.
- Parallel sequence comparisons.
- GPU memory allocation and management.
- Host-to-device and device-to-host transfers.
- CPU versus GPU benchmarking.
- GPU memory monitoring.

### 3. Distributed Scheduler

Responsible for coordinating genome-processing workloads.

Planned capabilities:

- Genome chunk distribution.
- Dask task scheduling.
- Parallel worker execution.
- Processing progress tracking.
- Result aggregation.
- Worker error handling.
- Scalable execution where hardware is available.

### 4. Live Terminal Dashboard

Built using Textual.

Planned dashboard sections:

- Genome input information.
- Sequence-processing status.
- Current processing progress.
- CPU and GPU execution statistics.
- GPU memory information.
- Chunk-processing status.
- Potential off-target candidates.
- Mismatch information.
- Alignment results.
- Final analysis summary.

## 🧬 CRISPR Off-Target Analysis

The analysis pipeline is designed to:

1. Accept a reference genome in FASTA format.
2. Accept a CRISPR guide sequence.
3. Configure the target PAM pattern.
4. Search for matching genomic regions.
5. Identify sequence mismatches.
6. Evaluate candidate locations.
7. Rank potential off-target sites.
8. Generate a structured analysis report.

The scoring system will consider sequence mismatches and PAM compatibility.

Results represent computationally identified candidates and do not establish actual biological editing activity or clinical safety.

## 📅 Four-Week Development Roadmap

### Week 1 – Genome Processing Foundation

- Implement a CPU-based DNA sequence alignment baseline.
- Build the genomic parser using BioPython.
- Read and process FASTA files.
- Divide genome sequences into manageable chunks.
- Establish baseline execution-time measurements.

### Week 2 – GPU Acceleration & Dashboard

- Implement CUDA kernels using Numba.
- Handle CPU and GPU memory transfers.
- Develop GPU-based sequence comparisons.
- Build the initial Textual dashboard.
- Display genome-processing progress.
- Monitor GPU memory usage.

### Mid-Project Review

- Benchmark GPU implementation against CPU baseline.
- Evaluate memory consumption.
- Validate genome-processing functionality.
- Test large genomic datasets.
- Evaluate the assignment's 100x GPU speedup target.

### Week 3 – Distributed Processing & Biological Scoring

- Integrate Dask for distributed processing.
- Implement parallel genome-chunk scheduling.
- Develop off-target candidate scoring.
- Evaluate PAM sequence compatibility.
- Rank potential off-target matches.
- Improve processing and result aggregation.

### Week 4 – Optimization & Finalization

- Explore CUDA shared-memory optimization.
- Improve processing performance.
- Refine the Textual dashboard.
- Highlight DNA mismatches.
- Improve error handling and validation.
- Complete testing and documentation.
- Prepare final analysis reports.

## 📊 Performance Evaluation

GeneWeaver will evaluate performance through measurable benchmarks.

| Metric | Purpose |
|---|---|
| CPU execution time | Establish baseline performance |
| GPU execution time | Measure accelerated processing |
| Speedup ratio | Compare CPU and GPU execution |
| Genome processing rate | Measure bases processed per second |
| GPU memory consumption | Monitor device memory usage |
| Processing throughput | Evaluate chunk-processing performance |
| Alignment accuracy | Validate sequence-comparison results |

Speedup will be calculated as:

```text
Speedup = CPU Execution Time / GPU Execution Time
```

Performance measurements will be recorded from actual executions. No benchmark values or acceleration claims will be reported without testing.

## 💻 Hardware Requirements

- 64-bit operating system.
- Python 3.14.6 development environment.
- NVIDIA GPU with CUDA-compatible hardware and drivers for GPU execution.
- Sufficient system RAM for the selected genome dataset.
- NVIDIA GPU memory appropriate for the workload.

CPU-based analysis and other supported functionality should remain available when CUDA hardware is unavailable.

## 🚀 Installation & Setup

Clone the repository:

```bash
git clone https://github.com/Sreevalli20/GeneWeaver.git
```

Navigate into the project:

```bash
cd GeneWeaver
```

Create a virtual environment using Python 3.14:

```bash
py -3.14 -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies after verifying Python 3.14.6 compatibility:

```bash
python -m pip install --upgrade pip
```

Install the project dependencies using the finalized requirements file:

```bash
pip install -r requirements.txt
```

**Note:** GPU execution requires compatible NVIDIA drivers, CUDA support, and compatible Numba dependencies. CPU-only operation does not demonstrate CUDA acceleration.

## ▶️ Running the Project

The exact execution commands will be documented after the application entry points and modules are implemented.

The intended workflow is:

1. Launch GeneWeaver.
2. Load a reference genome FASTA file.
3. Enter a CRISPR guide sequence.
4. Configure the PAM pattern and mismatch threshold.
5. Start genomic analysis.
6. Monitor processing through the live dashboard.
7. Review potential off-target locations.
8. Export the analysis report.

## 🧪 Testing & Validation

The project will include tests for:

- FASTA parsing.
- Invalid sequence handling.
- Genome chunking.
- DNA alignment correctness.
- Mismatch detection.
- PAM compatibility.
- CPU/GPU result consistency.
- Dask task execution.
- Memory management.
- Dashboard updates.
- Error handling.

## 📁 Planned Repository Structure

```text
GeneWeaver/
│
├── geneweaver/
│   ├── __init__.py
│   │
│   ├── parser/
│   │   ├── __init__.py
│   │   ├── fasta_parser.py
│   │   └── genome_chunker.py
│   │
│   ├── alignment/
│   │   ├── __init__.py
│   │   ├── cpu_engine.py
│   │   ├── cuda_engine.py
│   │   └── scoring.py
│   │
│   ├── scheduler/
│   │   ├── __init__.py
│   │   └── dask_scheduler.py
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   └── app.py
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── report_generator.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── validation.py
│
├── tests/
│   ├── test_parser.py
│   ├── test_alignment.py
│   ├── test_scoring.py
│   └── test_scheduler.py
│
├── data/
│   └── sample/
│
├── benchmarks/
│   └── benchmark.py
│
├── docs/
│
├── requirements.txt
├── .gitignore
├── README.md
└── main.py
```

This is the planned structure and may evolve as development progresses.

## 🔐 Data & Responsible Use

- Use public reference genome datasets or synthetic DNA sequences for development and testing.
- Do not upload private or identifiable human genomic data.
- Keep large datasets outside GitHub.
- Do not treat computational predictions as confirmed biological findings.
- Preserve reproducibility through documented parameters and benchmark configurations.

## 👩‍💻 Project Information

**Project Name:** GeneWeaver  
**Project Category:** Bioinformatics & High-Performance Computing  
**Development Language:** Python  
**Internship:** Infotact Solutions – Advanced Python Engineering  
**Project Duration:** Second-Month Project

**GitHub Repository:**  
https://github.com/Sreevalli20/GeneWeaver

---

## 🌟 Project Vision

GeneWeaver aims to demonstrate how Python can be extended beyond traditional scripting through GPU acceleration, parallel computing, memory-efficient data processing, and biological sequence analysis.

The ultimate objective is to develop a reliable, scalable, and performance-oriented CRISPR off-target screening engine using modern Python engineering techniques.

**Built with Python. Accelerated with CUDA. Designed for genomic discovery.**
