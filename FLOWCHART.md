# 🧬 GeneWeaver – System Flowchart

## 1. Project Workflow Overview

GeneWeaver is a GPU-Accelerated CRISPR Alignment Engine designed to identify potential CRISPR off-target sites in genomic DNA sequences.

The system follows a structured pipeline consisting of genomic input, sequence parsing, validation, genome chunking, CPU/GPU alignment, distributed processing, biological scoring, and result visualization.

## 2. Main System Flowchart

```mermaid
flowchart TD
    A([Start GeneWeaver]) --> B[Load Reference Genome]
    B --> C[Read FASTA File]
    C --> D[BioPython Genomic Parser]
    D --> E{Is Genome Valid?}

    E -- No --> F[Display Validation Error]
    F --> B

    E -- Yes --> G[Normalize DNA Sequences]
    G --> H[Extract Chromosome Information]
    H --> I[Divide Genome into Chunks]

    I --> J[Initialize Processing Engine]
    J --> K{CUDA GPU Available?}

    K -- Yes --> L[Initialize Numba CUDA]
    K -- No --> M[Use CPU Alignment Engine]

    L --> N[Parallel DNA Sequence Alignment]
    M --> N

    N --> O[Identify Potential Matching Sites]
    O --> P[Calculate Sequence Mismatches]
    P --> Q[Evaluate PAM Compatibility]
    Q --> R[Calculate Off-Target Scores]
    R --> S[Rank Candidate Locations]

    S --> T[Aggregate Processing Results]
    T --> U[Update Textual Dashboard]
    U --> V[Generate Analysis Report]

    V --> W([Analysis Completed])
```

## 3. Detailed Module Flowcharts

### Module 1 – Genomic Parser

Responsible for reading, validating, and preparing genomic FASTA data.

```mermaid
flowchart TD
    A([Genome Input]) --> B[Load FASTA File]
    B --> C[BioPython Sequence Parser]
    C --> D[Extract DNA Records]
    D --> E[Validate DNA Characters]
    E --> F{Valid Sequence?}

    F -- No --> G[Report Invalid Input]
    F -- Yes --> H[Normalize Sequence]
    H --> I[Extract Sequence Metadata]
    I --> J[Calculate Sequence Statistics]
    J --> K[Divide Genome into Chunks]
    K --> L([Prepared Genome Data])
```

### Module 2 – CUDA Alignment Engine

Responsible for sequence comparison and GPU acceleration.

```mermaid
flowchart TD
    A([Prepared Genome Chunks]) --> B[Initialize Alignment]
    B --> C[Run CPU Baseline]
    C --> D[Record CPU Execution Time]
    D --> E{CUDA Available?}

    E -- No --> F[Continue CPU Processing]
    E -- Yes --> G[Allocate GPU Memory]
    G --> H[Transfer Sequence Data]
    H --> I[Execute Numba CUDA Kernel]
    I --> J[Parallel Sequence Comparison]
    J --> K[Collect GPU Results]
    K --> L[Release GPU Resources]

    F --> M[Collect Alignment Results]
    L --> M
    M --> N[Compare CPU and GPU Results]
    N --> O([Alignment Results])
```

### Module 3 – Distributed Scheduler

Responsible for distributing genomic chunks and collecting processing results.

```mermaid
flowchart TD
    A([Genome Chunks]) --> B[Initialize Dask Scheduler]
    B --> C[Create Processing Tasks]
    C --> D[Assign Tasks to Workers]
    D --> E[Process Genome Chunks]
    E --> F{Task Successful?}

    F -- No --> G[Handle Processing Error]
    G --> H[Retry or Report Failure]

    F -- Yes --> I[Collect Worker Results]
    I --> J[Aggregate Results]
    H --> K[Update Task Status]
    J --> K
    K --> L([Combined Results])
```

### Module 4 – Off-Target Scoring Engine

Responsible for evaluating potential CRISPR off-target matches.

```mermaid
flowchart TD
    A([Alignment Candidates]) --> B[Extract Candidate DNA]
    B --> C[Compare with Guide Sequence]
    C --> D[Calculate Mismatch Count]
    D --> E[Evaluate PAM Compatibility]
    E --> F[Apply Biological Scoring]
    F --> G[Calculate Candidate Scores]
    G --> H[Rank Potential Off-Target Sites]
    H --> I[Generate Ranked Results]
    I --> J([Scoring Completed])
```

### Module 5 – Live Textual Dashboard

Responsible for presenting processing information and analysis results.

```mermaid
flowchart TD
    A([Start Dashboard]) --> B[Load Genome Information]
    B --> C[Initialize Monitoring]
    C --> D[Display Processing Progress]
    D --> E[Monitor CPU and GPU Metrics]
    E --> F[Display Chunk Processing]
    F --> G[Display Alignment Results]
    G --> H[Display Off-Target Candidates]
    H --> I{Processing Complete?}

    I -- No --> D
    I -- Yes --> J[Display Final Statistics]
    J --> K[Generate Analysis Summary]
    K --> L([Dashboard Complete])
```

## 4. Data Processing Pipeline

| Stage | Module | Responsibility |
|---|---|---|
| 1 | Input | Load reference genome and guide RNA |
| 2 | Parser | Parse FASTA records |
| 3 | Validation | Verify DNA sequence integrity |
| 4 | Chunking | Divide genome into manageable sections |
| 5 | Alignment | Compare genomic sequences |
| 6 | Acceleration | Execute supported GPU computation |
| 7 | Scheduling | Manage parallel processing tasks |
| 8 | Scoring | Evaluate mismatches and PAM compatibility |
| 9 | Ranking | Rank potential off-target candidates |
| 10 | Dashboard | Display processing and analysis results |
| 11 | Reporting | Generate structured analysis reports |

## 5. System Architecture

```mermaid
flowchart TB
    subgraph INPUT["Input Layer"]
        A[FASTA Genome]
        B[CRISPR Guide RNA]
        C[PAM Configuration]
    end

    subgraph PROCESSING["Processing Layer"]
        D[BioPython Parser]
        E[Genome Chunker]
        F[Dask Scheduler]
    end

    subgraph COMPUTE["Computational Layer"]
        G[CPU Baseline Engine]
        H[Numba CUDA Engine]
    end

    subgraph ANALYSIS["Analysis Layer"]
        I[Sequence Alignment]
        J[Mismatch Detection]
        K[Biological Scoring]
        L[Candidate Ranking]
    end

    subgraph OUTPUT["Output Layer"]
        M[Textual Dashboard]
        N[Performance Metrics]
        O[Analysis Reports]
    end

    A --> D
    B --> I
    C --> K
    D --> E
    E --> F
    F --> G
    F --> H
    G --> I
    H --> I
    I --> J
    J --> K
    K --> L
    L --> M
    L --> N
    L --> O
```

## 6. Expected System Outputs

The completed application is intended to provide:

- Genome processing statistics.
- FASTA validation results.
- Chromosome and sequence information.
- CPU and GPU execution benchmarks.
- Genome chunk-processing progress.
- Potential CRISPR off-target locations.
- Sequence mismatch counts.
- PAM compatibility information.
- Ranked candidate results.
- Performance summaries.
- Structured analysis reports.

## 7. Development Roadmap

| Development Phase | Main Deliverable |
|---|---|
| Week 1 | FASTA parser, CPU baseline and genome chunking |
| Week 2 | CUDA alignment engine and Textual dashboard |
| Mid-Project Review | Performance benchmarking and memory validation |
| Week 3 | Dask integration and biological scoring |
| Week 4 | CUDA optimization, dashboard refinement and final testing |

## 8. Important Implementation Notes

- CPU and GPU execution must be independently identifiable.
- GPU acceleration must only be reported when CUDA kernels actually execute.
- Dask task scheduling must not be confused with GPU acceleration.
- Sequence alignment and biological scoring must use documented algorithms.
- Processing errors and invalid genomic input must be handled safely.
- Performance metrics must come from actual measurements.
- Potential off-target matches are computational predictions, not experimentally confirmed mutations.

---

**GeneWeaver – Accelerating Genomic Analysis Through High-Performance Python Computing.**
