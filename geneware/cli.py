"""GeneWeaver command-line interface.

Subcommands:
    parse       - parse a FASTA file and print validation + statistics
    scan        - run the sequential CPU alignment scan (Week 1 baseline)
    bench       - timed CPU baseline benchmark
    analyze     - full pipeline (cpu|gpu|dask backend) with scoring
    report      - run analysis and export CSV/JSON/HTML reports
    gpu-status  - probe GPU/CUDA availability and print requirements
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aligner import AlignError, scan_records, validate_guide
from .benchmark import format_result
from .fasta import FastaParseError, compute_stats, iter_fasta
from .pipeline import AnalysisError, run_analysis
from .pipeline import run_benchmark as run_pipeline_benchmark


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--chunk-size", type=int, default=1_000_000,
                   help="bases per genome chunk (default: 1000000)")
    p.add_argument("--overlap", type=int, default=64,
                   help="bases of overlap between chunks (default: 64)")
    p.add_argument("--pam", default="NGG",
                   help="comma-separated IUPAC PAM patterns (default: NGG)")
    p.add_argument("--strand", choices=["both", "+", "-"], default="both")
    p.add_argument("--max-mismatches", type=int, default=0)
    p.add_argument("--allow-pam-mismatch", action="store_true",
                   help="keep candidates even when PAM is incompatible")


def _add_backend(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["cpu", "gpu", "dask"],
                   default="cpu",
                   help="execution backend (default: cpu)")
    p.add_argument("--workers", type=int, default=None,
                   help="dask worker count (default: auto, cpu only)")
    p.add_argument("--threads", action="store_true",
                   help="use dask threads instead of worker processes")


def _records_from(path: str):
    return list(iter_fasta(path))


def _strands(args: argparse.Namespace) -> tuple[str, ...]:
    return ("+", "-") if args.strand == "both" else (args.strand,)


def _pams(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(s.strip() for s in args.pam.split(",") if s.strip())


def cmd_parse(args: argparse.Namespace) -> int:
    try:
        records = _records_from(args.fasta)
    except FastaParseError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    stats = compute_stats(records)
    print(json.dumps(stats.to_dict(), indent=2))
    for rec in records:
        print(f"record {rec.record_id!r}: {rec.length:,} bases "
              f"({rec.description[:60]!r})")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    try:
        records = _records_from(args.fasta)
        guide = validate_guide(args.guide)
        candidates, stats = scan_records(
            records, guide,
            chunk_size=args.chunk_size, overlap=args.overlap,
            pam_patterns=_pams(args), strands=_strands(args),
            max_mismatches=args.max_mismatches,
            require_pam=not args.allow_pam_mismatch,
        )
    except (FastaParseError, AlignError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for c in candidates:
        print(json.dumps(c.to_dict()))
    print(f"# scan_stats: {stats.to_dict()}", file=sys.stderr)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    # --compare runs the same input/settings on CPU then GPU (when a GPU
    # backend genuinely works) so both measured results are shown.
    if getattr(args, "compare", False):
        backends = ["cpu", "gpu"]
    else:
        backends = [args.backend]

    rc = 0
    for backend in backends:
        if backend == "gpu":
            from .gpu import gpu_available
            if not gpu_available():
                print("SKIP gpu: no usable GPU backend "
                      "(see `python -m geneware.cli gpu-status`)",
                      file=sys.stderr)
                continue
        try:
            result, analysis = run_pipeline_benchmark(
                args.fasta, args.guide, backend=backend,
                chunk_size=args.chunk_size, overlap=args.overlap,
                pam_patterns=_pams(args), strands=_strands(args),
                max_mismatches=args.max_mismatches,
                require_pam=not args.allow_pam_mismatch,
                n_workers=getattr(args, "workers", None),
                processes=not getattr(args, "threads", False),
                progress=_progress,
            )
        except (FastaParseError, AlignError, AnalysisError) as exc:
            print(f"ERROR [{backend}]: {exc}", file=sys.stderr)
            rc = 2
            continue

        print(format_result(result))
        print(f"candidates (deduplicated): {len(analysis.candidates):,}")
        if args.json:
            print(json.dumps(result.to_dict(), indent=2))
    return rc


def _progress(done: int, total: int) -> None:
    if total:
        print(f"\rprogress: {done}/{total} chunks", end="", file=sys.stderr)
    if done == total:
        print(file=sys.stderr)


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        result = run_analysis(
            args.fasta, args.guide,
            backend=args.backend,
            chunk_size=args.chunk_size, overlap=args.overlap,
            pam_patterns=_pams(args), strands=_strands(args),
            max_mismatches=args.max_mismatches,
            require_pam=not args.allow_pam_mismatch,
            n_workers=args.workers, processes=not args.threads,
            progress=_progress,
        )
    except (FastaParseError, AlignError, AnalysisError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.analysis_dict(), indent=2))
    else:
        print(f"backend={result.backend}  candidates={len(result.candidates):,}  "
              f"elapsed={result.elapsed_seconds:.3f}s")
        print(f"score_summary={json.dumps(result.score_summary)}")
        for rank, sc in enumerate(result.scored[: args.top], start=1):
            c = sc.candidate
            print(f"{rank:>3}. score={sc.score:5.1f} {c.chromosome}:"
                  f"{c.start}-{c.end} {c.strand} mm={c.mismatches} "
                  f"pam={c.pam} ({'ok' if c.pam_compatible else 'BAD'})")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    try:
        result = run_analysis(
            args.fasta, args.guide,
            backend=args.backend,
            chunk_size=args.chunk_size, overlap=args.overlap,
            pam_patterns=_pams(args), strands=_strands(args),
            max_mismatches=args.max_mismatches,
            require_pam=not args.allow_pam_mismatch,
            n_workers=args.workers, processes=not args.threads,
            progress=_progress,
        )
        from .reports import write_report_bundle
        paths = write_report_bundle(result, args.out_dir, args.stem)
    except (FastaParseError, AlignError, AnalysisError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    for kind, p in paths.items():
        print(f"{kind}: {p}")
    return 0


def cmd_gpu_status(args: argparse.Namespace) -> int:
    from .gpu import detect_gpu
    status = detect_gpu(force=True)
    print(json.dumps(status.to_dict(), indent=2))
    return 0 if status.available else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geneware",
        description="GeneWeaver: GPU-accelerated CRISPR alignment engine "
                    "(cpu/gpu/dask backends, scoring, reports).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="parse and validate a FASTA file")
    p_parse.add_argument("fasta", help="path to FASTA file")
    p_parse.set_defaults(func=cmd_parse)

    p_scan = sub.add_parser("scan", help="scan genome for guide candidates (CPU)")
    p_scan.add_argument("fasta", help="path to FASTA file")
    p_scan.add_argument("--guide", required=True, help="guide spacer sequence")
    _add_common(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_bench = sub.add_parser("bench", help="timed end-to-end benchmark")
    p_bench.add_argument("fasta", help="path to FASTA file")
    p_bench.add_argument("--guide", required=True, help="guide spacer sequence")
    p_bench.add_argument("--json", action="store_true", help="also emit JSON")
    p_bench.add_argument("--compare", action="store_true",
                         help="run the same workload on cpu then gpu and "
                              "print both measured results")
    _add_common(p_bench)
    _add_backend(p_bench)
    p_bench.set_defaults(func=cmd_bench)

    p_an = sub.add_parser("analyze", help="full analysis with scoring")
    p_an.add_argument("fasta", help="path to FASTA file")
    p_an.add_argument("--guide", required=True, help="guide spacer sequence")
    p_an.add_argument("--json", action="store_true", help="emit JSON payload")
    p_an.add_argument("--top", type=int, default=20,
                      help="max rows in text output (default: 20)")
    _add_common(p_an)
    _add_backend(p_an)
    p_an.set_defaults(func=cmd_analyze)

    p_rep = sub.add_parser("report", help="analysis + CSV/JSON/HTML export")
    p_rep.add_argument("fasta", help="path to FASTA file")
    p_rep.add_argument("--guide", required=True, help="guide spacer sequence")
    p_rep.add_argument("--out-dir", default="reports",
                       help="output directory (default: reports/)")
    p_rep.add_argument("--stem", default="geneware_report",
                       help="report filename stem")
    _add_common(p_rep)
    _add_backend(p_rep)
    p_rep.set_defaults(func=cmd_report)

    p_gpu = sub.add_parser("gpu-status", help="probe GPU/CUDA availability")
    p_gpu.set_defaults(func=cmd_gpu_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
