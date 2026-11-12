"""GeneWeaver command-line interface (Week 1 CPU baseline).

Subcommands:
    parse    - parse a FASTA file and print validation + statistics
    scan     - run the CPU alignment scan and print candidates
    bench    - run the timed benchmark pipeline and print measurements
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .aligner import AlignError, scan_records, validate_guide
from .benchmark import format_result, run_benchmark
from .fasta import FastaParseError, compute_stats, iter_fasta


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


def _records_from(path: str):
    return list(iter_fasta(path))


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
        strands = ("+", "-") if args.strand == "both" else (args.strand,)
        pams = tuple(s.strip() for s in args.pam.split(",") if s.strip())
        candidates, stats = scan_records(
            records, guide,
            chunk_size=args.chunk_size, overlap=args.overlap,
            pam_patterns=pams, strands=strands,
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
    try:
        strands = ("+", "-") if args.strand == "both" else (args.strand,)
        pams = tuple(s.strip() for s in args.pam.split(",") if s.strip())
        result, candidates, _ = run_benchmark(
            args.fasta, args.guide,
            chunk_size=args.chunk_size, overlap=args.overlap,
            pam_patterns=pams, strands=strands,
            max_mismatches=args.max_mismatches,
            require_pam=not args.allow_pam_mismatch,
        )
    except (FastaParseError, AlignError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(format_result(result))
    print(f"candidates (deduplicated): {len(candidates):,}")
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geneware",
        description="GeneWeaver Week 1 CPU baseline: FASTA parsing, "
                    "chunking, protospacer scan, benchmarking.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser("parse", help="parse and validate a FASTA file")
    p_parse.add_argument("fasta", help="path to FASTA file")
    p_parse.set_defaults(func=cmd_parse)

    p_scan = sub.add_parser("scan", help="scan genome for guide candidates")
    p_scan.add_argument("fasta", help="path to FASTA file")
    p_scan.add_argument("--guide", required=True, help="guide spacer sequence")
    _add_common(p_scan)
    p_scan.set_defaults(func=cmd_scan)

    p_bench = sub.add_parser("bench", help="timed end-to-end benchmark")
    p_bench.add_argument("fasta", help="path to FASTA file")
    p_bench.add_argument("--guide", required=True, help="guide spacer sequence")
    p_bench.add_argument("--json", action="store_true", help="also emit JSON")
    _add_common(p_bench)
    p_bench.set_defaults(func=cmd_bench)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
