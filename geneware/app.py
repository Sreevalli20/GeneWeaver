"""GeneWeaver terminal dashboard (Textual) - Week 1 CPU baseline.

Every control performs a real operation:
  * Load Genome      -> parses the FASTA file with BioPython, validates
                        sequences, and displays real statistics.
  * Run Scan         -> runs the CPU alignment baseline and fills the
                        results table with real candidates.
  * Run Benchmark    -> runs the timed pipeline and shows measured
                        throughput/memory figures.

No GPU/CUDA metrics are shown: GPU acceleration is not implemented yet.
"""

from __future__ import annotations

import sys
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
)

from .aligner import DEFAULT_PAM_PATTERNS, scan_records, validate_guide, AlignError
from .benchmark import format_result, run_benchmark
from .fasta import (
    FastaParseError,
    compute_stats,
    parse_fasta,
)

EXAMPLE_FASTA = Path(__file__).resolve().parent.parent / "data" / "example.fasta"


class GeneWeaverApp(App):
    """GeneWeaver Week 1 dashboard."""

    TITLE = "GeneWeaver"
    SUB_TITLE = "GPU-Accelerated CRISPR Alignment Engine - Week 1 CPU Baseline"

    CSS = """
    Screen { layout: vertical; }
    #branding { text-align: center; padding: 1 0; }
    #branding .title { text-style: bold; color: $primary; }
    #branding .sub { color: $text-muted; }
    #main { layout: horizontal; height: 1fr; }
    #controls { width: 42; border: solid $primary-darken-2; padding: 1; overflow-y: auto; }
    #results { width: 1fr; padding: 1; }
    .section-label { margin-top: 1; color: $primary; text-style: bold; }
    #genome-info { height: auto; max-height: 12; overflow-y: auto; }
    #stats { height: auto; }
    #log { height: 10; border: solid $primary-darken-2; }
    #progress { margin: 1 0; }
    DataTable { height: 1fr; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+r", "run_scan", "Run Scan"),
        ("ctrl+b", "run_bench", "Benchmark"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.records = []
        self.stats = None
        self.last_scan = None  # (candidates, scan_stats)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Vertical(id="branding"):
            yield Static("GENEWEAVER", classes="title")
            yield Static(self.SUB_TITLE, classes="sub")
        with Horizontal(id="main"):
            with Vertical(id="controls"):
                yield Label("Genome", classes="section-label")
                yield Input(
                    placeholder="path/to/genome.fasta",
                    id="fasta-path",
                )
                yield Button("Load Genome", id="load-btn", variant="primary")
                yield Button("Load Example Genome", id="example-btn")
                yield Static("", id="genome-info")
                yield Label("Guide RNA", classes="section-label")
                yield Input(
                    placeholder="20-nt spacer, e.g. GAGTCCGAGCAGAAGAAGAA",
                    id="guide-input",
                )
                yield Label("PAM patterns (comma-separated)", classes="section-label")
                yield Input("NGG", id="pam-input")
                yield Label("Strand", classes="section-label")
                yield Select(
                    [("Both (+ and -)", "both"), ("Forward (+) only", "+"),
                     ("Reverse (-) only", "-")],
                    value="both",
                    id="strand-select",
                    allow_blank=False,
                )
                yield Label("Max mismatches", classes="section-label")
                yield Input("0", id="mm-input")
                yield Static("", id="stats")
                yield Button("Run Scan", id="scan-btn", variant="success")
                yield Button("Run Benchmark", id="bench-btn", variant="warning")
            with Vertical(id="results"):
                yield Static("Candidates", classes="section-label")
                yield ProgressBar(total=100, show_eta=False, id="progress")
                yield DataTable(id="results-table")
                yield Label("Benchmark", classes="section-label")
                yield Static("(no benchmark run yet)", id="bench-out")
                yield Label("Log", classes="section-label")
                yield RichLog(id="log", wrap=True, markup=False)
        yield Footer()

    # ---------------- helpers ----------------

    def _log(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(msg)

    def _set_progress(self, done: int, total: int) -> None:
        bar = self.query_one("#progress", ProgressBar)
        bar.total = max(total, 1)
        bar.progress = done

    def _strand_tuple(self) -> tuple[str, ...]:
        sel = self.query_one("#strand-select", Select).value
        return ("+", "-") if sel == "both" else (sel,)

    def _guide(self) -> str:
        return self.query_one("#guide-input", Input).value

    def _pam(self) -> tuple[str, ...]:
        raw = self.query_one("#pam-input", Input).value.strip()
        return tuple(p for p in (s.strip() for s in raw.split(",")) if p) or DEFAULT_PAM_PATTERNS

    def _max_mm(self) -> int:
        try:
            return max(0, int(self.query_one("#mm-input", Input).value.strip() or "0"))
        except ValueError:
            return 0

    def _fill_results(self, candidates) -> None:
        table = self.query_one("#results-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Chr", "Start", "End", "Strand", "MM", "PAM", "PAM ok", "MM pos")
        for c in candidates:
            table.add_row(
                c.chromosome, str(c.start), str(c.end), c.strand,
                str(c.mismatches), c.pam, "yes" if c.pam_compatible else "no",
                ",".join(map(str, c.mismatch_positions)) or "-",
            )

    # ---------------- actions ----------------

    @on(Button.Pressed, "#load-btn")
    @on(Button.Pressed, "#example-btn")
    def load_genome(self, event: Button.Pressed) -> None:
        if event.button.id == "example-btn":
            path = EXAMPLE_FASTA
        else:
            path = Path(self.query_one("#fasta-path", Input).value.strip())
            if not path:
                self._log("ERROR: enter a FASTA file path first.")
                return
        self._load_genome_worker(str(path))

    @work(thread=True)
    def _load_genome_worker(self, path_str: str) -> None:
        try:
            records = parse_fasta(path_str)
            stats = compute_stats(records)
        except FastaParseError as exc:
            self.call_from_thread(self._log, f"ERROR: {exc}")
            return
        except Exception as exc:  # noqa: BLE001 - surface real errors to user
            self.call_from_thread(self._log, f"ERROR: {type(exc).__name__}: {exc}")
            return

        self.records = records
        self.stats = stats
        self.call_from_thread(self._on_genome_loaded, path_str, records, stats)

    def _on_genome_loaded(self, path_str, records, stats) -> None:
        info = self.query_one("#genome-info", Static)
        lines = [f"file: {path_str}", f"records: {stats.record_count}"]
        for rec, st in zip(records, stats.per_record, strict=True):
            lines.append(
                f"  {rec.record_id} len={rec.length:,} desc='{rec.description[:40]}'"
            )
        info.update("\n".join(lines))

        s = self.query_one("#stats", Static)
        bc = stats.base_counts
        s.update(
            "VALID\n"
            f"bases: {stats.total_bases:,}\n"
            f"A={bc.get('A', 0):,} T={bc.get('T', 0):,} "
            f"G={bc.get('G', 0):,} C={bc.get('C', 0):,}\n"
            f"GC={stats.gc_percent:.2f}%  ambiguous={stats.ambiguous_count:,}"
        )
        self._log(
            f"Loaded {stats.record_count} record(s), "
            f"{stats.total_bases:,} bases from {path_str}"
        )

    def action_run_scan(self) -> None:
        self._start_scan()

    @on(Button.Pressed, "#scan-btn")
    def _start_scan(self) -> None:
        if not self.records:
            self._log("ERROR: load a genome first.")
            return
        try:
            guide = validate_guide(self._guide())
        except AlignError as exc:
            self._log(f"ERROR: {exc}")
            return
        self._scan_worker(guide)

    @work(thread=True)
    def _scan_worker(self, guide: str) -> None:
        def progress(done: int, total: int) -> None:
            self.call_from_thread(self._set_progress, done, total)

        try:
            candidates, scan_stats = scan_records(
                self.records,
                guide,
                chunk_size=250_000,
                overlap=64,
                pam_patterns=self._pam(),
                strands=self._strand_tuple(),
                max_mismatches=self._max_mm(),
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._log, f"ERROR during scan: {exc}")
            return

        self.last_scan = (candidates, scan_stats)
        self.call_from_thread(self._on_scan_done, candidates, scan_stats)

    def _on_scan_done(self, candidates, scan_stats) -> None:
        self._fill_results(candidates)
        self._log(
            f"Scan complete: {scan_stats.windows_evaluated:,} windows, "
            f"{scan_stats.candidates_found:,} candidates, "
            f"{scan_stats.bases_processed:,} bases"
        )

    def action_run_bench(self) -> None:
        path = self.query_one("#fasta-path", Input).value.strip()
        if not path:
            if EXAMPLE_FASTA.exists():
                path = str(EXAMPLE_FASTA)
            else:
                self._log("ERROR: enter a FASTA path for benchmarking.")
                return
        try:
            guide = validate_guide(self._guide() or "GAGTCCGAGCAGAAGAAGAA")
        except AlignError as exc:
            self._log(f"ERROR: {exc}")
            return
        self._bench_worker(path, guide)

    @on(Button.Pressed, "#bench-btn")
    def _bench_pressed(self) -> None:
        self.action_run_bench()

    @work(thread=True)
    def _bench_worker(self, path: str, guide: str) -> None:
        def progress(done: int, total: int) -> None:
            self.call_from_thread(self._set_progress, done, total)

        try:
            result, candidates, _ = run_benchmark(
                path,
                guide,
                chunk_size=250_000,
                overlap=64,
                pam_patterns=self._pam(),
                strands=self._strand_tuple(),
                max_mismatches=self._max_mm(),
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._log, f"ERROR during benchmark: {exc}")
            return

        text = format_result(result) + f"\n  candidates listed : {len(candidates):,}"
        self.call_from_thread(self._on_bench_done, text)

    def _on_bench_done(self, text: str) -> None:
        self.query_one("#bench-out", Static).update(text)
        self._log("Benchmark finished; see Benchmark panel for measured results.")


def main() -> None:
    GeneWeaverApp().run()


if __name__ == "__main__":
    sys.exit(main())
