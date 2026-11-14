"""GeneWeaver terminal dashboard (Textual).

Ten functional tabs matching the application specification:

  1. Home / Overview
  2. Genome File Manager
  3. CRISPR Sequence Analyzer
  4. Alignment Results
  5. Biological Scoring
  6. CPU / GPU Performance
  7. Dask Processing Status
  8. Benchmark Comparison
  9. Reports
  10. Settings

Every control performs a real operation: genome loading parses and
validates FASTA; analysis runs the selected backend (cpu/gpu/dask);
benchmarks are measured; exports write real files. GPU metrics are only
ever shown when :func:`geneware.gpu.detect_gpu` genuinely detects a
usable backend — otherwise the UI states that GPU is unavailable and
lists the requirements.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    Button,
    Checkbox,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Select,
    Static,
    TabbedContent,
    TabPane,
)

from .aligner import AlignError, validate_guide
from .benchmark import format_result
from .dask_engine import dask_status
from .fasta import FastaParseError, compute_stats, iter_fasta
from .gpu import detect_gpu
from .pipeline import AnalysisError, run_analysis
from .reports import write_csv, write_html, write_json
from .scoring import score_formula_text

EXAMPLE_FASTA = Path(__file__).resolve().parent.parent / "data" / "example.fasta"
DEFAULT_OUT_DIR = Path("reports")

STRAND_OPTIONS = [("Both (+ and -)", "both"), ("Forward (+) only", "+"),
                  ("Reverse (-) only", "-")]
BACKEND_OPTIONS = [("CPU (sequential)", "cpu"), ("GPU (CUDA)", "gpu"),
                   ("Dask (distributed CPU)", "dask")]


@dataclass
class AppState:
    """Shared application state — all real measured data."""

    fasta_path: str | None = None
    records: list = field(default_factory=list)
    sequence_stats: dict | None = None
    analysis: object | None = None          # AnalysisResult
    benchmarks: list = field(default_factory=list)  # list[BenchmarkResult]
    # settings
    chunk_size: int = 250_000
    overlap: int = 64
    n_workers: int = 2
    use_processes: bool = True
    out_dir: str = str(DEFAULT_OUT_DIR)


class GeneWeaverApp(App):
    """GeneWeaver dashboard."""

    TITLE = "GeneWeaver"
    SUB_TITLE = "GPU-Accelerated CRISPR Alignment Engine"

    CSS = """
    Screen { layout: vertical; }
    #branding { text-align: center; padding: 0 0 1 0; }
    #branding .title { text-style: bold; color: $primary; }
    #branding .sub { color: $text-muted; }
    TabbedContent { height: 1fr; }
    TabPane { padding: 1 2; overflow-y: auto; }
    .section-label { margin-top: 1; color: $primary; text-style: bold; }
    .grid { layout: horizontal; height: auto; }
    .grid > * { width: 1fr; margin-right: 2; }
    .grid > *:last-child { margin-right: 0; }
    #genome-table, #results-table, #bench-table { height: 16; }
    #home-log, #analyzer-log, #perf-log { height: 10; border: solid $primary-darken-2; }
    .kv { color: $text-muted; }
    ProgressBar { margin: 1 0; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("ctrl+r", "refresh_status", "Refresh Status"),
    ]

    # ------------------------------------------------------------- results

    # column_key -> sort key function over ScoredCandidate
    _RESULT_SORT_KEYS = {
        "score": lambda sc: sc.score,
        "chr": lambda sc: sc.candidate.chromosome,
        "start": lambda sc: sc.candidate.start,
        "end": lambda sc: sc.candidate.end,
        "strand": lambda sc: sc.candidate.strand,
        "mm": lambda sc: sc.candidate.mismatches,
        "pam": lambda sc: sc.candidate.pam,
        "pam_ok": lambda sc: int(sc.candidate.pam_compatible),
        "mm_pos": lambda sc: sc.candidate.mismatch_positions,
    }

    _RESULT_COLUMNS = [
        ("Score", "score"), ("Chr", "chr"), ("Start", "start"),
        ("End", "end"), ("Strand", "strand"), ("MM", "mm"),
        ("PAM", "pam"), ("PAM ok", "pam_ok"), ("MM pos", "mm_pos"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self._visible_scored: list = []
        self._sort_key: str | None = None
        self._sort_reverse: bool = False

    # ------------------------------------------------------------------ UI

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="branding"):
            yield Static("GENEWEAVER", classes="title")
            yield Static(self.SUB_TITLE, classes="sub")
        tabs = TabbedContent(id="tabs")
        with tabs:
            yield from self._home_tab()
            yield from self._genome_tab()
            yield from self._analyzer_tab()
            yield from self._results_tab()
            yield from self._scoring_tab()
            yield from self._performance_tab()
            yield from self._dask_tab()
            yield from self._benchmark_tab()
            yield from self._reports_tab()
            yield from self._settings_tab()
        yield Footer()

    # -- tab builders ----------------------------------------------------

    def _home_tab(self) -> TabPane:
        with TabPane("Home", id="tab-home"):
            yield Label("Status overview", classes="section-label")
            yield Static("(refreshing…)", id="home-status")
            yield Label("Quick start", classes="section-label")
            yield Static(
                "1. Open the 'Genome' tab and load a FASTA file "
                "(or 'Load Example Genome').\n"
                "2. Open 'CRISPR Analyzer', enter a guide sequence, choose "
                "PAM / mismatches / backend, then 'Analyze'.\n"
                "3. Review scored results in 'Alignment Results', compare "
                "backends in 'Benchmark Comparison', export via 'Reports'.",
                id="home-guide",
            )
            yield Label("Log", classes="section-label")
            yield RichLog(id="home-log", wrap=True)

    def _genome_tab(self) -> TabPane:
        with TabPane("Genome", id="tab-genome"):
            with Horizontal():
                yield Input(placeholder="path/to/genome.fasta",
                             id="genome-path")
                yield Button("Load Genome", id="genome-load", variant="primary")
                yield Button("Load Example", id="genome-example")
            yield ProgressBar(total=100, show_eta=False, id="genome-progress")
            yield Static("(no genome loaded)", id="genome-summary")
            yield Label("Records", classes="section-label")
            yield DataTable(id="genome-table")

    def _analyzer_tab(self) -> TabPane:
        with TabPane("Analyzer", id="tab-analyzer"):
            yield Static("(load a genome in the Genome tab first)",
                         id="analyzer-genome-status")
            with Horizontal():
                with Vertical():
                    yield Label("Guide RNA spacer (5'->3')",
                                classes="section-label")
                    yield Input(
                        placeholder="e.g. GAGTCCGAGCAGAAGAAGAA",
                        id="guide-input")
                    yield Label("PAM patterns (comma-separated IUPAC)",
                                classes="section-label")
                    yield Input("NGG", id="pam-input")
                    yield Label("Strand", classes="section-label")
                    yield Select(STRAND_OPTIONS, value="both",
                                 allow_blank=False, id="strand-select")
                    yield Label("Max mismatches", classes="section-label")
                    yield Input("0", id="mm-input")
                with Vertical():
                    yield Label("Backend", classes="section-label")
                    yield Select(BACKEND_OPTIONS, value="cpu",
                                 allow_blank=False, id="backend-select")
                    yield Label("Dask workers (cpu backend ignores)",
                                classes="section-label")
                    yield Input("2", id="workers-input")
                    yield Checkbox("Dask worker processes",
                                   value=True, id="processes-check")
                    yield Checkbox("Require PAM match", value=True,
                                   id="require-pam-check")
                    yield Button("Analyze", id="analyze-btn",
                                 variant="success")
            yield ProgressBar(total=100, show_eta=False, id="analyze-progress")
            yield Static("(no analysis yet)", id="analyze-status")
            yield RichLog(id="analyzer-log", wrap=True)

    def _results_tab(self) -> TabPane:
        with TabPane("Results", id="tab-results"):
            with Horizontal():
                yield Input(placeholder="min score (e.g. 80)", id="f-min-score")
                yield Input(placeholder="chromosome filter", id="f-chrom")
                yield Input(placeholder="search chr/PAM/strand/mm-pos",
                            id="f-search")
                yield Select([("All strands", "all"), ("+", "+"), ("-", "-")],
                             value="all", allow_blank=False, id="f-strand")
                yield Button("Apply filters", id="filter-btn", variant="primary")
                yield Button("Clear", id="filter-clear")
            yield Static("(run an analysis first)", id="results-metrics")
            yield Static("(run an analysis first)", id="results-count")
            yield DataTable(id="results-table")
            with Horizontal():
                yield Button("Export CSV", id="exp-csv")
                yield Button("Export JSON", id="exp-json")
                yield Button("Export HTML", id="exp-html")

    def _scoring_tab(self) -> TabPane:
        with TabPane("Scoring", id="tab-scoring"):
            yield Static(score_formula_text(), id="score-formula")
            yield Label("Score summary (last analysis)",
                        classes="section-label")
            yield Static("(no analysis yet)", id="score-summary")

    def _performance_tab(self) -> TabPane:
        with TabPane("Performance", id="tab-performance"):
            with Horizontal():
                yield Select(BACKEND_OPTIONS, value="cpu", allow_blank=False,
                             id="bench-backend")
                yield Button("Run Benchmark", id="bench-run", variant="warning")
            yield ProgressBar(total=100, show_eta=False, id="bench-progress")
            yield Static("(no benchmark run yet)", id="bench-out")
            yield RichLog(id="perf-log", wrap=True)

    def _dask_tab(self) -> TabPane:
        with TabPane("Dask", id="tab-dask"):
            yield Static("(probing…)", id="dask-status")
            yield Button("Refresh Dask status", id="dask-refresh")
            yield Label("Run a Dask-distributed scan", classes="section-label")
            yield Button("Analyze with Dask", id="dask-analyze",
                         variant="primary")
            yield ProgressBar(total=100, show_eta=False, id="dask-progress")
            yield Static("(no dask scan yet)", id="dask-out")

    def _benchmark_tab(self) -> TabPane:
        with TabPane("Benchmarks", id="tab-benchmarks"):
            yield Static("(no benchmarks recorded — run one in the "
                         "Performance tab)", id="bench-count")
            yield DataTable(id="bench-table")

    def _reports_tab(self) -> TabPane:
        with TabPane("Reports", id="tab-reports"):
            yield Static("Export the last completed analysis "
                         "(set output dir in Settings).",
                         classes="section-label")
            with Horizontal():
                yield Button("Write CSV", id="rep-csv", variant="primary")
                yield Button("Write JSON", id="rep-json", variant="primary")
                yield Button("Write HTML", id="rep-html", variant="primary")
                yield Button("Write all (bundle)", id="rep-bundle",
                             variant="success")
            yield Static("(nothing exported yet)", id="rep-out")

    def _settings_tab(self) -> TabPane:
        with TabPane("Settings", id="tab-settings"):
            yield Label("Chunking", classes="section-label")
            yield Input(str(self.state.chunk_size), id="set-chunk")
            yield Input(str(self.state.overlap), id="set-overlap")
            yield Label("Dask", classes="section-label")
            yield Input(str(self.state.n_workers), id="set-workers")
            yield Checkbox("Use worker processes", value=True,
                           id="set-processes")
            yield Label("Reports", classes="section-label")
            yield Input(self.state.out_dir, id="set-outdir")
            yield Button("Save settings", id="set-save", variant="primary")
            yield Static("(settings apply to the next run)",
                         id="set-status")

    # ------------------------------------------------------------- helpers

    def _log(self, tab_log_id: str, msg: str) -> None:
        try:
            self.query_one(tab_log_id, RichLog).write(msg)
        except Exception:
            pass

    def _set_progress(self, bar_id: str, done: int, total: int) -> None:
        try:
            bar = self.query_one(bar_id, ProgressBar)
            bar.total = max(total, 1)
            bar.progress = done
        except Exception:
            pass

    def _strand_tuple(self) -> tuple[str, ...]:
        sel = self.query_one("#strand-select", Select).value
        return ("+", "-") if sel == "both" else (sel,)

    def _backend(self, default_select: str | None = None) -> str:
        sel_id = default_select or "#backend-select"
        return str(self.query_one(sel_id, Select).value)

    def _guide(self) -> str:
        return self.query_one("#guide-input", Input).value

    def _pam(self) -> tuple[str, ...]:
        raw = self.query_one("#pam-input", Input).value.strip()
        return tuple(p for p in (s.strip() for s in raw.split(",")) if p) or ("NGG",)

    def _max_mm(self) -> int:
        try:
            return max(0, int(self.query_one("#mm-input", Input).value.strip() or "0"))
        except ValueError:
            return 0

    def _require_pam(self) -> bool:
        return bool(self.query_one("#require-pam-check", Checkbox).value)

    def _refresh_home_status(self) -> None:
        gpu = detect_gpu()
        ds = dask_status()
        st = self.state
        lines = [
            f"Genome : {'loaded — ' + st.fasta_path if st.fasta_path else 'not loaded'}",
            f"GPU    : {'AVAILABLE — ' + str(gpu.device_name) if gpu.available else 'unavailable — ' + gpu.reason}",
            f"Dask   : {'available — ' + str(ds.get('workers')) + ' workers' if ds.get('available') else 'unavailable — ' + str(ds.get('reason'))}",
            f"Last analysis: {st.analysis.backend if st.analysis else 'none'} "
            f"({len(st.analysis.candidates) if st.analysis else 0} candidates)",
            f"Benchmarks recorded: {len(st.benchmarks)}",
        ]
        try:
            self.query_one("#home-status", Static).update("\n".join(lines))
        except Exception:
            pass

    def action_refresh_status(self) -> None:
        self._refresh_home_status()
        self._refresh_dask_status()

    # ------------------------------------------------------- genome loading

    @on(Button.Pressed, "#genome-load")
    @on(Button.Pressed, "#genome-example")
    def load_genome(self, event: Button.Pressed) -> None:
        if event.button.id == "genome-example":
            path = str(EXAMPLE_FASTA)
        else:
            path = self.query_one("#genome-path", Input).value.strip()
            if not path:
                self._log("#analyzer-log", "ERROR: enter a FASTA path first.")
                return
        self._load_genome_worker(path)

    @work(thread=True)
    def _load_genome_worker(self, path_str: str) -> None:
        try:
            records = list(iter_fasta(path_str))
            stats = compute_stats(records).to_dict()
        except FastaParseError as exc:
            self.call_from_thread(self._log, "#analyzer-log",
                                  f"ERROR: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._log, "#analyzer-log",
                                  f"ERROR: {type(exc).__name__}: {exc}")
            return
        self.call_from_thread(self._on_genome_loaded, path_str, records, stats)

    def _on_genome_loaded(self, path_str, records, stats) -> None:
        st = self.state
        st.fasta_path = path_str
        st.records = records
        st.sequence_stats = stats

        self.query_one("#genome-summary", Static).update(
            f"file: {path_str}\nrecords: {stats['record_count']}  "
            f"bases: {stats['total_bases']:,}  GC: {stats['gc_percent']}%  "
            f"ambiguous: {stats['ambiguous_count']:,}"
        )
        table = self.query_one("#genome-table", DataTable)
        table.clear(columns=True)
        table.add_columns("ID", "Length", "GC %", "Description")
        for rec, sr in zip(records, stats["per_record"], strict=True):
            table.add_row(rec.record_id, f"{rec.length:,}",
                          f"{sr['gc_percent']}", rec.description[:60])
        self._log("#analyzer-log",
                  f"Loaded {stats['record_count']} record(s), "
                  f"{stats['total_bases']:,} bases from {path_str}")
        self._log("#home-log",
                  f"Genome loaded: {path_str} ({stats['total_bases']:,} bases)")
        self._refresh_home_status()
        self._set_progress("#genome-progress", 100, 100)

    # ------------------------------------------------------------- analyze

    @on(Button.Pressed, "#analyze-btn")
    @on(Button.Pressed, "#dask-analyze")
    def start_analysis(self, event: Button.Pressed) -> None:
        if not self.state.records:
            self._log("#analyzer-log", "ERROR: load a genome first.")
            return
        backend = ("dask" if event.button.id == "dask-analyze"
                   else self._backend())
        if backend == "gpu":
            gpu = detect_gpu()
            if not gpu.available:
                self._log(
                    "#analyzer-log",
                    "ERROR: GPU unavailable — " + gpu.reason
                    + ". See Home tab / `python -m geneware.cli gpu-status`.")
                return
        try:
            guide = validate_guide(self._guide())
        except AlignError as exc:
            self._log("#analyzer-log", f"ERROR: {exc}")
            return
        try:
            workers = int(self.query_one("#workers-input", Input).value or "2")
        except ValueError:
            workers = 2
        self._analyze_worker(
            backend=backend, guide=guide, workers=workers,
            progress_bar="#analyze-progress" if backend != "dask"
            else "#dask-progress")

    @work(thread=True)
    def _analyze_worker(self, backend: str, guide: str, workers: int,
                        progress_bar: str) -> None:
        def progress(done: int, total: int) -> None:
            self.call_from_thread(self._set_progress, progress_bar, done, total)

        try:
            result = run_analysis(
                self.state.fasta_path, guide, backend=backend,
                chunk_size=self.state.chunk_size,
                overlap=self.state.overlap,
                pam_patterns=self._pam(),
                strands=self._strand_tuple(),
                max_mismatches=self._max_mm(),
                require_pam=self._require_pam(),
                n_workers=workers,
                processes=self.state.use_processes,
                records=self.state.records,
                progress=progress,
            )
        except (AnalysisError, FastaParseError, AlignError) as exc:
            self.call_from_thread(self._log, "#analyzer-log",
                                  f"ERROR: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._log, "#analyzer-log",
                                  f"ERROR: {type(exc).__name__}: {exc}")
            return
        self.call_from_thread(self._on_analysis_done, result)

    def _on_analysis_done(self, result) -> None:
        self.state.analysis = result
        self._fill_results(result)
        self._fill_score_summary(result)
        msg = (f"Analysis done [{result.backend}]: "
               f"{result.stats.candidates_found:,} candidates, "
               f"{result.stats.windows_evaluated:,} windows, "
               f"{result.elapsed_seconds:.3f}s")
        self._log("#analyzer-log", msg)
        self._log("#home-log", msg)
        if result.extra.get("tasks_failed"):
            self._log("#analyzer-log",
                      f"WARNING: {result.extra['tasks_failed']} dask task(s) failed")
        self._refresh_home_status()

    def _fill_results(self, result) -> None:
        self._visible_scored = list(result.scored)
        self._sort_key = None
        self._sort_reverse = False
        self._render_results()
        sstats = result.sequence_stats or {}
        scan = result.stats
        total_mm = sum(c.mismatches for c in result.candidates)
        self.query_one("#results-metrics", Static).update(
            f"sequences processed: {sstats.get('record_count', len(result.records))}"
            f"  |  total bases: {scan.bases_processed:,}"
            f"  |  windows evaluated: {scan.windows_evaluated:,}"
            f"  |  matches found: {scan.candidates_found:,}"
            f"  |  total mismatches: {total_mm:,}"
            f"  |  backend: {result.backend}"
            f"  |  elapsed: {result.elapsed_seconds:.3f}s"
        )

    def _render_results(self) -> None:
        """Render ``self._visible_scored`` into the results table."""
        scored = self._visible_scored
        if self._sort_key in self._RESULT_SORT_KEYS:
            keyfn = self._RESULT_SORT_KEYS[self._sort_key]
            scored = sorted(scored, key=keyfn, reverse=self._sort_reverse)
        table = self.query_one("#results-table", DataTable)
        table.clear(columns=True)
        table.cursor_type = "row"
        table.add_columns(*self._RESULT_COLUMNS)
        for sc in scored:
            c = sc.candidate
            table.add_row(
                f"{sc.score:.1f}", c.chromosome, str(c.start),
                str(c.end), c.strand, str(c.mismatches), c.pam,
                "yes" if c.pam_compatible else "no",
                ",".join(map(str, c.mismatch_positions)) or "-",
            )
        analysis = self.state.analysis
        total = len(analysis.scored) if analysis is not None else 0
        if not scored:
            if total == 0:
                msg = ("No matches found. Try increasing 'Max mismatches', "
                       "relaxing the PAM pattern, or checking the guide "
                       "sequence.")
            else:
                msg = (f"0 of {total:,} candidates match the current "
                       "filters.")
        elif total and len(scored) != total:
            msg = f"{len(scored):,} of {total:,} candidates (filtered)"
        else:
            msg = (f"{len(scored):,} candidates — click a column header "
                   "to sort")
        self.query_one("#results-count", Static).update(msg)

    @on(DataTable.HeaderSelected, "#results-table")
    def sort_results(self, event: DataTable.HeaderSelected) -> None:
        key = str(event.column_key)
        if key not in self._RESULT_SORT_KEYS:
            return
        if self._sort_key == key:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_key, self._sort_reverse = key, False
        self._render_results()

    def _fill_score_summary(self, result) -> None:
        s = result.score_summary
        self.query_one("#score-summary", Static).update(
            f"count={s['count']}  mean={s['mean_score']}  "
            f"min={s['min_score']}  max={s['max_score']}  "
            f"perfect={s['perfect_matches']}\n\n{score_formula_text()}"
        )

    # -------------------------------------------------------------- filters

    @on(Button.Pressed, "#filter-btn")
    def apply_filters(self) -> None:
        result = self.state.analysis
        if result is None:
            self._log("#analyzer-log", "ERROR: no analysis to filter.")
            return
        try:
            min_score = float(self.query_one("#f-min-score", Input).value or "-inf")
        except ValueError:
            min_score = float("-inf")
        chrom = self.query_one("#f-chrom", Input).value.strip().lower()
        search = self.query_one("#f-search", Input).value.strip().lower()
        strand = self.query_one("#f-strand", Select).value

        def _matches(sc) -> bool:
            c = sc.candidate
            if sc.score < min_score:
                return False
            if chrom and chrom not in c.chromosome.lower():
                return False
            if strand != "all" and c.strand != strand:
                return False
            if search:
                haystack = " ".join([
                    c.chromosome, c.strand, c.pam,
                    ",".join(map(str, c.mismatch_positions)),
                    str(c.start), str(c.end),
                ]).lower()
                if search not in haystack:
                    return False
            return True

        self._visible_scored = [sc for sc in result.scored if _matches(sc)]
        self._render_results()

    @on(Button.Pressed, "#filter-clear")
    def clear_filters(self) -> None:
        self.query_one("#f-min-score", Input).value = ""
        self.query_one("#f-chrom", Input).value = ""
        self.query_one("#f-search", Input).value = ""
        self.query_one("#f-strand", Select).value = "all"
        if self.state.analysis is not None:
            self._visible_scored = list(self.state.analysis.scored)
            self._render_results()

    # ------------------------------------------------------------ benchmark

    @on(Button.Pressed, "#bench-run")
    def run_benchmark(self) -> None:
        if not self.state.fasta_path:
            path = str(EXAMPLE_FASTA) if EXAMPLE_FASTA.exists() else ""
            if not path:
                self._log("#perf-log", "ERROR: load a genome first.")
                return
        else:
            path = self.state.fasta_path
        try:
            guide = validate_guide(self._guide() or "GAGTCCGAGCAGAAGAAGAA")
        except AlignError as exc:
            self._log("#perf-log", f"ERROR: {exc}")
            return
        backend = self._backend("#bench-backend")
        if backend == "gpu" and not detect_gpu().available:
            self._log("#perf-log",
                      "ERROR: GPU unavailable — benchmark requires a "
                      "working CUDA backend.")
            return
        self._bench_worker(path, guide, backend)

    @work(thread=True)
    def _bench_worker(self, path: str, guide: str, backend: str) -> None:
        def progress(done: int, total: int) -> None:
            self.call_from_thread(self._set_progress, "#bench-progress",
                                  done, total)

        try:
            from .pipeline import run_benchmark as _rb
            result, analysis = _rb(
                path, guide, backend=backend,
                chunk_size=self.state.chunk_size,
                overlap=self.state.overlap,
                pam_patterns=self._pam(),
                strands=self._strand_tuple(),
                max_mismatches=self._max_mm(),
                require_pam=self._require_pam(),
                n_workers=self.state.n_workers,
                processes=self.state.use_processes,
                records=self.state.records or None,
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._log, "#perf-log",
                                  f"ERROR: {type(exc).__name__}: {exc}")
            return

        self.call_from_thread(self._on_bench_done, result, analysis)

    def _on_bench_done(self, result, analysis) -> None:
        self.state.benchmarks.append(result)
        self.state.analysis = analysis  # keep results consistent
        self._fill_results(analysis)
        self._fill_score_summary(analysis)
        self.query_one("#bench-out", Static).update(
            format_result(result)
            + f"\n  candidates : {len(analysis.candidates):,}"
            + f"\n  backend    : {analysis.backend}"
        )
        self._fill_bench_table()
        self._log("#perf-log",
                  f"Benchmark [{analysis.backend}] finished: "
                  f"{result.throughput_bases_per_sec:,.0f} bases/s")

    def _fill_bench_table(self) -> None:
        table = self.query_one("#bench-table", DataTable)
        table.clear(columns=True)
        table.add_columns("Backend", "Align s", "Total s", "Bases", "Windows",
                          "Candidates", "Bases/s", "Peak mem")
        for r in self.state.benchmarks:
            backend = next((n for n in r.notes if n.startswith("backend:")),
                           "cpu")
            table.add_row(
                backend.split(":", 1)[-1].strip(),
                f"{r.align_ns / 1e9:.4f}", f"{r.total_seconds:.4f}",
                f"{r.bases_processed:,}", f"{r.windows_evaluated:,}",
                f"{r.candidates_found:,}",
                f"{r.throughput_bases_per_sec:,.0f}",
                "n/a" if r.memory_peak_bytes is None else
                f"{r.memory_peak_bytes / 1048576:.1f} MiB",
            )
        self.query_one("#bench-count", Static).update(
            f"{len(self.state.benchmarks)} benchmark(s) recorded "
            "(all values measured at runtime)"
        )

    # ---------------------------------------------------------------- dask

    @on(Button.Pressed, "#dask-refresh")
    def refresh_dask(self) -> None:
        self._refresh_dask_status()

    def _refresh_dask_status(self) -> None:
        st = dask_status()
        if st.get("available"):
            text = (f"Dask available: {st.get('workers')} worker(s), "
                    f"{st.get('total_threads')} thread(s), "
                    f"{(st.get('total_memory_bytes') or 0) / 1073741824:.2f} GiB")
        else:
            text = f"Dask unavailable: {st.get('reason')}"
        try:
            self.query_one("#dask-status", Static).update(text)
            self.query_one("#dask-out", Static).update(text)
        except Exception:
            pass

    # -------------------------------------------------------------- reports

    def _out_dir(self) -> Path:
        return Path(self.state.out_dir)

    @on(Button.Pressed, "#rep-csv")
    @on(Button.Pressed, "#rep-json")
    @on(Button.Pressed, "#rep-html")
    @on(Button.Pressed, "#rep-bundle")
    def export_report(self, event: Button.Pressed) -> None:
        result = self.state.analysis
        if result is None:
            self._log("#analyzer-log", "ERROR: run an analysis first.")
            self.query_one("#rep-out", Static).update(
                "ERROR: run an analysis first.")
            return
        out = self._out_dir()
        try:
            if event.button.id == "rep-csv":
                p = write_csv(result, out / "geneware_report.csv")
                msg = f"CSV written: {p}"
            elif event.button.id == "rep-json":
                p = write_json(result, out / "geneware_report.json")
                msg = f"JSON written: {p}"
            elif event.button.id == "rep-html":
                p = write_html(result, out / "geneware_report.html")
                msg = f"HTML written: {p}"
            else:
                from .reports import write_report_bundle
                paths = write_report_bundle(result, out)
                msg = "Bundle written: " + ", ".join(
                    f"{k}={v}" for k, v in paths.items())
        except Exception as exc:  # noqa: BLE001
            self.query_one("#rep-out", Static).update(
                f"ERROR: {type(exc).__name__}: {exc}")
            return
        self.query_one("#rep-out", Static).update(msg)
        self._log("#home-log", msg)

    # ------------------------------------------------------------- settings

    @on(Button.Pressed, "#set-save")
    def save_settings(self) -> None:
        try:
            self.state.chunk_size = max(1_000, int(
                self.query_one("#set-chunk", Input).value))
            self.state.overlap = max(0, int(
                self.query_one("#set-overlap", Input).value))
            self.state.n_workers = max(1, int(
                self.query_one("#set-workers", Input).value))
            self.state.use_processes = bool(
                self.query_one("#set-processes", Checkbox).value)
            self.state.out_dir = self.query_one(
                "#set-outdir", Input).value.strip() or str(DEFAULT_OUT_DIR)
        except ValueError as exc:
            self.query_one("#set-status", Static).update(
                f"ERROR: invalid number — {exc}")
            return
        self.query_one("#set-status", Static).update(
            f"Saved: chunk={self.state.chunk_size:,} "
            f"overlap={self.state.overlap} workers={self.state.n_workers} "
            f"processes={self.state.use_processes} out={self.state.out_dir}")

    def on_mount(self) -> None:
        self._refresh_home_status()
        self._refresh_dask_status()


def main() -> None:
    GeneWeaverApp().run()


if __name__ == "__main__":
    sys.exit(main())
