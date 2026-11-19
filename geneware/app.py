"""GeneWeaver terminal dashboard (Textual).

Ten functional sections matching the application specification:

  1. Overview
  2. Genome Files
  3. CRISPR Analysis
  4. Alignment Results
  5. Biological Scoring
  6. Processing Performance
  7. CPU and GPU Status
  8. Dask Processing
  9. Reports and Exports
  10. Settings

Every control performs a real operation: genome loading parses and
validates FASTA; analysis runs the selected backend (cpu/gpu/dask);
benchmarks are measured; exports write real files. GPU metrics are only
ever shown when :func:`geneware.gpu.detect_gpu` genuinely detects a
usable backend — otherwise the UI states that GPU is unavailable and
lists the requirements.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

from rich.text import Text
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
from .alignment_view import alignment_segments, candidate_view
from .benchmark import format_bytes, format_result
from .dask_engine import dask_status
from .fasta import FastaParseError, compute_stats, iter_fasta
from .gpu import detect_gpu
from .pipeline import AnalysisError, _resolve_auto_backend, run_analysis
from .reports import write_csv, write_html, write_json, write_pdf
from .scoring import score_breakdown, score_formula_text, score_summary

EXAMPLE_FASTA = Path(__file__).resolve().parent.parent / "data" / "example.fasta"
DEFAULT_OUT_DIR = Path("reports")

STRAND_OPTIONS = [("Both (+ and -)", "both"), ("Forward (+) only", "+"),
                  ("Reverse (-) only", "-")]
BACKEND_OPTIONS = [("Auto (best available)", "auto"),
                   ("CPU (sequential)", "cpu"), ("GPU (CUDA)", "gpu"),
                   ("Dask (distributed CPU)", "dask")]


class _AnalysisCancelled(Exception):
    """Raised inside a scan worker when the user requests cancellation."""


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
    #results-detail { height: 12; border: solid $primary-darken-2; }
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
        "type": lambda sc: (sc.candidate.match_type, sc.candidate.mismatches),
        "chr": lambda sc: sc.candidate.chromosome,
        "start": lambda sc: sc.candidate.start,
        "end": lambda sc: sc.candidate.end,
        "strand": lambda sc: sc.candidate.strand,
        "mm": lambda sc: sc.candidate.mismatches,
        "mm_pos": lambda sc: sc.candidate.mismatch_positions,
        "ambig": lambda sc: sc.candidate.ambiguous_positions,
        "pam": lambda sc: sc.candidate.pam,
        "pam_ok": lambda sc: int(sc.candidate.pam_compatible),
        "guide": lambda sc: sc.candidate.guide,
        "match": lambda sc: sc.candidate.matched_sequence,
    }

    _RESULT_COLUMNS = [
        ("Score", "score"), ("Type", "type"), ("Chr", "chr"),
        ("Start", "start"), ("End", "end"), ("Strand", "strand"),
        ("MM", "mm"), ("MM pos", "mm_pos"), ("Ambig", "ambig"),
        ("PAM", "pam"), ("PAM ok", "pam_ok"), ("Match", "match"),
        ("Guide", "guide"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.state = AppState()
        self._visible_scored: list = []
        self._displayed_scored: list = []
        self._sort_key: str | None = None
        self._sort_reverse: bool = False
        self._analysis_worker = None
        self._cancel_requested = False
        self._active_status_id = "#analyze-status"

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
            yield from self._systems_tab()
            yield from self._dask_tab()
            yield from self._reports_tab()
            yield from self._settings_tab()
        yield Footer()

    # -- tab builders ----------------------------------------------------

    def _home_tab(self) -> TabPane:
        with TabPane("Overview", id="tab-home"):
            yield Label("Current state", classes="section-label")
            yield Static("(refreshing…)", id="home-status")
            yield Label("Quick start", classes="section-label")
            yield Static(
                "1. Open 'Genome Files' and load a FASTA file "
                "(or 'Load Example Genome').\n"
                "2. Open 'CRISPR Analysis', enter a guide sequence, choose "
                "PAM / mismatches / backend, then 'Analyze'.\n"
                "3. Follow live progress, review scored results in "
                "'Alignment Results', compare backends in 'Processing "
                "Performance', and export via 'Reports and Exports'.",
                id="home-guide",
            )
            yield Label("Log", classes="section-label")
            yield RichLog(id="home-log", wrap=True)

    def _genome_tab(self) -> TabPane:
        with TabPane("Genome Files", id="tab-genome"):
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
        with TabPane("CRISPR Analysis", id="tab-analyzer"):
            yield Static("(load a genome in 'Genome Files' first)",
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
                    yield Button("Cancel", id="analyze-cancel")
            yield ProgressBar(total=100, show_eta=False, id="analyze-progress")
            yield Static("(no analysis yet)", id="analyze-status")
            yield RichLog(id="analyzer-log", wrap=True)

    def _results_tab(self) -> TabPane:
        with TabPane("Alignment Results", id="tab-results"):
            with Horizontal():
                yield Input(placeholder="min score (e.g. 80)", id="f-min-score")
                yield Input(placeholder="max mismatches", id="f-max-mm")
                yield Input(placeholder="chromosome filter", id="f-chrom")
                yield Input(placeholder="search chr/PAM/strand/seq/coords",
                            id="f-search")
            with Horizontal():
                yield Select([("All strands", "all"), ("+", "+"), ("-", "-")],
                             value="all", allow_blank=False, id="f-strand")
                yield Select([("All PAM", "all"), ("PAM ok", "yes"),
                              ("PAM bad", "no")],
                             value="all", allow_blank=False, id="f-pam")
                yield Select([("All types", "all"), ("Exact", "exact"),
                              ("Near", "near")],
                             value="all", allow_blank=False, id="f-type")
                yield Button("Apply filters", id="filter-btn", variant="primary")
                yield Button("Clear", id="filter-clear")
            yield Static("(run an analysis first)", id="results-guide")
            yield Static("(run an analysis first)", id="results-metrics")
            yield Static("(run an analysis first)", id="results-count")
            yield DataTable(id="results-table")
            yield Label("Candidate detail / alignment", classes="section-label")
            yield RichLog(id="results-detail", wrap=True, max_lines=14)
            with Horizontal():
                yield Button("Export CSV", id="exp-csv")
                yield Button("Export JSON", id="exp-json")
                yield Button("Export HTML", id="exp-html")
                yield Button("Export PDF", id="exp-pdf")

    def _scoring_tab(self) -> TabPane:
        with TabPane("Biological Scoring", id="tab-scoring"):
            yield Static(score_formula_text(), id="score-formula")
            yield Label("Score summary (last analysis)",
                        classes="section-label")
            yield Static("(no analysis yet)", id="score-summary")

    def _performance_tab(self) -> TabPane:
        with TabPane("Processing Performance", id="tab-performance"):
            yield Label("Run a measured benchmark", classes="section-label")
            with Horizontal():
                yield Select(BACKEND_OPTIONS, value="cpu", allow_blank=False,
                             id="bench-backend")
                yield Button("Run Benchmark", id="bench-run", variant="warning")
            yield ProgressBar(total=100, show_eta=False, id="bench-progress")
            yield Static("(no benchmark run yet)", id="bench-out")
            yield Label("Last run (measured)", classes="section-label")
            yield Static("(no analysis yet)", id="perf-last")
            yield Label("Recorded benchmarks", classes="section-label")
            yield Static("(no benchmarks recorded — run one above)",
                         id="bench-count")
            yield DataTable(id="bench-table")
            yield RichLog(id="perf-log", wrap=True)

    def _systems_tab(self) -> TabPane:
        with TabPane("CPU/GPU Status", id="tab-systems"):
            yield Label("CPU", classes="section-label")
            yield Static("(probing…)", id="sys-cpu")
            yield Label("GPU", classes="section-label")
            yield Static("(probing…)", id="sys-gpu")
            yield Label("Execution engines", classes="section-label")
            yield Static("(probing…)", id="sys-engines")
            with Horizontal():
                yield Button("Refresh status", id="sys-refresh",
                             variant="primary")
            yield RichLog(id="sys-log", wrap=True)

    def _dask_tab(self) -> TabPane:
        with TabPane("Dask Processing", id="tab-dask"):
            yield Static("(probing…)", id="dask-status")
            yield Button("Refresh Dask status", id="dask-refresh")
            yield Label("Run a Dask-distributed scan", classes="section-label")
            yield Button("Analyze with Dask", id="dask-analyze",
                         variant="primary")
            yield ProgressBar(total=100, show_eta=False, id="dask-progress")
            yield Static("(no dask scan yet)", id="dask-out")

    def _reports_tab(self) -> TabPane:
        with TabPane("Reports and Exports", id="tab-reports"):
            yield Static("Export the last completed analysis "
                         "(set output dir in Settings).",
                         classes="section-label")
            with Horizontal():
                yield Button("Write CSV", id="rep-csv", variant="primary")
                yield Button("Write JSON", id="rep-json", variant="primary")
                yield Button("Write HTML", id="rep-html", variant="primary")
                yield Button("Write PDF", id="rep-pdf", variant="primary")
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
        raw = self.query_one("#mm-input", Input).value.strip() or "0"
        try:
            value = int(raw)
        except ValueError as exc:
            raise AlignError("max mismatches must be an integer") from exc
        if value < 0:
            raise AlignError("max mismatches must be >= 0")
        return value

    def _require_pam(self) -> bool:
        return bool(self.query_one("#require-pam-check", Checkbox).value)

    def _refresh_home_status(self) -> None:
        """Render the Overview panel from real application state only."""
        st = self.state
        gpu = detect_gpu()
        ds = dask_status()

        if st.fasta_path and st.sequence_stats:
            s = st.sequence_stats
            genome = (
                f"Genome : {Path(st.fasta_path).name}\n"
                f"  records: {s.get('record_count')}   "
                f"bases: {s.get('total_bases'):,}   "
                f"GC: {s.get('gc_percent')}%   "
                f"ambiguous: {s.get('ambiguous_count'):,}"
            )
        elif st.fasta_path:
            genome = f"Genome : {st.fasta_path} (statistics unavailable)"
        else:
            genome = ("Genome : not loaded — open 'Genome Files' and load "
                      "a FASTA file (or 'Load Example Genome').")

        a = st.analysis
        if a is not None:
            analysis = (
                f"Last analysis [{a.backend}]: "
                f"{len(a.scored):,} candidates exported, "
                f"{a.stats.candidates_found:,} matches found, "
                f"{a.elapsed_seconds:.3f}s\n"
                f"  method: {a.processing_method}   "
                f"guide: {a.guide}"
            )
        else:
            analysis = ("Last analysis: none — configure a guide in "
                        "'CRISPR Analysis' and press Analyze.")

        if st.benchmarks:
            b = st.benchmarks[-1]
            bench = (
                f"Latest benchmark [{b.backend}]: "
                f"{b.total_seconds:.3f}s total, "
                f"{b.throughput_bases_per_sec:,.0f} bases/s, "
                f"{b.candidates_found:,} candidates"
            )
        else:
            bench = ("Latest benchmark: none — run one in 'Processing "
                     "Performance'.")

        backends = (
            f"Backends: GPU {'available' if gpu.available else 'unavailable'}"
            f" | Dask {'available' if ds.get('available') else 'unavailable'}"
            f" | auto -> {_resolve_auto_backend()}"
        )
        text = "\n".join([genome, "", analysis, "", bench, "", backends])
        try:
            self.query_one("#home-status", Static).update(text)
        except Exception:
            pass

    # ----------------------------------------------------- systems status

    def _refresh_systems(self, force: bool = False) -> None:
        """Probe and display real CPU/GPU/engine status."""
        try:
            cpu_count = os.cpu_count() or 0
            try:
                proc_count = os.process_cpu_count() or cpu_count
            except (AttributeError, OSError):
                proc_count = cpu_count
            cpu_text = (
                f"processor : {platform.processor() or 'unknown'}\n"
                f"cores     : {cpu_count} logical "
                f"({proc_count} usable by this process)\n"
                f"platform  : {platform.system()} {platform.release()} "
                f"{platform.machine()}\n"
                f"python    : {platform.python_version()} "
                f"({sys.executable})"
            )
            self.query_one("#sys-cpu", Static).update(cpu_text)
        except Exception as exc:  # noqa: BLE001
            try:
                self.query_one("#sys-cpu", Static).update(
                    f"CPU probe failed: {type(exc).__name__}: {exc}")
            except Exception:
                pass

        gpu = detect_gpu(force=force)
        if gpu.available:
            gpu_text = (
                f"GPU: AVAILABLE — backend {gpu.backend}\n"
                f"  device    : {gpu.device_name}\n"
                f"  compute   : CC {gpu.compute_capability}\n"
                f"  VRAM      : {gpu.vram_free_mb:.0f} MiB free / "
                f"{gpu.vram_total_mb:.0f} MiB total\n"
                f"  note      : {gpu.driver_note or 'driver OK'}"
                if gpu.vram_total_mb is not None
                and gpu.vram_free_mb is not None else
                f"GPU: AVAILABLE — backend {gpu.backend}, "
                f"device {gpu.device_name}"
            )
        else:
            gpu_text = (
                f"GPU: unavailable — {gpu.reason}\n"
                "Requirements:\n"
                + "\n".join(f"  - {r}" for r in gpu.requirements)
            )
        try:
            self.query_one("#sys-gpu", Static).update(gpu_text)
        except Exception:
            pass

        engine_lines = [
            f"aligner engine : {os.environ.get('GENEWEARE_ALIGNER_ENGINE', 'auto')}"
            " (auto = numba-JIT when importable, else encoded pure Python)",
        ]
        try:
            import numba  # noqa: F401
            engine_lines.append(
                f"numba         : {numba.__version__} — JIT CPU kernel "
                "and CUDA backend available")
        except Exception as exc:
            engine_lines.append(
                f"numba         : unavailable ({type(exc).__name__}) — "
                "pure-Python aligner fallback in use")
        try:
            import dask  # noqa: F401
            import distributed  # noqa: F401
            engine_lines.append(
                f"dask          : {dask.__version__} / "
                f"{distributed.__version__} — distributed backend usable")
        except Exception:
            engine_lines.append("dask          : unavailable")
        try:
            engine_lines.append(
                f"auto resolves : {_resolve_auto_backend()} "
                "(gpu > dask > cpu, probed live)")
        except Exception:
            pass
        try:
            self.query_one("#sys-engines", Static).update(
                "\n".join(engine_lines))
        except Exception:
            pass

    @on(Button.Pressed, "#sys-refresh")
    def refresh_systems(self) -> None:
        self._refresh_systems(force=True)

    def action_refresh_status(self) -> None:
        self._refresh_home_status()
        self._refresh_dask_status()
        self._refresh_systems()

    def _update_perf_last(self, analysis) -> None:
        """Render the last run's measured performance into the panel."""
        st = analysis.stats
        ex = analysis.extra or {}
        elapsed = analysis.elapsed_seconds
        tput = st.bases_processed / elapsed if elapsed > 0 else 0.0
        wput = st.windows_evaluated / elapsed if elapsed > 0 else 0.0
        chunks_done = ex.get("chunks_completed", st.chunks_scanned)
        chunks_total = ex.get("chunks_total", st.chunks_scanned)
        mem = ex.get("memory_peak_bytes")
        mode = ex.get("mode") or ex.get("dask_backend") or ""
        failed = ex.get("tasks_failed", 0)
        status = "completed" if not failed else f"completed, {failed} task(s) failed"
        lines = [
            f"method      : {analysis.backend} {mode}".rstrip(),
            f"status      : {status}",
            f"chunks      : {chunks_done}/{chunks_total}",
            f"time        : {elapsed:.3f} s",
            f"throughput  : {tput:,.0f} bases/s ({wput:,.0f} windows/s)",
            f"bases       : {st.bases_processed:,}   "
            f"windows: {st.windows_evaluated:,}   "
            f"candidates: {st.candidates_found:,}",
            f"peak python memory: {format_bytes(mem)}",
        ]
        try:
            self.query_one("#perf-last", Static).update("\n".join(lines))
        except Exception:
            pass

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
        try:
            self.query_one("#analyzer-genome-status", Static).update(
                f"Genome ready: {Path(path_str).name} — "
                f"{stats['record_count']} record(s), "
                f"{stats['total_bases']:,} bases, GC {stats['gc_percent']}%")
        except Exception:
            pass
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
                    + ". See 'CPU/GPU Status' / `python -m geneware.cli gpu-status`.")
                return
        try:
            guide = validate_guide(self._guide())
            max_mm = self._max_mm()
            pams = self._pam()
            strands = self._strand_tuple()
            require_pam = self._require_pam()
            workers = int(self.query_one("#workers-input", Input).value or "2")
        except (AlignError, ValueError) as exc:
            self._log("#analyzer-log", f"ERROR: {exc}")
            return
        self._cancel_requested = False
        progress_bar = "#dask-progress" if backend == "dask" else "#analyze-progress"
        self._active_status_id = ("#dask-out" if backend == "dask"
                                  else "#analyze-status")
        self._set_progress(progress_bar, 0, 1)
        self._set_analysis_status(
            f"Starting {backend} analysis…", self._active_status_id)
        self._analysis_worker = self._analyze_worker(
            backend=backend, guide=guide, workers=workers,
            pam_patterns=pams, strands=strands, max_mismatches=max_mm,
            require_pam=require_pam, progress_bar=progress_bar)

    @on(Button.Pressed, "#analyze-cancel")
    def cancel_analysis(self) -> None:
        worker = self._analysis_worker
        running = worker is not None and getattr(worker, "is_running", False)
        if self._cancel_requested or running:
            self._cancel_requested = True
            if running:
                try:
                    worker.cancel()
                except Exception:
                    pass
            self._log("#analyzer-log",
                      "Cancellation requested — the scan will stop at the "
                      "next chunk boundary.")
            self._set_analysis_status(
                "Cancelling…", "#analyze-status")
        else:
            self._log("#analyzer-log", "No analysis is currently running.")

    def _set_analysis_status(self, text: str, widget_id: str) -> None:
        try:
            self.query_one(widget_id, Static).update(text)
        except Exception:
            pass

    @work(thread=True)
    def _analyze_worker(self, backend: str, guide: str, workers: int,
                        pam_patterns, strands, max_mismatches: int,
                        require_pam: bool, progress_bar: str) -> None:
        status_id = self._active_status_id
        t0 = time.perf_counter()

        def progress(done: int, total: int) -> None:
            if self._cancel_requested:
                raise _AnalysisCancelled("cancelled by user")
            elapsed = time.perf_counter() - t0
            self.call_from_thread(self._set_progress, progress_bar, done, total)
            self.call_from_thread(
                self._set_analysis_status,
                f"{backend}: {done}/{total} chunks — {elapsed:.1f}s elapsed",
                status_id)

        try:
            result = run_analysis(
                self.state.fasta_path, guide, backend=backend,
                chunk_size=self.state.chunk_size,
                overlap=self.state.overlap,
                pam_patterns=pam_patterns,
                strands=strands,
                max_mismatches=max_mismatches,
                require_pam=require_pam,
                n_workers=workers,
                processes=self.state.use_processes,
                records=self.state.records,
                progress=progress,
            )
        except _AnalysisCancelled as exc:
            self.call_from_thread(
                self._set_analysis_status, f"Cancelled: {exc}", status_id)
            self.call_from_thread(self._log, "#analyzer-log",
                                  f"Analysis cancelled by user ({backend}).")
            return
        except (AnalysisError, FastaParseError, AlignError) as exc:
            self.call_from_thread(self._log, "#analyzer-log",
                                  f"ERROR: {exc}")
            self.call_from_thread(
                self._set_analysis_status, f"Failed: {exc}", status_id)
            return
        except Exception as exc:  # noqa: BLE001
            self.call_from_thread(self._log, "#analyzer-log",
                                  f"ERROR: {type(exc).__name__}: {exc}")
            self.call_from_thread(
                self._set_analysis_status,
                f"Failed: {type(exc).__name__}: {exc}", status_id)
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
        self._set_analysis_status(
            f"{result.backend}: complete — "
            f"{result.stats.candidates_found:,} matches, "
            f"{result.elapsed_seconds:.3f}s", self._active_status_id)
        self._log("#analyzer-log", msg)
        self._log("#home-log", msg)
        self._update_perf_last(result)
        if result.extra.get("tasks_failed"):
            self._log("#analyzer-log",
                      f"WARNING: {result.extra['tasks_failed']} dask task(s) failed")
        self._refresh_home_status()

    def _fill_results(self, result) -> None:
        self._visible_scored = list(result.scored)
        self._displayed_scored = []
        self._sort_key = None
        self._sort_reverse = False
        self._render_results()
        sstats = result.sequence_stats or {}
        scan = result.stats
        total_mm = sum(c.mismatches for c in result.candidates)
        self.query_one("#results-guide", Static).update(
            f"guide: {result.guide}  |  PAM: {', '.join(result.pam_patterns)}"
            f"  |  max mismatches: {result.max_mismatches}"
            f"  |  strands: {''.join(result.strands)}"
            f"  |  require PAM: {'yes' if result.require_pam else 'no'}"
        )
        self.query_one("#results-metrics", Static).update(
            f"sequences processed: {sstats.get('record_count', len(result.records))}"
            f"  |  total bases: {scan.bases_processed:,}"
            f"  |  windows evaluated: {scan.windows_evaluated:,}"
            f"  |  matches found: {scan.candidates_found:,}"
            f"  |  total mismatches: {total_mm:,}"
            f"  |  method: {result.processing_method}"
            f"  |  elapsed: {result.elapsed_seconds:.3f}s"
        )

    def _render_results(self) -> None:
        """Render ``self._visible_scored`` into the results table."""
        scored = list(self._visible_scored)
        if self._sort_key in self._RESULT_SORT_KEYS:
            keyfn = self._RESULT_SORT_KEYS[self._sort_key]
            scored = sorted(scored, key=keyfn, reverse=self._sort_reverse)
        self._displayed_scored = scored
        table = self.query_one("#results-table", DataTable)
        table.clear(columns=True)
        table.cursor_type = "row"
        table.add_columns(*self._RESULT_COLUMNS)
        for sc in scored:
            c = sc.candidate
            table.add_row(
                f"{sc.score:.1f}", c.match_type, c.chromosome,
                str(c.start), str(c.end), c.strand, str(c.mismatches),
                ",".join(map(str, c.mismatch_positions)) or "-",
                ",".join(map(str, c.ambiguous_positions)) or "-",
                c.pam, "yes" if c.pam_compatible else "no",
                c.matched_sequence or "-",
                c.guide,
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
            msg = (f"{len(scored):,} candidates — click a row for details; "
                   "click a column header to sort")
        self.query_one("#results-count", Static).update(msg)
        self._clear_detail()

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

    @on(DataTable.RowSelected, "#results-table")
    def show_candidate_detail(self, event: DataTable.RowSelected) -> None:
        row_index = event.cursor_row
        if 0 <= row_index < len(self._displayed_scored):
            self._show_candidate_detail(self._displayed_scored[row_index])

    def _clear_detail(self) -> None:
        try:
            log = self.query_one("#results-detail", RichLog)
            log.clear()
            log.write("(select a result row to inspect the real alignment)")
        except Exception:
            pass

    def _show_candidate_detail(self, sc) -> None:
        c = sc.candidate
        view = candidate_view(c, self.state.records)
        breakdown = score_breakdown(c)
        log = self.query_one("#results-detail", RichLog)
        log.clear()

        meta = Text()
        meta.append("Candidate ", style="bold")
        meta.append(f"{c.chromosome}:{c.start}-{c.end}({c.strand}) ",
                    style="bold cyan")
        meta.append(f"score={sc.score:.2f} type={c.match_type} ",
                    style="bold")
        meta.append(f"MM={c.mismatches} positions={list(c.mismatch_positions)} "
                    f"ambiguous={list(c.ambiguous_positions)}\n")
        meta.append(
            f"guide={view.guide}  matched={view.matched_sequence or 'n/a'}  "
            f"genomic={view.genomic_sequence or 'n/a'}\n"
        )
        meta.append(
            f"PAM={view.pam_sequence} (candidate strand; plus footprint "
            f"{view.pam_plus_sequence or 'n/a'})  PAM coords "
            f"{c.pam_start}..{c.pam_end}\n"
        )
        meta.append(
            "score parts: "
            f"mm_penalty={breakdown['mismatch_penalty']} "
            f"seed_penalty={breakdown['seed_penalty']} "
            f"pam_penalty={breakdown['pam_penalty']} "
            f"ambiguity_penalty={breakdown['ambiguity_penalty']} "
            "(experimental ranking score, not validated)"
        )
        log.write(meta)

        if not view.matched_sequence:
            log.write("Sequence context unavailable for this candidate.")
            return

        guide_line = Text("guide     ", style="dim")
        guide_line.append(view.guide, style="bold cyan")
        match_line = Text("candidate ", style="dim")
        marker_line = Text("          ", style="dim")
        for seg in alignment_segments(view):
            if seg["kind"] == "mismatch":
                match_line.append(str(seg["matched"]), style="bold red")
                marker_line.append("^", style="bold red")
            elif seg["kind"] == "ambiguous":
                match_line.append(str(seg["matched"]), style="bold yellow")
                marker_line.append("~", style="bold yellow")
            else:
                match_line.append(str(seg["matched"]), style="bold green")
                marker_line.append(" ")
        log.write(guide_line)
        log.write(match_line)
        log.write(marker_line)
        log.write("legend: ^ mismatch, ~ compatible IUPAC ambiguity, "
                  "green exact base match")

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
        try:
            max_mm_raw = self.query_one("#f-max-mm", Input).value.strip()
            max_mm = int(max_mm_raw) if max_mm_raw else None
            if max_mm is not None and max_mm < 0:
                raise ValueError
        except ValueError:
            self._log("#analyzer-log", "ERROR: max mismatches filter must be "
                                   "a non-negative integer.")
            return
        chrom = self.query_one("#f-chrom", Input).value.strip().lower()
        search = self.query_one("#f-search", Input).value.strip().lower()
        strand = self.query_one("#f-strand", Select).value
        pam_filter = self.query_one("#f-pam", Select).value
        type_filter = self.query_one("#f-type", Select).value

        def _matches(sc) -> bool:
            c = sc.candidate
            if sc.score < min_score:
                return False
            if max_mm is not None and c.mismatches > max_mm:
                return False
            if chrom and chrom not in c.chromosome.lower():
                return False
            if strand != "all" and c.strand != strand:
                return False
            if pam_filter == "yes" and not c.pam_compatible:
                return False
            if pam_filter == "no" and c.pam_compatible:
                return False
            if type_filter != "all" and c.match_type != type_filter:
                return False
            if search:
                haystack = " ".join([
                    c.chromosome, c.strand, c.pam, c.guide,
                    c.matched_sequence, c.genomic_sequence,
                    ",".join(map(str, c.mismatch_positions)),
                    ",".join(map(str, c.ambiguous_positions)),
                    str(c.start), str(c.end), str(c.pam_start), str(c.pam_end),
                ]).lower()
                if search not in haystack:
                    return False
            return True

        self._visible_scored = [sc for sc in result.scored if _matches(sc)]
        self._render_results()

    @on(Button.Pressed, "#filter-clear")
    def clear_filters(self) -> None:
        self.query_one("#f-min-score", Input).value = ""
        self.query_one("#f-max-mm", Input).value = ""
        self.query_one("#f-chrom", Input).value = ""
        self.query_one("#f-search", Input).value = ""
        self.query_one("#f-strand", Select).value = "all"
        self.query_one("#f-pam", Select).value = "all"
        self.query_one("#f-type", Select).value = "all"
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
            max_mm = self._max_mm()
            pams = self._pam()
            strands = self._strand_tuple()
            require_pam = self._require_pam()
        except AlignError as exc:
            self._log("#perf-log", f"ERROR: {exc}")
            return
        backend = self._backend("#bench-backend")
        if backend == "gpu" and not detect_gpu().available:
            self._log("#perf-log",
                      "ERROR: GPU unavailable — benchmark requires a "
                      "working CUDA backend.")
            return
        self._bench_worker(path, guide, backend, pams, strands, max_mm,
                           require_pam)

    @work(thread=True)
    def _bench_worker(self, path: str, guide: str, backend: str,
                      pam_patterns, strands, max_mismatches: int,
                      require_pam: bool) -> None:
        def progress(done: int, total: int) -> None:
            self.call_from_thread(self._set_progress, "#bench-progress",
                                  done, total)

        try:
            from .pipeline import run_benchmark as _rb
            result, analysis = _rb(
                path, guide, backend=backend,
                chunk_size=self.state.chunk_size,
                overlap=self.state.overlap,
                pam_patterns=pam_patterns,
                strands=strands,
                max_mismatches=max_mismatches,
                require_pam=require_pam,
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
        self._update_perf_last(analysis)
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
            table.add_row(
                r.backend,
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

    def _analysis_for_export(self):
        """Return the currently visible scored candidates as an analysis."""
        result = self.state.analysis
        if result is None:
            return None
        visible = list(self._visible_scored)
        filtered = len(visible) != len(result.scored)
        extra = dict(result.extra or {})
        extra["export_scope"] = "filtered_results" if filtered else "all_results"
        extra["exported_candidates"] = len(visible)
        return replace(
            result,
            candidates=[sc.candidate for sc in visible],
            scored=visible,
            score_summary=score_summary(visible),
            extra=extra,
        )

    @on(Button.Pressed, "#exp-csv")
    @on(Button.Pressed, "#exp-json")
    @on(Button.Pressed, "#exp-html")
    @on(Button.Pressed, "#exp-pdf")
    def export_quick_report(self, event: Button.Pressed) -> None:
        export = self._analysis_for_export()
        if export is None:
            self._log("#analyzer-log", "ERROR: run an analysis first.")
            self.query_one("#results-count", Static).update(
                "ERROR: run an analysis first.")
            return
        out = self._out_dir()
        try:
            if event.button.id == "exp-csv":
                p = write_csv(export, out / "geneware_results.csv")
                msg = f"CSV written: {p}"
            elif event.button.id == "exp-json":
                p = write_json(export, out / "geneware_results.json")
                msg = f"JSON written: {p}"
            elif event.button.id == "exp-html":
                p = write_html(export, out / "geneware_results.html")
                msg = f"HTML written: {p}"
            else:
                p = write_pdf(export, out / "geneware_results.pdf")
                msg = f"PDF written: {p}"
        except Exception as exc:  # noqa: BLE001
            self.query_one("#results-count", Static).update(
                f"ERROR: {type(exc).__name__}: {exc}")
            return
        self.query_one("#results-count", Static).update(msg)
        self._log("#home-log", msg)

    @on(Button.Pressed, "#rep-csv")
    @on(Button.Pressed, "#rep-json")
    @on(Button.Pressed, "#rep-html")
    @on(Button.Pressed, "#rep-pdf")
    @on(Button.Pressed, "#rep-bundle")
    def export_report(self, event: Button.Pressed) -> None:
        result = self._analysis_for_export()
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
            elif event.button.id == "rep-pdf":
                p = write_pdf(result, out / "geneware_report.pdf")
                msg = f"PDF written: {p}"
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
        self._refresh_systems()


def main() -> None:
    GeneWeaverApp().run()


if __name__ == "__main__":
    sys.exit(main())
