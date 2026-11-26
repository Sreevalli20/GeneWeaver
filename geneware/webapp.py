"""GeneWeaver — CRISPR Guide Analyzer (Streamlit web app).

A friendly graphical front-end over the real analysis engine. Every page
calls the actual Python functions — :mod:`geneware.fasta` for parsing,
:mod:`geneware.pipeline` for the CPU / Dask / GPU scan,
:mod:`geneware.alignment_view` for sequence-alignment views, and
:mod:`geneware.reports` for exports. Backend diagnostics (GPU probes,
Dask workers, DLL errors) live in an optional *System Details* section;
the main screens stay simple. No analysis logic is duplicated here and
no metrics are fabricated.

Run with::

    streamlit run geneware/webapp.py
    # or simply:
    python -m geneware            # launches this app in the browser
    python -m geneware --textual  # legacy Textual dashboard
"""

from __future__ import annotations

import html
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st

from geneware import __version__
from geneware.alignment_view import alignment_segments, candidate_view
from geneware.aligner import AlignError, validate_guide
from geneware.fasta import FastaParseError, compute_stats, parse_fasta
from geneware.pipeline import AnalysisError, run_benchmark
from geneware.reports import write_csv, write_html, write_json, write_pdf

EXAMPLE_FASTA = Path(__file__).resolve().parent.parent / "data" / "example.fasta"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "geneware_uploads"
ALLOWED_SUFFIXES = {".fasta", ".fa", ".fna"}
PAGES = ["Home", "Upload Genome", "Run Analysis", "Results",
         "Performance", "Reports"]

st.set_page_config(
    page_title="GeneWeaver — CRISPR Guide Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .gw-hero { text-align: center; padding: 2.2rem 1rem 1rem 1rem; }
    .gw-hero h1 { font-size: 2.4rem; color: #1f4e79; margin-bottom: .2rem; }
    .gw-hero p  { color: #51606f; font-size: 1.05rem; margin-top: 0; }
    .gw-step { border: 1px solid #d8e0ea; border-radius: 12px;
               padding: 1rem 1.1rem; background: #ffffff; height: 100%; }
    .gw-step .num { display:inline-block; width: 1.7rem; height: 1.7rem;
                    border-radius: 50%; background: #1f4e79; color: #fff;
                    text-align: center; line-height: 1.7rem; font-weight: 600;
                    margin-bottom: .5rem; }
    .gw-card { border: 1px solid #d8e0ea; border-radius: 12px;
               padding: 1rem 1.2rem; background: #ffffff;
               margin-bottom: .8rem; }
    .gw-muted { color: #667085; font-size: .88rem; }
    .aln { font-family: Consolas, monospace; font-size: .92rem;
           background: #f7f9fc; border: 1px solid #d8e0ea;
           border-radius: 8px; padding: .7rem .9rem; overflow-x: auto; }
    .aln .lbl { color: #51606f; display: inline-block; width: 8.5rem; }
    .aln .b-match { color: #067647; font-weight: 600; }
    .aln .b-mismatch { color: #b42318; font-weight: 700;
                       text-decoration: underline; }
    .aln .b-ambiguous { color: #b54708; font-weight: 700; }
    .aln .marker { color: #b42318; letter-spacing: 0; }
    .strand-pill { display:inline-block; padding: .1rem .55rem;
                   border-radius: 999px; font-size: .8rem;
                   font-weight: 600; }
    .strand-fwd { background: #dcfae6; color: #067647; }
    .strand-rev { background: #fee4e2; color: #b42318; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _state() -> dict:
    return st.session_state


def genome_loaded() -> bool:
    return bool(_state().get("fasta_path")
                and _state().get("records") is not None)


def load_genome(path: Path, display_name: str) -> str | None:
    """Parse a FASTA file into session state. Returns an error or None."""
    try:
        records = parse_fasta(path)
    except FastaParseError as exc:
        _state()["load_error"] = str(exc)
        return str(exc)
    stats = compute_stats(records)
    _state()["fasta_path"] = str(path)
    _state()["fasta_name"] = display_name
    _state()["records"] = records
    _state()["seq_stats"] = stats.to_dict()
    _state()["load_error"] = None
    _state()["analysis"] = None
    _state()["benchmark"] = None
    _state()["analyze_error"] = None
    _state()["uploaded_key"] = None
    return None


def gpu_status():
    """Probe GPU availability once per session (only for System Details)."""
    if "gpu_status" not in _state():
        try:
            from geneware.gpu import detect_gpu
            _state()["gpu_status"] = detect_gpu()
        except Exception as exc:  # probe itself must never crash the UI
            _state()["gpu_status"] = None
            _state()["gpu_error"] = f"{type(exc).__name__}: {exc}"
    return _state()["gpu_status"]


def dask_available() -> bool:
    try:
        import distributed  # noqa: F401
        return True
    except Exception:
        return False


def _go(page: str) -> None:
    """Navigate to another page on the next script run."""
    _state()["page_target"] = page
    st.rerun()


# ---------------------------------------------------------------------------
# Shared UI bits
# ---------------------------------------------------------------------------

def _require_genome(page_hint: str = "Upload Genome") -> bool:
    if genome_loaded():
        return True
    st.info("Please load a genome first.")
    if st.button("Upload a genome", type="primary"):
        _go(page_hint)
    return False


def _strand_label(strand: str) -> str:
    return "→ Forward (+)" if strand == "+" else "← Reverse (−)"


def _strand_pill(strand: str) -> str:
    css = "strand-fwd" if strand == "+" else "strand-rev"
    arrow = "→" if strand == "+" else "←"
    return (f'<span class="strand-pill {css}">{arrow} '
            f'{"Forward" if strand == "+" else "Reverse"}</span>')


def _alignment_html(view) -> str:
    """Guide-vs-candidate alignment with mismatches highlighted."""
    seg = alignment_segments(view)
    if not seg:
        return ("<p class='gw-muted'>Sequence context unavailable for "
                "this candidate.</p>")
    guide_html = "".join(
        f'<span class="b-{s["kind"]}">{html.escape(s["guide"])}</span>'
        for s in seg)
    match_html = "".join(
        f'<span class="b-{s["kind"]}">{html.escape(s["matched"])}</span>'
        for s in seg)
    marker = "".join(
        "^" if s["kind"] == "mismatch"
        else "~" if s["kind"] == "ambiguous" else " "
        for s in seg)
    return (
        "<div class='aln'>"
        f"<div><span class='lbl'>Guide (5'→3')</span>"
        f"<code>{guide_html}</code></div>"
        "<div><span class='lbl'>Candidate match</span>"
        f"<code>{match_html}</code></div>"
        f"<div class='marker'><span class='lbl'></span>"
        f"<code>{html.escape(marker)}</code></div>"
        f"<div><span class='lbl'>PAM</span>"
        f"<code>{html.escape(view.pam_sequence)}</code> "
        f"<span class='gw-muted'>(candidate strand; plus-strand footprint "
        f"{html.escape(view.pam_plus_sequence) or 'n/a'})</span></div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home() -> None:
    st.markdown(
        "<div class='gw-hero'>"
        "<h1>🧬 GeneWeaver</h1>"
        "<p><b>CRISPR Guide Analyzer</b> — explore DNA sequences and "
        "identify potential CRISPR guide matches.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center;color:#51606f'>"
        "Analyze a DNA sequence and explore potential CRISPR guide "
        "matches in a few simple steps.</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "<div class='gw-step'><span class='num'>1</span><br>"
            "<b>Upload a DNA FASTA file</b><br>"
            "<span class='gw-muted'>Your genome or a small example.</span>"
            "</div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            "<div class='gw-step'><span class='num'>2</span><br>"
            "<b>Enter your guide &amp; settings</b><br>"
            "<span class='gw-muted'>Guide sequence, PAM pattern, "
            "mismatch limit.</span></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(
            "<div class='gw-step'><span class='num'>3</span><br>"
            "<b>Run the analysis</b><br>"
            "<span class='gw-muted'>Browse ranked matches and download "
            "a report.</span></div>", unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("⬆️  Upload Genome", type="primary",
                     use_container_width=True):
            _go("Upload Genome")
    with b2:
        if st.button("🧪  Try Example Genome", use_container_width=True):
            err = load_genome(EXAMPLE_FASTA, EXAMPLE_FASTA.name)
            if err is None:
                st.success("Example genome loaded — you can start an "
                           "analysis right away.")
                _go("Run Analysis")
            else:
                st.error(f"Could not load the example genome: {err}")

    st.divider()

    if genome_loaded():
        stats = _state()["seq_stats"]
        st.markdown("<div class='gw-card'><b>Current genome</b><br>"
                    f"<code>{html.escape(_state()['fasta_name'])}</code> · "
                    f"{stats['record_count']:,} sequences · "
                    f"{stats['total_bases']:,} bases · "
                    f"GC {stats['gc_percent']}%</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(
            "<div class='gw-card gw-muted'>No genome loaded yet — upload "
            "a FASTA file or try the bundled example to get started.</div>",
            unsafe_allow_html=True)

    st.markdown("<div class='gw-card'><b>Last analysis</b><br>",
                unsafe_allow_html=True)
    analysis = _state().get("analysis")
    if analysis is None:
        st.markdown(
            "<span class='gw-muted'>No analyses yet. Once you run one, "
            "a summary will appear here.</span></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<b>{len(analysis.scored):,}</b> potential matches found · "
            f"guide <code>{html.escape(analysis.guide)}</code> · "
            f"{analysis.elapsed_seconds:.2f} s · "
            f"{html.escape(analysis.processing_method)}<br>"
            f"<span class='gw-muted'>File "
            f"<code>{html.escape(Path(analysis.fasta_path).name)}</code> · "
            f"{datetime.now():%Y-%m-%d}</span></div>",
            unsafe_allow_html=True)
        if st.button("View results"):
            _go("Results")

    with st.expander("How scoring works"):
        st.markdown(
            "GeneWeaver ranks candidate matches with an experimental "
            "heuristic score (0–100). It rewards fewer mismatches and "
            "fewer mismatches near the PAM-proximal end of the guide. "
            "**This is a computational ranking only — it is not a "
            "validated biological prediction and does not prove that a "
            "CRISPR edit will occur.**"
        )


def page_upload() -> None:
    st.title("Upload Genome")
    st.caption("Choose a DNA FASTA file (.fasta, .fa, or .fna) — or try "
               "the bundled example.")

    uploaded = st.file_uploader(
        "DNA FASTA file", type=["fasta", "fa", "fna"],
        accept_multiple_files=False,
        help="Upload a .fasta, .fa, or .fna file containing DNA sequences",
    )

    # Only (re)parse when a *new* file is selected.
    if uploaded is not None:
        key = (uploaded.name, uploaded.size)
        if _state().get("uploaded_key") != key:
            suffix = Path(uploaded.name).suffix.lower()
            if suffix not in ALLOWED_SUFFIXES:
                _state()["load_error"] = (
                    f"Unsupported file type '{suffix}'. Please upload a "
                    ".fasta, .fa, or .fna file.")
                _state()["uploaded_key"] = key
            else:
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                dest = UPLOAD_DIR / Path(uploaded.name).name
                dest.write_bytes(uploaded.getbuffer())
                err = load_genome(dest, uploaded.name)
                _state()["uploaded_key"] = key
                if err is not None:
                    st.error(f"That file could not be parsed as FASTA: {err}")

    col_ex = st.columns([1, 3])
    with col_ex[0]:
        if st.button("🧪 Try example genome"):
            err = load_genome(EXAMPLE_FASTA, EXAMPLE_FASTA.name)
            if err is None:
                st.success("Example genome loaded.")
                _go("Run Analysis")
            else:
                st.error(f"Could not load the example genome: {err}")
    with col_ex[1]:
        st.caption("The example is a tiny synthetic genome — great for a "
                   "first look around.")

    err = _state().get("load_error")
    if err and not genome_loaded():
        st.warning(err)

    if not genome_loaded():
        return

    stats = _state()["seq_stats"]
    st.success("**Genome loaded successfully**")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sequences", f"{stats['record_count']:,}")
    c2.metric("Total bases", f"{stats['total_bases']:,}")
    c3.metric("GC content", f"{stats['gc_percent']}%")
    c4.metric("File", _state()["fasta_name"])

    st.subheader("Sequences in this genome")
    rows = [{
        "ID": rec.record_id,
        "Description": rec.description,
        "Length (bp)": f"{rec.length:,}",
        "GC %": pr["gc_percent"],
    } for rec, pr in zip(_state()["records"], stats["per_record"],
                         strict=True)]
    st.dataframe(rows, width="stretch", hide_index=True)

    if _state()["records"]:
        rec = _state()["records"][0]
        preview = rec.sequence[:200]
        st.subheader(f"Preview — {rec.record_id}")
        st.code(preview + ("…" if rec.length > 200 else ""),
                language=None)

    if st.button("Continue to analysis →", type="primary"):
        _go("Run Analysis")


def _backend_options() -> list[tuple[str, str]]:
    """(value, friendly label) pairs; GPU only when genuinely detected."""
    opts = [("auto", "Automatic (recommended)"),
            ("cpu", "CPU (reliable default)")]
    if dask_available():
        opts.append(("dask", "Parallel CPU (Dask)"))
    gpu = gpu_status()
    if gpu is not None and gpu.available:
        opts.append(("gpu", "GPU (CUDA)"))
    return opts


def page_analysis() -> None:
    st.title("Run Analysis")
    st.caption("Find places in your genome that resemble your guide "
               "sequence, allowing for a configurable number of "
               "mismatches.")

    if not _require_genome():
        return

    stats = _state()["seq_stats"]
    st.caption(f"Genome: **{_state()['fasta_name']}** — "
               f"{stats['record_count']:,} sequences, "
               f"{stats['total_bases']:,} bases.")

    if _state().get("analyzing"):
        st.info("An analysis is already running — please wait for it "
                "to finish.")
        return

    opts = _backend_options()
    labels = [label for _, label in opts]
    with st.form("analysis_form"):
        guide = st.text_input(
            "Guide sequence",
            value="ACGTACGTACGTACGTACGTACGT",
            help="Your guide RNA spacer as DNA, 5'→3' (10–30 bases). "
                 "Example: ACGTACGTACGTACGTACGTACGT",
        )
        pam_raw = st.text_input(
            "PAM pattern",
            value="NGG",
            help="The short motif required next to the match. "
                 "NGG is the classic Cas9 PAM; N means any base. "
                 "You can list several separated by commas, e.g. NGG,NAG.",
        )
        col1, col2 = st.columns(2)
        with col1:
            max_mm = st.slider(
                "Maximum mismatches",
                min_value=0, max_value=10, value=0,
                help="How many bases in the guide are allowed to differ "
                     "from the genome. 0 = perfect matches only; "
                     "higher values find more, weaker candidates.",
            )
        with col2:
            strands = st.multiselect(
                "Search strands",
                ["+", "-"], default=["+", "-"],
                format_func=lambda s: ("Forward (+)" if s == "+"
                                       else "Reverse (−)"),
                help="Search the forward strand, the reverse strand, "
                     "or both.",
            )
        backend_label = st.selectbox(
            "Processing mode",
            labels,
            index=0,
            help="Automatic picks the best genuinely available engine. "
                 "CPU is the reliable default; GPU is only offered when "
                 "your machine really has one.",
        )
        run = st.form_submit_button("▶  Run Analysis",
                                    type="primary", use_container_width=True)

    if not run:
        return

    backend = opts[labels.index(backend_label)][0]

    # Validate inputs before touching the engine.
    try:
        guide_norm = validate_guide(guide)
        pams = tuple(s.strip() for s in pam_raw.split(",") if s.strip())
        if not pams:
            raise AlignError("at least one PAM pattern is required")
        if not strands:
            raise AlignError("select at least one strand")
    except AlignError as exc:
        st.error(f"Please check your inputs: {exc}")
        return

    _state()["analyzing"] = True
    progress = st.progress(0.0, text="Starting analysis…")
    status = st.empty()

    def _progress(done: int, total: int) -> None:
        if total:
            progress.progress(
                min(done / total, 1.0),
                text=f"Scanning genome — chunk {done} of {total}")
        else:
            progress.progress(0.5, text="Scanning genome…")

    status.info(f"Running analysis on **{_state()['fasta_name']}** "
                f"with guide `{guide_norm}`…")
    try:
        benchmark, analysis = run_benchmark(
            _state()["fasta_path"], guide_norm,
            backend=backend,
            pam_patterns=pams,
            strands=tuple(strands),
            max_mismatches=int(max_mm),
            progress=_progress,
        )
    except (FastaParseError, AlignError, AnalysisError, MemoryError) as exc:
        progress.empty()
        st.error(f"Analysis failed: {exc}")
        return
    except Exception as exc:  # surface real backend errors honestly
        progress.empty()
        st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc(limit=5))
        return
    finally:
        _state()["analyzing"] = False

    progress.progress(1.0, text="Complete")
    _state()["analysis"] = analysis
    _state()["benchmark"] = benchmark
    _state()["analyze_error"] = None

    st.success(
        f"**Analysis complete** — found **{len(analysis.scored):,}** "
        f"potential matches in {analysis.elapsed_seconds:.2f} seconds "
        f"using {analysis.processing_method}.")
    if st.button("View results →", type="primary"):
        _go("Results")


def _results_rows(analysis) -> list[dict]:
    rows = []
    for rank, sc in enumerate(analysis.scored, start=1):
        d = sc.to_dict()
        rows.append({
            "Rank": rank,
            "Score": round(d["score"], 2),
            "Type": "Exact" if d["match_type"] == "exact" else "Near match",
            "Chromosome": d["chromosome"],
            "Position": f"{d['start']}–{d['end']}",
            "Strand": d["strand"],
            "PAM": d["pam"],
            "Mismatches": d["mismatches"],
            "Matched sequence": d["matched_sequence"],
        })
    return rows


def page_results() -> None:
    st.title("Results")
    analysis = _state().get("analysis")
    if analysis is None:
        st.info("No analysis yet — run one first and your matches will "
                "appear here.")
        if st.button("Run an analysis"):
            _go("Run Analysis")
        return

    n_exact = sum(1 for s in analysis.scored
                  if s.candidate.match_type == "exact")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Potential matches", f"{len(analysis.scored):,}")
    c2.metric("Exact matches", f"{n_exact:,}")
    c3.metric("Sequences searched",
              f"{analysis.sequence_stats.get('record_count', 0):,}")
    c4.metric("Analysis time",
              f"{analysis.elapsed_seconds:.2f} s")

    strand_txt = ("both strands" if len(analysis.strands) == 2
                  else _strand_label(analysis.strands[0]))
    st.caption(
        f"Guide `{analysis.guide}` · PAM "
        f"{', '.join(analysis.pam_patterns)} · up to "
        f"{analysis.max_mismatches} mismatches · {strand_txt} · "
        f"genome `{Path(analysis.fasta_path).name}`")

    if not analysis.scored:
        st.warning(
            "**No matches found** with these settings. This can happen "
            "when the guide is very different from the genome, the PAM "
            "pattern is too strict, or the mismatch limit is too low. "
            "Try a different guide, relax the PAM (e.g. allow PAM "
            "mismatches), or increase the mismatch limit.")
        if st.button("← Adjust settings and re-run"):
            _go("Run Analysis")
        return

    rows = _results_rows(analysis)
    chromosomes = sorted({r["Chromosome"] for r in rows}, key=str)

    f1, f2, f3, f4 = st.columns([2, 1, 1, 1])
    search = f1.text_input("Search", placeholder="chromosome, PAM, "
                           "sequence…")
    strand_f = f2.selectbox("Strand", ["all", "+", "-"],
                            format_func=lambda s:
                            "All strands" if s == "all" else
                            ("Forward (+)" if s == "+" else "Reverse (−)"))
    chr_f = f3.selectbox("Chromosome", ["all"] + chromosomes,
                         format_func=lambda s:
                         "All chromosomes" if s == "all" else str(s))
    max_mm_f = f4.slider("Max mismatches", 0, 10, 10)

    filtered = rows
    if search.strip():
        q = search.strip().lower()
        filtered = [r for r in filtered
                    if q in str(r["Chromosome"]).lower()
                    or q in str(r["Matched sequence"]).lower()
                    or q in str(r["PAM"]).lower()
                    or q in str(r["Position"])]
    if strand_f != "all":
        filtered = [r for r in filtered if r["Strand"] == strand_f]
    if chr_f != "all":
        filtered = [r for r in filtered if r["Chromosome"] == chr_f]
    filtered = [r for r in filtered if r["Mismatches"] <= max_mm_f]

    st.caption(f"Showing **{len(filtered):,}** of **{len(rows):,}** "
               "matches")
    st.dataframe(
        filtered, width="stretch", hide_index=True, height=420,
        column_config={
            "Score": st.column_config.NumberColumn(
                "Score", format="%.1f",
                help="Experimental ranking score (0–100) — higher is "
                     "better, but it is a heuristic, not a validated "
                     "biological prediction."),
            "Matched sequence": st.column_config.TextColumn(
                "Matched sequence", width="large"),
        },
    )

    # ---- Candidate detail view -------------------------------------
    st.subheader("Inspect a match")
    if not filtered:
        st.caption("No matches match the current filters — adjust them "
                   "above.")
    else:
        labels = [
            f"#{r['Rank']}  {r['Chromosome']}:{r['Position']}  "
            f"{r['Strand']}  · score {r['Score']}"
            for r in filtered[:500]
        ]
        pick = st.selectbox("Choose a candidate", labels,
                            key="cand_pick")
        chosen = filtered[labels.index(pick)]
        sc = analysis.scored[chosen["Rank"] - 1]
        view = candidate_view(sc.candidate, analysis.records)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score (experimental)", f"{sc.score:.1f}")
        m2.metric("Mismatches", sc.candidate.mismatches)
        m3.metric("Strand",
                  "Forward (+)" if sc.candidate.strand == "+"
                  else "Reverse (−)")
        m4.metric("Match type", "Exact" if sc.candidate.match_type
                  == "exact" else "Near match")

        st.markdown(
            f"**Location:** `{sc.candidate.chromosome}` "
            f"positions {sc.candidate.start}–{sc.candidate.end} "
            f"(0-based) · PAM `{sc.candidate.pam}` "
            f"(positions {sc.candidate.pam_start}–{sc.candidate.pam_end})")
        st.markdown(_alignment_html(view), unsafe_allow_html=True)
        st.caption(
            "**How to read this:** green = guide base matches the "
            "genome; red underlined = mismatch; orange = ambiguous base. "
            "The score is an experimental ranking heuristic — it does "
            "not prove biological effectiveness.")

    st.divider()
    st.markdown(
        "<div class='gw-card gw-muted'><b>About these results.</b> "
        "GeneWeaver finds sequence similarity between your guide and "
        "the genome. The score is an internal experimental ranking "
        "heuristic — higher scores mean fewer mismatches, especially "
        "near the PAM-proximal end. It does <b>not</b> predict whether "
        "a real CRISPR experiment will cut or edit at that site.</div>",
        unsafe_allow_html=True)
    if st.button("Download reports for these results →"):
        _go("Reports")


def page_performance() -> None:
    st.title("Performance")
    bench = _state().get("benchmark")
    analysis = _state().get("analysis")

    if bench is None or analysis is None:
        st.info("No analysis has been run yet — timings will appear "
                "here after your first analysis.")
        return

    d = bench.to_dict()
    throughput = d.get("throughput_bases_per_sec")
    method = analysis.processing_method
    friendly = {
        "cpu": "CPU", "gpu": "GPU (CUDA)",
    }.get(analysis.backend, analysis.backend)
    if "dask" in method:
        friendly = "Parallel CPU (Dask)"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analysis duration", f"{d['total_seconds']:.2f} s")
    c2.metric("Sequences processed",
              f"{d.get('records_parsed', 0):,}")
    c3.metric("Candidates found", f"{d.get('candidates_found', 0):,}")
    c4.metric("Throughput",
              "n/a" if throughput is None else
              f"{throughput:,.0f} bases/s")

    c1, c2 = st.columns(2)
    c1.metric("Execution mode", friendly)
    mem = d.get("memory_peak_bytes")
    c2.metric("Peak memory",
              "n/a" if mem is None else f"{mem / (1024 * 1024):.1f} MiB")

    st.caption(
        f"Genome `{Path(d['fasta_path']).name}` · guide "
        f"`{d['guide']}` · {d.get('bases_processed', 0):,} bases scanned "
        f"in {d.get('chunks_scanned', 0):,} chunks.")

    with st.expander("System Details"):
        st.markdown("**Hardware & runtime probes** (for debugging — "
                    "none of this affects your results)")
        gpu = gpu_status()
        if gpu is None:
            st.write("GPU probe failed:",
                     _state().get("gpu_error", "unknown error"))
        elif gpu.available:
            st.success(f"GPU available via {gpu.backend}")
            st.json(gpu.to_dict())
        else:
            st.write("**GPU:** unavailable —", gpu.reason)
        st.write("**Dask distributed:**",
                 "available" if dask_available() else "not installed")
        import platform
        import sys
        st.write("**Python:**", sys.version.split()[0],
                 "·", platform.platform())
        if d.get("notes"):
            st.write("**Benchmark notes:**", "; ".join(d["notes"]))
        extra = analysis.extra or {}
        if extra:
            st.json({k: v for k, v in extra.items()
                     if k != "gpu"})


def page_reports() -> None:
    st.title("Reports")
    analysis = _state().get("analysis")
    if analysis is None:
        st.info("No analysis to export yet — run one first.")
        if st.button("Run an analysis"):
            _go("Run Analysis")
        return

    st.caption(f"Reports for guide `{analysis.guide}` on "
               f"`{Path(analysis.fasta_path).name}` "
               f"({len(analysis.scored):,} matches, "
               f"{analysis.elapsed_seconds:.2f} s).")

    out_dir = Path(tempfile.gettempdir()) / "geneware_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"geneware_{Path(analysis.fasta_path).stem}_"
            f"{datetime.now():%Y%m%d_%H%M%S}")

    try:
        csv_path = write_csv(analysis, out_dir / f"{stem}.csv")
        json_path = write_json(analysis, out_dir / f"{stem}.json")
        html_path = write_html(analysis, out_dir / f"{stem}.html")
        pdf_path = write_pdf(analysis, out_dir / f"{stem}.pdf")
    except Exception as exc:
        st.error(f"Report generation failed: {exc}")
        return

    st.success("**Report ready** — your results are packaged below.")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("⬇️ CSV spreadsheet", csv_path.read_bytes(),
                       file_name=csv_path.name, mime="text/csv",
                       use_container_width=True)
    c2.download_button("⬇️ JSON data", json_path.read_bytes(),
                       file_name=json_path.name,
                       mime="application/json",
                       use_container_width=True)
    c3.download_button("⬇️ HTML report", html_path.read_bytes(),
                       file_name=html_path.name, mime="text/html",
                       use_container_width=True)
    c4.download_button("⬇️ PDF report", pdf_path.read_bytes(),
                       file_name=pdf_path.name, mime="application/pdf",
                       use_container_width=True)
    st.caption("Every report contains your guide, PAM settings, genome "
               "info, analysis date, timings, and the full ranked "
               "candidate list — all real analysis output.")


PAGES_FUNCS = {
    "Home": page_home,
    "Upload Genome": page_upload,
    "Run Analysis": page_analysis,
    "Results": page_results,
    "Performance": page_performance,
    "Reports": page_reports,
}


def main() -> None:
    # Programmatic navigation: _go() stores a target; apply it to the
    # sidebar radio *before* the widget is instantiated this run.
    target = _state().pop("page_target", None)
    if target in PAGES:
        _state()["nav"] = target
    with st.sidebar:
        st.markdown("# 🧬 GeneWeaver")
        st.caption("CRISPR Guide Analyzer")
        page = st.radio("Navigate", PAGES, key="nav",
                        label_visibility="collapsed")
        st.divider()
        if genome_loaded():
            st.caption(f"Genome: `{_state()['fasta_name']}`")
        else:
            st.caption("No genome loaded")
        analysis = _state().get("analysis")
        if analysis is not None:
            st.caption(f"Last run: {len(analysis.scored):,} matches")

    PAGES_FUNCS[page]()


if __name__ == "__main__":
    main()
