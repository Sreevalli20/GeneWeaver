"""GeneWeaver browser dashboard (Streamlit).

A thin graphical front-end over the real analysis engine. Every page calls
the actual Python functions — :mod:`geneware.fasta` for parsing/stats,
:mod:`geneware.pipeline` (``run_benchmark``/``run_analysis``) for the CPU /
Dask / GPU scan, :mod:`geneware.reports` for exports, and
:mod:`geneware.gpu` for honest GPU availability. No analysis logic is
duplicated here and no metrics are fabricated.

Run with::

    streamlit run geneware/webapp.py
    # or simply:
    python -m geneware          # launches this app in the browser
    python -m geneware --textual  # legacy Textual dashboard
"""

from __future__ import annotations

import tempfile
import traceback
from pathlib import Path

import streamlit as st

from geneware import __version__
from geneware.aligner import AlignError, validate_guide
from geneware.fasta import FastaParseError, compute_stats, iter_fasta, parse_fasta
from geneware.pipeline import AnalysisError, run_benchmark
from geneware.reports import write_csv, write_html, write_json

EXAMPLE_FASTA = Path(__file__).resolve().parent.parent / "data" / "example.fasta"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "geneware_uploads"
PAGES = [
    "Home Dashboard",
    "Genome Upload",
    "CRISPR Analysis",
    "Analysis Results",
    "Performance",
    "Reports",
]

st.set_page_config(
    page_title="GeneWeaver",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _state() -> dict:
    return st.session_state


def genome_loaded() -> bool:
    return bool(_state().get("fasta_path") and _state().get("records") is not None)


def load_genome(path: Path, display_name: str) -> str | None:
    """Parse a FASTA file into session state. Returns an error string or None."""
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
    return None


def gpu_status():
    """Probe GPU availability once per session (real detect_gpu result)."""
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


# ---------------------------------------------------------------------------
# Shared UI bits
# ---------------------------------------------------------------------------

def _card(title: str, body: str = "") -> None:
    st.markdown(
        f"<div style='border:1px solid #d8e0ea;border-radius:10px;"
        f"padding:1rem 1.2rem;margin-bottom:.8rem;background:#ffffff'>"
        f"<div style='font-weight:600;color:#1f4e79;font-size:.95rem'>{title}</div>"
        f"<div style='color:#33404f;font-size:.9rem;margin-top:.3rem'>{body}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _require_genome() -> bool:
    if genome_loaded():
        return True
    st.info("No genome loaded yet. Go to **Genome Upload** to load a FASTA file.")
    return False


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_home() -> None:
    st.title("GeneWeaver")
    st.caption(
        "GPU/Dask-accelerated CRISPR protospacer & PAM alignment engine "
        f"(v{__version__}) — experimental ranking scores, not validated "
        "biological predictions."
    )

    st.subheader("Genome")
    if genome_loaded():
        stats = _state()["seq_stats"]
        cols = st.columns(4)
        cols[0].metric("FASTA file", _state()["fasta_name"])
        cols[1].metric("Sequences", f"{stats['record_count']:,}")
        cols[2].metric("Total bases", f"{stats['total_bases']:,}")
        cols[3].metric("GC %", f"{stats['gc_percent']}%")
    else:
        st.warning("**Genome: not loaded**")
        if st.button("Upload a FASTA file", type="primary"):
            _state()["nav"] = "Genome Upload"
            st.rerun()

    st.subheader("Last analysis")
    analysis = _state().get("analysis")
    if analysis is None:
        st.info("**Last analysis: none.** Run an analysis on the "
                "**CRISPR Analysis** page.")
    else:
        cols = st.columns(4)
        cols[0].metric("Status", "completed")
        cols[1].metric("Candidates", f"{len(analysis.scored):,}")
        cols[2].metric("Backend", analysis.processing_method)
        cols[3].metric("Elapsed", f"{analysis.elapsed_seconds:.3f} s")
        st.caption(
            f"Guide `{analysis.guide}` · PAM {', '.join(analysis.pam_patterns)} · "
            f"max mismatches {analysis.max_mismatches} · file "
            f"`{Path(analysis.fasta_path).name}`"
        )

    st.subheader("Backend availability")
    cols = st.columns(3)
    cols[0].metric("CPU engine", "available")
    gpu = gpu_status()
    if gpu is not None and gpu.available:
        cols[1].metric("GPU", gpu.backend or "available")
    else:
        reason = (gpu.reason if gpu is not None else
                  _state().get("gpu_error", "probe failed"))
        cols[1].metric("GPU", "unavailable")
        st.caption(f"GPU unavailable: {reason}")
    cols[2].metric("Dask", "available" if dask_available() else "unavailable")


def page_upload() -> None:
    st.title("Genome Upload")
    st.caption("Upload a `.fasta`/`.fa` file or load the bundled example. "
               "Parsing uses the real GeneWeaver FASTA parser.")

    col_a, col_b = st.columns(2)
    with col_a:
        uploaded = st.file_uploader(
            "Upload FASTA", type=["fasta", "fa", "txt"],
            help="Your own FASTA file (.fasta or .fa)",
        )
        if uploaded is not None:
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dest = UPLOAD_DIR / Path(uploaded.name).name
            dest.write_bytes(uploaded.getbuffer())
            err = load_genome(dest, uploaded.name)
            if err is None:
                st.success(f"Loaded **{uploaded.name}**: "
                           f"{len(_state()['records']):,} sequences, "
                           f"{_state()['seq_stats']['total_bases']:,} bases")
                st.rerun()
            else:
                st.error(f"FASTA parse error: {err}")
    with col_b:
        if st.button("Load example FASTA", width="stretch"):
            err = load_genome(EXAMPLE_FASTA, EXAMPLE_FASTA.name)
            if err is None:
                st.success(f"Loaded **{EXAMPLE_FASTA.name}**")
                st.rerun()
            else:
                st.error(f"FASTA parse error: {err}")

    err = _state().get("load_error")
    if err:
        st.error(f"Last load failed: {err}")

    if not genome_loaded():
        return

    stats = _state()["seq_stats"]
    cols = st.columns(4)
    cols[0].metric("Sequences", f"{stats['record_count']:,}")
    cols[1].metric("Total bases", f"{stats['total_bases']:,}")
    cols[2].metric("GC %", f"{stats['gc_percent']}%")
    cols[3].metric("Ambiguous bases", f"{stats['ambiguous_count']:,}")

    st.subheader("Sequences")
    rows = []
    for rec, pr in zip(_state()["records"], stats["per_record"], strict=True):
        rows.append({
            "ID": rec.record_id,
            "Description": rec.description,
            "Length (bp)": rec.length,
            "GC %": pr["gc_percent"],
        })
    st.dataframe(rows, width="stretch", hide_index=True)


def _backend_options() -> list[str]:
    opts = ["cpu", "dask", "auto"]
    gpu = gpu_status()
    if gpu is not None and gpu.available:
        opts.append("gpu")
    return opts


def page_analysis() -> None:
    st.title("CRISPR Analysis")
    st.caption("Runs the real alignment engine (FASTA parse → chunked "
               "scan → experimental scoring). CPU is the reliable default.")

    if not _require_genome():
        return

    gpu = gpu_status()
    gpu_ok = gpu is not None and gpu.available

    with st.form("analysis_form"):
        guide = st.text_input(
            "Guide DNA sequence (5'→3')",
            value="ACGTACGTACGTACGTACGTACGTACGTAC",
            help="10–30 IUPAC DNA bases. U is normalised to T.",
        )
        pam_raw = st.text_input(
            "PAM patterns (comma-separated IUPAC)",
            value="NGG",
            help="e.g. NGG or NGG,NAG — all patterns must have equal length",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            max_mm = st.number_input("Max mismatches", min_value=0,
                                     max_value=30, value=0, step=1)
        with col2:
            strands = st.multiselect("Strands", ["+", "-"], default=["+", "-"],
                                     help="Search plus, minus, or both strands")
        with col3:
            require_pam = st.checkbox("Require PAM", value=True,
                                      help="Keep only PAM-compatible candidates")
        backend = st.radio(
            "Backend",
            _backend_options(),
            index=0,
            horizontal=True,
            help=("cpu: sequential engine · dask: parallel CPU · "
                  "auto: best available · gpu: CUDA (only when probe passes)"),
        )
        if not gpu_ok:
            reason = (gpu.reason if gpu is not None else
                      _state().get("gpu_error", "GPU probe failed"))
            st.caption(f"GPU backend unavailable on this machine: {reason}. "
                       "CPU is used by default — nothing is faked.")
        run = st.form_submit_button("Analyze", type="primary")

    if not run:
        return

    # Validate inputs before touching the engine.
    try:
        guide_norm = validate_guide(guide)
        pams = tuple(s.strip() for s in pam_raw.split(",") if s.strip())
        if not pams:
            raise AlignError("at least one PAM pattern is required")
        if not strands:
            raise AlignError("select at least one strand")
    except AlignError as exc:
        _state()["analyze_error"] = str(exc)
        st.error(f"Invalid input: {exc}")
        return

    fasta_path = _state()["fasta_path"]
    progress = st.progress(0.0, text="Starting analysis…")
    status = st.empty()

    def _progress(done: int, total: int) -> None:
        if total:
            progress.progress(min(done / total, 1.0),
                              text=f"Scanning chunks {done}/{total}…")
        else:
            progress.progress(1.0, text="Scanning…")

    status.text(f"Running {backend} analysis on "
                f"{_state()['fasta_name']}…")
    try:
        benchmark, analysis = run_benchmark(
            fasta_path, guide_norm,
            backend=backend,
            pam_patterns=pams,
            strands=tuple(strands),
            max_mismatches=int(max_mm),
            require_pam=require_pam,
            progress=_progress,
        )
    except (FastaParseError, AlignError, AnalysisError, MemoryError) as exc:
        progress.empty()
        _state()["analyze_error"] = str(exc)
        st.error(f"Analysis failed: {exc}")
        return
    except Exception as exc:  # surface real backend errors honestly
        progress.empty()
        _state()["analyze_error"] = f"{type(exc).__name__}: {exc}"
        st.error(f"Analysis failed: {type(exc).__name__}: {exc}")
        st.code(traceback.format_exc(limit=3))
        return

    progress.progress(1.0, text="Complete")
    _state()["analysis"] = analysis
    _state()["benchmark"] = benchmark
    _state()["analyze_error"] = None
    st.success(
        f"Analysis complete — **{len(analysis.scored):,} candidates** found "
        f"({analysis.stats.candidates_found:,} matches before export) in "
        f"{analysis.elapsed_seconds:.3f} s on the **{analysis.backend}** backend."
    )
    st.info("View ranked results on the **Analysis Results** page and "
            "timings on **Performance**.")


def _results_rows(analysis) -> list[dict]:
    rows = []
    for rank, sc in enumerate(analysis.scored, start=1):
        d = sc.to_dict()
        rows.append({
            "Rank": rank,
            "Score": round(d["score"], 2),
            "Type": d["match_type"],
            "Chromosome": d["chromosome"],
            "Start": d["start"],
            "End": d["end"],
            "Strand": d["strand"],
            "Mismatches": d["mismatches"],
            "MM positions": ";".join(map(str, d["mismatch_positions"])) or "-",
            "PAM": d["pam"],
            "PAM ok": "yes" if d["pam_compatible"] else "no",
            "Matched seq": d["matched_sequence"],
            "Genomic seq": d["genomic_sequence"],
        })
    return rows


def page_results() -> None:
    st.title("Analysis Results")
    analysis = _state().get("analysis")
    if analysis is None:
        st.info("No analysis yet. Run one on the **CRISPR Analysis** page.")
        return

    st.caption(
        f"Guide `{analysis.guide}` · file `{Path(analysis.fasta_path).name}` · "
        f"backend `{analysis.processing_method}` · "
        f"{analysis.elapsed_seconds:.3f} s"
    )

    if not analysis.scored:
        st.warning("**No candidates found** for these settings. Try a "
                   "different guide, more mismatches, or PAM "
                   "`require_pam=False`.")
        return

    rows = _results_rows(analysis)

    c1, c2, c3 = st.columns([2, 1, 1])
    search = c1.text_input("Search", placeholder="chromosome, sequence, PAM…")
    strand_f = c2.selectbox("Strand", ["all", "+", "-"])
    max_mm_f = c3.slider("Max mismatches", 0, 30, 30)

    filtered = rows
    if search.strip():
        q = search.strip().lower()
        filtered = [r for r in filtered
                    if q in str(r["Chromosome"]).lower()
                    or q in str(r["Matched seq"]).lower()
                    or q in str(r["Genomic seq"]).lower()
                    or q in str(r["PAM"]).lower()
                    or q in str(r["MM positions"])]
    if strand_f != "all":
        filtered = [r for r in filtered if r["Strand"] == strand_f]
    filtered = [r for r in filtered if r["Mismatches"] <= max_mm_f]

    st.caption(f"Showing **{len(filtered):,}** of **{len(rows):,}** candidates")
    st.dataframe(filtered, width="stretch", hide_index=True,
                 height=600)


def page_performance() -> None:
    st.title("Performance")
    bench = _state().get("benchmark")
    if bench is None:
        st.info("No benchmark yet. Run an analysis — timings are recorded "
                "automatically on the **CRISPR Analysis** page.")
    else:
        d = bench.to_dict()
        cols = st.columns(4)
        cols[0].metric("Parse time", f"{d['parse_seconds']:.3f} s")
        cols[1].metric("Scan time", f"{d['align_seconds']:.3f} s")
        cols[2].metric("Total time", f"{d['total_seconds']:.3f} s")
        cols[3].metric("Throughput",
                       f"{d['throughput_bases_per_sec']:,.0f} bases/s")

        cols = st.columns(4)
        cols[0].metric("Bases processed", f"{d['bases_processed']:,}")
        cols[1].metric("Windows evaluated", f"{d['windows_evaluated']:,}")
        cols[2].metric("Candidates", f"{d['candidates_found']:,}")
        cols[3].metric("Chunks scanned", f"{d['chunks_scanned']:,}")

        mem = d["memory_peak_bytes"]
        st.metric("Peak memory (tracemalloc)",
                  "n/a" if mem is None else
                  f"{mem / (1024 * 1024):.1f} MiB")

        st.subheader("Run details")
        st.write({
            "FASTA": Path(d["fasta_path"]).name,
            "Guide": d["guide"],
            "Backend": d["backend"],
            "Notes": d["notes"],
        })

    st.subheader("GPU status")
    gpu = gpu_status()
    if gpu is None:
        st.error(f"GPU probe failed: {_state().get('gpu_error', 'unknown')}")
    elif gpu.available:
        st.success(f"GPU available via **{gpu.backend}**")
        st.json(gpu.to_dict())
    else:
        st.warning("GPU unavailable")
        st.caption(gpu.reason)


def page_reports() -> None:
    st.title("Reports")
    analysis = _state().get("analysis")
    if analysis is None:
        st.info("No analysis to export yet. Run one on the "
                "**CRISPR Analysis** page.")
        return

    st.caption(f"Exports for guide `{analysis.guide}` on "
               f"`{Path(analysis.fasta_path).name}` "
               f"({len(analysis.scored):,} candidates).")

    out_dir = Path(tempfile.gettempdir()) / "geneware_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"geneware_{Path(analysis.fasta_path).stem}"

    try:
        csv_path = write_csv(analysis, out_dir / f"{stem}.csv")
        json_path = write_json(analysis, out_dir / f"{stem}.json")
        html_path = write_html(analysis, out_dir / f"{stem}.html")
    except Exception as exc:
        st.error(f"Report generation failed: {exc}")
        return

    col1, col2, col3 = st.columns(3)
    col1.download_button("Download CSV", csv_path.read_bytes(),
                         file_name=csv_path.name,
                         mime="text/csv")
    col2.download_button("Download JSON", json_path.read_bytes(),
                         file_name=json_path.name,
                         mime="application/json")
    col3.download_button("Download HTML report", html_path.read_bytes(),
                         file_name=html_path.name,
                         mime="text/html")
    st.caption("All reports contain only real analysis data "
               "(settings, timings, sequence stats, ranked candidates).")


PAGES_FUNCS = {
    "Home Dashboard": page_home,
    "Genome Upload": page_upload,
    "CRISPR Analysis": page_analysis,
    "Analysis Results": page_results,
    "Performance": page_performance,
    "Reports": page_reports,
}


def main() -> None:
    with st.sidebar:
        st.markdown("# 🧬 GeneWeaver")
        st.caption("CRISPR alignment engine")
        page = st.radio("Navigate", PAGES, key="nav",
                        label_visibility="collapsed")
        st.divider()
        if genome_loaded():
            st.caption(f"Genome: `{_state()['fasta_name']}`")
        else:
            st.caption("Genome: not loaded")
        if _state().get("analysis") is not None:
            st.caption(f"Last analysis: {len(_state()['analysis'].scored):,} "
                       "candidates")
        else:
            st.caption("Last analysis: none")

    PAGES_FUNCS[page]()


if __name__ == "__main__":
    main()
