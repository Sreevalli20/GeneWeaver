"""GeneWeaver — CRISPR Guide Analyzer (Streamlit web app).

A guided bioinformatics workbench over the real analysis engine. Every
page calls the actual Python functions — :mod:`geneware.fasta` for
parsing, :mod:`geneware.pipeline` for the CPU / Dask / GPU scan,
:mod:`geneware.alignment_view` for sequence views, and
:mod:`geneware.reports` for exports. Backend diagnostics live in a
collapsible *System Information* section; the main screens stay clean.
No analysis logic is duplicated here and no metrics are fabricated.

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
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from geneware import __version__
from geneware.alignment_view import alignment_segments, candidate_view
from geneware.aligner import AlignError, validate_guide
from geneware.fasta import FastaParseError, compute_stats, parse_fasta
from geneware.pipeline import AnalysisError, run_benchmark
from geneware.reports import write_csv, write_html, write_json, write_pdf
from geneware.scoring import score_breakdown, score_formula_text

EXAMPLE_FASTA = Path(__file__).resolve().parent.parent / "data" / "example.fasta"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "geneware_uploads"
ALLOWED_SUFFIXES = {".fasta", ".fa", ".fna"}
PAGES = [
    "Overview",
    "Genome Workspace",
    "Guide Designer",
    "Analysis Workbench",
    "Results Explorer",
    "Performance",
    "Reports",
]

st.set_page_config(
    page_title="GeneWeaver — CRISPR Guide Analyzer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    :root {
        --gw-bg: #0a1526;
        --gw-panel: #101f38;
        --gw-panel-2: #132544;
        --gw-border: #1f3a5f;
        --gw-teal: #2dd4bf;
        --gw-cyan: #38bdf8;
        --gw-text: #e6eefb;
        --gw-muted: #93a7c4;
        --gw-good: #34d399;
        --gw-bad: #f87171;
        --gw-warn: #fbbf24;
    }
    .stApp { background: #0a1526; }
    html, body, [class*="css"], .stMarkdown, p, span, div, label {
        color: var(--gw-text);
    }
    section[data-testid="stSidebar"] {
        background: #0d1a2e;
        border-right: 1px solid var(--gw-border);
    }
    section[data-testid="stSidebar"] * { color: var(--gw-text); }
    .block-container { padding-top: 1.6rem; }
    h1, h2, h3 { color: #f0f6ff !important; letter-spacing: .2px; }
    .gw-muted, .gw-muted * { color: var(--gw-muted) !important; }

    /* Cards */
    .gw-card {
        background: linear-gradient(180deg, #12233f 0%, #0f1d36 100%);
        border: 1px solid var(--gw-border);
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: .9rem;
    }
    .gw-card h4 { margin: 0 0 .35rem 0; color: #f0f6ff; font-size: 1.02rem; }
    .gw-card .gw-sub { color: var(--gw-muted); font-size: .86rem; }
    .gw-hero {
        text-align: center;
        padding: 2.4rem 1rem 1.6rem 1rem;
        border: 1px solid var(--gw-border);
        border-radius: 18px;
        background:
            radial-gradient(1200px 300px at 50% -10%, #14406b55, transparent),
            linear-gradient(180deg, #0f1f3a, #0b1830);
        margin-bottom: 1.4rem;
    }
    .gw-hero h1 { font-size: 2.5rem; margin: 0 0 .3rem 0;
                  background: linear-gradient(90deg,#7dd3fc,#2dd4bf);
                  -webkit-background-clip: text; background-clip: text;
                  -webkit-text-fill-color: transparent; }
    .gw-hero p { color: var(--gw-muted); font-size: 1.05rem; margin: 0; }

    /* Step cards */
    .gw-steps { display: flex; gap: .8rem; flex-wrap: wrap; }
    .gw-step {
        flex: 1 1 200px;
        background: #101f38;
        border: 1px solid var(--gw-border);
        border-radius: 12px;
        padding: 1rem 1.1rem;
    }
    .gw-step .num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 1.9rem; height: 1.9rem; border-radius: 50%;
        background: linear-gradient(135deg, #0e7490, #2dd4bf);
        color: #04202a; font-weight: 700; margin-bottom: .5rem;
    }
    .gw-step b { color: #f0f6ff; }
    .gw-step span.d { color: var(--gw-muted); font-size: .85rem; }

    /* Buttons */
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid var(--gw-border) !important;
        background: #142638 !important;
        color: var(--gw-text) !important;
        font-weight: 600;
    }
    .stButton > button:hover { border-color: var(--gw-teal) !important; }
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #0e7490, #0891b2) !important;
        border: none !important;
        color: #f0fdfa !important;
    }

    /* Nucleotide visualization */
    .nt {
        display: inline-block;
        min-width: 1.15rem;
        text-align: center;
        font-family: Consolas, "Courier New", monospace;
        font-weight: 700;
        font-size: .95rem;
        padding: .05rem .1rem;
        border-radius: 3px;
    }
    .nt-wrap {
        font-family: Consolas, monospace; font-size: .95rem;
        line-height: 1.9; word-break: break-all;
        background: #0b1830; border: 1px solid var(--gw-border);
        border-radius: 10px; padding: .7rem .9rem;
    }
    .aln { font-family: Consolas, monospace; font-size: .95rem; }
    .aln .row { display: flex; align-items: baseline; margin: .15rem 0; }
    .aln .lbl {
        color: var(--gw-muted); width: 8.6rem; flex: none;
        font-family: "Segoe UI", sans-serif; font-size: .8rem;
    }
    .aln .bases { letter-spacing: .5px; }
    .aln .k-match .nt { background: transparent; }
    .aln .k-mismatch .nt {
        background: #7f1d1d; outline: 1px solid #f87171;
        color: #fecaca !important;
    }
    .aln .k-ambiguous .nt { background: #78350f; color: #fde68a !important; }
    .aln .marker { color: var(--gw-bad); }
    .strand-pill {
        display: inline-block; padding: .12rem .6rem; border-radius: 999px;
        font-size: .78rem; font-weight: 700; letter-spacing: .3px;
    }
    .strand-fwd { background: #064e3b; color: #6ee7b7; border: 1px solid #059669; }
    .strand-rev { background: #7f1d1d; color: #fca5a5; border: 1px solid #dc2626; }

    /* Status banners */
    .gw-banner {
        border-radius: 12px; padding: .9rem 1.1rem; margin-bottom: 1rem;
        border: 1px solid var(--gw-border); background: #101f38;
    }
    .gw-banner.good { border-color: #059669; background: #052e22; }
    .gw-banner.warn { border-color: #b45309; background: #2b1d05; }
    .gw-banner.info { border-color: #0369a1; background: #082f49; }

    /* Inputs */
    .stTextInput input, .stSelectbox div[data-baseweb="select"] > div,
    .stSlider, .stMultiSelect, .stNumberInput input {
        background: #0b1830 !important;
        color: var(--gw-text) !important;
        border-color: var(--gw-border) !important;
    }
    label, .stWidgetLabel { color: var(--gw-muted) !important; }

    /* Misc */
    .gw-kpi { text-align: center; padding: .4rem 0; }
    .gw-kpi .v { font-size: 1.7rem; font-weight: 800; color: #f0f6ff; }
    .gw-kpi .k { color: var(--gw-muted); font-size: .8rem;
                 text-transform: uppercase; letter-spacing: .6px; }
    hr { border-color: var(--gw-border) !important; }
    .stCaption, .stCaption p { color: var(--gw-muted) !important; }
    .stAlert { border-radius: 10px; }
    table { color: var(--gw-text); }
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
    """Probe GPU availability once per session (System Info only)."""
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


def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "n/a"


# ---------------------------------------------------------------------------
# Visual components (all driven by real data)
# ---------------------------------------------------------------------------

_BASE_COLORS = {
    "A": "#4ade80", "C": "#60a5fa", "G": "#fbbf24", "T": "#f87171",
    "U": "#f87171",
}


def _base_color(base: str) -> str:
    return _BASE_COLORS.get(base.upper(), "#94a3b8")


def seq_blocks(seq: str, max_len: int | None = None,
               kind_map: dict[int, str] | None = None,
               guide: str | None = None) -> str:
    """Render a nucleotide sequence as coloured blocks.

    ``kind_map`` optionally maps positions to 'match' / 'mismatch' /
    'ambiguous' which adds a highlight class per position.
    """
    text = seq if max_len is None else seq[:max_len]
    out = []
    for i, b in enumerate(text):
        cls = "nt"
        if kind_map and i in kind_map:
            cls += f" k-{kind_map[i]}"
        elif guide is not None and i < len(guide):
            cls += " k-match"
        out.append(
            f'<span class="{cls}" style="color:{_base_color(b)}">'
            f"{html.escape(b)}</span>")
    suffix = "…" if max_len is not None and len(seq) > max_len else ""
    return f'<span class="nt-wrap"><span class="bases">{"".join(out)}</span>{suffix}</span>'


def strand_pill(strand: str) -> str:
    if strand == "+":
        return ('<span class="strand-pill strand-fwd">→ FORWARD (+)</span>')
    return '<span class="strand-pill strand-rev">← REVERSE (−)</span>'


def alignment_panel(view) -> str:
    """Guide-vs-candidate comparison with mismatch highlighting."""
    seg = alignment_segments(view)
    if not seg:
        return ("<p class='gw-muted'>Sequence context unavailable for "
                "this candidate.</p>")

    def row(label: str, key: str) -> str:
        spans = []
        for s in seg:
            kind = s["kind"] if key == "matched" else "match"
            spans.append(
                f'<span class="nt k-{kind}" '
                f'style="color:{_base_color(s[key])}">'
                f'{html.escape(s[key])}</span>')
        return (f'<div class="row"><span class="lbl">{label}</span>'
                f'<span class="bases">{"".join(spans)}</span></div>')

    marker = "".join(
        "^" if s["kind"] == "mismatch"
        else "~" if s["kind"] == "ambiguous" else " "
        for s in seg)
    pam = (f'<div class="row"><span class="lbl">PAM motif</span>'
           f'<span class="bases" style="font-family:Consolas,monospace;'
           f'color:#7dd3fc">{html.escape(view.pam_sequence)}</span>'
           f'<span class="gw-muted" style="font-size:.78rem"> '
           f'(candidate strand · plus-strand footprint '
           f'{html.escape(view.pam_plus_sequence) or "n/a"})</span></div>')
    legend = (
        '<div class="gw-muted" style="font-size:.8rem;margin-top:.5rem">'
        '<span style="color:#4ade80">■</span> match &nbsp; '
        '<span style="color:#fca5a5">■</span> mismatch &nbsp; '
        '<span style="color:#fcd34d">■</span> ambiguous base &nbsp; '
        '<span style="color:#f87171">^</span> mismatch position'
        '</div>')
    return (f'<div class="aln">{row("Guide 5\'→3\'", "guide")}'
            f'{row("Genome match", "matched")}'
            f'<div class="row marker"><span class="lbl"></span>'
            f'<span class="bases">{html.escape(marker)}</span></div>'
            f'{pam}</div>{legend}')


def card(title: str, body_html: str = "", sub: str = "") -> None:
    sub_html = f'<div class="gw-sub">{sub}</div>' if sub else ""
    st.markdown(
        f'<div class="gw-card"><h4>{html.escape(title)}</h4>'
        f'{sub_html}{body_html}</div>', unsafe_allow_html=True)


def banner(kind: str, text_html: str) -> None:
    st.markdown(f'<div class="gw-banner {kind}">{text_html}</div>',
                unsafe_allow_html=True)


def empty_state(icon: str, title: str, text: str,
                button_label: str | None = None,
                target: str | None = None) -> None:
    body = (f'<div style="text-align:center;padding:2.2rem 1rem">'
            f'<div style="font-size:2.6rem">{icon}</div>'
            f'<h3 style="margin:.4rem 0">{html.escape(title)}</h3>'
            f'<p class="gw-muted" style="max-width:34rem;margin:.3rem auto">'
            f'{html.escape(text)}</p>')
    if button_label and target:
        body += "</div>"
        st.markdown(body, unsafe_allow_html=True)
        if st.button(button_label, type="primary"):
            _go(target)
    else:
        st.markdown(body + "</div>", unsafe_allow_html=True)


def _require_genome(target: str = "Genome Workspace") -> bool:
    if genome_loaded():
        return True
    empty_state("🧬", "No genome loaded",
                "Upload a FASTA genome or load the bundled example to "
                "start analyzing CRISPR guide matches.",
                "Open Genome Workspace", target)
    return False


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.markdown(
        '<div class="gw-hero">'
        '<h1>🧬 GeneWeaver</h1>'
        '<p><b>CRISPR Guide Analyzer</b> — explore DNA sequences and '
        'identify potential CRISPR guide matches.</p>'
        '</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="gw-steps">'
        '<div class="gw-step"><span class="num">1</span><br>'
        '<b>Load a genome</b><br><span class="d">Upload a FASTA file '
        'or try the example genome.</span></div>'
        '<div class="gw-step"><span class="num">2</span><br>'
        '<b>Design a guide</b><br><span class="d">Enter your guide '
        'sequence, PAM pattern, and mismatch tolerance.</span></div>'
        '<div class="gw-step"><span class="num">3</span><br>'
        '<b>Analyze &amp; explore</b><br><span class="d">Run the '
        'engine, inspect candidates, and export reports.</span></div>'
        '</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("🧬  Start New Analysis", type="primary",
                     use_container_width=True):
            _go("Genome Workspace" if not genome_loaded()
                else "Guide Designer")
    with b2:
        if st.button("📂  Genome Workspace", use_container_width=True):
            _go("Genome Workspace")
    with b3:
        if st.button("🧪  Load Example Genome",
                     use_container_width=True):
            err = load_genome(EXAMPLE_FASTA, EXAMPLE_FASTA.name)
            if err is None:
                banner("good", "<b>Example genome loaded.</b> You can "
                       "now design a guide and run an analysis.")
            else:
                banner("warn", f"Could not load example: {err}")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="gw-card"><h4>🧬 Current genome</h4>',
                    unsafe_allow_html=True)
        if genome_loaded():
            stats = _state()["seq_stats"]
            st.markdown(
                f'<div class="gw-kpi"><div class="v">'
                f'{_fmt_int(stats["record_count"])}</div>'
                f'<div class="k">sequences</div></div>'
                f'<div class="gw-kpi"><div class="v">'
                f'{_fmt_int(stats["total_bases"])}</div>'
                f'<div class="k">total bases</div></div>'
                f'<div class="gw-kpi"><div class="v">'
                f'{stats["gc_percent"]}%</div>'
                f'<div class="k">GC content</div></div>'
                f'<p class="gw-muted" style="text-align:center;margin:0">'
                f'<code>{html.escape(_state()["fasta_name"])}</code></p>',
                unsafe_allow_html=True)
        else:
            st.markdown('<p class="gw-muted">No genome loaded yet. '
                        'Upload a FASTA file or load the example.</p>',
                        unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="gw-card"><h4>🧪 Latest analysis</h4>',
                    unsafe_allow_html=True)
        analysis = _state().get("analysis")
        if analysis is None:
            st.markdown('<p class="gw-muted">No analyses yet. Once you '
                        'run one, a summary will appear here.</p>',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="gw-kpi"><div class="v">'
                f'{_fmt_int(len(analysis.scored))}</div>'
                f'<div class="k">potential matches</div></div>'
                f'<p class="gw-muted" style="text-align:center;margin:0">'
                f'guide <code>{html.escape(analysis.guide)}</code> · '
                f'{analysis.elapsed_seconds:.2f} s · '
                f'{html.escape(analysis.processing_method)}<br>'
                f'{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC</p>',
                unsafe_allow_html=True)
            if st.button("Open Results Explorer"):
                _go("Results Explorer")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("ℹ️  About GeneWeaver"):
        st.markdown(
            "GeneWeaver scans a genome for sites that resemble a CRISPR "
            "guide RNA, on both strands, with a configurable PAM motif "
            "and mismatch tolerance. Candidates are ranked with an "
            "**experimental heuristic score** — it is a computational "
            "ranking only, **not** a validated biological prediction, "
            "and does not prove that an edit will occur."
        )


def page_workspace() -> None:
    st.title("🧬 Genome Workspace")
    st.caption("Load the DNA sequences you want to search. FASTA "
               "(.fasta, .fa, .fna) files are parsed by the real "
               "GeneWeaver parser.")

    col_up, col_ex = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader(
            "Upload a FASTA genome",
            type=["fasta", "fa", "fna"],
            accept_multiple_files=False,
            help="Choose a .fasta, .fa, or .fna file containing DNA "
                 "sequences in FASTA format",
        )
        if uploaded is not None:
            key = (uploaded.name, uploaded.size)
            if _state().get("uploaded_key") != key:
                suffix = Path(uploaded.name).suffix.lower()
                if suffix not in ALLOWED_SUFFIXES:
                    _state()["load_error"] = (
                        f"Unsupported file type '{suffix}'. Please upload "
                        "a .fasta, .fa, or .fna file.")
                    _state()["uploaded_key"] = key
                else:
                    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                    dest = UPLOAD_DIR / Path(uploaded.name).name
                    dest.write_bytes(uploaded.getbuffer())
                    err = load_genome(dest, uploaded.name)
                    _state()["uploaded_key"] = key
                    if err is not None:
                        banner("warn",
                               f"<b>Could not parse that file:</b> {err}")
                    else:
                        banner("good",
                               f"<b>Genome loaded:</b> "
                               f"{_state()['fasta_name']}")
    with col_ex:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧪  Example genome",
                     use_container_width=True):
            err = load_genome(EXAMPLE_FASTA, EXAMPLE_FASTA.name)
            if err is None:
                banner("good", "<b>Example genome loaded.</b>")
            else:
                banner("warn", f"Could not load example: {err}")

    err = _state().get("load_error")
    if err and not genome_loaded():
        banner("warn", f"<b>Last load failed:</b> {html.escape(err)}")

    if not genome_loaded():
        empty_state("📂", "Your workspace is empty",
                    "Upload a FASTA file above, or load the bundled "
                    "example genome to explore the workflow.")
        return

    stats = _state()["seq_stats"]
    banner("good", f"<b>Genome loaded successfully</b> — "
           f"{_fmt_int(stats['record_count'])} sequences · "
           f"{_fmt_int(stats['total_bases'])} bases · "
           f"GC {stats['gc_percent']}% · "
           f"{_fmt_int(stats.get('ambiguous_count', 0))} ambiguous bases")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sequences", _fmt_int(stats["record_count"]))
    c2.metric("Total bases", _fmt_int(stats["total_bases"]))
    c3.metric("GC content", f"{stats['gc_percent']}%")
    c4.metric("File", _state()["fasta_name"])

    st.subheader("Sequences")
    rows = [{
        "ID": rec.record_id,
        "Description": rec.description,
        "Length (bp)": _fmt_int(rec.length),
        "GC %": pr["gc_percent"],
    } for rec, pr in zip(_state()["records"], stats["per_record"],
                         strict=True)]
    st.dataframe(rows, width="stretch", hide_index=True)

    if _state()["records"]:
        st.subheader("Sequence preview")
        rec = _state()["records"][0]
        st.caption(f"First bases of `{rec.record_id}` — "
                   f"{_fmt_int(rec.length)} bp total. "
                   "A = green · C = blue · G = amber · T = red · "
                   "ambiguous = grey.")
        st.markdown(seq_blocks(rec.sequence, max_len=240),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    b1, b2 = st.columns([1, 3])
    with b1:
        if st.button("Replace genome", use_container_width=True):
            _state()["uploaded_key"] = None
            st.rerun()
    with b2:
        if st.button("Next: design a guide →", type="primary",
                     use_container_width=True):
            _go("Guide Designer")


def page_guide_designer() -> None:
    st.title("🧬 Guide Designer")
    st.caption("Configure the CRISPR guide and search settings. "
               "Everything is validated before the analysis runs.")

    if not _require_genome():
        return

    stats = _state()["seq_stats"]
    st.caption(f"Genome: **{_state()['fasta_name']}** — "
               f"{_fmt_int(stats['record_count'])} sequences, "
               f"{_fmt_int(stats['total_bases'])} bases.")

    cfg = _state().get("guide_config") or {}
    with st.form("guide_form"):
        guide = st.text_input(
            "Guide sequence",
            value=cfg.get("guide", "ACGTACGTACGTACGTACGTACGT"),
            help="Your guide RNA spacer written as DNA, 5'→3' — "
                 "10 to 30 bases (letters A C G T and IUPAC ambiguity "
                 "codes like N are allowed; U is read as T).",
        )
        pam_raw = st.text_input(
            "PAM pattern",
            value=cfg.get("pam", "NGG"),
            help="The short motif that must sit next to a match. "
                 "NGG is the classic Cas9 PAM; N means 'any base'. "
                 "Separate several patterns with commas, e.g. NGG,NAG.",
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            max_mm = st.slider(
                "Max mismatches", 0, 10,
                int(cfg.get("max_mm", 0)),
                help="How many guide bases may differ from the "
                     "genome. 0 = perfect matches only; higher values "
                     "return more, weaker candidates.",
            )
        with col2:
            strands = st.multiselect(
                "Search strands", ["+", "-"],
                default=cfg.get("strands", ["+", "-"]),
                format_func=lambda s: ("Forward (+)" if s == "+"
                                       else "Reverse (−)"),
                help="Scan the forward strand, the reverse strand, "
                     "or both.",
            )
        with col3:
            backend_label = st.selectbox(
                "Processing mode",
                _backend_labels(),
                index=0,
                help="CPU is the reliable default and is recommended. "
                     "GPU/Dask are only used when genuinely available "
                     "on this machine.",
            )
        if guide:
            st.markdown(
                "Guide preview: " +
                "".join(
                    f'<span class="nt" style="color:{_base_color(b)}">'
                    f"{html.escape(b)}</span>"
                    for b in guide.upper()),
                unsafe_allow_html=True)
        save = st.form_submit_button("Save guide design",
                                     type="primary",
                                     use_container_width=True)

    if save:
        # Validate before accepting the configuration.
        try:
            guide_norm = validate_guide(guide)
            pams = tuple(s.strip() for s in pam_raw.split(",")
                         if s.strip())
            if not pams:
                raise AlignError("at least one PAM pattern is required")
            if not strands:
                raise AlignError("select at least one strand")
        except AlignError as exc:
            banner("warn", f"<b>Check your guide design:</b> {exc}")
        else:
            backend = {label: value for value, label in
                       _backend_options()}[backend_label]
            _state()["guide_config"] = {
                "guide": guide_norm,
                "pam": ", ".join(pams),
                "max_mm": int(max_mm),
                "strands": list(strands),
                "backend": backend,
                "backend_label": backend_label,
            }
            banner("good",
                   f"<b>Guide design saved.</b> Guide "
                   f"<code>{guide_norm}</code> · PAM "
                   f"<code>{html.escape(', '.join(pams))}</code> · up to "
                   f"{max_mm} mismatch{'es' if max_mm != 1 else ''} · "
                   f"{html.escape(backend_label)}.")

    if _state().get("guide_config"):
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Continue to Analysis Workbench →",
                     type="primary"):
            _go("Analysis Workbench")


def _backend_options() -> list[tuple[str, str]]:
    """(value, friendly label); GPU only when genuinely detected."""
    opts = [("cpu", "CPU — recommended (reliable)"),
            ("auto", "Automatic (best available)")]
    if dask_available():
        opts.append(("dask", "Parallel CPU (Dask)"))
    gpu = gpu_status()
    if gpu is not None and gpu.available:
        opts.append(("gpu", "GPU (CUDA)"))
    return opts


def _backend_labels() -> list[str]:
    return [label for _, label in _backend_options()]


def page_workbench() -> None:
    st.title("⚙️ Analysis Workbench")
    st.caption("Run the real GeneWeaver alignment engine on your "
               "loaded genome.")

    if not _require_genome():
        return
    cfg = _state().get("guide_config")
    if not cfg:
        empty_state("🧬", "No guide designed yet",
                    "Design a guide sequence and search settings first.",
                    "Open Guide Designer", "Guide Designer")
        return

    # Configuration summary card
    strand_txt = ("both strands" if len(cfg["strands"]) == 2
                  else ("forward (+)" if cfg["strands"] == ["+"]
                        else "reverse (−)"))
    st.markdown(
        '<div class="gw-card"><h4>Current configuration</h4>'
        f'<p style="margin:.2rem 0">Guide '
        f'<code>{html.escape(cfg["guide"])}</code> · PAM '
        f'<code>{html.escape(cfg["pam"])}</code> · up to '
        f'{cfg["max_mm"]} mismatch'
        f'{"es" if cfg["max_mm"] != 1 else ""} · {strand_txt} · '
        f'{html.escape(cfg["backend_label"])}</p>'
        '<p class="gw-muted" style="margin:.2rem 0">Genome: '
        f'<code>{html.escape(_state()["fasta_name"])}</code></p></div>',
        unsafe_allow_html=True)

    if _state().get("analyzing"):
        banner("info", "<b>Analysis in progress…</b> please wait.")
        return

    col_run, col_edit = st.columns([1, 3])
    with col_run:
        run = st.button("▶  Run Analysis", type="primary",
                        use_container_width=True)
    with col_edit:
        st.caption("Uses the real backend — CPU is the reliable "
                   "default; progress reflects actual chunks scanned.")

    if not run:
        # Show the previous result banner if present.
        analysis = _state().get("analysis")
        if analysis is not None:
            banner("good",
                   f"<b>Last run complete:</b> "
                   f"{_fmt_int(len(analysis.scored))} potential matches "
                   f"in {analysis.elapsed_seconds:.2f} s "
                   f"({html.escape(analysis.processing_method)}).")
            if st.button("Open Results Explorer →"):
                _go("Results Explorer")
        return

    # ---- execute the real backend ----------------------------------
    pams = tuple(s.strip() for s in cfg["pam"].split(",") if s.strip())
    _state()["analyzing"] = True
    stage = st.empty()
    progress = st.progress(0.0)

    def _progress(done: int, total: int) -> None:
        if total:
            stage.info(f"⏳ Scanning genome — chunk **{done} of {total}**")
            progress.progress(min(done / total, 0.95),
                              text=f"Chunk {done}/{total}")
        else:
            stage.info("⏳ Scanning genome…")
            progress.progress(0.5, text="Scanning…")

    stage.info("⏳ Starting analysis — parsing genome and preparing "
               "scan…")
    try:
        benchmark, analysis = run_benchmark(
            _state()["fasta_path"], cfg["guide"],
            backend=cfg["backend"],
            pam_patterns=pams,
            strands=tuple(cfg["strands"]),
            max_mismatches=int(cfg["max_mm"]),
            progress=_progress,
        )
    except (FastaParseError, AlignError, AnalysisError, MemoryError) as exc:
        stage.empty()
        progress.empty()
        banner("warn", f"<b>Analysis failed:</b> {html.escape(str(exc))}")
        return
    except Exception as exc:
        stage.empty()
        progress.empty()
        banner("warn",
               f"<b>Analysis failed:</b> {type(exc).__name__}: "
               f"{html.escape(str(exc))}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc(limit=5))
        return
    finally:
        _state()["analyzing"] = False

    stage.empty()
    progress.progress(1.0, text="Complete")
    _state()["analysis"] = analysis
    _state()["benchmark"] = benchmark
    _state()["analyze_error"] = None

    banner("good",
           f"<b>✔ Analysis complete.</b> Found "
           f"<b>{_fmt_int(len(analysis.scored))}</b> potential matches "
           f"in <b>{analysis.elapsed_seconds:.2f} s</b> using "
           f"<b>{html.escape(analysis.processing_method)}</b>.")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Explore Results →", type="primary",
                     use_container_width=True):
            _go("Results Explorer")
    with c2:
        if st.button("View Performance", use_container_width=True):
            _go("Performance")


def _results_rows(analysis) -> list[dict]:
    rows = []
    for rank, sc in enumerate(analysis.scored, start=1):
        d = sc.to_dict()
        rows.append({
            "Rank": rank,
            "Score": round(d["score"], 1),
            "Type": "Exact" if d["match_type"] == "exact" else "Near",
            "Chromosome": d["chromosome"],
            "Position": f"{d['start']}–{d['end']}",
            "Strand": d["strand"],
            "PAM": d["pam"],
            "Mismatches": d["mismatches"],
            "Matched sequence": d["matched_sequence"],
        })
    return rows


def page_results() -> None:
    st.title("🔬 Results Explorer")
    analysis = _state().get("analysis")
    if analysis is None:
        empty_state("🔬", "No results yet",
                    "Run an analysis first — your candidate matches "
                    "will appear here.",
                    "Open Analysis Workbench", "Analysis Workbench")
        return

    n_exact = sum(1 for s in analysis.scored
                  if s.candidate.match_type == "exact")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Potential matches", _fmt_int(len(analysis.scored)))
    c2.metric("Exact matches", _fmt_int(n_exact))
    c3.metric("Sequences searched",
              _fmt_int(analysis.sequence_stats.get("record_count", 0)))
    c4.metric("Analysis time", f"{analysis.elapsed_seconds:.2f} s")
    c5.metric("Execution", analysis.processing_method)

    strand_txt = ("both strands" if len(analysis.strands) == 2
                  else ("forward (+)" if analysis.strands[0] == "+"
                        else "reverse (−)"))
    st.caption(
        f"Guide `{analysis.guide}` · PAM "
        f"{', '.join(analysis.pam_patterns)} · ≤ "
        f"{analysis.max_mismatches} mismatches · {strand_txt} · "
        f"genome `{Path(analysis.fasta_path).name}`")

    if not analysis.scored:
        banner("warn",
               "<b>No matches found</b> with these settings. This can "
               "happen when the guide differs a lot from the genome, "
               "the PAM pattern is too strict, or the mismatch limit "
               "is too low — try a different guide, relax the PAM, or "
               "raise the mismatch limit.")
        if st.button("Adjust guide design"):
            _go("Guide Designer")
        return

    rows = _results_rows(analysis)
    chromosomes = sorted({str(r["Chromosome"]) for r in rows})

    f1, f2, f3, f4, f5 = st.columns([2, 1, 1, 1, 1])
    search = f1.text_input("Search", placeholder="chromosome, PAM, "
                           "sequence…")
    strand_f = f2.selectbox("Strand", ["all", "+", "-"],
                            format_func=lambda s:
                            "All strands" if s == "all" else
                            ("Forward (+)" if s == "+" else "Reverse (−)"))
    chr_f = f3.selectbox("Chromosome", ["all"] + chromosomes,
                         format_func=lambda s:
                         "All" if s == "all" else s)
    type_f = f4.selectbox("Match type", ["all", "exact", "near"],
                          format_func=lambda s:
                          {"all": "All", "exact": "Exact",
                           "near": "Near match"}[s])
    max_mm_f = f5.slider("Max mismatches", 0, 10, 10)

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
        filtered = [r for r in filtered
                    if str(r["Chromosome"]) == chr_f]
    if type_f != "all":
        want = "Exact" if type_f == "exact" else "Near"
        filtered = [r for r in filtered if r["Type"] == want]
    filtered = [r for r in filtered if r["Mismatches"] <= max_mm_f]

    st.caption(f"Showing **{_fmt_int(len(filtered))}** of "
               f"**{_fmt_int(len(rows))}** matches")
    st.dataframe(
        filtered, width="stretch", hide_index=True, height=400,
        column_config={
            "Score": st.column_config.NumberColumn(
                "Score", format="%.1f",
                help="Experimental ranking score (0–100) — a "
                     "heuristic, not a validated biological prediction."),
            "Matched sequence": st.column_config.TextColumn(
                "Matched sequence", width="large"),
        })

    # ---- Candidate inspector ---------------------------------------
    st.subheader("🔬 Candidate inspector")
    if not filtered:
        st.caption("No candidates match the current filters — adjust "
                   "them above.")
    else:
        labels = [
            f"#{r['Rank']} · {r['Chromosome']}:{r['Position']} · "
            f"{'→' if r['Strand'] == '+' else '←'} · score {r['Score']}"
            for r in filtered[:500]]
        pick = st.selectbox("Select a candidate to inspect", labels,
                            key="cand_pick")
        chosen = filtered[labels.index(pick)]
        sc = analysis.scored[chosen["Rank"] - 1]
        cand = sc.candidate
        view = candidate_view(cand, analysis.records)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Score", f"{sc.score:.1f}")
        m2.metric("Mismatches", cand.mismatches)
        m3.metric("Strand",
                  "Forward (+)" if cand.strand == "+" else "Reverse (−)")
        m4.metric("PAM", cand.pam)

        st.markdown(
            f'<div class="gw-card"><h4>Sequence comparison</h4>'
            f'<p class="gw-sub">{strand_pill(cand.strand)} &nbsp; '
            f'<code>{html.escape(cand.chromosome)}</code> positions '
            f'{cand.start}–{cand.end} (0-based) · PAM footprint '
            f'{cand.pam_start}–{cand.pam_end}</p>'
            f'{alignment_panel(view)}</div>',
            unsafe_allow_html=True)

        with st.expander("Score breakdown (experimental)"):
            bd = score_breakdown(cand)
            st.markdown(score_formula_text())
            st.json({k: v for k, v in bd.items()})

    st.markdown("<br>", unsafe_allow_html=True)
    banner("info",
           "<b>How to read these results.</b> Green bases match the "
           "guide; red-highlighted bases are mismatches; amber means "
           "an ambiguous (IUPAC) base. The score is an internal "
           "experimental ranking heuristic — higher is better, but it "
           "does <b>not</b> prove biological effectiveness.")
    if st.button("Export these results →"):
        _go("Reports")


def page_performance() -> None:
    st.title("⚡ Performance")
    bench = _state().get("benchmark")
    analysis = _state().get("analysis")

    if bench is None or analysis is None:
        empty_state("⚡", "No measurements yet",
                    "Run an analysis and real timing, throughput, and "
                    "memory figures will appear here.",
                    "Open Analysis Workbench", "Analysis Workbench")
        return

    d = bench.to_dict()
    throughput = d.get("throughput_bases_per_sec")
    method = analysis.processing_method
    friendly = ("Parallel CPU (Dask)" if "dask" in method
                else "GPU (CUDA)" if analysis.backend == "gpu"
                else "CPU")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Analysis duration", f"{d['total_seconds']:.2f} s")
    c2.metric("Sequences processed", _fmt_int(d.get("records_parsed")))
    c3.metric("Candidates found", _fmt_int(d.get("candidates_found")))
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
        f"`{d['guide']}` · {_fmt_int(d.get('bases_processed'))} bases "
        f"scanned in {_fmt_int(d.get('chunks_scanned'))} chunks.")

    if d.get("notes"):
        st.caption("Notes: " + "; ".join(d["notes"]))

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🛠  System Information"):
        st.markdown("Diagnostics only — none of this changes your "
                    "results.")
        gpu = gpu_status()
        if gpu is None:
            st.write("**GPU probe:** failed —",
                     _state().get("gpu_error", "unknown error"))
        elif gpu.available:
            st.success(f"GPU available via **{gpu.backend}**")
            st.json(gpu.to_dict())
        else:
            st.write("**GPU:** unavailable —", gpu.reason)
        st.write("**Dask distributed:**",
                 "available" if dask_available() else "not installed")
        import platform
        import sys
        st.write("**Python:**", sys.version.split()[0], "·",
                 platform.platform())
        extra = analysis.extra or {}
        if extra:
            st.json({k: v for k, v in extra.items() if k != "gpu"})


def page_reports() -> None:
    st.title("📄 Reports")
    analysis = _state().get("analysis")
    if analysis is None:
        empty_state("📄", "Nothing to export yet",
                    "Run an analysis first — reports are generated "
                    "from real results only.",
                    "Open Analysis Workbench", "Analysis Workbench")
        return

    st.caption(f"Reports for guide `{analysis.guide}` on "
               f"`{Path(analysis.fasta_path).name}` "
               f"({_fmt_int(len(analysis.scored))} matches, "
               f"{analysis.elapsed_seconds:.2f} s).")

    out_dir = Path(tempfile.gettempdir()) / "geneware_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (f"geneware_{Path(analysis.fasta_path).stem}_"
            f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}")

    try:
        csv_path = write_csv(analysis, out_dir / f"{stem}.csv")
        json_path = write_json(analysis, out_dir / f"{stem}.json")
        html_path = write_html(analysis, out_dir / f"{stem}.html")
        pdf_path = write_pdf(analysis, out_dir / f"{stem}.pdf")
    except Exception as exc:
        banner("warn", f"<b>Report generation failed:</b> "
               f"{html.escape(str(exc))}")
        return

    banner("good", "<b>✔ Report ready.</b> Downloads contain your "
           "guide, PAM settings, genome info, analysis date, timings, "
           "and the full ranked candidate list.")

    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("⬇️ CSV", csv_path.read_bytes(),
                       file_name=csv_path.name, mime="text/csv",
                       use_container_width=True)
    c2.download_button("⬇️ JSON", json_path.read_bytes(),
                       file_name=json_path.name,
                       mime="application/json",
                       use_container_width=True)
    c3.download_button("⬇️ HTML", html_path.read_bytes(),
                       file_name=html_path.name, mime="text/html",
                       use_container_width=True)
    c4.download_button("⬇️ PDF", pdf_path.read_bytes(),
                       file_name=pdf_path.name,
                       mime="application/pdf",
                       use_container_width=True)

    with st.expander("👁  Preview HTML report"):
        st.html(html_path.read_text(encoding="utf-8"))


PAGES_FUNCS = {
    "Overview": page_overview,
    "Genome Workspace": page_workspace,
    "Guide Designer": page_guide_designer,
    "Analysis Workbench": page_workbench,
    "Results Explorer": page_results,
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
            st.caption(f"Last run: {_fmt_int(len(analysis.scored))} "
                       "matches")
        st.divider()
        st.caption(f"v{__version__} · experimental scoring")

    PAGES_FUNCS[page]()


if __name__ == "__main__":
    main()
