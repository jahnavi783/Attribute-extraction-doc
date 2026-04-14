import sys
import io
import json
from html import escape
from pathlib import Path



import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from services.normalization_engine import NormalizationEngine, NormalizationReport
from services.attribute_matcher import DEFAULT_FUZZY_THRESHOLD, DEFAULT_SEMANTIC_THRESHOLD

# ── Theme ─────────────────────────────────────────────────────────────────────
PRIMARY = "#0B7A75"
PRIMARY_DARK = "#075E63"
PRIMARY_DEEP = "#084C61"
PRIMARY_LIGHT = "#EAF7F6"
PRIMARY_LIGHTER = "#F4FBFB"
BORDER = "#CFE7E5"
TEXT_DARK = "#14323A"
TEXT_MUTE = "#5F6F76"
WHITE = "#FFFFFF"

MATCH_META = {
    "exact": {"label": "Exact", "bg": "#E8F6F1", "fg": "#1F7A53"},
    "synonym": {"label": "Synonym", "bg": "#E8F3FB", "fg": "#2E6EA8"},
    "fuzzy": {"label": "Fuzzy", "bg": "#FFF4E6", "fg": "#B26A00"},
    "semantic": {"label": "Semantic", "bg": "#F1ECFA", "fg": "#6C46B5"},
    "unmatched": {"label": "Unmatched", "bg": "#FDECEC", "fg": "#B93A3A"},
}

st.set_page_config(
    page_title="DocNorm — Attribute Normalizer",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
<style>
  header[data-testid="stHeader"] {{display:none;}}
  [data-testid="stToolbar"] {{display:none !important;}}
  .stApp {{background: #F7FBFC; color: {TEXT_DARK};}}
  .block-container {{padding-top: 1.1rem; padding-bottom: 2rem;}}

  section[data-testid="stSidebar"] {{
    background: white !important;
    border-right: 1px solid #E2EFED;
  }}

  .main-header {{
    background: linear-gradient(135deg, {PRIMARY_DEEP} 0%, {PRIMARY} 65%, #138E89 100%);
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    margin-bottom: 1.2rem;
    color: white;
    box-shadow: 0 8px 22px rgba(8, 76, 97, 0.12);
  }}
  .main-header h1 {{margin:0; font-size:1.85rem; color:white;}}
  .main-header p {{margin:0.35rem 0 0; color:#DDF3F2; font-size:0.98rem;}}

  .section-title {{
    font-size: 1.08rem;
    font-weight: 700;
    color: {PRIMARY_DEEP};
    margin: 0.2rem 0 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #D8ECEA;
  }}

  .metric-card {{
    background: white;
    border: 1px solid {BORDER};
    border-radius: 14px;
    padding: 1rem 1.1rem;
    min-height: 92px;
    box-shadow: 0 5px 16px rgba(8, 76, 97, 0.06);
    text-align:center;
  }}
  .metric-card .val {{font-size:1.75rem; font-weight:700; color:{PRIMARY_DEEP};}}
  .metric-card .lbl {{font-size:0.88rem; color:{TEXT_MUTE}; margin-top:0.15rem;}}

  .theme-card {{
    background:white;
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 1rem 1rem 0.9rem;
    box-shadow: 0 5px 16px rgba(8, 76, 97, 0.05);
  }}

  .upload-panel {{
    background: {PRIMARY_LIGHTER};
    border: 2px dashed #9CCFCC;
    border-radius: 18px;
    padding: 0.8rem;
  }}

  div[data-testid="stFileUploader"] {{
    border: 2px dashed #9CCFCC;
    border-radius: 18px;
    padding: 0.85rem;
    background: {PRIMARY_LIGHTER};
  }}
  div[data-testid="stFileUploader"] section {{cursor:pointer;}}
  div[data-testid="stFileUploaderDropzone"] {{
    background: {PRIMARY_LIGHTER} !important;
    border: none !important;
    min-height: 250px;
    cursor: pointer;
  }}
  div[data-testid="stFileUploaderDropzone"] > div {{padding: 1.2rem 0.75rem;}}
  div[data-testid="stFileUploaderDropzoneInstructions"] small,
  div[data-testid="stFileUploaderDropzoneInstructions"] span,
  div[data-testid="stFileUploaderDropzoneInstructions"] p {{
    color: {TEXT_MUTE} !important;
  }}
  div[data-testid="stFileUploader"] button {{
    border-radius: 12px !important;
    border: 1px solid #A8D7D3 !important;
    color: {PRIMARY_DEEP} !important;
  }}

  .sidebar-card {{
    background: #DCEFF0;
    border: 1px solid #C3E2E1;
    color: {PRIMARY_DEEP};
    padding: 1rem;
    border-radius: 14px;
    font-weight: 600;
  }}

  .badge {{display:inline-flex; align-items:center; gap:6px; padding: 0.36rem 0.8rem; border-radius: 999px; font-size:0.84rem; font-weight:600;}}

  .table-wrap {{
    border: 1px solid {BORDER};
    border-radius: 14px;
    overflow: hidden;
    background: white;
  }}
  .table-scroller {{max-height: 520px; overflow-y: auto; overflow-x: auto;}}
  .theme-table {{width:100%; border-collapse: collapse; min-width: 760px;}}
  .theme-table thead th {{
    position: sticky; top: 0; z-index: 2;
    background: {PRIMARY_DEEP};
    color: white;
    text-align: left;
    padding: 0.88rem 0.95rem;
    font-size: 0.88rem;
    white-space: nowrap;
  }}
  .theme-table tbody td {{
    padding: 0.9rem 0.95rem;
    border-bottom: 1px solid #E6F1F0;
    color: {TEXT_DARK};
    vertical-align: top;
  }}
  .theme-table th:first-child,
  .theme-table td.sno-col {{
    text-align: center;
    width: 72px;
    min-width: 72px;
    white-space: nowrap;
  }}
  .theme-table tbody tr:nth-child(odd) {{background: #F8FCFC;}}
  .theme-table tbody tr:hover {{background: #EEF8F7;}}

  .confidence-track {{width: 150px; background:#E6F1F0; border-radius:999px; height:10px; overflow:hidden; margin-bottom:6px;}}
  .confidence-bar {{background: linear-gradient(90deg, {PRIMARY} 0%, {PRIMARY_DEEP} 100%); height:100%; border-radius:999px;}}
  .canon-text {{font-weight:700; color:{PRIMARY_DEEP};}}
  .muted {{color:{TEXT_MUTE}; font-size:0.86rem;}}

  .preview-controls {{display:flex; justify-content:flex-end; gap:0.4rem; margin-bottom:0.45rem;}}
  .preview-icon {{
    width: 34px; height:34px; border-radius: 10px; display:flex; align-items:center; justify-content:center;
    border:1px solid {BORDER}; color:{PRIMARY_DEEP}; background:white; font-size:0.95rem;
  }}

  .stTabs [data-baseweb="tab-list"] {{gap: 0.35rem;}}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 10px 10px 0 0;
    padding: 0.7rem 0.9rem;
    color: {TEXT_MUTE};
    font-weight: 600;
  }}
  .stTabs [aria-selected="true"] {{
    color: {PRIMARY_DEEP} !important;
    border-bottom-color: {PRIMARY} !important;
  }}
  .stTabs [data-baseweb="tab-highlight"] {{background: {PRIMARY} !important; height: 3px !important;}}

  div[data-baseweb="slider"] [role="slider"] {{
    background: {PRIMARY} !important;
    border-color: {PRIMARY} !important;
    box-shadow: none !important;
  }}
  div[data-baseweb="slider"] > div > div:first-child {{background: #D6EAE9 !important;}}
  div[data-baseweb="slider"] > div > div > div {{background: {PRIMARY} !important;}}
  .stSlider p {{color:{PRIMARY_DEEP}; font-weight:600;}}

  .stButton > button, .stDownloadButton > button {{
    border-radius: 12px;
    border: 1px solid {PRIMARY};
    color: white;
    background: linear-gradient(135deg, {PRIMARY_DEEP} 0%, {PRIMARY} 100%);
    box-shadow: 0 6px 16px rgba(11, 122, 117, 0.15);
  }}
  .stButton > button:hover, .stDownloadButton > button:hover {{
    border-color: {PRIMARY_DEEP};
    color: white;
    background: linear-gradient(135deg, {PRIMARY_DEEP} 0%, #117F81 100%);
  }}

  div[data-testid="stAlert"] {{border-radius: 14px; border: 1px solid {BORDER};}}

  /* Hide first two dataframe toolbar buttons when Streamlit still renders them */
  div[data-testid="stDataFrameToolbar"] button:nth-child(1),
  div[data-testid="stDataFrameToolbar"] button:nth-child(2) {{display:none !important;}}

  /* Compact analytics distribution table */
  #dist-table .table-scroller {{
    max-height: 235px !important;
    overflow-y: hidden !important;
    overflow-x: hidden !important;
  }}

  #dist-table .theme-table {{
    min-width: 100% !important;
    width: 100% !important;
  }}

  #dist-table .theme-table thead th,
  #dist-table .theme-table tbody td {{
    padding: 0.7rem 0.75rem !important;
    font-size: 0.95rem !important;
  }}

  #dist-table .theme-table thead th:first-child,
  #dist-table .theme-table tbody td.sno-col {{
    width: 70px !important;
    min-width: 70px !important;
  }}
    /* Compact analytics distribution table */
  #dist-table .table-scroller {{
    max-height: 215px !important;
    overflow: hidden !important;
  }}

  #dist-table .theme-table {{
    min-width: 100% !important;
    width: 100% !important;
  }}

  #dist-table .theme-table thead th {{
    font-size: 0.78rem !important;
    padding: 0.5rem 0.6rem !important;
    white-space: nowrap !important;
  }}

  #dist-table .theme-table tbody td {{
    font-size: 0.82rem !important;
    padding: 0.5rem 0.6rem !important;
    white-space: nowrap !important;
  }}

  #dist-table .theme-table thead th:first-child,
  #dist-table .theme-table tbody td.sno-col {{
    width: 58px !important;
    min-width: 58px !important;
  }}

  #dist-table .theme-table tbody tr {{
    height: 32px !important;
  }}

  /* Only preview table font control */
  #normalized-preview .theme-table thead th {{
    font-size: 0.78rem !important;
    padding: 0.5rem 0.65rem !important;
  }}

  #normalized-preview .theme-table tbody td {{
    font-size: 0.82rem !important;
    padding: 0.5rem 0.65rem !important;
  }}
</style>
""",
    unsafe_allow_html=True,
)

MASTER_PATH = str(Path(__file__).parent / "data" / "master_attributes.json")


@st.cache_resource
def get_engine():
    return NormalizationEngine(MASTER_PATH)


def safe_text(value) -> str:
    if value is None:
        return ""
    return escape(str(value))


def badge_html(match_type: str) -> str:
    meta = MATCH_META.get(match_type, MATCH_META["unmatched"])
    icon = {
        "exact": "✅",
        "synonym": "🔵",
        "fuzzy": "🟠",
        "semantic": "🟣",
        "unmatched": "❌",
    }.get(match_type, "•")
    return (
        f"<span class='badge' style='background:{meta['bg']};color:{meta['fg']};'>"
        f"{icon} {meta['label']}</span>"
    )


def confidence_html(confidence: float) -> str:
    pct = max(0, min(100, int(round(confidence * 100))))
    return (
        f"<div class='confidence-track'><div class='confidence-bar' style='width:{pct}%'></div></div>"
        f"<div class='muted' style='font-size:0.95rem'>{pct}%</div>"
    )


def render_theme_table(df: pd.DataFrame, table_id: str, max_height: int = 520, canonical_cols=None, add_sno: bool = True):
    canonical_cols = canonical_cols or []
    render_df = df.copy()

    if add_sno:
        render_df.insert(0, "S.No", range(1, len(render_df) + 1))

    headers = "".join(f"<th>{safe_text(c)}</th>" for c in render_df.columns)
    rows = []
    for _, row in render_df.iterrows():
        tds = []
        for col in render_df.columns:
            cls = "canon-text" if col in canonical_cols else ""
            if col == "S.No":
                cls = f"{cls} sno-col".strip()
            tds.append(f"<td class='{cls}'>{safe_text(row[col])}</td>")
        rows.append(f"<tr>{''.join(tds)}</tr>")
    body = "".join(rows) if rows else f"<tr><td colspan='{len(render_df.columns)}'>No data available</td></tr>"
    st.markdown(
        f"""
        <div class="table-wrap" id="{table_id}">
          <div class="table-scroller" style="max-height:{max_height}px;">
            <table class="theme-table">
              <thead><tr>{headers}</tr></thead>
              <tbody>{body}</tbody>
            </table>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_match_details_table(results):
    rows = []
    for idx, r in enumerate(results, start=1):
        rows.append(
            "<tr>"
            f"<td class='sno-col'>{idx}</td>"
            f"<td>{safe_text(r.raw_attr)}</td>"
            f"<td class='canon-text'>{safe_text(r.canonical_attr)}</td>"
            f"<td>{badge_html(r.match_type)}</td>"
            f"<td>{confidence_html(r.confidence)}</td>"
            f"<td class='muted'>{safe_text(r.matched_variation or '—')}</td>"
            "</tr>"
        )
        st.markdown(
        """
        <div class="preview-controls" aria-hidden="true">
          <div class="preview-icon" title="Search">🔎</div>
          <div class="preview-icon" title="Expand">⤢</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="table-wrap">
          <div class="table-scroller" style="max-height:620px;">
            <table class="theme-table" style="min-width:1060px;">
              <thead>
                <tr>
                  <th>S.No</th>
                  <th>Raw Attribute</th>
                  <th>Canonical Attribute</th>
                  <th>Match Type</th>
                  <th>Confidence</th>
                  <th>Matched Variation</th>
                </tr>
              </thead>
              <tbody>{''.join(rows)}</tbody>
            </table>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_preview_table(df: pd.DataFrame):
    st.markdown(
        """
        <div class="preview-controls" aria-hidden="true">
          <div class="preview-icon" title="Search">🔎</div>
          <div class="preview-icon" title="Expand">⤢</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_theme_table(df, table_id="normalized-preview", max_height=520, canonical_cols=["Output Attribute"])


def build_distribution_chart(df: pd.DataFrame):
    if df.empty:
        st.info("No analytics available.")
        return
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, color=PRIMARY)
        .encode(
            x=alt.X(
                "Match Type:N",
                sort="-y",
                axis=alt.Axis(
                    title=None,
                    labelColor=TEXT_MUTE,
                    labelAngle=0,
                    labelLimit=200,
                ),
            ),
            y=alt.Y(
                "Count:Q",
                axis=alt.Axis(title=None, gridColor="#E6F1F0", tickColor="#E6F1F0")
            ),
            tooltip=["Match Type", "Count"],
        )
        .properties(height=320)
        .configure_view(strokeOpacity=0)
        .configure_axis(labelFontSize=12)
    )
    st.altair_chart(chart, use_container_width=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Theme Settings")
    fuzzy_threshold = st.slider(
        "Fuzzy Match Threshold",
        50,
        99,
        int(DEFAULT_FUZZY_THRESHOLD),
        help="Minimum RapidFuzz score needed for fuzzy matching.",
    )
    semantic_threshold = st.slider(
        "Semantic Match Threshold",
        0.10,
        0.90,
        float(DEFAULT_SEMANTIC_THRESHOLD),
        0.05,
        help="Minimum cosine similarity needed for semantic matching.",
    )

    st.markdown("---")
    st.markdown("### 📚 Master Attributes")
    with open(MASTER_PATH, encoding="utf-8") as f:
        master = json.load(f)
    st.markdown(
        f"<div class='sidebar-card'>{len(master['master_attributes'])} canonical attributes loaded</div>",
        unsafe_allow_html=True,
    )
    with st.expander("View Master List"):
        for entry in master["master_attributes"]:
            st.markdown(f"**{entry['canonical']}**")
            variations = entry.get("variations", [])
            if variations:
                st.caption(", ".join(variations[:5]))

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
      <h1>Document Attribute Normalization System</h1>
      <p>Upload PDF, Excel, or CSV files and map raw attribute names into clean canonical master attributes.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">📂 Upload Document</div>', unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Upload a document to get started",
    type=["pdf", "xlsx", "xls", "csv"],
    help="Supports PDF, XLSX, XLS, and CSV.",
)

if uploaded:
    engine = get_engine()
    with st.spinner("Extracting attributes and applying normalization..."):
        try:
            report: NormalizationReport = engine.process(
                io.BytesIO(uploaded.read()),
                uploaded.name,
                fuzzy_threshold=float(fuzzy_threshold),
                semantic_threshold=float(semantic_threshold),
            )
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.stop()

    match_counts = {}
    for r in report.match_details:
        match_counts[r.match_type] = match_counts.get(r.match_type, 0) + 1

    st.markdown('<div class="section-title">📊 Normalization Summary</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    exact_syn = match_counts.get("exact", 0) + match_counts.get("synonym", 0)
    fuzzy_sem = match_counts.get("fuzzy", 0) + match_counts.get("semantic", 0)
    cards = [
        (c1, report.total_attributes, "Total Attributes"),
        (c2, report.matched, "Matched"),
        (c3, report.unmatched, "Unmatched"),
        (c4, exact_syn, "Exact / Synonym"),
        (c5, fuzzy_sem, "Fuzzy / Semantic"),
    ]
    for col, value, label in cards:
        with col:
            st.markdown(
                f"<div class='metric-card'><div class='val'>{value}</div><div class='lbl'>{label}</div></div>",
                unsafe_allow_html=True,
            )

    unique_results = []
    seen = set()
    for r in report.match_details:
        if r.raw_attr not in seen:
            seen.add(r.raw_attr)
            unique_results.append(r)

    tab_preview, tab_debug, tab_analytics, tab_download = st.tabs(
        ["📋 Normalized Preview", "🔎 Match Details", "📈 Analytics", "⬇️ Download"]
    )

    with tab_preview:
        st.markdown('<div class="section-title">Normalized Output Preview</div>', unsafe_allow_html=True)
        st.caption("Final business output with normalized attribute names and original values.")
        preview_rows = [
            {"Output Attribute": rec.get("attribute", ""), "Value": rec.get("value", "")}
            for rec in report.normalized_records
            if str(rec.get("attribute", "")).strip() or str(rec.get("value", "")).strip()
        ]
        if preview_rows:
            preview_df = pd.DataFrame(preview_rows)
            render_preview_table(preview_df)
        else:
            st.info("No attributes found in the document.")

    with tab_debug:
        st.markdown('<div class="section-title">Attribute Matching Details</div>', unsafe_allow_html=True)
        st.caption("Technical matching diagnostics including match type, confidence, and matched variation.")
        render_match_details_table(unique_results)
        unmatched = [r for r in unique_results if r.match_type == "unmatched"]
        if unmatched:
            st.markdown('<div class="section-title">⚠️ Unmatched Attributes</div>', unsafe_allow_html=True)
            for r in unmatched:
                st.warning(f"{r.raw_attr} has no confident canonical match yet. Add it to the master attribute list if needed.")

    with tab_analytics:
        st.markdown('<div class="section-title">Match Type Distribution</div>', unsafe_allow_html=True)
        dist_df = pd.DataFrame(
            [{"Match Type": m.title(), "Count": c} for m, c in match_counts.items()]
        ).sort_values("Count", ascending=False)
        left, right = st.columns([1, 2])
        with left:
            render_theme_table(dist_df, table_id="dist-table", max_height=215)
        with right:
            build_distribution_chart(dist_df)

        st.markdown('<div class="section-title">Confidence Score Distribution</div>', unsafe_allow_html=True)
        conf_df = pd.DataFrame(
            {
                "Attribute": [r.raw_attr for r in unique_results],
                "Confidence (%)": [round(r.confidence * 100, 1) for r in unique_results],
                "Match Type": [r.match_type.title() for r in unique_results],
            }
        )
        render_theme_table(conf_df, table_id="conf-table", max_height=420)

    with tab_download:
        st.markdown('<div class="section-title">Download Normalized Output</div>', unsafe_allow_html=True)
        st.success(
            f"Output format: {report.output_ext.upper()} — same as input type, with normalized attribute names and original values."
        )
        ext_mime = {
            "pdf": "application/pdf",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "csv": "text/csv",
        }
        out_name = Path(uploaded.name).stem + "_normalized." + report.output_ext
        st.download_button(
            label=f"Download {out_name}",
            data=report.output_bytes,
            file_name=out_name,
            mime=ext_mime[report.output_ext],
            use_container_width=True,
        )
        info1, info2 = st.columns(2)
        with info1:
            st.info(
                f"**Input:** {uploaded.name}\n\n**Format:** {report.input_format.upper()}\n\n**Structure:** {report.doc_type}"
            )
        with info2:
            rate = (report.matched / report.total_attributes * 100) if report.total_attributes else 0
            st.info(
                f"**Attributes normalized:** {report.matched}/{report.total_attributes}\n\n**Match rate:** {rate:.0f}%\n\n**Output file:** {out_name}"
            )
else:
    st.markdown(
        f"""
        <div class="table-wrap" style="background:{PRIMARY_LIGHTER}; border-style:dashed; border-width:2px;">
          <div style="text-align:center; padding:3rem 1.5rem;">
            <div style="font-size:4rem; line-height:1;">📁</div>
            <h2 style="margin:0.9rem 0 0.4rem; color:{PRIMARY_DEEP};">Upload a document to get started</h2>
            <p style="max-width:620px; margin:0 auto; color:{TEXT_MUTE}; font-size:1.05rem;">
              Supports PDF, Excel (.xlsx/.xls), and CSV files. Raw attribute names are mapped to canonical master attributes using exact, synonym, fuzzy, and semantic matching.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )