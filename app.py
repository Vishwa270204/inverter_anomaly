"""
Inverter Anomaly Detection -- Streamlit Dashboard
====================================================
Production frontend only. All ML/training happens in inverter_anomaly.ipynb.

Reads:
    dashboard_data.parquet  -> evaluation-period observations + model output

Does NOT retrain or re-run the notebook. Does NOT treat anomaly_score_ratio
as a probability. Feature contributions are reported as "contributed most
to reconstruction error," never as a proven cause.
"""

import html
import json
import os
import re
import textwrap
from datetime import datetime, timedelta
from data_validation import validate_event_data, summarize as summarize_data
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Inverter Anomaly Detection",
    page_icon="⚡",
    layout="wide",
)

# Professional, compact theme. Filters live in the sidebar and content is
# split into tabs so the page fits the screen instead of scrolling forever.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Keep the dashboard compact, but do not make the text look tiny. */
    html { font-size: 16px; }

    :root {
        --primary: #0F3554;
        --primary-light: #164A73;
        --warn: #DC6803;
        --danger: #E4463F;
        --border: #E4E9F0;
        --muted: #64748B;
        --surface: #FFFFFF;
        --bg: #F5F7FA;
    }

    .stApp { background-color: var(--bg); }

    /* Fit the whole app to the viewport width and cut wasted vertical
       space so pages need far less scrolling. */
    /* Streamlit's fixed top toolbar (Share/star/menu icons) sits above our
       content. Shrink it and give the block-container just enough top
       clearance to sit below it instead of being hidden underneath it. */
    header[data-testid="stHeader"] {
        height: 2.4rem;
        background: transparent;
    }
    .block-container {
        padding-top: 2.6rem;
        padding-bottom: 0.5rem;
        padding-left: 1.5rem;
        padding-right: 1.5rem;
        max-width: 100%;
    }

    /* Tighten the default gap Streamlit puts between stacked elements */
    [data-testid="stVerticalBlock"] { gap: 0.35rem; }
    div[data-testid="stElementContainer"] { margin-bottom: 0 !important; }

    /* Date inputs / selects / checkboxes: shrink their control height */
    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        min-height: 2rem !important;
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
    }
    [data-testid="stWidgetLabel"] p { margin-bottom: 0.1rem !important; }

    /* ---------- Metric / KPI cards ---------- */
    [data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }
    [data-testid="stMetricValue"] {
        font-size: 1.25rem;
        font-weight: 700;
        color: #0F172A;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.74rem;
        font-weight: 600;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }
    .baseline-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 10px 0 20px 0;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}

.baseline-title {
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 4px;
}

.baseline-subtitle {
    font-size: 13px;
    color: #64748b;
    margin-bottom: 12px;
}

.baseline-values {
    font-size: 15px;
    font-weight: 600;
}
    /* ---------- Headings ---------- */
    h2, [data-testid="stMarkdownContainer"] h2 {
        color: #0F172A;
        font-size: 1.55rem !important;
        font-weight: 700;
        margin-top: 1.1rem !important;
        margin-bottom: 0.35rem !important;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--border);
    }
    h3, [data-testid="stMarkdownContainer"] h3 {
        color: #0F172A;
        font-size: 1.22rem !important;
        font-weight: 700;
        margin-top: 0.7rem !important;
        margin-bottom: 0.25rem !important;
    }
    h4 {
        color: #1E293B;
        font-size: 1.05rem !important;
        font-weight: 600;
        margin-top: 0 !important;
        margin-bottom: 0.2rem !important;
    }

    hr { border-color: var(--border) !important; margin: 0.5rem 0 !important; }

    /* ---------- Captions ---------- */
    [data-testid="stCaptionContainer"], .stCaption {
        margin-bottom: 0.2rem !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid var(--border);
    }
    .stButton > button[kind="primary"] {
        background-color: var(--primary);
        border-color: var(--primary);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #0B2740;
        border-color: #0B2740;
    }

    /* ---------- Expanders / bordered containers ---------- */
    .streamlit-expanderHeader { font-weight: 600; border-radius: 8px; }
    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--surface);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px !important;
        padding: 0.5rem 0.9rem !important;
    }
    /* Bordered containers (e.g. the filter bar) get compact inner padding */
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
        gap: 0.3rem;
    }

    /* ---------- Dataframes ---------- */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    /* ---------- Form labels ---------- */
    label, .stSelectbox label, .stDateInput label {
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        color: #334155 !important;
    }

    /* ---------- AI Explanation ---------- */
    .ai-title {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.3;
        margin-top: 0.25rem;
        margin-bottom: 0.15rem;
    }

    .ai-button-spacer { min-height: 2.7rem; }

    .ai-subtitle {
        color: #64748B;
        font-size: 1rem;
        line-height: 1.45;
        margin-bottom: 0.65rem;
    }

    .ai-card {
        width: 100%;
        box-sizing: border-box;
        background: #FFFFFF;
        border: 1px solid #D9E1EA;
        border-radius: 12px;
        padding: 1.35rem 1.5rem;
        box-shadow: 0 2px 7px rgba(15, 23, 42, 0.05);
    }

    .ai-row {
        padding: 0.15rem 0 0.9rem 0;
    }

    .ai-row:last-child {
        padding-bottom: 0;
    }

    .ai-label {
        color: #0F3554;
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.35;
        margin-bottom: 0.28rem;
    }

    .ai-body {
        color: #1E293B;
        font-size: 1.06rem;
        line-height: 1.7;
        word-break: normal;
        overflow-wrap: anywhere;
    }

    .ai-body strong {
        font-weight: 700;
        color: #0F3554;
    }

    .ai-checks {
        margin: 0.15rem 0 0 1.35rem;
        padding: 0;
    }

    .ai-checks li {
        color: #1E293B;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 0.28rem;
        padding-left: 0.15rem;
    }

    .ai-generating {
        width: 100%;
        box-sizing: border-box;
        background: #F7F9FC;
        border: 1px solid #D9E1EA;
        border-radius: 12px;
        padding: 1.05rem 1.25rem;
        min-height: 72px;
        display: flex;
        align-items: center;
    }

    .ai-generating span {
        color: #475569;
        font-size: 1rem;
        line-height: 1.5;
    }

    /* Give the AI action row a little more breathing room. */
    .ai-action-row {
        margin-top: 0.1rem;
        margin-bottom: 0.65rem;
    }

    /* ---------- AI Explanation: structured card ---------- */
    .ai-alert-box {
        display: flex;
        gap: 0.65rem;
        background: #FEF3F2;
        border: 1px solid #FECDCA;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }
    .ai-alert-icon { font-size: 1.3rem; line-height: 1.4; flex-shrink: 0; }
    .ai-alert-title { font-weight: 700; color: #B42318; font-size: 1.02rem; margin-bottom: 0.15rem; }
    .ai-alert-desc { color: #7A271A; font-size: 0.95rem; line-height: 1.5; }

    .ai-section { margin-top: 1.1rem; }
    .ai-section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 700;
        color: #0F172A;
        font-size: 1rem;
        margin-bottom: 0.45rem;
    }
    .ai-section-icon { font-size: 1.05rem; }

    .ai-bullets { margin: 0; padding-left: 1.3rem; }
    .ai-bullets li { color: #1E293B; font-size: 0.96rem; line-height: 1.55; margin-bottom: 0.3rem; }

    .ai-meta-row { display: flex; gap: 2.5rem; flex-wrap: wrap; }
    .ai-meta-label { font-size: 0.82rem; color: #64748B; font-weight: 600; }
    .ai-meta-value { font-size: 0.98rem; color: #0F172A; font-weight: 600; }

    .ai-footer-note {
        display: flex;
        gap: 0.6rem;
        background: #F0F6FF;
        border: 1px solid #D6E4FA;
        border-radius: 10px;
        padding: 0.75rem 0.9rem;
        margin-top: 1.1rem;
        font-size: 0.85rem;
        color: #375273;
        line-height: 1.5;
    }

    /* ---------- Selected event / validation layout ---------- */
    .event-summary-card { background:#FFFFFF; border:1px solid #D9E1EA; border-radius:12px; padding:1rem 1.1rem; box-shadow:0 1px 4px rgba(15,23,42,0.04); margin-bottom:0.7rem; }
    .event-summary-title { color:#0F172A; font-size:1.15rem; font-weight:700; margin-bottom:0.7rem; }
    .event-badge { display:inline-block; padding:0.22rem 0.55rem; border-radius:999px; background:#FFF1F2; color:#C0262D; border:1px solid #FECDD3; font-size:0.78rem; font-weight:600; margin-left:0.35rem; }
    .validation-card { background:#FFFFFF; border:1px solid #D9E1EA; border-radius:12px; padding:1rem 1.1rem; margin-top:0.75rem; box-shadow:0 1px 4px rgba(15,23,42,0.04); }
    .validation-header { display:flex; justify-content:space-between; align-items:center; gap:0.5rem; margin-bottom:0.3rem; }
    .validation-title { color:#0F172A; font-size:1.15rem; font-weight:700; }
    .validation-subtitle { color:#64748B; font-size:0.82rem; line-height:1.45; margin-bottom:0.75rem; }
    .validation-pill { border-radius:999px; padding:0.3rem 0.65rem; font-size:0.78rem; font-weight:700; white-space:nowrap; }
    .validation-pass { background:#ECFDF3; color:#027A48; border:1px solid #ABEFC6; }
    .validation-warn { background:#FFFAEB; color:#B54708; border:1px solid #FEDF89; }
    .validation-fail { background:#FEF3F2; color:#B42318; border:1px solid #FECDCA; }
    .check-row { display:flex; align-items:flex-start; gap:0.65rem; padding:0.55rem 0; border-top:1px solid #EEF2F6; }
    .check-icon { width:1.25rem; height:1.25rem; border-radius:50%; display:inline-flex; align-items:center; justify-content:center; flex:0 0 1.25rem; font-size:0.72rem; font-weight:700; }
    .check-pass { background:#D1FADF; color:#027A48; }
    .check-warn { background:#FEF0C7; color:#B54708; }
    .check-fail { background:#FEE4E2; color:#B42318; }
    .check-name { color:#1E293B; font-size:0.88rem; font-weight:600; line-height:1.35; }
    .check-detail { color:#64748B; font-size:0.76rem; line-height:1.35; margin-top:0.12rem; }
    .validation-note { margin-top:0.65rem; padding:0.65rem 0.75rem; border-radius:8px; background:#F8FAFC; border:1px solid #E2E8F0; color:#475569; font-size:0.78rem; line-height:1.45; }

    </style>
    """,
    unsafe_allow_html=True,
)


st.session_state.setdefault("ai_explanations", {})

# One-time migration: the explanation format changed from a plain paragraph
# string to a structured dict (headline / summary / why / actions). Drop any
# old-format cached entries so they regenerate instead of breaking the new
# renderer or being mistaken for the new shape.
_stale_keys = [
    k for k, v in st.session_state["ai_explanations"].items()
    if isinstance(v, dict) and isinstance(v.get("explanation"), str)
]
for _k in _stale_keys:
    del st.session_state["ai_explanations"][_k]

# ============================================================
# DATA LOADING (cached -- parquet is read once per session)
# ============================================================


@st.cache_data
def load_dashboard_data(path="dashboard_data.parquet"):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    if "anomaly_flag" in df.columns:
        df["anomaly_flag"] = df["anomaly_flag"].fillna(False).astype(bool)
    else:
        df["anomaly_flag"] = False
    return df

@st.cache_data
def load_trend_data(path="trend_data.parquet"):
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df

@st.cache_data
def load_dashboard_baseline(path="dashboard_baseline.parquet"):
    baseline = pd.read_parquet(path)

    numeric_cols = [
        c for c in baseline.columns
        if c.endswith("_median")
        or c.endswith("_q10")
        or c.endswith("_q90")
    ]

    for col in numeric_cols:
        baseline[col] = pd.to_numeric(
            baseline[col],
            errors="coerce"
        )

    if "healthy_sample_count" in baseline.columns:
        baseline["healthy_sample_count"] = pd.to_numeric(
            baseline["healthy_sample_count"],
            errors="coerce"
        )

    return baseline
def safe_load(loader, path, label):
    try:
        return loader(path)
    except FileNotFoundError:
        st.error(
            f"**{label} not found** (`{path}`). Run `inverter_anomaly.ipynb` "
            "to generate it, then place it next to `app.py`."
        )
        st.stop()
    except Exception as e:
        st.error(f"Failed to load **{label}** (`{path}`): {e}")
        st.stop()

df = safe_load(load_dashboard_data, "dashboard_data.parquet", "Dashboard data")
persistence_config = load_persistence_config(
    "persistence_config.json"
)


baseline_df = safe_load(
    load_dashboard_baseline,
    "dashboard_baseline.parquet",
    "Dashboard healthy baseline",
)
if df.empty:
    st.error("`dashboard_data.parquet` loaded but contains no rows.")
    st.stop()

# trend_data.parquet is optional: it adds pre-evaluation history for the
# Trends tab and for the 24-hour "before an anomaly" context. If it hasn't
# been generated yet (see notebook Section 19), fall back to using the
# dashboard data alone instead of crashing the whole app.
try:
    trend_df = load_trend_data("trend_data.parquet")
    has_trend_data = True
except Exception:
    trend_df = df
    has_trend_data = False


# ============================================================
# SMALL HELPERS
# ============================================================


def fmt_num(value, decimals=2, suffix="", dash="—"):
    """Format a possibly-missing numeric value; never raises on NaN/None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return dash
    return f"{value:,.{decimals}f}{suffix}"
def fmt_time(value, dash="—"):
    if value is None or pd.isna(value):
        return dash
    return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")

def get_trend_window(source, end_time, hours_back=24):
    """Slice trend_data.parquet (or dashboard_data.parquet as fallback) to
    [end_time - hours_back, end_time]."""
    end_time = pd.to_datetime(end_time)
    start_time = end_time - timedelta(hours=hours_back)
    window = source[(source["timestamp"] >= start_time) & (source["timestamp"] <= end_time)].copy()
    return window, start_time


def get_contribution_cols(frame):
    """Feature-contribution columns exported from the notebook (Section 14),
    named '<feature>_contribution_pct'."""
    return [c for c in frame.columns if c.endswith("_contribution_pct")]


@st.cache_resource
def get_groq_client():
    """Create one Groq client per Streamlit session/process."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["groq"]["api_key"]
        except Exception:
            api_key = None
    if not api_key:
        return None
    return Groq(api_key=api_key)


# Controlled backend tools used by the AI analyst. The LLM can request
# evidence, but it never gets arbitrary Python/database access.
def clean_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def row_to_dict(row):
    return {str(k): clean_value(v) for k, v in row.items()}


def get_anomaly_details(timestamp):
    target = pd.to_datetime(timestamp)
    source = df.copy()
    if source.empty:
        return {"error": "No inverter data available."}
    idx = (
        source["timestamp"] - target
    ).abs().idxmin()
    return row_to_dict(source.loc[idx])


def get_pre_anomaly_trend(timestamp, hours=24):
    target = pd.to_datetime(timestamp)

    start = target - timedelta(hours=float(hours))

    source = trend_df[
        (trend_df["timestamp"] >= start) &
        (trend_df["timestamp"] <= target)
    ].copy()

    if source.empty:
        return {
            "error": "No trend observations found for the requested window."
        }

    numeric = [
        c
        for c in [
            "ac_power_kw",
            "dc_power_kw",
            "dc_current_a",
            "ac_current_a",
            "inverter_temperature_c",
            "ambient_temperature_c",
            "inverter_ambient_temp_delta",
            "efficiency_pct",
            "power_factor",
        ]
        if c in source.columns
    ]

    stats = {}

    for col in numeric:

        series = pd.to_numeric(
            source[col],
            errors="coerce"
        ).dropna()

        if len(series) >= 2:

            stats[col] = {
                "start": clean_value(series.iloc[0]),
                "end": clean_value(series.iloc[-1]),
                "net_change": clean_value(
                    series.iloc[-1] - series.iloc[0]
                ),
                "min": clean_value(series.min()),
                "max": clean_value(series.max()),
                "median": clean_value(series.median()),
            }

    return {
        "window_start": clean_value(
            source["timestamp"].min()
        ),
        "window_end": clean_value(
            source["timestamp"].max()
        ),
        "observations": int(len(source)),
        "statistics": stats,
    }

def get_feature_contributions(timestamp, inverter_id=None):
    target = pd.to_datetime(timestamp)
    source = df.copy()
    if inverter_id is not None and "inverter_id" in source.columns:
        source = source[source["inverter_id"].astype(str) == str(inverter_id)]
    if source.empty:
        return {"error": "No matching data found."}
    idx = (source["timestamp"] - target).abs().idxmin()
    row = source.loc[idx]
    cols = [c for c in source.columns if c.endswith("_contribution_pct")]
    values = [
        {"feature": c.replace("_contribution_pct", ""), "contribution_pct": clean_value(row.get(c))}
        for c in cols
        if pd.notna(row.get(c))
    ]
    values.sort(
        key=lambda x: x["contribution_pct"] if x["contribution_pct"] is not None else -1,
        reverse=True,
    )
    return {
        "timestamp": clean_value(row.get("timestamp")),
        "top_contributing_feature": clean_value(row.get("top_contributing_feature")),
        "contributions": values,
    }


def get_operating_context(timestamp):
    target = pd.to_datetime(timestamp)

    source = df.copy()

    if source.empty:
        return {"error": "No inverter data available."}

    idx = (
        source["timestamp"] - target
    ).abs().idxmin()

    row = source.loc[idx]

    wanted = [
        "timestamp",
        "hour",
        "minute",
        "month",
        "is_daylight",
        "inverter_status",
        "dc_power_kw",
        "ac_power_kw",
        "dc_current_a",
        "ac_current_a",
        "power_factor",
        "frequency_hz",
        "efficiency_pct",
        "inverter_temperature_c",
        "ambient_temperature_c",
        "poa_w_m2",
        "ghi_w_m2",
        "quality_code",
        "communication_status",
        "fault_code",
        "alarm_code",
    ]

    return {
        k: clean_value(row.get(k))
        for k in wanted
        if k in source.columns
    }
def get_baseline_context(timestamp):
    """
    Retrieve healthy reference values for conditions similar to
    the selected event.

    This is a comparison reference only. It does not establish cause.
    """

    target = pd.to_datetime(timestamp)

    source = df.copy()

    if source.empty:
        return {
            "error": "No inverter data available."
        }

    idx = (
        source["timestamp"] - target
    ).abs().idxmin()

    row = source.loc[idx]

    conditions = {
        "inverter_status": clean_value(
            row.get("inverter_status")
        ),
        "is_daylight": clean_value(
            row.get("is_daylight")
        ),
        "hour": clean_value(
            row.get("hour")
        ),
        "month": clean_value(
            row.get("month")
        ),
    }

    baseline = baseline_df.copy()

    match_cols = [
        "inverter_status",
        "is_daylight",
        "hour",
        "month",
    ]

    match_cols = [
        c
        for c in match_cols
        if c in baseline.columns
        and conditions.get(c) is not None
    ]

    matched = baseline.copy()

    for col in match_cols:

        matched = matched[
            matched[col].astype(str)
            == str(conditions[col])
        ]

    if matched.empty:

        return {
            "error": (
                "No healthy baseline group is available "
                "for the selected operating conditions."
            ),
            "conditions": conditions,
        }

    result = {
        "conditions": conditions,
        "healthy_sample_count": int(
            matched["healthy_sample_count"].sum()
        ),
        "reference": {},
    }

    metric_names = sorted({
        col[:-7]
        for col in matched.columns
        if col.endswith("_median")
    })

    for metric in metric_names:

        median_col = f"{metric}_median"
        q10_col = f"{metric}_q10"
        q90_col = f"{metric}_q90"

        values = matched[
            [median_col, q10_col, q90_col]
        ].dropna(how="all")

        if values.empty:
            continue

        result["reference"][metric] = {
            "median": clean_value(
                values[median_col].iloc[0]
            ),
            "typical_low_q10": clean_value(
                values[q10_col].iloc[0]
            ),
            "typical_high_q90": clean_value(
                values[q90_col].iloc[0]
            ),
        }

    return result


def generate_ai_explanation(selected_event, event_rows):
    """
    Generate one structured explanation for a complete anomaly event.

    One event = continuous occurrence of anomaly observations.
    Python collects the event evidence first. Groq is used only
    to convert that evidence into a structured JSON explanation.

    Returns (explanation_dict, evidence_dict).
    Always either returns that 2-tuple, or raises. Never returns None.
    """

    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set GROQ_API_KEY in the environment "
            "or Streamlit secrets."
        )

    # ------------------------------------------------------------
    # 1. EVENT INFORMATION
    # ------------------------------------------------------------

    event_start = clean_value(selected_event.get("start_time"))
    event_end = clean_value(selected_event.get("end_time"))
    event_duration = clean_value(selected_event.get("duration_min"))
    anomaly_count = clean_value(selected_event.get("anomaly_count"))

    # ------------------------------------------------------------
    # 2. COLLECT EVENT EVIDENCE
    # ------------------------------------------------------------

    evidence = {
        "event": {
            "event_id": clean_value(selected_event.get("event_id")),
            "start_time": event_start,
            "end_time": event_end,
            "duration_min": event_duration,
            "anomaly_count": anomaly_count,
            "max_severity": clean_value(
                selected_event.get("max_severity")
            ),
            "mean_severity": clean_value(
                selected_event.get("mean_severity")
            ),
            "dominant_status": clean_value(
                selected_event.get("dominant_status")
            ),
            "anomaly_type": clean_value(
                selected_event.get("anomaly_type")
            ),
        }
    }

    # ------------------------------------------------------------
    # 3. EVENT OBSERVATIONS
    # ------------------------------------------------------------

    if event_rows is not None and not event_rows.empty:

        numeric_cols = [
            "dc_power_kw",
            "dc_current_a",
            "ac_power_kw",
            "ac_current_a",
            "power_factor",
            "frequency_hz",
            "efficiency_pct",
            "inverter_temperature_c",
            "ambient_temperature_c",
            "poa_w_m2",
            "ghi_w_m2",
            "packet_loss_pct",
            "communication_latency_ms",
        ]

        event_statistics = {}

        for col in numeric_cols:

            if col not in event_rows.columns:
                continue

            series = pd.to_numeric(
                event_rows[col],
                errors="coerce"
            ).dropna()

            if len(series) == 0:
                continue

            event_statistics[col] = {
                "start": clean_value(series.iloc[0]),
                "end": clean_value(series.iloc[-1]),
                "min": clean_value(series.min()),
                "max": clean_value(series.max()),
                "mean": clean_value(series.mean()),
                "median": clean_value(series.median()),
            }

        evidence["event_observations"] = {
            "observation_count": int(len(event_rows)),
            "statistics": event_statistics,
        }

        # --------------------------------------------------------
        # 4. OPERATING CONDITIONS DURING EVENT
        # --------------------------------------------------------

        context_columns = [
            "inverter_status",
            "is_daylight",
            "hour",
            "month",
            "quality_code",
            "communication_status",
            "fault_code",
            "alarm_code",
        ]

        operating_context = {}

        for col in context_columns:

            if col not in event_rows.columns:
                continue

            values = (
                event_rows[col]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if values:
                operating_context[col] = values

        evidence["operating_context"] = operating_context

        # --------------------------------------------------------
        # 5. FEATURE CONTRIBUTIONS
        # --------------------------------------------------------

        contribution_cols = get_contribution_cols(event_rows)

        if contribution_cols:

            contribution_values = (
                event_rows[contribution_cols]
                .apply(pd.to_numeric, errors="coerce")
                .mean()
                .dropna()
                .sort_values(ascending=False)
            )

            evidence["feature_contributions"] = [
                {
                    "feature": col.replace(
                        "_contribution_pct",
                        ""
                    ),
                    "mean_contribution_pct": clean_value(value),
                }
                for col, value in contribution_values.head(5).items()
            ]

        # --------------------------------------------------------
        # 6. PRE-EVENT TREND
        # --------------------------------------------------------

        trend, _ = get_trend_window(
            trend_df,
            pd.to_datetime(event_start),
            hours_back=24,
        )

        if not trend.empty:

            trend_statistics = {}

            trend_columns = [
                "dc_power_kw",
                "ac_power_kw",
                "dc_current_a",
                "ac_current_a",
                "inverter_temperature_c",
                "efficiency_pct",
                "power_factor",
            ]

            for col in trend_columns:

                if col not in trend.columns:
                    continue

                series = pd.to_numeric(
                    trend[col],
                    errors="coerce"
                ).dropna()

                if len(series) >= 2:

                    trend_statistics[col] = {
                        "start": clean_value(series.iloc[0]),
                        "end": clean_value(series.iloc[-1]),
                        "min": clean_value(series.min()),
                        "max": clean_value(series.max()),
                        "median": clean_value(series.median()),
                    }

            evidence["pre_event_trend"] = {
                "window_start": clean_value(
                    trend["timestamp"].min()
                ),
                "window_end": clean_value(
                    trend["timestamp"].max()
                ),
                "observations": int(len(trend)),
                "statistics": trend_statistics,
            }

    # ------------------------------------------------------------
    # 7. HEALTHY BASELINE
    # ------------------------------------------------------------

    try:

        if event_rows is not None and not event_rows.empty:

            reference_row = event_rows.iloc[0]

            baseline = get_baseline_context(
                reference_row["timestamp"]
            )

            evidence["healthy_baseline"] = baseline

    except Exception as e:

        evidence["healthy_baseline_error"] = str(e)

    # ------------------------------------------------------------
    # 8. AI PROMPT
    # ------------------------------------------------------------

    prompt = """
        You are an explanation assistant inside a solar inverter anomaly detection dashboard.
        Respond with ONLY a single JSON object — no markdown fences, no extra text.

        JSON shape (all fields required, all values are plain strings unless noted):
        {
          "headline": "One short sentence naming what happened, simple language, no ML terms.",
          "summary": "1-2 plain sentences expanding on the headline (what changed, roughly how much).",
          "why_it_happened": ["bullet 1", "bullet 2", "bullet 3"],
          "recommended_actions": ["bullet 1", "bullet 2", "bullet 3"]
        }

        IMPORTANT RULES:
        - Use simple words. Avoid technical/ML terms (no autoencoder, reconstruction error,
          threshold, anomaly score, probability, confidence, etc).
        - Explain as if talking to a plant operator, not a data scientist.
        - Use ONLY the supplied evidence. Never invent measurements, causes, or events.
        - Do not automatically call the anomaly a fault.
        - Feature contributions show which parameters were unusual; they do NOT prove root cause.
        - If the exact cause cannot be determined, say so plainly in "summary" or a bullet.
        - "why_it_happened": 2-4 short bullets, each a plain-language fact from the evidence
          (e.g. a parameter that moved outside its healthy range, or an unusual status/code).
        - "recommended_actions": 2-4 short, concrete, checkable next steps for an operator.
        - Keep every string free of markdown bold/asterisks — plain text only.

        EVENT EVIDENCE:
    """ + json.dumps(
            evidence,
            default=str,
            ensure_ascii=False
        )

    # ------------------------------------------------------------
    # 9. GROQ REQUEST
    # ------------------------------------------------------------
    # Two layers of defense against malformed JSON from the model:
    #   1. response_format={"type": "json_object"} asks Groq to constrain
    #      generation to valid JSON in the first place.
    #   2. If parsing still fails (rare -- e.g. a stray unescaped quote
    #      inside a bullet string), send the broken output back once with
    #      the exact parser error and ask the model to fix it, instead of
    #      failing the whole explanation.

    def _call_groq(messages):
        try:
            response = client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=messages,
                reasoning_effort="low",
                include_reasoning=False,
                temperature=0.2,
                max_completion_tokens=768,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            raise RuntimeError(f"Groq request failed: {e}") from e

        content = getattr(response.choices[0].message, "content", None)
        if content is None:
            raise RuntimeError("Groq returned no text content.")

        content = str(content).strip()
        # Strip accidental ```json fences some models add despite instructions.
        content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content).strip()

        if not content:
            raise RuntimeError("Groq returned an empty explanation.")

        return content

    base_messages = [{"role": "user", "content": prompt}]
    raw_content = _call_groq(base_messages)

    try:
        parsed = json.loads(raw_content)
    except Exception as first_error:
        # Repair attempt: show the model its own broken output and the
        # exact parser error, and ask for a corrected JSON object only.
        repair_messages = base_messages + [
            {"role": "assistant", "content": raw_content},
            {
                "role": "user",
                "content": (
                    "That was not valid JSON. The parser error was: "
                    f"{first_error}. Return ONLY a corrected, valid JSON "
                    "object with the same fields (headline, summary, "
                    "why_it_happened, recommended_actions). No markdown "
                    "fences, no explanation, just the JSON object."
                ),
            },
        ]
        try:
            raw_content = _call_groq(repair_messages)
            parsed = json.loads(raw_content)
        except Exception as second_error:
            raise RuntimeError(
                f"Groq did not return valid JSON after a retry: {second_error}"
            ) from second_error

    if not isinstance(parsed, dict):
        raise RuntimeError("Groq JSON response was not an object.")

    required = ["headline", "summary", "why_it_happened", "recommended_actions"]
    missing = [k for k in required if k not in parsed]
    if missing:
        raise RuntimeError(f"Groq JSON is missing required fields: {missing}")

    # "When it occurred" is computed from data Python already trusts --
    # never taken from the LLM.
    parsed["when_time"] = fmt_time(event_start)
    parsed["when_duration"] = (
        fmt_num(event_duration, 0, " min") if event_duration is not None else "—"
    )

    return parsed, evidence


def render_ai_explanation(data):
    """Render the structured LLM explanation as icon-header sections
    (alert banner, Why it happened, When it occurred, Recommended action)."""
    if not data:
        return

    # Backward compatibility: older cached entries (before the structured
    # JSON format) stored a plain paragraph string instead of a dict.
    if isinstance(data, str):
        text = re.sub(r"\s+", " ", data).strip()
        safe = html.escape(text)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        st.markdown(
            f'<div class="ai-card"><div class="ai-body">{safe}</div></div>',
            unsafe_allow_html=True,
        )
        return

    def esc(s):
        return html.escape(str(s)) if s is not None else ""

    headline = esc(data.get("headline", ""))
    summary = esc(data.get("summary", ""))
    why_bullets = [esc(b) for b in data.get("why_it_happened", []) if b]
    action_bullets = [esc(b) for b in data.get("recommended_actions", []) if b]
    when_time = esc(data.get("when_time", "—"))
    when_duration = esc(data.get("when_duration", "—"))

    why_html = "".join(f"<li>{b}</li>" for b in why_bullets)
    action_html = "".join(f"<li>{b}</li>" for b in action_bullets)

    html_block = textwrap.dedent(f'''
        <div class="ai-card">
            <div class="ai-alert-box">
                <div class="ai-alert-icon">⚠️</div>
                <div>
                    <div class="ai-alert-title">{headline}</div>
                    <div class="ai-alert-desc">{summary}</div>
                </div>
            </div>

            <div class="ai-section">
                <div class="ai-section-header">
                    <span class="ai-section-icon">🔍</span> Why it happened?
                </div>
                <ul class="ai-bullets">{why_html}</ul>
            </div>

            <div class="ai-section">
                <div class="ai-section-header">
                    <span class="ai-section-icon">🕐</span> When it occurred?
                </div>
                <div class="ai-meta-row">
                    <div>
                        <div class="ai-meta-label">Time</div>
                        <div class="ai-meta-value">{when_time}</div>
                    </div>
                    <div>
                        <div class="ai-meta-label">Duration</div>
                        <div class="ai-meta-value">~{when_duration}</div>
                    </div>
                </div>
            </div>

            <div class="ai-section">
                <div class="ai-section-header">
                    <span class="ai-section-icon">🔧</span> Recommended action
                </div>
                <ul class="ai-bullets">{action_html}</ul>
            </div>

            <div class="ai-footer-note">
                <span>ℹ️</span>
                <span>This explanation is based on the observed patterns in your data
                and the trained anomaly detection model. It does not confirm a fault
                but helps you understand the possible cause and impact.</span>
            </div>
        </div>
        ''')

    # textwrap.dedent only removes the COMMON leading whitespace across all
    # lines; deeply-nested lines still keep some indentation. Markdown's
    # HTML-block parser treats a block of raw HTML as ending at the first
    # blank line -- and once that happens, any left-over 4+ space indent on
    # the next line gets read as an "indented code block" instead of HTML.
    # Stripping ALL leading whitespace from every line (regardless of
    # nesting) prevents that, blank lines or not.
    html_block = re.sub(r"(?m)^[ \t]+", "", html_block).strip()

    st.markdown(html_block, unsafe_allow_html=True)


def validate_ai_explanation(explanation, evidence, event_rows, selected_event, baseline_df):
    """Validate only the four user-facing claims that can be checked reliably
    for the selected anomaly event. This is evidence-consistency validation,
    not root-cause or recommendation validation.
    """
    checks = []

    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})

    # 1) Event time & duration
    expected_start = pd.to_datetime(selected_event.get("start_time"))
    expected_end = pd.to_datetime(selected_event.get("end_time"))
    expected_duration = float(selected_event.get("duration_min", 0) or 0)
    ai_time = str(explanation.get("when_time", ""))
    ai_duration = str(explanation.get("when_duration", ""))
    time_ok = fmt_time(expected_start) in ai_time and f"{round(expected_duration):g}" in ai_duration
    add(
        "Event time & duration",
        "PASS" if time_ok else "FAIL",
        f"Expected {fmt_time(expected_start)} → {fmt_time(expected_end)}, {expected_duration:.0f} min." if time_ok
        else "The generated time/duration does not match the selected event.",
    )

    text_parts = [str(explanation.get("headline", "")), str(explanation.get("summary", ""))]
    text_parts += [str(x) for x in explanation.get("why_it_happened", []) if x]
    text_parts += [str(x) for x in explanation.get("recommended_actions", []) if x]
    ai_text = " ".join(text_parts).lower()

    # 2) Power change direction
    direction_errors = []
    for col, label in [("dc_power_kw", "DC power"), ("ac_power_kw", "AC power")]:
        stats = evidence.get("event_observations", {}).get("statistics", {}).get(col, {})
        if stats.get("start") is None or stats.get("end") is None:
            continue
        start_value = float(stats["start"])
        end_value = float(stats["end"])
        if end_value < start_value:
            direction = "decreased"
            words = ["decreased", "fell", "dropped", "declined", "reduced"]
        elif end_value > start_value:
            direction = "increased"
            words = ["increased", "rose", "grew", "climbed"]
        else:
            direction = "stable"
            words = ["stable", "unchanged", "remained similar", "remained steady"]
        metric_token = label.split()[0].lower()
        if metric_token in ai_text and not any(w in ai_text for w in words):
            direction_errors.append(f"{label} should be described as {direction}")

    add(
        "Power change direction",
        "PASS" if not direction_errors else "WARN",
        "AC/DC power direction matches the event data." if not direction_errors else "; ".join(direction_errors),
    )

    # 3) Numerical values. Approximate language such as "about", "roughly",
    # "approximately" and "~" is intentionally allowed within tolerance.
    known_numbers = []
    for metric in evidence.get("event_observations", {}).get("statistics", {}).values():
        if not isinstance(metric, dict):
            continue
        for key in ("start", "end", "min", "max", "mean", "median"):
            value = metric.get(key)
            if isinstance(value, (int, float)) and pd.notna(value):
                known_numbers.append(float(value))

    known_numbers.append(expected_duration)

    for item in evidence.get("feature_contributions", []):
        value = item.get("mean_contribution_pct")
        if isinstance(value, (int, float)) and pd.notna(value):
            known_numbers.append(float(value))

    for metric in evidence.get("healthy_baseline", {}).get("reference", {}).values():
        if not isinstance(metric, dict):
            continue
        for key in ("median", "typical_low_q10", "typical_high_q90"):
            value = metric.get(key)
            if isinstance(value, (int, float)) and pd.notna(value):
                known_numbers.append(float(value))

    measurement_pattern = re.compile(
        r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)\s*(kW|kw|A|a|°C|C|%|minutes?|mins?|min|kvar|kva|Hz|h)\b"
    )
    numeric_claims = measurement_pattern.findall(" ".join(text_parts))
    unmatched = []
    for raw, unit in numeric_claims:
        value = float(raw)
        # 1.5% relative tolerance, with a minimum absolute tolerance of 2
        # units, so "roughly 70 kW" can match an actual 70-71 kW change.
        if not any(abs(value - k) <= max(2.0, abs(k) * 0.015) for k in known_numbers):
            unmatched.append(f"{raw} {unit}")

    add(
        "Numerical values",
        "PASS" if not unmatched else "WARN",
        "Reported measurement values are consistent with supplied evidence." if not unmatched
        else "Unmatched numeric claims: " + ", ".join(unmatched[:5]),
    )

    # 4) Temperature / threshold claims.
    # If the explanation mentions a limit but no independently supplied
    # temperature reference exists, mark it REVIEW/WARN rather than FAIL.
    temp_stats = evidence.get("event_observations", {}).get("statistics", {}).get("inverter_temperature_c", {})
    temp_ref = evidence.get("healthy_baseline", {}).get("reference", {}).get("inverter_temperature_c", {})
    threshold_words = ["healthy limit", "upper healthy", "upper limit", "below", "within normal", "normal limit"]
    threshold_claim = any(w in ai_text for w in threshold_words)

    if threshold_claim:
        upper = temp_ref.get("typical_high_q90")
        if upper is None:
            add(
                "Temperature / threshold claim",
                "WARN",
                "Temperature limit is mentioned, but no independently supplied temperature upper reference is available.",
            )
        elif temp_stats.get("max") is not None and float(temp_stats["max"]) <= float(upper) + 0.01:
            add(
                "Temperature / threshold claim",
                "PASS",
                f"Event max temperature is within the supplied upper reference ({float(upper):.1f} °C).",
            )
        else:
            add(
                "Temperature / threshold claim",
                "FAIL",
                "Temperature claim does not match the supplied event/baseline values.",
            )
    else:
        add(
            "Temperature / threshold claim",
            "PASS",
            "No unsupported temperature-limit claim detected.",
        )

    return checks

# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="background:#FFFFFF;padding:0.95rem 1.15rem;border:1px solid #D9E1EA;border-radius:12px;margin-bottom:0.9rem;box-shadow:0 1px 4px rgba(15,23,42,0.04);display:flex;align-items:center;gap:0.75rem;">
        <div style="background:#EFF6FF;color:#2563EB;width:46px;height:46px;border-radius:11px;display:flex;align-items:center;justify-content:center;font-size:1.45rem;flex-shrink:0;">⚡</div>
        <div>
            <div style="color:#0F172A;font-size:1.65rem;font-weight:700;line-height:1.2;">Inverter Anomaly Detection</div>
            <div style="color:#64748B;font-size:0.92rem;margin-top:0.16rem;line-height:1.35;">Monitor inverter performance, investigate anomaly events, and review AI-generated explanations.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILTERS (inline row, real bordered container)
# ============================================================

# The date picker covers the FULL dataset (trend_data.parquet goes back
# further than the scored evaluation period in dashboard_data.parquet), so
# users can browse raw history even for dates that weren't scored for
# anomalies. If trend_data.parquet isn't available yet, this just falls
# back to the evaluation period's own range.
eval_min_date = df["timestamp"].min().date()
eval_max_date = df["timestamp"].max().date()
min_date = min(eval_min_date, trend_df["timestamp"].min().date())
max_date = max(eval_max_date, trend_df["timestamp"].max().date())
show_inverter_filter = "inverter_id" in df.columns and df["inverter_id"].nunique() > 1

with st.container(border=True):
    if not has_trend_data:
        st.caption(
            "ℹ️ `trend_data.parquet` not found — showing the evaluation "
            "period only. Run notebook Section 19 to enable full history."
        )

    filter_cols = st.columns([1.2, 1.2, 1, 1, 2] if show_inverter_filter else [1.2, 1.2, 1, 2])

    with filter_cols[0]:
        start_date = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )

    with filter_cols[1]:
        end_date = st.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

    with filter_cols[2]:
        st.write("")  # align checkbox with the date inputs
        show_anomalies_only = st.checkbox("Show anomalies only", value=False)

    if show_inverter_filter:
        with filter_cols[3]:
            inverter_options = ["All"] + sorted(df["inverter_id"].dropna().unique().tolist())
            selected_inverter = st.selectbox("Inverter", inverter_options)
        caption_col = filter_cols[4]
    else:
        selected_inverter = "All"
        caption_col = filter_cols[3]

    with caption_col:
        st.caption("**Severity**: higher = further outside normal range.")

    if start_date > end_date:
        st.warning("Start date is after end date — swap them to see results.")
    elif has_trend_data and (start_date < eval_min_date or end_date > eval_max_date):
        st.caption(
            f"ℹ️ Anomaly detection only covers **{eval_min_date} to {eval_max_date}**. "
            "Dates outside that window show raw power & temperature readings "
            "but no anomaly results."
        )

# ============================================================
# BUILD ANOMALY EVENTS
# ============================================================

def build_anomaly_events(anomaly_df):
    """
    Build persistent anomaly events.

    Fixed rule:
        - Event must contain at least 60 minutes of continuous anomaly.
        - Current data sampling interval is 5 minutes.
        - Therefore 12 consecutive anomaly points = 60 minutes of coverage.
        - A gap greater than 7.5 minutes breaks the event.
    """

    if anomaly_df.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # FIXED EVENT RULE
    # --------------------------------------------------------
    EVENT_DURATION_MIN = 60
    SAMPLE_INTERVAL_MIN = 5

    # 60 minutes / 5 minutes = 12 points
    PERSISTENCE_MIN_POINTS = int(
        EVENT_DURATION_MIN / SAMPLE_INTERVAL_MIN
    )

    # Allow normal timestamp spacing, but break larger gaps
    MAX_RUN_GAP_MIN = SAMPLE_INTERVAL_MIN * 1.5

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------
    work = anomaly_df.copy()

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        errors="coerce"
    )

    work = (
        work
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if work.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # FIND CONTINUOUS ANOMALY RUNS
    # --------------------------------------------------------
    time_gap = (
        work["timestamp"]
        .diff()
        .dt.total_seconds()
        / 60.0
    )

    work["event_break"] = (
        time_gap.isna()
        | (time_gap > MAX_RUN_GAP_MIN)
    )

    work["run_id"] = work["event_break"].cumsum()

    # --------------------------------------------------------
    # KEEP ONLY 60-MINUTE PERSISTENT EVENTS
    # --------------------------------------------------------
    run_sizes = work.groupby("run_id").size()

    persistent_run_ids = run_sizes[
        run_sizes >= PERSISTENCE_MIN_POINTS
    ].index

    work = work[
        work["run_id"].isin(persistent_run_ids)
    ].copy()

    if work.empty:
        return pd.DataFrame()

    # --------------------------------------------------------
    # CREATE EVENT SUMMARY
    # --------------------------------------------------------
    events = []

    for event_number, (run_id, event_rows) in enumerate(
        work.groupby("run_id"),
        start=1
    ):

        event_rows = (
            event_rows
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        start_time = event_rows["timestamp"].min()
        end_time = event_rows["timestamp"].max()

        # 12 points at 5-minute sampling cover 60 minutes
        duration_min = (
            (end_time - start_time).total_seconds() / 60.0
        ) + SAMPLE_INTERVAL_MIN

        event = {
            "event_id": f"Event {event_number}",
            "run_id": int(run_id),
            "start_time": start_time,
            "end_time": end_time,
            "duration_min": duration_min,
            "anomaly_count": len(event_rows),
            "is_persistent_event": True,
        }

        # ----------------------------------------------------
        # SEVERITY
        # ----------------------------------------------------
        if "anomaly_score_ratio" in event_rows.columns:

            scores = pd.to_numeric(
                event_rows["anomaly_score_ratio"],
                errors="coerce"
            )

            event["max_severity"] = scores.max()
            event["mean_severity"] = scores.mean()

        # ----------------------------------------------------
        # RECONSTRUCTION ERROR
        # ----------------------------------------------------
        if "reconstruction_error" in event_rows.columns:

            errors = pd.to_numeric(
                event_rows["reconstruction_error"],
                errors="coerce"
            )

            event["max_reconstruction_error"] = errors.max()
            event["mean_reconstruction_error"] = errors.mean()

        # ----------------------------------------------------
        # DOMINANT INVERTER STATUS
        # ----------------------------------------------------
        if "inverter_status" in event_rows.columns:

            mode = event_rows["inverter_status"].mode()

            event["dominant_status"] = (
                mode.iloc[0]
                if len(mode)
                else None
            )

        # ----------------------------------------------------
        # DOMINANT ANOMALY TYPE
        # ----------------------------------------------------
        if "anomaly_type" in event_rows.columns:

            mode = event_rows["anomaly_type"].mode()

            event["anomaly_type"] = (
                mode.iloc[0]
                if len(mode)
                else None
            )

        events.append(event)

    return pd.DataFrame(events).reset_index(drop=True)
# ============================================================
# FILTER DATA
# ============================================================

# filtered_df: the scored evaluation data (has anomaly results) -- powers
# the KPIs, Overview chart, Anomalies table, and Investigate tab.
if start_date <= end_date:
    filtered_df = df[
        (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    ].copy()
    # filtered_trend_df: the full raw history -- powers the Trends tab so
    # dates before the evaluation period still show something (falls back
    # to filtered_df itself when trend_data.parquet isn't available).
    filtered_trend_df = trend_df[
        (trend_df["timestamp"].dt.date >= start_date) & (trend_df["timestamp"].dt.date <= end_date)
    ].copy()
else:
    filtered_df = df.iloc[0:0].copy()  # empty until the dates are fixed
    filtered_trend_df = trend_df.iloc[0:0].copy()

if selected_inverter != "All" and "inverter_id" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["inverter_id"] == selected_inverter].copy()
if selected_inverter != "All" and "inverter_id" in filtered_trend_df.columns:
    filtered_trend_df = filtered_trend_df[
        filtered_trend_df["inverter_id"] == selected_inverter
    ].copy()

if show_anomalies_only:
    filtered_df = filtered_df[filtered_df["anomaly_flag"]].copy()

if filtered_df.empty and filtered_trend_df.empty:
    st.warning("No observations match the current filters. Adjust the filters above.")

total_observations = len(filtered_df)
total_anomalies = int(filtered_df["anomaly_flag"].sum()) if total_observations else 0
anomaly_rate = (total_anomalies / total_observations * 100) if total_observations else None
max_temperature = (
    filtered_df["inverter_temperature_c"].max()
    if "inverter_temperature_c" in filtered_df.columns and total_observations
    else None
)

anomaly_df = (
    filtered_df[filtered_df["anomaly_flag"]].copy()
    if total_observations
    else filtered_df.copy()
)

# Build continuous anomaly events
events_df = build_anomaly_events(anomaly_df)

# ------------------------------------------------------------
# OVERVIEW
# ------------------------------------------------------------
st.markdown("## Overview")
st.caption("A high-level view of inverter observations and detected anomalies for the selected period.")

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Total Observations", f"{total_observations:,}")
with kpi_cols[1]:
    st.metric(
        "Anomalous Observations",
        f"{total_anomalies:,}",
    )
with kpi_cols[2]:
    st.metric("Anomaly Rate", fmt_num(anomaly_rate, 2, "%"))

if total_observations > 0 and total_anomalies == 0:
    st.caption("✅ No anomalies found in this period — everything looks normal.")
# ============================================================
# HEALTHY BASELINE
# ============================================================

if not baseline_df.empty:

    baseline_features = {
        "dc_power_kw": "DC Power",
        "inverter_temperature_c": "Temperature",
    }

    baseline_text = []

    for feature, label in baseline_features.items():

        q10_col = f"{feature}_q10"
        q90_col = f"{feature}_q90"

        if q10_col in baseline_df.columns and q90_col in baseline_df.columns:

            low = baseline_df[q10_col].median()
            high = baseline_df[q90_col].median()

            if pd.notna(low) and pd.notna(high):

                if feature.endswith("_kw"):
                    unit = " kW"
                elif feature.endswith("_a"):
                    unit = " A"
                elif feature.endswith("_c"):
                    unit = " °C"
                else:
                    unit = ""

            baseline_text.append(
                f"{label} = {low:.1f}–{high:.1f}{unit}"
            )

st.markdown("### Healthy Baseline")
st.caption("Typical healthy operating range")

baseline_cols = st.columns(4)

for i, item in enumerate(baseline_text):
    label, value = item.split(" = ", 1)

    with baseline_cols[i]:
        st.metric(
            label=label,
            value=value
        )
st.markdown("### Anomaly Score Over Time")
st.caption("Higher points mean more unusual behavior. Red dots are flagged anomalies.")

if total_observations > 0 and "reconstruction_error" in filtered_df.columns:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=filtered_df["timestamp"],
            y=filtered_df["reconstruction_error"],
            mode="lines",
            name="Anomaly Score",
            line=dict(color="#4C78A8", width=1.5),
        )
    )
    anomaly_points = filtered_df[filtered_df["anomaly_flag"]]
    if len(anomaly_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomaly_points["timestamp"],
                y=anomaly_points["reconstruction_error"],
                mode="markers",
                name="Flagged anomaly",
                marker=dict(size=8, color="#E45756", symbol="circle"),
            )
        )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        yaxis_type="log",
        hovermode="x unified",
        height=380,
        template="plotly_white",
        margin=dict(t=20, l=55, r=25, b=45),
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No anomaly score data available for the selected period.")

# ------------------------------------------------------------
# SELECTED ANOMALY EVENT + AI EXPLANATION
# ------------------------------------------------------------
st.markdown("## Selected Anomaly Event")
st.caption("Select one continuous anomaly event to inspect its evidence and generate an AI explanation.")

if len(events_df) > 0:
    events_df = events_df.sort_values("start_time").reset_index(drop=True)

    selector_col, status_col = st.columns([5.5, 1.5])
    with selector_col:
        selected_event_index = st.selectbox(
            "Anomaly event",
            range(len(events_df)),
            format_func=lambda x: (
                f"Event {x + 1} | "
                f"{fmt_time(events_df.loc[x, 'start_time'])} → {fmt_time(events_df.loc[x, 'end_time'])}"
            ),
            key="ai_event_selector",
            label_visibility="collapsed",
        )
    selected_event = events_df.loc[selected_event_index]
    event_rows = anomaly_df[
        (anomaly_df["timestamp"] >= selected_event["start_time"]) &
        (anomaly_df["timestamp"] <= selected_event["end_time"])
    ].copy()

    with status_col:
        st.markdown(
            '<div style="text-align:right;padding-top:0.35rem;"><span class="event-badge">● Anomaly detected</span></div>',
            unsafe_allow_html=True,
        )

    summary_cols = st.columns(4)
    with summary_cols[0]: st.metric("Start Time", fmt_time(selected_event.get("start_time")))
    with summary_cols[1]: st.metric("End Time", fmt_time(selected_event.get("end_time")))
    with summary_cols[2]: st.metric("Duration", fmt_num(selected_event.get("duration_min"), 0, " min"))
    with summary_cols[3]: st.metric("Anomaly Points", f"{int(selected_event.get('anomaly_count', 0)):,}")

    event_key = clean_value(selected_event.get("event_id"))
    anomaly_key = f"{event_key}"
    cached = st.session_state["ai_explanations"].get(anomaly_key)

    evidence_col, ai_col = st.columns([1.45, 0.95], gap="large")

    # LEFT: selected event evidence
    with evidence_col:
        st.markdown("### Event Evidence")
        st.caption("Observed values for the selected event. Evidence is not a confirmed physical root cause.")

        metric_cols = st.columns(3)
        for i, (col, label, unit) in enumerate([
            ("dc_power_kw", "DC Power", " kW"),
            ("ac_power_kw", "AC Power", " kW"),
            ("inverter_temperature_c", "Temperature", " °C"),
        ]):
            with metric_cols[i]:
                if col in event_rows.columns and not event_rows[col].dropna().empty:
                    series = pd.to_numeric(event_rows[col], errors="coerce").dropna()
                    start_v = series.iloc[0]
                    end_v = series.iloc[-1]
                    delta_pct = ((end_v - start_v) / start_v * 100) if start_v != 0 else None
                    delta_text = f"{delta_pct:+.1f}%" if delta_pct is not None else "—"
                    st.metric(label, f"{start_v:.0f}{unit} → {end_v:.0f}{unit}", delta_text)
                else:
                    st.metric(label, "—")

        event_start = pd.to_datetime(selected_event["start_time"])
        event_end = pd.to_datetime(selected_event["end_time"])
        trend_window, _ = get_trend_window(trend_df, event_start, hours_back=24)

        st.markdown("### Power & Temperature Trend")
        if len(trend_window) > 1:
            fig_trend = go.Figure()
            if "dc_power_kw" in trend_window.columns:
                fig_trend.add_trace(go.Scatter(x=trend_window["timestamp"], y=trend_window["dc_power_kw"], mode="lines", name="DC Power (kW)", line=dict(color="#4C78A8", width=2)))
            if "ac_power_kw" in trend_window.columns:
                fig_trend.add_trace(go.Scatter(x=trend_window["timestamp"], y=trend_window["ac_power_kw"], mode="lines", name="AC Power (kW)", line=dict(color="#F28E2B", width=2)))
            if "inverter_temperature_c" in trend_window.columns:
                fig_trend.add_trace(go.Scatter(x=trend_window["timestamp"], y=trend_window["inverter_temperature_c"], mode="lines", name="Temperature (°C)", yaxis="y2", line=dict(color="#59A14F", width=2)))
            fig_trend.add_vrect(x0=event_start, x1=event_end, fillcolor="#E45756", opacity=0.10, line_color="#E45756", line_width=1)
            fig_trend.update_layout(xaxis_title="Time", yaxis=dict(title="Power (kW)"), yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"), hovermode="x unified", height=355, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5), margin=dict(l=50, r=55, t=45, b=40))
            st.plotly_chart(fig_trend, width="stretch")
        else:
            st.info("Not enough history is available to draw the event trend.")

    # RIGHT: AI explanation + compact validation
    with ai_col:
        ai_header_col, ai_button_col = st.columns([3.8, 1.5])
        with ai_header_col:
            st.markdown('<div class="ai-title">AI Explanation</div>', unsafe_allow_html=True)
        with ai_button_col:
            regenerate = st.button("Regenerate", type="primary", use_container_width=True, help="Generate a fresh explanation for this selected anomaly event.")

        if regenerate or cached is None:
            with st.spinner("Generating explanation from the selected event evidence..."):
                try:
                    ai_explanation, ai_evidence = generate_ai_explanation(selected_event, event_rows)
                    cached = {"explanation": ai_explanation, "evidence": ai_evidence, "error": None}
                except Exception as e:
                    cached = {"explanation": None, "evidence": None, "error": str(e)}
                st.session_state["ai_explanations"][anomaly_key] = cached

        if cached is None:
            st.info("Select an anomaly event to generate an AI explanation.")
        elif cached.get("error"):
            st.error(f"AI explanation failed: {cached['error']}")
        elif cached.get("explanation"):
            render_ai_explanation(cached["explanation"])

            # Run the detailed evidence/data plumbing validation internally,
            # but do not expose its many low-level checks in the dashboard.
            # The UI shows only the four meaningful explanation checks below.
            try:
                validate_event_data(
                    evidence=cached["evidence"],
                    df=df,
                    trend_df=trend_df if has_trend_data else None,
                    baseline_df=baseline_df,
                )
            except Exception:
                pass

            merged_checks = validate_ai_explanation(
                cached["explanation"],
                cached["evidence"],
                event_rows,
                selected_event,
                baseline_df,
            )
            pass_count = sum(r["status"] == "PASS" for r in merged_checks)
            fail_count = sum(r["status"] == "FAIL" for r in merged_checks)
            warn_count = sum(r["status"] == "WARN" for r in merged_checks)
            verdict = "FAIL" if fail_count else ("NEEDS REVIEW" if warn_count else "PASS")
            verdict_class = "validation-fail" if verdict == "FAIL" else ("validation-warn" if verdict == "NEEDS REVIEW" else "validation-pass")

            st.markdown(
                f'''<div class="validation-card">
                <div class="validation-header"><div class="validation-title">Validation</div><span class="validation-pill {verdict_class}">{pass_count}/{len(merged_checks)} checks verified</span></div>
                <div class="validation-subtitle">Checks whether the explanation is consistent with the selected event's data and supplied evidence. It does not prove the physical root cause.</div>
                </div>''',
                unsafe_allow_html=True,
            )

            important_names = {"Event time & duration", "Numerical values", "Power change direction", "Temperature / threshold claim"}
            important = [r for r in merged_checks if r["name"] in important_names]
            for check in important:
                status = check["status"]
                icon = "✓" if status == "PASS" else ("!" if status == "WARN" else "×")
                cls = "check-pass" if status == "PASS" else ("check-warn" if status == "WARN" else "check-fail")
                st.markdown(
                    f'''<div class="check-row"><span class="check-icon {cls}">{icon}</span><div><div class="check-name">{html.escape(check["name"])}</div><div class="check-detail">{html.escape(check["detail"])}</div></div></div>''',
                    unsafe_allow_html=True,
                )
            st.markdown('<div class="validation-note">Validation checks evidence consistency. A passed explanation is not proof of a physical fault or root cause.</div>', unsafe_allow_html=True)
        else:
            st.warning("AI did not return an explanation for this anomaly event.")
else:
    st.info("No anomaly events are available in the selected period.")

# ------------------------------------------------------------
# DETECTED ANOMALIES
# ------------------------------------------------------------
st.markdown("## Detected Anomalies")
st.caption("Observations flagged by the anomaly detector in the selected period.")

if len(events_df) > 0:

    event_table = events_df.copy()

    event_table["Event"] = [
        f"Event {i + 1}"
        for i in range(len(event_table))
    ]

    display_columns = [
        "Event",
        "start_time",
        "end_time",
        "duration_min",
        "anomaly_count",
    ]

    display_columns = [
        c for c in display_columns
        if c in event_table.columns
    ]

    friendly_names = {
        "start_time": "Start Time",
        "end_time": "End Time",
        "duration_min": "Duration (min)",
        "anomaly_count": "Anomaly Points"
    }

    table = (
        event_table[display_columns]
        .rename(columns=friendly_names)
        .sort_values("Start Time")
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True
    )

else:
    if total_observations > 0:
        st.success(
            "✅ No anomaly events were detected in this period."
        )
    else:
        st.info(
            "No readings in this date range. Try a different range above."
        )
