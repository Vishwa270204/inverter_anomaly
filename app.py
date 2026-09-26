"""
Inverter Anomaly Detection -- Streamlit Dashboard
====================================================
Production frontend only. All ML/training happens in inverter_anomaly.ipynb.

Reads:
    dashboard_data.parquet      -> evaluation-period observations + model output
    trend_data.parquet          -> optional extended raw history
    dashboard_baseline.parquet  -> healthy operating baseline (quantiles)

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

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from groq import Groq

# data_validation.py is an optional integration: if it isn't present next to
# app.py, the dashboard should degrade gracefully instead of crashing on import.
try:
    from data_validation import validate_event_data, summarize as summarize_data
    HAS_DATA_VALIDATION = True
except Exception:
    HAS_DATA_VALIDATION = False

    def validate_event_data(*args, **kwargs):
        raise RuntimeError("data_validation module is not available.")

    def summarize_data(*args, **kwargs):
        raise RuntimeError("data_validation module is not available.")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Inverter Anomaly Detection",
    page_icon="⚡",
    layout="wide",
)

# Professional, compact theme. Filters live near the top and content is
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

# One-time migration: the explanation format has changed shape over time
# (plain paragraph string -> flat when_time/when_duration dict -> the
# current when_occurred dict). Drop any cached entries that don't match the
# current shape so they regenerate instead of breaking the renderer.
_stale_keys = []
for _k, _v in st.session_state["ai_explanations"].items():
    if not isinstance(_v, dict):
        _stale_keys.append(_k)
        continue
    _exp = _v.get("explanation")
    if isinstance(_exp, str):
        _stale_keys.append(_k)
    elif isinstance(_exp, dict) and "when_occurred" not in _exp:
        _stale_keys.append(_k)
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
        if c.endswith("_median") or c.endswith("_q10") or c.endswith("_q90")
    ]
    for col in numeric_cols:
        baseline[col] = pd.to_numeric(baseline[col], errors="coerce")

    if "healthy_sample_count" in baseline.columns:
        baseline["healthy_sample_count"] = pd.to_numeric(
            baseline["healthy_sample_count"], errors="coerce"
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


# Metrics used for the population-level baseline comparison. Kept as one
# list so the comparison table, the aggregated reference, and the AI
# evidence all agree on the same set of variables.
COMPARISON_METRICS = [
    ("dc_power_kw", "DC Power", " kW"),
    ("ac_power_kw", "AC Power", " kW"),
    ("inverter_temperature_c", "Temperature", " °C"),
    ("efficiency_pct", "Efficiency", "%"),
    ("power_factor", "Power Factor", ""),
    ("dc_current_a", "DC Current", " A"),
]


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
    idx = (source["timestamp"] - target).abs().idxmin()
    return row_to_dict(source.loc[idx])


def get_pre_anomaly_trend(timestamp, hours=24):
    target = pd.to_datetime(timestamp)
    start = target - timedelta(hours=float(hours))

    source = trend_df[(trend_df["timestamp"] >= start) & (trend_df["timestamp"] <= target)].copy()

    if source.empty:
        return {"error": "No trend observations found for the requested window."}

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
        series = pd.to_numeric(source[col], errors="coerce").dropna()
        if len(series) >= 2:
            stats[col] = {
                "start": clean_value(series.iloc[0]),
                "end": clean_value(series.iloc[-1]),
                "net_change": clean_value(series.iloc[-1] - series.iloc[0]),
                "min": clean_value(series.min()),
                "max": clean_value(series.max()),
                "median": clean_value(series.median()),
            }

    return {
        "window_start": clean_value(source["timestamp"].min()),
        "window_end": clean_value(source["timestamp"].max()),
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
    idx = (source["timestamp"] - target).abs().idxmin()
    row = source.loc[idx]

    wanted = [
        "timestamp", "hour", "minute", "month", "is_daylight", "inverter_status",
        "dc_power_kw", "ac_power_kw", "dc_current_a", "ac_current_a", "power_factor",
        "frequency_hz", "efficiency_pct", "inverter_temperature_c", "ambient_temperature_c",
        "poa_w_m2", "ghi_w_m2", "quality_code", "communication_status", "fault_code", "alarm_code",
    ]
    return {k: clean_value(row.get(k)) for k in wanted if k in source.columns}


def get_baseline_context(timestamp):
    """Retrieve healthy reference values for conditions similar to the
    selected event. This is a comparison reference only. It does not
    establish cause."""
    target = pd.to_datetime(timestamp)
    source = df.copy()
    if source.empty:
        return {"error": "No inverter data available."}

    idx = (source["timestamp"] - target).abs().idxmin()
    row = source.loc[idx]

    conditions = {
        "inverter_status": clean_value(row.get("inverter_status")),
        "is_daylight": clean_value(row.get("is_daylight")),
        "hour": clean_value(row.get("hour")),
        "month": clean_value(row.get("month")),
    }

    baseline = baseline_df.copy()
    match_cols = [c for c in ["inverter_status", "is_daylight", "hour", "month"]
                  if c in baseline.columns and conditions.get(c) is not None]

    matched = baseline.copy()
    for col in match_cols:
        matched = matched[matched[col].astype(str) == str(conditions[col])]

    if matched.empty:
        return {
            "error": "No healthy baseline group is available for the selected operating conditions.",
            "conditions": conditions,
        }

    result = {
        "conditions": conditions,
        "healthy_sample_count": int(matched["healthy_sample_count"].sum()) if "healthy_sample_count" in matched.columns else None,
        "reference": {},
    }

    metric_names = sorted({col[:-7] for col in matched.columns if col.endswith("_median")})
    for metric in metric_names:
        median_col, q10_col, q90_col = f"{metric}_median", f"{metric}_q10", f"{metric}_q90"
        if median_col not in matched.columns:
            continue
        cols_present = [c for c in [median_col, q10_col, q90_col] if c in matched.columns]
        values = matched[cols_present].dropna(how="all")
        if values.empty:
            continue
        result["reference"][metric] = {
            "median": clean_value(values[median_col].iloc[0]) if median_col in values.columns else None,
            "typical_low_q10": clean_value(values[q10_col].iloc[0]) if q10_col in values.columns else None,
            "typical_high_q90": clean_value(values[q90_col].iloc[0]) if q90_col in values.columns else None,
        }

    return result


def get_healthy_baseline_summary(baseline_df):
    """Aggregate the healthy-baseline quantile table into simple headline
    ranges. Returns a list of (label, value_str) pairs. Never raises --
    returns an empty list if the baseline has no usable data. This replaces
    the previous version's scope bug (an undefined `unit` could leak across
    features when a metric had no valid quantiles)."""
    if baseline_df is None or baseline_df.empty:
        return []

    rows = []
    for feature, label, unit in COMPARISON_METRICS:
        q10_col, q90_col = f"{feature}_q10", f"{feature}_q90"
        if q10_col not in baseline_df.columns or q90_col not in baseline_df.columns:
            continue
        low = pd.to_numeric(baseline_df[q10_col], errors="coerce").median()
        high = pd.to_numeric(baseline_df[q90_col], errors="coerce").median()
        if pd.notna(low) and pd.notna(high):
            rows.append((label, f"{low:.1f}–{high:.1f}{unit}"))
    return rows


def aggregate_baseline_reference(baseline_df, metrics):
    """Median-of-medians reference values for each metric across all
    healthy baseline groups. A population-level reference (not matched to
    any single timestamp's operating conditions)."""
    reference = {}
    if baseline_df is None or baseline_df.empty:
        return reference
    for metric in metrics:
        median_col, q10_col, q90_col = f"{metric}_median", f"{metric}_q10", f"{metric}_q90"
        if median_col not in baseline_df.columns:
            continue
        med = pd.to_numeric(baseline_df[median_col], errors="coerce").dropna()
        if med.empty:
            continue
        q10 = pd.to_numeric(baseline_df[q10_col], errors="coerce").dropna() if q10_col in baseline_df.columns else pd.Series(dtype=float)
        q90 = pd.to_numeric(baseline_df[q90_col], errors="coerce").dropna() if q90_col in baseline_df.columns else pd.Series(dtype=float)
        reference[metric] = {
            "median": clean_value(med.median()),
            "typical_low_q10": clean_value(q10.median()) if not q10.empty else None,
            "typical_high_q90": clean_value(q90.median()) if not q90.empty else None,
        }
    return reference


def compare_population_to_baseline(anomaly_df, baseline_reference):
    """Compare mean anomaly-population values against the aggregated
    healthy baseline reference. Supporting evidence only -- does not
    establish cause."""
    rows = []
    if anomaly_df is None or anomaly_df.empty:
        return rows
    for metric, label, unit in COMPARISON_METRICS:
        if metric not in anomaly_df.columns or metric not in baseline_reference:
            continue
        values = pd.to_numeric(anomaly_df[metric], errors="coerce").dropna()
        if values.empty:
            continue
        observed = values.mean()
        ref = baseline_reference[metric]
        low, high = ref.get("typical_low_q10"), ref.get("typical_high_q90")
        if low is not None and high is not None:
            if observed < low:
                status = "Below typical range"
            elif observed > high:
                status = "Above typical range"
            else:
                status = "Within typical range"
            range_str = f"{low:.1f}–{high:.1f}{unit}"
        else:
            status = "No healthy reference available"
            range_str = "—"
        rows.append({
            "label": label,
            "observed": f"{observed:.1f}{unit}",
            "typical_range": range_str,
            "status": status,
        })
    return rows


def build_key_observations(anomaly_df, events_df, comparison_rows):
    """Operator-facing summary answering 'what should I investigate first?'.
    Built only from values already computed elsewhere -- no new claims."""
    obs = {
        "anomaly_count": int(len(anomaly_df)),
        "event_count": int(len(events_df)),
        "typical_duration": None,
        "max_severity": None,
        "top_features": [],
        "off_baseline_metrics": [],
    }
    if not events_df.empty and "duration_min" in events_df.columns:
        durations = pd.to_numeric(events_df["duration_min"], errors="coerce").dropna()
        if not durations.empty:
            obs["typical_duration"] = float(durations.median())

    if "anomaly_score_ratio" in anomaly_df.columns:
        scores = pd.to_numeric(anomaly_df["anomaly_score_ratio"], errors="coerce").dropna()
        if not scores.empty:
            obs["max_severity"] = float(scores.max())

    contribution_cols = get_contribution_cols(anomaly_df)
    if contribution_cols:
        means = (
            anomaly_df[contribution_cols].apply(pd.to_numeric, errors="coerce")
            .mean().dropna().sort_values(ascending=False)
        )
        obs["top_features"] = [c.replace("_contribution_pct", "") for c in means.head(3).index]

    obs["off_baseline_metrics"] = [
        r["label"] for r in comparison_rows if r["status"] in ("Above typical range", "Below typical range")
    ]
    return obs


def render_key_observations(obs):
    lines = [f"**{obs['anomaly_count']:,}** anomalous observations across **{obs['event_count']}** persistent event(s)."]
    if obs["typical_duration"] is not None:
        lines.append(f"Typical persistent-event duration: **~{obs['typical_duration']:.0f} min**.")
    if obs["max_severity"] is not None:
        lines.append(f"Maximum observed anomaly score (not a probability): **{obs['max_severity']:.2f}**.")
    if obs["top_features"]:
        lines.append("Variables contributing most to unusual reconstruction error: **" + ", ".join(obs["top_features"]) + "**.")
    if obs["off_baseline_metrics"]:
        lines.append(
            "Outside the typical healthy range during anomalies: **" + ", ".join(obs["off_baseline_metrics"])
            + "** (supporting evidence only, not proof of cause)."
        )
    if len(lines) == 1 and obs["anomaly_count"] == 0:
        st.caption("No anomalies in the selected period.")
        return
    for line in lines:
        st.markdown(f"- {line}")


def run_optional_data_validation(frame):
    """Best-effort integration with data_validation.py. Exact function
    signatures can vary by deployment, so every call is guarded -- a
    failure here must never crash the dashboard."""
    checks, summary = None, None
    if not HAS_DATA_VALIDATION:
        return checks, summary
    try:
        checks = validate_event_data(frame)
    except Exception:
        checks = None
    try:
        summary = summarize_data(frame)
    except Exception:
        summary = None
    return checks, summary


def render_optional_data_validation(frame, label="selected data"):
    if not HAS_DATA_VALIDATION:
        st.caption("Data validation module is not available in this deployment.")
        return
    if frame is None or frame.empty:
        st.caption(f"No {label} available to validate.")
        return

    checks, summary = run_optional_data_validation(frame)
    if checks is None and summary is None:
        st.caption("Data validation did not return a result for this selection.")
        return

    if summary is not None:
        with st.expander("Data summary", expanded=False):
            if isinstance(summary, dict):
                st.json(summary)
            else:
                st.write(summary)

    if checks is not None:
        with st.expander("Data validation checks", expanded=False):
            if isinstance(checks, (list, tuple)):
                for c in checks:
                    if isinstance(c, dict) and "name" in c and "status" in c:
                        status = str(c["status"]).upper()
                        icon_class = {"PASS": "check-pass", "OK": "check-pass", "WARN": "check-warn", "FAIL": "check-fail"}.get(status, "check-warn")
                        icon = {"PASS": "✓", "OK": "✓", "WARN": "!", "FAIL": "✕"}.get(status, "!")
                        st.markdown(
                            f'<div class="check-row"><div class="check-icon {icon_class}">{icon}</div>'
                            f'<div><div class="check-name">{html.escape(str(c["name"]))}</div>'
                            f'<div class="check-detail">{html.escape(str(c.get("detail","")))}</div></div></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.write(c)
            else:
                st.write(checks)


def build_population_evidence(anomaly_df, events_df, baseline_df, start_date, end_date, selected_inverter):
    """Single evidence structure shared by the AI generator AND the
    evidence-consistency validator, so the two never disagree on field
    names (this was the previous inconsistency between
    'anomaly_observations'/'healthy_baseline_summary' in the generator and
    'event_observations'/'healthy_baseline' in the validator)."""

    evidence = {
        "analysis_scope": {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "inverter": "All" if selected_inverter == "All" else str(selected_inverter),
        },
        "anomaly_summary": {
            "anomaly_observation_count": int(len(anomaly_df)),
            "persistent_event_count": int(len(events_df)),
        },
    }

    event_patterns = {}
    if not events_df.empty:
        for col in ["duration_min", "anomaly_count", "max_severity", "mean_severity"]:
            if col in events_df.columns:
                values = pd.to_numeric(events_df[col], errors="coerce").dropna()
                if not values.empty:
                    event_patterns[col] = {
                        "min": clean_value(values.min()),
                        "mean": clean_value(values.mean()),
                        "median": clean_value(values.median()),
                        "max": clean_value(values.max()),
                    }
        if "start_time" in events_df.columns:
            event_patterns["earliest_event_start"] = clean_value(events_df["start_time"].min())
        if "end_time" in events_df.columns:
            event_patterns["latest_event_end"] = clean_value(events_df["end_time"].max())
    evidence["event_patterns"] = event_patterns

    operating_patterns = {}
    for col in ["inverter_status", "is_daylight", "quality_code", "communication_status", "fault_code", "alarm_code"]:
        if col not in anomaly_df.columns:
            continue
        values = anomaly_df[col].dropna().astype(str)
        if not values.empty:
            counts = values.value_counts().head(10)
            operating_patterns[col] = {str(k): int(v) for k, v in counts.items()}
    evidence["operating_patterns"] = operating_patterns

    contribution_cols = get_contribution_cols(anomaly_df)
    if contribution_cols:
        means = (
            anomaly_df[contribution_cols].apply(pd.to_numeric, errors="coerce")
            .mean().dropna().sort_values(ascending=False)
        )
        evidence["feature_contributions"] = [
            {"feature": col.replace("_contribution_pct", ""), "mean_contribution_pct": clean_value(value)}
            for col, value in means.head(10).items()
        ]
    else:
        evidence["feature_contributions"] = []

    numeric_cols = [
        "dc_power_kw", "dc_current_a", "ac_power_kw", "ac_current_a",
        "power_factor", "frequency_hz", "efficiency_pct",
        "inverter_temperature_c", "ambient_temperature_c",
        "poa_w_m2", "ghi_w_m2", "packet_loss_pct", "communication_latency_ms",
    ]
    stats = {}
    for col in numeric_cols:
        if col not in anomaly_df.columns:
            continue
        values = pd.to_numeric(anomaly_df[col], errors="coerce").dropna()
        if not values.empty:
            stats[col] = {
                "min": clean_value(values.min()),
                "mean": clean_value(values.mean()),
                "median": clean_value(values.median()),
                "max": clean_value(values.max()),
            }
    evidence["anomaly_observations"] = {"observation_count": int(len(anomaly_df)), "statistics": stats}

    evidence["healthy_baseline"] = {"reference": aggregate_baseline_reference(baseline_df, numeric_cols)}

    return evidence


def generate_overall_ai_explanation(evidence):
    """Generate one evidence-based explanation for the whole anomaly
    population in scope (not a single selected event)."""

    client = get_groq_client()
    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set GROQ_API_KEY in the environment "
            "or Streamlit secrets."
        )

    prompt = """
You are an explanation assistant inside a solar inverter anomaly detection dashboard.

Analyze ALL detected anomalies in the supplied period as one population. Do not
describe a single selected event -- describe recurring patterns across the group.

Return ONLY valid JSON with this exact structure:
{
  "headline": "One short sentence describing the overall anomaly pattern.",
  "summary": "2-3 plain-language sentences explaining what is generally happening.",
  "why_it_happened": ["pattern-based reason 1", "pattern-based reason 2", "pattern-based reason 3"],
  "when_occurred": {
    "time_pattern": "The overall date/time range in scope, and whether anomalies cluster during daylight/night or particular hours, only if supported by evidence.",
    "duration_pattern": "The persistent-event count and typical/shortest/longest duration, using the supplied event statistics.",
    "operating_pattern": "Recurring operating-condition patterns (status, communication, daylight) only if supported by evidence."
  },
  "recommended_actions": ["practical operator check 1", "practical operator check 2", "practical operator check 3"]
}

Rules:
- Use only the supplied evidence; never invent measurements, dates, or counts.
- Feature contributions show variables associated with unusual reconstruction error;
  they do not prove physical root cause. Never say "root cause" or "caused by".
- Never call anomaly_score_ratio or reconstruction_error a probability or confidence value.
- Healthy baseline values are supporting comparison only, not proof of a fault.
- If the physical cause cannot be determined from the evidence, say so explicitly.
- Prefer patterns supported by multiple observations or events over isolated ones.
- Avoid ML jargon (autoencoder, threshold, reconstruction error, probability) in the
  operator-facing text -- write for a plant operator.
- Keep every field concise and grounded only in the SUPPLIED EVIDENCE below.

SUPPLIED EVIDENCE:
""" + json.dumps(evidence, default=str, ensure_ascii=False)

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Return ONLY a valid JSON object. Do not use markdown fences. "
                    "Do not add commentary before or after the JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=1600,
        response_format={
    "type": "json_schema",
    "json_schema": {
        "name": "inverter_anomaly_explanation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "headline": {
                    "type": "string"
                },
                "summary": {
                    "type": "string"
                },
                "why_it_happened": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                },
                "when_occurred": {
                    "type": "object",
                    "properties": {
                        "time_pattern": {
                            "type": "string"
                        },
                        "duration_pattern": {
                            "type": "string"
                        },
                        "operating_pattern": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "time_pattern",
                        "duration_pattern",
                        "operating_pattern"
                    ],
                    "additionalProperties": False
                },
                "recommended_actions": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    }
                }
            },
            "required": [
                "headline",
                "summary",
                "why_it_happened",
                "when_occurred",
                "recommended_actions"
            ],
            "additionalProperties": False
        }
    }
},
    )

    raw = (response.choices[0].message.content or "").strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start:end + 1]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"The AI returned malformed JSON. Parser error: {exc}.") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("The AI returned valid JSON, but it was not a JSON object.")

    return parsed


def render_ai_explanation(data):
    """Render the structured LLM explanation as icon-header sections
    (alert banner, Why it happened, When it occurred, Recommended action)."""
    if not data:
        return

    # Backward compatibility: much older cached entries stored a plain
    # paragraph string instead of a dict.
    if isinstance(data, str):
        text = re.sub(r"\s+", " ", data).strip()
        safe = html.escape(text)
        safe = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", safe)
        st.markdown(f'<div class="ai-card"><div class="ai-body">{safe}</div></div>', unsafe_allow_html=True)
        return

    def esc(s):
        return html.escape(str(s)) if s is not None else ""

    headline = esc(data.get("headline", ""))
    summary = esc(data.get("summary", ""))
    why_bullets = [esc(b) for b in data.get("why_it_happened", []) if b]
    action_bullets = [esc(b) for b in data.get("recommended_actions", []) if b]

    when = data.get("when_occurred")
    if isinstance(when, dict):
        time_pattern = esc(when.get("time_pattern", "—"))
        duration_pattern = esc(when.get("duration_pattern", "—"))
        operating_pattern = esc(when.get("operating_pattern", ""))
    else:
        # Backward compatibility with the older flat when_time/when_duration shape.
        time_pattern = esc(data.get("when_time", "—"))
        duration_pattern = esc(data.get("when_duration", "—"))
        operating_pattern = ""

    why_html = "".join(f"<li>{b}</li>" for b in why_bullets)
    action_html = "".join(f"<li>{b}</li>" for b in action_bullets)
    operating_row_html = (
        f'<div class="ai-row"><div class="ai-label">Operating pattern</div><div class="ai-body">{operating_pattern}</div></div>'
        if operating_pattern else ""
    )

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
                <div class="ai-row">
                    <div class="ai-label">Time pattern</div>
                    <div class="ai-body">{time_pattern}</div>
                </div>
                <div class="ai-row">
                    <div class="ai-label">Duration pattern</div>
                    <div class="ai-body">{duration_pattern}</div>
                </div>
                {operating_row_html}
            </div>

            <div class="ai-section">
                <div class="ai-section-header">
                    <span class="ai-section-icon">🔧</span> Recommended action
                </div>
                <ul class="ai-bullets">{action_html}</ul>
            </div>

            <div class="ai-footer-note">
                <span>ℹ️</span>
                <span>This explanation is based on patterns observed across the selected
                anomaly population and the trained anomaly detection model. It does not
                confirm a fault but helps you understand where to look first.</span>
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


def validate_overall_explanation(explanation, evidence):
    """Evidence-consistency checks for the OVERALL (population) explanation.
    This checks whether the AI's claims are consistent with the supplied
    evidence. It does NOT validate a physical diagnosis, and it is not tied
    to any single selected event's timestamp/duration."""
    checks = []

    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})

    text_parts = [str(explanation.get("headline", "")), str(explanation.get("summary", ""))]
    text_parts += [str(x) for x in explanation.get("why_it_happened", []) if x]
    text_parts += [str(x) for x in explanation.get("recommended_actions", []) if x]
    when = explanation.get("when_occurred")
    if isinstance(when, dict):
        text_parts += [str(when.get(k, "")) for k in ("time_pattern", "duration_pattern", "operating_pattern")]
    full_text = " ".join(text_parts)
    lower_text = full_text.lower()

    # 1) Terminology safety.
    banned_phrases = ["root cause", "probability of", "confidence score", "proven cause", "confirmed fault"]
    found_banned = [p for p in banned_phrases if p in lower_text]
    add(
        "Terminology safety",
        "PASS" if not found_banned else "FAIL",
        "No unsupported causal or probability language detected." if not found_banned
        else "Unsupported phrasing found: " + ", ".join(found_banned),
    )

    # 2) Persistent event count.
    expected_events = evidence.get("anomaly_summary", {}).get("persistent_event_count")
    duration_pattern_text = str(when.get("duration_pattern", "")) if isinstance(when, dict) else ""
    if expected_events is not None:
        count_match = re.search(r"\b(\d+)\b", duration_pattern_text)
        if count_match is None:
            add("Persistent event count", "WARN", "Duration pattern does not mention an event count to check.")
        elif int(count_match.group(1)) == int(expected_events):
            add("Persistent event count", "PASS", f"Matches the {expected_events} persistent event(s) in scope.")
        else:
            add("Persistent event count", "WARN", f"Mentions {count_match.group(1)}, evidence shows {expected_events}.")
    else:
        add("Persistent event count", "PASS", "No persistent events to check.")

    # 3) Numerical values -- every number+unit mentioned should be
    # traceable to a number in the supplied evidence (with tolerance).
    known_numbers = []
    for stats in evidence.get("event_patterns", {}).values():
        if isinstance(stats, dict):
            known_numbers += [v for v in stats.values() if isinstance(v, (int, float))]
    for stats in evidence.get("anomaly_observations", {}).get("statistics", {}).values():
        if isinstance(stats, dict):
            known_numbers += [v for v in stats.values() if isinstance(v, (int, float))]
    for item in evidence.get("feature_contributions", []):
        v = item.get("mean_contribution_pct")
        if isinstance(v, (int, float)):
            known_numbers.append(v)
    for ref in evidence.get("healthy_baseline", {}).get("reference", {}).values():
        if isinstance(ref, dict):
            known_numbers += [v for v in ref.values() if isinstance(v, (int, float))]
    for key in ("anomaly_observation_count", "persistent_event_count"):
        v = evidence.get("anomaly_summary", {}).get(key)
        if isinstance(v, (int, float)):
            known_numbers.append(v)
    known_numbers = [float(n) for n in known_numbers if isinstance(n, (int, float)) and pd.notna(n)]

    measurement_pattern = re.compile(
        r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)\s*(kW|kw|A|a|°C|C|%|minutes?|mins?|min|Hz|events?)\b"
    )
    numeric_claims = measurement_pattern.findall(full_text)
    unmatched = []
    for raw, unit in numeric_claims:
        value = float(raw)
        if not any(abs(value - k) <= max(2.0, abs(k) * 0.02) for k in known_numbers):
            unmatched.append(f"{raw} {unit}")
    add(
        "Numerical values",
        "PASS" if not unmatched else "WARN",
        "Reported figures are consistent with the supplied evidence." if not unmatched
        else "Unmatched figures: " + ", ".join(unmatched[:5]),
    )

    return checks


def render_evidence_consistency(checks):
    if not checks:
        return

    worst = "PASS"
    for c in checks:
        if c["status"] == "FAIL":
            worst = "FAIL"
            break
        if c["status"] == "WARN" and worst == "PASS":
            worst = "WARN"

    label = "Passed" if worst == "PASS" else "Review"

    with st.expander(f"Evidence consistency: {label}", expanded=False):
        st.caption(
            "These checks confirm the explanation's claims are consistent with the "
            "supplied data. They do not validate a physical diagnosis."
        )
        for c in checks:
            icon_class = {"PASS": "check-pass", "WARN": "check-warn", "FAIL": "check-fail"}[c["status"]]
            icon = {"PASS": "✓", "WARN": "!", "FAIL": "✕"}[c["status"]]
            st.markdown(
                f'<div class="check-row"><div class="check-icon {icon_class}">{icon}</div>'
                f'<div><div class="check-name">{html.escape(c["name"])}</div>'
                f'<div class="check-detail">{html.escape(c["detail"])}</div></div></div>',
                unsafe_allow_html=True,
            )


def build_anomaly_events(anomaly_df):
    """
    Build persistent anomaly events.

    Fixed rule (unchanged):
        - Event must contain at least 60 minutes of continuous anomaly.
        - Current data sampling interval is 5 minutes.
        - Therefore 12 consecutive anomaly points = 60 minutes of coverage.
        - A gap greater than 7.5 minutes breaks the event.
    """
    if anomaly_df.empty:
        return pd.DataFrame()

    EVENT_DURATION_MIN = 60
    SAMPLE_INTERVAL_MIN = 5
    PERSISTENCE_MIN_POINTS = int(EVENT_DURATION_MIN / SAMPLE_INTERVAL_MIN)
    MAX_RUN_GAP_MIN = SAMPLE_INTERVAL_MIN * 1.5

    work = anomaly_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if work.empty:
        return pd.DataFrame()

    time_gap = work["timestamp"].diff().dt.total_seconds() / 60.0
    work["event_break"] = time_gap.isna() | (time_gap > MAX_RUN_GAP_MIN)
    work["run_id"] = work["event_break"].cumsum()

    run_sizes = work.groupby("run_id").size()
    persistent_run_ids = run_sizes[run_sizes >= PERSISTENCE_MIN_POINTS].index
    work = work[work["run_id"].isin(persistent_run_ids)].copy()

    if work.empty:
        return pd.DataFrame()

    events = []
    for event_number, (run_id, event_rows) in enumerate(work.groupby("run_id"), start=1):
        event_rows = event_rows.sort_values("timestamp").reset_index(drop=True)
        start_time = event_rows["timestamp"].min()
        end_time = event_rows["timestamp"].max()
        duration_min = ((end_time - start_time).total_seconds() / 60.0) + SAMPLE_INTERVAL_MIN

        event = {
            "event_id": f"Event {event_number}",
            "run_id": int(run_id),
            "start_time": start_time,
            "end_time": end_time,
            "duration_min": duration_min,
            "anomaly_count": len(event_rows),
            "is_persistent_event": True,
        }

        if "anomaly_score_ratio" in event_rows.columns:
            scores = pd.to_numeric(event_rows["anomaly_score_ratio"], errors="coerce")
            event["max_severity"] = scores.max()
            event["mean_severity"] = scores.mean()

        if "reconstruction_error" in event_rows.columns:
            errors = pd.to_numeric(event_rows["reconstruction_error"], errors="coerce")
            event["max_reconstruction_error"] = errors.max()
            event["mean_reconstruction_error"] = errors.mean()

        if "inverter_status" in event_rows.columns:
            mode = event_rows["inverter_status"].mode()
            event["dominant_status"] = mode.iloc[0] if len(mode) else None

        if "anomaly_type" in event_rows.columns:
            mode = event_rows["anomaly_type"].mode()
            event["anomaly_type"] = mode.iloc[0] if len(mode) else None

        events.append(event)

    return pd.DataFrame(events).reset_index(drop=True)


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
        start_date = st.date_input("Start date", value=min_date, min_value=min_date, max_value=max_date)

    with filter_cols[1]:
        end_date = st.date_input("End date", value=max_date, min_value=min_date, max_value=max_date)

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
# FILTER DATA
# ============================================================

# filtered_df: the scored evaluation data (has anomaly results) -- powers
# the KPIs, Overview chart, Anomalies table, and Investigate tab.
if start_date <= end_date:
    filtered_df = df[(df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)].copy()
    # filtered_trend_df: the full raw history -- powers the Trends tab so
    # dates before the evaluation period still show something (falls back
    # to filtered_df itself when trend_data.parquet isn't available).
    filtered_trend_df = trend_df[(trend_df["timestamp"].dt.date >= start_date) & (trend_df["timestamp"].dt.date <= end_date)].copy()
else:
    filtered_df = df.iloc[0:0].copy()  # empty until the dates are fixed
    filtered_trend_df = trend_df.iloc[0:0].copy()

if selected_inverter != "All" and "inverter_id" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["inverter_id"] == selected_inverter].copy()
if selected_inverter != "All" and "inverter_id" in filtered_trend_df.columns:
    filtered_trend_df = filtered_trend_df[filtered_trend_df["inverter_id"] == selected_inverter].copy()

if show_anomalies_only:
    filtered_df = filtered_df[filtered_df["anomaly_flag"]].copy()

if filtered_df.empty and filtered_trend_df.empty:
    st.warning("No observations match the current filters. Adjust the filters above.")

total_observations = len(filtered_df)
total_anomalies = int(filtered_df["anomaly_flag"].sum()) if total_observations else 0
anomaly_rate = (total_anomalies / total_observations * 100) if total_observations else None

anomaly_df = filtered_df[filtered_df["anomaly_flag"]].copy() if total_observations else filtered_df.copy()

# Build continuous anomaly events (selected inverter and date range only).
events_df = build_anomaly_events(anomaly_df)

# Population-level baseline comparison, reused by the Overview tab and by
# the AI evidence (so the numbers displayed and the numbers given to the
# AI are exactly the same).
comparison_metric_names = [m for m, _, _ in COMPARISON_METRICS]
baseline_reference = aggregate_baseline_reference(baseline_df, comparison_metric_names)
comparison_rows = compare_population_to_baseline(anomaly_df, baseline_reference)
key_obs = build_key_observations(anomaly_df, events_df, comparison_rows)

# ------------------------------------------------------------
# OVERVIEW / KPIs
# ------------------------------------------------------------
st.markdown("## Overview")
st.caption("A high-level view of inverter observations and detected anomalies for the selected period.")

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Total Observations", f"{total_observations:,}")
with kpi_cols[1]:
    st.metric("Anomalous Observations", f"{total_anomalies:,}")
with kpi_cols[2]:
    st.metric("Anomaly Rate", fmt_num(anomaly_rate, 2, "%"))
with kpi_cols[3]:
    st.metric("Persistent Events", f"{len(events_df):,}")

if total_observations > 0 and total_anomalies == 0:
    st.caption("✅ No anomalies found in this period — everything looks normal.")

# ============================================================
# HEALTHY BASELINE
# ============================================================

st.markdown("### Healthy Baseline")
st.caption(
    "Typical healthy operating range, derived from the trained model's healthy "
    "reference data. A reading outside this range is not automatically a fault."
)

baseline_rows = get_healthy_baseline_summary(baseline_df)
if baseline_rows:
    baseline_cols = st.columns(len(baseline_rows))
    for col, (label, value) in zip(baseline_cols, baseline_rows):
        with col:
            st.metric(label=label, value=value)
else:
    st.caption("No healthy baseline reference values are available for this dataset.")

# ============================================================
# TABS
# ============================================================

tab_overview, tab_ai, tab_events, tab_investigate, tab_trends = st.tabs(
    ["Overview", "AI Analysis", "Anomaly Events", "Investigation", "Trends"]
)

# ------------------------------------------------------------
# TAB: OVERVIEW (anomaly chart + key observations + baseline comparison)
# ------------------------------------------------------------
with tab_overview:
    st.markdown("### Anomaly Score Over Time")
    st.caption(
        "Higher points mean more unusual behavior relative to the trained model. "
        "Red dots are flagged anomalies; shaded bands mark persistent events."
    )

    if total_observations > 0 and "reconstruction_error" in filtered_df.columns:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"],
                y=filtered_df["reconstruction_error"],
                mode="lines",
                name="Anomaly Score",
                line=dict(color="#4C78A8", width=1.5),
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Anomaly score: %{y:.3f}<extra></extra>",
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
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>Anomaly score: %{y:.3f}<extra>Flagged</extra>",
                )
            )
        for _, ev in events_df.iterrows():
            fig.add_vrect(x0=ev["start_time"], x1=ev["end_time"], fillcolor="#E45756", opacity=0.08, line_width=0)

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Anomaly Score",
            yaxis_type="log",
            hovermode="x unified",
            height=380,
            template="plotly_white",
            margin=dict(t=20, l=55, r=25, b=45),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, width="stretch")
        if not events_df.empty:
            st.caption("Shaded bands mark persistent anomaly events (≥60 min of continuous anomalous readings).")
    else:
        st.info("No anomaly score data available for the selected period.")

    st.markdown("### Key Observations")
    st.caption("What to investigate first, based only on the evidence above.")
    render_key_observations(key_obs)

    st.markdown("### Anomalies vs Healthy Baseline")
    st.caption(
        "Average values observed during anomalies compared with the typical healthy "
        "range. Supporting evidence only — it does not prove a fault or its cause."
    )
    if comparison_rows:
        comparison_table = pd.DataFrame(comparison_rows).rename(
            columns={
                "label": "Variable",
                "observed": "Observed during anomalies",
                "typical_range": "Typical healthy range",
                "status": "Status",
            }
        )
        st.dataframe(comparison_table, width="stretch", hide_index=True)
    else:
        st.caption("No overlapping variables between the anomaly data and the healthy baseline.")

    st.markdown("### Data Quality")
    render_optional_data_validation(filtered_df, label="the selected period's data")

# ------------------------------------------------------------
# TAB: AI ANALYSIS (overall population explanation)
# ------------------------------------------------------------
with tab_ai:
    st.markdown("## Overall Anomaly Explanation")
    st.caption(
        "AI analysis of the recurring patterns across all detected anomalies "
        "in the selected period. No individual anomaly is selected."
    )

    if len(anomaly_df) == 0:
        st.info("No anomalies were detected in the selected period, so there is no anomaly pattern to explain.")
    else:
        header_col, button_col = st.columns([5.5, 1.5])
        with header_col:
            st.markdown('<div class="ai-title">Why are anomalies occurring?</div>', unsafe_allow_html=True)
        with button_col:
            regenerate_overall = st.button(
                "Regenerate",
                type="primary",
                use_container_width=True,
                help="Generate a fresh overall explanation from all detected anomalies.",
            )

        overall_cache_key = f"overall_{selected_inverter}_{start_date}_{end_date}"
        cached_overall = st.session_state["ai_explanations"].get(overall_cache_key)

        if regenerate_overall or cached_overall is None:
            evidence = build_population_evidence(
                anomaly_df, events_df, baseline_df, start_date, end_date, selected_inverter
            )
            with st.spinner("Analyzing all detected anomalies..."):
                try:
                    explanation = generate_overall_ai_explanation(evidence)
                    checks = validate_overall_explanation(explanation, evidence)
                    cached_overall = {
                        "explanation": explanation,
                        "evidence": evidence,
                        "checks": checks,
                        "error": None,
                    }
                except Exception as e:
                    cached_overall = {"explanation": None, "evidence": None, "checks": None, "error": str(e)}
            st.session_state["ai_explanations"][overall_cache_key] = cached_overall

        if cached_overall.get("error"):
            st.error(f"AI explanation failed: {cached_overall['error']}")
        elif cached_overall.get("explanation"):
            render_ai_explanation(cached_overall["explanation"])
            render_evidence_consistency(cached_overall.get("checks"))
        else:
            st.warning("AI did not return an overall explanation.")

# ------------------------------------------------------------
# TAB: ANOMALY EVENTS
# ------------------------------------------------------------
with tab_events:
    st.markdown("## Detected Anomaly Events")
    st.caption("Persistent anomaly events detected in the selected period (≥60 minutes of continuous anomalous readings).")

    if len(events_df) > 0:
        event_table = events_df.copy()
        event_table["Event"] = [f"Event {i + 1}" for i in range(len(event_table))]

        display_columns = [
            "Event", "start_time", "end_time", "duration_min", "anomaly_count",
            "max_severity", "mean_severity", "dominant_status", "anomaly_type",
        ]
        display_columns = [c for c in display_columns if c in event_table.columns]

        friendly_names = {
            "start_time": "Start Time",
            "end_time": "End Time",
            "duration_min": "Duration (min)",
            "anomaly_count": "Anomaly Points",
            "max_severity": "Max Severity",
            "mean_severity": "Mean Severity",
            "dominant_status": "Dominant Status",
            "anomaly_type": "Dominant Anomaly Type",
        }

        table = event_table[display_columns].rename(columns=friendly_names).sort_values("Start Time")
        for col in ["Max Severity", "Mean Severity", "Duration (min)"]:
            if col in table.columns:
                table[col] = pd.to_numeric(table[col], errors="coerce").round(2)

        st.dataframe(table, width="stretch", hide_index=True)
        st.caption("Severity values reflect the anomaly score, not a probability of failure.")
    else:
        if total_observations > 0:
            st.success("✅ No anomaly events were detected in this period.")
        else:
            st.info("No readings in this date range. Try a different range above.")

# ------------------------------------------------------------
# TAB: INVESTIGATION (per-event detail)
# ------------------------------------------------------------
with tab_investigate:
    st.markdown("## Investigate a Specific Anomaly Event")
    st.caption(
        "Detailed evidence for one persistent anomaly event, for operator investigation. "
        "Feature contributions and baseline comparisons are supporting evidence only — "
        "they do not prove a physical cause."
    )

    if events_df.empty:
        st.info("No persistent anomaly events in the selected period to investigate.")
    else:
        events_sorted = events_df.sort_values("start_time").reset_index(drop=True)
        event_labels = [
            f"Event {i + 1}: {fmt_time(row['start_time'])} → {fmt_time(row['end_time'])}"
            for i, row in events_sorted.iterrows()
        ]
        chosen_idx = st.selectbox("Select an event", range(len(event_labels)), format_func=lambda i: event_labels[i])
        event_row = events_sorted.iloc[chosen_idx]
        reference_time = event_row["start_time"]

        st.markdown(
            f'<div class="event-summary-card"><div class="event-summary-title">'
            f'{fmt_time(event_row["start_time"])} → {fmt_time(event_row["end_time"])}'
            f'<span class="event-badge">{fmt_num(event_row.get("duration_min"), 0, " min")}</span></div>'
            f'{event_row.get("anomaly_count", 0)} anomalous readings in this event.</div>',
            unsafe_allow_html=True,
        )

        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.markdown("#### Operating Context")
            context = get_operating_context(reference_time)
            if context.get("error"):
                st.caption(context["error"])
            else:
                st.json(context)

        with detail_cols[1]:
            st.markdown("#### Feature Contributions")
            contrib = get_feature_contributions(
                reference_time, selected_inverter if selected_inverter != "All" else None
            )
            if contrib.get("error"):
                st.caption(contrib["error"])
            elif contrib.get("contributions"):
                for item in contrib["contributions"][:5]:
                    st.markdown(
                        f"- **{item['feature']}**: {fmt_num(item['contribution_pct'], 1, '%')} "
                        "contribution to unusual reconstruction error"
                    )
                st.caption(
                    "These variables contributed most to unusual reconstruction error — "
                    "this does not identify which one caused the anomaly."
                )
            else:
                st.caption("No feature-contribution data available for this event.")

        st.markdown("#### Comparison to Healthy Baseline")
        baseline_context = get_baseline_context(reference_time)
        if baseline_context.get("error"):
            st.caption(baseline_context["error"])
        else:
            ref_rows = []
            for metric, values in baseline_context.get("reference", {}).items():
                ref_rows.append({
                    "Metric": metric,
                    "Typical healthy range": f"{fmt_num(values.get('typical_low_q10'))}–{fmt_num(values.get('typical_high_q90'))}",
                    "Typical (median)": fmt_num(values.get("median")),
                })
            if ref_rows:
                st.dataframe(pd.DataFrame(ref_rows), width="stretch", hide_index=True)
                st.caption("This is a comparison reference only. It does not establish cause.")
            else:
                st.caption("No matching healthy reference group for these operating conditions.")

        st.markdown("#### 24-Hour Trend Before This Event")
        pre_trend = get_pre_anomaly_trend(reference_time, hours=24)
        if pre_trend.get("error"):
            st.caption(pre_trend["error"])
        else:
            st.caption(
                f"{pre_trend['observations']} observations between "
                f"{fmt_time(pre_trend['window_start'])} and {fmt_time(pre_trend['window_end'])}."
            )
            stat_rows = [{"Variable": k, **v} for k, v in pre_trend.get("statistics", {}).items()]
            if stat_rows:
                st.dataframe(pd.DataFrame(stat_rows), width="stretch", hide_index=True)
            else:
                st.caption("No pre-event trend statistics available.")

        st.markdown("#### Data Quality for This Event")
        event_window_rows = filtered_df[
            (filtered_df["timestamp"] >= event_row["start_time"]) & (filtered_df["timestamp"] <= event_row["end_time"])
        ]
        render_optional_data_validation(event_window_rows, label="this event's readings")

# ------------------------------------------------------------
# TAB: TRENDS (raw history, including pre-evaluation period)
# ------------------------------------------------------------
with tab_trends:
    st.markdown("## Historical Trends")
    if not has_trend_data:
        st.caption("`trend_data.parquet` not found — showing the evaluation-period data only.")

    if filtered_trend_df.empty:
        st.info("No historical readings in this date range.")
    else:
        trend_metric_options = [
            c for c in [
                "ac_power_kw", "dc_power_kw", "inverter_temperature_c",
                "ambient_temperature_c", "efficiency_pct", "dc_current_a", "ac_current_a",
            ]
            if c in filtered_trend_df.columns
        ]
        if trend_metric_options:
            chosen_metrics = st.multiselect(
                "Variables to plot", trend_metric_options, default=trend_metric_options[:2]
            )
            if chosen_metrics:
                tfig = go.Figure()
                for m in chosen_metrics:
                    tfig.add_trace(
                        go.Scatter(x=filtered_trend_df["timestamp"], y=filtered_trend_df[m], mode="lines", name=m)
                    )
                if has_trend_data:
                    tfig.add_vrect(
                        x0=eval_min_date, x1=eval_max_date,
                        fillcolor="#0F3554", opacity=0.04, line_width=0,
                        annotation_text="Scored evaluation period", annotation_position="top left",
                    )
                tfig.update_layout(
                    height=380, template="plotly_white", hovermode="x unified",
                    margin=dict(t=30, l=55, r=25, b=45),
                )
                st.plotly_chart(tfig, width="stretch")
                st.caption(
                    "Dates outside the scored evaluation period show raw readings only — "
                    "they were not evaluated by the anomaly detector."
                )
        else:
            st.info("No plottable trend variables found in this dataset.")
