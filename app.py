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


st.session_state.setdefault("overall_ai_explanation", None)

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


def make_healthy_reference(source, start_time, days=7):
    start_time=pd.to_datetime(start_time)
    ref_start=start_time-timedelta(days=days)
    ref=source[(source["timestamp"]>=ref_start)&(source["timestamp"]<start_time)].copy()
    if "anomaly_flag" in ref.columns: ref=ref[~ref["anomaly_flag"].fillna(False)]
    if "inverter_status" in ref.columns: ref=ref[ref["inverter_status"].astype(str).str.upper()!="STANDBY"]
    return ref,ref_start,start_time

def calculate_overall_evidence(healthy_df, anomaly_df, events_df):
    features=["dc_power_kw","ac_power_kw","dc_current_a","ac_current_a","inverter_temperature_c","efficiency_pct","power_factor","frequency_hz"]
    rows=[]
    for feature in features:
        if feature not in healthy_df.columns or feature not in anomaly_df.columns: continue
        h=pd.to_numeric(healthy_df[feature],errors="coerce").dropna(); a=pd.to_numeric(anomaly_df[feature],errors="coerce").dropna()
        if len(h)<10 or a.empty: continue
        q=h.quantile([.01,.05,.25,.5,.75,.95,.99]); h50=float(q.loc[.5]); a50=float(a.median())
        change=(a50-h50)/h50*100 if h50 else None; below=int((a<q.loc[.05]).sum()); above=int((a>q.loc[.95]).sum())
        unit="kW" if feature.endswith("_kw") else "A" if feature.endswith("_a") else "°C" if feature.endswith("_c") else "%" if feature.endswith("_pct") else "Hz" if feature.endswith("_hz") else ""
        rows.append({"feature":feature,"unit":unit,"healthy_count":len(h),"abnormal_count":len(a),"healthy_p1":float(q.loc[.01]),"healthy_p5":float(q.loc[.05]),"healthy_p25":float(q.loc[.25]),"healthy_p50":h50,"healthy_p75":float(q.loc[.75]),"healthy_p95":float(q.loc[.95]),"healthy_p99":float(q.loc[.99]),"abnormal_median":a50,"abnormal_min":float(a.min()),"abnormal_max":float(a.max()),"median_change_pct":change,"outside_p5_p95_count":below+above,"outside_p5_p95_pct":(below+above)/len(a)*100,"below_p1_count":int((a<q.loc[.01]).sum()),"above_p99_count":int((a>q.loc[.99]).sum())})
    if not rows: return {"metrics":[],"top_deviations":[],"healthy_reference_rows":len(healthy_df),"anomaly_rows":len(anomaly_df),"episode_count":len(events_df),"total_anomaly_duration_min":float(events_df["duration_min"].sum()) if not events_df.empty else 0}
    ranked=sorted(rows,key=lambda x:abs(x["median_change_pct"]) if x["median_change_pct"] is not None else 0,reverse=True)
    return {"metrics":rows,"top_deviations":ranked[:6],"healthy_reference_rows":len(healthy_df),"anomaly_rows":len(anomaly_df),"episode_count":len(events_df),"total_anomaly_duration_min":float(events_df["duration_min"].sum()) if not events_df.empty else 0}

@st.cache_resource
def get_groq_client():
    key=os.getenv("GROQ_API_KEY")
    if not key:
        try: key=st.secrets["groq"]["api_key"]
        except Exception: key=None
    return Groq(api_key=key) if key else None

def generate_overall_ai_explanation(evidence):
    client=get_groq_client()
    if client is None: raise RuntimeError("GROQ_API_KEY is not configured. Set GROQ_API_KEY in the environment or Streamlit secrets.")
    prompt="""You are an explanation assistant inside a solar inverter anomaly detection dashboard.
Generate ONE overall explanation for ALL anomaly episodes in the selected period. Do not explain events separately.
Return ONLY valid JSON with exactly these fields:
{\"headline\":\"one short sentence\",\"summary\":\"2-3 simple sentences\",\"key_findings\":[\"fact 1\",\"fact 2\",\"fact 3\"],\"recommended_actions\":[\"action 1\",\"action 2\",\"action 3\"]}
Rules: use only supplied evidence; use simple plant-operator language; avoid ML jargon; compare anomalous medians with previous 7-day healthy percentiles; for low power use P5/P1; mention median percentage change and outside-P5-P95 percentage when useful; statistical deviation is evidence of unusual behavior, not proof of a physical fault/root cause; do not claim a definite physical cause; recommendations must be practical checks; no markdown bold.
EVIDENCE:
"""+json.dumps(evidence,default=str,ensure_ascii=False)
    try:
        r=client.chat.completions.create(model="openai/gpt-oss-20b",messages=[{"role":"user","content":prompt}],reasoning_effort="low",include_reasoning=False,temperature=0.2,max_completion_tokens=900,response_format={"type":"json_object"})
    except Exception as e: raise RuntimeError(f"Groq request failed: {e}") from e
    content=getattr(r.choices[0].message,"content",None)
    if not content: raise RuntimeError("Groq returned no text content.")
    try: data=json.loads(re.sub(r"^```(?:json)?\s*|\s*```$","",str(content).strip()))
    except Exception as e: raise RuntimeError(f"Groq returned invalid JSON: {e}") from e
    missing=[k for k in ["headline","summary","key_findings","recommended_actions"] if k not in data]
    if missing: raise RuntimeError(f"Groq JSON is missing required fields: {missing}")
    return data

def render_overall_ai(data):
    esc=lambda x:html.escape(str(x)) if x is not None else ""
    findings="".join(f"<li>{esc(x)}</li>" for x in data.get("key_findings",[]) if x)
    actions="".join(f"<li>{esc(x)}</li>" for x in data.get("recommended_actions",[]) if x)
    st.markdown(f"<div class=\"ai-card\"><div class=\"ai-alert-box\"><div class=\"ai-alert-icon\">⚠️</div><div><div class=\"ai-alert-title\">{esc(data.get('headline',''))}</div><div class=\"ai-alert-desc\">{esc(data.get('summary',''))}</div></div></div><div class=\"ai-section\"><div class=\"ai-section-header\">🔍 What the data shows</div><ul class=\"ai-bullets\">{findings}</ul></div><div class=\"ai-section\"><div class=\"ai-section-header\">🔧 Recommended action</div><ul class=\"ai-bullets\">{actions}</ul></div><div class=\"ai-footer-note\"><span>ℹ️</span><span>This is a statistical explanation of overall anomaly behavior. It does not confirm a physical fault or root cause.</span></div></div>",unsafe_allow_html=True)

def validate_overall_ai(explanation,evidence):
    checks=[]
    def add(n,s,d): checks.append({"name":n,"status":s,"detail":d})
    parts=[str(explanation.get("headline","")),str(explanation.get("summary",""))]+[str(x) for x in explanation.get("key_findings",[])]+[str(x) for x in explanation.get("recommended_actions",[])]
    text=" ".join(parts).lower(); known=[]
    for m in evidence.get("metrics",[]):
        for k in ["healthy_p1","healthy_p5","healthy_p25","healthy_p50","healthy_p75","healthy_p95","healthy_p99","abnormal_median","abnormal_min","abnormal_max","median_change_pct","outside_p5_p95_pct"]:
            v=m.get(k)
            if isinstance(v,(int,float)) and pd.notna(v): known.append(float(v))
    claims=re.findall(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)\s*(?:kW|kw|A|°C|C|%|Hz|minutes?|mins?|min)\b"," ".join(parts))
    unmatched=[x for x in claims if not any(abs(float(x)-k)<=max(2,abs(k)*.02) for k in known)]
    add("Numerical values","PASS" if not unmatched else "WARN","Reported measurements match supplied evidence." if not unmatched else "Unmatched numeric claims: "+", ".join(unmatched[:5]))
    issues=[]
    for feature,label in [("dc_power_kw","dc power"),("ac_power_kw","ac power")]:
        row=next((m for m in evidence.get("metrics",[]) if m.get("feature")==feature),None)
        if not row or row.get("median_change_pct") is None or label not in text: continue
        c=float(row["median_change_pct"]); words=["decreased","fell","dropped","lower","reduced","declined","low"] if c<0 else ["increased","rose","higher","grew","climbed"] if c>0 else ["stable","unchanged","similar"]
        if not any(w in text for w in words): issues.append(f"{label} direction does not match calculated change")
    add("Power change direction","PASS" if not issues else "WARN","Power direction is consistent with calculated median change." if not issues else "; ".join(issues))
    ref=any(x in text for x in ["healthy range","healthy distribution","p5","p95","p1","p99","outside","percentile"])
    add("Healthy-reference claims","PASS" if ref else "WARN","Explanation uses the healthy-reference comparison." if ref else "Explanation does not explicitly mention the healthy-reference comparison.")
    bad=[x for x in ["definitely caused by","confirmed fault","proves the fault","root cause is","caused by"] if x in text]
    add("Causal certainty","FAIL" if bad else "PASS","No unsupported causal certainty detected." if not bad else "Unsupported causal wording: "+", ".join(bad))
    return checks

def render_validation(checks):
    p=sum(x["status"]=="PASS" for x in checks); w=sum(x["status"]=="WARN" for x in checks); f=sum(x["status"]=="FAIL" for x in checks); verdict="FAIL" if f else "NEEDS REVIEW" if w else "PASS"; cls="validation-fail" if f else "validation-warn" if w else "validation-pass"
    st.markdown(f"<div class=\"validation-card\"><div class=\"validation-header\"><div class=\"validation-title\">Explanation Validation</div><span class=\"validation-pill {cls}\">{p}/{len(checks)} checks passed</span></div></div>",unsafe_allow_html=True)
    for c in checks:
        icon="✓" if c["status"]=="PASS" else "!" if c["status"]=="WARN" else "×"; rc="check-pass" if c["status"]=="PASS" else "check-warn" if c["status"]=="WARN" else "check-fail"
        st.markdown(f"<div class=\"check-row\"><span class=\"check-icon {rc}\">{icon}</span><div><div class=\"check-name\">{html.escape(c['name'])}</div><div class=\"check-detail\">{html.escape(c['detail'])}</div></div></div>",unsafe_allow_html=True)
    st.caption(f"Verdict: {verdict} · {p} passed · {w} review · {f} failed. Validation checks explanation consistency, not physical root cause.")

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

def build_anomaly_events(anomaly_df, gap_minutes=6):
    """
    Group continuous anomaly observations into anomaly events.

    One event = continuous occurrence of anomaly.
    A new event starts when the gap between two anomaly
    observations is greater than gap_minutes.
    """

    if anomaly_df.empty:
        return pd.DataFrame()

    work = anomaly_df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"])
    work = work.sort_values("timestamp").reset_index(drop=True)

    time_gap = work["timestamp"].diff().dt.total_seconds() / 60

    work["event_break"] = (
        time_gap.isna() |
        (time_gap > gap_minutes)
    )

    work["event_number"] = work["event_break"].cumsum()

    events = []

    for event_number, event_rows in work.groupby("event_number"):

        start_time = event_rows["timestamp"].min()
        end_time = event_rows["timestamp"].max()

        event = {
            "event_id": f"Event {int(event_number)}",
            "start_time": start_time,
            "end_time": end_time,
            "duration_min": (
                end_time - start_time
            ).total_seconds() / 60,
            "anomaly_count": len(event_rows),
        }

        if "anomaly_score_ratio" in event_rows.columns:
            event["max_severity"] = event_rows["anomaly_score_ratio"].max()
            event["mean_severity"] = event_rows["anomaly_score_ratio"].mean()

        if "reconstruction_error" in event_rows.columns:
            event["max_reconstruction_error"] = (
                event_rows["reconstruction_error"].max()
            )
            event["mean_reconstruction_error"] = (
                event_rows["reconstruction_error"].mean()
            )

        if "inverter_status" in event_rows.columns:
            mode = event_rows["inverter_status"].mode()
            event["dominant_status"] = (
                mode.iloc[0] if len(mode) else None
            )

        if "anomaly_type" in event_rows.columns:
            mode = event_rows["anomaly_type"].mode()
            event["anomaly_type"] = (
                mode.iloc[0] if len(mode) else None
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
events_df = build_anomaly_events(
    anomaly_df,
    gap_minutes=6
)


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
# OVERALL AI EXPLANATION
# ------------------------------------------------------------
st.markdown("## Overall AI Explanation")
st.caption("One explanation for all ML-detected anomaly episodes in the selected period, using the previous 7 days as the healthy statistical reference.")

if len(anomaly_df) == 0:
    st.info("No anomalies are available for an overall AI explanation in the selected period.")
else:
    analysis_start=pd.to_datetime(anomaly_df["timestamp"].min())
    healthy_df,healthy_start,healthy_end=make_healthy_reference(trend_df if has_trend_data else df,analysis_start,days=7)
    if selected_inverter != "All" and "inverter_id" in healthy_df.columns:
        healthy_df=healthy_df[healthy_df["inverter_id"]==selected_inverter].copy()
    overall_evidence=calculate_overall_evidence(healthy_df,anomaly_df,events_df)
    c1,c2,c3,c4=st.columns(4)
    with c1: st.metric("Anomaly Observations",f"{len(anomaly_df):,}")
    with c2: st.metric("Anomaly Episodes",f"{len(events_df):,}")
    with c3: st.metric("Healthy Reference",f"{len(healthy_df):,}")
    with c4: st.metric("Anomaly Duration",fmt_num(overall_evidence.get("total_anomaly_duration_min",0),0," min"))
    st.caption(f"Healthy reference window: {fmt_time(healthy_start)} → {fmt_time(healthy_end)}. The AI receives percentile statistics, not the full raw dataset.")
    if overall_evidence["metrics"]:
        table=pd.DataFrame(overall_evidence["metrics"])[["feature","healthy_p5","healthy_p50","healthy_p95","abnormal_median","median_change_pct","outside_p5_p95_pct"]].copy()
        table["feature"]=table["feature"].str.replace("_"," ",regex=False).str.title()
        table=table.rename(columns={"feature":"Feature","healthy_p5":"Healthy P5","healthy_p50":"Healthy P50","healthy_p95":"Healthy P95","abnormal_median":"Abnormal Median","median_change_pct":"Median Change %","outside_p5_p95_pct":"Outside P5-P95 %"})
        st.dataframe(table.round(2),width="stretch",hide_index=True)
        st.caption("P5-P95 is the main healthy comparison band. P1/P99 are also included in the evidence for extreme-tail context.")
    else:
        st.warning("Not enough healthy reference data or anomaly observations are available for percentile comparison.")
    bcol,scol=st.columns([1.5,5])
    with bcol: regenerate=st.button("Generate Explanation",type="primary",use_container_width=True)
    with scol: st.caption(f"All {len(anomaly_df):,} anomalous observations across {len(events_df):,} episodes are used together. No event selector is used for AI generation.")
    if regenerate or st.session_state["overall_ai_explanation"] is None:
        with st.spinner("Comparing anomaly behavior with the previous 7-day healthy reference..."):
            try:
                result=generate_overall_ai_explanation(overall_evidence)
                st.session_state["overall_ai_explanation"]={"explanation":result,"evidence":overall_evidence,"error":None}
            except Exception as exc:
                st.session_state["overall_ai_explanation"]={"explanation":None,"evidence":overall_evidence,"error":str(exc)}
    cached=st.session_state["overall_ai_explanation"]
    if cached and cached.get("error"): st.error(f"AI explanation failed: {cached['error']}")
    elif cached and cached.get("explanation"):
        render_overall_ai(cached["explanation"])
        render_validation(validate_overall_ai(cached["explanation"],cached["evidence"]))

# ------------------------------------------------------------
# DETECTED ANOMALIES
# ------------------------------------------------------------
st.markdown("## Detected Anomaly Episodes")
st.caption("Continuous anomaly observations grouped for display. These episodes are not individually sent to the AI.")

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
