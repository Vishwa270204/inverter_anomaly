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

    try:

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            reasoning_effort="low",
            include_reasoning=False,
            temperature=0.2,
            max_completion_tokens=768,
        )

    except Exception as e:

        raise RuntimeError(
            f"Groq request failed: {e}"
        ) from e

    raw_content = getattr(
        response.choices[0].message,
        "content",
        None,
    )

    if raw_content is None:
        raise RuntimeError(
            "Groq returned no text content."
        )

    raw_content = str(raw_content).strip()
    # Strip accidental ```json fences some models add despite instructions.
    raw_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content).strip()

    if not raw_content:
        raise RuntimeError(
            "Groq returned an empty explanation."
        )

    try:
        parsed = json.loads(raw_content)
    except Exception as e:
        raise RuntimeError(f"Groq did not return valid JSON: {e}") from e

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

        st.markdown(
        textwrap.dedent(f'''
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
        '''),
        unsafe_allow_html=True,
    )
# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #0F3554 0%, #164A73 100%);
        padding: 1.25rem 1.6rem;
        border-radius: 14px;
        margin-bottom: 1.1rem;
        box-shadow: 0 4px 14px rgba(15, 53, 84, 0.16);
        display: flex;
        align-items: center;
        gap: 0.7rem;
    ">
        <div style="
            background: rgba(255,255,255,0.12);
            width: 54px; height: 54px;
            border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.65rem;
            flex-shrink: 0;
        ">⚡</div>
        <div>
            <div style="color:#FFFFFF; font-size:1.8rem; font-weight:700; line-height:1.2;">
                Inverter Anomaly Detection
            </div>
            <div style="color:#C9DDED; font-size:1rem; margin-top:0.22rem; line-height:1.4;">
                Spot unusual inverter behavior early
            </div>
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
# AI EXPLANATION
# --------------------------------------------------------
if len(events_df) > 0:

    events_df = events_df.sort_values("start_time").reset_index(drop=True)

    selected_event_index = st.selectbox(
        "Select an anomaly event for AI explanation",
        range(len(events_df)),
        format_func=lambda x: (
            f"Event {x + 1} | "
            f"{fmt_time(events_df.loc[x, 'start_time'])} → "
            f"{fmt_time(events_df.loc[x, 'end_time'])}"
        ),
        key="ai_event_selector",
    )

    selected_event = events_df.loc[selected_event_index]

    event_rows = anomaly_df[
        (anomaly_df["timestamp"] >= selected_event["start_time"]) &
        (anomaly_df["timestamp"] <= selected_event["end_time"])
    ].copy()

ai_header_col, ai_button_col = st.columns([7, 1.35])

with ai_header_col:
    st.markdown(
        '<div class="ai-title">AI Explanation</div>',
        unsafe_allow_html=True,
    )

with ai_button_col:
    regenerate = st.button(
        "Regenerate",
        type="primary",
        use_container_width=True,
        help="Generate a fresh explanation for this anomaly.",
    )

event_key = clean_value(selected_event.get("event_id"))

anomaly_key = f"{event_key}"
cached = st.session_state["ai_explanations"].get(anomaly_key)

if regenerate or cached is None:

    with st.spinner("Generating explanation from the anomaly evidence..."):

        try:
            ai_explanation, ai_evidence = generate_ai_explanation(
                selected_event,
                event_rows
            )
            cached = {
                "explanation": ai_explanation,
                "evidence": ai_evidence,
                "error": None,
            }

        except Exception as e:

            cached = {
                "explanation": None,
                "evidence": None,
                "error": str(e),
            }

        st.session_state["ai_explanations"][anomaly_key] = cached


# ------------------------------------------------------------
# ALWAYS RENDER AN AI RESULT AREA
# ------------------------------------------------------------

if cached is None:
    st.info("Select an anomaly event to generate an AI explanation.")

elif cached.get("error"):

    st.error(
        f"AI explanation failed: {cached['error']}"
    )

elif cached.get("explanation"):
    render_ai_explanation(cached["explanation"])

    ev_meta = cached["evidence"].get("event", {})
    event_label = (
        f"{ev_meta.get('event_id', event_key)} "
        f"({ev_meta.get('start_time', '?')} → {ev_meta.get('end_time', '?')})"
    )
    st.caption(f"Validating data for **{event_label}**")

    # --------------------------------------------------------
    # DATA-FETCH VALIDATION -- is the evidence itself correct?
    # Reads the event window straight out of evidence["event"] (not a
    # separately-passed selected_event), so this always matches whatever
    # event actually produced this cached evidence.
    # --------------------------------------------------------
    data_results = validate_event_data(
        evidence=cached["evidence"],
        df=df,
        trend_df=trend_df if has_trend_data else None,
        baseline_df=baseline_df,
    )
    data_verdict = summarize_data(data_results)
    badge = {"PASS": "✅", "NEEDS_REVIEW": "⚠️", "FAIL": "❌"}[data_verdict["verdict"]]

    # Always-visible summary row (no click needed to see the verdict).
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Verdict", f"{badge} {data_verdict['verdict']}")
    m2.metric("Passed", data_verdict["pass_count"])
    m3.metric("Failed", data_verdict["fail_count"])
    m4.metric("Needs review", data_verdict["warn_count"])

    # Always-visible table (no expander) -- only shows FAIL/WARN rows by
    # default so it stays short; flip show_all to see every check.
    show_all = st.checkbox("Show all checks (including passed)", value=False)
    rows_to_show = data_results if show_all else [
        r for r in data_results if r["status"] != "PASS"
    ]

    if rows_to_show:
        table = pd.DataFrame(rows_to_show)[["status", "rule", "detail"]]
        table.columns = ["Status", "Check", "Detail"]

        def _highlight(row):
            color = {"FAIL": "#FDECEA", "WARN": "#FFF7E0", "PASS": "#EAF7EE"}[row["Status"]]
            return [f"background-color: {color}"] * len(row)

        st.dataframe(
            table.style.apply(_highlight, axis=1),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("All data checks passed for this event.") 
else:

    st.warning(
        "AI did not return an explanation for this anomaly event."
    )

# ------------------------------------------------------------
# INVESTIGATE AN ANOMALY
# ------------------------------------------------------------
st.markdown("## Investigate an Anomaly")
st.caption("Select one detected anomaly to inspect its preceding trend, contributing parameters, and AI explanation.")

if len(events_df) > 0:

    selected_event = events_df.loc[selected_event_index]

    event_rows = anomaly_df[
        (anomaly_df["timestamp"] >= selected_event["start_time"]) &
        (anomaly_df["timestamp"] <= selected_event["end_time"])
    ].copy()

    inv_cols = st.columns(3)
    with inv_cols[0]:
        st.metric(
            "Event Time",
            f"{fmt_time(selected_event.get('start_time'))} → "
            f"{fmt_time(selected_event.get('end_time'))}"
        )
    with inv_cols[1]:
        st.metric(
            "Event Duration",
            fmt_num(selected_event.get("duration_min"), 1, " min")
        )

    with inv_cols[2]:
        st.metric(
            "Anomaly Points",
            f"{int(selected_event.get('anomaly_count', 0)):,}"
        )

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("### What Happened Before")
        st.caption("Power and temperature in the 24 hours leading up to the anomaly.")

        event_start = pd.to_datetime(selected_event["start_time"])
        event_end = pd.to_datetime(selected_event["end_time"])
        trend_window, window_start = get_trend_window(trend_df,event_start,hours_back=24)

        if len(trend_window) > 1:
            fig_trend = go.Figure()
            if "dc_power_kw" in trend_window.columns:
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_window["timestamp"],
                        y=trend_window["ac_power_kw"],
                        mode="lines",
                        name="DC Power (kW)",
                        yaxis="y",
                        line=dict(color="#4C78A8"),
                    )
                )
            if "inverter_temperature_c" in trend_window.columns:
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_window["timestamp"],
                        y=trend_window["inverter_temperature_c"],
                        mode="lines",
                        name="Temp (°C)",
                        yaxis="y2",
                        line=dict(color="#E45756"),
                    )
                )
            fig_trend.add_vrect(
                x0=event_start,
                x1=event_end,
                fillcolor="red",
                opacity=0.12,
                line_width=1,
            )
            fig_trend.update_layout(
                xaxis=dict(title="Time"),
                yaxis=dict(title="AC Power (kW)", side="left"),
                yaxis2=dict(title="Temp (°C)", overlaying="y", side="right"),
                hovermode="x unified",
                height=360,
                template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
                margin=dict(l=55, r=55, t=45, b=45),
            )
            st.plotly_chart(fig_trend, width="stretch")
        else:
            st.info(
                "Not enough history before this anomaly to draw a trend — "
                "it is at or very near the start of the available data."
            )

    with detail_col2:
        st.markdown("### What Contributed Most")
        st.caption("Parameters that contributed most to the unusual pattern — not confirmed causes.")

        contribution_cols = get_contribution_cols(anomaly_df)
        if contribution_cols:
            contrib_values = (event_rows[contribution_cols].mean().dropna())
            if len(contrib_values) > 0:
                contrib_plot_df = pd.DataFrame(
                    {
                        "feature": [c.replace("_contribution_pct", "") for c in contrib_values.index],
                        "contribution_pct": contrib_values.values,
                    }
                ).sort_values("contribution_pct", ascending=True).tail(5)

                fig_contrib = go.Figure()
                fig_contrib.add_trace(
                    go.Bar(
                        x=contrib_plot_df["contribution_pct"],
                        y=contrib_plot_df["feature"],
                        orientation="h",
                        marker_color="#4C78A8",
                    )
                )
                fig_contrib.update_layout(
                    xaxis_title="Contribution (%)",
                    yaxis_title="",
                    height=360,
                    template="plotly_white",
                    margin=dict(t=15, l=10, r=10, b=45),
                )
                st.plotly_chart(fig_contrib, width="stretch")
            else:
                st.info("No contribution data available for this observation.")
        else:
            st.info("No contribution data available for this observation.")

    st.divider()


else:
    st.info(
        "Nothing to investigate here — there are no anomalies in the selected "
        "date range. Widen the date range or clear **Show anomalies only** "
        "to bring up more data."
    )

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
