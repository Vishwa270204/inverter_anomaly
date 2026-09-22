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

    </style>
    """,
    unsafe_allow_html=True,
)


st.session_state.setdefault("ai_explanations", {})

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


def get_anomaly_details(timestamp, inverter_id=None):
    target = pd.to_datetime(timestamp)
    source = df.copy()
    if inverter_id is not None and "inverter_id" in source.columns:
        source = source[source["inverter_id"].astype(str) == str(inverter_id)]
    if source.empty:
        return {"error": "No matching inverter data found."}
    idx = (source["timestamp"] - target).abs().idxmin()
    return row_to_dict(source.loc[idx])


def get_pre_anomaly_trend(timestamp, inverter_id=None, hours=24):
    target = pd.to_datetime(timestamp)
    start = target - timedelta(hours=float(hours))
    source = trend_df[(trend_df["timestamp"] >= start) & (trend_df["timestamp"] <= target)].copy()
    if inverter_id is not None and "inverter_id" in source.columns:
        source = source[source["inverter_id"].astype(str) == str(inverter_id)]
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


def get_operating_context(timestamp, inverter_id=None):
    target = pd.to_datetime(timestamp)
    source = df.copy()
    if inverter_id is not None and "inverter_id" in source.columns:
        source = source[source["inverter_id"].astype(str) == str(inverter_id)]
    if source.empty:
        return {"error": "No matching data found."}
    idx = (source["timestamp"] - target).abs().idxmin()
    row = source.loc[idx]
    wanted = [
        "timestamp",
        "inverter_id",
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
        "quality_code_inv",
        "communication_status_inv",
    ]
    return {k: clean_value(row.get(k)) for k in wanted if k in source.columns}
def get_baseline_context(timestamp, inverter_id=None):
    """
    Retrieve healthy reference values for conditions similar to the
    selected anomaly.

    This is a comparison reference only. It does not establish cause.
    """
    target = pd.to_datetime(timestamp)

    source = df.copy()

    if inverter_id is not None and "inverter_id" in source.columns:
        source = source[
            source["inverter_id"].astype(str) == str(inverter_id)
        ]

    if source.empty:
        return {"error": "No matching inverter data found."}

    idx = (source["timestamp"] - target).abs().idxmin()
    row = source.loc[idx]

    conditions = {
        "inverter_id": clean_value(row.get("inverter_id")),
        "inverter_status": clean_value(row.get("inverter_status")),
        "is_daylight": clean_value(row.get("is_daylight")),
        "hour": clean_value(row.get("hour")),
        "month": clean_value(row.get("month")),
    }

    baseline = baseline_df.copy()

    # Match the most specific available healthy condition first.
    match_cols = [
        "inverter_id",
        "inverter_status",
        "is_daylight",
        "hour",
        "month",
    ]

    match_cols = [
        c for c in match_cols
        if c in baseline.columns and conditions.get(c) is not None
    ]

    matched = baseline.copy()

    for col in match_cols:
        matched = matched[
            matched[col].astype(str) == str(conditions[col])
        ]

    if matched.empty:
        return {
            "error": "No healthy baseline group is available for the selected operating conditions.",
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

        values = matched[[median_col, q10_col, q90_col]].dropna(
            how="all"
        )

        if values.empty:
            continue

        result["reference"][metric] = {
            "median": clean_value(values[median_col].iloc[0]),
            "typical_low_q10": clean_value(values[q10_col].iloc[0]),
            "typical_high_q90": clean_value(values[q90_col].iloc[0]),
        }

    return result




def generate_ai_explanation(selected_anomaly):
    """
    Generate one explanation for the currently selected anomaly.

    Python collects the evidence first. Groq is used only for the final
    natural-language explanation, so the LLM does not control the tool-call
    loop and cannot get stuck repeatedly requesting tools.
    """
    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set GROQ_API_KEY in the environment "
            "or Streamlit secrets."
        )

    # ------------------------------------------------------------
    # 0. RESOLVE THE SCALAR TIMESTAMP / INVERTER ID UP FRONT
    # ------------------------------------------------------------
    # NOTE: this used to be computed *after* the evidence-gathering block
    # below, which meant every get_* call was accidentally passed the
    # whole `selected_anomaly` Series as `timestamp` instead of a scalar.
    # pd.to_datetime() on a full row tries to convert every value in it
    # (including the numpy.bool_ anomaly_flag), which is exactly the
    # "<class 'numpy.bool'> is not convertible to datetime" error. Moving
    # this up and passing the scalars explicitly fixes it.
    timestamp = clean_value(selected_anomaly.get("timestamp"))
    inverter_id = (
        clean_value(selected_anomaly.get("inverter_id"))
        if "inverter_id" in selected_anomaly.index
        else None
    )

    # ------------------------------------------------------------
    # 1. COLLECT ALL EVIDENCE IN PYTHON
    # ------------------------------------------------------------
    evidence = {}

    # Use the existing evidence/tool functions already defined in the app.
    # These functions are deterministic and do not call the LLM.
    try:
        evidence = get_anomaly_details(timestamp, inverter_id)
    except Exception as e:
        evidence["anomaly_details_error"] = str(e)

    try:
        trend = get_pre_anomaly_trend(timestamp, inverter_id)
        evidence["pre_anomaly_trend"] = trend
    except Exception as e:
        evidence["pre_anomaly_trend_error"] = str(e)

    try:
        contributions = get_feature_contributions(timestamp, inverter_id)
        evidence["feature_contributions"] = contributions
    except Exception as e:
        evidence["feature_contributions_error"] = str(e)

    try:
        operating_context = get_operating_context(timestamp, inverter_id)
        evidence["operating_context"] = operating_context
    except Exception as e:
        evidence["operating_context_error"] = str(e)

    try:
        baseline = get_baseline_context(timestamp, inverter_id)
        evidence["healthy_baseline"] = baseline
    except Exception as e:
        evidence["healthy_baseline_error"] = str(e)

    # ------------------------------------------------------------
    # 2. ASK GROQ ONLY FOR THE FINAL EXPLANATION
    # ------------------------------------------------------------
    # GPT-OSS is a reasoning model. Keep the instruction and evidence in
    # one user message, disable returned reasoning, and give the model enough
    # completion budget for BOTH reasoning and the final paragraph.
    prompt = """
You are the explanation assistant inside an inverter anomaly detection
dashboard.

Generate ONE concise factual paragraph from the supplied evidence.

Rules:
- Use only the supplied evidence. Never invent measurements, causes, events,
  or trends.
- Feature contributions mean parameters that contributed to the unusual
  pattern; they are not automatically causes or root causes.
- Do not automatically call an anomaly a fault.
- Consider operating condition, daylight, power level, temperature,
  communication condition, baseline comparison, and pre-anomaly trend when
  available.
- If the evidence does not establish a cause, say so.
- Do not mention autoencoder, reconstruction error, threshold, anomaly-score
  ratio, probability, confidence, or other internal ML details.
- Do not use headings, bullets, labels, Markdown, or separate sections.
- Return only ONE paragraph of about 4-6 sentences and preferably under
  110 words.
- Naturally cover what happened, when it happened, why it was flagged based
  on the evidence, and what should be checked next.

Selected anomaly:
""" + json.dumps(
        {
            "timestamp": timestamp,
            "inverter_id": inverter_id,
        },
        default=str,
        ensure_ascii=False,
    ) + """

Evidence:
""" + json.dumps(
        evidence,
        default=str,
        ensure_ascii=False,
    )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            # GPT-OSS uses reasoning tokens internally. A small max_tokens
            # budget can be consumed by reasoning before any final text is
            # produced, which is why the old code sometimes got content="".
            reasoning_effort="low",
            include_reasoning=False,
            temperature=0.2,
            max_completion_tokens=768,
        )
    except Exception as e:
        raise RuntimeError(f"Groq request failed: {e}") from e

    explanation = getattr(response.choices[0].message, "content", None)

    if explanation is None:
        raise RuntimeError("Groq returned no text content.")

    explanation = str(explanation).strip()

    if not explanation:
        raise RuntimeError("Groq returned an empty explanation.")

    return explanation, evidence

def render_ai_explanation(explanation):
    """Render the LLM response as one clean paragraph."""
    if not explanation:
        return

    text = str(explanation).strip()

    # Remove accidental Markdown markers if the model returns them.
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"^\s*(?:What happened|When|Why it was flagged|What to check)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()

    safe = html.escape(text)
    st.markdown(
        f'<div class="ai-card"><div class="ai-body">{safe}</div></div>',
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


anomaly_df = (
    filtered_df[filtered_df["anomaly_flag"]].copy() if total_observations else filtered_df.copy()
)


# ============================================================
# DASHBOARD CONTENT -- single scrollable page
# ============================================================

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

 # --------------------------------------------------------
    # AI EXPLANATION
    # --------------------------------------------------------
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

    ts_key = clean_value(selected_anomaly.get("timestamp"))
    inv_key = (
        clean_value(selected_anomaly.get("inverter_id"))
        if "inverter_id" in selected_anomaly.index
        else None
    )
    anomaly_key = f"{ts_key}|{inv_key}"
    cached = st.session_state["ai_explanations"].get(anomaly_key)

    if regenerate or cached is None:

        with st.spinner("Generating explanation from the anomaly evidence..."):

            try:
                ai_explanation, ai_evidence = generate_ai_explanation(
                    selected_anomaly
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
        st.info("Select an anomaly to generate an AI explanation.")

    elif cached.get("error"):

        st.error(
            f"AI explanation failed: {cached['error']}"
        )

    elif cached.get("explanation"):

        render_ai_explanation(
            cached["explanation"]
        )

    else:

        st.warning(
            "Groq did not return an explanation for this anomaly."
        )

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

if len(anomaly_df) > 0:
    display_columns = [
        "timestamp",
        "anomaly_score_ratio",
        "inverter_temperature_c",
        "dc_power_kw",
    ]
    display_columns = [c for c in display_columns if c in anomaly_df.columns]
    friendly_names = {
        "timestamp": "Time",
        "anomaly_score_ratio": "Severity",
        "inverter_temperature_c": "Inverter Temp (°C)",
        "dc_power_kw": "DC Power (kW)",
    }
    table = anomaly_df[display_columns].sort_values("timestamp").rename(columns=friendly_names)
    st.dataframe(table, width="stretch", hide_index=True)
else:
    if total_observations > 0:
        st.success(
            "✅ No anomalies in this period — the inverter behaved normally "
            f"across all {total_observations:,} readings."
        )
    else:
        st.info("No readings in this date range. Try a different range above.")

# ------------------------------------------------------------
# INVESTIGATE AN ANOMALY
# ------------------------------------------------------------
st.markdown("## Investigate an Anomaly")
st.caption("Select one detected anomaly to inspect its preceding trend, contributing parameters, and AI explanation.")

if len(anomaly_df) > 0:
    anomaly_df = anomaly_df.sort_values("timestamp").reset_index(drop=True)

    selected_index = st.selectbox(
        "Select an anomaly",
        range(len(anomaly_df)),
        format_func=lambda x: fmt_time(anomaly_df.loc[x, "timestamp"]),
    )
    selected_anomaly = anomaly_df.loc[selected_index]

    inv_cols = st.columns(3)
    with inv_cols[0]:
        st.metric("Time", fmt_time(selected_anomaly.get("timestamp")))
    with inv_cols[1]:
        st.metric("Severity", fmt_num(selected_anomaly.get("anomaly_score_ratio"), 2, "×"))
    with inv_cols[2]:
        st.metric(
            "Inverter Temperature",
            fmt_num(selected_anomaly.get("inverter_temperature_c"), 1, " °C"),
        )

    detail_col1, detail_col2 = st.columns(2)

    with detail_col1:
        st.markdown("### What Happened Before")
        st.caption("Power and temperature in the 24 hours leading up to the anomaly.")

        end_time = pd.to_datetime(selected_anomaly["timestamp"])
        trend_window, window_start = get_trend_window(trend_df, end_time, hours_back=24)

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
            fig_trend.add_vline(
                x=end_time, line_dash="dash", line_width=2, line_color="#B10318"
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
            contrib_values = selected_anomaly[contribution_cols].dropna()
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

   
