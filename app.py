"""
Inverter Anomaly Detection -- Streamlit Dashboard
====================================================
Production frontend only. All ML/training happens in inverter_anomaly.ipynb.

Reads:
    dashboard_data.parquet  -> evaluation-period observations + model output
    trend_data.parquet      -> full historical time series (pre-anomaly context)

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


AI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_anomaly_details",
            "description": "Retrieve the selected anomaly observation and model outputs.",
            "parameters": {
                "type": "object",
                "properties": {"timestamp": {"type": "string"}, "inverter_id": {"type": "string"}},
                "required": ["timestamp"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_pre_anomaly_trend",
            "description": "Retrieve descriptive statistics for the 24 hours before the selected anomaly.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "string"},
                    "inverter_id": {"type": "string"},
                    "hours": {"type": "number"},
                },
                "required": ["timestamp"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_feature_contributions",
            "description": "Retrieve feature contributions to reconstruction error for the selected anomaly.",
            "parameters": {
                "type": "object",
                "properties": {"timestamp": {"type": "string"}, "inverter_id": {"type": "string"}},
                "required": ["timestamp"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_operating_context",
            "description": "Retrieve daylight, status, power, environmental, and communication context at the selected anomaly.",
            "parameters": {
                "type": "object",
                "properties": {"timestamp": {"type": "string"}, "inverter_id": {"type": "string"}},
                "required": ["timestamp"],
                "additionalProperties": False,
            },
        },
    },
]


def execute_ai_tool(name, args):
    if name == "get_anomaly_details":
        return get_anomaly_details(**args)
    if name == "get_pre_anomaly_trend":
        return get_pre_anomaly_trend(**args)
    if name == "get_feature_contributions":
        return get_feature_contributions(**args)
    if name == "get_operating_context":
        return get_operating_context(**args)
    return {"error": f"Unknown tool: {name}"}


def generate_ai_explanation(selected_anomaly):
    client = get_groq_client()
    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Add [groq] api_key to Streamlit Cloud Secrets."
        )
    timestamp = clean_value(selected_anomaly.get("timestamp"))
    inverter_id = (
        clean_value(selected_anomaly.get("inverter_id"))
        if "inverter_id" in selected_anomaly.index
        else None
    )
    evidence = {}
    system_prompt = """You are an assistant that explains inverter anomalies to a normal \
dashboard user (not a technical or ML user), using only evidence returned by tools.

TOOL USE
- Before writing your final answer, call the available tools -- get_anomaly_details, \
get_pre_anomaly_trend, get_feature_contributions, get_operating_context -- for the given \
timestamp and inverter_id, to gather the anomaly's operating status, daylight/time-of-day \
context, power and current values, inverter and ambient temperature, efficiency, power \
factor, frequency, communication/quality info, feature contributions, anomaly reason, and \
the 24-hour pre-anomaly trend. Do not rely only on the timestamp/inverter_id given to you --
use the tools to gather this evidence yourself.
- Never invent, estimate, or assume a value that was not returned by a tool. If a tool \
returns an error or is missing data, work only with what is available.
- If the remaining evidence is not enough to explain why the anomaly happened, say exactly: \
"The available data is not sufficient to determine the exact reason." Do not guess.

CAUSALITY RULES (critical)
- A feature's contribution means it was one of the readings that stood out as unusual in \
the anomaly evidence. It does NOT mean that feature caused the anomaly, and it does NOT mean \
a component failed. Never state or imply a root cause (e.g. never say "high temperature \
caused the anomaly" or "the cooling system failed").
- Instead, describe a contributing parameter as something that stood out, and offer a \
practical check rather than a diagnosis, e.g. "Temperature was one of the parameters that \
contributed strongly to the unusual pattern" and "Cooling performance should be checked if \
this pattern persists."
- Keep observed fact, possible explanation, and recommended check clearly separate -- do not \
blur them into a single causal claim.

NORMAL-OPERATING-CONDITIONS RULE
- Before calling anything unusual, consider is_daylight, hour, inverter_status, power level, \
ambient temperature, and the normal pre-anomaly trend from the tools.
- Do not call a change abnormal just because a value is higher than at an earlier time. For \
example, a temperature rise around midday alongside a normal rise in ambient temperature and \
power output is not automatically unusual -- say so plainly rather than flagging it as \
abnormal when it looks consistent with normal operation.

LANGUAGE RULES
- Use simple, plain, professional language.
- Never mention: autoencoder, reconstruction error, EVT, POT, anomaly_score_ratio, threshold, \
feature contribution percentage, model score, confidence score, probability, or any other ML \
model internals.
- Never say things like "the model predicts with X% confidence", "the probability of failure \
is...", "the root cause is...", "the ML model determined that...", or "the autoencoder \
detected...".
- Prefer phrasing like: "The inverter showed unusual behavior...", "The main parameter \
contributing to the unusual pattern was...", "Before the anomaly, AC power decreased \
while...", "This should be checked...", "The available data does not confirm the exact \
cause."

OUTPUT FORMAT
Return ONE single paragraph only.

The paragraph must naturally include:
- what happened
- when it happened
- why it was flagged
- what should be checked

Do NOT use headings.
Do NOT use bullet points.
Do NOT use numbered lists.
Do NOT use Markdown or bold markers.
Do NOT use labels such as "What happened:", "When:", "Why it was flagged:", or "What to check:".

Write 4-6 concise sentences in plain, professional language. Keep the entire response under about 110 words. Do not add any other sections or ML terminology.
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "instruction": (
                        "Explain this anomaly for a dashboard user. Call the tools to "
                        "gather the evidence you need -- do not rely on this message alone."
                    ),
                    "selected_timestamp": timestamp,
                    "inverter_id": inverter_id,
                },
                default=str,
            ),
        },
    ]

    for _ in range(6):
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=AI_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content, evidence

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except Exception:
                args = {}
            result = execute_ai_tool(tc.function.name, args)
            evidence[tc.function.name] = result
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str),
                }
            )

    raise RuntimeError(
        "AI investigation reached the maximum tool-call steps without producing an explanation."
    )


def build_llm_evidence(anomaly_row, trend_window, contribution_cols):
    """
    Package only the available, factual evidence for a selected anomaly into
    a JSON-serializable structure suitable for passing to an LLM later.

    Evidence is descriptive only. No causal claims, fault labels, or
    probability language are added here -- that judgment is left entirely
    to whatever consumes this evidence, and even then should be bounded by
    the same rule: describe what the data shows, state what it does not
    establish.
    """

    def clean(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        if isinstance(v, (pd.Timestamp, datetime)):
            return str(v)
        if hasattr(v, "item"):  # numpy scalar
            return v.item()
        return v

    top_contributions = {}
    for col in contribution_cols:
        feature_name = col.replace("_contribution_pct", "")
        top_contributions[feature_name] = clean(anomaly_row.get(col))

    evidence = {
        "anomaly_observation": {
            "timestamp": clean(anomaly_row.get("timestamp")),
            "reconstruction_error": clean(anomaly_row.get("reconstruction_error")),
            "anomaly_score_ratio": clean(anomaly_row.get("anomaly_score_ratio")),
            "note_on_score": (
                "anomaly_score_ratio is reconstruction_error divided by the "
                "detection threshold. It is a ratio, not a probability or "
                "confidence level."
            ),
            "top_contributing_feature": clean(anomaly_row.get("top_contributing_feature")),
            "feature_contribution_pct": top_contributions,
            "anomaly_reason": clean(anomaly_row.get("anomaly_reason")),
        },
        "values_during_anomaly": {
            k: clean(anomaly_row.get(k))
            for k in [
                "inverter_temperature_c",
                "ambient_temperature_c",
                "inverter_ambient_temp_delta",
                "temp_delta_zscore",
                "dc_power_kw",
                "ac_power_kw",
                "quality_code_inv",
                "communication_status_inv",
            ]
            if k in anomaly_row.index
        },
        "trend_window_before_anomaly": None,
        "guidance_for_explanation": [
            "Explain what was detected (which observation(s) exceeded the "
            "reconstruction-error threshold).",
            "Explain which feature(s) contributed most to reconstruction "
            "error, described as a contribution -- never as a 'cause' or "
            "'root cause' unless the evidence itself establishes that.",
            "Describe what changed before vs. during the anomaly, using "
            "only the trend statistics provided.",
            "State what the data supports.",
            "State plainly what cannot be concluded from this evidence "
            "alone (e.g. do not assert a temperature rise caused the "
            "anomaly merely because temperature was elevated; do not call "
            "an anomaly a fault automatically).",
        ],
    }

    if trend_window is not None and len(trend_window) > 0:
        numeric_cols = [
            c
            for c in [
                "ac_power_kw",
                "dc_current_a",
                "dc_power_kw",
                "inverter_temperature_c",
                "inverter_ambient_temp_delta",
                "ambient_temperature_c",
            ]
            if c in trend_window.columns
        ]
        stats = {}
        for col in numeric_cols:
            series = pd.to_numeric(trend_window[col], errors="coerce").dropna()
            if len(series) >= 2:
                stats[col] = {
                    "start": clean(series.iloc[0]),
                    "end": clean(series.iloc[-1]),
                    "net_change": clean(series.iloc[-1] - series.iloc[0]),
                    "min": clean(series.min()),
                    "max": clean(series.max()),
                }
        evidence["trend_window_before_anomaly"] = {
            "window_start": clean(trend_window["timestamp"].min()),
            "window_end": clean(trend_window["timestamp"].max()),
            "number_of_observations": int(len(trend_window)),
            "net_changes": stats,
        }

    return evidence



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
max_temperature = (
    filtered_df["inverter_temperature_c"].max()
    if "inverter_temperature_c" in filtered_df.columns and total_observations
    else None
)

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
# TRENDS
# ------------------------------------------------------------
st.markdown("## Trends")
st.caption("Historical power and temperature behavior for the selected period.")

trend_total = len(filtered_trend_df)
trend_col1, trend_col2 = st.columns(2)

with trend_col1:
    st.markdown("### Power Output")
    st.caption("How much power the inverter produced.")
    has_dc = "dc_power_kw" in filtered_trend_df.columns
    has_ac = "ac_power_kw" in filtered_trend_df.columns
    if trend_total > 0 and (has_dc or has_ac):
        fig_power = go.Figure()
        if has_ac:
            fig_power.add_trace(
                go.Scatter(
                    x=filtered_trend_df["timestamp"],
                    y=filtered_trend_df["ac_power_kw"],
                    mode="lines",
                    name="AC Power (kW)",
                    line=dict(color="#4C78A8"),
                )
            )
        if has_dc:
            fig_power.add_trace(
                go.Scatter(
                    x=filtered_trend_df["timestamp"],
                    y=filtered_trend_df["dc_power_kw"],
                    mode="lines",
                    name="DC Power (kW)",
                    line=dict(color="#72B7B2"),
                )
            )
        fig_power.update_layout(
            xaxis_title="Time",
            yaxis_title="Power (kW)",
            hovermode="x unified",
            height=340,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=35, l=55, r=20, b=45),
        )
        st.plotly_chart(fig_power, width="stretch")
    else:
        st.info("Power data not available.")

with trend_col2:
    st.markdown("### Temperature")
    st.caption("Inverter temperature vs. the surrounding air.")
    has_temp = "inverter_temperature_c" in filtered_trend_df.columns
    has_ambient = "ambient_temperature_c" in filtered_trend_df.columns
    if trend_total > 0 and (has_temp or has_ambient):
        fig_temp = go.Figure()
        if has_temp:
            fig_temp.add_trace(
                go.Scatter(
                    x=filtered_trend_df["timestamp"],
                    y=filtered_trend_df["inverter_temperature_c"],
                    mode="lines",
                    name="Inverter Temp (°C)",
                    line=dict(color="#E45756"),
                )
            )
        if has_ambient:
            fig_temp.add_trace(
                go.Scatter(
                    x=filtered_trend_df["timestamp"],
                    y=filtered_trend_df["ambient_temperature_c"],
                    mode="lines",
                    name="Ambient Temp (°C)",
                    line=dict(color="#F58518"),
                )
            )
        fig_temp.update_layout(
            xaxis_title="Time",
            yaxis_title="Temperature (°C)",
            hovermode="x unified",
            height=340,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=35, l=55, r=20, b=45),
        )
        st.plotly_chart(fig_temp, width="stretch")
    else:
        st.info("Temperature data not available.")

# ------------------------------------------------------------
# DETECTED ANOMALIES
# ------------------------------------------------------------
st.markdown("## Detected Anomalies")
st.caption("Observations flagged by the anomaly detector in the selected period.")

if len(anomaly_df) > 0:
    display_columns = [
        "timestamp",
        "anomaly_score_ratio",
        "anomaly_reason",
        "inverter_temperature_c",
        "ac_power_kw",
    ]
    display_columns = [c for c in display_columns if c in anomaly_df.columns]
    friendly_names = {
        "timestamp": "Time",
        "anomaly_score_ratio": "Severity",
        "anomaly_reason": "Anomaly Description",
        "inverter_temperature_c": "Inverter Temp (°C)",
        "ac_power_kw": "AC Power (kW)",
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
            if "ac_power_kw" in trend_window.columns:
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_window["timestamp"],
                        y=trend_window["ac_power_kw"],
                        mode="lines",
                        name="AC Power (kW)",
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
                ai_explanation, ai_evidence = generate_ai_explanation(selected_anomaly)
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

    if cached.get("error"):
        st.error(f"AI explanation failed: {cached['error']}")
    elif cached.get("explanation"):
        render_ai_explanation(cached["explanation"])

else:
    st.info(
        "Nothing to investigate here — there are no anomalies in the selected "
        "date range. Widen the date range or clear **Show anomalies only** "
        "to bring up more data."
    )
