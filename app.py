"""
Inverter Anomaly Detection -- Streamlit Dashboard
====================================================
Production frontend only. All ML/training happens in inverter_anomaly.ipynb.

Reads:
    dashboard_data.parquet  -> evaluation-period observations + model output

Does NOT retrain or re-run the notebook. Does NOT treat anomaly_score_ratio
as a probability. Feature contributions are reported as "contributed most
to reconstruction error," never as a proven cause.

NOTE ON "anomaly_type": the underlying data/model does not export an event
classification (CURTAILMENT / GRID_OUTAGE_TRIP / etc). classify_anomaly_type()
below derives a best-effort label from inverter_status + power behavior +
the top contributing feature. Treat it as a heuristic display label, not a
verified event type -- adjust the rules to your own domain logic as needed.
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

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    html { font-size: 16px; }

    :root {
        --primary: #0F3554;
        --primary-light: #164A73;
        --warn: #DC6803;
        --warn-bg: #FEF3E8;
        --danger: #E4463F;
        --danger-bg: #FDECEC;
        --success: #1F9D55;
        --success-bg: #EAF7EF;
        --border: #E4E9F0;
        --muted: #64748B;
        --surface: #FFFFFF;
        --bg: #F5F7FA;
    }

    .stApp { background-color: var(--bg); }

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

    [data-testid="stVerticalBlock"] { gap: 0.35rem; }
    div[data-testid="stElementContainer"] { margin-bottom: 0 !important; }

    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        min-height: 2rem !important;
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
    }
    [data-testid="stWidgetLabel"] p { margin-bottom: 0.1rem !important; }

    /* ---------- KPI cards (icon + label + value) ---------- */
    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        min-height: 88px;
    }
    .kpi-icon {
        width: 38px; height: 38px;
        border-radius: 9px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
        flex-shrink: 0;
    }
    .kpi-icon.blue   { background: #E8F0F9; }
    .kpi-icon.warn   { background: var(--warn-bg); }
    .kpi-icon.danger { background: var(--danger-bg); }
    .kpi-icon.success{ background: var(--success-bg); }
    .kpi-label {
        font-size: 0.8rem;
        color: var(--muted);
        font-weight: 500;
        margin-bottom: 0.1rem;
    }
    .kpi-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.78rem;
        color: var(--muted);
        margin-top: 0.1rem;
    }

    /* ---------- Status badges ---------- */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
    }
    .badge-running { background: var(--success-bg); color: var(--success); }
    .badge-derated { background: var(--warn-bg); color: var(--warn); }
    .badge-fault   { background: var(--danger-bg); color: var(--danger); }
    .badge-default { background: #EEF2F7; color: var(--muted); }

    /* ---------- Headings ---------- */
    h2, [data-testid="stMarkdownContainer"] h2 {
        color: #0F172A;
        font-size: 1.4rem !important;
        font-weight: 700;
        margin-top: 0.9rem !important;
        margin-bottom: 0.35rem !important;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--border);
    }
    h3, [data-testid="stMarkdownContainer"] h3 {
        color: #0F172A;
        font-size: 1.12rem !important;
        font-weight: 700;
        margin-top: 0.5rem !important;
        margin-bottom: 0.25rem !important;
    }

    hr { border-color: var(--border) !important; margin: 0.5rem 0 !important; }
    [data-testid="stCaptionContainer"], .stCaption { margin-bottom: 0.2rem !important; }

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

    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--surface);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px !important;
        padding: 0.5rem 0.9rem !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
        gap: 0.3rem;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    label, .stSelectbox label, .stDateInput label {
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        color: #334155 !important;
    }

    /* ---------- AI Explanation panel ---------- */
    .ai-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 2px 7px rgba(15, 23, 42, 0.05);
    }
    .ai-panel-title {
        color: #0F172A;
        font-size: 1.25rem;
        font-weight: 700;
    }
    .ai-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: #E8F0F9;
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 700;
        padding: 0.2rem 0.6rem;
        border-radius: 999px;
    }
    .ai-headline-box {
        background: #F3F7FB;
        border: 1px solid #DCE6F0;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin: 0.7rem 0 0.9rem 0;
        display: flex;
        gap: 0.6rem;
        align-items: flex-start;
    }
    .ai-headline-box .icon { font-size: 1.1rem; flex-shrink: 0; }
    .ai-headline-title { font-weight: 700; color: #0F172A; margin-bottom: 0.2rem; }
    .ai-headline-body { color: #1E293B; font-size: 0.95rem; line-height: 1.55; }

    .ai-section { margin-bottom: 0.9rem; }
    .ai-section:last-of-type { margin-bottom: 0; }
    .ai-section-title {
        display: flex; align-items: center; gap: 0.4rem;
        font-weight: 700; color: #0F172A; font-size: 1rem;
        margin-bottom: 0.35rem;
    }
    .ai-section ul { margin: 0 0 0 1.25rem; padding: 0; }
    .ai-section li {
        color: #1E293B; font-size: 0.92rem; line-height: 1.55;
        margin-bottom: 0.2rem;
    }
    .ai-section .plain { color: #1E293B; font-size: 0.92rem; line-height: 1.55; }
    .ai-when-time { font-weight: 600; color: #0F172A; }
    .ai-when-sub { color: var(--muted); font-size: 0.85rem; margin-top: 0.1rem; }

    .ai-footnote {
        display: flex; gap: 0.5rem; align-items: flex-start;
        background: #EFF6FF; border: 1px solid #DCEAFB;
        border-radius: 10px; padding: 0.7rem 0.9rem;
        margin-top: 0.9rem;
        color: #334155; font-size: 0.82rem; line-height: 1.5;
    }

    .ai-generating {
        background: #F7F9FC; border: 1px solid #D9E1EA; border-radius: 10px;
        padding: 0.85rem 1rem; color: #475569; font-size: 0.9rem;
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
    load_dashboard_baseline, "dashboard_baseline.parquet", "Dashboard healthy baseline"
)
if df.empty:
    st.error("`dashboard_data.parquet` loaded but contains no rows.")
    st.stop()

try:
    trend_df = load_trend_data("trend_data.parquet")
    has_trend_data = True
except Exception:
    trend_df = df
    has_trend_data = False


# ============================================================
# ANOMALY TYPE (heuristic display label -- see module docstring)
# ============================================================


def classify_anomaly_type(row):
    status = str(row.get("inverter_status", "") or "").upper()

    if "OUTAGE" in status or "TRIP" in status:
        return "GRID_OUTAGE_TRIP"
    if "FAULT" in status:
        return "FAULT"
    if "DERATE" in status or "CURTAIL" in status:
        return "CURTAILMENT"

    dc = row.get("dc_power_kw")
    poa = row.get("poa_w_m2")
    if pd.notna(dc) and pd.notna(poa) and poa and poa > 50:
        if dc is not None and dc < 0.05 * poa:
            return "GRID_OUTAGE_TRIP"

    top_feat = row.get("top_contributing_feature")
    if isinstance(top_feat, str) and top_feat.strip():
        return f"{top_feat.strip().upper()}_DEVIATION"

    return "UNCLASSIFIED"


if "anomaly_flag" in df.columns:
    df["anomaly_type"] = df.apply(
        lambda r: classify_anomaly_type(r) if r.get("anomaly_flag") else None, axis=1
    )
else:
    df["anomaly_type"] = None


def status_badge_class(status):
    s = str(status or "").upper()
    if "FAULT" in s or "TRIP" in s or "OUTAGE" in s:
        return "badge-fault"
    if "DERATE" in s or "CURTAIL" in s:
        return "badge-derated"
    if "RUN" in s or "NORMAL" in s or "OK" in s:
        return "badge-running"
    return "badge-default"


# ============================================================
# SMALL HELPERS
# ============================================================


def fmt_num(value, decimals=2, suffix="", dash="—"):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return dash
    return f"{value:,.{decimals}f}{suffix}"


def fmt_time(value, dash="—"):
    if value is None or pd.isna(value):
        return dash
    return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")


def fmt_hm(value, dash="—"):
    if value is None or pd.isna(value):
        return dash
    return pd.to_datetime(value).strftime("%H:%M")


def get_trend_window(source, end_time, hours_back=24):
    end_time = pd.to_datetime(end_time)
    start_time = end_time - timedelta(hours=hours_back)
    window = source[(source["timestamp"] >= start_time) & (source["timestamp"] <= end_time)].copy()
    return window, start_time


def get_contribution_cols(frame):
    return [c for c in frame.columns if c.endswith("_contribution_pct")]


def kpi_card(icon, icon_class, label, value, sub=None):
    sub_html = f'<div class="kpi-sub">{html.escape(sub)}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-icon {icon_class}">{icon}</div>
            <div>
                <div class="kpi-label">{html.escape(label)}</div>
                <div class="kpi-value">{html.escape(str(value))}</div>
                {sub_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["groq"]["api_key"]
        except Exception:
            api_key = None
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ============================================================
# EVIDENCE TOOLS (deterministic, no LLM calls)
# ============================================================


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
        c for c in [
            "ac_power_kw", "dc_power_kw", "dc_current_a", "ac_current_a",
            "inverter_temperature_c", "ambient_temperature_c",
            "inverter_ambient_temp_delta", "efficiency_pct", "power_factor",
        ] if c in source.columns
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
        for c in cols if pd.notna(row.get(c))
    ]
    values.sort(key=lambda x: x["contribution_pct"] if x["contribution_pct"] is not None else -1, reverse=True)
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
        "timestamp", "inverter_id", "hour", "minute", "month", "is_daylight",
        "inverter_status", "dc_power_kw", "ac_power_kw", "dc_current_a",
        "ac_current_a", "power_factor", "frequency_hz", "efficiency_pct",
        "inverter_temperature_c", "ambient_temperature_c", "poa_w_m2",
        "ghi_w_m2", "quality_code_inv", "communication_status_inv",
    ]
    return {k: clean_value(row.get(k)) for k in wanted if k in source.columns}


def get_baseline_context(timestamp, inverter_id=None):
    target = pd.to_datetime(timestamp)
    source = df.copy()
    if inverter_id is not None and "inverter_id" in source.columns:
        source = source[source["inverter_id"].astype(str) == str(inverter_id)]
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
    match_cols = ["inverter_id", "inverter_status", "is_daylight", "hour", "month"]
    match_cols = [c for c in match_cols if c in baseline.columns and conditions.get(c) is not None]

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
        "healthy_sample_count": int(matched["healthy_sample_count"].sum()),
        "reference": {},
    }

    metric_names = sorted({col[:-7] for col in matched.columns if col.endswith("_median")})
    for metric in metric_names:
        median_col, q10_col, q90_col = f"{metric}_median", f"{metric}_q10", f"{metric}_q90"
        values = matched[[median_col, q10_col, q90_col]].dropna(how="all")
        if values.empty:
            continue
        result["reference"][metric] = {
            "median": clean_value(values[median_col].iloc[0]),
            "typical_low_q10": clean_value(values[q10_col].iloc[0]),
            "typical_high_q90": clean_value(values[q90_col].iloc[0]),
        }
    return result


# ============================================================
# AI EXPLANATION (structured JSON -> sectioned card)
# ============================================================


def generate_ai_explanation(selected_anomaly):
    """
    Generate one structured explanation for the currently selected anomaly.
    Python collects the evidence first; Groq only writes the final text, in
    a fixed JSON shape so the UI can render distinct sections.
    """
    client = get_groq_client()
    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. Set GROQ_API_KEY in the environment "
            "or Streamlit secrets."
        )

    timestamp = clean_value(selected_anomaly.get("timestamp"))
    inverter_id = (
        clean_value(selected_anomaly.get("inverter_id"))
        if "inverter_id" in selected_anomaly.index else None
    )
    anomaly_type = clean_value(selected_anomaly.get("anomaly_type"))

    evidence = {}
    try:
        evidence = get_anomaly_details(timestamp, inverter_id)
    except Exception as e:
        evidence["anomaly_details_error"] = str(e)
    try:
        evidence["pre_anomaly_trend"] = get_pre_anomaly_trend(timestamp, inverter_id)
    except Exception as e:
        evidence["pre_anomaly_trend_error"] = str(e)
    try:
        evidence["feature_contributions"] = get_feature_contributions(timestamp, inverter_id)
    except Exception as e:
        evidence["feature_contributions_error"] = str(e)
    try:
        evidence["operating_context"] = get_operating_context(timestamp, inverter_id)
    except Exception as e:
        evidence["operating_context_error"] = str(e)
    try:
        evidence["healthy_baseline"] = get_baseline_context(timestamp, inverter_id)
    except Exception as e:
        evidence["healthy_baseline_error"] = str(e)

    prompt = """
You are the explanation assistant inside an inverter anomaly detection
dashboard.

Return ONLY valid JSON (no Markdown fences, no commentary) matching exactly
this schema:

{
  "description": "1-2 sentence factual description of what happened",
  "why_it_happened": ["short factual bullet", "short factual bullet"],
  "duration_note": "short note on how long it appears to have lasted, or null if evidence doesn't establish this",
  "recommended_actions": ["short actionable bullet", "short actionable bullet"]
}

Rules:
- Use only the supplied evidence. Never invent measurements, causes, events, or trends.
- "why_it_happened" should give 2-4 short bullets grounded in the evidence
  (e.g. which readings moved, by how much, versus the healthy baseline).
- Feature contributions mean parameters that contributed to the unusual
  pattern; they are not automatically causes or root causes.
- Do not automatically call an anomaly a fault unless the evidence (status
  field) says so.
- "recommended_actions" should be 2-4 concrete, practical checks a field
  technician or plant operator could do next.
- If the evidence does not establish a cause, say so plainly in "description".
- Do not mention autoencoder, reconstruction error, threshold, anomaly-score
  ratio, probability, confidence, or other internal ML details.
- Keep "description" under 60 words. Keep each bullet under 20 words.

Selected anomaly:
""" + json.dumps(
        {"timestamp": timestamp, "inverter_id": inverter_id, "anomaly_type": anomaly_type},
        default=str, ensure_ascii=False,
    ) + """

Evidence:
""" + json.dumps(evidence, default=str, ensure_ascii=False)

    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[{"role": "user", "content": prompt}],
            reasoning_effort="low",
            include_reasoning=False,
            temperature=0.2,
            max_completion_tokens=768,
        )
    except Exception as e:
        raise RuntimeError(f"Groq request failed: {e}") from e

    raw = getattr(response.choices[0].message, "content", None)
    if raw is None:
        raise RuntimeError("Groq returned no text content.")
    raw = str(raw).strip()
    if not raw:
        raise RuntimeError("Groq returned an empty explanation.")

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.I)
    try:
        parsed = json.loads(cleaned)
    except Exception:
        # Fall back to treating the whole response as a plain description.
        parsed = {
            "description": re.sub(r"\s+", " ", cleaned).strip(),
            "why_it_happened": [],
            "duration_note": None,
            "recommended_actions": [],
        }

    parsed.setdefault("description", "")
    parsed.setdefault("why_it_happened", [])
    parsed.setdefault("duration_note", None)
    parsed.setdefault("recommended_actions", [])
    return parsed, evidence


def render_ai_explanation(parsed, timestamp, anomaly_type):
    description = html.escape(str(parsed.get("description") or "").strip())
    why_bullets = [html.escape(str(b)) for b in (parsed.get("why_it_happened") or []) if str(b).strip()]
    actions = [html.escape(str(b)) for b in (parsed.get("recommended_actions") or []) if str(b).strip()]
    duration_note = parsed.get("duration_note")
    duration_html = f'<div class="ai-when-sub">{html.escape(str(duration_note))}</div>' if duration_note else ""

    type_label = html.escape(str(anomaly_type).replace("_", " ")) if anomaly_type else "anomaly"

    headline = f"""
    <div class="ai-headline-box">
        <div class="icon">⚠️</div>
        <div>
            <div class="ai-headline-title">An anomaly was detected at {html.escape(fmt_time(timestamp))}.</div>
            <div class="ai-headline-body">{description}</div>
        </div>
    </div>
    """

    why_html = ""
    if why_bullets:
        items = "".join(f"<li>{b}</li>" for b in why_bullets)
        why_html = f"""
        <div class="ai-section">
            <div class="ai-section-title">💡 Why it happened?</div>
            <ul>{items}</ul>
        </div>
        """

    when_html = f"""
    <div class="ai-section">
        <div class="ai-section-title">🕒 When it occurred?</div>
        <div class="ai-when-time">{html.escape(fmt_time(timestamp))} &middot; {type_label}</div>
        {duration_html}
    </div>
    """

    actions_html = ""
    if actions:
        items = "".join(f"<li>{a}</li>" for a in actions)
        actions_html = f"""
        <div class="ai-section">
            <div class="ai-section-title">🔧 Recommended action</div>
            <ul>{items}</ul>
        </div>
        """

    footnote = """
    <div class="ai-footnote">
        <div>ℹ️</div>
        <div>This explanation is based on the observed patterns in your data and the
        trained anomaly detection model. It does not confirm a fault but helps you
        understand the possible cause and impact.</div>
    </div>
    """

    st.markdown(headline + why_html + when_html + actions_html + footnote, unsafe_allow_html=True)


# ============================================================
# HEADER
# ============================================================

available_inverters = (
    sorted(df["inverter_id"].dropna().unique().tolist()) if "inverter_id" in df.columns else []
)

header_left, header_right = st.columns([3, 1.2])
with header_left:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:0.7rem;">
            <div style="
                background: linear-gradient(135deg, #0F3554 0%, #164A73 100%);
                width: 50px; height: 50px; border-radius: 12px;
                display:flex; align-items:center; justify-content:center;
                font-size: 1.5rem; flex-shrink:0;
                box-shadow: 0 4px 14px rgba(15, 53, 84, 0.16);
            ">⚡</div>
            <div>
                <div style="color:#0F172A; font-size:1.65rem; font-weight:700; line-height:1.2;">
                    Inverter Anomaly Detection
                </div>
                <div style="color:#64748B; font-size:0.92rem; margin-top:0.1rem;">
                    Monitor &middot; Detect &middot; Understand
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown(
        f"""
        <div style="text-align:right; padding-top:0.3rem;">
            <div style="color:#0F172A; font-size:0.95rem; font-weight:600;">☀️ Solar Plant Dashboard</div>
            <div style="color:#64748B; font-size:0.85rem;">Inverter ID: {{inverter_label}}</div>
        </div>
        """.replace("{inverter_label}", "All" if not available_inverters else "—"),
        unsafe_allow_html=True,
    )

# ============================================================
# FILTERS
# ============================================================

eval_min_date = df["timestamp"].min().date()
eval_max_date = df["timestamp"].max().date()
min_date = min(eval_min_date, trend_df["timestamp"].min().date())
max_date = max(eval_max_date, trend_df["timestamp"].max().date())
show_inverter_filter = "inverter_id" in df.columns and df["inverter_id"].nunique() > 1

status_options = ["All"]
if "inverter_status" in df.columns:
    status_options += sorted(df["inverter_status"].dropna().astype(str).unique().tolist())

with st.container(border=True):
    if not has_trend_data:
        st.caption(
            "ℹ️ `trend_data.parquet` not found — showing the evaluation "
            "period only. Run notebook Section 19 to enable full history."
        )

    filter_cols = st.columns([1.6, 1, 1, 1.1] if show_inverter_filter else [1.6, 1, 1.1])

    with filter_cols[0]:
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = min_date, max_date

    if show_inverter_filter:
        with filter_cols[1]:
            inverter_options = ["All"] + available_inverters
            selected_inverter = st.selectbox("Inverter", inverter_options)
        status_col, toggle_col = filter_cols[2], filter_cols[3]
    else:
        selected_inverter = "All"
        status_col, toggle_col = filter_cols[1], filter_cols[2]

    with status_col:
        selected_status = st.selectbox("Status", status_options)

    with toggle_col:
        st.write("")
        show_anomalies_only = st.toggle("Show anomalies only", value=False)

    if start_date > end_date:
        st.warning("Start date is after end date — swap them to see results.")
    elif has_trend_data and (start_date < eval_min_date or end_date > eval_max_date):
        st.caption(
            f"ℹ️ Anomaly detection only covers **{eval_min_date} to {eval_max_date}**. "
            "Dates outside that window show raw power & temperature readings "
            "but no anomaly results."
        )

inverter_label = selected_inverter if selected_inverter != "All" else (
    "All" if not available_inverters else "All"
)

# ============================================================
# FILTER DATA
# ============================================================

if start_date <= end_date:
    filtered_df = df[
        (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    ].copy()
    filtered_trend_df = trend_df[
        (trend_df["timestamp"].dt.date >= start_date) & (trend_df["timestamp"].dt.date <= end_date)
    ].copy()
else:
    filtered_df = df.iloc[0:0].copy()
    filtered_trend_df = trend_df.iloc[0:0].copy()

if selected_inverter != "All" and "inverter_id" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["inverter_id"] == selected_inverter].copy()
if selected_inverter != "All" and "inverter_id" in filtered_trend_df.columns:
    filtered_trend_df = filtered_trend_df[filtered_trend_df["inverter_id"] == selected_inverter].copy()

if selected_status != "All" and "inverter_status" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["inverter_status"].astype(str) == selected_status].copy()

if show_anomalies_only:
    filtered_df = filtered_df[filtered_df["anomaly_flag"]].copy()

if filtered_df.empty and filtered_trend_df.empty:
    st.warning("No observations match the current filters. Adjust the filters above.")

total_observations = len(filtered_df)
total_anomalies = int(filtered_df["anomaly_flag"].sum()) if total_observations else 0
normal_count = total_observations - total_anomalies
anomaly_rate = (total_anomalies / total_observations * 100) if total_observations else None
normal_rate = (normal_count / total_observations * 100) if total_observations else None

anomaly_df = filtered_df[filtered_df["anomaly_flag"]].copy() if total_observations else filtered_df.copy()

most_common_type, most_common_count = "—", 0
if len(anomaly_df) and "anomaly_type" in anomaly_df.columns:
    counts = anomaly_df["anomaly_type"].dropna().value_counts()
    if len(counts):
        most_common_type = str(counts.index[0]).replace("_", " ")
        most_common_count = int(counts.iloc[0])

# ============================================================
# KPI ROW
# ============================================================

kpi_cols = st.columns(4)
with kpi_cols[0]:
    kpi_card("🗄️", "blue", "Total Records", f"{total_observations:,}", "in selected period")
with kpi_cols[1]:
    kpi_card("⚠️", "danger", "Anomalies Detected", f"{total_anomalies:,}", fmt_num(anomaly_rate, 2, "%"))
with kpi_cols[2]:
    kpi_card("✅", "success", "Normal", f"{normal_count:,}", fmt_num(normal_rate, 2, "%"))
with kpi_cols[3]:
    kpi_card(
        "📊", "blue", "Most Common Anomaly Type", most_common_type,
        f"{most_common_count} occurrence{'s' if most_common_count != 1 else ''}" if most_common_count else None,
    )

# ============================================================
# MAIN LAYOUT: power trend (left) + AI explanation (right)
# ============================================================

main_col, ai_col = st.columns([2, 1.15], gap="medium")

with main_col:
    st.markdown("### ⚡ DC & AC Power Trend")

    if total_observations > 0 and ("dc_power_kw" in filtered_df.columns or "ac_power_kw" in filtered_df.columns):
        fig = go.Figure()
        if "dc_power_kw" in filtered_df.columns:
            fig.add_trace(go.Scatter(
                x=filtered_df["timestamp"], y=filtered_df["dc_power_kw"],
                mode="lines", name="DC Power (kW)", line=dict(color="#4C78A8", width=1.6),
            ))
        if "ac_power_kw" in filtered_df.columns:
            fig.add_trace(go.Scatter(
                x=filtered_df["timestamp"], y=filtered_df["ac_power_kw"],
                mode="lines", name="AC Power (kW)", line=dict(color="#F58518", width=1.6),
            ))

        anomaly_points = filtered_df[filtered_df["anomaly_flag"]].copy()
        power_col = "dc_power_kw" if "dc_power_kw" in filtered_df.columns else "ac_power_kw"
        if len(anomaly_points) > 0 and power_col in anomaly_points.columns:
            fig.add_trace(go.Scatter(
                x=anomaly_points["timestamp"], y=anomaly_points[power_col],
                mode="markers", name="Anomaly",
                marker=dict(size=9, color="#E45756", symbol="circle", line=dict(width=1, color="white")),
            ))
            y_max = float(filtered_df[power_col].max()) if pd.notna(filtered_df[power_col].max()) else 0
            for _, pt in anomaly_points.iterrows():
                label = f"{fmt_hm(pt['timestamp'])}<br>{str(pt.get('anomaly_type') or '').replace('_', ' ')}"
                fig.add_vline(x=pt["timestamp"], line_dash="dash", line_width=1, line_color="#E45756", opacity=0.5)
                fig.add_annotation(
                    x=pt["timestamp"], y=y_max, text=label, showarrow=False,
                    yshift=18, font=dict(size=10, color="#B10318"),
                    bgcolor="#FDECEC", bordercolor="#E45756", borderwidth=1, borderpad=3,
                )

        fig.update_layout(
            xaxis_title="Time", yaxis_title="Power (kW)", hovermode="x unified",
            height=380, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.15, xanchor="center", x=0.5),
            margin=dict(t=55, l=55, r=25, b=45),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No power data available for the selected period.")

    temp_col, donut_col = st.columns([1.5, 1])

    with temp_col:
        st.markdown("### 🌡️ Inverter Temperature")
        st.caption("Split by operating status where available.")
        trend_total = len(filtered_trend_df)
        has_temp = "inverter_temperature_c" in filtered_trend_df.columns
        if trend_total > 0 and has_temp:
            fig_temp = go.Figure()
            if "inverter_status" in filtered_trend_df.columns:
                for status_val, color in [("Running", "#4C78A8"), ("Derated", "#F58518")]:
                    mask = filtered_trend_df["inverter_status"].astype(str).str.contains(status_val, case=False, na=False)
                    if mask.any():
                        y = filtered_trend_df["inverter_temperature_c"].where(mask)
                        fig_temp.add_trace(go.Scatter(
                            x=filtered_trend_df["timestamp"], y=y, mode="lines",
                            name=status_val, connectgaps=False, line=dict(color=color),
                        ))
            else:
                fig_temp.add_trace(go.Scatter(
                    x=filtered_trend_df["timestamp"], y=filtered_trend_df["inverter_temperature_c"],
                    mode="lines", name="Inverter Temp (°C)", line=dict(color="#E45756"),
                ))
            fig_temp.update_layout(
                xaxis_title="Time", yaxis_title="Temperature (°C)", hovermode="x unified",
                height=300, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
                margin=dict(t=35, l=55, r=20, b=45),
            )
            st.plotly_chart(fig_temp, width="stretch")
        else:
            st.info("Temperature data not available.")

    with donut_col:
        st.markdown("### 🧭 Inverter Status")
        if total_observations > 0 and "inverter_status" in filtered_df.columns:
            status_counts = filtered_df["inverter_status"].astype(str).value_counts()
            color_map = {"badge-running": "#4C78A8", "badge-derated": "#F58518", "badge-fault": "#E45756", "badge-default": "#94A3B8"}
            colors = [color_map[status_badge_class(s)] for s in status_counts.index]
            top_label = status_counts.index[0]
            top_pct = status_counts.iloc[0] / status_counts.sum() * 100

            fig_donut = go.Figure(data=[go.Pie(
                labels=status_counts.index, values=status_counts.values,
                hole=0.68, marker=dict(colors=colors), textinfo="none",
                hovertemplate="%{label}: %{percent}<extra></extra>",
            )])
            fig_donut.update_layout(
                height=260, showlegend=True,
                legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.0, font=dict(size=11)),
                margin=dict(t=10, l=10, r=10, b=10),
                annotations=[dict(
                    text=f"<b>{top_pct:.2f}%</b><br><span style='font-size:11px'>{top_label}</span>",
                    x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#0F172A"),
                )],
            )
            st.plotly_chart(fig_donut, width="stretch")
        else:
            st.info("Status data not available.")

with ai_col:
    st.markdown(
        '<div class="ai-panel-title">AI Explanation '
        '<span class="ai-pill">✨ Powered by AI</span></div>',
        unsafe_allow_html=True,
    )

    if len(anomaly_df) > 0:
        anomaly_df_sorted = anomaly_df.sort_values("timestamp").reset_index(drop=True)
        default_idx = len(anomaly_df_sorted) - 1  # most recent anomaly by default

        selected_index = st.selectbox(
            "Select anomaly to explain",
            range(len(anomaly_df_sorted)),
            index=default_idx,
            format_func=lambda x: f"{fmt_time(anomaly_df_sorted.loc[x, 'timestamp'])} — "
                                   f"{str(anomaly_df_sorted.loc[x, 'anomaly_type'] or '').replace('_', ' ')}",
        )
        selected_anomaly = anomaly_df_sorted.loc[selected_index]

        regenerate = st.button("🔄 Regenerate", type="primary", use_container_width=True)

        ts_key = clean_value(selected_anomaly.get("timestamp"))
        inv_key = clean_value(selected_anomaly.get("inverter_id")) if "inverter_id" in selected_anomaly.index else None
        anomaly_key = f"{ts_key}|{inv_key}"
        cached = st.session_state["ai_explanations"].get(anomaly_key)

        if regenerate or cached is None:
            with st.spinner("Generating explanation from the anomaly evidence..."):
                try:
                    parsed, ai_evidence = generate_ai_explanation(selected_anomaly)
                    cached = {"parsed": parsed, "evidence": ai_evidence, "error": None}
                except Exception as e:
                    cached = {"parsed": None, "evidence": None, "error": str(e)}
                st.session_state["ai_explanations"][anomaly_key] = cached

        if cached.get("error"):
            st.error(f"AI explanation failed: {cached['error']}")
        elif cached.get("parsed"):
            render_ai_explanation(
                cached["parsed"], selected_anomaly.get("timestamp"), selected_anomaly.get("anomaly_type"),
            )
        else:
            st.warning("The AI service did not return an explanation for this anomaly.")
    else:
        st.info(
            "No anomalies in the selected filters — nothing to explain. "
            "Widen the date range or clear **Show anomalies only**."
        )

# ============================================================
# DETECTED ANOMALIES TABLE
# ============================================================

st.markdown("## Detected Anomalies")

if len(anomaly_df) > 0:
    show_all = st.session_state.get("show_all_anomalies", False)
    header_col, link_col = st.columns([6, 1])
    with link_col:
        if st.button("View All" if not show_all else "Show Top 5", use_container_width=True):
            st.session_state["show_all_anomalies"] = not show_all
            st.rerun()

    contribution_cols = get_contribution_cols(anomaly_df)

    def key_contributors(row):
        vals = row[contribution_cols].dropna() if contribution_cols else pd.Series(dtype=float)
        if vals.empty:
            return "—"
        top = vals.sort_values(ascending=False).head(3)
        return ", ".join(c.replace("_contribution_pct", "").replace("_", " ").title() for c in top.index)

    table_df = anomaly_df.sort_values("timestamp", ascending=False).reset_index(drop=True)
    if not show_all:
        table_df = table_df.head(5)

    rows_html = ""
    for _, r in table_df.iterrows():
        status_val = r.get("inverter_status", "—")
        badge_cls = status_badge_class(status_val)
        rows_html += f"""
        <tr>
            <td>{html.escape(fmt_time(r.get('timestamp')))}</td>
            <td>{html.escape(str(r.get('inverter_id', '—')))}</td>
            <td><b>{html.escape(str(r.get('anomaly_type') or '—').replace('_', ' '))}</b></td>
            <td><span class="badge {badge_cls}">{html.escape(str(status_val or '—'))}</span></td>
            <td>{html.escape(key_contributors(r))}</td>
        </tr>
        """

    st.markdown(
        f"""
        <div style="overflow-x:auto; border:1px solid var(--border); border-radius:10px;">
        <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
            <thead style="background:#F8FAFC; text-align:left; color:#64748B; font-size:0.78rem; text-transform:uppercase;">
                <tr>
                    <th style="padding:0.6rem 0.8rem;">Time (UTC)</th>
                    <th style="padding:0.6rem 0.8rem;">Inverter ID</th>
                    <th style="padding:0.6rem 0.8rem;">Anomaly Type</th>
                    <th style="padding:0.6rem 0.8rem;">Status</th>
                    <th style="padding:0.6rem 0.8rem;">Key Contributors</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    if total_observations > 0:
        st.success(
            f"✅ No anomalies in this period — the inverter behaved normally "
            f"across all {total_observations:,} readings."
        )
    else:
        st.info("No readings in this date range. Try a different range above.")
