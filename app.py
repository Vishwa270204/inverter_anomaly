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

AI Explanation is generated live via the Groq API (same provider/model used
in the notebook, openai/gpt-oss-20b), grounded only in the evidence package
already shown on screen.
"""

import json
import os
from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Inverter Anomaly Detection",
    page_icon="⚡",
    layout="wide",
)

# ------------------------------------------------------------
# THEME
# Light, clean palette -- no sidebar, no dark theme. Filters stay
# inline at the top of the page like the original layout.
# ------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_white"
PANEL = "#FFFFFF"
BORDER = "#E3E8EF"
TEXT = "#1B2430"
MUTED = "#5B6B82"
ACCENT = "#4C78A8"     # blue -- primary series
ACCENT2 = "#F58518"    # orange -- secondary series / warnings
DANGER = "#E45756"     # anomaly markers / temperature
BLUE = "#72B7B2"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FFFFFF; }}
    .block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; }}

    [data-testid="stMetric"] {{
        background-color: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.9rem 1rem 0.7rem 1rem;
    }}
    [data-testid="stMetricValue"] {{ font-size: 1.35rem; color: {TEXT}; }}

    .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {BORDER}; }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent; color: {MUTED};
        border-radius: 8px 8px 0 0; padding: 0.5rem 1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: #F0F4F9; color: {ACCENT} !important;
        border: 1px solid {BORDER}; border-bottom: none;
    }}

    .app-header {{
        background-color: #F5F8FC;
        border: 1px solid {BORDER};
        border-left: 5px solid {ACCENT};
        padding: 1.6rem 1.8rem; border-radius: 8px; margin-bottom: 1.4rem;
    }}
    .app-header .title {{ color: {TEXT}; font-size: 1.9rem; font-weight: 600; }}
    .app-header .subtitle {{ color: {MUTED}; font-size: 0.95rem; margin-top: 0.3rem; }}

    .note-box {{
        background-color: #F0F4F9; border-left: 3px solid {ACCENT};
        padding: 0.7rem 0.9rem; border-radius: 6px; font-size: 0.88rem; color: {MUTED};
    }}
    .llm-box {{
        background-color: #FFF8EF; border: 1px solid {BORDER};
        border-left: 3px solid {ACCENT2}; border-radius: 8px;
        padding: 1rem 1.1rem; color: {TEXT}; line-height: 1.55;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


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
trend_df = safe_load(load_trend_data, "trend_data.parquet", "Trend data")

if df.empty:
    st.error("`dashboard_data.parquet` loaded but contains no rows.")
    st.stop()


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


def get_trend_window(trend_source, end_time, hours_back=24):
    """Slice trend_data.parquet to [end_time - hours_back, end_time]."""
    end_time = pd.to_datetime(end_time)
    start_time = end_time - timedelta(hours=hours_back)
    window = trend_source[
        (trend_source["timestamp"] >= start_time) & (trend_source["timestamp"] <= end_time)
    ].copy()
    return window, start_time


def get_contribution_cols(frame):
    """Feature-contribution columns exported from the notebook (Section 14),
    named '<feature>_contribution_pct'."""
    return [c for c in frame.columns if c.endswith("_contribution_pct")]


FEATURE_ICONS = {
    "temperature": "🌡️", "temp": "🌡️",
    "voltage": "⚡", "current": "〰️", "power": "🔋",
    "ambient": "☀️", "irradiance": "☀️", "poa": "☀️", "ghi": "☀️",
    "frequency": "🎛️", "efficiency": "📈", "quality": "🔧",
    "communication": "📡",
}


def feature_icon(feature_name: str) -> str:
    name = feature_name.lower()
    for key, icon in FEATURE_ICONS.items():
        if key in name:
            return icon
    return "▪️"


def compute_persistent_events(period_df):
    """
    Group consecutive flagged anomalies into 'persistent events' the same
    way the notebook does for run detection (Section 13): anomalies that
    are close together in time (within ~2 sampling intervals) are treated
    as one ongoing event rather than separate, unrelated readings.

    Nothing here is invented -- the gap threshold is derived from the
    median spacing of the observations actually in `period_df`.
    """
    if period_df.empty or "anomaly_flag" not in period_df.columns:
        return []

    ordered = period_df.sort_values("timestamp")
    diffs = ordered["timestamp"].diff().dropna()
    sampling_interval = diffs.median() if len(diffs) else timedelta(minutes=15)
    if pd.isna(sampling_interval) or sampling_interval <= timedelta(0):
        sampling_interval = timedelta(minutes=15)
    gap_threshold = max(sampling_interval * 2, timedelta(minutes=1))

    anomalies = ordered[ordered["anomaly_flag"]].reset_index(drop=True)
    if anomalies.empty:
        return []

    events = []
    current_rows = [anomalies.iloc[0]]
    for i in range(1, len(anomalies)):
        row = anomalies.iloc[i]
        if row["timestamp"] - current_rows[-1]["timestamp"] > gap_threshold:
            events.append(current_rows)
            current_rows = [row]
        else:
            current_rows.append(row)
    events.append(current_rows)

    contribution_cols = get_contribution_cols(anomalies)

    out = []
    for rows in events:
        block = pd.DataFrame(rows)
        start_time = block["timestamp"].min()
        end_time = block["timestamp"].max()
        duration_min = max((end_time - start_time).total_seconds() / 60.0, 0.0) + (
            sampling_interval.total_seconds() / 60.0
        )
        peak_score = block["anomaly_score_ratio"].max() if "anomaly_score_ratio" in block else None
        peak_error = block["reconstruction_error"].max() if "reconstruction_error" in block else None

        top_feature = None
        top_feature_pct = None
        if contribution_cols:
            means = block[contribution_cols].mean(numeric_only=True).dropna()
            if len(means) > 0:
                top_col = means.idxmax()
                top_feature = top_col.replace("_contribution_pct", "")
                top_feature_pct = means[top_col]
        elif "top_contributing_feature" in block.columns and block["top_contributing_feature"].notna().any():
            top_feature = block["top_contributing_feature"].mode().iloc[0]

        out.append({
            "start": start_time, "end": end_time, "duration_min": duration_min,
            "count": len(block), "peak_score": peak_score, "peak_error": peak_error,
            "top_feature": top_feature, "top_feature_pct": top_feature_pct,
        })
    return sorted(out, key=lambda e: e["start"])


def ring_gauge_html(percent, color, label, size=76):
    """Small CSS conic-gradient ring (no extra chart library needed)."""
    percent = 0 if percent is None or pd.isna(percent) else max(0.0, min(100.0, percent))
    deg = percent / 100 * 360
    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; gap:0.35rem;">
      <div style="
          width:{size}px; height:{size}px; border-radius:50%;
          background: conic-gradient({color} {deg}deg, #E9EDF3 {deg}deg 360deg);
          display:flex; align-items:center; justify-content:center;">
        <div style="
            width:{size-16}px; height:{size-16}px; border-radius:50%;
            background:#FFFFFF; display:flex; align-items:center; justify-content:center;
            font-size:0.95rem; font-weight:700; color:{color};">
          {percent:.0f}%
        </div>
      </div>
      <div style="font-size:0.78rem; color:#5B6B82;">{label}</div>
    </div>
    """


def kpi_card_html(icon, icon_bg, label, value, sub, value_color="#1B2430"):
    return f"""
    <div style="
        background:#FFFFFF; border:1px solid #E3E8EF; border-radius:10px;
        padding:0.9rem 1rem; display:flex; gap:0.8rem; align-items:flex-start;
        height:100%;">
      <div style="
          width:40px; height:40px; min-width:40px; border-radius:10px;
          background:{icon_bg}; display:flex; align-items:center; justify-content:center;
          font-size:1.2rem;">{icon}</div>
      <div>
        <div style="font-size:0.8rem; color:#5B6B82;">{label}</div>
        <div style="font-size:1.25rem; font-weight:700; color:{value_color}; line-height:1.3;">{value}</div>
        <div style="font-size:0.75rem; color:#8CA0C2;">{sub}</div>
      </div>
    </div>
    """


def build_llm_evidence(anomaly_row, trend_window, contribution_cols):
    """
    Package only the available, factual evidence for a selected anomaly into
    a JSON-serializable structure passed to the LLM.

    Evidence is descriptive only. No causal claims, fault labels, or
    probability language are added here -- that judgment is left to the
    model, and even then bounded by the same rule: describe what the data
    shows, state what it does not establish.
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
            c for c in ["ac_power_kw", "dc_current_a", "dc_power_kw", "inverter_temperature_c",
                        "inverter_ambient_temp_delta", "ambient_temperature_c"]
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


# ============================================================
# GROQ LLM CALL
# ============================================================

GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are an assistant that explains solar-inverter anomaly-detection "
    "output to a technician. You are given a JSON evidence package built "
    "from an Autoencoder's reconstruction error and an EVT/POT-derived "
    "threshold. Rules you must follow strictly:\n"
    "1. anomaly_score_ratio is reconstruction_error / threshold. It is a "
    "ratio, never a probability or confidence percentage.\n"
    "2. Feature contribution percentages describe what the model found "
    "hardest to reconstruct. Call this a 'contribution', never a proven "
    "'cause' or 'root cause' unless the evidence itself unambiguously "
    "establishes causation (it usually does not).\n"
    "3. Only describe correlations/co-occurrence from the trend window as "
    "'changed alongside' the anomaly, not as having caused it.\n"
    "4. Do not invent values, thresholds, fault types, or maintenance "
    "actions that are not present in the evidence.\n"
    "5. End with a short 'What this data does not establish' note.\n"
    "Write 3-5 short paragraphs, plain technical language, no headers."
)


def generate_ai_explanation(evidence: dict, api_key: str) -> str:
    """Call Groq's chat-completions endpoint with the evidence package.
    Raises requests.HTTPError / requests.RequestException / RuntimeError
    on failure.

    openai/gpt-oss-20b is a *reasoning* model: it spends tokens on hidden
    internal reasoning before writing the visible answer. Two things
    matter here to avoid empty/truncated answers:
      - use `max_completion_tokens` (not the older `max_tokens`), and
        give it a generous budget so reasoning doesn't eat the whole thing.
      - set `reasoning_effort` low, since this is a short explanatory
        write-up, not a hard multi-step problem, so little reasoning is
        actually needed.
    """
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0,
        "reasoning_effort": "low",
        "include_reasoning": False,
        "max_completion_tokens": 1536,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(evidence, indent=2)},
        ],
    }
    resp = requests.post(
        GROQ_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise RuntimeError(data["error"].get("message", "Unknown Groq API error."))

    choice = data["choices"][0]
    content = (choice.get("message") or {}).get("content")
    finish_reason = choice.get("finish_reason")

    if not content:
        raise RuntimeError(
            f"Groq returned no visible answer text (finish_reason: "
            f"{finish_reason!r}). This usually means the token budget ran "
            "out during hidden reasoning -- try again, or raise "
            "max_completion_tokens in app.py."
        )
    if finish_reason == "length":
        content += (
            "\n\n*(Note: this explanation was cut off because it hit the "
            "token limit -- raise max_completion_tokens in app.py for "
            "longer answers.)*"
        )
    return content.strip()


# ============================================================
# GROQ API KEY
# ------------------------------------------------------------
# Put your key directly here (simplest option), OR set it as an
# environment variable GROQ_API_KEY before running the app --
# either way works, this line just picks whichever is set.
# ============================================================

GROQ_API_KEY = "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"  # <-- put your key between the quotes
groq_api_key = GROQ_API_KEY if GROQ_API_KEY and "PASTE_YOUR" not in GROQ_API_KEY else os.environ.get("GROQ_API_KEY", "")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="title">⚡ Inverter Anomaly Detection Dashboard</div>
        <div class="subtitle">Autoencoder-based anomaly detection with EVT/POT reconstruction-error thresholding</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILTERS (inline, no sidebar)
# ============================================================

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

show_inverter_filter = "inverter_id" in df.columns and df["inverter_id"].nunique() > 1
filter_cols = st.columns([2, 1, 1] if show_inverter_filter else [2, 1])

with filter_cols[0]:
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

with filter_cols[1]:
    st.write("")  # vertical alignment spacer
    show_anomalies_only = st.checkbox("Show anomalies only", value=False)

if show_inverter_filter:
    with filter_cols[2]:
        inverter_options = ["All"] + sorted(df["inverter_id"].dropna().unique().tolist())
        selected_inverter = st.selectbox("Inverter", inverter_options)
else:
    selected_inverter = "All"

st.caption(
    "`anomaly_score_ratio` is reconstruction error divided by the detection "
    "threshold -- a ratio, not a probability."
)
if not groq_api_key:
    st.caption(
        "⚠️ No Groq API key set. Paste one into the `GROQ_API_KEY` variable "
        "near the top of app.py (or set the GROQ_API_KEY environment "
        "variable) to enable the AI Explanation button below."
    )
st.divider()


# ============================================================
# FILTER DATA
# ============================================================

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    filtered_df = df[
        (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    ].copy()
else:
    # User has only picked one end of the range so far -- show everything
    # rather than erroring, and let them finish picking.
    filtered_df = df.copy()

if selected_inverter != "All" and "inverter_id" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["inverter_id"] == selected_inverter].copy()

if show_anomalies_only:
    filtered_df = filtered_df[filtered_df["anomaly_flag"]].copy()

if filtered_df.empty:
    st.warning("No observations match the current filters. Adjust the date range or filters in the sidebar.")


# ============================================================
# OVERVIEW -- KPI cards, main chart + latest event panel,
# key features / recent anomalies / quick insights
# (layout modeled on the reference dashboard, light theme)
# ============================================================

total_observations = len(filtered_df)
total_anomalies = int(filtered_df["anomaly_flag"].sum()) if total_observations else 0
anomaly_rate = (total_anomalies / total_observations * 100) if total_observations else None
normal_rate = (100 - anomaly_rate) if anomaly_rate is not None else None

persistent_events = compute_persistent_events(filtered_df)
latest_event = persistent_events[-1] if persistent_events else None

latest_row = filtered_df.sort_values("timestamp").iloc[-1] if total_observations else None
is_anomalous_now = bool(latest_row["anomaly_flag"]) if latest_row is not None else False
latest_ratio = (
    latest_row.get("anomaly_score_ratio")
    if latest_row is not None and "anomaly_score_ratio" in filtered_df.columns
    else None
)
severity_percentile = None
if (
    latest_ratio is not None and not pd.isna(latest_ratio)
    and "anomaly_score_ratio" in filtered_df.columns and total_observations > 1
):
    all_ratios = filtered_df["anomaly_score_ratio"].dropna()
    if len(all_ratios) > 1:
        severity_percentile = (all_ratios <= latest_ratio).mean() * 100

kpi_cols = st.columns(5)
with kpi_cols[0]:
    st.markdown(
        kpi_card_html(
            "⚠️" if is_anomalous_now else "✅",
            "#FDE8E8" if is_anomalous_now else "#E3F6EA",
            "Current Status",
            "ANOMALY DETECTED" if is_anomalous_now else "NORMAL",
            "Most recent observation in range",
            DANGER if is_anomalous_now else "#2E8B57",
        ),
        unsafe_allow_html=True,
    )
with kpi_cols[1]:
    st.markdown(
        kpi_card_html(
            "📊", "#E9EEF9", "Anomaly Score",
            fmt_num(latest_ratio, 1, "×") if latest_ratio is not None else "—",
            "Most recent reading, vs. threshold 1.0×",
        ),
        unsafe_allow_html=True,
    )
with kpi_cols[2]:
    ring_color = ACCENT if (severity_percentile or 0) < 66 else ACCENT2 if (severity_percentile or 0) < 90 else DANGER
    st.markdown(
        f'<div style="background:#FFFFFF; border:1px solid {BORDER}; border-radius:10px; '
        f'padding:0.7rem 1rem; height:100%; display:flex; align-items:center; gap:0.8rem;">'
        f'{ring_gauge_html(severity_percentile, ring_color, "Severity")}'
        f'<div><div style="font-size:0.8rem; color:{MUTED};">Severity Percentile</div>'
        f'<div style="font-size:0.75rem; color:#8CA0C2;">vs. other readings in range</div></div></div>',
        unsafe_allow_html=True,
    )
with kpi_cols[3]:
    st.markdown(
        kpi_card_html(
            "🗓️", "#E9EEF9", "Persistent Events",
            f"{len(persistent_events)}",
            "Grouped anomaly runs in range",
        ),
        unsafe_allow_html=True,
    )
with kpi_cols[4]:
    st.markdown(
        kpi_card_html(
            "✅", "#E3F6EA", "Normal-Operation Rate",
            fmt_num(normal_rate, 1, "%"),
            "Share of readings not flagged",
            "#2E8B57",
        ),
        unsafe_allow_html=True,
    )

st.write("")


# ------------------------------------------------------------
# Main chart (left) + Latest Event Details (right)
# ------------------------------------------------------------
main_col, side_col = st.columns([2.1, 1])

with main_col:
    st.markdown("##### Reconstruction Error & Anomaly Detection")
    if total_observations > 0 and "reconstruction_error" in filtered_df.columns:
        fig = go.Figure()

        # Shade each persistent event so runs of anomalies read as one
        # event, the way "Persistent Event" / "Short Event" are called
        # out in the reference design.
        for ev in persistent_events:
            fig.add_vrect(
                x0=ev["start"], x1=ev["end"],
                fillcolor=DANGER, opacity=0.10, line_width=0,
            )

        fig.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"], y=filtered_df["reconstruction_error"],
                mode="lines", name="Reconstruction Error", line=dict(color=ACCENT, width=1.5),
            )
        )

        # Threshold isn't stored directly, but anomaly_score_ratio is
        # defined as reconstruction_error / threshold -- so the threshold
        # can be recovered exactly from rows where the ratio is known,
        # rather than guessed or hard-coded.
        valid_ratio = filtered_df[filtered_df.get("anomaly_score_ratio", pd.Series(dtype=float)) > 0]
        threshold_est = None
        if len(valid_ratio) > 0:
            threshold_est = (valid_ratio["reconstruction_error"] / valid_ratio["anomaly_score_ratio"]).median()
        if threshold_est is not None and not pd.isna(threshold_est):
            fig.add_hline(
                y=threshold_est, line_dash="dash", line_color=DANGER, line_width=1.5,
                annotation_text=f"Threshold ({threshold_est:.6g})", annotation_position="top left",
            )

        anomaly_points = filtered_df[filtered_df["anomaly_flag"]]
        if len(anomaly_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_points["timestamp"], y=anomaly_points["reconstruction_error"],
                    mode="markers", name="Anomaly", marker=dict(size=7, color=DANGER),
                )
            )

        err_series = pd.to_numeric(filtered_df["reconstruction_error"], errors="coerce").dropna()
        use_log = len(err_series) > 0 and (err_series > 0).all()

        fig.update_layout(
            xaxis_title="Time", yaxis_title="Reconstruction Error",
            yaxis_type="log" if use_log else "linear",
            hovermode="x unified", height=430,
            template=PLOTLY_TEMPLATE, paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            margin=dict(t=30), showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "Shaded bands mark grouped persistent events; the dashed line is the "
            "detection threshold recovered from anomaly_score_ratio. "
            f"{'Linear scale used because some values are at/near zero.' if not use_log else ''}"
        )
    else:
        st.info("No reconstruction-error data available for the selected period.")

with side_col:
    st.markdown("##### Latest Event Details")
    if latest_event is not None:
        tag = "Persistent Event" if latest_event["count"] > 1 else "Single Reading"
        st.markdown(
            f'<div style="background:#FFFFFF; border:1px solid {BORDER}; border-radius:10px; '
            f'padding:1rem; height:430px; overflow-y:auto;">'
            f'<span style="background:{"#FDE8E8" if latest_event["count"]>1 else "#F0F4F9"}; color:{DANGER if latest_event["count"]>1 else ACCENT}; '
            f'padding:0.15rem 0.6rem; border-radius:999px; font-size:0.75rem;">{tag}</span>'
            f'<div style="font-weight:700; font-size:1.05rem; margin-top:0.6rem;">'
            f'{fmt_time(latest_event["start"]).split(" ")[1]} – {fmt_time(latest_event["end"]).split(" ")[1]} '
            f'({latest_event["duration_min"]:.0f} min)</div>'
            f'<div style="color:{MUTED}; font-size:0.82rem; margin-bottom:0.8rem;">{fmt_time(latest_event["start"])}</div>'
            f'<div style="display:flex; gap:1.2rem; margin-bottom:0.7rem;">'
            f'<div><div style="font-size:0.75rem; color:{MUTED};">Peak Score</div>'
            f'<div style="font-weight:700;">{fmt_num(latest_event["peak_score"],1,"×")}</div></div>'
            f'<div><div style="font-size:0.75rem; color:{MUTED};">Readings</div>'
            f'<div style="font-weight:700;">{latest_event["count"]}</div></div></div>'
            + (
                f'<div style="font-size:0.75rem; color:{MUTED};">Top Contributing Feature</div>'
                f'<div style="font-weight:600;">{feature_icon(latest_event["top_feature"])} {latest_event["top_feature"]}'
                + (f' — {latest_event["top_feature_pct"]:.1f}% avg contribution' if latest_event["top_feature_pct"] is not None else "")
                + '</div>'
                if latest_event["top_feature"] else ""
            )
            + '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No anomalies in the current selection.")

st.write("")


# ------------------------------------------------------------
# Key Features / Recent Anomalies / Quick Insights
# ------------------------------------------------------------
feat_col, recent_col, insight_col = st.columns([1, 1.3, 1])

anomaly_df_all = filtered_df[filtered_df["anomaly_flag"]].copy() if total_observations else filtered_df.copy()
contribution_cols_all = get_contribution_cols(filtered_df)

with feat_col:
    st.markdown("##### Key Features (Top Contributors)")
    if contribution_cols_all and len(anomaly_df_all) > 0:
        avg_contrib = anomaly_df_all[contribution_cols_all].mean(numeric_only=True).dropna().sort_values(ascending=False)
        palette = [ACCENT, ACCENT2, DANGER, BLUE, "#8B7EC8", "#5B6B82"]
        for i, (col, val) in enumerate(avg_contrib.items()):
            feature_name = col.replace("_contribution_pct", "")
            color = palette[i % len(palette)]
            st.markdown(
                f'<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.55rem;">'
                f'<div style="width:1.3rem;">{feature_icon(feature_name)}</div>'
                f'<div style="flex:1;">'
                f'<div style="font-size:0.82rem; color:{TEXT}; margin-bottom:2px;">{feature_name}</div>'
                f'<div style="background:#EEF1F6; border-radius:6px; height:8px; width:100%;">'
                f'<div style="background:{color}; border-radius:6px; height:8px; width:{min(val,100):.1f}%;"></div>'
                f'</div></div>'
                f'<div style="width:3rem; text-align:right; font-size:0.82rem; color:{MUTED};">{val:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.caption("Average contribution to reconstruction error across current anomalies, not a confirmed cause.")
    else:
        st.info("No feature-contribution data available for this selection.")

with recent_col:
    st.markdown("##### Recent Anomalies")
    if len(anomaly_df_all) > 0:
        recent = anomaly_df_all.sort_values("timestamp", ascending=False).head(7)
        show_cols = [c for c in ["timestamp", "reconstruction_error", "anomaly_score_ratio", "top_contributing_feature"] if c in recent.columns]
        display_recent = recent[show_cols].rename(columns={
            "timestamp": "Time", "reconstruction_error": "Recon. Error",
            "anomaly_score_ratio": "Score", "top_contributing_feature": "Top Feature",
        })
        st.dataframe(display_recent, width="stretch", hide_index=True, height=280)
        st.caption(f"Showing {len(recent)} of {len(anomaly_df_all)} anomalies in range.")
    else:
        st.success("No anomalies detected in the selected period.")

with insight_col:
    st.markdown("##### Quick Insights")
    st.caption("Computed directly from the data above -- not a separate AI call.")
    bullets = []
    if latest_event is not None:
        if latest_event["count"] > 1:
            bullets.append(
                f"🔴 Persistent anomaly from {fmt_time(latest_event['start']).split(' ')[1]} to "
                f"{fmt_time(latest_event['end']).split(' ')[1]} ({latest_event['duration_min']:.0f} min, "
                f"{latest_event['count']} readings)."
            )
        else:
            bullets.append(f"🟠 A single flagged reading at {fmt_time(latest_event['start'])}.")
        if latest_event["top_feature"]:
            bullets.append(
                f"🌡️ **{latest_event['top_feature']}** contributed most to reconstruction error in the latest event."
            )
    if len(persistent_events) > 1:
        bullets.append(f"ℹ️ {len(persistent_events)} separate persistent events detected in the selected range.")
    if total_observations:
        bullets.append(
            f"✅ {total_anomalies:,} of {total_observations:,} readings flagged "
            f"({fmt_num(anomaly_rate, 2, '%')}) in the selected period."
        )
    if not bullets:
        bullets.append("No anomalies in the current selection.")
    for b in bullets:
        st.markdown(
            f'<div style="background:#FFFFFF; border:1px solid {BORDER}; border-radius:8px; '
            f'padding:0.6rem 0.8rem; margin-bottom:0.5rem; font-size:0.85rem;">{b}</div>',
            unsafe_allow_html=True,
        )

st.divider()


# ============================================================
# POWER ANALYSIS
# ============================================================

if True:
    st.subheader("Power Analysis")
    power_col1, power_col2 = st.columns(2)

    with power_col1:
        if total_observations > 0 and "dc_power_kw" in filtered_df.columns:
            fig_dc = go.Figure()
            fig_dc.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"], y=filtered_df["dc_power_kw"],
                    mode="lines", name="DC Power", line=dict(color=ACCENT),
                )
            )
            fig_dc.update_layout(
                title="DC Power", xaxis_title="Time", yaxis_title="DC Power (kW)",
                hovermode="x unified", height=380, template=PLOTLY_TEMPLATE,
                paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            )
            st.plotly_chart(fig_dc, width="stretch")
        else:
            st.info("DC power data not available.")

    with power_col2:
        if total_observations > 0 and "ac_power_kw" in filtered_df.columns:
            fig_ac = go.Figure()
            fig_ac.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"], y=filtered_df["ac_power_kw"],
                    mode="lines", name="AC Power", line=dict(color=BLUE),
                )
            )
            fig_ac.update_layout(
                title="AC Power", xaxis_title="Time", yaxis_title="AC Power (kW)",
                hovermode="x unified", height=380, template=PLOTLY_TEMPLATE,
                paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            )
            st.plotly_chart(fig_ac, width="stretch")
        else:
            st.info("AC power data not available.")

st.divider()


# ============================================================
# THERMAL ANALYSIS
# ============================================================

if True:
    st.subheader("Thermal Analysis")
    thermal_col1, thermal_col2 = st.columns(2)

    with thermal_col1:
        has_temp = "inverter_temperature_c" in filtered_df.columns
        has_ambient = "ambient_temperature_c" in filtered_df.columns
        if total_observations > 0 and (has_temp or has_ambient):
            fig_temp = go.Figure()
            if has_temp:
                fig_temp.add_trace(
                    go.Scatter(
                        x=filtered_df["timestamp"], y=filtered_df["inverter_temperature_c"],
                        mode="lines", name="Inverter Temp (°C)", line=dict(color=DANGER),
                    )
                )
            if has_ambient:
                fig_temp.add_trace(
                    go.Scatter(
                        x=filtered_df["timestamp"], y=filtered_df["ambient_temperature_c"],
                        mode="lines", name="Ambient Temp (°C)", line=dict(color=ACCENT2),
                    )
                )
            fig_temp.update_layout(
                title="Inverter vs. Ambient Temperature", xaxis_title="Time",
                yaxis_title="Temperature (°C)", hovermode="x unified",
                height=380, template=PLOTLY_TEMPLATE,
                paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            )
            st.plotly_chart(fig_temp, width="stretch")
        else:
            st.info("Temperature data not available.")

    with thermal_col2:
        if total_observations > 0 and "inverter_ambient_temp_delta" in filtered_df.columns:
            fig_delta = go.Figure()
            fig_delta.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"], y=filtered_df["inverter_ambient_temp_delta"],
                    mode="lines", name="Temp Delta (°C)", line=dict(color="#B279A2"),
                )
            )
            anomaly_points = filtered_df[filtered_df["anomaly_flag"]]
            if len(anomaly_points) > 0:
                fig_delta.add_trace(
                    go.Scatter(
                        x=anomaly_points["timestamp"], y=anomaly_points["inverter_ambient_temp_delta"],
                        mode="markers", name="Flagged anomaly", marker=dict(size=7, color=DANGER),
                    )
                )
            fig_delta.update_layout(
                title="Inverter-to-Ambient Temperature Difference", xaxis_title="Time",
                yaxis_title="Temperature Difference (°C)", hovermode="x unified",
                height=380, template=PLOTLY_TEMPLATE,
                paper_bgcolor=PANEL, plot_bgcolor=PANEL,
            )
            st.plotly_chart(fig_delta, width="stretch")
        else:
            st.info("Inverter-to-ambient temperature delta not available.")

st.divider()


# ============================================================
# DETECTED ANOMALIES + INVESTIGATION
# ============================================================

if True:
    st.subheader("Detected Anomalies")

    anomaly_df = filtered_df[filtered_df["anomaly_flag"]].copy() if total_observations else filtered_df.copy()

    if len(anomaly_df) > 0:
        display_columns = [
            "timestamp", "reconstruction_error", "anomaly_score_ratio",
            "top_contributing_feature", "inverter_temperature_c",
            "ac_power_kw", "dc_power_kw", "anomaly_reason",
        ]
        display_columns = [c for c in display_columns if c in anomaly_df.columns]
        st.dataframe(
            anomaly_df[display_columns].sort_values("timestamp"),
            width="stretch",
            hide_index=True,
        )
    else:
        st.success("No anomalies detected in the selected period.")

    st.divider()
    st.subheader("Anomaly Investigation")

    if len(anomaly_df) > 0:
        anomaly_df = anomaly_df.sort_values("timestamp").reset_index(drop=True)

        selected_index = st.selectbox(
            "Select an anomaly observation",
            range(len(anomaly_df)),
            format_func=lambda x: fmt_time(anomaly_df.loc[x, "timestamp"]),
        )
        selected_anomaly = anomaly_df.loc[selected_index]

        # --- Selected anomaly summary ---
        inv_cols = st.columns(4)
        with inv_cols[0]:
            st.metric("Time", fmt_time(selected_anomaly.get("timestamp")))
        with inv_cols[1]:
            st.metric("Reconstruction Error", fmt_num(selected_anomaly.get("reconstruction_error"), 4))
        with inv_cols[2]:
            st.metric("Score / Threshold Ratio", fmt_num(selected_anomaly.get("anomaly_score_ratio"), 2, "×"))
        with inv_cols[3]:
            st.metric("Inverter Temperature", fmt_num(selected_anomaly.get("inverter_temperature_c"), 1, " °C"))

        with st.expander("Other available values at this observation"):
            other_cols = [
                c for c in [
                    "ambient_temperature_c", "inverter_ambient_temp_delta", "temp_delta_zscore",
                    "dc_power_kw", "ac_power_kw", "dc_voltage_v", "dc_current_a",
                    "ac_voltage_v", "ac_current_a", "power_factor", "frequency_hz",
                    "efficiency_pct", "poa_w_m2", "ghi_w_m2", "quality_code_inv",
                    "communication_status_inv", "anomaly_reason",
                ]
                if c in selected_anomaly.index
            ]
            if other_cols:
                other_values = selected_anomaly[other_cols].rename("value").to_frame()
                other_values["value"] = other_values["value"].astype(str)
                st.dataframe(other_values, width="stretch")
            else:
                st.caption("No additional columns available.")

        # ========================================================
        # 24-HOUR PRE-ANOMALY TREND (from trend_data.parquet)
        # ========================================================
        st.markdown("#### 24-Hour Trend Before Selected Anomaly")

        end_time = pd.to_datetime(selected_anomaly["timestamp"])
        trend_window, window_start = get_trend_window(trend_df, end_time, hours_back=24)

        if len(trend_window) > 0:
            # --- Fix: three series sharing one plot area with three
            # overlaid y-axes is inherently misleading -- each axis
            # autoscales independently, so lines visually "cross" or
            # "track together" at points that mean nothing (a temperature
            # line dipping through a power line is not a relationship,
            # just two unrelated scales drawn on top of each other). That
            # is what read as "wrong info". Fixed by giving each series
            # its own stacked panel with its own y-axis, sharing only the
            # time axis -- nothing is ever plotted against the wrong scale.
            panel_specs = [
                ("ac_power_kw", "AC Power (kW)", ACCENT),
                ("dc_current_a", "DC Current (A)", ACCENT2),
                ("inverter_temperature_c", "Inverter Temp (°C)", DANGER),
            ]
            panel_specs = [p for p in panel_specs if p[0] in trend_window.columns]

            fig_trend = make_subplots(
                rows=len(panel_specs), cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=[label for _, label, _ in panel_specs],
            )

            for i, (col, label, color) in enumerate(panel_specs, start=1):
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_window["timestamp"], y=trend_window[col],
                        mode="lines", name=label, line=dict(color=color),
                        showlegend=False,
                    ),
                    row=i, col=1,
                )
                fig_trend.add_vline(
                    x=end_time, line_dash="dash", line_width=1.5, line_color=DANGER,
                    row=i, col=1,
                )
                fig_trend.update_yaxes(title_text=label, row=i, col=1)

            fig_trend.update_xaxes(title_text="Time", row=len(panel_specs), col=1)
            fig_trend.update_layout(
                title="AC Power, DC Current & Temperature — 24 Hours Before Anomaly (each on its own scale)",
                height=180 * len(panel_specs) + 120,
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=PANEL,
                plot_bgcolor=PANEL,
                hovermode="x unified",
                margin=dict(l=70, r=30, t=70, b=50),
            )
            st.plotly_chart(fig_trend, width="stretch")
            st.caption(
                f"Trend window: {fmt_time(trend_window['timestamp'].min())} → "
                f"{fmt_time(trend_window['timestamp'].max())} "
                f"({len(trend_window):,} observations). Each variable has its own "
                "panel and its own scale, so nothing is plotted against the wrong "
                "axis. The dashed red line marks the selected anomaly. This shows "
                "the historical period leading up to the anomaly; it does not "
                "imply any single variable caused it."
            )
        else:
            st.info(
                "No trend data is available in the 24 hours before this anomaly "
                "(e.g. it may be at the very start of the recorded history)."
            )

        st.divider()

        # ========================================================
        # FEATURE CONTRIBUTIONS
        # ========================================================
        st.markdown("#### Features Contributing to Reconstruction Error")

        contribution_cols = get_contribution_cols(anomaly_df)
        if contribution_cols:
            contrib_values = selected_anomaly[contribution_cols].dropna()
            if len(contrib_values) > 0:
                contrib_plot_df = pd.DataFrame({
                    "feature": [c.replace("_contribution_pct", "") for c in contrib_values.index],
                    "contribution_pct": contrib_values.values,
                }).sort_values("contribution_pct", ascending=True)

                fig_contrib = go.Figure()
                fig_contrib.add_trace(
                    go.Bar(
                        x=contrib_plot_df["contribution_pct"],
                        y=contrib_plot_df["feature"],
                        orientation="h",
                        marker_color=ACCENT,
                    )
                )
                fig_contrib.update_layout(
                    title="Features contributing to reconstruction error",
                    xaxis_title="Contribution to reconstruction error (%)",
                    yaxis_title="",
                    height=280,
                    template=PLOTLY_TEMPLATE,
                    paper_bgcolor=PANEL,
                    plot_bgcolor=PANEL,
                    margin=dict(t=50),
                )
                st.plotly_chart(fig_contrib, width="stretch")
                st.caption(
                    "This shows which feature the model found hardest to reconstruct "
                    "for this observation, not a confirmed root cause."
                )
            else:
                st.info("No feature-contribution values available for this observation.")
        else:
            st.info(
                "No feature-contribution columns were found in the dashboard data. "
                "Re-export `dashboard_data.parquet` from the notebook (Section 14 / 18A) "
                "to include them."
            )

        st.divider()

        # ========================================================
        # AI EXPLANATION -- live Groq call
        # ========================================================
        st.markdown("#### AI Explanation")
        st.caption(
            "Generated by Groq (openai/gpt-oss-20b), grounded only in the evidence "
            "package below -- the same values already shown above."
        )

        llm_evidence = build_llm_evidence(selected_anomaly, trend_window, contribution_cols)

        with st.expander("Evidence package sent to the model (JSON)"):
            st.json(llm_evidence)

        cache_key = f"ai_explanation::{selected_anomaly.get('timestamp')}"

        gen_clicked = st.button("Generate AI Explanation", disabled=not bool(groq_api_key))
        if not groq_api_key:
            st.caption("Set GROQ_API_KEY near the top of app.py to enable this.")

        if gen_clicked:
            with st.spinner("Calling Groq…"):
                try:
                    explanation = generate_ai_explanation(llm_evidence, groq_api_key)
                    st.session_state[cache_key] = explanation
                except requests.HTTPError as e:
                    st.error(f"Groq API returned an error: {e.response.status_code} {e.response.text[:300]}")
                except requests.RequestException as e:
                    st.error(f"Could not reach Groq API: {e}")
                except RuntimeError as e:
                    st.error(str(e))

        if cache_key in st.session_state:
            st.markdown(f'<div class="llm-box">{st.session_state[cache_key]}</div>', unsafe_allow_html=True)
    else:
        st.info("No anomalies in the current selection to investigate.")

st.divider()


# ============================================================
# RAW DATA
# ============================================================

with st.expander("View filtered data"):
    st.dataframe(filtered_df, width="stretch", hide_index=True)


# ============================================================
# FOOTER
# ============================================================

st.divider()
st.caption(
    "Anomaly detection results are generated by the trained Autoencoder and "
    "an EVT/POT-derived reconstruction-error threshold, computed in "
    "`inverter_anomaly.ipynb`. `anomaly_score_ratio` is a ratio of "
    "reconstruction error to that threshold, not a probability. Feature "
    "contributions describe what drove reconstruction error, not a "
    "confirmed physical cause. AI explanations are generated live and "
    "constrained to the evidence shown above them."
)
