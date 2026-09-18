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
import requests
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Inverter Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# THEME
# A dark, technical/industrial palette (navy + amber/teal accents)
# instead of the previous light default. Applied via CSS variables
# so Streamlit's native widgets pick it up too.
# ------------------------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"
BG = "#0B1220"
PANEL = "#111A2C"
PANEL_ALT = "#16223A"
BORDER = "#22314F"
TEXT = "#E7ECF5"
MUTED = "#8CA0C2"
ACCENT = "#3DD6C6"     # teal -- primary series
ACCENT2 = "#F5A623"    # amber -- secondary series / warnings
DANGER = "#FF5C7A"     # anomaly markers
BLUE = "#5B9BD5"

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    .block-container {{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1400px; }}

    section[data-testid="stSidebar"] {{
        background-color: {PANEL}; border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{ color: {TEXT}; }}

    [data-testid="stMetric"] {{
        background-color: {PANEL};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 0.9rem 1rem 0.7rem 1rem;
    }}
    [data-testid="stMetricValue"] {{ font-size: 1.35rem; color: {TEXT}; }}
    [data-testid="stMetricLabel"] {{ color: {MUTED}; }}

    h1, h2, h3, h4, h5 {{ color: {TEXT} !important; }}
    p, span, label, .stMarkdown, .stCaption {{ color: {TEXT}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {MUTED} !important; }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px; border-bottom: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent; color: {MUTED};
        border-radius: 8px 8px 0 0; padding: 0.5rem 1rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PANEL_ALT}; color: {ACCENT} !important;
        border: 1px solid {BORDER}; border-bottom: none;
    }}

    div[data-testid="stExpander"] {{
        background-color: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px;
    }}

    .app-header {{
        background: linear-gradient(120deg, #0F3554 0%, #0B2540 60%, #0B1220 100%);
        border: 1px solid {BORDER};
        padding: 1.5rem 1.8rem; border-radius: 12px; margin-bottom: 1.2rem;
    }}
    .app-header .title {{ color: #FFFFFF; font-size: 1.85rem; font-weight: 700; }}
    .app-header .subtitle {{ color: {MUTED}; font-size: 0.95rem; margin-top: 0.25rem; }}
    .app-header .badge {{
        display: inline-block; margin-top: 0.7rem; padding: 0.2rem 0.6rem;
        border: 1px solid {ACCENT}; color: {ACCENT}; border-radius: 999px;
        font-size: 0.75rem; letter-spacing: 0.02em;
    }}

    .note-box {{
        background-color: {PANEL_ALT}; border-left: 3px solid {ACCENT};
        padding: 0.7rem 0.9rem; border-radius: 6px; font-size: 0.88rem; color: {MUTED};
    }}
    .llm-box {{
        background-color: {PANEL_ALT}; border: 1px solid {BORDER};
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
    Raises requests.HTTPError / requests.RequestException on failure."""
    payload = {
        "model": GROQ_MODEL,
        "temperature": 0.2,
        "max_tokens": 700,
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
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="app-header">
        <div class="title">⚡ Inverter Anomaly Detection Dashboard</div>
        <div class="subtitle">Autoencoder-based anomaly detection with EVT/POT reconstruction-error thresholding</div>
        <div class="badge">Live • Groq-powered explanations</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR -- FILTERS + LLM SETTINGS
# ============================================================

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

with st.sidebar:
    st.markdown("### Filters")
    selected_dates = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    show_anomalies_only = st.checkbox("Show anomalies only", value=False)

    show_inverter_filter = "inverter_id" in df.columns and df["inverter_id"].nunique() > 1
    if show_inverter_filter:
        inverter_options = ["All"] + sorted(df["inverter_id"].dropna().unique().tolist())
        selected_inverter = st.selectbox("Inverter", inverter_options)
    else:
        selected_inverter = "All"

    st.markdown(
        '<div class="note-box">anomaly_score_ratio is reconstruction error '
        "divided by the detection threshold — a ratio, not a probability.</div>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### AI Explanation settings")
    default_key = os.environ.get("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API key",
        value=default_key,
        type="password",
        help="Uses the same model as the notebook (openai/gpt-oss-20b) via "
             "Groq's chat-completions API. Falls back to the GROQ_API_KEY "
             "environment variable if set.",
    )
    st.caption("Key is kept only in this session; never written to disk.")


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
# KPI SECTION (always visible, above tabs)
# ============================================================

total_observations = len(filtered_df)
total_anomalies = int(filtered_df["anomaly_flag"].sum()) if total_observations else 0
anomaly_rate = (total_anomalies / total_observations * 100) if total_observations else None
max_temperature = (
    filtered_df["inverter_temperature_c"].max()
    if "inverter_temperature_c" in filtered_df.columns and total_observations
    else None
)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric("Total Observations", f"{total_observations:,}")
with kpi_cols[1]:
    st.metric("Anomalous Observations", f"{total_anomalies:,}")
with kpi_cols[2]:
    st.metric("Anomaly Rate", fmt_num(anomaly_rate, 2, "%"))
with kpi_cols[3]:
    if "inverter_temperature_c" in filtered_df.columns:
        st.metric("Max Inverter Temperature", fmt_num(max_temperature, 1, " °C"))
    else:
        st.metric("Max Inverter Temperature", "—")

st.write("")


# ============================================================
# TABS
# ============================================================

tab_overview, tab_power, tab_thermal, tab_investigate, tab_raw = st.tabs(
    ["📈 Overview", "⚡ Power", "🌡️ Thermal", "🚨 Anomalies & Investigation", "🗂️ Raw Data"]
)

# ------------------------------------------------------------
# OVERVIEW TAB -- anomaly timeline
# ------------------------------------------------------------
with tab_overview:
    st.subheader("Anomaly Timeline")
    st.caption("Reconstruction error over time. Points above the threshold are flagged as anomalies.")

    if total_observations > 0 and "reconstruction_error" in filtered_df.columns:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"],
                y=filtered_df["reconstruction_error"],
                mode="lines",
                name="Reconstruction Error",
                line=dict(color=ACCENT, width=1.5),
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
                    marker=dict(size=8, color=DANGER, symbol="circle"),
                )
            )

        # Reconstruction error can legitimately be 0 or negative-adjacent
        # (near-perfect reconstruction). A log y-axis silently drops those
        # points instead of showing them at the bottom, which reads as
        # "missing" data. Use log scale only when every value is strictly
        # positive; otherwise fall back to linear so nothing is hidden.
        err_series = pd.to_numeric(filtered_df["reconstruction_error"], errors="coerce").dropna()
        use_log = len(err_series) > 0 and (err_series > 0).all()

        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Reconstruction Error",
            yaxis_type="log" if use_log else "linear",
            hovermode="x unified",
            height=440,
            template=PLOTLY_TEMPLATE,
            paper_bgcolor=PANEL,
            plot_bgcolor=PANEL,
            margin=dict(t=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, width="stretch")
        if not use_log:
            st.caption(
                "Linear scale used here because reconstruction error includes "
                "values at or near zero; a log scale would hide them."
            )
    else:
        st.info("No reconstruction-error data available for the selected period.")

# ------------------------------------------------------------
# POWER TAB
# ------------------------------------------------------------
with tab_power:
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

# ------------------------------------------------------------
# THERMAL TAB
# ------------------------------------------------------------
with tab_thermal:
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

# ------------------------------------------------------------
# ANOMALIES & INVESTIGATION TAB
# ------------------------------------------------------------
with tab_investigate:
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
            fig_trend = go.Figure()

            if "ac_power_kw" in trend_window.columns:
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_window["timestamp"], y=trend_window["ac_power_kw"],
                        mode="lines", name="AC Power (kW)", yaxis="y", line=dict(color=ACCENT),
                    )
                )
            if "dc_current_a" in trend_window.columns:
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_window["timestamp"], y=trend_window["dc_current_a"],
                        mode="lines", name="DC Current (A)", yaxis="y2", line=dict(color=ACCENT2),
                    )
                )
            if "inverter_temperature_c" in trend_window.columns:
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_window["timestamp"], y=trend_window["inverter_temperature_c"],
                        mode="lines", name="Inverter Temperature (°C)", yaxis="y3",
                        line=dict(color=DANGER),
                    )
                )

            fig_trend.add_vline(x=end_time, line_dash="dash", line_width=2, line_color=DANGER)
            fig_trend.add_annotation(
                x=end_time, y=1.1, xref="x", yref="paper", showarrow=False,
                text="Selected anomaly", font=dict(color=DANGER, size=12),
            )

            # --- Fix: with 3 y-axes, the 3rd axis must be anchored "free"
            # and the plotting area narrowed, or it renders on top of / at
            # the same position as the 2nd axis, making both sets of tick
            # labels overlap and misread against the wrong line -- this is
            # what made the chart look like it had "false" values.
            fig_trend.update_layout(
                title="AC Power, DC Current & Temperature — 24 Hours Before Anomaly",
                xaxis=dict(title="Time", domain=[0.0, 0.86]),
                yaxis=dict(
                    title=dict(text="AC Power (kW)", font=dict(color=ACCENT)),
                    tickfont=dict(color=ACCENT),
                    side="left",
                ),
                yaxis2=dict(
                    title=dict(text="DC Current (A)", font=dict(color=ACCENT2)),
                    tickfont=dict(color=ACCENT2),
                    overlaying="y", side="right", anchor="free", position=0.86,
                    showgrid=False,
                ),
                yaxis3=dict(
                    title=dict(text="Temperature (°C)", font=dict(color=DANGER)),
                    tickfont=dict(color=DANGER),
                    overlaying="y", side="right", anchor="free", position=1.0,
                    showgrid=False,
                ),
                hovermode="x unified",
                height=520,
                template=PLOTLY_TEMPLATE,
                paper_bgcolor=PANEL,
                plot_bgcolor=PANEL,
                legend=dict(orientation="h", yanchor="bottom", y=1.16, xanchor="center", x=0.43),
                margin=dict(l=70, r=40, t=100, b=60),
            )
            st.plotly_chart(fig_trend, width="stretch")
            st.caption(
                f"Trend window: {fmt_time(trend_window['timestamp'].min())} → "
                f"{fmt_time(trend_window['timestamp'].max())} "
                f"({len(trend_window):,} observations). Each axis is colour-matched "
                "to its line and positioned independently so the three series "
                "cannot be misread against each other. This shows the historical "
                "period leading up to the anomaly; it does not imply any single "
                "variable caused it."
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
            st.caption("Enter a Groq API key in the sidebar to enable this.")

        if gen_clicked:
            with st.spinner("Calling Groq…"):
                try:
                    explanation = generate_ai_explanation(llm_evidence, groq_api_key)
                    st.session_state[cache_key] = explanation
                except requests.HTTPError as e:
                    st.error(f"Groq API returned an error: {e.response.status_code} {e.response.text[:300]}")
                except requests.RequestException as e:
                    st.error(f"Could not reach Groq API: {e}")

        if cache_key in st.session_state:
            st.markdown(f'<div class="llm-box">{st.session_state[cache_key]}</div>', unsafe_allow_html=True)
    else:
        st.info("No anomalies in the current selection to investigate.")

# ------------------------------------------------------------
# RAW DATA TAB
# ------------------------------------------------------------
with tab_raw:
    st.subheader("Filtered Data")
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
