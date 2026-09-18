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

import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Inverter Anomaly Detection",
    page_icon="☀️",
    layout="wide",
)

# ------------------------------------------------------------
# THEME -- warm solar-plant palette. Light background, no dark
# mode. One bold gradient moment (the header); everything else
# stays calm and readable.
# ------------------------------------------------------------
SUN = "#E8A33D"        # amber -- normal operation / "sun"
SKY = "#0E5C67"        # deep teal -- brand / primary
ALERT = "#D6483F"      # warm red -- anomalies
INK = "#22303C"        # body text
MUTED = "#5B6B73"      # secondary text
PAPER = "#FBF8F2"      # page background (warm off-white, not stark white)
LINE = "#E4DCC9"       # hairline borders

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {PAPER}; }}
    .block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
    html, body, [class*="css"] {{ color: {INK}; }}

    [data-testid="stMetricValue"] {{ font-size: 1.4rem; }}

    h2, h3 {{ color: {INK}; font-weight: 700; }}

    .section-lede {{
        color: {MUTED};
        font-size: 0.92rem;
        margin-top: -0.4rem;
        margin-bottom: 0.8rem;
    }}

    /* KPI strip -- colored top border instead of boxed shadow cards */
    .kpi {{
        border-top: 3px solid var(--accent, {SKY});
        background-color: #FFFFFF;
        border-radius: 6px;
        padding: 0.85rem 1rem 0.9rem 1rem;
    }}
    .kpi-label {{
        font-size: 0.8rem;
        color: {MUTED};
        margin-bottom: 0.15rem;
    }}
    .kpi-value {{
        font-size: 1.65rem;
        font-weight: 700;
        color: {INK};
        line-height: 1.15;
    }}

    /* Plain-language callout badges */
    .badge {{
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }}
    .badge-alert {{ background-color: #FBEAE8; color: {ALERT}; }}
    .badge-ok {{ background-color: #E9F3EC; color: #1E7A4C; }}

    .insight-card {{
        background-color: #FFFFFF;
        border: 1px solid {LINE};
        border-left: 4px solid {SUN};
        border-radius: 6px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
    }}
    .insight-card b {{ color: {INK}; }}
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
# LLM CLIENT (Groq) -- for AI-generated explanations
# ============================================================

# Hardcoded directly here -- replace with your actual Groq API key.
GROQ_API_KEY = "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"


@st.cache_resource
def get_groq_client():
    if not GROQ_API_KEY or GROQ_API_KEY == "YOUR_GROQ_API_KEY_HERE":
        return None
    return Groq(api_key=GROQ_API_KEY)


client = get_groq_client()


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


def kpi_card(col, label, value, accent):
    with col:
        st.markdown(
            f"""
            <div class="kpi" style="--accent:{accent};">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


FRIENDLY_METRICS = {
    "inverter_temperature_c": "Inverter Temperature (°C)",
    "ac_power_kw": "AC Power Output (kW)",
    "dc_power_kw": "DC Power Output (kW)",
}


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


def plot_metric_with_anomalies(data, metric_col, metric_label, height=460, mark_peak=True):
    """
    The plain-language anomaly chart: a real, physical quantity (temperature
    or power) plotted over time, with a shaded 'typical range' band and
    anomalies marked as clear red points. The single highest anomaly point
    is called out with a peak annotation, so a non-technical viewer can see
    at a glance *what* spiked and *by how much* -- without needing to
    understand reconstruction error or z-scores.
    """
    plot_df = data.dropna(subset=[metric_col]).copy()
    if plot_df.empty:
        st.info(f"No {metric_label} data available for the selected period.")
        return

    series = plot_df[metric_col]
    mean_val = series.mean()
    std_val = series.std() if series.std() and not pd.isna(series.std()) else 0
    band_upper = mean_val + 2 * std_val
    band_lower = mean_val - 2 * std_val

    fig = go.Figure()

    # Typical-range shading, drawn first so the line sits on top of it.
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"], y=[band_upper] * len(plot_df),
            mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"], y=[band_lower] * len(plot_df),
            mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False,
            fill="tonexty", fillcolor="rgba(232,163,61,0.12)",
            name="Typical range",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"], y=series,
            mode="lines", name=metric_label,
            line=dict(color=SKY, width=1.8),
        )
    )

    anomaly_points = plot_df[plot_df["anomaly_flag"]] if "anomaly_flag" in plot_df.columns else plot_df.iloc[0:0]
    if len(anomaly_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=anomaly_points["timestamp"], y=anomaly_points[metric_col],
                mode="markers", name="Flagged as unusual",
                marker=dict(size=9, color=ALERT, line=dict(width=1, color="white")),
            )
        )

        if mark_peak:
            peak_row = anomaly_points.loc[anomaly_points[metric_col].idxmax()]
            fig.add_annotation(
                x=peak_row["timestamp"], y=peak_row[metric_col],
                text=f"Peak: {peak_row[metric_col]:,.1f}<br>{fmt_time(peak_row['timestamp'])}",
                showarrow=True, arrowhead=2, arrowcolor=ALERT, ax=0, ay=-45,
                bgcolor="white", bordercolor=ALERT, borderwidth=1, borderpad=4,
                font=dict(color=ALERT, size=12),
            )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=metric_label,
        hovermode="x unified",
        height=height,
        template="plotly_white",
        margin=dict(t=20, l=60, r=30, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    st.plotly_chart(fig, width="stretch")


# ============================================================
# HEADER
# ============================================================

st.markdown(
    f"""
    <div style="
        background: linear-gradient(120deg, {SKY} 0%, #16777F 55%, {SUN} 130%);
        padding:1.7rem 1.9rem;
        border-radius:10px;
        margin-bottom:1.4rem;
    ">
        <div style="color:#FFFFFF; font-size:1.9rem; font-weight:700; line-height:1.2;">
            ☀️ Inverter Health &amp; Anomaly Dashboard
        </div>
        <div style="color:#EAF6F2; font-size:0.97rem; margin-top:0.35rem;">
            A plain-language view of how your inverters are running, with unusual
            behavior flagged and explained.
        </div>
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
    show_anomalies_only = st.checkbox("Show unusual periods only", value=False)

if show_inverter_filter:
    with filter_cols[2]:
        inverter_options = ["All"] + sorted(df["inverter_id"].dropna().unique().tolist())
        selected_inverter = st.selectbox("Inverter", inverter_options)
else:
    selected_inverter = "All"

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
    st.warning("No observations match the current filters. Adjust the date range or filters above.")


# ============================================================
# KPI SECTION
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
kpi_card(kpi_cols[0], "Observations", f"{total_observations:,}", SKY)
kpi_card(kpi_cols[1], "Unusual Periods Flagged", f"{total_anomalies:,}", ALERT)
kpi_card(kpi_cols[2], "Share Flagged", fmt_num(anomaly_rate, 2, "%"), SUN)
kpi_card(kpi_cols[3], "Peak Inverter Temperature", fmt_num(max_temperature, 1, " °C"), "#7A5195")

st.write("")
st.divider()


# ============================================================
# ANOMALY DETECTION -- the main, plain-language chart
# ============================================================

st.subheader("What's Unusual, and When")
st.markdown(
    '<div class="section-lede">The line below is a real measurement from your '
    "inverter. The shaded band is its typical range. Red dots mark moments the "
    "system flagged as unusual, and the peak is called out directly.</div>",
    unsafe_allow_html=True,
)

available_metrics = [c for c in FRIENDLY_METRICS if c in filtered_df.columns]
if available_metrics:
    metric_choice = st.radio(
        "Metric",
        available_metrics,
        format_func=lambda c: FRIENDLY_METRICS[c],
        horizontal=True,
        label_visibility="collapsed",
    )
    if total_observations > 0:
        plot_metric_with_anomalies(filtered_df, metric_choice, FRIENDLY_METRICS[metric_choice])
    else:
        st.info("No data available for the selected period.")
else:
    st.info("None of the plain-language metrics (temperature, AC/DC power) are available in this data.")

st.divider()


# ============================================================
# POWER & TEMPERATURE -- simple, understandable trend charts
# ============================================================

st.subheader("Power Output & Temperature")
st.markdown(
    '<div class="section-lede">A quick look at overall output and heat, side by side.</div>',
    unsafe_allow_html=True,
)

trend_col1, trend_col2 = st.columns(2)

with trend_col1:
    has_ac = "ac_power_kw" in filtered_df.columns
    has_dc = "dc_power_kw" in filtered_df.columns
    if total_observations > 0 and (has_ac or has_dc):
        fig_power = go.Figure()
        if has_ac:
            fig_power.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"], y=filtered_df["ac_power_kw"],
                    mode="lines", name="AC Power (kW)", line=dict(color=SKY, width=1.8),
                )
            )
        if has_dc:
            fig_power.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"], y=filtered_df["dc_power_kw"],
                    mode="lines", name="DC Power (kW)", line=dict(color=SUN, width=1.8),
                )
            )
        fig_power.update_layout(
            title="Power Output", xaxis_title="Time", yaxis_title="Power (kW)",
            hovermode="x unified", height=360, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=60),
        )
        st.plotly_chart(fig_power, width="stretch")
    else:
        st.info("Power data not available.")

with trend_col2:
    has_temp = "inverter_temperature_c" in filtered_df.columns
    has_ambient = "ambient_temperature_c" in filtered_df.columns
    if total_observations > 0 and (has_temp or has_ambient):
        fig_temp = go.Figure()
        if has_temp:
            fig_temp.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"], y=filtered_df["inverter_temperature_c"],
                    mode="lines", name="Inverter Temp (°C)", line=dict(color=ALERT, width=1.8),
                )
            )
        if has_ambient:
            fig_temp.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"], y=filtered_df["ambient_temperature_c"],
                    mode="lines", name="Outside Temp (°C)", line=dict(color=SUN, width=1.8, dash="dot"),
                )
            )
        fig_temp.update_layout(
            title="Inverter vs. Outside Temperature", xaxis_title="Time",
            yaxis_title="Temperature (°C)", hovermode="x unified",
            height=360, template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=60),
        )
        st.plotly_chart(fig_temp, width="stretch")

        if has_temp and has_ambient and total_observations > 0:
            avg_delta = (filtered_df["inverter_temperature_c"] - filtered_df["ambient_temperature_c"]).mean()
            if pd.notna(avg_delta):
                st.caption(f"On average, the inverter runs about **{avg_delta:.1f}°C warmer** than the outside air.")
    else:
        st.info("Temperature data not available.")

st.divider()


# ============================================================
# ADVANCED / TECHNICAL DETAIL -- kept, but out of the way
# ============================================================

with st.expander("Advanced detail (for engineers): model score & feature contributions"):
    st.caption(
        "`anomaly_score_ratio` is reconstruction error divided by the detection "
        "threshold -- a ratio, not a probability."
    )

    if total_observations > 0 and "reconstruction_error" in filtered_df.columns:
        fig_re = go.Figure()
        fig_re.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"], y=filtered_df["reconstruction_error"],
                mode="lines", name="Reconstruction Error", line=dict(color=SKY, width=1.3),
            )
        )
        anomaly_points = filtered_df[filtered_df["anomaly_flag"]]
        if len(anomaly_points) > 0:
            fig_re.add_trace(
                go.Scatter(
                    x=anomaly_points["timestamp"], y=anomaly_points["reconstruction_error"],
                    mode="markers", name="Flagged anomaly", marker=dict(size=7, color=ALERT),
                )
            )
        fig_re.update_layout(
            title="Reconstruction Error Over Time (log scale)",
            xaxis_title="Time", yaxis_title="Reconstruction Error", yaxis_type="log",
            hovermode="x unified", height=360, template="plotly_white", margin=dict(t=50),
        )
        st.plotly_chart(fig_re, width="stretch")
    else:
        st.info("No reconstruction-error data available for the selected period.")

    if "inverter_ambient_temp_delta" in filtered_df.columns and total_observations > 0:
        fig_delta = go.Figure()
        fig_delta.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"], y=filtered_df["inverter_ambient_temp_delta"],
                mode="lines", name="Temp Delta (°C)", line=dict(color="#7A5195"),
            )
        )
        fig_delta.update_layout(
            title="Inverter-to-Ambient Temperature Difference", xaxis_title="Time",
            yaxis_title="Temperature Difference (°C)", hovermode="x unified",
            height=320, template="plotly_white", margin=dict(t=50),
        )
        st.plotly_chart(fig_delta, width="stretch")

st.divider()


# ============================================================
# DETECTED ANOMALIES TABLE
# ============================================================

st.subheader("Flagged Periods")

anomaly_df = filtered_df[filtered_df["anomaly_flag"]].copy() if total_observations else filtered_df.copy()

if len(anomaly_df) > 0:
    display_columns = [
        "timestamp", "inverter_temperature_c", "ac_power_kw", "dc_power_kw",
        "anomaly_reason", "top_contributing_feature",
        "reconstruction_error", "anomaly_score_ratio",
    ]
    display_columns = [c for c in display_columns if c in anomaly_df.columns]
    friendly_headers = {
        "timestamp": "Time",
        "inverter_temperature_c": "Inverter Temp (°C)",
        "ac_power_kw": "AC Power (kW)",
        "dc_power_kw": "DC Power (kW)",
        "anomaly_reason": "Reason Flagged",
        "top_contributing_feature": "Main Factor",
        "reconstruction_error": "Model Score (raw)",
        "anomaly_score_ratio": "Model Score (× threshold)",
    }
    st.dataframe(
        anomaly_df[display_columns].sort_values("timestamp").rename(columns=friendly_headers),
        width="stretch",
        hide_index=True,
    )
else:
    st.markdown('<span class="badge badge-ok">✓ No unusual periods in the selected range</span>', unsafe_allow_html=True)

st.divider()


# ============================================================
# ANOMALY INVESTIGATION
# ============================================================

st.subheader("Look Into a Flagged Period")

if len(anomaly_df) > 0:
    anomaly_df = anomaly_df.sort_values("timestamp").reset_index(drop=True)

    selected_index = st.selectbox(
        "Select a flagged observation",
        range(len(anomaly_df)),
        format_func=lambda x: fmt_time(anomaly_df.loc[x, "timestamp"]),
    )
    selected_anomaly = anomaly_df.loc[selected_index]

    # --- Selected anomaly summary ---
    inv_cols = st.columns(4)
    with inv_cols[0]:
        st.metric("Time", fmt_time(selected_anomaly.get("timestamp")))
    with inv_cols[1]:
        st.metric("Inverter Temperature", fmt_num(selected_anomaly.get("inverter_temperature_c"), 1, " °C"))
    with inv_cols[2]:
        st.metric("AC Power", fmt_num(selected_anomaly.get("ac_power_kw"), 1, " kW"))
    with inv_cols[3]:
        st.metric("Model Score (× threshold)", fmt_num(selected_anomaly.get("anomaly_score_ratio"), 2, "×"))

    reason = selected_anomaly.get("anomaly_reason")
    top_feature = selected_anomaly.get("top_contributing_feature")
    if isinstance(reason, str) and reason.strip():
        st.markdown(
            f'<div class="insight-card"><b>Why it was flagged:</b> {reason}</div>',
            unsafe_allow_html=True,
        )
    elif isinstance(top_feature, str) and top_feature.strip():
        st.markdown(
            f'<div class="insight-card"><b>Main factor the model noticed:</b> {top_feature.replace("_", " ")}</div>',
            unsafe_allow_html=True,
        )

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
    st.markdown("#### The 24 Hours Leading Up to This")

    end_time = pd.to_datetime(selected_anomaly["timestamp"])
    trend_window, window_start = get_trend_window(trend_df, end_time, hours_back=24)

    if len(trend_window) > 0:
        investigate_metric = "inverter_temperature_c" if "inverter_temperature_c" in trend_window.columns else (
            "ac_power_kw" if "ac_power_kw" in trend_window.columns else None
        )

        if investigate_metric:
            fig_inv = go.Figure()
            fig_inv.add_trace(
                go.Scatter(
                    x=trend_window["timestamp"], y=trend_window[investigate_metric],
                    mode="lines", name=FRIENDLY_METRICS.get(investigate_metric, investigate_metric),
                    line=dict(color=SKY, width=1.8),
                )
            )
            fig_inv.add_vline(x=end_time, line_dash="dash", line_width=2, line_color=ALERT)
            fig_inv.add_annotation(
                x=end_time, y=1.06, xref="x", yref="paper", showarrow=False,
                text="Flagged moment", font=dict(color=ALERT, size=12),
            )
            fig_inv.update_layout(
                title=f"{FRIENDLY_METRICS.get(investigate_metric, investigate_metric)} — 24 Hours Before",
                xaxis_title="Time", yaxis_title=FRIENDLY_METRICS.get(investigate_metric, investigate_metric),
                hovermode="x unified", height=420, template="plotly_white", margin=dict(t=60),
            )
            st.plotly_chart(fig_inv, width="stretch")

        st.caption(
            f"Window shown: {fmt_time(trend_window['timestamp'].min())} → "
            f"{fmt_time(trend_window['timestamp'].max())} "
            f"({len(trend_window):,} readings). This shows what led up to the "
            "flagged moment; it does not by itself prove what caused it."
        )

        with st.expander("Full technical trend (AC power, DC current & temperature together)"):
            fig_trend = go.Figure()
            if "ac_power_kw" in trend_window.columns:
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_window["timestamp"], y=trend_window["ac_power_kw"],
                        mode="lines", name="AC Power (kW)", yaxis="y", line=dict(color=SKY),
                    )
                )
            if "dc_current_a" in trend_window.columns:
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_window["timestamp"], y=trend_window["dc_current_a"],
                        mode="lines", name="DC Current (A)", yaxis="y2", line=dict(color=SUN),
                    )
                )
            if "inverter_temperature_c" in trend_window.columns:
                fig_trend.add_trace(
                    go.Scatter(
                        x=trend_window["timestamp"], y=trend_window["inverter_temperature_c"],
                        mode="lines", name="Inverter Temperature (°C)", yaxis="y3",
                        line=dict(color=ALERT),
                    )
                )
            fig_trend.add_vline(x=end_time, line_dash="dash", line_width=2, line_color=ALERT)
            fig_trend.update_layout(
                xaxis=dict(title="Time"),
                yaxis=dict(title="AC Power (kW)", side="left"),
                yaxis2=dict(title="DC Current (A)", overlaying="y", side="right"),
                yaxis3=dict(title="Temperature (°C)", overlaying="y", side="right", position=0.94),
                hovermode="x unified", height=480, template="plotly_white",
                legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
                margin=dict(l=70, r=110, t=70, b=60),
            )
            st.plotly_chart(fig_trend, width="stretch")
    else:
        st.info(
            "No trend data is available in the 24 hours before this observation "
            "(e.g. it may be at the very start of the recorded history)."
        )

    st.divider()

    # ========================================================
    # FEATURE CONTRIBUTIONS -- moved into an expander (technical)
    # ========================================================
    contribution_cols = get_contribution_cols(anomaly_df)
    with st.expander("Advanced detail: which measurements the model found hardest to explain"):
        if contribution_cols:
            contrib_values = selected_anomaly[contribution_cols].dropna()
            if len(contrib_values) > 0:
                contrib_plot_df = pd.DataFrame({
                    "feature": [c.replace("_contribution_pct", "").replace("_", " ") for c in contrib_values.index],
                    "contribution_pct": contrib_values.values,
                }).sort_values("contribution_pct", ascending=True)

                fig_contrib = go.Figure()
                fig_contrib.add_trace(
                    go.Bar(
                        x=contrib_plot_df["contribution_pct"],
                        y=contrib_plot_df["feature"],
                        orientation="h",
                        marker_color=SKY,
                    )
                )
                fig_contrib.update_layout(
                    xaxis_title="Contribution to reconstruction error (%)",
                    yaxis_title="",
                    height=260,
                    template="plotly_white",
                    margin=dict(t=20),
                )
                st.plotly_chart(fig_contrib, width="stretch")
                st.caption(
                    "This shows which measurement the model found hardest to reconstruct "
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
    # AI EXPLANATION (evidence only -- LLM call added later)
    # ========================================================
    st.markdown("#### AI Explanation")
    st.caption(
        "An LLM-generated explanation will be added here in a future version. "
        "Below is the exact evidence package that will be passed to it -- built "
        "only from values already shown above, so the explanation stays "
        "grounded in what the data actually supports."
    )

    llm_evidence = build_llm_evidence(selected_anomaly, trend_window, contribution_cols)

    with st.expander("Evidence package for AI explanation (JSON)"):
        st.json(llm_evidence)

    EXPLANATION_PROMPT_TEMPLATE = """
You are an AI assistant explaining an industrial inverter anomaly
detected by a machine-learning model.

STRICT EVIDENCE RULES
- Use ONLY the numbers and timestamps given in EVIDENCE below.
- Do NOT invent sensor readings, fault codes, weather, or maintenance history.
- Keep "before_anomaly" and "during_anomaly" evidence strictly separate.
- A temporal relationship does NOT prove causation -- phrase causes as
  "likely" or "consistent with", never as certain fact.
- If the evidence is insufficient to say why the anomaly happened, say so
  plainly instead of guessing.

OUTPUT FORMAT (keep it SHORT -- max ~120 words total, plain text, no markdown headers)
When: <one line -- start time, end time>
Why: <1-2 sentences -- most likely cause(s), based only on the evidence,
      referencing which measurement(s) moved and by how much>
Solution: <1-2 sentences -- concrete, practical next step a technician or
           engineer could take to confirm the cause and fix/prevent it>

EVIDENCE
============================================================
{evidence_json}
"""

    if client is None:
        st.warning(
            "AI explanations aren't configured yet. Add `GROQ_API_KEY` to "
            "`.streamlit/secrets.toml` or set it as an environment variable "
            "to enable this."
        )
    elif st.button("Generate AI Explanation"):
        with st.spinner("Generating explanation..."):
            try:
                prompt = EXPLANATION_PROMPT_TEMPLATE.format(
                    evidence_json=json.dumps(llm_evidence, indent=2)
                )
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "system", "content": (
                            "You are an industrial anomaly-analysis assistant. "
                            "Use only the supplied evidence. Keep before-anomaly "
                            "and during-anomaly evidence strictly separate. "
                            "Never invent measurements, faults, causes, or "
                            "operating conditions. A temporal relationship does "
                            "not establish causation. A net change does not "
                            "prove a gradual trend. Always follow the requested "
                            "OUTPUT FORMAT exactly and keep the whole answer short."
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                explanation_text = response.choices[0].message.content
                st.markdown(
                    f'<div class="insight-card">{explanation_text}</div>',
                    unsafe_allow_html=True,
                )
            except Exception as e:
                st.error(f"AI explanation failed: {e}")

else:
    st.info("No unusual periods in the current selection to investigate.")

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
    "confirmed physical cause."
)
