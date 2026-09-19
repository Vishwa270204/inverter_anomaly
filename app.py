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
import os
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

# Light, minimal styling -- no dark/neon theme, no animation.
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


st.session_state.setdefault("last_ai_explanation", None)
st.session_state.setdefault("last_ai_evidence", None)

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
    numeric = [c for c in ["ac_power_kw", "dc_power_kw", "dc_current_a", "ac_current_a", "inverter_temperature_c", "ambient_temperature_c", "inverter_ambient_temp_delta", "efficiency_pct", "power_factor"] if c in source.columns]
    stats = {}
    for col in numeric:
        series = pd.to_numeric(source[col], errors="coerce").dropna()
        if len(series) >= 2:
            stats[col] = {"start": clean_value(series.iloc[0]), "end": clean_value(series.iloc[-1]), "net_change": clean_value(series.iloc[-1] - series.iloc[0]), "min": clean_value(series.min()), "max": clean_value(series.max()), "median": clean_value(series.median())}
    return {"window_start": clean_value(source["timestamp"].min()), "window_end": clean_value(source["timestamp"].max()), "observations": int(len(source)), "statistics": stats}


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
    values = [{"feature": c.replace("_contribution_pct", ""), "contribution_pct": clean_value(row.get(c))} for c in cols if pd.notna(row.get(c))]
    values.sort(key=lambda x: x["contribution_pct"] if x["contribution_pct"] is not None else -1, reverse=True)
    return {"timestamp": clean_value(row.get("timestamp")), "top_contributing_feature": clean_value(row.get("top_contributing_feature")), "contributions": values}


def get_operating_context(timestamp, inverter_id=None):
    target = pd.to_datetime(timestamp)
    source = df.copy()
    if inverter_id is not None and "inverter_id" in source.columns:
        source = source[source["inverter_id"].astype(str) == str(inverter_id)]
    if source.empty:
        return {"error": "No matching data found."}
    idx = (source["timestamp"] - target).abs().idxmin()
    row = source.loc[idx]
    wanted = ["timestamp", "inverter_id", "hour", "minute", "month", "is_daylight", "inverter_status", "dc_power_kw", "ac_power_kw", "dc_current_a", "ac_current_a", "power_factor", "frequency_hz", "efficiency_pct", "inverter_temperature_c", "ambient_temperature_c", "poa_w_m2", "ghi_w_m2", "quality_code_inv", "communication_status_inv"]
    return {k: clean_value(row.get(k)) for k in wanted if k in source.columns}


AI_TOOLS = [
    {"type":"function","function":{"name":"get_anomaly_details","description":"Retrieve the selected anomaly observation and model outputs.","parameters":{"type":"object","properties":{"timestamp":{"type":"string"},"inverter_id":{"type":"string"}},"required":["timestamp"],"additionalProperties":False}}},
    {"type":"function","function":{"name":"get_pre_anomaly_trend","description":"Retrieve descriptive statistics for the 24 hours before the selected anomaly.","parameters":{"type":"object","properties":{"timestamp":{"type":"string"},"inverter_id":{"type":"string"},"hours":{"type":"number"}},"required":["timestamp"],"additionalProperties":False}}},
    {"type":"function","function":{"name":"get_feature_contributions","description":"Retrieve feature contributions to reconstruction error for the selected anomaly.","parameters":{"type":"object","properties":{"timestamp":{"type":"string"},"inverter_id":{"type":"string"}},"required":["timestamp"],"additionalProperties":False}}},
    {"type":"function","function":{"name":"get_operating_context","description":"Retrieve daylight, status, power, environmental, and communication context at the selected anomaly.","parameters":{"type":"object","properties":{"timestamp":{"type":"string"},"inverter_id":{"type":"string"}},"required":["timestamp"],"additionalProperties":False}}},
]

def execute_ai_tool(name, args):
    if name == "get_anomaly_details": return get_anomaly_details(**args)
    if name == "get_pre_anomaly_trend": return get_pre_anomaly_trend(**args)
    if name == "get_feature_contributions": return get_feature_contributions(**args)
    if name == "get_operating_context": return get_operating_context(**args)
    return {"error": f"Unknown tool: {name}"}


def generate_ai_explanation(selected_anomaly):
    client = get_groq_client()
    if client is None:
        raise RuntimeError("GROQ_API_KEY is not configured. Add [groq] api_key to Streamlit Cloud Secrets.")
    timestamp = clean_value(selected_anomaly.get("timestamp"))
    inverter_id = clean_value(selected_anomaly.get("inverter_id")) if "inverter_id" in selected_anomaly.index else None
    evidence = {}
    messages = [{"role":"system","content":"You are an industrial inverter anomaly explanation assistant. Use only evidence returned by the supplied tools and the selected anomaly context. The Autoencoder flags an unusual observation when reconstruction error exceeds the EVT/POT detection threshold. anomaly_score_ratio is reconstruction_error divided by threshold; it is NOT probability, confidence, severity, or failure probability. Feature contributions show what was hardest for the model to reconstruct; they are NOT proof of physical cause or root cause. Do not automatically call an anomaly a fault, failure, or breakdown. Account for daylight, time, operating status, and operating conditions. Describe pre-anomaly trends as observed changes, not causation. If evidence is insufficient, say so. Recommendations must be engineering checks/investigations, not confirmed diagnoses. Final response sections: What was detected; What the model found; What changed before the anomaly; What this evidence supports; What cannot be concluded; Recommended checks."},{"role":"user","content":json.dumps({"selected_timestamp":timestamp,"inverter_id":inverter_id,"selected_row":row_to_dict(selected_anomaly)},default=str)}]
    for _ in range(6):
        response = client.chat.completions.create(model="llama-3.3-70b-versatile",messages=messages,tools=AI_TOOLS,tool_choice="auto",temperature=0.2)
        msg = response.choices[0].message
        if not msg.tool_calls:
            return msg.content, evidence
        messages.append({"role":"assistant","content":msg.content or "","tool_calls":[{"id":tc.id,"type":"function","function":{"name":tc.function.name,"arguments":tc.function.arguments}} for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            try: args=json.loads(tc.function.arguments)
            except Exception: args={}
            result=execute_ai_tool(tc.function.name,args)
            evidence[tc.function.name]=result
            messages.append({"role":"tool","tool_call_id":tc.id,"content":json.dumps(result,default=str)})
    raise RuntimeError("AI investigation reached the maximum tool-call steps without producing an explanation.")


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


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        background-color:#0F3554;
        padding:1.6rem 1.8rem;
        border-radius:8px;
        margin-bottom:1.4rem;
    ">
        <div style="color:#FFFFFF; font-size:1.9rem; font-weight:600; line-height:1.2;">
            ⚡ Inverter Anomaly Detection Dashboard
        </div>
        <div style="color:#CBD9E5; font-size:0.95rem; margin-top:0.3rem;">
            Autoencoder-based anomaly detection with EVT/POT reconstruction-error thresholding
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

st.divider()


# ============================================================
# ANOMALY TIMELINE
# ============================================================

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
        yaxis_title="Reconstruction Error",
        yaxis_type="log",
        hovermode="x unified",
        height=420,
        template="plotly_white",
        margin=dict(t=20),
    )
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No reconstruction-error data available for the selected period.")

st.divider()


# ============================================================
# POWER ANALYSIS
# ============================================================

st.subheader("Power Analysis")

power_col1, power_col2 = st.columns(2)

with power_col1:
    if total_observations > 0 and "dc_power_kw" in filtered_df.columns:
        fig_dc = go.Figure()
        fig_dc.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"], y=filtered_df["dc_power_kw"],
                mode="lines", name="DC Power", line=dict(color="#4C78A8"),
            )
        )
        fig_dc.update_layout(
            title="DC Power", xaxis_title="Time", yaxis_title="DC Power (kW)",
            hovermode="x unified", height=380, template="plotly_white",
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
                mode="lines", name="AC Power", line=dict(color="#72B7B2"),
            )
        )
        fig_ac.update_layout(
            title="AC Power", xaxis_title="Time", yaxis_title="AC Power (kW)",
            hovermode="x unified", height=380, template="plotly_white",
        )
        st.plotly_chart(fig_ac, width="stretch")
    else:
        st.info("AC power data not available.")

st.divider()


# ============================================================
# THERMAL ANALYSIS
# ============================================================

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
                    mode="lines", name="Inverter Temp (°C)", line=dict(color="#E45756"),
                )
            )
        if has_ambient:
            fig_temp.add_trace(
                go.Scatter(
                    x=filtered_df["timestamp"], y=filtered_df["ambient_temperature_c"],
                    mode="lines", name="Ambient Temp (°C)", line=dict(color="#F58518"),
                )
            )
        fig_temp.update_layout(
            title="Inverter vs. Ambient Temperature", xaxis_title="Time",
            yaxis_title="Temperature (°C)", hovermode="x unified",
            height=380, template="plotly_white",
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
                    mode="markers", name="Flagged anomaly", marker=dict(size=7, color="#E45756"),
                )
            )
        fig_delta.update_layout(
            title="Inverter-to-Ambient Temperature Difference", xaxis_title="Time",
            yaxis_title="Temperature Difference (°C)", hovermode="x unified",
            height=380, template="plotly_white",
        )
        st.plotly_chart(fig_delta, width="stretch")
    else:
        st.info("Inverter-to-ambient temperature delta not available.")

st.divider()


# ============================================================
# DETECTED ANOMALIES TABLE
# ============================================================

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


# ============================================================
# ANOMALY INVESTIGATION
# ============================================================

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
                    mode="lines", name="AC Power (kW)", yaxis="y", line=dict(color="#4C78A8"),
                )
            )
        if "dc_current_a" in trend_window.columns:
            fig_trend.add_trace(
                go.Scatter(
                    x=trend_window["timestamp"], y=trend_window["dc_current_a"],
                    mode="lines", name="DC Current (A)", yaxis="y2", line=dict(color="#F58518"),
                )
            )
        if "inverter_temperature_c" in trend_window.columns:
            fig_trend.add_trace(
                go.Scatter(
                    x=trend_window["timestamp"], y=trend_window["inverter_temperature_c"],
                    mode="lines", name="Inverter Temperature (°C)", yaxis="y3",
                    line=dict(color="#E45756"),
                )
            )

        fig_trend.add_vline(x=end_time, line_dash="dash", line_width=2, line_color="#B10318")
        fig_trend.add_annotation(
            x=end_time, y=1.06, xref="x", yref="paper", showarrow=False,
            text="Selected anomaly", font=dict(color="#B10318", size=12),
        )

        fig_trend.update_layout(
            title="AC Power, DC Current & Temperature — 24 Hours Before Anomaly",
            xaxis=dict(title="Time"),
            yaxis=dict(title="AC Power (kW)", side="left"),
            yaxis2=dict(title="DC Current (A)", overlaying="y", side="right"),
            yaxis3=dict(title="Temperature (°C)", overlaying="y", side="right", position=0.94),
            hovermode="x unified",
            height=520,
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
            margin=dict(l=70, r=110, t=90, b=60),
        )
        st.plotly_chart(fig_trend, width="stretch")
        st.caption(
            f"Trend window: {fmt_time(trend_window['timestamp'].min())} → "
            f"{fmt_time(trend_window['timestamp'].max())} "
            f"({len(trend_window):,} observations). This shows the historical "
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
                    marker_color="#4C78A8",
                )
            )
            fig_contrib.update_layout(
                title="Features contributing to reconstruction error",
                xaxis_title="Contribution to reconstruction error (%)",
                yaxis_title="",
                height=280,
                template="plotly_white",
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
    # AI EXPLANATION
    # ========================================================
    st.markdown("#### AI Explanation")
    st.caption(
        "AI explanation based on the selected anomaly and the evidence "
        "retrieved from the model output."
    )

    # Keep the explanation and Generate button aligned in one row.
    ai_text_col, ai_button_col = st.columns([5, 1])

    with ai_text_col:
        st.markdown(
            """
            <div style="
                background:#F7F9FC;
                border:1px solid #D9E1EA;
                border-radius:8px;
                padding:14px 18px;
                min-height:72px;
                display:flex;
                align-items:center;
            ">
                <span style="
                    color:#64748B;
                    font-size:0.92rem;
                ">
                    Generate an AI explanation to see why the anomaly was
                    flagged, when it occurred, and what should be checked.
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with ai_button_col:
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        generate_ai = st.button(
            "Generate",
            type="primary",
            use_container_width=True,
            help="Generate a concise explanation from the selected anomaly evidence.",
        )

    if generate_ai:
        with st.spinner("Generating explanation..."):
            try:
                ai_explanation, ai_evidence = generate_ai_explanation(selected_anomaly)

                st.session_state["last_ai_explanation"] = ai_explanation
                st.session_state["last_ai_evidence"] = ai_evidence

            except Exception as e:
                st.error(f"AI explanation failed: {e}")

    # --------------------------------------------------------
    # Display the latest AI explanation
    # --------------------------------------------------------
    if st.session_state.get("last_ai_explanation"):

        st.markdown(
            """
            <div style="
                margin-top:12px;
                border:1px solid #D9E1EA;
                border-radius:8px;
                background:#FFFFFF;
                padding:16px 18px;
            ">
            """,
            unsafe_allow_html=True,
        )

        st.markdown(st.session_state["last_ai_explanation"])

        st.markdown("</div>", unsafe_allow_html=True)

    # Keep raw evidence available for debugging, but hidden by default.
    if st.session_state.get("last_ai_evidence"):
        with st.expander("View AI evidence"):
            st.json(st.session_state["last_ai_evidence"])


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
    "confirmed physical cause."
)
