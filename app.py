import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Inverter Anomaly Detection",
    page_icon="⚡",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_parquet("dashboard_data.parquet")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)

    if "anomaly_flag" in df.columns:
        df["anomaly_flag"] = df["anomaly_flag"].astype(bool)

    return df


df = load_data()
def get_trend_window(df, end_time, hours_back=24):
    start_time = pd.to_datetime(end_time) - pd.Timedelta(hours=hours_back)

    window = df[
        (df["timestamp"] >= start_time) &
        (df["timestamp"] <= end_time)
    ].copy()

    return window

# ============================================================
# TITLE
# ============================================================

st.title("Inverter Anomaly Detection Dashboard")

st.caption(
    "Autoencoder-based anomaly detection with reconstruction-error thresholding"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Filters")

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

selected_dates = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

show_anomalies_only = st.sidebar.checkbox(
    "Show anomalies only",
    value=False
)


# ============================================================
# FILTER DATA
# ============================================================

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:

    start_date, end_date = selected_dates

    filtered_df = df[
        (df["timestamp"].dt.date >= start_date) &
        (df["timestamp"].dt.date <= end_date)
    ].copy()

else:
    filtered_df = df.copy()


if show_anomalies_only:
    filtered_df = filtered_df[
        filtered_df["anomaly_flag"]
    ].copy()


# ============================================================
# KPI VALUES
# ============================================================

total_observations = len(filtered_df)

total_anomalies = int(
    filtered_df["anomaly_flag"].sum()
)

anomaly_rate = (
    total_anomalies / total_observations * 100
    if total_observations > 0
    else 0
)

if "reconstruction_error" in filtered_df.columns and len(filtered_df) > 0:
    max_error = filtered_df["reconstruction_error"].max()
else:
    max_error = 0

if "inverter_temperature_c" in filtered_df.columns and len(filtered_df) > 0:
    max_temperature = filtered_df["inverter_temperature_c"].max()
else:
    max_temperature = 0


# ============================================================
# KPI CARDS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Observations",
        f"{total_observations:,}"
    )

with col2:
    st.metric(
        "Anomalous Observations",
        f"{total_anomalies:,}"
    )

with col3:
    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )

with col4:
    st.metric(
        "Maximum Temperature",
        f"{max_temperature:.1f} °C"
    )


st.divider()


# ============================================================
# ANOMALY TIMELINE
# ============================================================

st.subheader("Anomaly Timeline")

if len(filtered_df) > 0:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered_df["timestamp"],
            y=filtered_df["reconstruction_error"],
            mode="lines",
            name="Reconstruction Error"
        )
    )

    if "anomaly_score_ratio" in filtered_df.columns:

        anomaly_points = filtered_df[
            filtered_df["anomaly_flag"]
        ]

        fig.add_trace(
            go.Scatter(
                x=anomaly_points["timestamp"],
                y=anomaly_points["reconstruction_error"],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    size=8,
                    symbol="circle"
                )
            )
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Reconstruction Error",
        hovermode="x unified",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

else:
    st.info("No data available for the selected period.")


# ============================================================
# POWER ANALYSIS
# ============================================================

st.subheader("Power Analysis")

if len(filtered_df) > 0:

    fig = go.Figure()

    if "dc_power_kw" in filtered_df.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"],
                y=filtered_df["dc_power_kw"],
                mode="lines",
                name="DC Power"
            )
        )

    if "ac_power_kw" in filtered_df.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"],
                y=filtered_df["ac_power_kw"],
                mode="lines",
                name="AC Power"
            )
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Power (kW)",
        hovermode="x unified",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TEMPERATURE ANALYSIS
# ============================================================

st.subheader("Thermal Analysis")

if len(filtered_df) > 0:

    fig = go.Figure()

    if "inverter_temperature_c" in filtered_df.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"],
                y=filtered_df["inverter_temperature_c"],
                mode="lines",
                name="Inverter Temperature"
            )
        )

    if "ambient_temperature_c" in filtered_df.columns:
        fig.add_trace(
            go.Scatter(
                x=filtered_df["timestamp"],
                y=filtered_df["ambient_temperature_c"],
                mode="lines",
                name="Ambient Temperature"
            )
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# TEMPERATURE DELTA
# ============================================================

if "inverter_ambient_temp_delta" in filtered_df.columns:

    st.subheader("Inverter-to-Ambient Temperature Difference")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered_df["timestamp"],
            y=filtered_df["inverter_ambient_temp_delta"],
            mode="lines",
            name="Temperature Delta"
        )
    )

    anomaly_points = filtered_df[
        filtered_df["anomaly_flag"]
    ]

    if len(anomaly_points) > 0:

        fig.add_trace(
            go.Scatter(
                x=anomaly_points["timestamp"],
                y=anomaly_points["inverter_ambient_temp_delta"],
                mode="markers",
                name="Anomaly",
                marker=dict(
                    size=8
                )
            )
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Temperature Difference (°C)",
        hovermode="x unified",
        height=400
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# ANOMALY EVENTS
# ============================================================

st.subheader("Detected Anomalies")

anomaly_df = filtered_df[
    filtered_df["anomaly_flag"]
].copy()


if len(anomaly_df) > 0:

    display_columns = [
        "timestamp",
        "reconstruction_error",
        "anomaly_score_ratio",
        "inverter_temperature_c",
        "ac_power_kw",
        "dc_power_kw"
    ]

    display_columns = [
        c for c in display_columns
        if c in anomaly_df.columns
    ]

    st.dataframe(
        anomaly_df[display_columns],
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No anomalies detected in the selected period."
    )


# ============================================================
# SELECT ANOMALY
# ============================================================

st.subheader("Anomaly Investigation")

if len(anomaly_df) > 0:

    anomaly_df = anomaly_df.reset_index(drop=True)

    selected_index = st.selectbox(
        "Select an anomaly observation",
        range(len(anomaly_df)),
        format_func=lambda x: (
            anomaly_df.loc[x, "timestamp"].strftime(
                "%Y-%m-%d %H:%M"
            )
        )
    )

    selected_anomaly = anomaly_df.loc[selected_index]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Time",
            selected_anomaly["timestamp"].strftime(
                "%Y-%m-%d %H:%M"
            )
        )

    with col2:
        st.metric(
            "Reconstruction Error",
            f"{selected_anomaly['reconstruction_error']:.4f}"
        )

    with col3:

        if "anomaly_score_ratio" in selected_anomaly:
            st.metric(
                "Score / Threshold",
                f"{selected_anomaly['anomaly_score_ratio']:.2f}×"
            )

    with col4:

        if "inverter_temperature_c" in selected_anomaly:
            st.metric(
                "Temperature",
                f"{selected_anomaly['inverter_temperature_c']:.1f} °C"
            )

# ============================================================
# 24-HOUR PRE-ANOMALY TREND
# ============================================================

if len(anomaly_df) > 0:

    st.subheader("24-Hour Trend Before Selected Anomaly")

    end_time = selected_anomaly["timestamp"]

    trend_window = get_trend_window(
        filtered_df,
        end_time,
        hours_back=24
    )

    if len(trend_window) > 0:

        # ----------------------------------------
        # AC POWER
        # ----------------------------------------

        if "ac_power_kw" in trend_window.columns:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=trend_window["timestamp"],
                    y=trend_window["ac_power_kw"],
                    mode="lines",
                    name="AC Power"
                )
            )

            fig.update_layout(
                title="AC Power — 24 Hours Before Anomaly",
                xaxis_title="Time",
                yaxis_title="AC Power (kW)",
                hovermode="x unified",
                height=350
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------
        # DC CURRENT
        # ----------------------------------------

        if "dc_current_a" in trend_window.columns:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=trend_window["timestamp"],
                    y=trend_window["dc_current_a"],
                    mode="lines",
                    name="DC Current"
                )
            )

            fig.update_layout(
                title="DC Current — 24 Hours Before Anomaly",
                xaxis_title="Time",
                yaxis_title="DC Current (A)",
                hovermode="x unified",
                height=350
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


        # ----------------------------------------
        # INVERTER TEMPERATURE
        # ----------------------------------------

        if "inverter_temperature_c" in trend_window.columns:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=trend_window["timestamp"],
                    y=trend_window["inverter_temperature_c"],
                    mode="lines",
                    name="Inverter Temperature"
                )
            )

            fig.update_layout(
                title="Inverter Temperature — 24 Hours Before Anomaly",
                xaxis_title="Time",
                yaxis_title="Temperature (°C)",
                hovermode="x unified",
                height=350
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:
        st.info(
            "No trend data available for the selected anomaly."
        )
# ============================================================
# RAW DATA
# ============================================================

with st.expander("View filtered data"):

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Anomaly detection results are generated from the trained "
    "Autoencoder and established anomaly threshold."
)
