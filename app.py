"""
Inverter Anomaly Detection -- Streamlit Dashboard
==================================================

Production frontend only. All ML/training happens in inverter_anomaly.ipynb.

Reads:
    dashboard_data.parquet
        Evaluation-period observations + ML anomaly output.

    trend_data.parquet
        Optional historical data used for the 7-day healthy baseline.

IMPORTANT:
    dashboard_baseline.parquet is NOT used.

The healthy baseline is created dynamically from the data:
    - previous 7 days
    - healthy observations only
    - operating-condition aware
    - P1/P5/P25/P50/P75/P95/P99

The AI generates ONE overall explanation from ALL detected anomaly
episodes in the selected period.

It does NOT generate one explanation per event.

The LLM does not determine whether a physical fault occurred.
It only explains statistical evidence supplied by Python.
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

from data_validation import validate_event_data


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Inverter Anomaly Detection",
    page_icon="⚡",
    layout="wide",
)


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    html {
        font-size: 16px;
    }

    :root {
        --primary: #0F3554;
        --border: #E4E9F0;
        --muted: #64748B;
        --surface: #FFFFFF;
        --bg: #F5F7FA;
    }

    .stApp {
        background-color: var(--bg);
    }

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

    [data-testid="stVerticalBlock"] {
        gap: 0.35rem;
    }

    div[data-testid="stElementContainer"] {
        margin-bottom: 0 !important;
    }

    [data-testid="stDateInput"] input,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        min-height: 2rem !important;
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
    }

    [data-testid="stWidgetLabel"] p {
        margin-bottom: 0.1rem !important;
    }

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

    h2,
    [data-testid="stMarkdownContainer"] h2 {
        color: #0F172A;
        font-size: 1.55rem !important;
        font-weight: 700;
        margin-top: 1.1rem !important;
        margin-bottom: 0.35rem !important;
        padding-bottom: 0.35rem;
        border-bottom: 1px solid var(--border);
    }

    h3,
    [data-testid="stMarkdownContainer"] h3 {
        color: #0F172A;
        font-size: 1.22rem !important;
        font-weight: 700;
        margin-top: 0.7rem !important;
        margin-bottom: 0.25rem !important;
    }

    label,
    .stSelectbox label,
    .stDateInput label {
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        color: #334155 !important;
    }

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

    /* ========================================================
       AI
       ======================================================== */

    .ai-title {
        color: #0F172A;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.3;
        margin-top: 0.25rem;
        margin-bottom: 0.15rem;
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

    .ai-alert-box {
        display: flex;
        gap: 0.65rem;
        background: #FEF3F2;
        border: 1px solid #FECDCA;
        border-radius: 10px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
    }

    .ai-alert-icon {
        font-size: 1.3rem;
        line-height: 1.4;
        flex-shrink: 0;
    }

    .ai-alert-title {
        font-weight: 700;
        color: #B42318;
        font-size: 1.02rem;
        margin-bottom: 0.15rem;
    }

    .ai-alert-desc {
        color: #7A271A;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .ai-section {
        margin-top: 1.1rem;
    }

    .ai-section-header {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        font-weight: 700;
        color: #0F172A;
        font-size: 1rem;
        margin-bottom: 0.45rem;
    }

    .ai-bullets {
        margin: 0;
        padding-left: 1.3rem;
    }

    .ai-bullets li {
        color: #1E293B;
        font-size: 0.96rem;
        line-height: 1.55;
        margin-bottom: 0.3rem;
    }

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

    /* ========================================================
       BASELINE
       ======================================================== */

    .baseline-card {
        background: #FFFFFF;
        border: 1px solid #D9E1EA;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-top: 0.5rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }

    .baseline-title {
        color: #0F172A;
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .baseline-subtitle {
        color: #64748B;
        font-size: 0.82rem;
        line-height: 1.45;
        margin-bottom: 0.75rem;
    }

    /* ========================================================
       VALIDATION
       ======================================================== */

    .validation-card {
        background: #FFFFFF;
        border: 1px solid #D9E1EA;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin-top: 0.75rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
    }

    .validation-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.3rem;
    }

    .validation-title {
        color: #0F172A;
        font-size: 1.15rem;
        font-weight: 700;
    }

    .validation-subtitle {
        color: #64748B;
        font-size: 0.82rem;
        line-height: 1.45;
        margin-bottom: 0.75rem;
    }

    .validation-pill {
        border-radius: 999px;
        padding: 0.3rem 0.65rem;
        font-size: 0.78rem;
        font-weight: 700;
        white-space: nowrap;
    }

    .validation-pass {
        background: #ECFDF3;
        color: #027A48;
        border: 1px solid #ABEFC6;
    }

    .validation-warn {
        background: #FFFAEB;
        color: #B54708;
        border: 1px solid #FEDF89;
    }

    .validation-fail {
        background: #FEF3F2;
        color: #B42318;
        border: 1px solid #FECDCA;
    }

    .check-row {
        display: flex;
        align-items: flex-start;
        gap: 0.65rem;
        padding: 0.55rem 0;
        border-top: 1px solid #EEF2F6;
    }

    .check-icon {
        width: 1.25rem;
        height: 1.25rem;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 1.25rem;
        font-size: 0.72rem;
        font-weight: 700;
    }

    .check-pass {
        background: #D1FADF;
        color: #027A48;
    }

    .check-warn {
        background: #FEF0C7;
        color: #B54708;
    }

    .check-fail {
        background: #FEE4E2;
        color: #B42318;
    }

    .check-name {
        color: #1E293B;
        font-size: 0.88rem;
        font-weight: 600;
        line-height: 1.35;
    }

    .check-detail {
        color: #64748B;
        font-size: 0.76rem;
        line-height: 1.35;
        margin-top: 0.12rem;
    }

    .validation-note {
        margin-top: 0.65rem;
        padding: 0.65rem 0.75rem;
        border-radius: 8px;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        color: #475569;
        font-size: 0.78rem;
        line-height: 1.45;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault("ai_overall_explanation", None)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dashboard_data(path="dashboard_data.parquet"):
    data = pd.read_parquet(path)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    data = (
        data
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if "anomaly_flag" in data.columns:
        data["anomaly_flag"] = (
            data["anomaly_flag"]
            .fillna(False)
            .astype(bool)
        )
    else:
        data["anomaly_flag"] = False

    return data


@st.cache_data
def load_trend_data(path="trend_data.parquet"):
    data = pd.read_parquet(path)

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce",
    )

    return (
        data
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def safe_load(loader, path, label):
    try:
        return loader(path)

    except FileNotFoundError:
        st.error(
            f"**{label} not found** (`{path}`). "
            "Run the required notebook section first."
        )
        st.stop()

    except Exception as exc:
        st.error(
            f"Failed to load **{label}** (`{path}`): {exc}"
        )
        st.stop()


df = safe_load(
    load_dashboard_data,
    "dashboard_data.parquet",
    "Dashboard data",
)

if df.empty:
    st.error("`dashboard_data.parquet` contains no rows.")
    st.stop()


# trend data is optional
try:
    trend_df = load_trend_data("trend_data.parquet")
    has_trend_data = True

except Exception:
    trend_df = df.copy()
    has_trend_data = False


# ============================================================
# HELPERS
# ============================================================

def clean_value(value):
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def fmt_num(value, decimals=2, suffix="", dash="—"):
    if value is None:
        return dash

    try:
        if pd.isna(value):
            return dash
    except Exception:
        pass

    return f"{float(value):,.{decimals}f}{suffix}"


def fmt_time(value, dash="—"):
    if value is None:
        return dash

    try:
        if pd.isna(value):
            return dash
    except Exception:
        pass

    return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")


def get_contribution_cols(frame):
    return [
        c for c in frame.columns
        if c.endswith("_contribution_pct")
    ]


# ============================================================
# GROQ
# ============================================================

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
# ANOMALY EVENTS
# ============================================================

def build_anomaly_events(anomaly_df, gap_minutes=6):

    if anomaly_df.empty:
        return pd.DataFrame()

    work = anomaly_df.copy()

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        errors="coerce",
    )

    work = (
        work
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if work.empty:
        return pd.DataFrame()

    gaps = (
        work["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
    )

    work["event_break"] = (
        gaps.isna()
        | (gaps > gap_minutes)
    )

    work["event_number"] = work["event_break"].cumsum()

    events = []

    for event_number, rows in work.groupby("event_number"):

        rows = rows.sort_values("timestamp")

        start = rows["timestamp"].min()
        end = rows["timestamp"].max()

        duration = (
            (end - start).total_seconds() / 60
        )

        event = {
            "event_id": f"Event {int(event_number)}",
            "start_time": start,
            "end_time": end,
            "duration_min": duration,
            "anomaly_count": len(rows),
        }

        if "anomaly_score_ratio" in rows.columns:
            score = pd.to_numeric(
                rows["anomaly_score_ratio"],
                errors="coerce",
            )

            event["max_severity"] = score.max()
            event["mean_severity"] = score.mean()

        if "reconstruction_error" in rows.columns:
            error = pd.to_numeric(
                rows["reconstruction_error"],
                errors="coerce",
            )

            event["max_reconstruction_error"] = error.max()
            event["mean_reconstruction_error"] = error.mean()

        if "inverter_status" in rows.columns:
            mode = rows["inverter_status"].mode()

            event["dominant_status"] = (
                mode.iloc[0]
                if len(mode)
                else None
            )

        if "anomaly_type" in rows.columns:
            mode = rows["anomaly_type"].mode()

            event["anomaly_type"] = (
                mode.iloc[0]
                if len(mode)
                else None
            )

        events.append(event)

    return pd.DataFrame(events).reset_index(drop=True)


# ============================================================
# HEALTHY BASELINE
# ============================================================

BASELINE_METRICS = [
    "dc_power_kw",
    "ac_power_kw",
    "dc_current_a",
    "ac_current_a",
    "power_factor",
    "frequency_hz",
    "efficiency_pct",
    "inverter_temperature_c",
    "ambient_temperature_c",
]


def get_healthy_reference_window(
    target_time,
    source,
    days=7,
):
    """
    Return healthy observations from the previous 7 days.

    Only observations before the anomaly are used.

    Healthy definition:
        anomaly_flag == False

    RUNNING observations are preferred when inverter_status exists.

    Daylight/night and hour are retained so the baseline can be
    condition-aware.
    """

    target_time = pd.to_datetime(target_time)

    start_time = target_time - timedelta(days=days)

    reference = source[
        (source["timestamp"] >= start_time)
        & (source["timestamp"] < target_time)
    ].copy()

    if reference.empty:
        return reference

    if "anomaly_flag" in reference.columns:
        reference = reference[
            ~reference["anomaly_flag"].fillna(False)
        ].copy()

    # Prefer normal RUNNING operation.
    if "inverter_status" in reference.columns:

        running = reference[
            reference["inverter_status"]
            .astype(str)
            .str.upper()
            .eq("RUNNING")
        ].copy()

        if len(running) > 0:
            reference = running

    return reference


def calculate_baseline(
    target_time,
    source,
    days=7,
):
    """
    Create the healthy baseline dynamically.

    Percentiles:
        P1
        P5
        P25
        P50
        P75
        P95
        P99

    The baseline is created from the previous 7 days of healthy
    observations.

    For condition-aware matching:
        - same daylight state
        - same hour ±1 hour

    If too few observations exist after matching, progressively
    relax the matching condition.
    """

    reference = get_healthy_reference_window(
        target_time,
        source,
        days=days,
    )

    result = {
        "reference_start": None,
        "reference_end": None,
        "healthy_sample_count": 0,
        "matching_method": None,
        "statistics": {},
    }

    if reference.empty:
        return result

    target_row = (
        source.iloc[
            (
                source["timestamp"] - pd.to_datetime(target_time)
            ).abs().argsort()[:1]
        ]
        .iloc[0]
    )

    target_hour = None
    target_daylight = None

    if "hour" in source.columns:
        target_hour = clean_value(
            target_row.get("hour")
        )

    if "is_daylight" in source.columns:
        target_daylight = clean_value(
            target_row.get("is_daylight")
        )

    # --------------------------------------------------------
    # CONDITION MATCHING
    # --------------------------------------------------------

    matched = reference.copy()

    # First attempt:
    # same daylight + hour ±1
    if (
        target_hour is not None
        and target_daylight is not None
        and "hour" in matched.columns
        and "is_daylight" in matched.columns
    ):

        hour_distance = (
            pd.to_numeric(
                matched["hour"],
                errors="coerce",
            )
            - float(target_hour)
        ).abs()

        condition_matched = matched[
            (hour_distance <= 1)
            & (
                matched["is_daylight"].astype(str)
                == str(target_daylight)
            )
        ].copy()

        if len(condition_matched) >= 20:
            matched = condition_matched
            result["matching_method"] = (
                "same daylight state and hour ±1"
            )

    # Second attempt:
    # same daylight
    if (
        result["matching_method"] is None
        and target_daylight is not None
        and "is_daylight" in matched.columns
    ):

        condition_matched = matched[
            matched["is_daylight"].astype(str)
            == str(target_daylight)
        ].copy()

        if len(condition_matched) >= 20:
            matched = condition_matched
            result["matching_method"] = (
                "same daylight state"
            )

    # Third fallback:
    # all healthy observations from previous 7 days
    if result["matching_method"] is None:
        result["matching_method"] = (
            "all healthy observations from previous 7 days"
        )

    result["reference_start"] = clean_value(
        matched["timestamp"].min()
    )

    result["reference_end"] = clean_value(
        matched["timestamp"].max()
    )

    result["healthy_sample_count"] = int(
        len(matched)
    )

    # --------------------------------------------------------
    # PERCENTILES
    # --------------------------------------------------------

    for metric in BASELINE_METRICS:

        if metric not in matched.columns:
            continue

        series = pd.to_numeric(
            matched[metric],
            errors="coerce",
        ).dropna()

        if len(series) < 5:
            continue

        result["statistics"][metric] = {
            "p1": clean_value(series.quantile(0.01)),
            "p5": clean_value(series.quantile(0.05)),
            "p25": clean_value(series.quantile(0.25)),
            "p50": clean_value(series.quantile(0.50)),
            "p75": clean_value(series.quantile(0.75)),
            "p95": clean_value(series.quantile(0.95)),
            "p99": clean_value(series.quantile(0.99)),
        }

    return result


def build_overall_baseline(
    anomaly_events,
    source,
):
    """
    Create healthy references for every anomaly event.

    The resulting baseline is NOT one static parquet file.

    Each anomaly event gets a 7-day healthy reference immediately
    preceding that event.

    The individual event baselines are then aggregated into an
    overall healthy reference.
    """

    if anomaly_events.empty:
        return {
            "event_baselines": [],
            "overall": {},
        }

    event_baselines = []

    for _, event in anomaly_events.iterrows():

        baseline = calculate_baseline(
            event["start_time"],
            source,
            days=7,
        )

        baseline["event_id"] = event["event_id"]
        baseline["event_start"] = clean_value(
            event["start_time"]
        )

        event_baselines.append(baseline)

    overall = {}

    # --------------------------------------------------------
    # Combine percentile information across event baselines.
    #
    # We use the median of the corresponding healthy percentile
    # values across events.
    # --------------------------------------------------------

    for metric in BASELINE_METRICS:

        values = {
            "p1": [],
            "p5": [],
            "p25": [],
            "p50": [],
            "p75": [],
            "p95": [],
            "p99": [],
        }

        for baseline in event_baselines:

            stats = baseline.get(
                "statistics",
                {},
            ).get(metric)

            if not stats:
                continue

            for percentile in values:
                value = stats.get(percentile)

                if value is not None:
                    values[percentile].append(
                        float(value)
                    )

        metric_result = {}

        for percentile, vals in values.items():

            if vals:
                metric_result[percentile] = float(
                    pd.Series(vals).median()
                )

        if metric_result:
            overall[metric] = metric_result

    return {
        "event_baselines": event_baselines,
        "overall": overall,
    }


# ============================================================
# ABNORMAL PERIOD STATISTICS
# ============================================================

def calculate_abnormal_statistics(
    anomaly_df,
):
    """
    Calculate statistics across ALL ML-detected abnormal
    observations in the selected period.
    """

    result = {}

    if anomaly_df.empty:
        return result

    for metric in BASELINE_METRICS:

        if metric not in anomaly_df.columns:
            continue

        series = pd.to_numeric(
            anomaly_df[metric],
            errors="coerce",
        ).dropna()

        if len(series) == 0:
            continue

        result[metric] = {
            "count": int(len(series)),
            "p1": clean_value(series.quantile(0.01)),
            "p5": clean_value(series.quantile(0.05)),
            "p25": clean_value(series.quantile(0.25)),
            "p50": clean_value(series.quantile(0.50)),
            "p75": clean_value(series.quantile(0.75)),
            "p95": clean_value(series.quantile(0.95)),
            "p99": clean_value(series.quantile(0.99)),
        }

    return result


# ============================================================
# COMPARE ABNORMAL VS HEALTHY
# ============================================================

def compare_against_baseline(
    abnormal_stats,
    healthy_stats,
):
    """
    Compare abnormal statistics against the healthy reference.

    Important:
        This identifies statistical deviation.
        It does NOT prove physical root cause.
    """

    result = {}

    for metric, abnormal in abnormal_stats.items():

        healthy = healthy_stats.get(metric)

        if not healthy:
            continue

        comparison = {}

        healthy_median = healthy.get("p50")
        abnormal_median = abnormal.get("p50")

        if (
            healthy_median is not None
            and abnormal_median is not None
            and healthy_median != 0
        ):

            comparison["median_change_pct"] = (
                (
                    abnormal_median
                    - healthy_median
                )
                / healthy_median
            ) * 100

        # ----------------------------------------------------
        # Outside healthy P5-P95
        # ----------------------------------------------------

        lower = healthy.get("p5")
        upper = healthy.get("p95")

        if (
            lower is not None
            and upper is not None
            and metric in abnormal_stats
        ):

            if metric in abnormal_stats:
                # This will be calculated from the actual
                # abnormal observations later.
                comparison["healthy_p5"] = lower
                comparison["healthy_p95"] = upper

        # ----------------------------------------------------
        # Outside healthy P1-P99
        # ----------------------------------------------------

        if healthy.get("p1") is not None:
            comparison["healthy_p1"] = healthy["p1"]

        if healthy.get("p99") is not None:
            comparison["healthy_p99"] = healthy["p99"]

        result[metric] = comparison

    return result


def calculate_outside_counts(
    anomaly_df,
    healthy_stats,
):
    """
    Count abnormal observations outside the healthy P5-P95
    and P1-P99 ranges.
    """

    result = {}

    if anomaly_df.empty:
        return result

    for metric, healthy in healthy_stats.items():

        if metric not in anomaly_df.columns:
            continue

        series = pd.to_numeric(
            anomaly_df[metric],
            errors="coerce",
        ).dropna()

        if len(series) == 0:
            continue

        item = {
            "observations": int(len(series)),
        }

        p5 = healthy.get("p5")
        p95 = healthy.get("p95")
        p1 = healthy.get("p1")
        p99 = healthy.get("p99")

        if p5 is not None and p95 is not None:

            outside = (
                (series < p5)
                | (series > p95)
            )

            item["outside_p5_p95_count"] = int(
                outside.sum()
            )

            item["outside_p5_p95_pct"] = (
                float(outside.mean() * 100)
            )

        if p1 is not None and p99 is not None:

            outside_extreme = (
                (series < p1)
                | (series > p99)
            )

            item["outside_p1_p99_count"] = int(
                outside_extreme.sum()
            )

            item["outside_p1_p99_pct"] = (
                float(outside_extreme.mean() * 100)
            )

        result[metric] = item

    return result


# ============================================================
# TOP DEVIATING VARIABLES
# ============================================================

def get_top_deviating_metrics(
    comparisons,
):
    """
    Rank variables by absolute median percentage change.

    This is only used to select which evidence is sent to the
    LLM. It is not presented as a causal ranking.
    """

    scored = []

    for metric, values in comparisons.items():

        change = values.get("median_change_pct")

        if change is None:
            continue

        scored.append(
            {
                "metric": metric,
                "median_change_pct": change,
                "absolute_change_pct": abs(change),
            }
        )

    scored.sort(
        key=lambda x: x["absolute_change_pct"],
        reverse=True,
    )

    return scored[:5]


# ============================================================
# OVERALL EVIDENCE
# ============================================================

def build_overall_evidence(
    anomaly_df,
    events_df,
    source,
):
    """
    Build the compact statistical evidence package sent to the LLM.
    """

    if anomaly_df.empty:
        return {
            "status": "no_anomalies"
        }

    # --------------------------------------------------------
    # Healthy baseline
    # --------------------------------------------------------

    baseline_package = build_overall_baseline(
        events_df,
        source,
    )

    healthy_stats = baseline_package.get(
        "overall",
        {},
    )

    # --------------------------------------------------------
    # Abnormal statistics
    # --------------------------------------------------------

    abnormal_stats = calculate_abnormal_statistics(
        anomaly_df
    )

    # --------------------------------------------------------
    # Comparisons
    # --------------------------------------------------------

    comparisons = compare_against_baseline(
        abnormal_stats,
        healthy_stats,
    )

    # --------------------------------------------------------
    # Outside-range counts
    # --------------------------------------------------------

    outside_counts = calculate_outside_counts(
        anomaly_df,
        healthy_stats,
    )

    # --------------------------------------------------------
    # Top deviations
    # --------------------------------------------------------

    top_metrics = get_top_deviating_metrics(
        comparisons
    )

    # --------------------------------------------------------
    # Feature contributions
    # --------------------------------------------------------

    feature_contributions = []

    contribution_cols = get_contribution_cols(
        anomaly_df
    )

    if contribution_cols:

        contribution_values = (
            anomaly_df[contribution_cols]
            .apply(pd.to_numeric, errors="coerce")
            .mean()
            .dropna()
            .sort_values(ascending=False)
        )

        feature_contributions = [
            {
                "feature": col.replace(
                    "_contribution_pct",
                    "",
                ),
                "mean_contribution_pct": clean_value(
                    value
                ),
            }
            for col, value
            in contribution_values.head(5).items()
        ]

    # --------------------------------------------------------
    # Event summary
    # --------------------------------------------------------

    total_duration = 0

    if not events_df.empty:
        total_duration = float(
            events_df["duration_min"]
            .fillna(0)
            .sum()
        )

    event_types = []

    if "anomaly_type" in events_df.columns:

        event_types = (
            events_df["anomaly_type"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )

    statuses = []

    if "dominant_status" in events_df.columns:

        statuses = (
            events_df["dominant_status"]
            .dropna()
            .astype(str)
            .value_counts()
            .to_dict()
        )

    return {
        "analysis": {
            "anomaly_observations": int(
                len(anomaly_df)
            ),
            "anomaly_events": int(
                len(events_df)
            ),
            "total_anomaly_duration_min": total_duration,
            "event_types": event_types,
            "statuses": statuses,
        },

        "healthy_baseline": {
            "method": (
                "Previous 7 days of healthy observations "
                "for each anomaly event"
            ),
            "percentiles": [
                "P1",
                "P5",
                "P25",
                "P50",
                "P75",
                "P95",
                "P99",
            ],
            "overall_statistics": healthy_stats,
            "event_baseline_count": len(
                baseline_package["event_baselines"]
            ),
        },

        "abnormal_statistics": abnormal_stats,

        "comparison": comparisons,

        "outside_healthy_range": outside_counts,

        "top_deviating_metrics": top_metrics,

        "feature_contributions": feature_contributions,
    }


# ============================================================
# AI GENERATION
# ============================================================

def generate_overall_ai_explanation(
    evidence,
):
    """
    Generate ONE overall anomaly explanation from the complete
    statistical evidence package.
    """

    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Set GROQ_API_KEY in the environment or Streamlit secrets."
        )

    prompt = """
You are an explanation assistant inside a solar inverter anomaly
detection dashboard.

Your task is to explain ALL detected anomaly behavior during the
selected analysis period as ONE overall explanation.

Do NOT explain individual anomaly events separately.

Return ONLY one valid JSON object.

Required JSON:

{
  "headline": "One short sentence describing the overall abnormal behavior.",
  "summary": "2-3 simple sentences explaining what changed compared with healthy behavior.",
  "why_it_happened": [
    "Evidence-based point 1",
    "Evidence-based point 2",
    "Evidence-based point 3"
  ],
  "recommended_actions": [
    "Practical action 1",
    "Practical action 2",
    "Practical action 3"
  ]
}

RULES:

1. Use ONLY the supplied evidence.

2. Do not invent values.

3. Do not explain individual events one by one.

4. Explain the overall pattern across all anomaly observations.

5. Compare abnormal behavior with the healthy 7-day baseline.

6. Use the healthy P1/P5/P25/P50/P75/P95/P99 values supplied.

7. For low-power behavior, pay attention to the lower healthy percentiles
   such as P5 and P1. Do not use P95/P99 alone.

8. If an abnormal median is below the healthy median, describe the
   downward shift.

9. If an abnormal median is above the healthy median, describe the
   upward shift.

10. If abnormal observations are outside healthy P5-P95, this is
    statistical evidence of unusual behavior.

11. If observations are outside P1-P99, describe this as stronger
    statistical evidence of unusually extreme behavior.

12. Never say that crossing P95/P99 proves a physical fault.

13. Never claim that a feature contribution proves root cause.

14. Do not automatically call the anomaly a fault.

15. If the physical cause cannot be established from the supplied data,
    explicitly say that the data shows unusual behavior but does not
    establish the physical root cause.

16. Use simple language suitable for a plant operator.

17. Do not use technical ML terms such as:
    autoencoder,
    reconstruction error,
    anomaly score,
    probability,
    confidence,
    latent space.

18. Recommendations must be based only on available evidence.

19. Do not invent maintenance findings.

20. Keep all text plain. Do not use markdown bold or asterisks.

EVIDENCE:

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
            reasoning_effort="low",
            include_reasoning=False,
            temperature=0.2,
            max_completion_tokens=768,
            response_format={
                "type": "json_object"
            },
        )

    except Exception as exc:
        raise RuntimeError(
            f"Groq request failed: {exc}"
        ) from exc

    content = getattr(
        response.choices[0].message,
        "content",
        None,
    )

    if not content:
        raise RuntimeError(
            "Groq returned no explanation."
        )

    content = str(content).strip()

    content = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        content,
    ).strip()

    try:
        explanation = json.loads(content)

    except Exception as exc:
        raise RuntimeError(
            f"Groq returned invalid JSON: {exc}"
        ) from exc

    if not isinstance(explanation, dict):
        raise RuntimeError(
            "Groq response was not a JSON object."
        )

    required = [
        "headline",
        "summary",
        "why_it_happened",
        "recommended_actions",
    ]

    missing = [
        field
        for field in required
        if field not in explanation
    ]

    if missing:
        raise RuntimeError(
            f"Groq JSON is missing fields: {missing}"
        )

    return explanation


# ============================================================
# AI RENDERER
# ============================================================

def render_overall_ai_explanation(
    explanation,
):
    if not explanation:
        return

    def esc(value):
        return html.escape(
            str(value)
        )

    headline = esc(
        explanation.get(
            "headline",
            "",
        )
    )

    summary = esc(
        explanation.get(
            "summary",
            "",
        )
    )

    why = [
        esc(x)
        for x in explanation.get(
            "why_it_happened",
            [],
        )
        if x
    ]

    actions = [
        esc(x)
        for x in explanation.get(
            "recommended_actions",
            [],
        )
        if x
    ]

    why_html = "".join(
        f"<li>{item}</li>"
        for item in why
    )

    action_html = "".join(
        f"<li>{item}</li>"
        for item in actions
    )

    html_block = f"""
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
<span>🔍</span>
Why this pattern was detected
</div>
<ul class="ai-bullets">
{why_html}
</ul>
</div>

<div class="ai-section">
<div class="ai-section-header">
<span>🔧</span>
Recommended action
</div>
<ul class="ai-bullets">
{action_html}
</ul>
</div>

<div class="ai-footer-note">
<span>ℹ️</span>
<span>
This explanation summarizes statistical patterns observed across
the detected anomaly period and compares them with healthy historical
behavior. It does not confirm a physical fault or root cause.
</span>
</div>

</div>
"""

    st.markdown(
        html_block,
        unsafe_allow_html=True,
    )


# ============================================================
# VALIDATION OF OVERALL AI
# ============================================================

def validate_overall_ai(
    explanation,
    evidence,
):
    """
    Validate the overall explanation against the statistical evidence.

    Checks:
        1. Numeric claims
        2. Direction of major changes
        3. Healthy-range claims
        4. Unsupported threshold claims

    This checks evidence consistency only.
    """

    checks = []

    def add(
        name,
        status,
        detail,
    ):
        checks.append(
            {
                "name": name,
                "status": status,
                "detail": detail,
            }
        )

    text_parts = [
        str(
            explanation.get(
                "headline",
                "",
            )
        ),
        str(
            explanation.get(
                "summary",
                "",
            )
        ),
    ]

    text_parts += [
        str(x)
        for x in explanation.get(
            "why_it_happened",
            [],
        )
        if x
    ]

    text_parts += [
        str(x)
        for x in explanation.get(
            "recommended_actions",
            [],
        )
        if x
    ]

    ai_text = " ".join(
        text_parts
    ).lower()

    # ========================================================
    # 1. NUMERICAL CLAIMS
    # ========================================================

    known_numbers = []

    for section_name in [
        "healthy_baseline",
        "abnormal_statistics",
        "comparison",
        "outside_healthy_range",
    ]:

        section = evidence.get(
            section_name,
            {},
        )

        if isinstance(section, dict):

            def collect_numbers(obj):

                if isinstance(obj, dict):

                    for value in obj.values():
                        collect_numbers(value)

                elif isinstance(obj, list):

                    for value in obj:
                        collect_numbers(value)

                elif isinstance(obj, (int, float)):

                    if pd.notna(obj):
                        known_numbers.append(
                            float(obj)
                        )

            collect_numbers(section)

    measurement_pattern = re.compile(
        r"(?<![A-Za-z])"
        r"(-?\d+(?:\.\d+)?)"
        r"\s*"
        r"(kW|kw|A|a|°C|C|%|minutes?|mins?|min|Hz|h)"
        r"\b"
    )

    numeric_claims = (
        measurement_pattern.findall(
            " ".join(text_parts)
        )
    )

    unmatched = []

    for raw, unit in numeric_claims:

        value = float(raw)

        if not any(
            abs(value - known)
            <= max(
                2.0,
                abs(known) * 0.015,
            )
            for known in known_numbers
        ):

            unmatched.append(
                f"{raw} {unit}"
            )

    add(
        "Numerical values",
        "PASS" if not unmatched else "WARN",
        (
            "Reported measurements are consistent with supplied evidence."
            if not unmatched
            else
            "Unmatched numeric claims: "
            + ", ".join(
                unmatched[:5]
            )
        ),
    )

    # ========================================================
    # 2. POWER DIRECTION
    # ========================================================

    direction_errors = []

    for metric, label in [
        ("dc_power_kw", "DC power"),
        ("ac_power_kw", "AC power"),
    ]:

        comparison = evidence.get(
            "comparison",
            {},
        ).get(
            metric,
            {},
        )

        change = comparison.get(
            "median_change_pct"
        )

        if change is None:
            continue

        if change < -0.5:

            expected_words = [
                "decreased",
                "lower",
                "reduced",
                "dropped",
                "fell",
                "downward",
            ]

            expected = "decreased"

        elif change > 0.5:

            expected_words = [
                "increased",
                "higher",
                "rose",
                "grew",
                "upward",
            ]

            expected = "increased"

        else:

            expected_words = [
                "stable",
                "similar",
                "little change",
                "unchanged",
            ]

            expected = "roughly stable"

        if (
            label.lower() in ai_text
            and not any(
                word in ai_text
                for word in expected_words
            )
        ):

            direction_errors.append(
                f"{label} should be described as {expected}"
            )

    add(
        "Power change direction",
        "PASS"
        if not direction_errors
        else "WARN",
        (
            "The reported power direction is consistent with the "
            "statistical comparison."
            if not direction_errors
            else "; ".join(
                direction_errors
            )
        ),
    )

    # ========================================================
    # 3. HEALTHY RANGE CLAIMS
    # ========================================================

    range_words = [
        "outside healthy range",
        "outside normal range",
        "below healthy range",
        "above healthy range",
        "within healthy range",
        "outside the healthy range",
        "below normal",
        "above normal",
    ]

    has_range_claim = any(
        word in ai_text
        for word in range_words
    )

    if has_range_claim:

        comparable_metrics = 0

        for metric, values in evidence.get(
            "outside_healthy_range",
            {},
        ).items():

            if (
                values.get(
                    "outside_p5_p95_count"
                )
                is not None
            ):
                comparable_metrics += 1

        if comparable_metrics > 0:

            add(
                "Healthy-range claims",
                "PASS",
                "Healthy-range statements can be checked against the supplied percentile ranges.",
            )

        else:

            add(
                "Healthy-range claims",
                "WARN",
                "A healthy-range claim was made, but no matching percentile comparison was available.",
            )

    else:

        add(
            "Healthy-range claims",
            "PASS",
            "No unsupported healthy-range statement detected.",
        )

    # ========================================================
    # 4. CAUSALITY
    # ========================================================

    causal_words = [
        "caused by",
        "definitely caused",
        "root cause is",
        "confirmed fault",
        "proof of fault",
    ]

    unsupported_causal = [
        word
        for word in causal_words
        if word in ai_text
    ]

    add(
        "Causal claim",
        "WARN"
        if unsupported_causal
        else "PASS",
        (
            "Potentially unsupported causal wording detected: "
            + ", ".join(
                unsupported_causal
            )
            if unsupported_causal
            else
            "No unsupported definitive causal claim detected."
        ),
    )

    return checks


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div style="
background:#FFFFFF;
padding:0.95rem 1.15rem;
border:1px solid #D9E1EA;
border-radius:12px;
margin-bottom:0.9rem;
box-shadow:0 1px 4px rgba(15,23,42,0.04);
display:flex;
align-items:center;
gap:0.75rem;
">

<div style="
background:#EFF6FF;
color:#2563EB;
width:46px;
height:46px;
border-radius:11px;
display:flex;
align-items:center;
justify-content:center;
font-size:1.45rem;
flex-shrink:0;
">
⚡
</div>

<div>

<div style="
color:#0F172A;
font-size:1.65rem;
font-weight:700;
line-height:1.2;
">
Inverter Anomaly Detection
</div>

<div style="
color:#64748B;
font-size:0.92rem;
margin-top:0.16rem;
line-height:1.35;
">
Monitor inverter performance and understand overall abnormal behavior.
</div>

</div>

</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# FILTERS
# ============================================================

eval_min_date = df["timestamp"].min().date()
eval_max_date = df["timestamp"].max().date()

min_date = min(
    eval_min_date,
    trend_df["timestamp"].min().date(),
)

max_date = max(
    eval_max_date,
    trend_df["timestamp"].max().date(),
)

show_inverter_filter = (
    "inverter_id" in df.columns
    and df["inverter_id"].nunique() > 1
)

with st.container(border=True):

    if not has_trend_data:
        st.caption(
            "ℹ️ `trend_data.parquet` was not found. "
            "The 7-day healthy baseline can only use the available dashboard data."
        )

    filter_cols = st.columns(
        [1.2, 1.2, 1, 1, 2]
        if show_inverter_filter
        else [1.2, 1.2, 1, 2]
    )

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

        st.write("")

        show_anomalies_only = st.checkbox(
            "Show anomalies only",
            value=False,
        )

    if show_inverter_filter:

        with filter_cols[3]:

            inverter_options = (
                ["All"]
                + sorted(
                    df["inverter_id"]
                    .dropna()
                    .unique()
                    .tolist()
                )
            )

            selected_inverter = st.selectbox(
                "Inverter",
                inverter_options,
            )

        caption_col = filter_cols[4]

    else:

        selected_inverter = "All"
        caption_col = filter_cols[3]

    with caption_col:

        st.caption(
            "**Baseline**: previous 7 days of healthy data."
        )


# ============================================================
# FILTER DATA
# ============================================================

if start_date <= end_date:

    filtered_df = df[
        (df["timestamp"].dt.date >= start_date)
        & (df["timestamp"].dt.date <= end_date)
    ].copy()

    filtered_trend_df = trend_df[
        (trend_df["timestamp"].dt.date >= start_date)
        & (trend_df["timestamp"].dt.date <= end_date)
    ].copy()

else:

    filtered_df = df.iloc[0:0].copy()
    filtered_trend_df = trend_df.iloc[0:0].copy()


if (
    selected_inverter != "All"
    and "inverter_id" in filtered_df.columns
):

    filtered_df = filtered_df[
        filtered_df["inverter_id"]
        == selected_inverter
    ].copy()


if (
    selected_inverter != "All"
    and "inverter_id" in filtered_trend_df.columns
):

    filtered_trend_df = filtered_trend_df[
        filtered_trend_df["inverter_id"]
        == selected_inverter
    ].copy()


if show_anomalies_only:

    filtered_df = filtered_df[
        filtered_df["anomaly_flag"]
    ].copy()


# ============================================================
# OVERVIEW METRICS
# ============================================================

total_observations = len(
    filtered_df
)

total_anomalies = int(
    filtered_df["anomaly_flag"].sum()
) if total_observations else 0

anomaly_rate = (
    total_anomalies
    / total_observations
    * 100
) if total_observations else None


anomaly_df = (
    filtered_df[
        filtered_df["anomaly_flag"]
    ].copy()
    if total_observations
    else filtered_df.copy()
)


events_df = build_anomaly_events(
    anomaly_df
)


# ============================================================
# OVERVIEW
# ============================================================

st.markdown("## Overview")

st.caption(
    "High-level view of inverter observations and detected anomalies."
)

kpi_cols = st.columns(4)

with kpi_cols[0]:

    st.metric(
        "Total Observations",
        f"{total_observations:,}",
    )

with kpi_cols[1]:

    st.metric(
        "Anomalous Observations",
        f"{total_anomalies:,}",
    )

with kpi_cols[2]:

    st.metric(
        "Anomaly Rate",
        fmt_num(
            anomaly_rate,
            2,
            "%",
        ),
    )

with kpi_cols[3]:

    st.metric(
        "Anomaly Events",
        f"{len(events_df):,}",
    )


if (
    total_observations > 0
    and total_anomalies == 0
):

    st.success(
        "No anomalies were detected in the selected period."
    )


# ============================================================
# ANOMALY SCORE
# ============================================================

st.markdown("### Anomaly Score Over Time")

st.caption(
    "Higher values indicate more unusual behavior. Red points are ML-detected anomalies."
)

if (
    total_observations > 0
    and "reconstruction_error" in filtered_df.columns
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered_df["timestamp"],
            y=filtered_df["reconstruction_error"],
            mode="lines",
            name="Anomaly Score",
            line=dict(
                color="#4C78A8",
                width=1.5,
            ),
        )
    )

    anomaly_points = filtered_df[
        filtered_df["anomaly_flag"]
    ]

    if len(anomaly_points) > 0:

        fig.add_trace(
            go.Scatter(
                x=anomaly_points["timestamp"],
                y=anomaly_points["reconstruction_error"],
                mode="markers",
                name="Flagged anomaly",
                marker=dict(
                    size=8,
                    color="#E45756",
                    symbol="circle",
                ),
            )
        )

    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Anomaly Score",
        yaxis_type="log",
        hovermode="x unified",
        height=380,
        template="plotly_white",
        margin=dict(
            t=20,
            l=55,
            r=25,
            b=45,
        ),
    )

    st.plotly_chart(
        fig,
        width="stretch",
    )

else:

    st.info(
        "No anomaly score data available."
    )


# ============================================================
# OVERALL HEALTHY BASELINE
# ============================================================

st.markdown("## Healthy Baseline")

st.caption(
    "The baseline is created dynamically from healthy observations "
    "in the 7 days before detected anomaly events."
)

if len(events_df) > 0:

    with st.spinner(
        "Building 7-day healthy baseline..."
    ):

        baseline_package = build_overall_baseline(
            events_df,
            trend_df,
        )

    healthy_stats = baseline_package.get(
        "overall",
        {},
    )

    if healthy_stats:

        baseline_card = """
<div class="baseline-card">

<div class="baseline-title">
Healthy reference
</div>

<div class="baseline-subtitle">
Median healthy behavior with lower and upper percentile boundaries.
</div>

"""

        display_metrics = [
            (
                "dc_power_kw",
                "DC Power",
                "kW",
            ),
            (
                "ac_power_kw",
                "AC Power",
                "kW",
            ),
            (
                "dc_current_a",
                "DC Current",
                "A",
            ),
            (
                "ac_current_a",
                "AC Current",
                "A",
            ),
            (
                "inverter_temperature_c",
                "Temperature",
                "°C",
            ),
            (
                "efficiency_pct",
                "Efficiency",
                "%",
            ),
        ]

        available = [
            item
            for item in display_metrics
            if item[0] in healthy_stats
        ]

        cols = st.columns(
            min(
                4,
                max(
                    1,
                    len(available),
                ),
            )
        )

        for index, (
            metric,
            label,
            unit,
        ) in enumerate(available):

            stats = healthy_stats[
                metric
            ]

            with cols[
                index
                % len(cols)
            ]:

                st.metric(
                    label,
                    (
                        f"{stats['p5']:.1f}"
                        f"–"
                        f"{stats['p95']:.1f}"
                        f" {unit}"
                    ),
                    help=(
                        f"P5 = {stats['p5']:.2f}, "
                        f"P50 = {stats['p50']:.2f}, "
                        f"P95 = {stats['p95']:.2f}. "
                        "The displayed range is P5-P95."
                    ),
                )

        baseline_card += "</div>"

        st.markdown(
            baseline_card,
            unsafe_allow_html=True,
        )

        st.caption(
            f"Healthy observations used: "
            f"{sum(b.get('healthy_sample_count', 0) for b in baseline_package.get('event_baselines', [])):,} "
            f"across {len(baseline_package.get('event_baselines', []))} anomaly-event reference windows."
        )

    else:

        st.warning(
            "A healthy baseline could not be calculated from the available history."
        )

else:

    st.info(
        "A healthy baseline will be created when anomaly events are available."
    )


# ============================================================
# OVERALL ANOMALY EVIDENCE
# ============================================================

st.markdown("## Overall Anomaly Evidence")

st.caption(
    "All ML-detected anomaly observations are compared with their preceding 7-day healthy references."
)

if len(events_df) > 0:

    with st.spinner(
        "Calculating overall anomaly evidence..."
    ):

        overall_evidence = build_overall_evidence(
            anomaly_df,
            events_df,
            trend_df,
        )

    top_metrics = overall_evidence.get(
        "top_deviating_metrics",
        [],
    )

    if top_metrics:

        evidence_cols = st.columns(
            min(
                4,
                len(top_metrics),
            )
        )

        for index, item in enumerate(
            top_metrics[:4]
        ):

            metric = item["metric"]

            label_map = {
                "dc_power_kw": "DC Power",
                "ac_power_kw": "AC Power",
                "dc_current_a": "DC Current",
                "ac_current_a": "AC Current",
                "power_factor": "Power Factor",
                "frequency_hz": "Frequency",
                "efficiency_pct": "Efficiency",
                "inverter_temperature_c": "Temperature",
                "ambient_temperature_c": "Ambient Temperature",
            }

            label = label_map.get(
                metric,
                metric,
            )

            change = item[
                "median_change_pct"
            ]

            with evidence_cols[
                index
                % len(evidence_cols)
            ]:

                st.metric(
                    label,
                    f"{change:+.1f}%",
                    "median change vs healthy",
                )

        # ----------------------------------------------------
        # Detailed evidence table
        # ----------------------------------------------------

        rows = []

        healthy_stats = overall_evidence.get(
            "healthy_baseline",
            {},
        ).get(
            "overall_statistics",
            {},
        )

        abnormal_stats = overall_evidence.get(
            "abnormal_statistics",
            {},
        )

        comparisons = overall_evidence.get(
            "comparison",
            {},
        )

        outside = overall_evidence.get(
            "outside_healthy_range",
            {},
        )

        label_map = {
            "dc_power_kw": "DC Power",
            "ac_power_kw": "AC Power",
            "dc_current_a": "DC Current",
            "ac_current_a": "AC Current",
            "power_factor": "Power Factor",
            "frequency_hz": "Frequency",
            "efficiency_pct": "Efficiency",
            "inverter_temperature_c": "Temperature",
            "ambient_temperature_c": "Ambient Temperature",
        }

        for metric in healthy_stats:

            if metric not in abnormal_stats:
                continue

            h = healthy_stats[
                metric
            ]

            a = abnormal_stats[
                metric
            ]

            c = comparisons.get(
                metric,
                {},
            )

            o = outside.get(
                metric,
                {},
            )

            rows.append(
                {
                    "Metric": label_map.get(
                        metric,
                        metric,
                    ),
                    "Healthy P5": round(
                        h.get("p5", float("nan")),
                        2,
                    ),
                    "Healthy Median": round(
                        h.get("p50", float("nan")),
                        2,
                    ),
                    "Healthy P95": round(
                        h.get("p95", float("nan")),
                        2,
                    ),
                    "Abnormal Median": round(
                        a.get("p50", float("nan")),
                        2,
                    ),
                    "Median Change": (
                        f"{c.get('median_change_pct', 0):+.1f}%"
                        if c.get(
                            "median_change_pct"
                        )
                        is not None
                        else "—"
                    ),
                    "Outside P5-P95": (
                        f"{o.get('outside_p5_p95_pct', 0):.1f}%"
                        if o.get(
                            "outside_p5_p95_pct"
                        )
                        is not None
                        else "—"
                    ),
                }
            )

        if rows:

            st.dataframe(
                pd.DataFrame(rows),
                width="stretch",
                hide_index=True,
            )

else:

    st.info(
        "No anomaly observations are available for comparison."
    )


# ============================================================
# OVERALL AI EXPLANATION
# ============================================================

st.markdown("## Overall AI Explanation")

st.caption(
    "One explanation generated from all detected anomaly behavior and the healthy statistical baseline."
)

if len(events_df) > 0:

    ai_col, button_col = st.columns(
        [5.2, 1.2]
    )

    with ai_col:

        st.markdown(
            '<div class="ai-title">AI Explanation</div>',
            unsafe_allow_html=True,
        )

    with button_col:

        regenerate = st.button(
            "Regenerate",
            type="primary",
            use_container_width=True,
        )

    if (
        regenerate
        or st.session_state[
            "ai_overall_explanation"
        ] is None
    ):

        with st.spinner(
            "Generating overall explanation..."
        ):

            try:

                explanation = (
                    generate_overall_ai_explanation(
                        overall_evidence
                    )
                )

                st.session_state[
                    "ai_overall_explanation"
                ] = {
                    "explanation": explanation,
                    "evidence": overall_evidence,
                    "error": None,
                }

            except Exception as exc:

                st.session_state[
                    "ai_overall_explanation"
                ] = {
                    "explanation": None,
                    "evidence": overall_evidence,
                    "error": str(exc),
                }

    cached = st.session_state[
        "ai_overall_explanation"
    ]

    if cached and cached.get("error"):

        st.error(
            f"AI explanation failed: "
            f"{cached['error']}"
        )

    elif cached and cached.get(
        "explanation"
    ):

        render_overall_ai_explanation(
            cached["explanation"]
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        checks = validate_overall_ai(
            cached["explanation"],
            cached["evidence"],
        )

        pass_count = sum(
            x["status"] == "PASS"
            for x in checks
        )

        fail_count = sum(
            x["status"] == "FAIL"
            for x in checks
        )

        warn_count = sum(
            x["status"] == "WARN"
            for x in checks
        )

        verdict = (
            "FAIL"
            if fail_count
            else (
                "NEEDS REVIEW"
                if warn_count
                else "PASS"
            )
        )

        verdict_class = (
            "validation-fail"
            if verdict == "FAIL"
            else (
                "validation-warn"
                if verdict == "NEEDS REVIEW"
                else "validation-pass"
            )
        )

        st.markdown(
            f"""
<div class="validation-card">

<div class="validation-header">

<div class="validation-title">
Validation
</div>

<span class="validation-pill {verdict_class}">
{pass_count}/{len(checks)} checks verified
</span>

</div>

<div class="validation-subtitle">
Checks whether the AI explanation is consistent with the statistical
evidence supplied to it. A passed explanation does not prove a physical
fault or root cause.
</div>

</div>
""",
            unsafe_allow_html=True,
        )

        for check in checks:

            status = check["status"]

            icon = (
                "✓"
                if status == "PASS"
                else (
                    "!"
                    if status == "WARN"
                    else "×"
                )
            )

            cls = (
                "check-pass"
                if status == "PASS"
                else (
                    "check-warn"
                    if status == "WARN"
                    else "check-fail"
                )
            )

            st.markdown(
                f"""
<div class="check-row">

<span class="check-icon {cls}">
{icon}
</span>

<div>

<div class="check-name">
{html.escape(check["name"])}
</div>

<div class="check-detail">
{html.escape(check["detail"])}
</div>

</div>

</div>
""",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
<div class="validation-note">
Validation checks evidence consistency only. It does not establish
the physical root cause of the inverter anomaly.
</div>
""",
            unsafe_allow_html=True,
        )

    else:

        st.info(
            "Generate the overall AI explanation to continue."
        )

else:

    st.info(
        "No anomaly events are available for an overall explanation."
    )


# ============================================================
# DETECTED ANOMALIES
# ============================================================

st.markdown("## Detected Anomalies")

st.caption(
    "Continuous anomaly observations grouped into anomaly events."
)

if len(events_df) > 0:

    event_table = events_df.copy()

    event_table["Event"] = [
        f"Event {i + 1}"
        for i in range(
            len(event_table)
        )
    ]

    display_columns = [
        "Event",
        "start_time",
        "end_time",
        "duration_min",
        "anomaly_count",
    ]

    display_columns = [
        col
        for col in display_columns
        if col in event_table.columns
    ]

    friendly_names = {
        "start_time": "Start Time",
        "end_time": "End Time",
        "duration_min": "Duration (min)",
        "anomaly_count": "Anomaly Points",
    }

    table = (
        event_table[
            display_columns
        ]
        .rename(
            columns=friendly_names
        )
        .sort_values(
            "Start Time"
        )
    )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )

else:

    if total_observations > 0:

        st.success(
            "No anomaly events were detected in this period."
        )

    else:

        st.info(
            "No readings are available for this date range."
        )
