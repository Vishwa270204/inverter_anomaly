"""
Inverter Anomaly Detection — Streamlit Dashboard
==================================================
Production frontend only. All ML/training happens in inverter_anomaly.ipynb.

Reads:
    dashboard_data.parquet
    trend_data.parquet       (optional)
    dashboard_baseline.parquet

Does NOT:
    - retrain the model
    - re-run notebook ML code
    - treat anomaly_score_ratio as probability
    - claim feature contributions are proven physical causes
    - provide individual event selection/investigation
"""

import html
import json
import os
import re
import textwrap
from datetime import datetime

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


# ============================================================
# THEME / CSS
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

    /* ========================================================
       KPI CARDS
       ======================================================== */

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

    /* ========================================================
       HEADINGS
       ======================================================== */

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

    h4 {
        color: #1E293B;
        font-size: 1.05rem !important;
        font-weight: 600;
        margin-top: 0 !important;
        margin-bottom: 0.2rem !important;
    }

    hr {
        border-color: var(--border) !important;
        margin: 0.5rem 0 !important;
    }

    [data-testid="stCaptionContainer"],
    .stCaption {
        margin-bottom: 0.2rem !important;
    }

    /* ========================================================
       BUTTONS
       ======================================================== */

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
       CONTAINERS
       ======================================================== */

    [data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--surface);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 10px !important;
        padding: 0.5rem 0.9rem !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]
    > div
    > div[data-testid="stVerticalBlock"] {
        gap: 0.3rem;
    }

    /* ========================================================
       DATAFRAME
       ======================================================== */

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
    }

    /* ========================================================
       FORM LABELS
       ======================================================== */

    label,
    .stSelectbox label,
    .stDateInput label {
        font-weight: 600 !important;
        font-size: 0.83rem !important;
        color: #334155 !important;
    }

    /* ========================================================
       AI EXPLANATION
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

    .ai-section-icon {
        font-size: 1.05rem;
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

    .ai-meta-row {
        display: flex;
        gap: 2.5rem;
        flex-wrap: wrap;
    }

    .ai-meta-label {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 600;
    }

    .ai-meta-value {
        font-size: 0.98rem;
        color: #0F172A;
        font-weight: 600;
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
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

st.session_state.setdefault(
    "ai_explanations",
    {},
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dashboard_data(
    path="dashboard_data.parquet",
):
    df = pd.read_parquet(path)

    if "timestamp" not in df.columns:
        raise ValueError(
            "dashboard_data.parquet must contain a 'timestamp' column."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if "anomaly_flag" in df.columns:
        df["anomaly_flag"] = (
            df["anomaly_flag"]
            .fillna(False)
            .astype(bool)
        )
    else:
        df["anomaly_flag"] = False

    return df


@st.cache_data
def load_trend_data(
    path="trend_data.parquet",
):
    df = pd.read_parquet(path)

    if "timestamp" not in df.columns:
        raise ValueError(
            "trend_data.parquet must contain a 'timestamp' column."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    return (
        df.dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


@st.cache_data
def load_dashboard_baseline(
    path="dashboard_baseline.parquet",
):
    baseline = pd.read_parquet(path)

    required_columns = [
        "dc_power_bin",
        "dc_power_bin_lower",
        "dc_power_bin_upper",
        "healthy_sample_count",
    ]

    missing = [
        column
        for column in required_columns
        if column not in baseline.columns
    ]

    if missing:
        raise ValueError(
            "dashboard_baseline.parquet is missing required "
            f"columns: {missing}"
        )

    numeric_cols = [
        column
        for column in baseline.columns
        if (
            column.endswith("_median")
            or column.endswith("_q10")
            or column.endswith("_q90")
            or column.endswith("_q95")
            or column in [
                "dc_power_bin_lower",
                "dc_power_bin_upper",
                "healthy_sample_count",
            ]
        )
    ]

    for column in numeric_cols:
        baseline[column] = pd.to_numeric(
            baseline[column],
            errors="coerce",
        )

    baseline = baseline.dropna(
        subset=[
            "dc_power_bin_lower",
            "dc_power_bin_upper",
        ]
    ).copy()

    return baseline


def safe_load(
    loader,
    path,
    label,
):
    try:
        return loader(path)

    except FileNotFoundError:
        st.error(
            f"**{label} not found** (`{path}`). "
            "Run `inverter_anomaly.ipynb` to generate it, "
            "then place it next to `app.py`."
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

baseline_df = safe_load(
    load_dashboard_baseline,
    "dashboard_baseline.parquet",
    "Dashboard healthy baseline",
)

if df.empty:
    st.error(
        "`dashboard_data.parquet` loaded but contains no rows."
    )
    st.stop()


# trend_data is optional
try:
    trend_df = load_trend_data(
        "trend_data.parquet"
    )
    has_trend_data = True

except Exception:
    trend_df = df.copy()
    has_trend_data = False


# ============================================================
# SMALL HELPERS
# ============================================================

def fmt_num(
    value,
    decimals=2,
    suffix="",
    dash="—",
):
    """Safely format a numeric value."""

    if value is None:
        return dash

    try:
        if pd.isna(value):
            return dash
    except Exception:
        pass

    return (
        f"{float(value):,.{decimals}f}"
        f"{suffix}"
    )


def clean_value(value):
    """Convert pandas/numpy values into JSON-safe values."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
        ),
    ):
        return str(value)

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass

    return value


def get_contribution_cols(frame):
    """Return feature contribution columns."""

    return [
        column
        for column in frame.columns
        if column.endswith(
            "_contribution_pct"
        )
    ]


@st.cache_resource
def get_groq_client():
    """Create one Groq client per Streamlit process."""

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:
        try:
            api_key = st.secrets[
                "groq"
            ]["api_key"]
        except Exception:
            api_key = None

    if not api_key:
        return None

    return Groq(
        api_key=api_key
    )


# ============================================================
# BUILD PERSISTENT ANOMALY EVENTS
# ============================================================

def build_anomaly_events(
    anomaly_df,
):
    """
    Group continuous anomaly observations into persistent events.

    Rule:
        - 5-minute sampling
        - minimum 12 consecutive points
        - approximately 60 minutes of persistence
        - gaps > 7.5 minutes break an event

    IMPORTANT:
        Events are summary information only.
        There is intentionally NO event selection UI.
    """

    if anomaly_df.empty:
        return pd.DataFrame()

    event_duration_min = 60
    sample_interval_min = 5

    persistence_min_points = int(
        event_duration_min
        / sample_interval_min
    )

    max_run_gap_min = (
        sample_interval_min * 1.5
    )

    work = anomaly_df.copy()

    work["timestamp"] = pd.to_datetime(
        work["timestamp"],
        errors="coerce",
    )

    work = (
        work.dropna(
            subset=["timestamp"]
        )
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    if work.empty:
        return pd.DataFrame()

    time_gap = (
        work["timestamp"]
        .diff()
        .dt.total_seconds()
        / 60.0
    )

    work["event_break"] = (
        time_gap.isna()
        | (
            time_gap
            > max_run_gap_min
        )
    )

    work["run_id"] = (
        work["event_break"]
        .cumsum()
    )

    run_sizes = (
        work.groupby("run_id")
        .size()
    )

    persistent_run_ids = run_sizes[
        run_sizes
        >= persistence_min_points
    ].index

    work = work[
        work["run_id"].isin(
            persistent_run_ids
        )
    ].copy()

    if work.empty:
        return pd.DataFrame()

    events = []

    for event_number, (
        run_id,
        event_rows,
    ) in enumerate(
        work.groupby("run_id"),
        start=1,
    ):

        event_rows = (
            event_rows
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        start_time = (
            event_rows["timestamp"].min()
        )

        end_time = (
            event_rows["timestamp"].max()
        )

        duration_min = (
            (
                end_time
                - start_time
            ).total_seconds()
            / 60.0
        ) + sample_interval_min

        event = {
            "event_id": (
                f"Event {event_number}"
            ),
            "run_id": int(run_id),
            "start_time": start_time,
            "end_time": end_time,
            "duration_min": duration_min,
            "anomaly_count": len(
                event_rows
            ),
        }

        if (
            "anomaly_score_ratio"
            in event_rows.columns
        ):

            scores = pd.to_numeric(
                event_rows[
                    "anomaly_score_ratio"
                ],
                errors="coerce",
            )

            event[
                "max_severity"
            ] = scores.max()

            event[
                "mean_severity"
            ] = scores.mean()

        if (
            "reconstruction_error"
            in event_rows.columns
        ):

            errors = pd.to_numeric(
                event_rows[
                    "reconstruction_error"
                ],
                errors="coerce",
            )

            event[
                "max_reconstruction_error"
            ] = errors.max()

            event[
                "mean_reconstruction_error"
            ] = errors.mean()

        if (
            "inverter_status"
            in event_rows.columns
        ):

            mode = (
                event_rows[
                    "inverter_status"
                ].mode()
            )

            event[
                "dominant_status"
            ] = (
                mode.iloc[0]
                if len(mode)
                else None
            )

        if (
            "anomaly_type"
            in event_rows.columns
        ):

            mode = (
                event_rows[
                    "anomaly_type"
                ].mode()
            )

            event[
                "anomaly_type"
            ] = (
                mode.iloc[0]
                if len(mode)
                else None
            )

        events.append(event)

    return pd.DataFrame(
        events
    ).reset_index(drop=True)


# ============================================================
# AI EXPLANATION
# ============================================================

def generate_overall_ai_explanation(
    anomaly_df,
    events_df,
    start_date,
    end_date,
    selected_inverter,
):
    """
    Generate ONE explanation for the complete anomaly population.

    The AI receives aggregated evidence.
    It does not investigate or select an individual event.
    """

    client = get_groq_client()

    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is not configured. "
            "Set GROQ_API_KEY in the environment "
            "or Streamlit secrets."
        )

    evidence = {
        "analysis_scope": {
            "start_date": str(
                start_date
            ),
            "end_date": str(
                end_date
            ),
            "inverter": (
                "All"
                if selected_inverter
                == "All"
                else str(
                    selected_inverter
                )
            ),
        },
        "anomaly_summary": {
            "anomaly_observation_count": int(
                len(anomaly_df)
            ),
            "persistent_event_count": int(
                len(events_df)
            ),
        },
    }

    # --------------------------------------------------------
    # EVENT PATTERNS
    # --------------------------------------------------------

    if not events_df.empty:

        event_patterns = {}

        for column in [
            "duration_min",
            "anomaly_count",
            "max_severity",
            "mean_severity",
        ]:

            if column not in events_df.columns:
                continue

            values = pd.to_numeric(
                events_df[column],
                errors="coerce",
            ).dropna()

            if not values.empty:

                event_patterns[column] = {
                    "min": clean_value(
                        values.min()
                    ),
                    "mean": clean_value(
                        values.mean()
                    ),
                    "median": clean_value(
                        values.median()
                    ),
                    "max": clean_value(
                        values.max()
                    ),
                }

        evidence[
            "event_patterns"
        ] = event_patterns

    # --------------------------------------------------------
    # OPERATING PATTERNS
    # --------------------------------------------------------

    operating_patterns = {}

    for column in [
        "inverter_status",
        "is_daylight",
        "quality_code",
        "communication_status",
        "fault_code",
        "alarm_code",
    ]:

        if column not in anomaly_df.columns:
            continue

        values = (
            anomaly_df[column]
            .dropna()
            .astype(str)
        )

        if not values.empty:

            counts = (
                values
                .value_counts()
                .head(10)
            )

            operating_patterns[
                column
            ] = {
                str(key): int(value)
                for key, value
                in counts.items()
            }

    evidence[
        "operating_patterns"
    ] = operating_patterns

    # --------------------------------------------------------
    # FEATURE CONTRIBUTIONS
    # --------------------------------------------------------

    contribution_cols = (
        get_contribution_cols(
            anomaly_df
        )
    )

    if contribution_cols:

        means = (
            anomaly_df[
                contribution_cols
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .mean()
            .dropna()
            .sort_values(
                ascending=False
            )
        )

        evidence[
            "feature_contributions"
        ] = [
            {
                "feature": column.replace(
                    "_contribution_pct",
                    "",
                ),
                "mean_contribution_pct": (
                    clean_value(value)
                ),
            }
            for column, value
            in means.head(10).items()
        ]

    # --------------------------------------------------------
    # ANOMALY STATISTICS
    # --------------------------------------------------------

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

    anomaly_statistics = {}

    for column in numeric_cols:

        if column not in anomaly_df.columns:
            continue

        values = pd.to_numeric(
            anomaly_df[column],
            errors="coerce",
        ).dropna()

        if not values.empty:

            anomaly_statistics[
                column
            ] = {
                "min": clean_value(
                    values.min()
                ),
                "mean": clean_value(
                    values.mean()
                ),
                "median": clean_value(
                    values.median()
                ),
                "max": clean_value(
                    values.max()
                ),
            }

    evidence[
        "anomaly_observations"
    ] = {
        "observation_count": int(
            len(anomaly_df)
        ),
        "statistics": (
            anomaly_statistics
        ),
    }

    # --------------------------------------------------------
    # HEALTHY BASELINE
    # --------------------------------------------------------

    if (
        baseline_df is not None
        and not baseline_df.empty
    ):

        baseline_summary = {}

        for metric in [
            "dc_power_kw",
            "ac_power_kw",
            "inverter_temperature_c",
            "ambient_temperature_c",
            "inverter_ambient_temp_delta",
            "efficiency_pct",
            "power_factor",
            "frequency_hz",
        ]:

            median_col = (
                f"{metric}_median"
            )

            q10_col = (
                f"{metric}_q10"
            )

            q90_col = (
                f"{metric}_q90"
            )

            if (
                median_col
                not in baseline_df.columns
            ):
                continue

            med = pd.to_numeric(
                baseline_df[
                    median_col
                ],
                errors="coerce",
            ).dropna()

            if med.empty:
                continue

            if q10_col in baseline_df.columns:
                q10 = pd.to_numeric(
                    baseline_df[
                        q10_col
                    ],
                    errors="coerce",
                ).dropna()
            else:
                q10 = pd.Series(
                    dtype=float
                )

            if q90_col in baseline_df.columns:
                q90 = pd.to_numeric(
                    baseline_df[
                        q90_col
                    ],
                    errors="coerce",
                ).dropna()
            else:
                q90 = pd.Series(
                    dtype=float
                )

            baseline_summary[
                metric
            ] = {
                "median_of_healthy_group_medians": (
                    clean_value(
                        med.median()
                    )
                ),
                "median_q10": (
                    clean_value(
                        q10.median()
                    )
                    if not q10.empty
                    else None
                ),
                "median_q90": (
                    clean_value(
                        q90.median()
                    )
                    if not q90.empty
                    else None
                ),
            }

        evidence[
            "healthy_baseline_summary"
        ] = baseline_summary

    # --------------------------------------------------------
    # AI PROMPT
    # --------------------------------------------------------

    prompt = """
You are an explanation assistant inside a solar inverter
anomaly detection dashboard.

Analyze ALL detected anomalies in the supplied period as ONE GROUP.

There is deliberately NO individual anomaly/event selection.

Do not explain, select, rank, or prioritize any individual anomaly.

Return ONLY valid JSON with this exact structure:

{
  "headline": "One short sentence describing the overall anomaly pattern.",
  "summary": "2-3 plain-language sentences explaining what is generally happening.",
  "why_it_happened": [
    "pattern-based reason 1",
    "pattern-based reason 2",
    "pattern-based reason 3"
  ],
  "when_time": "Describe the overall time or operating-condition pattern only if supported.",
  "when_duration": "Describe the general persistence pattern of anomalies.",
  "recommended_actions": [
    "practical check 1",
    "practical check 2",
    "practical check 3"
  ]
}

Rules:

- Explain recurring patterns across the COMPLETE anomaly population.
- Do not mention a latest event.
- Do not explain one specific event.
- Use only supplied evidence.
- Never invent measurements, times, thresholds, or causes.
- Feature contributions indicate variables associated with unusual
  model error. They do NOT prove physical root cause.
- Healthy baseline information is supporting evidence, not proof of causality.
- Do not claim a confirmed physical fault unless the supplied evidence
  explicitly supports it.
- If the physical cause cannot be determined, say so.
- Prefer patterns supported by multiple observations or events.
- Avoid operator-facing ML terminology such as autoencoder,
  reconstruction error, threshold, probability, or confidence.
- Write for a solar plant operator.
- Keep the explanation concise and practical.

SUPPLIED EVIDENCE:
""" + json.dumps(
        evidence,
        default=str,
        ensure_ascii=False,
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "Return ONLY a valid JSON object. "
                    "Do not use markdown fences. "
                    "Do not add commentary before or after the JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
        max_tokens=1600,
        response_format={
            "type": "json_object"
        },
    )

    raw = (
        response
        .choices[0]
        .message
        .content
        or ""
    ).strip()

    # Remove accidental markdown fences
    if raw.startswith("```"):

        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )

        raw = re.sub(
            r"\s*```$",
            "",
            raw,
        )

    # Extract JSON object if surrounding text exists
    start = raw.find("{")
    end = raw.rfind("}")

    if (
        start >= 0
        and end > start
    ):
        raw = raw[
            start:end + 1
        ]

    try:

        parsed = json.loads(
            raw
        )

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            "The AI returned malformed JSON. "
            f"Parser error: {exc}."
        ) from exc

    if not isinstance(
        parsed,
        dict,
    ):
        raise RuntimeError(
            "The AI returned valid JSON, "
            "but it was not a JSON object."
        )

    return (
        parsed,
        evidence,
    )


# ============================================================
# AI RENDERER
# ============================================================

def render_ai_explanation(
    data,
):
    """Render the overall AI explanation."""

    if not data:
        return

    # Backward compatibility
    if isinstance(
        data,
        str,
    ):

        safe = html.escape(
            re.sub(
                r"\s+",
                " ",
                data,
            ).strip()
        )

        st.markdown(
            f"""
            <div class="ai-card">
                <div class="ai-alert-box">
                    <div class="ai-alert-icon">
                        ⚠️
                    </div>

                    <div class="ai-alert-desc">
                        {safe}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    def esc(value):

        return (
            html.escape(
                str(value)
            )
            if value is not None
            else ""
        )

    headline = esc(
        data.get(
            "headline",
            "",
        )
    )

    summary = esc(
        data.get(
            "summary",
            "",
        )
    )

    why_bullets = [
        esc(item)
        for item in data.get(
            "why_it_happened",
            [],
        )
        if item
    ]

    action_bullets = [
        esc(item)
        for item in data.get(
            "recommended_actions",
            [],
        )
        if item
    ]

    when_time = esc(
        data.get(
            "when_time",
            "—",
        )
    )

    when_duration = esc(
        data.get(
            "when_duration",
            "—",
        )
    )

    why_html = "".join(
        f"<li>{item}</li>"
        for item in why_bullets
    )

    action_html = "".join(
        f"<li>{item}</li>"
        for item in action_bullets
    )

    html_block = textwrap.dedent(
        f"""
        <div class="ai-card">

            <div class="ai-alert-box">

                <div class="ai-alert-icon">
                    ⚠️
                </div>

                <div>

                    <div class="ai-alert-title">
                        {headline}
                    </div>

                    <div class="ai-alert-desc">
                        {summary}
                    </div>

                </div>

            </div>


            <div class="ai-section">

                <div class="ai-section-header">
                    <span class="ai-section-icon">
                        🔍
                    </span>

                    Why it happened?
                </div>

                <ul class="ai-bullets">
                    {why_html}
                </ul>

            </div>


            <div class="ai-section">

                <div class="ai-section-header">
                    <span class="ai-section-icon">
                        🕐
                    </span>

                    When it occurred?
                </div>

                <div class="ai-meta-row">

                    <div>

                        <div class="ai-meta-label">
                            Overall time pattern
                        </div>

                        <div class="ai-meta-value">
                            {when_time}
                        </div>

                    </div>


                    <div>

                        <div class="ai-meta-label">
                            Duration pattern
                        </div>

                        <div class="ai-meta-value">
                            {when_duration}
                        </div>

                    </div>

                </div>

            </div>


            <div class="ai-section">

                <div class="ai-section-header">
                    <span class="ai-section-icon">
                        🔧
                    </span>

                    Recommended action
                </div>

                <ul class="ai-bullets">
                    {action_html}
                </ul>

            </div>


            <div class="ai-footer-note">

                <span>
                    ℹ️
                </span>

                <span>
                    This explanation is based on recurring
                    patterns observed in the selected period
                    and the trained anomaly detection model.
                    It does not confirm a physical fault
                    or root cause.
                </span>

            </div>

        </div>
        """
    )

    # Prevent Markdown from interpreting indentation as code
    html_block = re.sub(
        r"(?m)^[ \t]+",
        "",
        html_block,
    ).strip()

    st.markdown(
        html_block,
        unsafe_allow_html=True,
    )


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
                Monitor inverter performance, review anomaly
                patterns, and understand recurring behavior.
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FILTERS
# ============================================================

eval_min_date = (
    df["timestamp"]
    .min()
    .date()
)

eval_max_date = (
    df["timestamp"]
    .max()
    .date()
)

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


with st.container(
    border=True
):

    if not has_trend_data:

        st.caption(
            "ℹ️ `trend_data.parquet` not found — "
            "showing the evaluation period only. "
            "Run notebook Section 19 to enable full history."
        )

    filter_cols = st.columns(
        (
            [1.2, 1.2, 1, 1, 2]
            if show_inverter_filter
            else [1.2, 1.2, 1, 2]
        )
    )

    # Start date
    with filter_cols[0]:

        start_date = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )

    # End date
    with filter_cols[1]:

        end_date = st.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

    # Anomaly filter
    with filter_cols[2]:

        st.write("")

        show_anomalies_only = st.checkbox(
            "Show anomalies only",
            value=False,
        )

    # Inverter filter
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
            "**Severity:** higher = further outside normal range."
        )

    if start_date > end_date:

        st.warning(
            "Start date is after end date — "
            "swap them to see results."
        )

    elif has_trend_data and (
        start_date < eval_min_date
        or end_date > eval_max_date
    ):

        st.caption(
            f"ℹ️ Anomaly detection only covers "
            f"**{eval_min_date} to {eval_max_date}**. "
            "Dates outside that window show raw readings "
            "but no anomaly results."
        )


# ============================================================
# FILTER DATA
# ============================================================

if start_date <= end_date:

    filtered_df = df[
        (
            df["timestamp"].dt.date
            >= start_date
        )
        & (
            df["timestamp"].dt.date
            <= end_date
        )
    ].copy()

    filtered_trend_df = trend_df[
        (
            trend_df["timestamp"].dt.date
            >= start_date
        )
        & (
            trend_df["timestamp"].dt.date
            <= end_date
        )
    ].copy()

else:

    filtered_df = df.iloc[0:0].copy()

    filtered_trend_df = (
        trend_df.iloc[0:0].copy()
    )


# Apply inverter filter
if selected_inverter != "All":

    if "inverter_id" in filtered_df.columns:

        filtered_df = filtered_df[
            filtered_df["inverter_id"]
            == selected_inverter
        ].copy()

    if (
        "inverter_id"
        in filtered_trend_df.columns
    ):

        filtered_trend_df = (
            filtered_trend_df[
                filtered_trend_df[
                    "inverter_id"
                ]
                == selected_inverter
            ]
            .copy()
        )


# Apply anomaly-only filter
if show_anomalies_only:

    filtered_df = filtered_df[
        filtered_df["anomaly_flag"]
    ].copy()


# Empty state
if (
    filtered_df.empty
    and filtered_trend_df.empty
):

    st.warning(
        "No observations match the current filters. "
        "Adjust the filters above."
    )


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_observations = len(
    filtered_df
)

total_anomalies = (
    int(
        filtered_df[
            "anomaly_flag"
        ].sum()
    )
    if total_observations
    else 0
)

anomaly_rate = (
    total_anomalies
    / total_observations
    * 100
    if total_observations
    else None
)

max_temperature = (
    filtered_df[
        "inverter_temperature_c"
    ].max()
    if (
        "inverter_temperature_c"
        in filtered_df.columns
        and total_observations
    )
    else None
)


# All anomaly observations
anomaly_df = (
    filtered_df[
        filtered_df["anomaly_flag"]
    ].copy()
    if total_observations
    else filtered_df.copy()
)


# Build persistent event summary.
# This is NOT selectable.
events_df = build_anomaly_events(
    anomaly_df
)


# ============================================================
# OVERVIEW
# ============================================================

st.markdown(
    "## Overview"
)

st.caption(
    "High-level view of inverter observations and "
    "detected anomaly patterns for the selected period."
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
        "Max Temperature",
        fmt_num(
            max_temperature,
            1,
            " °C",
        ),
    )


if (
    total_observations > 0
    and total_anomalies == 0
):

    st.caption(
        "✅ No anomalies found in this period — "
        "everything looks normal."
    )


# ============================================================
# HEALTHY BASELINE
# ============================================================

st.markdown(
    "### Healthy Baseline"
)

st.caption(
    "Typical healthy operating range derived "
    "from the healthy baseline dataset."
)

baseline_features = {
    "dc_power_kw": "DC Power",
    "inverter_temperature_c": "Temperature",
}

baseline_text = []

for feature, label in (
    baseline_features.items()
):

    q10_col = (
        f"{feature}_q10"
    )

    q90_col = (
        f"{feature}_q90"
    )

    if (
        q10_col
        not in baseline_df.columns
        or q90_col
        not in baseline_df.columns
    ):
        continue

    low = (
        baseline_df[q10_col]
        .median()
    )

    high = (
        baseline_df[q90_col]
        .median()
    )

    if (
        pd.isna(low)
        or pd.isna(high)
    ):
        continue

    if feature.endswith("_kw"):
        unit = " kW"
    elif feature.endswith("_a"):
        unit = " A"
    elif feature.endswith("_c"):
        unit = " °C"
    else:
        unit = ""

    baseline_text.append(
        (
            label,
            f"{low:.1f}–{high:.1f}{unit}",
        )
    )


if baseline_text:

    baseline_cols = st.columns(
        min(
            4,
            len(baseline_text),
        )
    )

    for index, (
        label,
        value,
    ) in enumerate(
        baseline_text
    ):

        with baseline_cols[index]:

            st.metric(
                label=label,
                value=value,
            )

else:

    st.info(
        "Healthy baseline ranges are not available "
        "for the displayed metrics."
    )


# ============================================================
# ANOMALY SCORE OVER TIME
# ============================================================

st.markdown(
    "### Anomaly Score Over Time"
)

st.caption(
    "Higher points mean more unusual behavior. "
    "Red dots identify flagged anomaly observations."
)


if (
    total_observations > 0
    and "reconstruction_error"
    in filtered_df.columns
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=filtered_df[
                "timestamp"
            ],
            y=filtered_df[
                "reconstruction_error"
            ],
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

    if not anomaly_points.empty:

        fig.add_trace(
            go.Scatter(
                x=anomaly_points[
                    "timestamp"
                ],
                y=anomaly_points[
                    "reconstruction_error"
                ],
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
        "No anomaly score data available "
        "for the selected period."
    )


# ============================================================
# OVERALL ANOMALY EXPLANATION
# ============================================================

st.markdown(
    "## Overall Anomaly Explanation"
)

st.caption(
    "AI analysis of recurring patterns across ALL "
    "detected anomalies in the selected period. "
    "No individual event is selected or investigated."
)


if not anomaly_df.empty:

    header_col, button_col = st.columns(
        [5.5, 1.5]
    )

    with header_col:

        st.markdown(
            '<div class="ai-title">'
            'Why are anomalies occurring?'
            '</div>',
            unsafe_allow_html=True,
        )

    with button_col:

        regenerate_overall = st.button(
            "Regenerate",
            type="primary",
            use_container_width=True,
            help=(
                "Generate a fresh explanation using "
                "all detected anomalies in the selected period."
            ),
        )

    overall_cache_key = (
        f"overall_"
        f"{selected_inverter}_"
        f"{start_date}_"
        f"{end_date}"
    )

    cached_overall = (
        st.session_state[
            "ai_explanations"
        ].get(
            overall_cache_key
        )
    )

    if (
        regenerate_overall
        or cached_overall is None
    ):

        with st.spinner(
            "Analyzing all detected anomalies..."
        ):

            try:

                explanation, evidence = (
                    generate_overall_ai_explanation(
                        anomaly_df=anomaly_df,
                        events_df=events_df,
                        start_date=start_date,
                        end_date=end_date,
                        selected_inverter=selected_inverter,
                    )
                )

                cached_overall = {
                    "explanation": explanation,
                    "evidence": evidence,
                    "error": None,
                }

            except Exception as exc:

                cached_overall = {
                    "explanation": None,
                    "evidence": None,
                    "error": str(exc),
                }

            st.session_state[
                "ai_explanations"
            ][
                overall_cache_key
            ] = cached_overall

    if cached_overall.get(
        "error"
    ):

        st.error(
            "AI explanation failed: "
            f"{cached_overall['error']}"
        )

    elif cached_overall.get(
        "explanation"
    ):

        render_ai_explanation(
            cached_overall[
                "explanation"
            ]
        )

    else:

        st.warning(
            "AI did not return an "
            "overall explanation."
        )

else:

    st.info(
        "No anomalies were detected in the selected "
        "period, so there is no anomaly pattern to explain."
    )


# ============================================================
# DETECTED ANOMALIES
# ============================================================

st.markdown(
    "## Detected Anomalies"
)

st.caption(
    "Persistent anomaly events detected in the selected "
    "period. Events are summarized for monitoring only."
)


if not events_df.empty:

    event_table = events_df.copy()

    event_table["Event"] = [
        f"Event {index + 1}"
        for index in range(
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
        column
        for column in display_columns
        if column in event_table.columns
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
        .copy()
    )

    if "Start Time" in table.columns:

        table[
            "Start Time"
        ] = (
            pd.to_datetime(
                table[
                    "Start Time"
                ],
                errors="coerce",
            )
            .dt.strftime(
                "%Y-%m-%d %H:%M"
            )
        )

    if "End Time" in table.columns:

        table[
            "End Time"
        ] = (
            pd.to_datetime(
                table[
                    "End Time"
                ],
                errors="coerce",
            )
            .dt.strftime(
                "%Y-%m-%d %H:%M"
            )
        )

    if (
        "Duration (min)"
        in table.columns
    ):

        table[
            "Duration (min)"
        ] = (
            pd.to_numeric(
                table[
                    "Duration (min)"
                ],
                errors="coerce",
            )
            .round(0)
        )

    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
    )

else:

    if total_observations > 0:

        if total_anomalies > 0:

            st.info(
                f"{total_anomalies:,} anomaly observations "
                "were flagged, but none met the 60-minute "
                "persistence rule required to form an "
                "anomaly event."
            )

        else:

            st.success(
                "✅ No anomaly events were detected "
                "in this period."
            )

    else:

        st.info(
            "No readings in this date range. "
            "Try a different range above."
        )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Model outputs are shown for monitoring and investigation "
    "support. Anomaly scores are not probabilities, and feature "
    "contributions indicate variables associated with unusual "
    "model error rather than proven physical root causes."
)
