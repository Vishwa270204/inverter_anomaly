"""
Compact data validation for one selected anomaly event.

This module validates the DATA/EVIDENCE supplied to the LLM.
It does not judge the wording or style of the LLM response.

Only checks that are useful for the selected-event explanation are kept:
1. Event data / time window
2. Numerical evidence used by the explanation
3. 24-hour pre-event trend evidence
4. Healthy-baseline evidence (only when baseline data is supplied)

It intentionally does NOT validate:
- whether the user selected the "right" anomaly
- generic operating condition
- AI wording/style
- physical root cause
- recommendations
- anomaly-model accuracy
- individual min/max/mean values as separate UI checks
"""

import pandas as pd
from datetime import timedelta

EVENT_NUMERIC_COLS = [
    "dc_power_kw", "dc_current_a", "ac_power_kw", "ac_current_a",
    "power_factor", "frequency_hz", "efficiency_pct",
    "inverter_temperature_c", "ambient_temperature_c",
    "poa_w_m2", "ghi_w_m2", "packet_loss_pct",
    "communication_latency_ms",
]

TREND_NUMERIC_COLS = [
    "dc_power_kw", "ac_power_kw", "dc_current_a", "ac_current_a",
    "inverter_temperature_c", "efficiency_pct", "power_factor",
]

BASELINE_MATCH_COLS = ["inverter_status", "is_daylight", "hour", "month"]

REL_TOL = 0.01
ABS_TOL = 1e-6


def _clean(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _close(a, b):
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    return abs(a - b) <= max(ABS_TOL, REL_TOL * max(abs(a), abs(b)))


def _add(results, name, status, detail):
    results.append({
        "rule": name,
        "status": status,
        "detail": detail,
    })


def _event_rows(evidence, df):
    event = evidence.get("event", {})
    start = pd.to_datetime(event.get("start_time"), errors="coerce")
    end = pd.to_datetime(event.get("end_time"), errors="coerce")

    if pd.isna(start) or pd.isna(end) or "timestamp" not in df.columns:
        return None, start, end

    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    if "anomaly_flag" in df.columns:
        mask &= df["anomaly_flag"].fillna(False).astype(bool)

    return df.loc[mask], start, end


def check_event_data(evidence, df, results):
    """One compact check for the selected event's time/count/statistics."""
    event_rows, start, end = _event_rows(evidence, df)

    if event_rows is None:
        _add(results, "Event data", "WARN",
             "Cannot independently verify the event window from the supplied data.")
        return

    if event_rows.empty:
        _add(results, "Event data", "FAIL",
             f"No matching anomaly rows were found for {start} → {end}.")
        return

    event = evidence.get("event", {})
    claimed_count = event.get("anomaly_count")
    actual_count = len(event_rows)

    problems = []

    if claimed_count is not None:
        try:
            if int(claimed_count) != actual_count:
                problems.append(
                    f"anomaly count {claimed_count} vs {actual_count}"
                )
        except (TypeError, ValueError):
            problems.append("invalid anomaly_count in evidence")

    actual_start = event_rows["timestamp"].min()
    actual_end = event_rows["timestamp"].max()

    if actual_start != start:
        problems.append(f"start {start} vs {actual_start}")
    if actual_end != end:
        problems.append(f"end {end} vs {actual_end}")

    if problems:
        _add(results, "Event data", "FAIL",
             "Event evidence does not match the selected event: " +
             "; ".join(problems))
    else:
        _add(results, "Event data", "PASS",
             f"Selected event window and {actual_count} anomaly points match the source data.")


def check_numerical_evidence(evidence, df, results):
    """Aggregate check for the numeric evidence passed to the LLM."""
    event_rows, _, _ = _event_rows(evidence, df)

    if event_rows is None or event_rows.empty:
        _add(results, "Numerical evidence", "WARN",
             "No event rows available for an independent numeric check.")
        return

    cached_stats = evidence.get("event_observations", {}).get("statistics", {})
    mismatches = []

    for col in EVENT_NUMERIC_COLS:
        if col not in event_rows.columns or col not in cached_stats:
            continue

        series = pd.to_numeric(event_rows[col], errors="coerce").dropna()
        if series.empty:
            continue

        expected = {
            "min": _clean(series.min()),
            "max": _clean(series.max()),
            "mean": _clean(series.mean()),
        }
        cached = cached_stats.get(col, {})

        for stat, actual in expected.items():
            if stat not in cached or actual is None:
                continue
            if not _close(cached[stat], actual):
                mismatches.append(
                    f"{col}.{stat}: evidence={cached[stat]}, source={actual:.3f}"
                )

    if mismatches:
        _add(results, "Numerical evidence", "FAIL",
             "Some evidence values do not match the source data: " +
             "; ".join(mismatches[:4]))
    else:
        _add(results, "Numerical evidence", "PASS",
             "Event measurement statistics supplied to the LLM match the source data.")


def check_pre_event_trend(evidence, trend_df, results):
    """One compact check for the 24-hour pre-event evidence."""
    cached = evidence.get("pre_event_trend")
    if not cached:
        # Trend evidence was not used, so there is nothing to validate.
        return

    if trend_df is None or "timestamp" not in trend_df.columns:
        _add(results, "Pre-event trend", "WARN",
             "The explanation contains pre-event trend evidence, but trend data is unavailable.")
        return

    event = evidence.get("event", {})
    start = pd.to_datetime(event.get("start_time"), errors="coerce")
    if pd.isna(start):
        _add(results, "Pre-event trend", "WARN",
             "Cannot determine the start of the pre-event trend window.")
        return

    window_start = start - timedelta(hours=24)
    window = trend_df[
        (trend_df["timestamp"] >= window_start) &
        (trend_df["timestamp"] <= start)
    ]

    if window.empty:
        _add(results, "Pre-event trend", "WARN",
             "No source rows were found for the 24-hour pre-event window.")
        return

    problems = []

    claimed_obs = cached.get("observations")
    if claimed_obs is not None:
        try:
            if int(claimed_obs) != len(window):
                problems.append(f"observation count {claimed_obs} vs {len(window)}")
        except (TypeError, ValueError):
            problems.append("invalid observation count")

    cached_stats = cached.get("statistics", {})
    for col in TREND_NUMERIC_COLS:
        if col not in trend_df.columns or col not in cached_stats:
            continue

        series = pd.to_numeric(window[col], errors="coerce").dropna()
        if len(series) < 2:
            continue

        expected = {
            "start": _clean(series.iloc[0]),
            "end": _clean(series.iloc[-1]),
            "min": _clean(series.min()),
            "max": _clean(series.max()),
        }

        for stat, actual in expected.items():
            cached_value = cached_stats[col].get(stat)
            if cached_value is not None and actual is not None and not _close(cached_value, actual):
                problems.append(f"{col}.{stat}")

    if problems:
        _add(results, "Pre-event trend", "FAIL",
             "Pre-event evidence does not match the source trend data: " +
             ", ".join(problems[:5]))
    else:
        _add(results, "Pre-event trend", "PASS",
             "The supplied 24-hour pre-event trend evidence matches the source data.")


def check_healthy_baseline(evidence, df, baseline_df, results):
    """One compact check for baseline references used by the explanation."""
    cached = evidence.get("healthy_baseline")
    if not cached or "reference" not in cached:
        return

    if baseline_df is None or baseline_df.empty:
        _add(results, "Healthy baseline", "WARN",
             "The explanation contains baseline evidence, but the baseline data is unavailable.")
        return

    event_rows, _, _ = _event_rows(evidence, df)
    if event_rows is None or event_rows.empty:
        _add(results, "Healthy baseline", "WARN",
             "Cannot identify the event row needed to reproduce the baseline match.")
        return

    reference_row = event_rows.iloc[0]
    conditions = {
        c: _clean(reference_row.get(c))
        for c in BASELINE_MATCH_COLS
        if c in df.columns
    }

    matched = baseline_df.copy()
    for col, value in conditions.items():
        if col in matched.columns and value is not None:
            matched = matched[matched[col].astype(str) == str(value)]

    if matched.empty:
        _add(results, "Healthy baseline", "WARN",
             "No matching healthy-baseline group was found for the event.")
        return

    problems = []

    cached_count = cached.get("healthy_sample_count")
    if cached_count is not None and "healthy_sample_count" in matched.columns:
        actual_count = int(matched["healthy_sample_count"].sum())
        try:
            if int(cached_count) != actual_count:
                problems.append(
                    f"sample count {cached_count} vs {actual_count}"
                )
        except (TypeError, ValueError):
            problems.append("invalid healthy_sample_count")

    for metric, cached_ref in cached["reference"].items():
        median_col = f"{metric}_median"
        q10_col = f"{metric}_q10"
        q90_col = f"{metric}_q90"

        if median_col not in matched.columns:
            continue

        row = matched.iloc[0]

        expected = {
            "median": _clean(row.get(median_col)),
            "typical_low_q10": _clean(row.get(q10_col)),
            "typical_high_q90": _clean(row.get(q90_col)),
        }

        for stat, actual in expected.items():
            cached_value = cached_ref.get(stat)
            if cached_value is not None and actual is not None and not _close(cached_value, actual):
                problems.append(f"{metric}.{stat}")

    if problems:
        _add(results, "Healthy baseline", "FAIL",
             "Baseline evidence does not match the selected event's baseline group: " +
             ", ".join(problems[:5]))
    else:
        _add(results, "Healthy baseline", "PASS",
             "Baseline references supplied to the LLM match the selected event's baseline group.")


def validate_event_data(evidence, df, trend_df=None, baseline_df=None):
    """
    Validate only the evidence/data plumbing for the selected anomaly event.

    Returns at most four checks. These checks are intentionally separate from
    the LLM-claim checks in app.py.
    """
    results = []

    check_event_data(evidence, df, results)
    check_numerical_evidence(evidence, df, results)

    if trend_df is not None:
        check_pre_event_trend(evidence, trend_df, results)

    if baseline_df is not None:
        check_healthy_baseline(evidence, df, baseline_df, results)

    return results


def summarize(results):
    fails = [r for r in results if r["status"] == "FAIL"]
    warns = [r for r in results if r["status"] == "WARN"]

    if fails:
        verdict = "FAIL"
    elif warns:
        verdict = "NEEDS_REVIEW"
    else:
        verdict = "PASS"

    return {
        "verdict": verdict,
        "pass_count": sum(1 for r in results if r["status"] == "PASS"),
        "fail_count": len(fails),
        "warn_count": len(warns),
    }
