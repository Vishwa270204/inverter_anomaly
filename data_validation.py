"""
Data-Fetch Validation for a Selected Anomaly Event
====================================================
This does NOT check the LLM's wording. It checks whether the `evidence`
dict that was fed to the LLM (built in generate_ai_explanation(), app.py)
was itself correctly computed from the source data for the SPECIFIC event
the user has selected in the dropdown.

Method: recompute every number independently, straight from df / trend_df /
baseline_df using the same filters (event time window, 24h pre-event
window, matching baseline conditions), then diff the recomputed values
against what's already sitting in `evidence`. A mismatch means the cached
evidence is stale, mis-filtered, or was built from the wrong event -- a
data/plumbing bug, not a wording problem.

Usage:
    from data_validation import validate_event_data

    results = validate_event_data(
        selected_event=selected_event,   # row from events_df
        evidence=cached["evidence"],     # what was actually sent to the LLM
        df=df,                           # full scored dashboard_data.parquet
        trend_df=trend_df,               # full trend_data.parquet
        baseline_df=baseline_df,         # dashboard_baseline.parquet
    )
"""

import pandas as pd
from datetime import timedelta

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

EVENT_NUMERIC_COLS = [
    "dc_power_kw", "dc_current_a", "ac_power_kw", "ac_current_a",
    "power_factor", "frequency_hz", "efficiency_pct",
    "inverter_temperature_c", "ambient_temperature_c",
    "poa_w_m2", "ghi_w_m2", "packet_loss_pct", "communication_latency_ms",
]

TREND_NUMERIC_COLS = [
    "dc_power_kw", "ac_power_kw", "dc_current_a", "ac_current_a",
    "inverter_temperature_c", "efficiency_pct", "power_factor",
]

BASELINE_MATCH_COLS = ["inverter_status", "is_daylight", "hour", "month"]

REL_TOL = 0.01   # 1% relative tolerance
ABS_TOL = 1e-6   # absolute floor for near-zero values


def _close(a, b, rel_tol=REL_TOL, abs_tol=ABS_TOL):
    if a is None or b is None:
        return a is None and b is None
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return str(a) == str(b)
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


def _clean(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


# ------------------------------------------------------------
# 1. EVENT WINDOW / ANOMALY COUNT
# ------------------------------------------------------------

def check_event_window(selected_event, evidence, df, results):
    """Recompute which rows in df are (a) flagged anomalies and (b) inside
    the event's [start_time, end_time], independent of whatever event_rows
    the app already sliced. Confirms the event's own anomaly_count and time
    bounds are actually correct against the raw data."""
    start = pd.to_datetime(selected_event.get("start_time"))
    end = pd.to_datetime(selected_event.get("end_time"))

    if "anomaly_flag" not in df.columns:
        results.append({
            "rule": "event_window",
            "status": "WARN",
            "detail": "df has no anomaly_flag column -- cannot recheck event window.",
        })
        return

    recomputed = df[
        (df["anomaly_flag"]) &
        (df["timestamp"] >= start) &
        (df["timestamp"] <= end)
    ]

    claimed_count = evidence.get("event", {}).get("anomaly_count")
    actual_count = len(recomputed)
    if claimed_count is not None and int(claimed_count) != actual_count:
        results.append({
            "rule": "event_anomaly_count",
            "status": "FAIL",
            "detail": f"evidence claims anomaly_count={claimed_count}, but recomputing "
                      f"from df for [{start}, {end}] gives {actual_count}.",
        })
    else:
        results.append({
            "rule": "event_anomaly_count",
            "status": "PASS",
            "detail": f"anomaly_count={actual_count} matches evidence for this window.",
        })

    if actual_count > 0:
        actual_start, actual_end = recomputed["timestamp"].min(), recomputed["timestamp"].max()
        if actual_start != start or actual_end != end:
            results.append({
                "rule": "event_time_bounds",
                "status": "FAIL",
                "detail": f"Recomputed anomaly points span [{actual_start}, {actual_end}], "
                          f"but the event claims [{start}, {end}].",
            })
        else:
            results.append({
                "rule": "event_time_bounds",
                "status": "PASS",
                "detail": f"Event time bounds [{start}, {end}] match the raw flagged points.",
            })


# ------------------------------------------------------------
# 2. EVENT OBSERVATION STATISTICS
# ------------------------------------------------------------

def check_event_statistics(selected_event, evidence, df, results):
    """Recompute min/max/mean for each numeric column directly from df,
    filtered to this event's flagged points, and diff against
    evidence['event_observations']['statistics']."""
    start = pd.to_datetime(selected_event.get("start_time"))
    end = pd.to_datetime(selected_event.get("end_time"))

    if "anomaly_flag" in df.columns:
        event_rows = df[(df["anomaly_flag"]) & (df["timestamp"] >= start) & (df["timestamp"] <= end)]
    else:
        event_rows = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    cached_stats = evidence.get("event_observations", {}).get("statistics", {})

    for col in EVENT_NUMERIC_COLS:
        if col not in df.columns or col not in cached_stats:
            continue
        series = pd.to_numeric(event_rows[col], errors="coerce").dropna()
        if series.empty:
            continue

        recomputed = {
            "min": _clean(series.min()),
            "max": _clean(series.max()),
            "mean": _clean(series.mean()),
        }
        cached = cached_stats[col]

        for stat in ("min", "max", "mean"):
            if stat not in cached:
                continue
            if _close(cached[stat], recomputed[stat]):
                results.append({
                    "rule": f"event_stat[{col}.{stat}]",
                    "status": "PASS",
                    "detail": f"evidence {stat}={cached[stat]:.3f} matches recomputed "
                              f"{recomputed[stat]:.3f}.",
                })
            else:
                results.append({
                    "rule": f"event_stat[{col}.{stat}]",
                    "status": "FAIL",
                    "detail": f"evidence {stat}={cached[stat]} does NOT match recomputed "
                              f"{recomputed[stat]:.3f} from raw df for this event window.",
                })


# ------------------------------------------------------------
# 3. PRE-EVENT TREND (24h before)
# ------------------------------------------------------------

def check_pre_event_trend(selected_event, evidence, trend_df, results):
    """Recompute the 24h pre-event trend window stats from trend_df and
    diff against evidence['pre_event_trend']['statistics']."""
    cached = evidence.get("pre_event_trend")
    if not cached:
        return  # nothing to check if the app didn't attach a trend section

    start = pd.to_datetime(selected_event.get("start_time"))
    window_start = start - timedelta(hours=24)
    window = trend_df[(trend_df["timestamp"] >= window_start) & (trend_df["timestamp"] <= start)]

    if window.empty:
        results.append({
            "rule": "pre_event_trend_window",
            "status": "WARN",
            "detail": "No rows found in trend_df for the recomputed 24h pre-event window.",
        })
        return

    claimed_obs = cached.get("observations")
    if claimed_obs is not None and int(claimed_obs) != len(window):
        results.append({
            "rule": "pre_event_trend_observations",
            "status": "FAIL",
            "detail": f"evidence claims {claimed_obs} pre-event observations, "
                      f"recomputed count is {len(window)}.",
        })
    else:
        results.append({
            "rule": "pre_event_trend_observations",
            "status": "PASS",
            "detail": f"Pre-event observation count {len(window)} matches evidence.",
        })

    cached_stats = cached.get("statistics", {})
    for col in TREND_NUMERIC_COLS:
        if col not in trend_df.columns or col not in cached_stats:
            continue
        series = pd.to_numeric(window[col], errors="coerce").dropna()
        if len(series) < 2:
            continue
        recomputed = {
            "start": _clean(series.iloc[0]),
            "end": _clean(series.iloc[-1]),
            "min": _clean(series.min()),
            "max": _clean(series.max()),
        }
        for stat in ("start", "end", "min", "max"):
            if stat not in cached_stats[col]:
                continue
            if _close(cached_stats[col][stat], recomputed[stat]):
                results.append({
                    "rule": f"pre_event_trend[{col}.{stat}]",
                    "status": "PASS",
                    "detail": f"evidence {stat}={cached_stats[col][stat]:.3f} matches "
                              f"recomputed {recomputed[stat]:.3f}.",
                })
            else:
                results.append({
                    "rule": f"pre_event_trend[{col}.{stat}]",
                    "status": "FAIL",
                    "detail": f"evidence {stat}={cached_stats[col][stat]} does NOT match "
                              f"recomputed {recomputed[stat]:.3f} from trend_df.",
                })


# ------------------------------------------------------------
# 4. HEALTHY BASELINE MATCH
# ------------------------------------------------------------

def check_healthy_baseline(selected_event, evidence, df, baseline_df, results):
    """Recompute which baseline group SHOULD have been matched (same
    inverter_status / is_daylight / hour / month as the event's first
    flagged row) and diff its median/q10/q90 against
    evidence['healthy_baseline']['reference']."""
    cached = evidence.get("healthy_baseline")
    if not cached or "reference" not in cached:
        return

    start = pd.to_datetime(selected_event.get("start_time"))
    end = pd.to_datetime(selected_event.get("end_time"))
    if "anomaly_flag" in df.columns:
        event_rows = df[(df["anomaly_flag"]) & (df["timestamp"] >= start) & (df["timestamp"] <= end)]
    else:
        event_rows = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

    if event_rows.empty:
        return

    reference_row = event_rows.iloc[0]
    conditions = {c: _clean(reference_row.get(c)) for c in BASELINE_MATCH_COLS if c in df.columns}

    matched = baseline_df.copy()
    for col, val in conditions.items():
        if col in matched.columns and val is not None:
            matched = matched[matched[col].astype(str) == str(val)]

    if matched.empty:
        results.append({
            "rule": "healthy_baseline_match",
            "status": "WARN",
            "detail": f"Recomputing the baseline match for conditions {conditions} "
                      f"found no matching group -- cannot verify.",
        })
        return

    recomputed_count = int(matched["healthy_sample_count"].sum()) if "healthy_sample_count" in matched.columns else None
    cached_count = cached.get("healthy_sample_count")
    if recomputed_count is not None and cached_count is not None:
        if int(cached_count) == recomputed_count:
            results.append({
                "rule": "healthy_baseline_sample_count",
                "status": "PASS",
                "detail": f"healthy_sample_count={cached_count} matches recomputed {recomputed_count}.",
            })
        else:
            results.append({
                "rule": "healthy_baseline_sample_count",
                "status": "FAIL",
                "detail": f"evidence healthy_sample_count={cached_count}, recomputed "
                          f"from baseline_df is {recomputed_count} for conditions {conditions}.",
            })

    for metric, cached_ref in cached["reference"].items():
        median_col, q10_col, q90_col = f"{metric}_median", f"{metric}_q10", f"{metric}_q90"
        if median_col not in matched.columns:
            continue
        values = matched[[median_col, q10_col, q90_col]].dropna(how="all")
        if values.empty:
            continue
        recomputed_ref = {
            "median": _clean(values[median_col].iloc[0]),
            "typical_low_q10": _clean(values[q10_col].iloc[0]),
            "typical_high_q90": _clean(values[q90_col].iloc[0]),
        }
        for stat in ("median", "typical_low_q10", "typical_high_q90"):
            if stat not in cached_ref or recomputed_ref.get(stat) is None:
                continue
            if _close(cached_ref[stat], recomputed_ref[stat]):
                results.append({
                    "rule": f"baseline_ref[{metric}.{stat}]",
                    "status": "PASS",
                    "detail": f"evidence {stat}={cached_ref[stat]:.3f} matches recomputed "
                              f"{recomputed_ref[stat]:.3f}.",
                })
            else:
                results.append({
                    "rule": f"baseline_ref[{metric}.{stat}]",
                    "status": "FAIL",
                    "detail": f"evidence {stat}={cached_ref[stat]} does NOT match recomputed "
                              f"{recomputed_ref[stat]:.3f} for conditions {conditions}.",
                })


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------

def validate_event_data(evidence, df, trend_df=None, baseline_df=None):
    """Run all data-fetch checks for ONE specific anomaly event.

    evidence:  cached["evidence"] -- what was actually sent to the LLM for
               this event. The event's time window and anomaly_count are
               read directly from evidence["event"], NOT from any separate
               `selected_event` variable in the caller -- this guarantees
               the check always validates the exact event that generated
               this evidence, even if the caller's own selection state has
               since moved on to a different event.
    df:        full scored dashboard_data.parquet (with anomaly_flag).
    trend_df:  full trend_data.parquet (optional; skips trend checks if None).
    baseline_df: dashboard_baseline.parquet (optional; skips baseline checks if None).
    """
    ev = evidence.get("event", {})
    selected_event = pd.Series({
        "start_time": ev.get("start_time"),
        "end_time": ev.get("end_time"),
        "anomaly_count": ev.get("anomaly_count"),
    })

    results = []
    check_event_window(selected_event, evidence, df, results)
    check_event_statistics(selected_event, evidence, df, results)
    if trend_df is not None:
        check_pre_event_trend(selected_event, evidence, trend_df, results)
    if baseline_df is not None:
        check_healthy_baseline(selected_event, evidence, df, baseline_df, results)
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
