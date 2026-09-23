"""
Rule-Based Validation for AI Explanations
==========================================
Checks the LLM-generated explanation text against the structured `evidence`
dict that was actually fed to the model (see generate_ai_explanation() in
app.py). Every rule here is deterministic: no second LLM call, no judgment
calls -- just explicit pass/fail/warn conditions.

Usage:
    from validation import validate_explanation
    results = validate_explanation(explanation_text, evidence)
    # results is a list of dicts: {"rule", "status", "detail"}

Statuses:
    PASS  -> rule satisfied
    FAIL  -> rule violated (numeric mismatch, banned term found, etc.)
    WARN  -> could not fully verify (e.g. evidence missing for that field)
    INFO  -> structural/informational, not a strict correctness check
"""

import re

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

# Maps phrases that might appear in the explanation -> evidence field name,
# display unit, and a regex fragment for the unit as it might actually be
# written (used to anchor the number search so we don't grab a stray digit
# from a neighbouring clause). unit_pattern=None means "just take the
# nearest following number" (used for unitless fields like power factor).
FEATURE_KEYWORDS = [
    ("ambient temperature", "ambient_temperature_c", "°C", r"°?\s?C\b"),
    ("inverter temperature", "inverter_temperature_c", "°C", r"°?\s?C\b"),
    ("dc power", "dc_power_kw", "kW", r"kW\b"),
    ("ac power", "ac_power_kw", "kW", r"kW\b"),
    ("dc current", "dc_current_a", "A", r"\bA\b"),
    ("ac current", "ac_current_a", "A", r"\bA\b"),
    ("power factor", "power_factor", "", None),
    ("frequency", "frequency_hz", "Hz", r"Hz\b"),
    ("efficiency", "efficiency_pct", "%", r"%"),
]

BANNED_TERMS = [
    "autoencoder", "reconstruction error", "threshold", "anomaly score",
    "probability", "confidence", "z-score", "neural network", "model score",
]

CAUSAL_PHRASES = [
    "caused by", "because of", "due to", "results from", "resulted from",
    "was caused", "led to this",
]

RANGE_LABELS = {
    "upper end": "high",
    "top end": "high",
    "near the maximum": "high",
    "lower end": "low",
    "bottom end": "low",
    "near the minimum": "low",
}

NUMBER_WINDOW = 80      # chars to search around a keyword for a number
NUMERIC_TOLERANCE = 0.05  # 5% slack for rounding ("around X")
MAX_WORDS = 120          # soft ceiling from the prompt's "~100 words" rule


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def _find_number_near(text, idx, window=NUMBER_WINDOW):
    """Look for a number in a window of text around index idx (either
    direction). Used only where no unit is available to anchor on
    (e.g. range-label phrases, power factor)."""
    start = max(0, idx - window)
    end = min(len(text), idx + window)
    snippet = text[start:end]
    match = re.search(r"-?\d+\.?\d*", snippet)
    return float(match.group()) if match else None


def _find_number_with_unit(text, from_idx, unit_pattern, window=NUMBER_WINDOW):
    """Search FORWARD from from_idx for 'number <unit>', anchoring on the
    unit so we don't accidentally grab a number from a different clause
    (e.g. the temperature value while checking DC power). Falls back to
    the nearest forward number if no unit-anchored match is found."""
    snippet = text[from_idx: from_idx + window]
    if unit_pattern:
        m = re.search(r"(-?\d+\.?\d*)\s*\(?\s*" + unit_pattern, snippet)
        if m:
            return float(m.group(1))
    m = re.search(r"-?\d+\.?\d*", snippet)
    return float(m.group()) if m else None


def _get_stat_bounds(evidence, field):
    """Pull (min, max) for a field from whichever evidence section has it."""
    for section in ("event_observations", "pre_event_trend"):
        stats = evidence.get(section, {}).get("statistics", {})
        if field in stats:
            s = stats[field]
            if s.get("min") is not None and s.get("max") is not None:
                return s["min"], s["max"]
    return None


def _get_baseline_quantiles(evidence, field):
    """Pull (q10, median, q90) for a field from the healthy baseline."""
    ref = evidence.get("healthy_baseline", {}).get("reference", {})
    if field in ref:
        r = ref[field]
        return r.get("typical_low_q10"), r.get("median"), r.get("typical_high_q90")
    return None


# ------------------------------------------------------------
# RULES
# ------------------------------------------------------------

def rule_numeric_claims(text, evidence, results):
    """Every number attached to a known feature keyword must fall within
    that feature's observed min/max for the event (with small tolerance)."""
    for phrase, field, unit, unit_pattern in FEATURE_KEYWORDS:
        for m in re.finditer(re.escape(phrase), text, flags=re.I):
            value = _find_number_with_unit(text, m.end(), unit_pattern)
            if value is None:
                results.append({
                    "rule": f"numeric_claim[{field}]",
                    "status": "WARN",
                    "detail": f"'{phrase}' is mentioned but no number/unit could be "
                              f"matched nearby -- rule could not check this claim.",
                })
                continue
            bounds = _get_stat_bounds(evidence, field)
            if bounds is None:
                results.append({
                    "rule": f"numeric_claim[{field}]",
                    "status": "WARN",
                    "detail": f"Claimed {value}{unit} for '{phrase}', but no "
                              f"evidence stats exist for {field} to check against.",
                })
                continue
            lo, hi = bounds
            slack = (hi - lo) * NUMERIC_TOLERANCE if hi != lo else abs(hi) * NUMERIC_TOLERANCE
            if lo - slack <= value <= hi + slack:
                results.append({
                    "rule": f"numeric_claim[{field}]",
                    "status": "PASS",
                    "detail": f"Claimed {value}{unit} for '{phrase}' is within "
                              f"observed range [{lo:.2f}, {hi:.2f}].",
                })
            else:
                results.append({
                    "rule": f"numeric_claim[{field}]",
                    "status": "FAIL",
                    "detail": f"Claimed {value}{unit} for '{phrase}' is OUTSIDE "
                              f"observed range [{lo:.2f}, {hi:.2f}].",
                })


def rule_duration(text, evidence, results):
    """A number followed by 'minute(s)' should match the event duration."""
    duration = evidence.get("event", {}).get("duration_min")
    if duration is None:
        return
    for m in re.finditer(r"(\d+\.?\d*)\s*minute", text, flags=re.I):
        claimed = float(m.group(1))
        if abs(claimed - duration) <= 1:  # allow 1-minute rounding
            results.append({
                "rule": "duration_claim",
                "status": "PASS",
                "detail": f"Claimed duration {claimed} min matches event duration {duration:.1f} min.",
            })
        else:
            results.append({
                "rule": "duration_claim",
                "status": "FAIL",
                "detail": f"Claimed duration {claimed} min does not match event duration {duration:.1f} min.",
            })


def rule_time_window(text, evidence, results):
    """Times mentioned (HH:MM) should fall within [start_time, end_time]."""
    start = evidence.get("event", {}).get("start_time")
    end = evidence.get("event", {}).get("end_time")
    times_found = re.findall(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if not times_found or start is None or end is None:
        return
    try:
        start_hm = (int(start[11:13]), int(start[14:16]))
        end_hm = (int(end[11:13]), int(end[14:16]))
    except Exception:
        results.append({
            "rule": "time_window",
            "status": "WARN",
            "detail": "Could not parse event start/end timestamps for comparison.",
        })
        return
    for h, mnt in times_found:
        t = (int(h), int(mnt))
        if start_hm <= t <= end_hm:
            results.append({
                "rule": "time_window",
                "status": "PASS",
                "detail": f"Claimed time {h}:{mnt} falls within event window {start} - {end}.",
            })
        else:
            results.append({
                "rule": "time_window",
                "status": "FAIL",
                "detail": f"Claimed time {h}:{mnt} falls OUTSIDE event window {start} - {end}.",
            })


def rule_range_label(text, evidence, results):
    """Phrases like 'near the upper end of the healthy range' must be
    consistent with where the claimed value sits vs. baseline quantiles."""
    for phrase, direction in RANGE_LABELS.items():
        if phrase.lower() not in text.lower():
            continue
        # Find the nearest preceding feature keyword to know which field this refers to.
        phrase_idx = text.lower().find(phrase.lower())
        preceding = text[:phrase_idx].lower()
        matched_field = None
        matched_kw_idx = -1
        for kw, field, unit, unit_pattern in FEATURE_KEYWORDS:
            kw_idx = preceding.rfind(kw)
            if kw_idx > matched_kw_idx:  # keep the CLOSEST preceding keyword
                matched_kw_idx = kw_idx
                matched_field = (field, unit, unit_pattern, kw_idx)
        if not matched_field:
            results.append({
                "rule": f"range_label[{phrase}]",
                "status": "WARN",
                "detail": f"Found range phrase '{phrase}' but could not tell which "
                          f"parameter it refers to.",
            })
            continue
        field, unit, unit_pattern, kw_idx = matched_field
        # Search forward from the matched keyword (not the phrase) so we stay
        # anchored to the right unit, e.g. "...DC power was near the upper
        # end... (approx 2.3 kW)" -> anchor on "DC power", not "upper end".
        value = _find_number_with_unit(text, kw_idx, unit_pattern, window=100)
        quantiles = _get_baseline_quantiles(evidence, field)
        if value is None or quantiles is None or None in quantiles:
            results.append({
                "rule": f"range_label[{phrase}]",
                "status": "WARN",
                "detail": f"Cannot verify '{phrase}' claim for {field} -- missing "
                          f"value or baseline quantiles.",
            })
            continue
        q10, median, q90 = quantiles
        is_high = value >= median
        claim_ok = (direction == "high" and is_high) or (direction == "low" and not is_high)
        results.append({
            "rule": f"range_label[{phrase}]",
            "status": "PASS" if claim_ok else "FAIL",
            "detail": f"'{phrase}' for {field}={value}{unit}: baseline q10={q10:.2f}, "
                      f"median={median:.2f}, q90={q90:.2f}.",
        })


def rule_banned_terms(text, results):
    lowered = text.lower()
    hits = [term for term in BANNED_TERMS if term in lowered]
    if hits:
        results.append({
            "rule": "banned_ml_terms",
            "status": "FAIL",
            "detail": f"Explanation leaks internal ML terminology: {hits}",
        })
    else:
        results.append({
            "rule": "banned_ml_terms",
            "status": "PASS",
            "detail": "No internal ML terminology found.",
        })


def rule_unsupported_causality(text, evidence, results):
    """Flag causal language unless it's explicitly hedged (e.g. 'does not
    confirm the cause') or the named feature is a top contributor."""
    top_features = {
        c["feature"].lower()
        for c in evidence.get("feature_contributions", [])
    }
    lowered = text.lower()
    for phrase in CAUSAL_PHRASES:
        for m in re.finditer(re.escape(phrase), lowered):
            window = lowered[max(0, m.start() - 60): m.end() + 60]
            hedged = any(neg in window for neg in ["cannot", "does not", "can't", "not confirm", "no clear cause"])
            if hedged:
                continue
            supported = any(feat in window for feat in top_features) if top_features else False
            results.append({
                "rule": "unsupported_causality",
                "status": "PASS" if supported else "FAIL",
                "detail": f"Causal phrase '{phrase}' found near: \"...{window.strip()}...\" "
                          + ("(matches a top contributing feature)" if supported
                             else "(not clearly tied to a listed contributing feature)"),
            })


def rule_structure(text, results):
    """Checks the formatting rules the prompt itself imposes: one paragraph,
    no bullets/headings, roughly under ~100-120 words."""
    word_count = len(text.split())
    if word_count > MAX_WORDS:
        results.append({
            "rule": "word_count",
            "status": "WARN",
            "detail": f"{word_count} words, exceeds the ~{MAX_WORDS}-word soft limit.",
        })
    else:
        results.append({
            "rule": "word_count",
            "status": "PASS",
            "detail": f"{word_count} words, within limit.",
        })

    if re.search(r"^\s*[-*•]\s", text, flags=re.M) or "\n\n" in text.strip():
        results.append({
            "rule": "single_paragraph",
            "status": "FAIL",
            "detail": "Explanation contains bullet points or multiple paragraphs.",
        })
    else:
        results.append({
            "rule": "single_paragraph",
            "status": "PASS",
            "detail": "Explanation is a single paragraph with no bullets.",
        })


# ------------------------------------------------------------
# MAIN ENTRY POINT
# ------------------------------------------------------------

def validate_explanation(explanation_text, evidence):
    """Run every rule and return a flat list of {rule, status, detail} dicts."""
    results = []
    rule_numeric_claims(explanation_text, evidence, results)
    rule_duration(explanation_text, evidence, results)
    rule_time_window(explanation_text, evidence, results)
    rule_range_label(explanation_text, evidence, results)
    rule_banned_terms(explanation_text, results)
    rule_unsupported_causality(explanation_text, evidence, results)
    rule_structure(explanation_text, results)
    return results


def summarize(results):
    """Roll results up into an overall verdict for quick display."""
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
