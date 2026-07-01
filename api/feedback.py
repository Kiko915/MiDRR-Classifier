"""SHAP-informed adaptive feedback generator.

How this is better than pure rule-based:
- Previously: "you scored LOW, and evacuation_time is globally important → here is a
  generic message about evacuation time."
- Now: "for THIS student, the model's SHAP values say decision_latency pushed the
  prediction toward LOW the most — and path_efficiency_ratio actually pushed it
  AWAY from LOW (a strength) — so we tell them specifically."

The SHAP value sign encodes direction:
  positive → that feature hurt the student (pushed toward a worse label)
  negative → that feature helped the student (pushed toward a better label)

This module uses that to:
  1. Generate a primary message about the top WEAKNESS (highest positive SHAP)
  2. Append a "bright spot" about the biggest STRENGTH (most negative SHAP)
     when the student didn't score HIGH — so the feedback is constructive,
     not just critical.
"""

from __future__ import annotations

# ── Primary feedback templates ────────────────────────────────────────────────
# Keyed by (predicted_level, feature_name).
# These describe the most influential feature for this student's prediction.

_TEMPLATES: dict[tuple[str, str], str] = {
    # HIGH — reinforce the specific behaviour that drove the good outcome
    ("HIGH", "evacuation_time"): (
        "Excellent work! You reached the assembly area quickly — your speed was the biggest factor in your HIGH rating."
    ),
    ("HIGH", "decision_latency"): (
        "Outstanding! Reacting immediately when you detected the hazard was your strongest behaviour."
    ),
    ("HIGH", "path_efficiency_ratio"): (
        "Great job! Taking a direct route to the assembly area was your top contribution to a HIGH rating."
    ),
    ("HIGH", "hazard_avoidance_ratio"): (
        "Well done! Consistently maintaining distance from the hazard was what drove your HIGH preparedness score."
    ),
    ("HIGH", "interaction_frequency"): (
        "Excellent! Actively engaging with safety procedures — alarms, exits, and assembly points — was your key strength."
    ),
    ("HIGH", "panic_proxy"): (
        "Great composure! Your calm, deliberate movement was the behaviour that most contributed to your HIGH rating."
    ),
    ("HIGH", "spray_accuracy"): (
        "Excellent technique! Your extinguisher sprays consistently landed on the fire — your PASS technique was a key strength."
    ),
    ("HIGH", "resource_utilization"): (
        "Well executed! You followed the correct safety sequence — pulling the pin before spraying, or taking sturdy cover promptly — precisely when it mattered."
    ),
    ("HIGH", "situational_awareness"): (
        "Outstanding overall awareness! You read the situation correctly and combined the right actions with a calm, efficient response."
    ),
    # MODERATE — acknowledge the leading issue, stay encouraging
    ("MODERATE", "evacuation_time"): (
        "You made it to safety, but your evacuation took longer than ideal. Practicing the route will build speed."
    ),
    ("MODERATE", "decision_latency"): (
        "You responded to the hazard, but the model detected a hesitation before your first action. Train yourself to react the moment you sense danger."
    ),
    ("MODERATE", "path_efficiency_ratio"): (
        "You reached the assembly area, but with some detours. Study the floor plan so the correct path becomes automatic."
    ),
    ("MODERATE", "hazard_avoidance_ratio"): (
        "You showed awareness, but spent more time near the hazard than was safe. Prioritise distance first, then evacuation."
    ),
    ("MODERATE", "interaction_frequency"): (
        "You used safety procedures, but not consistently enough. Engage earlier — activate alarms and notify others before you move."
    ),
    ("MODERATE", "panic_proxy"): (
        "Your movement was somewhat scattered under pressure. Repeated drills will help you move calmly and decisively."
    ),
    ("MODERATE", "spray_accuracy"): (
        "Some of your extinguisher sprays missed the fire. Practice aiming at the base of the flame and sweeping side to side."
    ),
    ("MODERATE", "resource_utilization"): (
        "Your safety sequencing was inconsistent — remember to pull the pin before spraying, or take cover fully before assessing the room. Drill the exact order until it's automatic."
    ),
    ("MODERATE", "situational_awareness"): (
        "You showed partial awareness of the situation, but combining the right action with a calmer, more direct response will raise your score."
    ),
    # LOW — direct and constructive, not discouraging
    ("LOW", "evacuation_time"): (
        "Your evacuation took too long. In a real emergency, every second costs lives — memorise the exit routes and walk them until they are automatic."
    ),
    ("LOW", "decision_latency"): (
        "You waited too long after detecting the hazard before taking action. Train yourself to move immediately — hesitation is the most dangerous behaviour in an emergency."
    ),
    ("LOW", "path_efficiency_ratio"): (
        "You took an inefficient route to the assembly area. Study the building's floor plan and physically walk the correct evacuation path until it is second nature."
    ),
    ("LOW", "hazard_avoidance_ratio"): (
        "You spent too much time near the hazard. Your first instinct must be to move away from danger — everything else comes after that."
    ),
    ("LOW", "interaction_frequency"): (
        "You did not engage with safety procedures during the scenario. Practice locating and using fire alarms, emergency exits, and assembly areas before the next drill."
    ),
    ("LOW", "panic_proxy"): (
        "Your movement showed signs of disorientation and panic. Breathing exercises and repeated scenario practice will help you stay focused under real pressure."
    ),
    ("LOW", "spray_accuracy"): (
        "Most of your extinguisher sprays missed the fire, or you fought the fire without pulling the pin first. Review the PASS technique (Pull, Aim, Squeeze, Sweep) before your next drill."
    ),
    ("LOW", "resource_utilization"): (
        "You did not follow the correct safety sequence — remember: pull the pin before you spray, and never fight a fire alone. During an earthquake, drop, cover, and hold immediately."
    ),
    ("LOW", "situational_awareness"): (
        "The model detected low overall awareness — a mix of delayed action, an inefficient route, and erratic movement. Focus on one BFP procedure at a time in your next drill."
    ),
}

# ── Bright-spot messages ──────────────────────────────────────────────────────
# Appended when a feature's SHAP value is strongly negative for a LOW/MODERATE
# student — meaning that feature actually helped them despite the overall result.

_BRIGHT_SPOT: dict[str, str] = {
    "evacuation_time": "One strength to build on: your evacuation speed was not your main problem.",
    "decision_latency": "One strength to build on: your initial reaction time was not what held you back.",
    "path_efficiency_ratio": "One strength to build on: your evacuation route was relatively direct.",
    "hazard_avoidance_ratio": "One strength to build on: you generally maintained a safe distance from the hazard.",
    "interaction_frequency": "One strength to build on: your engagement with safety procedures was above the critical threshold.",
    "panic_proxy": "One strength to build on: your movement was relatively calm — that composure will help you in real drills.",
    "spray_accuracy": "One strength to build on: your extinguisher technique (or Drop-Cover-Hold) was reasonably accurate.",
    "resource_utilization": "One strength to build on: your safety sequencing (pull-before-spray, or cover discipline) was not the issue.",
    "situational_awareness": "One strength to build on: your overall read of the situation was better than the final score suggests.",
}

_FALLBACK: dict[str, str] = {
    "HIGH": "Excellent performance! You demonstrated strong disaster preparedness across all behaviours.",
    "MODERATE": "Good effort. Identify your weakest behaviour and target it in your next drill.",
    "LOW": "Your performance shows significant room for improvement. Regular drills focused on the steps above will make a real difference.",
}

# How negative does a SHAP value need to be to count as a meaningful "bright spot"?
# This is a soft threshold — tune it once you have real data.
_BRIGHT_SPOT_THRESHOLD = -0.05

# ── BFP-revised design's numeric preparedness thresholds ──────────────────────
# These are fixed, rule-based cutoffs from the 2026-07-01 simulation-design
# diagrams — independent of the model's SHAP attribution. A concerning value
# is always surfaced even if it wasn't the top-ranked SHAP feature for this
# particular student (SHAP only ranks features RELATIVE to each other; a
# genuinely bad decision_latency can still lose the "top" spot to an even
# worse feature). Checked in this fixed priority order.
_CRITICAL_THRESHOLDS: list[tuple[str, str, float]] = [
    # (feature, comparison, threshold) — comparison is ">" (bad if above) or "<" (bad if below)
    ("decision_latency", ">", 30.0),
    ("spray_accuracy", "<", 0.40),
    ("path_efficiency_ratio", "<", 0.50),
    ("panic_proxy", ">", 2.0),
]

_THRESHOLD_NOTES: dict[str, str] = {
    "decision_latency": (
        "Flag: your reaction time was over 30 seconds — the BFP-flagged cutoff for a slow response."
    ),
    "spray_accuracy": (
        "Flag: fewer than 40% of your extinguisher sprays (or Drop-Cover-Hold attempts) were on-target."
    ),
    "path_efficiency_ratio": (
        "Flag: your evacuation route fell below half of the ideal directness — review the designated exit path."
    ),
    "panic_proxy": (
        "Flag: your movement speed was highly inconsistent — a sign of panic the model flags independently."
    ),
}


def check_thresholds(features: dict[str, float]) -> dict[str, str]:
    """Flag features that cross the diagram's fixed numeric thresholds.

    Rule-based and independent of SHAP — a feature can cross its critical
    threshold without being the top |SHAP| feature (SHAP only ranks features
    relative to each other for this one prediction). Only features present
    in *features* are checked.

    Args:
        features: ``{feature_name: value}`` for the student's session.

    Returns:
        ``{feature_name: note}`` for every feature that crossed its threshold,
        in the fixed priority order of ``_CRITICAL_THRESHOLDS``.
    """
    flagged: dict[str, str] = {}
    for feat, op, threshold in _CRITICAL_THRESHOLDS:
        if feat not in features:
            continue
        value = features[feat]
        crossed = value > threshold if op == ">" else value < threshold
        if crossed:
            flagged[feat] = _THRESHOLD_NOTES[feat]
    return flagged


def generate_result_text(
    level: str,
    top_feature: str,
    top_shap_value: float = 0.0,
    all_shap_values: dict[str, float] | None = None,
    features: dict[str, float] | None = None,
) -> str:
    """Build a personalised feedback string from SHAP attribution scores.

    Args:
        level: Predicted preparedness level — ``"HIGH"``, ``"MODERATE"``, or ``"LOW"``.
        top_feature: Feature with the highest absolute SHAP value for this student.
        top_shap_value: The SHAP value of ``top_feature`` (can be negative).
        all_shap_values: Full ``{feature: shap_value}`` dict for this student.
            When provided, a "bright spot" sentence is appended for LOW/MODERATE
            students if another feature has a meaningfully negative SHAP value.
        features: Full ``{feature: value}`` dict for this student. When
            provided, at most one rule-based threshold flag (see
            :func:`check_thresholds`) is appended for a feature other than
            *top_feature* — surfacing a BFP-critical value even when it
            wasn't the model's single most-influential feature.

    Returns:
        One to three sentences of personalised, constructive feedback.
    """
    main = _TEMPLATES.get((level, top_feature), _FALLBACK.get(level, ""))

    # For LOW/MODERATE students, find if there is a feature with a strongly
    # negative SHAP value (meaning: the model says this feature was a strength
    # for this student, even though the overall prediction was bad).
    if level in ("LOW", "MODERATE") and all_shap_values:
        # Find the feature with the most negative SHAP (biggest relative strength).
        bright_feat, bright_val = min(all_shap_values.items(), key=lambda kv: kv[1])
        if bright_val < _BRIGHT_SPOT_THRESHOLD and bright_feat != top_feature:
            bright = _BRIGHT_SPOT.get(bright_feat, "")
            if bright:
                main = f"{main} {bright}"

    if features:
        for feat, note in check_thresholds(features).items():
            if feat != top_feature:
                main = f"{main} {note}"
                break  # one flag is enough to keep the message readable

    return main
