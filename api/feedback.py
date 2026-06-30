"""SHAP-informed adaptive feedback generator.

How this is better than pure rule-based:
- Previously: "you scored LOW, and evacuation_time is globally important → here is a
  generic message about evacuation time."
- Now: "for THIS student, the model's SHAP values say decision_delay pushed the
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
    ("HIGH", "decision_delay"): (
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
    # MODERATE — acknowledge the leading issue, stay encouraging
    ("MODERATE", "evacuation_time"): (
        "You made it to safety, but your evacuation took longer than ideal. Practicing the route will build speed."
    ),
    ("MODERATE", "decision_delay"): (
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
    # LOW — direct and constructive, not discouraging
    ("LOW", "evacuation_time"): (
        "Your evacuation took too long. In a real emergency, every second costs lives — memorise the exit routes and walk them until they are automatic."
    ),
    ("LOW", "decision_delay"): (
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
}

# ── Bright-spot messages ──────────────────────────────────────────────────────
# Appended when a feature's SHAP value is strongly negative for a LOW/MODERATE
# student — meaning that feature actually helped them despite the overall result.

_BRIGHT_SPOT: dict[str, str] = {
    "evacuation_time": "One strength to build on: your evacuation speed was not your main problem.",
    "decision_delay": "One strength to build on: your initial reaction time was not what held you back.",
    "path_efficiency_ratio": "One strength to build on: your evacuation route was relatively direct.",
    "hazard_avoidance_ratio": "One strength to build on: you generally maintained a safe distance from the hazard.",
    "interaction_frequency": "One strength to build on: your engagement with safety procedures was above the critical threshold.",
    "panic_proxy": "One strength to build on: your movement was relatively calm — that composure will help you in real drills.",
}

_FALLBACK: dict[str, str] = {
    "HIGH": "Excellent performance! You demonstrated strong disaster preparedness across all behaviours.",
    "MODERATE": "Good effort. Identify your weakest behaviour and target it in your next drill.",
    "LOW": "Your performance shows significant room for improvement. Regular drills focused on the steps above will make a real difference.",
}

# How negative does a SHAP value need to be to count as a meaningful "bright spot"?
# This is a soft threshold — tune it once you have real data.
_BRIGHT_SPOT_THRESHOLD = -0.05


def generate_result_text(
    level: str,
    top_feature: str,
    top_shap_value: float = 0.0,
    all_shap_values: dict[str, float] | None = None,
) -> str:
    """Build a personalised feedback string from SHAP attribution scores.

    Args:
        level: Predicted preparedness level — ``"HIGH"``, ``"MODERATE"``, or ``"LOW"``.
        top_feature: Feature with the highest absolute SHAP value for this student.
        top_shap_value: The SHAP value of ``top_feature`` (can be negative).
        all_shap_values: Full ``{feature: shap_value}`` dict for this student.
            When provided, a "bright spot" sentence is appended for LOW/MODERATE
            students if another feature has a meaningfully negative SHAP value.

    Returns:
        One or two sentences of personalised, constructive feedback.
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
                return f"{main} {bright}"

    return main
