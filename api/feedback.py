"""Rule-based adaptive feedback generator (Phase 8 starter).

Given a predicted preparedness level and the top-contributing feature,
this module returns a human-readable message for the student.

The logic is intentionally simple: a lookup table of (level, feature) →
message.  In Phase 6, this will be replaced with per-session SHAP values
so the top feature is the one that most influenced *this specific student's*
prediction, not just the globally most important one.
"""

from __future__ import annotations

# Human-readable labels for each feature name (for use in feedback text)
_FEATURE_LABELS: dict[str, str] = {
    "evacuation_time": "evacuation speed",
    "decision_delay": "reaction time",
    "path_efficiency_ratio": "route efficiency",
    "hazard_avoidance_ratio": "hazard avoidance",
    "interaction_frequency": "safety procedure engagement",
    "panic_proxy": "movement calmness",
}

# (level, top_feature) → feedback sentence
# Written to be direct, actionable, and appropriate for a student audience.
_TEMPLATES: dict[tuple[str, str], str] = {
    # HIGH — reinforce what they did well
    ("HIGH", "evacuation_time"): (
        "Excellent work! You reached the assembly area quickly, which is the most critical skill in a real emergency."
    ),
    ("HIGH", "decision_delay"): (
        "Outstanding! Your immediate reaction upon detecting the hazard shows strong situational awareness."
    ),
    ("HIGH", "path_efficiency_ratio"): (
        "Great job! You followed a direct evacuation route, minimizing your exposure time."
    ),
    ("HIGH", "hazard_avoidance_ratio"): (
        "Well done! You consistently maintained a safe distance from the hazard throughout the scenario."
    ),
    ("HIGH", "interaction_frequency"): (
        "Excellent! You actively engaged with safety procedures — activating alarms and using exits correctly."
    ),
    ("HIGH", "panic_proxy"): (
        "Great composure! You moved calmly and deliberately, which helps everyone around you stay safe too."
    ),
    # MODERATE — acknowledge effort, point to one improvement
    ("MODERATE", "evacuation_time"): (
        "Good effort. Your evacuation time was within range, but practicing the route will help you move faster next time."
    ),
    ("MODERATE", "decision_delay"): (
        "You're on the right track, but try to act the moment you first sense the hazard — hesitation costs critical seconds."
    ),
    ("MODERATE", "path_efficiency_ratio"): (
        "Not bad! You reached safety, but your path had some detours. Study the floor plan so you can take a more direct route."
    ),
    ("MODERATE", "hazard_avoidance_ratio"): (
        "You showed awareness of the hazard, but you were still too close at times. Make 'move away first' your instinct."
    ),
    ("MODERATE", "interaction_frequency"): (
        "You used some safety procedures correctly. Try to engage more — activate alarms and notify others earlier."
    ),
    ("MODERATE", "panic_proxy"): (
        "Your movement was a bit scattered. Practice the scenario a few more times to build muscle memory and stay calm."
    ),
    # LOW — direct and constructive, not discouraging
    ("LOW", "evacuation_time"): (
        "Your evacuation took too long. In a real emergency, every second matters — memorize the exit routes and practice them."
    ),
    ("LOW", "decision_delay"): (
        "You waited too long after detecting the hazard before taking action. Drill yourself to react immediately — act first, think on the way."
    ),
    ("LOW", "path_efficiency_ratio"): (
        "You took an inefficient route to the assembly area. Study the building's floor plan and walk the correct path until it becomes automatic."
    ),
    ("LOW", "hazard_avoidance_ratio"): (
        "You spent too much time near the hazard. Your top priority in any emergency is to move away from danger before doing anything else."
    ),
    ("LOW", "interaction_frequency"): (
        "You did not engage enough with safety procedures. Practice locating and using fire alarms, exits, and assembly areas before the real scenario."
    ),
    ("LOW", "panic_proxy"): (
        "Your movement showed signs of panic and disorientation. Repeated drills and breathing practice can help you stay focused under pressure."
    ),
}

# Fallback messages when no specific template matches
_FALLBACK: dict[str, str] = {
    "HIGH": "Excellent performance! You demonstrated strong disaster preparedness.",
    "MODERATE": "Good effort! There is room to improve — keep practicing the evacuation procedures.",
    "LOW": "Your performance shows areas that need significant improvement. Regular drills will help build confidence and speed.",
}


def generate_result_text(level: str, top_feature: str) -> str:
    """Return an adaptive feedback string for a student.

    Args:
        level: Predicted preparedness level — ``"HIGH"``, ``"MODERATE"``, or ``"LOW"``.
        top_feature: The feature name with the highest importance weight for this
            prediction (used to personalise the message).

    Returns:
        A single feedback sentence appropriate for the student's level and
        most influential behavioral feature.
    """
    return _TEMPLATES.get((level, top_feature), _FALLBACK.get(level, ""))
