"""Rule-based preparedness labeling (Phase 2.5, BFP-revised design).

The revised BERONG SMP simulation runs its own 3-phase fire state machine
(Prevention -> Intervention -> Evacuation) and computes a numeric
``simulation_score`` (0-100), which it maps to its own rule-based
``prep_level``. That game-computed label is now a real input to this
pipeline, not just an ML output — see ``docs/telemetry_contract.md`` §3b/§5.

This module is the ML side's mirror of that scoring rule, plus the label
**provenance policy** locked in the approved plan:

- **Expert labels are gold.** A BFP-instructor override (from the
  validation UI) always wins, tagged ``label_source="expert"``.
- **Rule-based labels are weak labels for scale**, tagged
  ``label_source="rule"``. They exist to grow the training set beyond what
  hand-scoring can cover.
- **The test split must be expert-only.** This is the circularity guard
  from ``docs/labeling_rubric.md`` §7 — never report accuracy against a
  rule-labeled test set, since the rule and the model would just be
  agreeing with each other's arithmetic.
- Rule labels are only trustworthy if they agree well with the expert gold
  subset — see :func:`rule_expert_agreement` for the validation check
  ``labeling_rubric.md`` §7 requires before trusting them at scale.
"""

from __future__ import annotations

import pandas as pd

from midrr_classifier.data_schema import LABEL_CLASSES

# ---------------------------------------------------------------------------
# Score -> label thresholds, locked to the BFP-revised simulation design
# (2026-07-01 diagrams). The game's own state machine uses these same cuts.
# ---------------------------------------------------------------------------

HIGH_THRESHOLD: float = 75.0
MODERATE_THRESHOLD: float = 40.0

# Fire state machine phases, in the order a run progresses through them.
# Mirrors data_schema.SIM_PHASES.
_PHASE_ORDER: list[str] = ["prevention", "intervention", "evacuation"]


def rule_based_label(score: float) -> str:
    """Map the game's numeric ``simulation_score`` (0-100) to a label.

    ``>= 75`` -> HIGH, ``40-74`` -> MODERATE, ``< 40`` -> LOW. This mirrors
    the simulation's own rule-based ``prep_level`` computation (see
    ``telemetry_contract.md`` §5); it exists on the ML side so the pipeline
    can independently reproduce/validate the game's scoring, not just trust
    it blindly.

    Args:
        score: Numeric preparedness score in ``[0, 100]``.

    Returns:
        One of :data:`~midrr_classifier.data_schema.LABEL_CLASSES`.
    """
    if score >= HIGH_THRESHOLD:
        return "HIGH"
    if score >= MODERATE_THRESHOLD:
        return "MODERATE"
    return "LOW"


def phase_outcome_label(phase_reached: str, evacuated_safely: bool) -> str:
    """Coarse label derived from which fire-sim phase a run reached.

    This is a **cross-check** against :func:`rule_based_label`, not a
    primary label source — the numeric score is. Use it to sanity-check
    that the phase the state machine reached and the score it produced
    tell the same story (large disagreements are worth investigating).

    - ``prevention`` (fire never escalated past the preventable stage) -> HIGH
    - ``intervention`` (player had to actively respond) -> MODERATE
    - ``evacuation`` -> MODERATE if ``evacuated_safely`` else LOW
    - anything else (unrecognized phase, run abandoned) -> LOW

    Args:
        phase_reached: Last ``phase`` value from a ``phase_transition``
            event (one of :data:`~midrr_classifier.data_schema.SIM_PHASES`).
        evacuated_safely: Whether ``assembly_area_reached`` was observed.

    Returns:
        One of :data:`~midrr_classifier.data_schema.LABEL_CLASSES`.
    """
    if phase_reached == "prevention":
        return "HIGH"
    if phase_reached == "intervention":
        return "MODERATE"
    if phase_reached == "evacuation":
        return "MODERATE" if evacuated_safely else "LOW"
    return "LOW"


def resolve_label(
    expert_label: str | None,
    rule_score: float | None = None,
    rule_label: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve the single ``(preparedness_level, label_source)`` pair to use.

    Precedence: **expert override > pre-computed rule label > rule score**.
    A session with no label information at all resolves to ``(None, None)``
    — callers should drop such rows before training, not silently zero-fill.

    Args:
        expert_label: BFP-instructor override, if any (already HIGH/
            MODERATE/LOW). ``None``/empty if no override was recorded.
        rule_score: The game's numeric ``simulation_score``, if available.
        rule_label: The game's own precomputed ``prep_level``, if available
            (preferred over deriving one from ``rule_score`` — it reflects
            whatever exact rule the game used, including any state-machine
            logic beyond the numeric threshold).

    Returns:
        ``(label, label_source)`` where ``label_source`` is ``"expert"``,
        ``"rule"``, or ``None``.
    """
    if expert_label:
        return expert_label, "expert"
    if rule_label:
        return rule_label, "rule"
    if rule_score is not None:
        return rule_based_label(rule_score), "rule"
    return None, None


def attach_labels(
    df: pd.DataFrame,
    expert_col: str = "expert_label",
    rule_label_col: str = "prep_level",
    rule_score_col: str = "simulation_score",
) -> pd.DataFrame:
    """Vectorized :func:`resolve_label` over a session-level DataFrame.

    Adds/overwrites ``preparedness_level`` and ``label_source`` columns.
    Any of the three source columns may be absent — this is the normal case
    for CSV batches that only carry a rule score, or Turso rows that always
    carry ``prep_level`` but only sometimes carry an expert override.

    This is the label-resolution step the ingestion adapter (Phase 2.5
    step 4) calls for every session before it reaches
    :func:`~midrr_classifier.feature_engineering.build_feature_table`.

    Args:
        df: Session-level (or raw-log) DataFrame, one row per session or
            one row per event with session-level columns broadcast.
        expert_col: Column holding the BFP-instructor override label.
        rule_label_col: Column holding the game's precomputed ``prep_level``.
        rule_score_col: Column holding the game's numeric ``simulation_score``.

    Returns:
        A copy of *df* with ``preparedness_level`` and ``label_source`` set.
    """
    df = df.copy()
    n = len(df)
    expert = df[expert_col] if expert_col in df.columns else pd.Series([None] * n, index=df.index)
    rule_label = df[rule_label_col] if rule_label_col in df.columns else pd.Series([None] * n, index=df.index)
    rule_score = df[rule_score_col] if rule_score_col in df.columns else pd.Series([None] * n, index=df.index)

    resolved = [
        resolve_label(e, s, r) for e, s, r in zip(expert, rule_score, rule_label)
    ]
    df["preparedness_level"] = [label for label, _source in resolved]
    df["label_source"] = [source for _label, source in resolved]
    return df


def rule_expert_agreement(
    labels_df: pd.DataFrame,
    rule_col: str = "rule_label",
    expert_col: str = "expert_label",
) -> dict:
    """Cohen's kappa between rule-based and expert labels, on the overlap.

    Implements the validation ``docs/labeling_rubric.md`` §7 requires:
    rule-based weak labels are only trustworthy at scale if they agree well
    with the expert gold subset. Sessions missing either label are dropped
    before computing agreement (kappa needs a paired sample).

    Args:
        labels_df: DataFrame with one row per session carrying both a rule
            label and an expert label (typically the gold subset that has
            both, e.g. from a pilot where every run was also hand-scored).
        rule_col: Column with the rule-based label.
        expert_col: Column with the expert label.

    Returns:
        ``{"kappa": float, "n": int, "agreement_rate": float}``.

    Raises:
        ValueError: If no rows have both a rule and an expert label.
    """
    paired = labels_df[[rule_col, expert_col]].dropna()
    if paired.empty:
        raise ValueError("No overlapping rule+expert labels to compute agreement on.")

    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(paired[expert_col], paired[rule_col], labels=LABEL_CLASSES)
    agreement_rate = float((paired[rule_col] == paired[expert_col]).mean())
    return {"kappa": float(kappa), "n": int(len(paired)), "agreement_rate": agreement_rate}
