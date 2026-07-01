# BERONG SMP — Preparedness Labeling Rubric & Protocol

**Version:** 1.1 (draft)
**Owner:** AI/ML Engineer (MiDRR-Classifier)
**Raters:** BFP officers + teachers (per Ch3 and the system architecture's BFP validation loop)
**Grounding source:** LSPU Sta. Cruz Administration Building Evacuation Plan (BFP-validated) + standard PHIVOLCS/DRRMO earthquake guidance.

This document defines how each simulation run gets its `preparedness_level` label (**High / Moderate / Low**) — the `y` the Random Forest learns to predict. **The gold label still comes from expert judgment scored against BFP-validated procedures — not from the game's own arithmetic or from survey scores.** What changed in v1.1 is that the game *itself* now also computes a rule-based `prep_level` inside its 3-phase state machine (§7), and that rule-based label is now used as a **weak label for scale**, not as the gold standard.

### Changes since v1.0 (driven by the 2026-07-01 BFP consultation / revised simulation design)
- **The game now computes its own numeric score and rule-based label** (`simulation_score` → `prep_level` via thresholds ≥75 HIGH / 40–74 MODERATE / <40 LOW), written to the Turso `sessions` table alongside a `confidence` value. See `docs/telemetry_contract.md` §3b/§5.
- **`label_source` column** (`expert` / `rule`) now travels with every labeled row, implemented by `src/midrr_classifier/labeling.py` (`rule_based_label()`, `phase_outcome_label()`, `resolve_label()`/`attach_labels()`). Precedence is **expert override > rule label > rule score** — see §7 below (unchanged principle, now with code behind it).
- **`rule_expert_agreement()`** (in `labeling.py`) implements the §7 validation requirement in code: compute Cohen's κ between the game's rule labels and the expert gold subset before trusting rule labels at scale.
- **The circularity guard is unchanged and still binding: the test split must be expert-only** (`label_source == "expert"`), enforced by the ingestion adapter (`data_ingestion.resolve_session_labels()`), not just a paper policy.

---

## 1. Why this rubric exists (and the one trap to avoid)

The model predicts preparedness *level*. That target must be defined by something independent of the model's own arithmetic, or the result is circular.

> **Label-leakage / circularity warning.** If raters score a run by eyeballing the *same quantities* the model uses as features (e.g. "was the path efficient?" = `path_efficiency_ratio`), the Random Forest just re-derives your scoring formula, and high accuracy proves nothing. **Defensible framing:** the rubric captures **holistic expert judgment** of disaster-response competency — including *appropriateness* and *correctness* of actions in context, which raw features can't see (e.g. fighting a fire while alone is *wrong* even though it's an "interaction"). The features are cheap, automatically-extracted **proxies**. The research question becomes: *can automatic behavioral proxies recover expert preparedness judgment?* That is a real, defensible finding.

So: rate **behavior quality and appropriateness**, not feature values.

---

## 2. Label scheme (locked to the manuscript)

| Level | Meaning |
|---|---|
| **High** | Responded correctly, calmly, and in time; followed BFP procedure; evacuated safely to assembly area. |
| **Moderate** | Mostly correct with notable lapses (e.g. delayed, missed a step, inefficient but safe evacuation). |
| **Low** | Unsafe, incorrect, panicked, or failed to evacuate; violated key procedures (e.g. fought fire alone, ignored alarm, wrong route). |

Three ordinal classes. Casing must match across repos — **pick one** (`High/Moderate/Low` in the schema vs `HIGH/MODERATE/LOW` in the dashboard) and fix it before labeling starts.

---

## 3. Grounding: the BFP-validated behaviors

From the LSPU Sta. Cruz evacuation plan, the **correct** response set is explicit. The rubric scores against these.

**Fire — IN CASE OF EMERGENCY: ISOLATE → COMMUNICATE → EVACUATE → RECORD**
- **ISOLATE** — keep clear of danger; stay 2–3 m back from fire.
- **COMMUNICATE** — inform others; **press the fire alarm button**.
- **Extinguisher judgment (PASS)** — *only if appropriate*: **DO NOT FIGHT FIRE IF ALONE.** If using: Pull, Aim, Squeeze, Sweep; check type/pressure; keep exit behind you.
- **EVACUATE** — leave by designated/nearest safe exit; proceed to the **closest assembly area**.
- **Evacuation conduct** — remain calm, do not panic; follow directional arrows; await instructions at assembly area.

**Earthquake — (needs BFP/DRRMO validation like fire got; standard PH guidance)**
- **During shaking:** Duck, Cover, Hold; do **not** run during shaking; move away from windows/falling-object zones.
- **After shaking:** evacuate via safe route once shaking stops; proceed to assembly area; watch for aftershocks (mod emits `aftershock_count`/phase).

---

## 4. Scoring dimensions

Each run is scored on the dimensions below (0 / 1 / 2). The composite maps to a level (§5). Raters score from **session replays**, not live.

### 4A. Fire scenario

| # | Dimension | 0 (Poor) | 1 (Partial) | 2 (Correct) | Independent of features? |
|---|---|---|---|---|---|
| F1 | **Hazard avoidance / safe distance** | Repeatedly entered danger zone; reckless | Some unsafe proximity | Maintained safe distance (2–3 m) | overlaps `hazard_avoidance_ratio` — rate *appropriateness*, not raw distance |
| F2 | **Communication (alarm)** | Never alerted / never pressed alarm | Delayed alarm | Pressed fire alarm promptly | mostly independent (needs `fire_alarm_activate`) |
| F3 | **Intervention judgment** | Fought fire while alone / used extinguisher wrongly | Hesitant or partial correct use | Correct PASS use *or* correctly chose not to fight | **independent judgment** — the key non-feature dimension |
| F4 | **Evacuation route & success** | Wrong route / never reached safety | Inefficient but eventually safe | Designated/nearest exit → reached assembly area | overlaps `path_efficiency_ratio` — rate *correctness of choice* |
| F5 | **Composure** | Erratic, panicked movement | Some hesitation/erratic | Calm, controlled | overlaps `panic_proxy` — rate *observed* composure |
| F6 | **Timeliness** | No timely response | Slow but acted | Prompt appropriate first action | overlaps `decision_delay` |

### 4B. Earthquake scenario

| # | Dimension | 0 | 1 | 2 |
|---|---|---|---|---|
| E1 | **Protective action during shaking** | Ran/froze in danger during shaking | Late/partial Duck-Cover-Hold | Immediate Duck-Cover-Hold |
| E2 | **Falling-hazard avoidance** | Stayed near windows/heavy objects | Some exposure | Moved to safe spot away from hazards |
| E3 | **Post-shaking evacuation** | Evacuated during shaking / not at all | Delayed or inefficient | Evacuated safely after shaking |
| E4 | **Route & assembly success** | Wrong/failed | Safe but inefficient | Correct route → assembly area |
| E5 | **Composure** | Panicked | Some erratic | Calm |
| E6 | **Aftershock awareness** | Ignored aftershock risk | Partial | Re-took cover / waited appropriately |

---

## 5. From scores to a label

Sum the six dimensions (range 0–12), then apply tiers. **Calibrate cut-points with BFP/teacher raters during the pilot** — these are starting points:

| Composite (0–12) | Override rule | Label |
|---|---|---|
| 9–12 | — | **High** |
| 5–8 | — | **Moderate** |
| 0–4 | — | **Low** |
| any | **Critical-failure override:** fought fire alone (F3=0) **or** failed to evacuate/reach assembly (F4=0 / E4=0) | **cap at Low** |
| any | Never evacuated AND never took protective action | **Low** |

The override encodes the BFP non-negotiables — a fast, calm run that ends in an unsafe outcome is **not** High.

---

## 6. Rater protocol & reliability

1. **≥2 independent raters** per run (recommended: 1 BFP officer + 1 teacher), blind to each other's scores.
2. Score from a **standardized session replay** (same camera/data view for all raters).
3. Compute **inter-rater reliability**: Cohen's κ (2 raters) or Fleiss' κ (3+). **Target κ ≥ 0.60** (substantial). If lower, refine anchors and re-train raters before full labeling.
4. **Adjudicate disagreements** (third rater or consensus). Record final label + both raw scores.
5. **Report κ in Chapters 3–4.** This is your methodological defense against "where's the basis."

---

## 7. Scaling: the hybrid (expert gold + rule-based weak labels)

Hand-scoring every run may be infeasible at N≈300–400. Use the hybrid from the dev plan — **now implemented, not just planned:**

- **Expert gold set:** all runs if feasible, otherwise a stratified subset, entered through the **BFP-instructor validation UI** (override loop in the revised system architecture). **The test set is always expert-labeled** — `data_ingestion.resolve_session_labels()` / `split_train_test()` enforce this at the code level, not just by convention.
- **Rule-based weak labels** scale training data. The game's own 3-phase fire (and earthquake) state machine computes a numeric `simulation_score` and maps it to `prep_level` via `src/midrr_classifier/labeling.py::rule_based_label()`:
  - `score >= 75` → **High**
  - `40 <= score < 75` → **Moderate**
  - `score < 40` → **Low**
  - `labeling.py::phase_outcome_label()` provides an independent, coarser cross-check derived from which state-machine phase (`prevention` / `intervention` / `evacuation`) a run reached — use it to sanity-check the numeric rule, not as a second label source.
- **Label precedence** (`labeling.py::resolve_label()`): **expert override > game's precomputed `prep_level` > label derived from `simulation_score`**. Every resolved row carries a `label_source` of `"expert"` or `"rule"`.
- **Validate rule labels against the expert gold subset** using `labeling.py::rule_expert_agreement()` — computes Cohen's κ and raw agreement rate between the two on the overlapping (both-labeled) subset. Only trust weak labels at scale if they agree well with experts (target κ ≥ 0.60, matching §6's inter-rater bar).

> Keep rule-based labels **distinct in the data** via the `label_source` column (`expert` / `rule`, see `data_schema.LABEL_SOURCES`). Never report final accuracy on rule-labeled test data.

---

## 8. Surveys are *not* the label

Pre/post-test surveys (architecture diagram) measure **knowledge**. Chapter 2 argues preparedness is **behavioral**, not knowledge-based. So:
- Surveys → secondary **criterion/validity** variable (e.g. "does behavioral level correlate with knowledge gain?").
- Surveys → **not** the `preparedness_level` label.

---

## 9. Telemetry-contract deltas this rubric requires

Scoring the BFP procedures needs telemetry that v1.0 of the contract doesn't yet capture. **Update `telemetry_contract.md` to v1.1:**

| New element | Why (BFP procedure) | Add to contract |
|---|---|---|
| `fire_alarm_activate` event | COMMUNICATE step requires pressing the alarm (F2) | new `event_type` + `INTERACTION_EVENT_TYPES` |
| `assembly_area_reached` event | True evacuation success is reaching the assembly area, not touching an exit (F4/E4) | new `event_type`; becomes the real success signal |
| extinguisher context (`is_alone` / nearby-player count at event) | "DO NOT FIGHT FIRE IF ALONE" — F3 judgment depends on whether the player was alone | add a field on `extinguisher_use` events |
| map route metadata | `path_efficiency_ratio` should measure against *designated* exits from the floor plan | reference the LSPU floor plan when building the Minecraft map + define exit coordinates |

---

## 10. Open items for the team / BFP

- [ ] **Validate the earthquake dimensions (§4B) with BFP/DRRMO**, the same way fire was validated. The uploaded plan is fire-focused.
- [ ] Confirm the **Minecraft map replicates the LSPU Sta. Cruz floor plan** (exits, extinguisher + alarm positions, assembly areas) so behavior is measured against the real layout.
- [ ] Calibrate composite cut-points (§5) and `SAFE_HAZARD_DISTANCE` with raters during the pilot.
- [ ] Decide assembly-area coordinates in-game (the plan marks multiple "TO ASSEMBLY AREA" exits).
- [ ] Confirm `session_id` is the join key linking rubric scores → telemetry.

---

*Rubric v1.1 — grounded in the BFP-validated LSPU Sta. Cruz evacuation plan. Pair with `telemetry_contract.md` (v1.2) and the ML development plan §3.*
