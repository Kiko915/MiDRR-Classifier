# MiDRR-Classifier — Task Tracker

Derived from `docs/MiDRR_ML_Development_Plan.md`. Phases are ordered by dependency — everything before Phase 3 can be done without real data.

> **2026-07-01 update:** Post-BFP-consultation, the simulation/analytics design changed
> (see `Diagrams/01`–`06` and the approved plan at
> `C:\Users\ASUS-Pc\.claude\plans\there-has-been-some-goofy-sloth.md`). This added a new
> **Phase 2.5** (9-feature migration, Turso ingestion, rule-based labeling) — **now complete,
> all 7 steps done** (see below). Phases 6–8 were further along than this file originally
> showed (SHAP, FastAPI, streaming, feedback text already shipped) and have now been
> re-verified/extended for the 9-feature contract as part of Phase 2.5 steps 6–7.

---

## Phase 0 — Foundations & Decisions

- [x] Reconcile label casing (`High` → `HIGH/MODERATE/LOW`) across `data_schema.py` and tests; add contract test
- [x] Decide labeling strategy; draft expert rubric + κ protocol (`docs/labeling_rubric.md`)
- [x] Freeze telemetry contract v1.1 (`docs/telemetry_contract.md`)
- [x] Lock the six feature **operational definitions** to exact Chapter 3 wording — write them as docstrings in `data_schema.py` (apply two BFP semantics fixes below)
  - `evacuation_time` / `decision_delay` end at `assembly_area_reached`, **not** `emergency_exit`
  - `interaction_frequency` must NOT count extinguisher use when `nearby_player_count == 0` (that's a violation)
- [x] Resolve Chapter 3 **6-features-vs-8-attributes** mismatch — state which raw attributes map to which features, and whether `Decision Sequence`, `Task Completion Time`, `Safety Compliance` are features, raw inputs, or out of scope

---

## Phase 1 — Synthetic Data + Pipeline Hardening

- [x] Write `src/midrr_classifier/synth.py` — synthetic log generator emitting telemetry contract v1.1 format (fire + earthquake, controllable skill level producing separable HIGH/MODERATE/LOW)
- [x] Add a companion notebook `notebooks/synthetic_pipeline.ipynb` showing end-to-end usage
- [x] Replace placeholder `compute_*` formulas in `feature_engineering.py` with Chapter-3-exact definitions; verify against synthetic ground truth
- [x] End-to-end smoke run on synthetic data: raw → features → train → evaluate → confusion matrix PNG
- [x] Implement group-aware stratified train/test split (no `player_id` leaking across splits) in `data_ingestion.py`
- [x] Set up GitHub Actions CI running `pytest` on push (`.github/workflows/ci.yml`)
- [x] Ensure all synthetic data outputs are clearly labelled as synthetic (mirror the web repo's requirement)

---

## Phase 2 — Telemetry Contract (Cross-Repo)

- [x] Write `docs/telemetry_contract.md` v1.1 with JSON + CSV shapes, per-tick vs per-event spec, full event vocabulary
- [x] Document required new mod instrumentation (per-tick `x/y/z`, `hazard_distance`, `fire_alarm_activate`, `assembly_area_reached`, `nearby_player_count`, `map_metadata.json`)
- [x] **Send `telemetry_contract.md` to Necookie with a hard deadline** — this starts the clock on the longest pole
- [x] Agree on transport layer: **batched CSV export** (PostgreSQL direct read dropped)

---

## Phase 2.5 — BFP-Revised Design Migration (9 Features, Turso, Rule-Based Labeling) *(NEW)*

Source: BFP consultation diagrams (`01`–`06`) + approved plan. Locked decisions: adopt all
9 features; full fire/earthquake parity; train on expert labels + rule-based weak labels
(`label_source`), test set expert-only; ingestion adapter supports both Turso and CSV.
This phase **blocks** Phases 3–9 doing anything meaningful under the old 6-feature schema.

### Step 1 — Contract ✅ (2026-07-01)
- [x] `data_schema.py`: `FEATURE_SCHEMA` 6 → 9 keys (`decision_latency`, `spray_accuracy`, `path_efficiency_ratio`, `hazard_avoidance_ratio`, `evacuation_time`, `interaction_frequency`, `resource_utilization`, `panic_proxy`, `situational_awareness`)
- [x] Rewrite `FEATURE_DEFINITIONS` locked prose for all 9 (fire computation + earthquake analog each)
- [x] Add new event constants: `ext_spray` (+ `hit_fire` flag), `pin_pull`, `hazard_neutralize`, `phase_transition` (Prevention/Intervention/Evacuation), `drop_cover_hold`, `extinguisher_class` field
- [x] Declare `nearby_player_count` in `RAW_LOG_SCHEMA` (currently used in code but undeclared); `session_end` documented as an `EVENT_TYPES` value (it's an event, not a raw-log column)
- [x] Update `INTERACTION_EVENT_TYPES`, `DECISION_LATENCY_ACTION_TYPES` (renamed from `DECISION_DELAY_ACTION_TYPES`), `CH3_ATTRIBUTE_MAPPING` for the 9 features
- [x] Add `label_source` (`expert`/`rule`) to the schema (+ `LABEL_SOURCES` constant)
- [x] `config.py`: `feature_cols` → the 9 (fixed order); `max_depth` `None` → `8`; expose `class_weight` as a config field (still hardcoded in `model_definition.py:78` — wiring is Step 6); add Turso connection fields (`turso_database_url`/`turso_auth_token`, env-sourced)
- [x] `docs/telemetry_contract.md` → **v1.2**: new events, `hit_fire`/`extinguisher_class` fields, `sessions`-table shape (`event_log` JSON + `move_log_csv` + `prep_level`/`confidence`/`simulation_score`/`passed`), Turso transport, 9 features + quake analogs, fix the 10/20 Hz inconsistency (locked to 20 Hz)

  > This step intentionally broke `feature_engineering.py`/`streaming.py` imports (old names removed) — expected, fixed by Step 2.

### Step 2 — Feature engineering ✅ (2026-07-01)
- [x] Add `compute_spray_accuracy`, `compute_resource_utilization`, `compute_situational_awareness` (each dispatches on fire-vs-quake scenario family via `_is_earthquake_scenario`)
- [x] Redefine `compute_panic_proxy` → std-dev of per-tick movement **speed²** (was bearing turn-angles)
- [x] Re-anchor `compute_decision_delay` → `compute_decision_latency`, measured from **SIM_START** (`session_start` event) rather than first hazard exposure
- [x] `build_feature_table()` emits the 9 columns + `preparedness_level` + `label_source`; keep group-by `(player_id, scenario_type)`
- [x] Verified: `pytest tests/` — 23 tests still pass; `test_feature_engineering.py`/`test_streaming.py` fail on old 6-feature names as expected (fixed in Step 7 test bump; `streaming.py` itself still needs its Step 6 update)
- [ ] TODO carried forward: `compute_resource_utilization` doesn't yet check `extinguisher_class` against room type (no `room_type` raw field from the mod yet) — sequencing-only for now

### Step 3 — Labeling ✅ (2026-07-01)
- [x] New `src/midrr_classifier/labeling.py`: `rule_based_label(score)` (≥75 HIGH / 40–74 MODERATE / <40 LOW), `phase_outcome_label()` (Prevention=HIGH / Intervention=MOD / Evacuation=MOD|LOW / fail=LOW)
- [x] Helper to compute rule-vs-expert agreement (κ) for `labeling_rubric.md` validation — `rule_expert_agreement()` (Cohen's κ + raw agreement rate via sklearn)
- [x] Wire into ingestion: `resolve_label()`/`attach_labels()` implement expert > rule-label > rule-score precedence; `data_ingestion.resolve_session_labels()` is the call site the Step 4 Turso/CSV loader will route through
- [x] `docs/labeling_rubric.md` → **v1.1**: documents the game rule-based label + BFP override flow, `label_source`, reaffirms circularity guard, references `labeling.py` functions directly (earthquake DCH BFP/DRRMO sign-off still pending — tracked in Ongoing, unchanged)
- [x] Verified: manual sanity check of all 5 functions + `pytest tests/` — same 23 pass / 2 known-broken (Step 6/7) as before, no new breakage

### Step 4 — Ingestion adapter ✅ (2026-07-01)
- [x] `data_ingestion.py`: new `load_sessions(source=...)` with two backends — Turso (`load_sessions_from_turso`, deferred `libsql-client` import so it's a true optional dependency, added as a poetry extra `turso`; parses `event_log` JSON + `move_log_csv` via `_sessions_table_to_raw_log`/`_explode_session_row`, maps `student_name→player_id`/`simulation_type→scenario_type`) and CSV (`load_raw_logs`/`load_feature_table` unchanged; `source="csv"` optionally joins a companion `sessions_<batch>.csv` on `session_id`)
- [x] Label resolution: both backends route through `resolve_session_labels()` (expert override > rule label > rule score)
- [x] Keep group-aware `split_train_test()`; **enforce expert-only test set** — non-expert rows are now dropped from the test split (never moved to train) whenever `label_source` carries real data; fully backward compatible when it's absent
- [x] **Bug found + fixed during verification:** `labeling.resolve_label()` used Python truthiness (`if expert_label:`), and `NaN` is truthy — a missing expert override read back from CSV (`None` → `NaN` on the CSV round-trip) was silently mislabeled as `label_source="expert"` with a NaN label. Replaced with a proper `_has_value()` None/NaN/empty-string check.
- [x] Also fixed a pre-existing latent bug hit during testing: a `→` arrow character in a `data_ingestion.py` log message crashes on Windows `cp1252` consoles (`UnicodeEncodeError`) — replaced with ASCII `->`. (Same arrow pattern exists elsewhere, e.g. `inference.py` debug logs — not fixed, out of scope for this step, flagged for later.)
- [x] Verified: mocked Turso session row (JSON `event_log` + CSV `move_log_csv`) → exploded raw log → `build_feature_table()` 9-feature row; CSV + companion `sessions_<batch>.csv` label join; `split_train_test()` expert-only enforcement with no player overlap; unknown/incomplete `source=` raises `ValueError`. `pytest tests/` — same 23 pass / 2 known-broken, no new breakage.

### Step 5 — Synthetic data ✅ (2026-07-01)
- [x] `synth.py`: emit new events — fire: `pin_pull` → `ext_spray` (with `hit_fire`) → `hazard_neutralize` (capped at 5 per session) via `_emit_pass_sequence()`, plus `phase_transition` (prevention/intervention/evacuation); earthquake: `drop_cover_hold` (+ a second one modeling re-covering on an aftershock)
- [x] Locked 20 Hz sampling (`_DT` 0.1 → 0.05) to match `telemetry_contract.md` v1.2
- [x] Class distribution → **HIGH 35% / MODERATE 45% / LOW 20%** across **250 sessions** is now the *default* (`generate_logs()` with no args → 88/112/50 via `_allocate_class_counts()`, largest-remainder rounding); legacy uniform `n_per_class=` behavior still supported for notebooks/small samples
- [x] Extended `_PROFILE` per skill with `ext_engage_prob`/`pin_pull_correct_prob`/`spray_hit_prob`/`spray_count_range` (fire) and `dch_prob`/`dch_recover_prob` (quake) — `situational_awareness` needed no new synth knob (it's a composite of already-generated signals)
- [x] Verified: 250-session default run → exact 88/112/50 class split, 125/125 fire/earthquake, zero NaN in all 9 feature columns; per-class-per-scenario means show clean HIGH > MODERATE > LOW separation on every feature (e.g. fire `spray_accuracy` 0.53/0.31/0.16, earthquake `resource_utilization` 0.52/0.31/0.22); legacy `n_per_class=` path unchanged; `pytest tests/` — same 23 pass / 2 known-broken, no new breakage
- [x] **Bug found + fixed while starting Step 6:** the movement model normalized velocity to a *fixed* magnitude every tick, so per-tick step distance was ~constant regardless of skill/`noise_std` — `panic_proxy` (speed² std-dev, redefined in step 2) carried almost no real signal and what little existed was inverted (HIGH showed *higher* "panic" than LOW). Added a per-tick `speed_jitter` scaled by `noise_std` (calm players ≈ constant pace, panicked players alternate sprint/near-freeze bursts). Re-verified: `panic_proxy` now HIGH ~2.2 < MODERATE ~6.2 < LOW ~8.0-8.3 for both scenarios — correct direction and clean separation. All other features re-checked, still properly ordered; `pytest` unaffected.
- [ ] Not done: `notebooks/synthetic_pipeline.ipynb` still shows 6-feature-era output — left as-is, not in this step's scope

### Step 6 — Retrain + downstream (mechanical 6→9 propagation) ✅ (2026-07-01)
- [x] `model_definition.py`: `build_model()` now reads `class_weight` from config instead of a hardcoded `"balanced"` string
- [x] `inference.py`: docstrings updated to the 9-feature example + accurate `predict_preparedness_full()` return-value docs (stale "Replace with SHAP in Phase 6" note removed — SHAP has been in since the earlier commit); fixed a `→`-arrow Windows `cp1252` crash risk in two `logger.debug` calls (same class of bug as Step 4's)
- [x] `streaming.py`: `_FEATURE_COLS` and the features dict in `StreamingPredictor.update()` now use all 9 `compute_*` functions
- [x] `api/schemas.py` + `api/routes/predict.py`: `FeaturesRequest`/`_FEATURE_COLS` extended to the 9 fields; SHAP `feature_cols` already auto-follows config, no change needed there
- [x] `api/feedback.py`: added HIGH/MODERATE/LOW templates + bright-spot lines for `spray_accuracy`/`resource_utilization`/`situational_awareness`; renamed `decision_delay` template keys to `decision_latency`; added `check_thresholds()` encoding the diagram's fixed numeric cutoffs (`decision_latency` >30s, `spray_accuracy` <0.40, `path_efficiency_ratio` <0.50, `panic_proxy` >2.0) as a rule-based flag layered onto the SHAP-driven message (independent of SHAP ranking, so a BFP-critical value is always surfaced)
- [x] **New `api/routes/session.py`**: exposes `POST /session/{id}/events` (+ `DELETE /session/{id}`) backed by a module-level `StreamingPredictor` singleton, best-effort model load at import time, mounted in `api/main.py`. Closes the telemetry_contract.md §6 gap.
- [x] **Bug found + fixed:** `StreamingPredictor`/`normalize_raw_log` require `player_id`/`scenario_type` on *every event row* (confirmed via `test_streaming.py`'s own fixtures), but the new session route only had them at the request's top level — caused an immediate `KeyError: 'scenario_type'`. Fixed by broadcasting both fields onto each event dict before calling `_predictor.update()`.
- [x] Session-completion handling: the route closes the buffer specifically on receiving a `session_end` event (not merely on `is_complete`/`assembly_area_reached`), matching contract §6d — a batch can have trailing events after the assembly area is reached.
- [x] **Retrained the model** on fresh 9-feature synthetic data (250 sessions, new `config` defaults: `max_depth=8`, `class_weight="balanced"`) — 100% train/test accuracy (synthetic data is separable by design); `models/midrr_rf.pkl` is gitignored so this is a local verification artifact, not a commit.
- [x] Verified end-to-end: `/predict` (direct function call, since installed `httpx`/`starlette` versions are incompatible with `TestClient` in this env — pre-existing dependency mismatch, not fixed, out of scope) returns a sensible HIGH prediction with SHAP-ranked `featureImportance` and the new `panic_proxy` HIGH template; `/session/{id}/events` returns partial-then-complete snapshots across two batches with correct buffer cleanup on `session_end`. `pytest tests/` — 38 pass / 1 known-broken assertion (old feature name, Step 7) / 1 known-broken import (Step 7), i.e. `test_streaming.py` now *runs* (was a full import failure before).

### Step 7 — Tests + docs ✅ (2026-07-01)
- [x] Bumped all 6-feature assertions to 9 across `tests/` (`test_feature_engineering.py` full rewrite with fire+quake dispatch coverage for all 9 `compute_*` fns; `test_streaming.py`'s `decision_delay` assertion → `decision_latency` + range checks for the 3 new features; `test_model_definition.py` dataset shape 6→9 cols + new `class_weight` config-wiring tests; `test_data_ingestion.py` fixture bumped to 9 features)
- [x] New unit tests: `tests/test_labeling.py` (26 tests — `rule_based_label` boundary values, `phase_outcome_label` branches, `resolve_label` NaN-safety regression tests, `attach_labels`, `rule_expert_agreement`); Turso adapter tests in `test_data_ingestion.py` (`TestTursoIngestion`, mocked `libsql_client` via `sys.modules` injection — no real dependency or network needed); `tests/test_synth.py` (12 tests — locked 88/112/50 class split, legacy `n_per_class=` mode, new event vocabulary, panic_proxy/spray_accuracy skill-separation regression guards); new `tests/test_api.py` (10 tests — 9-field `FeaturesRequest`, `/predict` full contract + 503/422 error paths, `/session/{id}/events` snapshot + 400 mismatch + `session_end` buffer cleanup)
- [x] `test_label_contract.py` — unchanged, as planned (casing still `HIGH/MODERATE/LOW`)
- [x] `docs/MiDRR_ML_Development_Plan.md`: added a 2026-07-01 changelog note; 6→9 feature references updated throughout (§0.1, the scope diagram, the readiness table, Ch3 Table-1 mismatch section); Turso replaces the originally-planned Postgres transport; §3 documents `labeling.py` as the rule-based labeler + BFP-validation-UI expert-override loop; Phases 6–8 marked `⚠️ partially done` with per-bullet status (SHAP, the API incl. the streaming route, and SHAP+threshold feedback are all shipped; CV/metrics.json, deployment, and the mod-facing feedback payload schema are not)
- [x] **CI fix found during this step:** `tests/test_api.py` imports the `api` package (FastAPI/Pydantic), but `.github/workflows/ci.yml` only installed the root `requirements.txt` — `api/requirements.txt` (fastapi/uvicorn/pydantic/shap) was never installed in CI, which would have broken the pipeline the moment this test file landed. Added `pip install -r api/requirements.txt` to the workflow.
- [x] Noted but not fixed (pre-existing, unrelated): installed `httpx`/`starlette` versions in this dev environment are mutually incompatible with FastAPI's `TestClient`; `test_api.py` calls route functions directly instead (still exercises the real request/response mapping).

**Verification for this phase:** `pytest tests/ -v` → **131 passed**, 0 failed, 0 errors (up from 23 passed / 2 broken files at the start of Phase 2.5). Confirmed: 9-feature dispatch for both fire and earthquake on every `compute_*` function; synthetic smoke run (250 sessions, exact 88/112/50 split, zero NaN, both scenarios); Turso adapter parses a mocked `sessions` row into a valid 9-feature raw-log row; `/predict` returns the full contract with 9-feature SHAP importances; `/session/{id}/events` returns a 9-feature snapshot and closes correctly on `session_end`; rule-based labels reproduce the ≥75/40/<40 tiers with expert-override precedence and the expert-only test-split guard enforced in code.

---

**Phase 2.5 — COMPLETE (2026-07-01).** All 7 steps done: 9-feature contract locked (schema, config, telemetry contract v1.2) → feature engineering with fire/earthquake dispatch → rule-based labeling + expert-override precedence → Turso/CSV ingestion adapter → synthetic data regenerated to spec → model retrained and every downstream consumer (inference, streaming, API, feedback) updated → full test suite green (131 tests) and dev-plan docs reconciled. Phases 3–9 below are now unblocked to build on the 9-feature contract.

---

## Phase 3 — Real Data Collection *(blocked on real mod telemetry — see `telemetry_contract.md` §7 gap analysis; Phase 2.5's 9-feature contract itself is done)*

- [ ] Pilot run with small N — validate logs match contract v1.2 and 9 features compute correctly
- [ ] Capture expert-rubric labels for each run (raters watching/replaying sessions)
- [ ] Inter-rater reliability pass — resolve disagreements, compute Cohen's/Fleiss' κ
- [ ] Full data-collection runs (Santa Cruz / Calamba sites per Ch3)
- [ ] Version raw data with DVC

---

## Phase 4 — Feature Engineering on Real Data

- [ ] Run real logs through the pipeline; inspect per-feature distributions per class (9 features)
- [ ] Calibrate `SAFE_HAZARD_DISTANCE` (currently hardcoded `5.0`) with domain experts
- [ ] Handle real-world edge cases: missing ticks, players who never evacuate, scenario time-limit caps
- [ ] EDA: per-class feature separability, correlations (informs importance interpretation)

---

## Phase 5 — Model Training, Tuning, Validation

- [ ] Train baseline RF with defaults; record metrics as the floor
- [ ] Stratified k-fold CV (k=5) — report **mean ± std**, not a single split
- [ ] Hyperparameter search over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `class_weight` (diagram spec starting point: `n_estimators=100`, `max_depth=8`)
- [ ] Address class imbalance (`class_weight="balanced"` and/or resampling)
- [ ] Lock final hyperparameters into `config.py`; retrain on full train split; persist `models/midrr_rf.pkl`

---

## Phase 6 — Evaluation & Explainability

- [x] Compute feature importance using SHAP (`src/midrr_classifier/explainability.py`, `TreeExplainer`) — feature-count-agnostic, verified against the retrained 9-feature model (Phase 2.5 step 6)
- [x] Compute SHAP **per-session** (not only globally) — `predict_preparedness_full()` returns per-student SHAP values over all 9 features
- [ ] Report accuracy, per-class precision/recall/F1 (macro + weighted), confusion matrix
- [ ] Permutation importance as a cross-check against SHAP
- [ ] Persist all metrics to `models/metrics.json` for deterministic figure regeneration
- [ ] Sanity check importances against domain intuition (`decision_latency`, `hazard_avoidance_ratio` should rank high)

---

## Phase 7 — Serving / API

- [x] Build `api/` FastAPI service with `POST /predict` (9-feature body)
- [x] Return `{ prepLevel, prepScore, featureImportance[], resultText }` contract (`api/schemas.py`, `api/routes/predict.py`)
- [x] Expose the streaming/mid-session route — `api/routes/session.py` (`POST /session/{id}/events`, `DELETE /session/{id}`), mounted in `api/main.py` (Phase 2.5 step 6)
- [ ] Add `POST /leads` and pre/post survey endpoints only if team decides surveys flow through this API
- [ ] Containerize the API
- [ ] Deploy (Render free tier + UptimeRobot keep-alive, or Railway)
- [ ] Set `PUBLIC_API_BASE_URL` in `BERONG_SMP_WEB` to the deployed URL

---

## Phase 8 — Adaptive Feedback (ECD / Stealth-Assessment Layer)

- [x] Map predicted level + top SHAP feature contributions → human-readable `resultText` (`api/feedback.py`), templates covering all 9 features
- [x] Encode the diagram's numeric feedback thresholds (latency >30s, spray accuracy <0.40, path efficiency <0.50, panic proxy >2.0) — `check_thresholds()`, layered onto the SHAP-driven message (Phase 2.5 step 6)
- [ ] Define and document the feedback payload schema the mod consumes (cross-repo with Necookie)

---

## Phase 9 — Manuscript & Defense Support

- [ ] Generate publication-quality figures: confusion matrix, feature-importance bar chart, per-class metrics table, CV variance plot
- [ ] Write Chapter 4 results narrative (model performance + which behaviors drive preparedness)
- [ ] Prepare reproducibility appendix: fixed seeds, `config.py` snapshot, `requirements.txt`, data version hash
- [ ] Defense Q&A prep: why RF over alternatives, why 9 features (and how they replaced the original 6), how labels were obtained (expert + rule-based hybrid) and their κ, small-N generalization limits

---

## Ongoing / Cross-Cutting

- [ ] Get BFP/DRRMO to formally validate the earthquake rubric dimensions (`labeling_rubric.md` §4B is drafted but unvalidated)
- [ ] Reconcile `LABEL_CLASSES` order in confusion matrix axis vs dashboard display order
- [ ] Keep `docs/MiDRR_ML_Development_Plan.md`, `telemetry_contract.md`, and `labeling_rubric.md` versions in sync as Phase 2.5 lands
- [ ] Update this file as tasks are completed
