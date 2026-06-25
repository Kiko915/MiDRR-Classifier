# MiDRR-Classifier — Task Tracker

Derived from `docs/MiDRR_ML_Development_Plan.md`. Phases are ordered by dependency — everything before Phase 3 can be done without real data.

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
- [ ] **Send `telemetry_contract.md` to Necookie with a hard deadline** — this starts the clock on the longest pole
- [ ] Agree on transport layer: PostgreSQL direct read vs batched CSV export; add a Postgres loader to `data_ingestion.py` alongside the CSV loader

---

## Phase 3 — Real Data Collection *(blocked on Phase 2)*

- [ ] Pilot run with small N — validate logs match contract and features compute correctly
- [ ] Capture expert-rubric labels for each run (raters watching/replaying sessions)
- [ ] Inter-rater reliability pass — resolve disagreements, compute Cohen's/Fleiss' κ
- [ ] Full data-collection runs (Santa Cruz / Calamba sites per Ch3)
- [ ] Version raw data with DVC

---

## Phase 4 — Feature Engineering on Real Data

- [ ] Run real logs through the pipeline; inspect per-feature distributions per class
- [ ] Calibrate `SAFE_HAZARD_DISTANCE` (currently hardcoded `5.0`) with domain experts
- [ ] Handle real-world edge cases: missing ticks, players who never evacuate, scenario time-limit caps
- [ ] EDA: per-class feature separability, correlations (informs importance interpretation)

---

## Phase 5 — Model Training, Tuning, Validation

- [ ] Train baseline RF with defaults; record metrics as the floor
- [ ] Stratified k-fold CV (k=5) — report **mean ± std**, not a single split
- [ ] Hyperparameter search over `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `class_weight`
- [ ] Address class imbalance (`class_weight="balanced"` and/or resampling)
- [ ] Lock final hyperparameters into `config.py`; retrain on full train split; persist `models/midrr_rf.pkl`

---

## Phase 6 — Evaluation & Explainability

- [ ] Report accuracy, per-class precision/recall/F1 (macro + weighted), confusion matrix
- [ ] Compute feature importance using **permutation importance and/or SHAP** (not Gini — it is biased for correlated features)
- [ ] Compute SHAP **per-session** (not only globally) — feeds the stealth-assessment adaptive feedback layer
- [ ] Persist all metrics to `models/metrics.json` for deterministic figure regeneration
- [ ] Sanity check importances against domain intuition (`decision_delay`, `hazard_avoidance_ratio` should rank high)

---

## Phase 7 — Serving / API

- [ ] Build `midrr-api` FastAPI service with `POST /predict` (accepts six features or raw session logs)
- [ ] Return `{ prepLevel, prepScore (proba → 0–100), featureImportance[], resultText }` — exact dashboard `Session` contract
- [ ] Add `POST /leads` and pre/post survey endpoints only if team decides surveys flow through this API
- [ ] Containerize the API
- [ ] Deploy (Render free tier + UptimeRobot keep-alive, or Railway)
- [ ] Set `PUBLIC_API_BASE_URL` in `BERONG_SMP_WEB` to the deployed URL

---

## Phase 8 — Adaptive Feedback (ECD / Stealth-Assessment Layer)

- [ ] Map predicted level + top SHAP feature contributions → human-readable `resultText` and improvement recommendations
- [ ] Keep it rule-driven and explainable (e.g. "high `decision_delay` + low `hazard_avoidance_ratio` → recommend drill on immediate evacuation")
- [ ] Define and document the feedback payload schema the mod consumes (cross-repo with Necookie)

---

## Phase 9 — Manuscript & Defense Support

- [ ] Generate publication-quality figures: confusion matrix, feature-importance bar chart, per-class metrics table, CV variance plot
- [ ] Write Chapter 4 results narrative (model performance + which behaviors drive preparedness)
- [ ] Prepare reproducibility appendix: fixed seeds, `config.py` snapshot, `requirements.txt`, data version hash
- [ ] Defense Q&A prep: why RF over alternatives, why these six features, how labels were obtained and their κ, small-N generalization limits

---

## Ongoing / Cross-Cutting

- [ ] Get BFP/DRRMO to validate earthquake rubric dimensions (current `labeling_rubric.md` is fire-focused)
- [ ] Reconcile `LABEL_CLASSES` order in confusion matrix axis vs dashboard display order
- [ ] Update this file as tasks are completed
