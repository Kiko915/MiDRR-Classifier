# MiDRR-Classifier — AI/ML Engineer Development Plan

**Project:** *Minecraft as a Platform for AI-Enhanced Disaster Risk Simulation and Adaptive Educational Preparedness* (BERONG SMP)
**Your role:** AI/ML Engineer — own the feature pipeline, Random Forest model, evaluation, explainability, and the inference/serving boundary.
**Primary repo:** [`Kiko915/MiDRR-Classifier`](https://github.com/Kiko915/MiDRR-Classifier)
**Downstream consumers:** [`Necookie/BERONG_SMP_WEB`](https://github.com/Necookie/BERONG_SMP_WEB) (landing + dashboard), `Necookie/BERONG_SMP` (Minecraft mod/server — private)
**Companion docs (same `docs/` folder):** `telemetry_contract.md` (v1.1 — mod↔ML data spec) · `labeling_rubric.md` (how `y` labels are produced)

> Anchor the week numbers below to your real academic calendar (defense date, data-collection windows). They are relative durations, not fixed dates.

> **Updated** after the system architecture diagram and the BFP-validated LSPU Sta. Cruz evacuation plan were added. Key deltas: labeling rubric now exists and is grounded in the BFP plan; telemetry contract bumped to v1.1 (three new events + map metadata); PostgreSQL confirmed as the data-layer store; feature importance confirmed to feed the stealth-assessment layer.

> **Updated again (2026-07-01)** after a second BFP consultation revised the simulation/analytics design (Phase 2.5, see `tasks.md`). Key deltas: **6 → 9 engineered features** (new `spray_accuracy`, `resource_utilization`, `situational_awareness`; `decision_delay` renamed/re-anchored to `decision_latency`; `panic_proxy` redefined from bearing-change std-dev to movement-speed² std-dev — every feature now has an earthquake-parity computation, not just fire); telemetry contract bumped to **v1.2** (3-phase fire state machine, PASS-technique events, Drop-Cover-Hold); the transport layer is **Turso Cloud DB (libSQL)**, not PostgreSQL as originally planned here — `data_ingestion.load_sessions()` supports both Turso and CSV; the game's own **3-phase state machine now computes a rule-based `prep_level`** (`src/midrr_classifier/labeling.py`), used as a `label_source="rule"` weak label alongside `label_source="expert"` BFP-instructor overrides from a validation UI (test split remains expert-only — circularity guard unchanged). Phases 6–8 below were also already **partially shipped** before this revision (SHAP explainability, the FastAPI service, SHAP-informed feedback text) — see the phase-by-phase status notes.

---

## 0. Read this first — the two things that decide your whole timeline

Before any modeling, two facts from the actual repos govern everything:

### 0.1 The telemetry gap (your critical path, and it's not in your repo)
Your `data_schema.RAW_LOG_SCHEMA` and Chapters 1 & 3 assume **per-event/per-tick logs** — `x/y/z`, `timestamp`, `hazard_distance` over time, and interaction events (`door_open`, `extinguisher_use`, `emergency_exit`). All nine of your engineered features (v1.2 — see the 2026-07-01 update note above) are computed from that granularity.

But per `BERONG_SMP_WEB/CLAUDE.md`, the mod **currently emits session-level data only**:

| The mod produces today | Your nine features need |
|---|---|
| `disasterType` (FIRE/EARTHQUAKE) | per-tick `x,y,z` trajectory → `path_efficiency_ratio`, `panic_proxy` |
| session duration / ticks elapsed | `timestamp` per event → `evacuation_time`, `decision_latency` |
| `firesExtinguishedCount` (fire) | timestamped interaction/PASS events → `interaction_frequency`, `decision_latency`, `spray_accuracy`, `resource_utilization` |
| `magnitude`, `aftershockCount`, `EarthquakePhase` (quake) | `hazard_distance` per tick, `drop_cover_hold` events → `hazard_avoidance_ratio`, the quake analogs of `spray_accuracy`/`resource_utilization` |
| player UUID, start/end | — |

**Implication:** none of your nine features can be computed from real data until the mod ships new per-tick instrumentation. This is a **cross-repo dependency on Necookie**, and it is the single longest pole. Your `data_schema.py` already *is* the contract — your first high-leverage move is to formalize it as a logging spec and hand it over.

> **Now formalized** as `docs/telemetry_contract.md` (v1.1). The BFP evacuation plan added three required events the mod must build beyond the six-feature stream: `fire_alarm_activate` (COMMUNICATE step), `assembly_area_reached` (the *true* evacuation-success signal — not `emergency_exit`), and `nearby_player_count` on `extinguisher_use` (the "do not fight fire alone" rule). It also adds a one-time static `map_metadata.json` (designated exits, assembly areas, alarm/extinguisher positions) so `path_efficiency_ratio` measures against the real floor plan.

### 0.2 The labeling problem (no ground truth exists yet)
Random Forest is **supervised** — it needs labeled `preparedness_level` per run. Nothing in either repo produces those labels yet, and Chapter 2 itself argues preparedness is *behavioral, not knowledge-based*, so you cannot just use a quiz score as the label without weakening construct validity. You must decide the labeling strategy **before** collecting data, because it shapes the consent forms, the rubric, and the experimental protocol. See §3.

> **Now addressed** by `docs/labeling_rubric.md` — an expert-rubric + rule-based hybrid grounded in the BFP-validated LSPU evacuation plan (PASS, ISOLATE→COMMUNICATE→EVACUATE→RECORD, assembly-area success). It includes a **circularity/label-leakage guard** (don't let raters score the same quantities the model uses as features) and a critical-failure override (unsafe outcome caps at Low regardless of speed/calm).

> These two items are also the honest answer to your adviser's Chapter 2 comment (*"nasan ang basis like supporting articles"*) on the Algorithm Matrix: the basis for RF is in your lit review (Fife & D'Onofrio 2023; Wu & Jiang 2024; Chen 2022), but the basis for *your labels and features* must be made explicit in Chapter 3.

---

## 1. Scope boundary — what is yours vs. not

```mermaid
flowchart LR
    subgraph MOD["BERONG_SMP (mod) — Necookie"]
        A[Minecraft fire/earthquake sim] --> B[Per-tick + session telemetry]
    end
    subgraph YOURS["MiDRR-Classifier — YOU"]
        C[Ingestion + schema validation] --> D[Feature engineering<br/>9 features]
        D --> E[Random Forest train/tune/validate]
        E --> F[Evaluation + explainability]
        E --> G[predict_preparedness inference]
        G --> H[FastAPI /predict service]
    end
    subgraph WEB["BERONG_SMP_WEB — Necookie"]
        I[Dashboard: prepLevel, prepScore,<br/>featureImportance, resultText]
    end
    B -->|JSON logs per session| C
    H -->|label + scores + feature importance| I
    H -->|adaptive feedback payload| A
```

**Yours (own end to end):** §C–H above — ingestion, features, model, evaluation, explainability, the inference function, and the serving API.
**Shared/negotiated:** the telemetry contract (§0.1) and the adaptive-feedback payload schema (§9).
**Not yours:** mod gameplay logic, the dashboard UI, auth. You *specify* what you need from them; you don't build it.

---

## 2. Repo readiness — where MiDRR-Classifier actually stands

Good news: the scaffold is solid and well-aligned to Chapter 3. Honest status:

| Module | State | What's left |
|---|---|---|
| `config.py` | ✅ real | `n_estimators`/`max_depth`/`class_weight` locked to the diagram spec (100/8/balanced); add Turso connection fields — **done** |
| `data_schema.py` | ✅ real | 9-feature contract locked (v1.2); `LABEL_CLASSES` casing reconciled (`HIGH/MODERATE/LOW`, tested); `SAFE_HAZARD_DISTANCE` calibration still pending real data (Phase 4) |
| `data_ingestion.py` | ✅ real | group-aware stratified split done; Turso + CSV adapter (`load_sessions()`) done; expert-only test-split circularity guard enforced in code |
| `feature_engineering.py` | ✅ real | 9 `compute_*` functions with fire/earthquake dispatch, verified against synthetic ground truth; still placeholder-quality pending real-data calibration (Phase 4) |
| `labeling.py` | ✅ new (Phase 2.5) | rule-based label + phase-outcome cross-check + expert/rule precedence + κ helper |
| `model_definition.py` | ✅ real wrapper | `predict_proba`, config-driven `class_weight`, retrained on the 9-feature synthetic set |
| `train.py` / `evaluate.py` | ✅ runnable | still needs CV + persisted `metrics.json` (Phase 5/6) |
| `inference.py` | ✅ done | proba + per-student SHAP + global importances all returned |
| `explainability.py` | ✅ done (was Phase 6) | SHAP `TreeExplainer`, cached per model, feature-count-agnostic |
| `tests/` | ✅ | label casing, group split + expert-only enforcement, API contract, labeling thresholds, synth distribution all covered |
| **REST API** | ✅ built (was Phase 7) | `POST /predict` + `POST /session/{id}/events` (streaming) + `/health`; not yet containerized/deployed |
| **Dataset** | ❌ none | blocked on §0.1 — still true even under the revised 9-feature/Turso design |

**Quick win this week:** the casing mismatch (`High` vs `HIGH`) will silently break the dashboard integration. Pick one canonical form, fix `LABEL_CLASSES`, and add a test that asserts it. Small, but it's a real latent bug across repos.

---

## 3. Labeling strategy — decide before collecting data

You need ordinal labels (`High`/`Moderate`/`Low`) per run. Three viable routes; you most likely want the **hybrid**:

| Route | How | Strength | Weakness |
|---|---|---|---|
| **A. Expert-rubric (recommended gold standard)** | BFP officers / teachers (already your Ch3 stakeholders) score each session against a fixed disaster-response rubric → ordinal label | High construct validity; ECD-aligned; defensible at defense | Labor-intensive; needs **inter-rater reliability (Cohen's/Fleiss' κ)** |
| **B. Rule-based weak labels** | Threshold rules on raw outcomes (e.g. evacuated in time + low hazard exposure → High) | Scales to all sessions; cold-start friendly | Circular if rules ≈ features; must be validated against A |
| **C. Pre/post knowledge test as proxy** | Use test gain as the label | Cheap | Weak — contradicts your own Ch2 "behavior ≠ knowledge" argument; use only as a secondary criterion variable, not the primary label |

**Recommended:** Expert-rubric on the **full set if feasible, or a gold subset**, plus rule-based weak labels to scale, with the rule-based labels **validated against the expert gold set** (report agreement). Report κ in Chapter 3/4. This directly hardens the methodology against the "where's the basis" critique.

**Status — DONE:** this strategy is now fully specified in `docs/labeling_rubric.md`, grounded in the BFP-validated LSPU evacuation plan. It defines the six scoring dimensions per scenario, the composite→level tiers, the critical-failure override, the κ protocol, the `label_source` (`expert`/`rule`) separation, and the circularity guard.

**Status — DONE (2026-07-01 revision):** the rule-based side of the hybrid is now real code, not just a plan. The revised simulation design runs its own 3-phase fire (and earthquake) state machine, which computes a numeric `simulation_score` and maps it to a rule-based `prep_level` — mirrored on the ML side by `src/midrr_classifier/labeling.py` (`rule_based_label()`, `phase_outcome_label()` as an independent cross-check, `rule_expert_agreement()` for the κ-style validation this section calls for). The BFP-instructor validation UI is the **expert override loop**: `resolve_label()`/`attach_labels()` implement expert-override-wins-over-rule-label precedence, and `data_ingestion.split_train_test()` enforces the expert-only test split in code (not just as a documented convention). **Remaining action (unchanged):** get BFP/DRRMO to validate the *earthquake* dimensions (the uploaded plan is fire-focused) and run the rater-calibration pilot — that's a real-world data-collection step this revision doesn't change.

---

## 4. Development lifecycle (phased)

The ordering is deliberate: everything you *can* do without real data is front-loaded, so you're not idle while the mod telemetry and data collection happen in parallel.

> **Note:** the checkboxes below are the original phase plan and are frozen at roughly when this document was last substantially rewritten. **`tasks.md` is the live, currently-accurate phase tracker** — by the 2026-07-01 revision, Phases 0–2 are fully done and Phases 6–8 are partially done (see the phase notes below and `tasks.md`'s Phase 2.5 section for the 9-feature migration specifically).

### Phase 0 — Foundations & decisions *(do now, ~1 wk)*
- [ ] Reconcile label casing across `data_schema.py` ↔ dashboard `PrepLevel`. Add a test.
- [ ] Lock the **operational definitions** of all nine features (v1.2) to exact Chapter 3 wording (units, edge cases, time caps). Write them into `data_schema.py` docstrings as the source of truth.
- [ ] Decide labeling strategy (§3) and draft the rubric + κ protocol.
- [ ] Freeze the **telemetry contract v1** (next phase) from your `RAW_LOG_SCHEMA`.

### Phase 1 — Synthetic data + pipeline hardening *(parallel, ~1–2 wks, no real data needed)*
- [ ] Write a **synthetic log generator** that emits the `telemetry_contract.md` v1.2 format (including `fire_alarm_activate`, `assembly_area_reached`, `nearby_player_count`, the PASS-technique events, and `drop_cover_hold`) for fire and earthquake, with controllable "skill" so it produces separable High/Moderate/Low. Put it in `src/midrr_classifier/synth.py` and a notebook.
- [ ] Replace placeholder `compute_*` formulas with the locked Chapter-3 definitions; verify against synthetic ground truth. **Two semantics fixes from the BFP plan:** (a) `evacuation_time`/`decision_latency` end at `assembly_area_reached`, **not** `emergency_exit` (an exit is a waypoint); (b) `interaction_frequency` is **not** monotonically good — extinguisher use while `nearby_player_count == 0` is a *violation*, so don't treat all interactions as positive signal.
- [ ] End-to-end smoke run: raw → features → train → evaluate → confusion matrix PNG, **entirely on synthetic data**.
- [ ] Group-aware, stratified train/test split (never let one `player_id` leak across splits).
- [ ] CI: GitHub Actions running `pytest` on push.
- [ ] **Label all synthetic outputs clearly as synthetic** (the web CLAUDE.md insists on this separation; mirror it).

### Phase 2 — Telemetry contract (cross-repo) *(blocks real data; start ASAP)*
- [x] Turn `RAW_LOG_SCHEMA` into a versioned **`docs/telemetry_contract.md`** — **DONE (v1.1)**: JSON + batch-CSV shapes, per-tick vs per-event, 10 Hz sampling, coordinate frame, units, full `event_type` vocabulary.
- [x] Specify required new mod instrumentation — **DONE** (§7 gap analysis): per-tick `x,y,z`; running `hazard_distance`; `fire_alarm_activate`; `assembly_area_reached`; `nearby_player_count`; one-time `map_metadata.json`.
- [ ] **Hand to Necookie with a deadline (your single highest-leverage action — do today).** Track it as a cross-repo dependency.
- [x] Agree on transport — **DONE, superseded by the 2026-07-01 revision:** the original architecture diagram proposed PostgreSQL; the team settled on **Turso Cloud DB (libSQL)** instead, with a `sessions` table (`event_log` JSON + `move_log_csv`). `data_ingestion.load_sessions(source="turso"|"csv")` supports both, so batched CSV export remains the offline/reproducibility fallback.

### Phase 3 — Real data collection *(blocked on Phase 2; ~2–4 wks of runs)*
- [ ] Pilot: small N first to validate that logs match the contract and features compute sanely.
- [ ] Capture expert-rubric labels alongside each run (your raters watching/replaying sessions).
- [ ] Inter-rater reliability pass; resolve disagreements; compute κ.
- [ ] Full data-collection runs (Santa Cruz / Calamba sites per Ch3).
- [ ] Version raw data (consider DVC — already a TODO in your README).

### Phase 4 — Feature engineering on real data *(~1 wk)*
- [ ] Run real logs through the pipeline; inspect distributions per feature per class.
- [ ] Calibrate `SAFE_HAZARD_DISTANCE` and any thresholds with domain experts (currently a hardcoded `5.0` TODO).
- [ ] Handle real-world mess: missing ticks, players who never evacuate, scenario time-limit caps.
- [ ] EDA: per-class feature separability, correlations (matters for importance interpretation later).

### Phase 5 — Model training, tuning, validation *(~1–2 wks)*
- [ ] Baseline RF with defaults → record metrics as the floor.
- [ ] **Stratified k-fold CV (k=5)** — report **mean ± std**, not a single split (your N is small: ≤300–400 students, fewer complete sessions, 2 scenarios → high variance risk).
- [ ] Hyperparameter search (`n_estimators`, `max_depth`, `min_samples_leaf`, `max_features`, `class_weight`) via grid/random search inside CV.
- [ ] Address class imbalance (`class_weight="balanced"` and/or resampling) — preparedness classes will not be even.
- [ ] Lock final config into `config.py`; retrain on full train split; persist `models/midrr_rf.pkl`.

### Phase 6 — Evaluation & explainability *(~1 wk)* — ⚠️ partially done
- [x] **Feature importance: SHAP**, computed **per-session** via `explainability.py`'s `TreeExplainer` (cached per model, feature-count-agnostic — the 9-feature migration needed no changes here). `inference.predict_preparedness_full()` returns both per-student SHAP and global Gini importance (kept for training-time diagnostics only).
- [ ] Report accuracy, **per-class** precision/recall/F1 (macro + weighted), confusion matrix — all nine are explicit Chapter 1/3 objectives. Still pending real/synthetic-scale evaluation run.
- [ ] Persist all metrics to a `models/metrics.json` so figures regenerate deterministically for the manuscript.
- [ ] Sanity check importances against domain intuition (e.g. `decision_latency`, `hazard_avoidance_ratio` should rank high) — this *is* your "what behaviors drive preparedness" research finding.

### Phase 7 — Serving / API *(~1 wk; the web repo already expects it)* — ⚠️ partially done
- [x] Built `api/` (FastAPI): `POST /predict` taking the nine features, **and** `POST /session/{id}/events` for mid-session streaming predictions (the architecture diagram's real-time path — this wasn't in the original phase plan but was already implemented in `streaming.py` and is now mounted).
- [x] Returns `{ prepLevel, prepScore (proba→0–100), featureImportance[], resultText }` — matches the dashboard `Session` contract exactly.
- [ ] Add `POST /leads` and pre/post survey endpoints **only if** your team decides surveys flow through this API (web repo references them as design intent).
- [ ] Containerize; deploy somewhere free/cheap (Render free tier with an UptimeRobot keep-alive is fine for a thesis demo — same pattern you've used before; Railway or a small VPS if you want no cold starts).
- [ ] Set `PUBLIC_API_BASE_URL` in the web repo to your deployed URL.

### Phase 8 — Adaptive feedback (ECD / stealth-assessment layer) *(~1 wk)* — ⚠️ partially done
- [x] `api/feedback.py` maps predicted level + top SHAP feature contributions → human-readable `resultText`, with a "bright spot" callout when a feature helped despite the overall result. Extended (2026-07-01) with `check_thresholds()` — the diagram's fixed numeric cutoffs (`decision_latency` >30s, `spray_accuracy` <0.40, `path_efficiency_ratio` <0.50, `panic_proxy` >2.0) layered on top of the SHAP-driven message, kept rule-driven and explainable per the original intent below.
- [ ] Define the feedback payload the mod consumes (cross-repo with Necookie) — still undocumented as a formal contract.

### Phase 9 — Manuscript & defense support *(ongoing → final)*
- [ ] Generate publication-quality figures: confusion matrix, feature-importance bar, per-class metrics table, CV variance.
- [ ] Write the results narrative for Chapter 4 (model performance, which behaviors mattered).
- [ ] Prepare a reproducibility appendix: seeds, config, env (`requirements.txt`), data version.
- [ ] Defense Q&A prep: *why RF over alternatives* (your Algorithm Matrix), *why these features*, *how you got labels and their reliability (κ)*, *small-N generalization limits*.

---

## 5. Task → manuscript mapping

| Manuscript element | Phase / artifact that satisfies it |
|---|---|
| Ch1 obj: structured data logging architecture | §2 telemetry contract + `data_schema.py` |
| Ch1 obj: train & validate RF on behavioral features | Phases 4–5 |
| Ch1 obj: ECD stealth assessment + adaptive feedback | Phase 8 |
| Ch1 obj: accuracy/precision/recall/F1/confusion/importance | Phase 6 + `metrics.json` |
| Ch2 adviser comment ("basis…") on Algorithm Matrix | §3 labeling basis + §4 Phase 9 defense notes; cite Fife & D'Onofrio, Wu & Jiang, Chen |
| Ch3 Granular Logging Framework (Table 1) | §0.1 / §2 — reconciled: the original 6 features map to Table 1's 8 attributes via `CH3_ATTRIBUTE_MAPPING` in `data_schema.py` (`Decision Sequence`, `Task Completion Time`, `Safety Compliance` are raw-attribute groundings, not separate features). The 2026-07-01 revision adds 3 more features (`spray_accuracy`, `resource_utilization`, `situational_awareness`) that are **new BFP-driven instrumentation, not part of Ch3 Table 1** — call this out explicitly in Chapter 3/4 rather than forcing a Table-1 mapping that doesn't exist for them |
| Ch3 70/30 split + K-fold | Phase 5 (use grouped, stratified CV) |
| Ch3 System Architecture (Data/ML/Assessment layers) | Phases 1–8; architecture diagram **confirms PostgreSQL** as the data-layer store (Real-Time Logging → Postgres → Feature Engineering) and routes **Feature Importance → Stealth Assessment**. Add a Postgres loader to `data_ingestion.py`; compute SHAP per-session for the feedback layer |

> ⚠️ **Consistency fix for Ch3 — DONE, then extended:** Table 1 lists **8** logged attributes; the original model used **6** engineered features, resolved via `CH3_ATTRIBUTE_MAPPING` (`Decision Sequence`, `Task Completion Time`, `Safety Compliance` are raw-attribute groundings, not extra features). The model now uses **9** features — the 3 new ones (`spray_accuracy`, `resource_utilization`, `situational_awareness`) are BFP-driven additions with no Table-1 attribute to map to. State this explicitly in Chapter 3/4: don't force-fit them into Table 1, name them as a post-BFP-consultation design addition.

---

## 6. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Mod telemetry never reaches required granularity | **High** | Lock contract early (§2); have a fallback reduced feature set computable from session-level data only, so you can still ship *something* |
| Small N → overfitting, unstable metrics | High | Grouped stratified k-fold, report mean±std, prefer shallower trees, `class_weight=balanced`, permutation importance |
| Label quality / rater disagreement | Med | Rubric + κ; resolve via adjudication; report reliability |
| Label casing / schema drift across repos | Med | Single versioned contract; contract tests in CI |
| Correlated features inflate Gini importance | Med | SHAP / permutation importance |
| Scope creep into mod/dashboard work | Med | Hold the §1 boundary; specify, don't build, others' parts |
| **Your known pattern: motivation fading ~week 2, projects unfinished** | — | Define "done" per phase on day one (§7); ship the synthetic-data end-to-end pipeline *first* so you always have a working, demoable system even if real data slips |

---

## 7. "Done" criteria (define the finish line now)

The ML component is **defense-ready** when:
1. The full pipeline runs raw logs → features → trained model → evaluation **on real, labeled data** with one command.
2. Reported metrics use grouped stratified CV with mean ± std, plus a held-out confusion matrix.
3. Feature importance (permutation/SHAP) is computed, plotted, and interpreted against domain expectation.
4. `predict_preparedness` is wrapped in a deployed `/predict` API returning the dashboard's exact contract.
5. Adaptive feedback maps predictions → explainable recommendations.
6. Labeling reliability (κ) is reported; feature/label definitions match Chapter 3 verbatim.
7. Everything reproduces from `requirements.txt` + fixed seed + a versioned dataset.

A *minimum viable* version (so you're never stuck with nothing): items 1–3 on **synthetic** data. Build that in Phase 1 and you have a working system to demo regardless of mod/data delays.

---

## 8. Immediate next actions (this week)

> **This entire numbered list is now historical — all 6 items were completed** (see `tasks.md` Phases 0–2). Left as-is below for the record rather than rewritten; the six-feature wording in item 3 reflects the *original* v1.1 design before the 2026-07-01 revision to nine features.

Done since first draft: ~~write the telemetry contract~~ (✅ v1.1) · ~~draft the labeling rubric + κ protocol~~ (✅ `labeling_rubric.md`). Remaining:

1. **Commit the three `docs/` files and send `telemetry_contract.md` to Necookie with a deadline** — the clock-starter on the longest pole. Do this first.
2. Fix the `High`/`HIGH` label casing across repos + add a test.
3. Lock the six feature operational definitions to Chapter 3 wording in `data_schema.py` (apply the two BFP semantics fixes: assembly-area = success; extinguisher-while-alone ≠ good).
4. Resolve the Chapter 3 **6-features-vs-8-attributes** mapping with your team.
5. Build `synth.py` (emitting contract v1.1) + run the full pipeline end-to-end on synthetic data (your MVP safety net).
6. Get BFP/DRRMO to validate the **earthquake** rubric dimensions.

---

*Plan generated from your three repos and Chapter 1–3 drafts. Re-anchor phase durations to your defense calendar.*
