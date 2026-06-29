# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

MiDRR-Classifier is the ML backend for the *BERONG SMP* thesis project (LSPU CMSC 312). It trains a Random Forest classifier that predicts a student's disaster preparedness level (`HIGH` / `MODERATE` / `LOW`) from six behavioral features extracted from Minecraft fire/earthquake simulation logs.

**This repo owns:** ingestion → feature engineering → model training → evaluation → `predict_preparedness()` inference boundary.
**Out of scope here:** the Minecraft mod (`BERONG_SMP` — Necookie), the web dashboard (`BERONG_SMP_WEB` — Necookie), and the future `midrr-api` FastAPI service.

## Commands

```bash
# Install (editable mode so src/ imports resolve)
pip install -e .

# Run all tests
pytest tests/ -v

# Run a single test file
pytest tests/test_feature_engineering.py -v

# Train (requires data/processed/features.csv — will raise FileNotFoundError without it)
python -m midrr_classifier.train
python -m midrr_classifier.train --config config.yaml

# Evaluate (requires a trained model and test CSV)
python -m midrr_classifier.evaluate
python -m midrr_classifier.evaluate --model models/midrr_rf.pkl --test-csv data/processed/test.csv
```

## Architecture

Data flows in one direction: raw gameplay logs → feature engineering → processed feature table → model train/evaluate → serialized model → inference.

```
data/raw/gameplay_logs.csv
        ↓  data_ingestion.load_raw_logs()
        ↓  feature_engineering.build_feature_table()  ← group-by (player_id, scenario_type)
data/processed/features.csv
        ↓  data_ingestion.load_feature_table() + split_train_test()
        ↓  MiDRRClassifier.build_model() → fit()
models/midrr_rf.pkl
        ↓  inference.predict_preparedness(features_row)  ← integration boundary for external callers
```

### Key modules

- **`config.py`** — `MiDRRConfig` dataclass is the single source of truth for all paths and hyperparameters. Pass a YAML path to `load_config()` to override defaults.
- **`feature_engineering.py`** — Six `compute_*` functions (currently placeholder implementations; must be replaced with Chapter 3 exact definitions before real experiments). `build_feature_table()` groups raw events by `(player_id, scenario_type)` and applies each.
- **`model_definition.py`** — `MiDRRClassifier` wraps sklearn's `RandomForestClassifier`. Always call `build_model()` before `fit()`. Uses `class_weight="balanced"`.
- **`inference.py`** — `predict_preparedness(features_row)` is the intended API for external systems. The classifier is LRU-cached (`maxsize=1`) to avoid repeated disk I/O.
- **`data_schema.py`** — Defines `RAW_LOG_SCHEMA`, `FEATURE_SCHEMA`, and `LABEL_CLASSES`. This file doubles as the telemetry contract spec for the mod.

### The six engineered features

| Feature | Description |
|---|---|
| `evacuation_time` | Elapsed seconds from scenario start to `assembly_area_reached` |
| `decision_delay` | Latency from first `hazard_proximity` event to first valid action |
| `path_efficiency_ratio` | Straight-line distance ÷ total path length (0–1] |
| `hazard_avoidance_ratio` | Fraction of timesteps at safe distance from hazard [0–1] |
| `interaction_frequency` | Safety interactions per second (context-dependent — see below) |
| `panic_proxy` | Std-dev of consecutive bearing changes (erratic movement indicator) |

## Critical constraints

### Label casing — must be uppercase
`LABEL_CLASSES = ["HIGH", "MODERATE", "LOW"]`. The downstream dashboard (`BERONG_SMP_WEB`) expects uppercase labels. `tests/test_label_contract.py` enforces this. Never change label casing without updating both repos.

### Feature formula semantics (BFP-validated)
Two non-obvious domain rules must be reflected in `compute_*` implementations:
1. `evacuation_time` / `decision_delay` end at `assembly_area_reached`, **not** `emergency_exit` (an exit is a waypoint, not the success signal).
2. `interaction_frequency` — extinguisher use while alone (`nearby_player_count == 0`) is a *violation* per BFP procedure ("DO NOT FIGHT FIRE IF ALONE"), not positive signal. Don't treat all interactions as equivalent.

### Group-aware train/test split
Never allow the same `player_id` to appear in both train and test splits. Use `split_train_test()` with `group_col="player_id"`. Violating this leaks per-student patterns and inflates evaluation metrics.

### Synthetic data flag
All synthetic data must be labeled as such. The web repo insists on this separation. Add a `label_source` column (`"expert"` / `"rule"` / `"synthetic"`) when generating or loading data.

## Current status (scaffold)

- `feature_engineering.py` — placeholder formula implementations. Replace all `compute_*` with Chapter 3 exact definitions before running real experiments.
- `data/raw/` and `data/processed/` are empty (`.gitkeep`). `train.py` will raise `FileNotFoundError` until real or synthetic data is placed there.
- No REST API yet — `midrr-api` FastAPI service is a planned separate deliverable.
- `SAFE_HAZARD_DISTANCE = 5.0` (blocks) in `data_schema.py` is a placeholder; calibrate with domain experts.

## Commit style

Do not add `Co-Authored-By: Claude` trailers to commits in this repo.

## Docs worth reading

- `docs/telemetry_contract.md` — v1.1 spec for what the Minecraft mod must emit. This is the cross-repo dependency on Necookie.
- `docs/labeling_rubric.md` — how `preparedness_level` labels are produced by BFP/teacher expert raters; includes the circularity guard and κ protocol.
- `docs/MiDRR_ML_Development_Plan.md` — phased development roadmap (Phases 1–9) with manuscript mapping.
