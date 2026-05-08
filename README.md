# MiDRR-Classifier

> **Minecraft Disaster Risk and Resilience Classifier** — the ML backend for the thesis project *"Minecraft as a Platform for AI-Enhanced Disaster Risk Simulation and Adaptive Educational Preparedness."*

This repository contains a Random Forest classifier that predicts a student's **disaster preparedness level** (High / Moderate / Low) from behavioural data logged during Minecraft-based fire and earthquake simulations.

---

## Research Context

Minecraft is used as an immersive, low-cost platform for simulating disaster scenarios (fire evacuation, earthquake response). Students interact with the simulation without being aware they are being assessed — a stealth-assessment paradigm. The game logs fine-grained behavioural events (movement coordinates, hazard proximity, interaction events, timing) which are aggregated into six engineered features:

| Feature | Description |
|---|---|
| `evacuation_time` | Total elapsed seconds from scenario start to exit |
| `decision_delay` | Latency from first hazard exposure to first valid action |
| `path_efficiency_ratio` | Straight-line distance ÷ total path length |
| `hazard_avoidance_ratio` | Fraction of timesteps the player stayed at a safe distance |
| `interaction_frequency` | Safety interactions (door, extinguisher, exit) per second |
| `panic_proxy` | Std-dev of consecutive bearing changes (erratic movement indicator) |

A Random Forest classifier maps these features to one of three preparedness levels:

```
preparedness_level ∈ {"High", "Moderate", "Low"}
```

> **Scope:** This repository covers only the ML backend. The Minecraft client/server plugin and any web front-end live in separate repositories.

---

## Repository Structure

```
MiDRR-Classifier/
├── README.md                          # This file
├── requirements.txt                   # Pip-installable dependencies
├── pyproject.toml                     # Poetry project config + pytest settings
├── .gitignore
│
├── data/
│   ├── raw/                           # Place raw gameplay CSV logs here
│   └── processed/                     # Feature tables written here by feature_engineering
│
├── models/                            # Trained model .pkl files and evaluation PNGs
│
├── src/
│   └── midrr_classifier/
│       ├── __init__.py                # Package version
│       ├── config.py                  # MiDRRConfig dataclass + YAML loader
│       ├── data_schema.py             # Column definitions, schema validation helpers
│       ├── data_ingestion.py          # CSV loaders + train/test split
│       ├── feature_engineering.py     # compute_* functions + build_feature_table()
│       ├── model_definition.py        # MiDRRClassifier wrapper (build/fit/save/load)
│       ├── train.py                   # Training pipeline entry-point
│       ├── evaluate.py                # Evaluation pipeline entry-point
│       ├── inference.py               # predict_preparedness() — integration boundary
│       └── utils/
│           ├── logging_utils.py       # Project-wide logging setup
│           └── metrics.py             # compute_classification_metrics(), plot_confusion_matrix()
│
├── notebooks/
│   └── exploration_template.ipynb    # EDA starter notebook (stubs — plug in real data)
│
└── tests/
    ├── test_feature_engineering.py   # Tests for all compute_* functions + build_feature_table
    └── test_model_definition.py      # Tests for MiDRRClassifier build/fit/predict/errors
```

---

## Installation

```bash
git clone <repo-url>
cd MiDRR-Classifier

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# (Optional) install the package in editable mode so imports resolve cleanly
pip install -e .
```

> **Python version:** 3.10 or higher is required (uses `X | Y` union type syntax).

---

## Data Expectations

### Raw gameplay logs — `data/raw/gameplay_logs.csv`

One row per in-game event. Expected columns:

| Column | Type | Description |
|---|---|---|
| `player_id` | `str` | Unique student/player identifier |
| `scenario_type` | `str` | `"fire"` or `"earthquake"` |
| `timestamp` | `float` | Seconds since scenario start |
| `x` | `float` | Player X coordinate |
| `y` | `float` | Player Y coordinate (height) |
| `z` | `float` | Player Z coordinate |
| `event_type` | `str` | One of: `move`, `door_open`, `extinguisher_use`, `emergency_exit`, `hazard_proximity` |
| `hazard_distance` | `float` | Euclidean distance (blocks) to nearest hazard |
| `preparedness_level` | `str` | Run-level label: `High`, `Moderate`, or `Low` |

### Processed feature table — `data/processed/features.csv`

One row per player × scenario run. Produced by `feature_engineering.build_feature_table()`.
Columns: `player_id`, `scenario_type`, plus the six feature columns, plus `preparedness_level`.

---

## Basic Usage

### Step 1 — Engineer features from raw logs

```python
from midrr_classifier.data_ingestion import load_raw_logs
from midrr_classifier.feature_engineering import build_feature_table

raw_df = load_raw_logs("data/raw/gameplay_logs.csv")
feature_df = build_feature_table(raw_df)
feature_df.to_csv("data/processed/features.csv", index=False)
```

### Step 2 — Train the classifier

```bash
python -m midrr_classifier.train
# with a custom YAML config:
python -m midrr_classifier.train --config config.yaml
```

### Step 3 — Evaluate

```bash
python -m midrr_classifier.evaluate
# with explicit paths:
python -m midrr_classifier.evaluate \
    --model models/midrr_rf.pkl \
    --test-csv data/processed/test.csv
```

Outputs a classification report to stdout and saves `models/confusion_matrix.png`.

### Step 4 — Single-row inference

```python
import pandas as pd
from midrr_classifier.inference import predict_preparedness

features = pd.Series({
    "evacuation_time": 42.0,
    "decision_delay": 3.5,
    "path_efficiency_ratio": 0.72,
    "hazard_avoidance_ratio": 0.85,
    "interaction_frequency": 0.12,
    "panic_proxy": 15.3,
})

label = predict_preparedness(features)
print(label)  # "High", "Moderate", or "Low"
```

### Run tests

```bash
pytest tests/ -v
```

---

## Integration Notes

The `predict_preparedness()` function in [src/midrr_classifier/inference.py](src/midrr_classifier/inference.py) is the intended integration boundary for external systems.

**Suggested integration pattern:**

1. Wrap `predict_preparedness()` in a lightweight Flask or FastAPI endpoint (e.g. `POST /predict`).
2. The Minecraft server plugin (Spigot / Paper / Fabric) sends a JSON payload of the six feature values immediately after a simulation run ends.
3. The API returns the predicted preparedness label in the response body.

Real-time REST deployment and Minecraft-side integration code are **out of scope** for this repository and will be developed in a separate `midrr-api` service.

---

## Limitations / TODO

- **Scaffold only:** No real dataset is included. The repository runs tests with synthetic data but training and evaluation require actual gameplay logs.
- **Placeholder feature formulas:** All `compute_*` functions use illustrative implementations. These must be aligned with the precise operational definitions in Chapter 3 of the thesis before final experiments.
- **Hyperparameter defaults:** `n_estimators=100`, `max_depth=None` are initial guesses. Tune via cross-validation once the dataset is available.
- **`SAFE_HAZARD_DISTANCE`:** Currently set to 5 blocks. Calibrate with domain experts.
- **No data versioning:** Consider integrating DVC for tracking dataset and model versions.
- **No REST API:** See integration notes above.

---

## License

For academic research use within Laguna State Polytechnic University (LSPU). Update this notice if the repository is open-sourced.

## Authors

- MiDRR Research Team — LSPU, CMSC 312 Thesis (A.Y. 2025–2026)
