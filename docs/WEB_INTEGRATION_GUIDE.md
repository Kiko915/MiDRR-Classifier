# MiDRR API — Web Dashboard Integration Guide

For whoever is sitting in `BERONG_SMP_WEB` wiring this up (probably future-you).
This is the cross-repo contract doc for **that repo → this API**, the same way
`telemetry_contract.md` is the contract doc for **the mod → this repo**.

Think of this API the way you'd think of any third-party REST API you're
integrating into a Next.js app: it has endpoints, JSON contracts, error
codes, and a couple of gotchas you need to know about before you start
wiring up `fetch` calls.

---

## ⚠️ Read this first: the live model is synthetic, not real

The deployed model was trained on **fabricated Minecraft sessions**
(`scripts/bootstrap_synthetic_model.py`, seed 42), not real students —
because Necookie's mod hasn't shipped the real telemetry yet (Phase 3 of
`docs/MiDRR_ML_Development_Plan.md` is still blocked on that). The API
contract, response shapes, and error behavior are all real and stable — you
can build the entire dashboard against it right now. **The actual
`prepLevel`/`prepScore`/`resultText` values it returns are placeholder
demo data**, not meaningful assessments. Don't present them to real users as
real results. This gets replaced by retraining on real data once Phase 3/4
lands — no API contract changes expected when that happens.

---

## Base URL

| Environment | URL |
|---|---|
| Production (Render) | `https://midrr-api.onrender.com` |
| Local dev | `http://localhost:8000` (see this repo's README for `uvicorn api.main:app --reload`, or `docker run -p 8000:8000 midrr-api`) |

Interactive docs (try requests straight from the browser, no Postman needed):
- Swagger UI: `{BASE_URL}/docs`
- ReDoc: `{BASE_URL}/redoc`

**Cold starts:** Render's free tier spins the service down after ~15 min
idle. The first request after that takes several seconds while it wakes up.
If a live defense demo needs zero cold-start lag, either ping `/health`
from an UptimeRobot monitor beforehand, or hit it yourself a minute before
you need it.

---

## Two things to decide before you write fetch calls

### 1. CORS is not configured yet

There is currently **no CORS middleware** on this API. That means:

- ✅ Calling it from a Next.js **server component / API route / server action**
  (server-to-server request) works fine right now — no browser involved, no CORS.
- ❌ Calling it directly from **client-side JS** (`"use client"` component doing
  `fetch(BASE_URL)`) will be blocked by the browser.

If you want the client-side-fetch pattern, come back to this repo and ask
for CORS middleware to be added for your dashboard's origin
(`api/main.py` — a few lines with FastAPI's `CORSMiddleware`). Cheap to add,
just hasn't been needed until there's an actual frontend calling it.

### 2. No authentication

The API is fully open right now — anyone with the URL can call `/predict`.
Fine for a thesis demo; worth revisiting (API key header, or put it behind
your own backend so the key never reaches the browser) before any public
launch.

---

## Endpoints

### `GET /health`

Liveness check. No auth, no body.

```json
{ "status": "ok" }
```

### `POST /predict` — one finished session, get a full result

Use this when you already have a **complete** session's 9 engineered
features and want the final verdict + feedback text. This is the "batch"
endpoint — the caller must have already computed the features (this repo's
`feature_engineering.build_feature_table()` does that from raw telemetry;
whoever calls this endpoint needs to have run that step first, or be reading
already-engineered features from wherever they're stored).

**Request** (`FeaturesRequest` — see `api/schemas.py`):

```ts
interface PredictRequest {
  player_id: string;
  scenario_type: string; // "fire" | "earthquake" | "ccs_fire" | "ccs_earthquake"
  decision_latency: number;       // seconds, >= 0
  spray_accuracy: number;         // 0–1
  path_efficiency_ratio: number;  // 0–1
  hazard_avoidance_ratio: number; // 0–1
  evacuation_time: number;        // seconds, >= 0
  interaction_frequency: number;  // per second, >= 0
  resource_utilization: number;   // 0–1
  panic_proxy: number;            // >= 0, unbounded above
  situational_awareness: number;  // 0–1
}
```

**Response** (`PredictResponse` — this exact shape is a locked contract,
per `api/schemas.py`'s own docstring: *"Do not rename fields without
updating the web repo"* — i.e. this file and that promise are both about you):

```ts
interface PredictResponse {
  prepLevel: "HIGH" | "MODERATE" | "LOW"; // always uppercase, exact strings
  prepScore: number;   // 0–100, confidence in prepLevel
  featureImportance: { feature: string; weight: number }[]; // sorted by |weight| desc
  resultText: string;  // human-readable adaptive feedback, ready to display as-is
}
```

`featureImportance[].weight` is a **signed SHAP value** for this specific
student, not a global ranking: positive = pushed toward `prepLevel` (a
weakness), negative = pushed away from it (a relative strength). If you want
to visualize direction (e.g. red/green bars), the sign is meaningful and
intentional — don't take absolute values before displaying.

**Example:**

```bash
curl -X POST https://midrr-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "player_id": "stu_0412",
    "scenario_type": "fire",
    "decision_latency": 5.2,
    "spray_accuracy": 0.85,
    "path_efficiency_ratio": 0.9,
    "hazard_avoidance_ratio": 0.95,
    "evacuation_time": 12.0,
    "interaction_frequency": 0.3,
    "resource_utilization": 0.8,
    "panic_proxy": 1.1,
    "situational_awareness": 0.9
  }'
```

**Error responses:**

| Status | When | Body |
|---|---|---|
| `422` | Missing/invalid field (Pydantic validation) | FastAPI's standard validation error shape |
| `503` | No model loaded yet on the server | `{"detail": "Model not available. Train a model first..."}` |

### `POST /session/{session_id}/events` — live, mid-session streaming

Use this for a **real-time "prep meter" while the student is still playing**
— not a finished session. The full wire format (event vocabulary, batching
cadence, lifecycle) is documented in `docs/telemetry_contract.md` §6; this is
the short version.

**Who calls this is an open question worth confirming before you build
around it.** As documented, the *mod* samples at 20 Hz and POSTs batches
here every ~5 seconds, and "the dashboard displays it in real time" — but
nothing here specifies the transport from mod → dashboard for that live
value. Possible shapes:
- The mod calls this API directly and separately pushes the response to the
  web dashboard over some other channel (websocket, polling a shared store).
- The web backend is the one calling this endpoint (receiving raw events
  from the mod via its own channel), and the frontend then just talks to
  your own backend.

Confirm this with whoever owns the mod-side networking before assuming
either shape.

**Request:**

```ts
interface SessionEventsRequest {
  contract_version: string;   // "1.2"
  session_id: string;
  player_id: string;
  scenario_type: string;
  events: {
    timestamp: number;
    event_type: string;       // "session_start" | "move" | "pin_pull" | "ext_spray" | ... (full vocabulary in telemetry_contract.md §4)
    x?: number; y?: number; z?: number;
    hazard_distance?: number;
    [extra: string]: unknown; // event-specific fields (hit_fire, extinguisher_class, phase, nearby_player_count, ...) pass through
  }[];
}
```

Send only **new** events since the last POST — the server keeps a per-session
buffer. `session_start` must be in the first batch; `session_end` must be in
the last one (the server closes the buffer on receiving it).

**Response** (`SessionSnapshotResponse`):

```ts
interface SessionSnapshotResponse {
  session_id: string;
  player_id: string;
  scenario_type: string;
  elapsed_time: number;
  event_count: number;
  is_complete: boolean;              // true once assembly_area_reached is seen
  features: Record<string, number>;  // partial/running feature values — see note below
  prediction: string | null;         // null until a model is loaded (it is loaded here)
  prep_score: number | null;
}
```

Features computed on partial data are meaningful, not garbage: e.g.
`hazard_avoidance_ratio` is "fraction of ticks at safe distance *so far*."
But `resource_utilization`/`spray_accuracy` won't exist in the dict until at
least one relevant event (`pin_pull`/`ext_spray`/`drop_cover_hold`) has
happened — handle missing keys in the UI rather than assuming all 9 are
always present mid-session.

**Important operational limitation:** the session buffer is **in-process
memory**, not a database. If the API restarts (e.g. a Render redeploy, or
the free tier spinning down mid-session) an in-progress session's buffer is
lost. Fine for a thesis demo; would need Redis/Postgres before any real
multi-session, multi-instance deployment.

### `DELETE /session/{session_id}`

Optional explicit cleanup — the server auto-closes on `session_end` anyway.
Returns `{"status": "closed", "session_id": "..."}`.

---

## The 9 features, if you need to render them somewhere

These are what `featureImportance` and the streaming `features` dict are
keyed by. Full operational definitions (fire computation + earthquake
analog for each) live in `src/midrr_classifier/data_schema.py`'s
`FEATURE_DEFINITIONS` — this is the quick-reference version:

| Feature | Meaning | Range |
|---|---|---|
| `decision_latency` | Seconds from scenario start to first valid safety action | ≥ 0 |
| `spray_accuracy` | Fraction of extinguisher sprays that hit the fire (or Drop-Cover-Hold correctness, earthquake) | 0–1 |
| `path_efficiency_ratio` | Straight-line distance ÷ total path length to the assembly area | 0–1 |
| `hazard_avoidance_ratio` | Fraction of time spent at a safe distance from the hazard | 0–1 |
| `evacuation_time` | Seconds from scenario start to reaching the assembly area | ≥ 0 |
| `interaction_frequency` | Qualifying safety interactions per second (extinguisher use *while alone* is a violation, not credit — see CLAUDE.md) | ≥ 0 |
| `resource_utilization` | Correctness of the PASS-technique sequencing (pull-before-spray, etc.) | 0–1 |
| `panic_proxy` | Std-dev of movement speed² — higher means more erratic | ≥ 0, unbounded |
| `situational_awareness` | Composite "read the situation correctly" score | 0–1 |

---

## Label contract

`prepLevel` / `prediction` is always one of the **exact uppercase strings**
`"HIGH"`, `"MODERATE"`, `"LOW"` — never `"High"`, never lowercase.
`tests/test_label_contract.py` in this repo enforces this on the ML side; if
your dashboard does string comparisons or CSS class lookups keyed by level,
match the casing exactly rather than normalizing — that's the established
contract other tooling depends on.

---

## Known limitations to plan around

- **Free tier cold starts** (see above) — first request after idle is slow.
- **Single-process streaming state** — see the `/session/{id}/events` note above.
- **No auth, no CORS yet** — see "Two things to decide" above.
- **Synthetic model** — see the warning at the top. This is the big one.
- **No `/leads` or survey endpoints** — only get built if the team decides
  surveys should flow through this API (currently undecided, tracked in this
  repo's `tasks.md` Phase 7).

---

## Where to look if this doc doesn't answer something

- `api/schemas.py` — the literal Pydantic request/response definitions.
- `api/routes/predict.py`, `api/routes/session.py` — the actual route logic.
- `api/feedback.py` — how `resultText` is generated (SHAP-driven + fixed
  BFP threshold flags), if you ever need to explain *why* a message says
  what it says.
- `docs/telemetry_contract.md` — the mod-facing raw telemetry contract
  (event vocabulary, session metadata, the full streaming spec).
- `src/midrr_classifier/data_schema.py` — `FEATURE_DEFINITIONS`,
  `LABEL_CLASSES`, exact operational definitions.
- This repo's `tasks.md` Phase 7/8 — current status of anything not yet
  built (survey endpoints, feedback payload schema formalization, etc.).
