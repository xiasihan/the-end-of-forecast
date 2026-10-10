# EndForecast Web Console — User Guide

The EndForecast web console is a production-grade ML prediction platform. Upload data, describe your goal, and watch the entire 10-phase pipeline execute in real-time with streaming output, LLM decision logs, and a deployable API.

## Table of Contents

- [Starting the Platform](#starting-the-platform)
- [Interface Overview](#interface-overview)
- [Creating a Project](#creating-a-project)
- [Pipeline Execution](#pipeline-execution)
- [Dashboard & Results](#dashboard--results)
- [Deployment API](#deployment-api)
- [Internationalization](#internationalization)

---

## Starting the Platform

### Docker (Recommended)
```bash
docker-compose up
# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

### Manual
```bash
# Terminal 1 — Backend
cd endforecast
uvicorn endforecast.web.app:app --reload --port 8000

# Terminal 2 — Frontend
cd frontend
npm install
npm run dev
# Open http://localhost:3000
```

---

## Interface Overview

The platform uses a three-column layout inspired by professional developer tools:

```
┌──────────────────────────────────────────────────────────────────┐
│  🔮 EndForecast                     [🌐 EN/中文]  [API Docs]    │  ← Header (48px)
├──────────┬──────────────────────────────┬───────────────────────┤
│          │                              │                       │
│ PROJECT  │      Main Terminal Area      │    Context Panel      │
│ LIST     │                              │                       │
│          │  ┌────────────────────────┐  │  [▶ Run Pipeline]     │
│ ┌──────┐ │  │                        │  │                       │
│ │+ New │ │  │  Phase Output Cards    │  │  ● Phase Progress     │
│ └──────┘ │  │  (streaming, expand)   │  │    P0 ✓ Requirements  │
│          │  │                        │  │    P1 ✓ Understanding │
│ Project1 │  │  ┌──────────────────┐  │  │    P2 ✓ Preparation  │
│ (active) │  │  │ Phase 4: Baseline│  │  │    P3 ✓ LeakGuard    │
│          │  │  │ ✅ completed     │  │  │    P4 ● Baseline     │
│ Project2 │  │  │ naive=18.48      │  │  │    P5 — Experiment   │
│          │  │  └──────────────────┘  │  │    ...               │
│ Project3 │  │                        │  │                       │
│          │  │  ┌──────────────────┐  │  │  Live Metrics         │
│          │  │  │ Phase 5 ● Running│  │  │  Validation: 0.52    │
│          │  │  │ Round 2/3        │  │  │  Holdout:    0.70   │
│          │  │  └──────────────────┘  │  │  Best Model: lgbm   │
│          │  │                        │  │                       │
│          │  │  ▼ auto-scroll         │  │  🤖 LLM Decisions    │
│          │  └────────────────────────┘  │  ┌──────────────────┐│
│          │                              │  │ configurator:    ││
│ 240px    │           flex-1             │  │ holdout=chrono.. ││
│          │                              │  ├──────────────────┤│
│          │                              │  │ planner: 4 models││
│          │                              │  └──────────────────┘│
│          │                              │        320px          │
├──────────┴──────────────────────────────┴───────────────────────┤
│  proj_anhui | ● Running | Best: lightgbm | Holdout: 0.70  [⏏] │  ← Status Bar
└──────────────────────────────────────────────────────────────────┘
```

### Key Regions

| Region | Width | Purpose |
|--------|-------|---------|
| **Header** | Full | Logo, language toggle (🌐 EN/中文), API Docs link |
| **Sidebar** | 240px | Project list, search, new project button |
| **Terminal Area** | Flexible | Streaming phase cards during execution, dashboard afterward |
| **Context Panel** | 320px | Run button, phase progress tracker, live metrics, LLM decision log |
| **Status Bar** | Full | Project ID, run status, best model, holdout score, export button |

---

## Creating a Project

Click **New Project** in the sidebar to open the creation dialog:

```
┌─────────────────────────────────────────────┐
│  New Project                            [✕] │
│                                             │
│  Project Name                               │
│  ┌─────────────────────────────────────────┐│
│  │ e.g. Anhui Spread Prediction            ││
│  └─────────────────────────────────────────┘│
│                                             │
│  Prediction Goal                            │
│  ┌─────────────────────────────────────────┐│
│  │ e.g. Predict whether Anhui electricity  ││
│  │ price spread will go up or down         ││
│  └─────────────────────────────────────────┘│
│                                             │
│  Target Column  Time Column  ID (optional)  │
│  ┌──────┐       ┌──────┐     ┌──────┐      │
│  │ y    │       │ ds   │     │      │      │
│  └──────┘       └──────┘     └──────┘      │
│                                             │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┐  │
│  │        📁 Drop CSV/Parquet file       │  │
│  │        or click to browse             │  │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┘  │
│                                             │
│  [ Cancel ]    [ Create & Run Pipeline ]    │
└─────────────────────────────────────────────┘
```

1. **Project Name** — A human-readable identifier.
2. **Prediction Goal** — Describe your problem in natural language. This text is sent to the LLM agents to guide model selection, preprocessing, and feature engineering.
3. **Column Mapping** — Specify which columns are the target (`y`), timestamp (`ds`), and optional group ID.
4. **Upload Data** — Drag-and-drop or click to browse. Supports CSV, Parquet, and Excel files.
5. Click **Create & Run Pipeline** — the project is created and the pipeline starts immediately.

---

## Pipeline Execution

Once the pipeline starts, the terminal area streams output in real-time.

### Phase Cards

Each of the 10 phases appears as a collapsible card:

```
┌─ Phase 1: Data Understanding ──── 2.3s ── ✅ completed ── [▼] ─┐
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │ Task Type│ │ Rows     │ │ Columns  │ │ Missing  │          │
│  │ classif. │ │ 6,360    │ │ 3        │ │ 0.0%     │          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                  │
│  [label_noise]                                                   │
│                                                                  │
│  ┌─ Fingerprint Diagnostic ───────────────────────────────────┐ │
│  │ Stationarity: ADF=-0.25 — fails to reject unit-root.       │ │
│  │ Suggest first-order differencing.                          │ │
│  │ Autocorrelation: moderate (ACF1=0.588). Lag features help. │ │
│  │ Sparsity: 61% zeros. Consider zero-inflated models.        │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

**Card behavior:**
- **Running phases**: Auto-expanded, pulsing yellow border, spinner icon.
- **Completed phases**: Collapsed to summary line; click header to expand.
- **Failed phases**: Red border, error message always visible.
- Phase-specific content rendered based on phase type (baseline tables, fingerprint diagnostics, holdout scores).

### Phase Progress Tracker (Right Panel)

The right panel shows a vertical stepper that updates in real-time:

```
Phase Progress
 ● P0 ✓ Requirements Gathering   1.2s
 ● P1 ✓ Data Understanding       2.3s
 ● P2 ✓ Data Preparation         0.1s
 ● P3 ✓ LeakGuard Check          0.5s
 ● P4 ● Baseline Establishment   1.8s    ← currently running
 ○ P5 — Experiment Execution
 ○ P6 — Evaluation & Error Analysis
 ○ P7 — Iterative Refinement
 ○ P8 — Interpretation
 ○ P9 — Pipeline Construction
 ○ P10 — Drift Detection
```

### Rounds Within Phase 5

Phase 5 runs multiple experiment rounds. Each round produces a `round_complete` event visible in the LLM Decision log:

```
🤖 LLM Decisions
┌──────────────────────────────────────────────┐
│ configurator                                 │
│ holdout=chronological, cv=time_series,       │
│ route=fast, metric=mase                      │
│ ──────────────────────────────────────────── │
│ planner                                      │
│ 4 models, 3 feature combos                   │
│ Lag features essential with ACF1=0.85...     │
│ ──────────────────────────────────────────── │
│ diagnostician (Round 1)                      │
│ Ridge MASE 4.44 vs baseline 18.48...         │
└──────────────────────────────────────────────┘
```

---

## Dashboard & Results

When the pipeline completes, the terminal area switches to the Dashboard view:

```
┌──────────────────────────────────────────────────────────────┐
│  📊 Run Summary                                      [Export]│
├─────────────┬─────────────┬──────────────┬──────────────────┤
│ Validation  │ Holdout     │ Best Model   │ Rounds           │
│ ─────────── │ ─────────── │ ─────────────│ ───────────      │
│ 0.5168      │ 1.0000      │ logistic     │ 3                │
│ Best model  │ Unbiased    │ Selected     │ Experiment       │
│ on validat. │ estimate    │ algorithm    │ rounds           │
├─────────────┴─────────────┴──────────────┴──────────────────┤
│                                                              │
│  🔮 Deployment API                                           │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ curl -X POST .../api/projects/PROJ_ID/predict \        │  │
│  │   -H "Authorization: Bearer YOUR_API_KEY" \            │  │
│  │   -H "Content-Type: application/json" \                │  │
│  │   -d '{"data": [...]}'                                 │  │
│  └────────────────────────────────────────────────────────┘  │
│  [📋 Copy] [📥 Download predictor.py]                       │
│                                                              │
│  🤖 LLM Decisions                                            │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ configurator  holdout=chronological, route=fast...     │  │
│  │ planner       4 models, 3 feature combos               │  │
│  │ diagnostician Ridge MASE 4.44, stable across seeds     │  │
│  │ refiner       residual_correction (bias=+255.7)        │  │
│  │ ensemble      median_ensemble → 5 models               │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### Status Bar

The bottom bar shows live run status:

```
proj_anhui | ○ Idle | Best: logistic | Holdout: 1.0000 | 0 errors   [⏏ Export]
```

---

## Deployment API

After a successful run, you can deploy the model as a production API:

### 1. Create an API Key

```bash
curl -X POST http://localhost:8000/api/keys \
  -H "Content-Type: application/json" \
  -d '{"name": "production-key"}'

# Response:
# {"id": "abc123", "name": "production-key", "prefix": "ef_XyZ...",
#  "raw_key": "ef_XyZ1234567890abcdef..."}  ← Save this! Shown only once.
```

### 2. Make Predictions

```bash
curl -X POST http://localhost:8000/api/projects/PROJECT_ID/predict \
  -H "Authorization: Bearer ef_XyZ1234567890abcdef..." \
  -H "Content-Type: application/json" \
  -d '{"data": [
    {"ds": "2026-09-01 00:00:00", "y": 362.0}
  ]}'
```

### 3. Download Standalone Predictor

Click **Download predictor.py** in the dashboard, or:

```bash
curl http://localhost:8000/api/projects/PROJECT_ID/runs/RUN_ID/export \
  -o predictor.py
```

The exported file is a fully self-contained Python script with the trained model, preprocessing pipeline, and refinement — ready for CI/CD or manual deployment.

---

## Internationalization

The platform supports English and Chinese, with English as the default.

**To switch languages**:
- Click the 🌐 **EN/中文** button in the top header.
- Your preference is saved to `localStorage` and persists across sessions.

All interface text localizes automatically including:
- Navigation and buttons
- Phase names (10 phases)
- Pipeline status messages
- Dashboard metrics
- LLM agent decision labels
- Error messages and tooltips

To add a new language, create `frontend/src/locales/{lang}.json` following the structure of `en.json`.

---

## Project File Locations

| File | Purpose |
|------|---------|
| `endforecast_sessions/proj_*/meta.json` | Project metadata |
| `endforecast_sessions/proj_*/data.csv` | Uploaded data |
| `endforecast_sessions/proj_*/runs/run_*/result.json` | Full pipeline result |
| `endforecast_sessions/proj_*/runs/run_*/predictor.py` | Exported predictor |
| `endforecast_sessions/sessions.db` | SQLite index (projects + runs + API keys) |