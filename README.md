# PW Student Early-Warning System

> AI-powered pipeline that predicts student disengagement **7–14 days in advance** and recommends precise interventions.

## Example Output

```
Student 1042 | Risk: 78% | Tier: High | Urgency: 48h

⚠️  PRIMARY RISK SIGNALS:
  • Physics attendance dropped 63% over the past 14 days
  • Test accuracy declined from 72% → 51% (past 14 days)
  • No active session for 6 days

📋  RECOMMENDED INTERVENTIONS:
  1. Schedule a 1:1 mentor call within 48 hours
  2. Enrol student in 2-day Physics Recovery Plan
  3. Send personalised re-engagement push notification
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full pipeline
```bash
python src/pipeline.py
```
*(Generates synthetic data → trains model → scores students → saves risk report)*

### 3. Launch the Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```
Open: http://localhost:8501

### 4. Launch the REST API
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```
Docs: http://localhost:8000/docs

### 5. Run Tests
```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Architecture

```
Student Behavioral Data
        │
        ▼
Feature Engineering (40+ rolling-window features)
        │
        ▼
LightGBM Risk Model + Platt Calibration
        │
        ├──► Risk Score + Tier (Low/Medium/High/Critical)
        │
        ▼
SHAP Root-Cause Analysis
        │
        ├──► Top-3 Human-Readable Risk Factors
        │
        ▼
Intervention Engine (Rules + Gemini LLM)
        │
        ├──► Specific Actions + Counselor Narrative
        │
        ▼
Outcome Tracker (7d/14d follow-up re-scoring)
```

---

## Project Structure

```
├── data/
│   ├── synthetic_generator.py    # Generates 5,000 student records
│   ├── raw/                      # Generated CSVs
│   └── processed/                # risk_report.csv
│
├── src/
│   ├── ingestion/data_loader.py          # Load & validate data
│   ├── features/feature_engineering.py  # 40+ rolling-window features
│   ├── models/
│   │   ├── risk_model.py                # LightGBM + calibration
│   │   └── model_registry.py            # Save/load artifacts
│   ├── explainability/shap_explainer.py # SHAP root-cause analysis
│   ├── interventions/recommendation_engine.py  # Rules + Gemini LLM
│   ├── tracking/outcome_tracker.py      # Outcome logging & KPIs
│   └── pipeline.py                      # End-to-end orchestrator
│
├── api/app.py                    # FastAPI REST backend
├── dashboard/app.py              # Streamlit counselor UI
├── tests/                        # pytest test suite
├── config.yaml                   # All tunable thresholds
└── requirements.txt
```

---

## Feature Groups (40+ features)

| Group | Features |
|---|---|
| Attendance | `attend_rate_7d/14d/30d`, `attend_delta_7_14`, `completion_rate_14d` |
| Test Performance | `test_score_avg_7d/14d/30d`, `score_slope_14d`, `score_delta_14d` |
| Session Behavior | `days_since_last_session`, `session_freq_7d/14d` |
| Assignment | `assignment_rate_7d/14d` |
| Doubt Activity | `doubt_count_7d/14d`, `doubt_delta_7_14` |
| Revision | `revision_count_7d/14d` |
| App Engagement | `app_minutes_7d/14d`, `app_delta_7_14` |
| Peer Comparison | `attend_vs_batch`, `score_vs_batch`, `progress_vs_batch` |
| Subject-Level | `physics_attend_drop_pct`, `chemistry_score_drop_pct`, etc. |

---

## Success Metrics

| Metric | Measurement |
|---|---|
| AUC-ROC | Evaluated on 20% held-out test set |
| Precision/Recall at 50% threshold | Classification report |
| CV AUC (5-fold) | Cross-validation |
| Recovery rate @7d/@14d | Outcome tracker |
| Students recovered per 1,000 flagged | Dashboard KPI |
| Avg score improvement | Pre/post intervention comparison |

---

## Configuration

Edit `config.yaml` to tune:
- **Gemini API key** and model
- **Risk tier thresholds** (Low/Medium/High/Critical)
- **Intervention rules** (drop thresholds, inactivity thresholds)
- **Model hyperparameters** (n_estimators, learning_rate, etc.)
- **Outcome tracking** (recovery threshold, follow-up days)
