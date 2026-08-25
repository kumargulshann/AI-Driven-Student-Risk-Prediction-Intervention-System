"""
FastAPI REST Backend
=====================
Exposes the early-warning system as a REST API for integration
with PW's existing dashboards, CRM, and notification systems.

Endpoints
---------
GET  /health                       - API liveness check
GET  /students/at-risk             - Paginated at-risk student list
GET  /students/{id}/risk-report    - Full risk report for one student
POST /students/{id}/intervention   - Log a counselor intervention
POST /pipeline/run                 - Trigger a full pipeline run
GET  /metrics/model                - Model performance metrics
GET  /metrics/outcomes             - Intervention outcome metrics

Launch:  uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
Docs:    http://localhost:8000/docs
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import logging
import json
import yaml
from datetime import datetime

# ── Setup ─────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

app = FastAPI(
    title="PW Student Early-Warning API",
    description="AI-powered student disengagement prediction and intervention recommendation",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

REPORT_PATH = ROOT / "data" / "processed" / "risk_report.csv"

# ── Cached State ──────────────────────────────────────────────────────────────
_cache: dict = {"report_df": None, "model_metrics": None, "loaded_at": None}


def _load_report() -> pd.DataFrame:
    """Load (or return cached) risk report."""
    if _cache["report_df"] is None or not REPORT_PATH.exists():
        if not REPORT_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail="Risk report not found. Run POST /pipeline/run first.",
            )
        _cache["report_df"]  = pd.read_csv(REPORT_PATH)
        _cache["loaded_at"]  = datetime.now().isoformat()
    return _cache["report_df"]


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class StudentRiskSummary(BaseModel):
    student_id:    int
    risk_score:    float
    risk_tier:     str
    batch_id:      Optional[str]
    city:          Optional[str]
    course:        Optional[str]
    urgency_hours: Optional[int]
    top_reasons:   Optional[str]

class StudentRiskReport(StudentRiskSummary):
    interventions:  Optional[str]
    llm_narrative:  Optional[str]
    intervention_type: Optional[str]

class InterventionLog(BaseModel):
    intervention_type: str = Field(..., json_schema_extra={"example": "mentor_call"})
    counselor_id:      str = Field(..., json_schema_extra={"example": "counselor_01"})
    notes:             Optional[str] = None

class PipelineRunRequest(BaseModel):
    retrain: bool = Field(default=True, description="Retrain model from scratch")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_to_summary(row: pd.Series) -> StudentRiskSummary:
    return StudentRiskSummary(
        student_id=int(row["student_id"]),
        risk_score=round(float(row["risk_score"]), 4),
        risk_tier=str(row.get("risk_tier", "")),
        batch_id=str(row.get("batch_id", "")) or None,
        city=str(row.get("city", "")) or None,
        course=str(row.get("course", "")) or None,
        urgency_hours=int(row["urgency_hours"]) if pd.notna(row.get("urgency_hours")) else None,
        top_reasons=str(row.get("top_reasons", "")) or None,
    )


def _row_to_report(row: pd.Series) -> StudentRiskReport:
    s = _row_to_summary(row)
    return StudentRiskReport(
        **s.model_dump(),
        interventions=str(row.get("interventions", "")) or None,
        llm_narrative=str(row.get("llm_narrative", "")) or None,
        intervention_type=str(row.get("intervention_type", "")) or None,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "message": "PW Student Early-Warning API is running 🚀",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_url": "/health",
        "endpoints": {
            "GET /health": "API liveness & data load status",
            "GET /docs": "Interactive Swagger API documentation",
            "GET /students/at-risk": "List and filter at-risk students",
            "GET /students/{id}/risk-report": "Detailed risk breakdown & recommendations",
            "POST /students/{id}/intervention": "Log counselor intervention",
            "GET /metrics/model": "Model performance metrics",
            "GET /metrics/outcomes": "Intervention outcome & recovery rates",
            "POST /pipeline/run": "Trigger full pipeline re-run",
        }
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "report_loaded": _cache["report_df"] is not None,
        "report_loaded_at": _cache["loaded_at"],
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/students/at-risk", response_model=dict)
def get_at_risk_students(
    tier: Optional[str] = Query(None, description="Filter by tier: Low/Medium/High/Critical"),
    course: Optional[str] = Query(None, description="Filter by course"),
    min_risk: float = Query(0.0, ge=0, le=1),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    df = _load_report()

    # Apply filters
    mask = df["risk_score"] >= min_risk
    if tier:
        mask &= df["risk_tier"].str.lower() == tier.lower()
    if course and "course" in df.columns:
        mask &= df["course"].str.lower() == course.lower()

    filtered = df[mask].sort_values("risk_score", ascending=False)

    total    = len(filtered)
    start    = (page - 1) * page_size
    end      = start + page_size
    page_df  = filtered.iloc[start:end]

    students = [_row_to_summary(row) for _, row in page_df.iterrows()]
    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "students": [s.model_dump() for s in students],
    }


@app.get("/students/{student_id}/risk-report", response_model=StudentRiskReport)
def get_student_risk_report(student_id: int):
    df = _load_report()
    row = df[df["student_id"] == student_id]
    if len(row) == 0:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")
    return _row_to_report(row.iloc[0])


@app.post("/students/{student_id}/intervention")
def log_intervention(student_id: int, body: InterventionLog):
    df = _load_report()
    row = df[df["student_id"] == student_id]
    if len(row) == 0:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")

    from src.tracking.outcome_tracker import log_intervention as _log
    score = float(row.iloc[0]["risk_score"])
    tier  = str(row.iloc[0]["risk_tier"])
    _log(
        student_id=student_id,
        intervention_type=body.intervention_type,
        risk_score_before=score,
        risk_tier_before=tier,
        counselor_id=body.counselor_id,
    )
    return {
        "status": "logged",
        "student_id": student_id,
        "intervention_type": body.intervention_type,
        "counselor_id": body.counselor_id,
        "logged_at": datetime.now().isoformat(),
    }


@app.post("/pipeline/run")
def run_pipeline(body: PipelineRunRequest, background_tasks: BackgroundTasks):
    """Trigger a full pipeline run in the background."""
    def _run():
        try:
            from src.pipeline import run_pipeline as _pipeline
            result = _pipeline(train_model=body.retrain)
            _cache["report_df"]    = result["risk_report"]
            _cache["model_metrics"] = result["model_metrics"]
            _cache["loaded_at"]    = datetime.now().isoformat()
            logger.info("Pipeline run complete")
        except Exception as e:
            logger.error("Pipeline run failed: %s", e)

    background_tasks.add_task(_run)
    return {
        "status": "started",
        "retrain": body.retrain,
        "message": "Pipeline is running in the background. Poll /health for status.",
    }


@app.get("/metrics/model")
def get_model_metrics():
    from src.models.model_registry import load_model, model_exists
    if not model_exists():
        raise HTTPException(status_code=503, detail="No trained model. Run /pipeline/run first.")
    _, _, _, meta = load_model()
    return meta


@app.get("/metrics/outcomes")
def get_outcome_metrics():
    from src.tracking.outcome_tracker import compute_outcome_metrics
    return compute_outcome_metrics()


@app.get("/stats/overview")
def overview_stats():
    df = _load_report()
    tier_counts = df["risk_tier"].value_counts().to_dict()
    return {
        "total_students": int(len(df)),
        "tier_distribution": tier_counts,
        "avg_risk_score": float(df["risk_score"].mean()),
        "critical_count": int(tier_counts.get("Critical", 0)),
        "high_count": int(tier_counts.get("High", 0)),
        "report_generated_at": _cache.get("loaded_at"),
    }
