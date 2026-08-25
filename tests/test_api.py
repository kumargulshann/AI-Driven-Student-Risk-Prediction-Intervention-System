"""
Tests — FastAPI Endpoints
"""

import pytest
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client_with_data(tmp_path_factory):
    """Provide a TestClient with a pre-loaded mock risk report."""
    tmp_dir = tmp_path_factory.mktemp("data")

    # Create mock report
    mock_report = pd.DataFrame({
        "student_id":        [1, 2, 3, 4, 5],
        "risk_score":        [0.85, 0.72, 0.55, 0.30, 0.91],
        "risk_tier":         ["Critical", "High", "Medium", "Low", "Critical"],
        "batch_id":          ["BATCH_001"] * 5,
        "city":              ["Delhi"] * 5,
        "course":            ["JEE Advanced"] * 5,
        "urgency_hours":     [12, 48, 96, 168, 12],
        "top_reasons":       ["Low attendance | Score drop"] * 5,
        "interventions":     ["Mentor call | Recovery plan"] * 5,
        "llm_narrative":     ["Mock narrative"] * 5,
        "intervention_type": ["mentor_call"] * 5,
        "disengaged_in_14d": [1, 1, 0, 0, 1],
        "archetype":         ["at_risk_early"] * 5,
    })

    report_path = tmp_dir / "risk_report.csv"
    mock_report.to_csv(report_path, index=False)

    # Patch the path in the API module
    import api.app as api_module
    api_module.REPORT_PATH = report_path
    api_module._cache["report_df"]  = None  # force reload
    api_module._cache["loaded_at"]  = None

    from api.app import app
    return TestClient(app)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_health_check(client_with_data):
    resp = client_with_data.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_get_at_risk_returns_list(client_with_data):
    resp = client_with_data.get("/students/at-risk")
    assert resp.status_code == 200
    data = resp.json()
    assert "students" in data
    assert "total" in data
    assert data["total"] > 0


def test_get_at_risk_tier_filter(client_with_data):
    resp = client_with_data.get("/students/at-risk?tier=Critical")
    assert resp.status_code == 200
    data = resp.json()
    for s in data["students"]:
        assert s["risk_tier"] == "Critical"


def test_get_at_risk_min_risk_filter(client_with_data):
    resp = client_with_data.get("/students/at-risk?min_risk=0.80")
    assert resp.status_code == 200
    data = resp.json()
    for s in data["students"]:
        assert s["risk_score"] >= 0.80


def test_get_student_report_found(client_with_data):
    resp = client_with_data.get("/students/1/risk-report")
    assert resp.status_code == 200
    data = resp.json()
    assert data["student_id"] == 1
    assert "risk_score" in data
    assert "risk_tier" in data


def test_get_student_report_not_found(client_with_data):
    resp = client_with_data.get("/students/99999/risk-report")
    assert resp.status_code == 404


def test_overview_stats(client_with_data):
    resp = client_with_data.get("/stats/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_students" in data
    assert "tier_distribution" in data
    assert data["total_students"] == 5


def test_pagination(client_with_data):
    resp = client_with_data.get("/students/at-risk?page=1&page_size=2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["students"]) <= 2
