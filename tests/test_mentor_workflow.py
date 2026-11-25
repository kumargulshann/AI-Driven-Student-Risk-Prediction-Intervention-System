from fastapi.testclient import TestClient
import httpx
import pandas as pd

import api.app as api
from src.interventions import recommendation_engine as recommendations
from src.tracking import outcome_tracker as tracker


def test_gemini_uses_env_file_and_omits_student_id(tmp_path, monkeypatch):
    monkeypatch.setattr(recommendations, "ROOT", tmp_path)
    monkeypatch.setattr(recommendations, "USE_LLM", True)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    (tmp_path / ".env").write_text("GEMINI_API_KEY=local-test-key\n", encoding="utf-8")

    def post(url, *, headers, json, timeout):
        assert url.endswith("/gemini-3.5-flash-lite:generateContent")
        assert headers["x-goog-api-key"] == "local-test-key"
        prompt = json["contents"][0]["parts"][0]["text"]
        assert "824657" not in prompt
        assert "Schedule a mentor call" in prompt
        assert timeout == 12.0
        return httpx.Response(200, request=httpx.Request("POST", url), json={
            "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "A mentor call is recommended."}]}}],
        })

    monkeypatch.setattr(recommendations.httpx, "post", post)
    assert recommendations.gemini_available()
    brief = recommendations.generate_counselor_brief(
        824657, 0.76, "High", ["Fewer attended lectures"], ["Schedule a mentor call"],
    )
    assert brief.source == "gemini"
    assert brief.text == "A mentor call is recommended."


def test_gemini_quota_error_falls_back_without_logging_key(monkeypatch, caplog):
    monkeypatch.setattr(recommendations, "USE_LLM", True)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-quota-key")

    def post(url, **kwargs):
        return httpx.Response(429, request=httpx.Request("POST", url), json={"error": "fake-quota-key"})

    monkeypatch.setattr(recommendations.httpx, "post", post)
    brief = recommendations.generate_counselor_brief(21, 0.8, "Critical", ["Less activity"], ["Call mentor"])
    assert brief.source == "rules"
    assert "Call mentor" in brief.text
    assert "fake-quota-key" not in caplog.text


def test_mentor_log_survives_demo_regeneration_and_supports_followup(tmp_path, monkeypatch):
    report_path = tmp_path / "risk_report.csv"
    risk = pd.DataFrame({"student_id": [21], "risk_score": [0.81], "risk_tier": ["Critical"]})
    risk.to_csv(report_path, index=False)
    monkeypatch.setattr(api, "REPORT_PATH", report_path)
    monkeypatch.setattr(api, "_cache", {"report_df": None, "report_version": None, "loaded_at": None})
    monkeypatch.setattr(tracker, "OUTCOMES_FILE", tmp_path / "outcomes.csv")
    tracker.simulate_outcomes(risk)

    with TestClient(api.app) as client:
        followup = "/students/21/intervention/followup"
        assert client.post(followup, json={"day": 7, "risk_score": 0.50}).status_code == 404
        assert client.post("/students/21/intervention", json={
            "intervention_type": "mentor_call", "counselor_id": "mentor_1",
            "score_before": 55, "notes": "Agreed on a study plan",
        }).status_code == 200
        tracker.simulate_outcomes(risk)
        assert client.post(followup, json={"day": 14, "risk_score": 0.50, "test_score": 65}).status_code == 200
        assert client.post(followup, json={"day": 9, "risk_score": 0.50}).status_code == 422
        metrics = client.get("/metrics/outcomes?source=mentor").json()
        assert metrics["mentor_interventions"] == 1
        assert metrics["simulated_interventions"] == 0
        assert metrics["avg_score_improvement"] == 10
        assert metrics["recovery_rate_14d"] == 1

    demo = pd.read_csv(tracker.OUTCOMES_FILE)
    mentor = pd.read_csv(tracker.OUTCOMES_FILE.with_name("mentor_outcomes.csv"))
    assert set(demo["source"]) == {"demo"}
    assert mentor.iloc[0]["notes"] == "Agreed on a study plan"
