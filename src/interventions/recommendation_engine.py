"""
Intervention Recommendation Engine
=====================================
Stage 1 — Rule Engine: deterministic, fast, always available.
Stage 2 — Gemini LLM: rich counselor-facing narrative (optional).

The combined output for a high-risk student looks like:
─────────────────────────────────────────────────────
Student 1042 | Risk: 78% | Tier: High

⚠️  PRIMARY REASONS:
  • Physics attendance dropped 63% over the past 14 days
  • Test accuracy declined from 72% → 51% (past 14 days)
  • No active session for 6 days

📋  RECOMMENDED INTERVENTIONS:
  1. Schedule a 1:1 mentor call within 48 hours
  2. Enrol student in 2-day Physics Recovery Plan
  3. Send personalised re-engagement push notification

📊  EXPECTED OUTCOME: High probability of re-engagement if
     mentor call completed within 72 hours.
─────────────────────────────────────────────────────
"""

import logging
import os
from pathlib import Path
from dataclasses import dataclass

import httpx
import yaml
from dotenv import dotenv_values

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
with (ROOT / "config.yaml").open(encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

INT_CFG = CFG["interventions"]["rules"]
USE_LLM = CFG["gemini"]["use_llm_narratives"]
GEMINI_MODEL = CFG["gemini"]["model"]


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class InterventionResult:
    student_id:          int
    risk_score:          float
    risk_tier:           str
    top_reasons:         list[str]        # SHAP narratives
    interventions:       list[str]        # Rule-based recommendations
    urgency_hours:       int              # SLA for counselor action
    llm_narrative:       str = ""         # Gemini-generated rich text
    intervention_type:   str = ""         # e.g. "mentor_call+recovery_plan"


@dataclass(frozen=True)
class CounselorBrief:
    text: str
    source: str


# ── Rule Engine ────────────────────────────────────────────────────────────────

RULES: list[dict] = [
    {
        "id":          "score_drop_mentor",
        "condition":   lambda f: (
            f.get("score_delta_14d", 0) < -INT_CFG["score_drop_threshold"] * 50
            and f.get("attend_rate_14d", 1) < (1 - INT_CFG["attendance_drop_threshold"])
        ),
        "action":      "Schedule a 1:1 mentor call within 48 hours",
        "type":        "mentor_call",
        "urgency_h":   48,
    },
    {
        "id":          "physics_recovery",
        "condition":   lambda f: f.get("physics_attend_drop_pct", 0) > 0.40,
        "action":      "Enrol student in 2-day Physics Recovery Plan",
        "type":        "recovery_plan_physics",
        "urgency_h":   72,
    },
    {
        "id":          "chemistry_recovery",
        "condition":   lambda f: f.get("chemistry_attend_drop_pct", 0) > 0.40,
        "action":      "Enrol student in 2-day Chemistry Recovery Plan",
        "type":        "recovery_plan_chemistry",
        "urgency_h":   72,
    },
    {
        "id":          "math_recovery",
        "condition":   lambda f: f.get("mathematics_attend_drop_pct", 0) > 0.40,
        "action":      "Enrol student in 2-day Mathematics Recovery Plan",
        "type":        "recovery_plan_mathematics",
        "urgency_h":   72,
    },
    {
        "id":          "biology_recovery",
        "condition":   lambda f: f.get("biology_attend_drop_pct", 0) > 0.40,
        "action":      "Enrol student in 2-day Biology Recovery Plan",
        "type":        "recovery_plan_biology",
        "urgency_h":   72,
    },
    {
        "id":          "inactive_reengagement",
        "condition":   lambda f: f.get("days_since_last_session", 0) >= INT_CFG["days_inactive_threshold"],
        "action":      "Send personalised re-engagement push notification",
        "type":        "push_notification",
        "urgency_h":   24,
    },
    {
        "id":          "doubt_nudge",
        "condition":   lambda f: f.get("doubt_count_14d", 1) == 0,
        "action":      "Invite student to next live doubt-clearing session",
        "type":        "doubt_session_invite",
        "urgency_h":   96,
    },
    {
        "id":          "assignment_nudge",
        "condition":   lambda f: f.get("assignment_rate_14d", 1) < INT_CFG["assignment_completion_low"],
        "action":      "Send assignment nudge with 3-day deadline extension offer",
        "type":        "assignment_nudge",
        "urgency_h":   48,
    },
    {
        "id":          "batch_lag_catchup",
        "condition":   lambda f: f.get("progress_vs_batch", 0) < -INT_CFG["progress_lag_threshold"],
        "action":      "Enrol in weekly batch catch-up session",
        "type":        "batch_catchup",
        "urgency_h":   120,
    },
    {
        "id":          "critical_escalation",
        "condition":   lambda f: f.get("risk_score", 0) >= 0.80,
        "action":      "Escalate to senior academic counselor immediately",
        "type":        "senior_escalation",
        "urgency_h":   12,
    },
]


def apply_rules(features: dict) -> tuple[list[str], list[str], int]:
    """
    Apply all rules to a student's feature dict.
    Returns (actions, types, min_urgency_hours).
    """
    actions, types, urgencies = [], [], []
    for rule in RULES:
        try:
            if rule["condition"](features):
                actions.append(rule["action"])
                types.append(rule["type"])
                urgencies.append(rule["urgency_h"])
        except Exception:
            pass  # rule evaluation errors are non-fatal

    if not actions:
        actions   = ["Monitor closely — check in within 7 days"]
        types     = ["monitor"]
        urgencies = [168]

    return actions, types, min(urgencies)


# ── Gemini LLM Narrative ───────────────────────────────────────────────────────

def _build_llm_prompt(
    risk_score: float,
    risk_tier: str,
    reasons: list[str],
    interventions: list[str],
) -> str:
    reasons_str      = "\n".join(f"  - {r}" for r in reasons)
    intervention_str = "\n".join(f"  {i+1}. {v}" for i, v in enumerate(interventions))
    return f"""Write a concise, empathetic brief for an academic mentor.

Risk Score: {risk_score:.0%} probability of disengagement in 14 days
Risk Tier: {risk_tier}

Primary Risk Signals:
{reasons_str}

Recommended Interventions:
{intervention_str}

Write a 3-paragraph counselor brief:
1. Summary of student situation (1-2 sentences, include specific numbers)
2. Why these interventions are recommended (connect to the signals)
3. Follow-up timing where the listed interventions specify it

Use only the supplied facts, signals, and actions. Do not add actions, estimates, or outcome claims.
Be specific, data-driven, and empathetic. Use plain English. Do NOT use bullet points.
Keep total response under 150 words."""


def _gemini_api_key() -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        return key.strip()
    local_env = dotenv_values(ROOT / ".env", interpolate=False, encoding="utf-8-sig")
    return (local_env.get("GEMINI_API_KEY") or local_env.get("GOOGLE_API_KEY")
            or CFG["gemini"]["api_key"] or "").strip()


def gemini_available() -> bool:
    return USE_LLM and bool(_gemini_api_key())


def generate_counselor_brief(
    student_id: int,
    risk_score: float,
    risk_tier: str,
    reasons: list[str],
    interventions: list[str],
) -> CounselorBrief:
    """Draft one brief without sending the student ID or contact details."""
    key = _gemini_api_key() if USE_LLM else ""
    if key and interventions:
        try:
            response = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                headers={"x-goog-api-key": key},
                json={
                    "contents": [{"parts": [{"text": _build_llm_prompt(
                        risk_score, risk_tier, reasons, interventions,
                    )}]}],
                    "generationConfig": {"maxOutputTokens": 512},
                },
                timeout=12.0,
            )
            response.raise_for_status()
            candidate = response.json()["candidates"][0]
            if candidate.get("finishReason") == "STOP":
                text = "\n".join(part["text"] for part in candidate["content"]["parts"]
                                 if "text" in part and not part.get("thought")).strip()
                if text:
                    return CounselorBrief(text, "gemini")
            logger.warning("Gemini returned no complete brief. Using rule-based narrative.")
        except httpx.HTTPStatusError as error:
            logger.warning("Gemini returned HTTP %d. Using rule-based narrative.", error.response.status_code)
        except (httpx.RequestError, ValueError, TypeError, KeyError, IndexError) as error:
            logger.warning("Gemini brief failed (%s). Using rule-based narrative.", type(error).__name__)

    return CounselorBrief(
        _rule_based_narrative(student_id, risk_score, risk_tier, reasons, interventions), "rules",
    )


def generate_llm_narrative(
    student_id: int,
    risk_score: float,
    risk_tier: str,
    reasons: list[str],
    interventions: list[str],
) -> str:
    return generate_counselor_brief(student_id, risk_score, risk_tier, reasons, interventions).text


def _rule_based_narrative(
    student_id: int,
    risk_score: float,
    risk_tier: str,
    reasons: list[str],
    interventions: list[str],
) -> str:
    reasons_str      = "; ".join(reasons[:3])
    intervention_str = "; ".join(interventions[:3])
    return (
        f"Student {student_id} has a {risk_score:.0%} probability of disengagement in the next 14 days "
        f"({risk_tier} tier). Key signals: {reasons_str}. "
        f"Recommended actions: {intervention_str}."
    )


# ── Main Entry Point ──────────────────────────────────────────────────────────

def generate_intervention(
    student_id: int,
    risk_score: float,
    risk_tier: str,
    features: dict,
    shap_factors: list[dict],
    use_llm: bool = True,
) -> InterventionResult:
    """
    Generate a full intervention recommendation for a single student.

    Parameters
    ----------
    student_id   : unique student identifier
    risk_score   : calibrated probability (0-1)
    risk_tier    : "Low" / "Medium" / "High" / "Critical"
    features     : dict of feature values for this student
    shap_factors : top-K SHAP explanations (from shap_explainer.explain_student)
    use_llm      : whether to call Gemini (overrides config if False)

    Returns
    -------
    InterventionResult
    """
    features["risk_score"] = risk_score

    # Rule engine
    actions, types, urgency = apply_rules(features)

    # Human-readable reasons from SHAP
    reasons = [f["narrative"] for f in shap_factors]

    # LLM narrative
    narrative = ""
    if use_llm and USE_LLM:
        narrative = generate_llm_narrative(student_id, risk_score, risk_tier, reasons, actions)
    else:
        narrative = _rule_based_narrative(student_id, risk_score, risk_tier, reasons, actions)

    return InterventionResult(
        student_id=student_id,
        risk_score=risk_score,
        risk_tier=risk_tier,
        top_reasons=reasons,
        interventions=actions,
        urgency_hours=urgency,
        llm_narrative=narrative,
        intervention_type="+".join(types),
    )
