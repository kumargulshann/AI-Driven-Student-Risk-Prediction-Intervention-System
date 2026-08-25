"""
Outcome Tracker
================
Logs interventions, re-scores students at 7 and 14 days post-intervention,
and computes success metrics:
  - Intervention Success Rate (risk_score drop ≥ 0.15)
  - Students Recovered per 1,000 Flagged
  - Score improvement post-intervention

Data is persisted to data/outcomes.csv.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

OT_CFG         = CFG["outcome_tracking"]
OUTCOMES_FILE  = ROOT / OT_CFG["outcomes_file"]
RECOVERY_THR   = OT_CFG["recovery_threshold"]   # 0.15
FOLLOWUP_DAYS  = OT_CFG["followup_days"]         # [7, 14]

OUTCOMES_COLS = [
    "student_id", "intervention_type", "risk_score_before",
    "risk_tier_before", "intervened_at", "counselor_id",
    "risk_score_7d", "risk_score_14d",
    "recovered_7d", "recovered_14d",
    "score_before", "score_after", "score_improvement",
]


# ── Initialization ─────────────────────────────────────────────────────────────

def _load_outcomes() -> pd.DataFrame:
    if OUTCOMES_FILE.exists():
        return pd.read_csv(OUTCOMES_FILE, parse_dates=["intervened_at"])
    return pd.DataFrame(columns=OUTCOMES_COLS)


def _save_outcomes(df: pd.DataFrame) -> None:
    OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTCOMES_FILE, index=False)


# ── Log an Intervention ────────────────────────────────────────────────────────

def log_intervention(
    student_id: int,
    intervention_type: str,
    risk_score_before: float,
    risk_tier_before: str,
    counselor_id: str = "system",
    score_before: float | None = None,
) -> None:
    """Record that an intervention was taken for a student."""
    df = _load_outcomes()

    new_row = {
        "student_id":         student_id,
        "intervention_type":  intervention_type,
        "risk_score_before":  round(risk_score_before, 4),
        "risk_tier_before":   risk_tier_before,
        "intervened_at":      datetime.now().isoformat(),
        "counselor_id":       counselor_id,
        "risk_score_7d":      None,
        "risk_score_14d":     None,
        "recovered_7d":       None,
        "recovered_14d":      None,
        "score_before":       score_before,
        "score_after":        None,
        "score_improvement":  None,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_outcomes(df)
    logger.info("Logged intervention for student %d (%s)", student_id, intervention_type)


# ── Update Follow-Up Scores ────────────────────────────────────────────────────

def update_followup(
    student_id: int,
    followup_day: int,
    new_risk_score: float,
    new_test_score: float | None = None,
) -> None:
    """
    Called 7 or 14 days after intervention to record new risk score.
    Updates recovered flags and score improvement.
    """
    df = _load_outcomes()
    mask = df["student_id"] == student_id
    if not mask.any():
        logger.warning("No intervention record found for student %d", student_id)
        return

    # Update the most recent intervention for this student
    idx = df[mask].index[-1]
    score_before = df.at[idx, "risk_score_before"]
    drop         = score_before - new_risk_score
    recovered    = drop >= RECOVERY_THR

    if followup_day == 7:
        df.at[idx, "risk_score_7d"]  = round(new_risk_score, 4)
        df.at[idx, "recovered_7d"]   = recovered
    elif followup_day == 14:
        df.at[idx, "risk_score_14d"] = round(new_risk_score, 4)
        df.at[idx, "recovered_14d"]  = recovered

    if new_test_score is not None:
        df.at[idx, "score_after"]       = new_test_score
        score_b = df.at[idx, "score_before"]
        if score_b is not None and not np.isnan(float(score_b)):
            df.at[idx, "score_improvement"] = round(new_test_score - float(score_b), 2)

    _save_outcomes(df)
    logger.info(
        "Followup day=%d for student %d | new_risk=%.3f | recovered=%s",
        followup_day, student_id, new_risk_score, recovered,
    )


# ── Simulate Outcome Data (for demo) ──────────────────────────────────────────

def simulate_outcomes(risk_df: pd.DataFrame, n_intervened: int = 200) -> pd.DataFrame:
    """
    Generate synthetic outcome data for demonstration.
    Simulates that interventions reduce risk scores by varying degrees.
    """
    OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Select top-N at-risk students
    at_risk = risk_df[risk_df["risk_tier"].isin(["High", "Critical"])].head(n_intervened)
    rows = []

    for _, student in at_risk.iterrows():
        sid         = int(student["student_id"])
        score_b     = float(student["risk_score"])
        tier_b      = student["risk_tier"]

        # Simulate intervention effectiveness (higher risk → harder to recover)
        effectiveness = np.random.beta(2, 3)  # skewed towards partial recovery
        score_7d  = max(0.0, score_b - effectiveness * 0.30)
        score_14d = max(0.0, score_b - effectiveness * 0.45)

        rows.append({
            "student_id":         sid,
            "intervention_type":  "mentor_call+recovery_plan",
            "risk_score_before":  round(score_b, 4),
            "risk_tier_before":   tier_b,
            "intervened_at":      datetime.now().isoformat(),
            "counselor_id":       f"counselor_{np.random.randint(1, 11):02d}",
            "risk_score_7d":      round(score_7d, 4),
            "risk_score_14d":     round(score_14d, 4),
            "recovered_7d":       (score_b - score_7d) >= RECOVERY_THR,
            "recovered_14d":      (score_b - score_14d) >= RECOVERY_THR,
            "score_before":       round(np.random.uniform(40, 70), 1),
            "score_after":        round(np.random.uniform(50, 80), 1),
            "score_improvement":  None,
        })

    df = pd.DataFrame(rows)
    # Compute score improvement
    df["score_improvement"] = df["score_after"] - df["score_before"]
    _save_outcomes(df)
    logger.info("Simulated %d outcome records", len(df))
    return df


# ── Metric Computation ─────────────────────────────────────────────────────────

def compute_outcome_metrics() -> dict:
    """Compute all success metrics from the outcomes log."""
    df = _load_outcomes()
    if len(df) == 0:
        return {"error": "No outcome data yet"}

    total      = len(df)
    rec_7d     = df["recovered_7d"].dropna()
    rec_14d    = df["recovered_14d"].dropna()
    score_imp  = df["score_improvement"].dropna()

    metrics = {
        "total_interventions":           total,
        "recovery_rate_7d":              float(rec_7d.mean())  if len(rec_7d)  else 0.0,
        "recovery_rate_14d":             float(rec_14d.mean()) if len(rec_14d) else 0.0,
        "recovered_per_1000_flagged_7d": float(rec_7d.mean()  * 1000) if len(rec_7d)  else 0.0,
        "recovered_per_1000_flagged_14d":float(rec_14d.mean() * 1000) if len(rec_14d) else 0.0,
        "avg_score_improvement":         float(score_imp.mean()) if len(score_imp) else 0.0,
        "pct_score_improved":            float((score_imp > 0).mean()) if len(score_imp) else 0.0,
    }
    return metrics


def get_outcomes_df() -> pd.DataFrame:
    return _load_outcomes()
