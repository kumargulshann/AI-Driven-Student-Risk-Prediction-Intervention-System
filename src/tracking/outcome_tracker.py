"""
Outcome Tracker
================
Logs interventions, re-scores students at 7 and 14 days post-intervention,
and computes success metrics:
  - Intervention Success Rate (risk_score drop ≥ 0.15)
  - Students Recovered per 1,000 Flagged
  - Score improvement post-intervention

Demo outcomes and locally logged mentor activity are stored separately.
"""

import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from datetime import datetime
from tempfile import NamedTemporaryFile

from filelock import FileLock

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
with (ROOT / "config.yaml").open(encoding="utf-8") as f:
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
    "source", "notes",
]


# ── Initialization ─────────────────────────────────────────────────────────────

def _mentor_log() -> Path:
    return OUTCOMES_FILE.with_name("mentor_outcomes.csv")


def _read_outcomes(path: Path) -> pd.DataFrame:
    if path.exists():
        outcomes = pd.read_csv(path).reindex(columns=OUTCOMES_COLS)
        outcomes["source"] = outcomes["source"].fillna("mentor")
        return outcomes
    return pd.DataFrame(columns=OUTCOMES_COLS)


def _load_outcomes() -> pd.DataFrame:
    demo = _read_outcomes(OUTCOMES_FILE)
    mentor = _read_outcomes(_mentor_log())
    if demo.empty or mentor.empty:
        return mentor if demo.empty else demo
    return pd.concat([demo, mentor], ignore_index=True)


def _save_outcomes(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as staging:
        temporary = Path(staging.name)
    try:
        df.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_lock() -> FileLock:
    OUTCOMES_FILE.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(OUTCOMES_FILE.with_suffix(".csv.lock"), timeout=10)


def _load_mentor_log_for_write() -> pd.DataFrame:
    """Migrate entries from old mixed logs when their source was unspecified."""
    mentor = _read_outcomes(_mentor_log())
    old = _read_outcomes(OUTCOMES_FILE)
    legacy = old.loc[old["source"].ne("demo")]
    if legacy.empty:
        return mentor
    merged = pd.concat([legacy, mentor], ignore_index=True) if not mentor.empty else legacy
    merged = merged.drop_duplicates(
        subset=["student_id", "counselor_id", "intervened_at"], keep="last",
    ).reset_index(drop=True)
    _save_outcomes(merged, _mentor_log())
    _save_outcomes(old.loc[old["source"].eq("demo")], OUTCOMES_FILE)
    return merged


# ── Log an Intervention ────────────────────────────────────────────────────────

def log_intervention(
    student_id: int,
    intervention_type: str,
    risk_score_before: float,
    risk_tier_before: str,
    counselor_id: str = "system",
    score_before: float | None = None,
    notes: str | None = None,
) -> None:
    """Record that an intervention was taken for a student."""
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
        "source":             "mentor",
        "notes":              notes,
    }
    with _write_lock():
        outcomes = _load_mentor_log_for_write()
        outcomes.loc[len(outcomes)] = new_row
        _save_outcomes(outcomes, _mentor_log())
    logger.info("Logged intervention for student %d (%s)", student_id, intervention_type)


# ── Update Follow-Up Scores ────────────────────────────────────────────────────

def update_followup(
    student_id: int,
    followup_day: int,
    new_risk_score: float,
    new_test_score: float | None = None,
) -> bool:
    """
    Called 7 or 14 days after intervention to record new risk score.
    Updates recovered flags and score improvement.
    """
    if followup_day not in (7, 14):
        raise ValueError("Follow-up day must be 7 or 14.")
    with _write_lock():
        df = _load_mentor_log_for_write()
        matches = df["student_id"].eq(student_id) & df["source"].eq("mentor")
        if not matches.any():
            return False
        idx = df[matches].index[-1]
        recovered = round(df.at[idx, "risk_score_before"] - new_risk_score, 4) >= RECOVERY_THR
        df.at[idx, f"risk_score_{followup_day}d"] = round(new_risk_score, 4)
        recovered_column = f"recovered_{followup_day}d"
        df[recovered_column] = df[recovered_column].astype("boolean")
        df.at[idx, recovered_column] = recovered

        if new_test_score is not None:
            df.at[idx, "score_after"] = new_test_score
            score_before = df.at[idx, "score_before"]
            if pd.notna(score_before):
                df.at[idx, "score_improvement"] = round(new_test_score - score_before, 2)
        _save_outcomes(df, _mentor_log())
    logger.info(
        "Followup day=%d for student %d | new_risk=%.3f | recovered=%s",
        followup_day, student_id, new_risk_score, recovered,
    )
    return True


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
    rng = np.random.default_rng(42)

    for _, student in at_risk.iterrows():
        sid         = int(student["student_id"])
        score_b     = float(student["risk_score"])
        tier_b      = student["risk_tier"]

        # Simulate intervention effectiveness (higher risk → harder to recover)
        effectiveness = rng.beta(2, 3)
        score_7d  = max(0.0, score_b - effectiveness * 0.30)
        score_14d = max(0.0, score_b - effectiveness * 0.45)

        rows.append({
            "student_id":         sid,
            "intervention_type":  "mentor_call+recovery_plan",
            "risk_score_before":  round(score_b, 4),
            "risk_tier_before":   tier_b,
            "intervened_at":      datetime.now().isoformat(),
            "counselor_id":       f"demo_{rng.integers(1, 11):02d}",
            "risk_score_7d":      round(score_7d, 4),
            "risk_score_14d":     round(score_14d, 4),
            "recovered_7d":       (score_b - score_7d) >= RECOVERY_THR,
            "recovered_14d":      (score_b - score_14d) >= RECOVERY_THR,
            "score_before":       round(rng.uniform(40, 70), 1),
            "score_after":        round(rng.uniform(50, 80), 1),
            "score_improvement":  None,
            "source":             "demo",
        })

    df = pd.DataFrame(rows, columns=OUTCOMES_COLS)
    # Compute score improvement
    df["score_improvement"] = df["score_after"] - df["score_before"]
    with _write_lock():
        _load_mentor_log_for_write()
        _save_outcomes(df, OUTCOMES_FILE)
    logger.info("Simulated %d outcome records", len(df))
    return df


# ── Metric Computation ─────────────────────────────────────────────────────────

def compute_outcome_metrics(source: str | None = None) -> dict:
    """Compute all success metrics from the outcomes log."""
    df = _load_outcomes()
    if source is not None:
        df = df.loc[df["source"].eq(source)]
    if len(df) == 0:
        return {"error": "No outcome data yet"}

    total      = len(df)
    rec_7d     = df["recovered_7d"].dropna()
    rec_14d    = df["recovered_14d"].dropna()
    score_imp  = df["score_improvement"].dropna()

    metrics = {
        "total_interventions":           total,
        "simulated_interventions":       int(df["source"].eq("demo").sum()),
        "mentor_interventions":          int(df["source"].eq("mentor").sum()),
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
