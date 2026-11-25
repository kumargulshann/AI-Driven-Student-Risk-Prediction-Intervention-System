"""
SHAP Explainability — Root-Cause Analysis
==========================================
Computes SHAP values for each student's risk score and returns
top-K driving factors as human-readable narrative strings.

Example output for a student:
  [
    "Physics attendance dropped 63% over the past 14 days",
    "Test accuracy declined from 72% → 51% (past 14 days)",
    "No app activity in 6 days (batch average: 1.2 days)",
  ]
"""

import shap
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
with (ROOT / "config.yaml").open(encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SUBJECTS = CFG["data"]["subjects"]


# ── Human-Readable Feature Descriptions ───────────────────────────────────────

FEATURE_LABELS: dict[str, str] = {
    # Attendance
    "attend_rate_7d":               "Lecture attendance (last 7 days)",
    "attend_rate_14d":              "Lecture attendance (last 14 days)",
    "attend_rate_30d":              "Lecture attendance (last 30 days)",
    "attend_delta_7_14":            "Attendance drop (7d vs 14d)",
    "completion_rate_14d":          "Lecture completion rate (14 days)",
    # Test Performance
    "test_score_avg_7d":            "Average test score (last 7 days)",
    "test_score_avg_14d":           "Average test score (last 14 days)",
    "test_score_avg_30d":           "Average test score (last 30 days)",
    "test_attempts_7d":             "Test attempts (last 7 days)",
    "test_attempts_14d":            "Test attempts (last 14 days)",
    "score_slope_14d":              "Score trend (14 days) — negative = declining",
    "score_delta_14d":              "Score change: last 7d vs prior 7d",
    # Session
    "days_since_last_session":      "Days since last active session",
    "session_freq_7d":              "Session frequency (last 7 days)",
    "session_freq_14d":             "Session frequency (last 14 days)",
    # Assignment
    "assignment_rate_7d":           "Assignment completion (last 7 days)",
    "assignment_rate_14d":          "Assignment completion (last 14 days)",
    # Doubt
    "doubt_count_7d":               "Doubt questions asked (last 7 days)",
    "doubt_count_14d":              "Doubt questions asked (last 14 days)",
    "doubt_delta_7_14":             "Drop in doubt activity (7d vs 14d)",
    # Revision
    "revision_count_14d":           "Revision sessions (last 14 days)",
    "revision_count_7d":            "Revision sessions (last 7 days)",
    # App
    "app_minutes_7d":               "App usage (last 7 days, minutes)",
    "app_minutes_14d":              "App usage (last 14 days, minutes)",
    "app_delta_7_14":               "Drop in app usage (7d vs 14d)",
    # Peer
    "attend_vs_batch":              "Attendance vs batch average",
    "score_vs_batch":               "Score vs batch average",
    "progress_vs_batch":            "Progress vs batch average",
    "app_vs_batch":                 "App usage vs batch average",
}

# Add subject-level descriptions dynamically
for _subj in SUBJECTS:
    _sl = _subj.lower()
    FEATURE_LABELS[f"{_sl}_attend_14d"]       = f"{_subj} attendance (14 days)"
    FEATURE_LABELS[f"{_sl}_attend_drop_pct"]  = f"{_subj} attendance drop %"
    FEATURE_LABELS[f"{_sl}_score_14d"]        = f"{_subj} score (14 days)"
    FEATURE_LABELS[f"{_sl}_score_7d"]         = f"{_subj} score (7 days)"
    FEATURE_LABELS[f"{_sl}_score_drop_pct"]   = f"{_subj} score drop %"


def get_feature_label(col: str) -> str:
    return FEATURE_LABELS.get(col, col.replace("_", " ").title())


# ── SHAP Explainer ─────────────────────────────────────────────────────────────

class StudentShapExplainer:
    """Wraps a LightGBM model with SHAP for student-level explanations."""

    def __init__(self, model, feature_cols: list[str]):
        # Extract the underlying LightGBM estimator from the calibrated wrapper
        self.feature_cols = feature_cols
        self._lgbm_estimator = _extract_lgbm(model)
        self.explainer = shap.TreeExplainer(self._lgbm_estimator)
        logger.info("SHAP TreeExplainer initialized")

    def explain_batch(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return SHAP values matrix (n_students × n_features)."""
        shap_vals = self.explainer.shap_values(X_scaled)
        # For binary classification LightGBM, shap_values returns list [neg, pos]
        if isinstance(shap_vals, list):
            return shap_vals[1]  # positive class
        return shap_vals

    def top_factors(
        self,
        student_idx: int,
        X_scaled: np.ndarray,
        feat_df_row: pd.Series,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Return top-K risk-driving factors for a single student.

        Returns a list of dicts:
          {"feature": ..., "shap": ..., "label": ..., "narrative": ...}
        """
        sv = self.explain_batch(X_scaled[[student_idx]])[0]
        return self.factors_from_values(sv, feat_df_row, top_k)

    def factors_from_values(self, sv, feat_df_row, top_k=3):
        """Format the features with the largest contributions toward higher risk."""
        indices = np.argsort(-sv, kind="stable")[:top_k]

        factors = []
        for i in indices:
            col   = self.feature_cols[i]
            label = get_feature_label(col)
            val   = feat_df_row[col] if col in feat_df_row.index else None
            factors.append({
                "feature":   col,
                "shap":      float(sv[i]),
                "label":     label,
                "value":     val,
                "narrative": _to_narrative(col, val),
            })
        return factors


def _extract_lgbm(calibrated_model):
    """Extract the LightGBM booster from a CalibratedClassifierCV."""
    try:
        # CalibratedClassifierCV stores a list of calibrated classifiers
        return calibrated_model.calibrated_classifiers_[0].estimator
    except AttributeError:
        return calibrated_model


def _to_narrative(col: str, val) -> str:
    """Describe the observed value in its original units."""
    label = get_feature_label(col)

    if pd.isna(val):
        return f"{label} — data unavailable"

    val_f = float(val)

    if col == "score_delta_14d":
        return f"{label}: {val_f:+.1f} points"
    if col == "score_slope_14d":
        return f"{label}: {val_f:+.1f} points per observation"
    if col.endswith("_vs_batch"):
        if col.startswith("app"):
            return f"{label}: {val_f:+.1f} minutes"
        difference = val_f if col.startswith("score") else val_f * 100
        return f"{label}: {difference:+.1f} percentage points"
    if "drop_pct" in col or "delta" in col:
        return f"{label}: {val_f:.0%}"
    if "days_since" in col:
        return f"Days since last attended lecture: {int(val_f)}"
    if "count" in col or "attempts" in col or "freq" in col or "revision" in col:
        return f"{label}: {int(val_f)} times"
    if "score" in col:
        return f"{label}: {val_f:.1f}%"
    if "rate" in col or "_attend_" in col:
        return f"{label}: {val_f:.0%}"
    if "app_minutes" in col:
        return f"{label}: {val_f:.0f} minutes"

    return f"{label}: {val_f:.2f}"


# ── Convenience: Full Explanation for One Student ─────────────────────────────

def explain_student(
    student_id: int,
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    explainer: StudentShapExplainer,
    scaler,
    top_k: int = 3,
) -> list[dict]:
    """Return top-K explanations for a single student."""
    row_idx = feat_df.index[feat_df["student_id"] == student_id]
    if len(row_idx) == 0:
        return []
    idx_in_df = row_idx[0]
    pos = list(feat_df.index).index(idx_in_df)

    X = feat_df[feature_cols].values.astype(np.float32)
    X_s = scaler.transform(X)

    feat_row = feat_df.loc[idx_in_df]
    return explainer.top_factors(pos, X_s, feat_row, top_k=top_k)
