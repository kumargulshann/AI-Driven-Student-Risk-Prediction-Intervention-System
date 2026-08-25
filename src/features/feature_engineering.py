"""
Feature Engineering
====================
Computes rolling-window behavioral features per student
at a given snapshot day. All features are derived from
historical data *before* the snapshot — no data leakage.

Feature groups
--------------
1.  Attendance / Lecture                    (7d, 14d, delta)
2.  Test Performance                        (avg score, slope, attempts)
3.  Session Behavior                        (days_since_last, frequency)
4.  Assignment Completion
5.  Doubt Activity
6.  Revision Behavior
7.  App Engagement
8.  Peer / Batch Comparison                 (progress delta, score percentile)
9.  Subject-Level Signals                   (per-subject attendance & score drop)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

SUBJECTS = CFG["data"]["subjects"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _window(df: pd.DataFrame, day: int, window_days: int) -> pd.DataFrame:
    """Rows within the last `window_days` before (and including) `day`."""
    return df[(df["day"] > day - window_days) & (df["day"] <= day)]


def _score_slope(scores: pd.Series) -> float:
    """OLS slope of test scores over time (positive = improving)."""
    scores = scores.dropna()
    if len(scores) < 2:
        return 0.0
    x = np.arange(len(scores), dtype=float)
    x -= x.mean()
    y = scores.values - scores.mean()
    denom = (x ** 2).sum()
    return float((x * y).sum() / denom) if denom > 0 else 0.0


def _pct_drop(a: float, b: float) -> float:
    """Percentage drop from a to b (0 if a == 0)."""
    if a < 1e-6:
        return 0.0
    return max(0.0, (a - b) / a)


# ── Core Feature Builder ───────────────────────────────────────────────────────

def build_features(
    daily_df: pd.DataFrame,
    subject_dfs: dict[str, pd.DataFrame],
    snapshot_day: int,
    batch_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a feature row for every student in `daily_df` at `snapshot_day`.

    Parameters
    ----------
    daily_df     : aggregated daily signals (output of data_loader.pivot_to_daily)
    subject_dfs  : {subject_name: subject-level daily df}
    snapshot_day : the reference day (features use history before this day)
    batch_df     : pre-computed batch averages per day (optional, derived internally if None)

    Returns
    -------
    pd.DataFrame with one row per student and all features.
    """
    student_ids = daily_df["student_id"].unique()
    feature_rows = []

    # Compute batch-level aggregates once
    batch_avg = _compute_batch_averages(daily_df, snapshot_day)

    for sid in student_ids:
        sdf = daily_df[daily_df["student_id"] == sid].sort_values("day")
        row = _build_student_features(sid, sdf, subject_dfs, snapshot_day, batch_avg)
        feature_rows.append(row)

    feat_df = pd.DataFrame(feature_rows)
    logger.info("Built %d feature rows with %d features", len(feat_df), feat_df.shape[1] - 1)
    return feat_df


def _compute_batch_averages(daily_df: pd.DataFrame, snapshot_day: int) -> dict:
    """Compute batch-level averages in the 14-day window before snapshot."""
    window = _window(daily_df, snapshot_day, 14)
    return {
        "attend_14d":    window["attend_rate"].mean(),
        "score_14d":     window["test_score_mean"].mean(),
        "progress_14d":  window["completion_rate"].mean(),
        "app_min_14d":   window["app_minutes"].mean(),
    }


def _build_student_features(
    sid: int,
    sdf: pd.DataFrame,
    subject_dfs: dict[str, pd.DataFrame],
    snapshot_day: int,
    batch_avg: dict,
) -> dict:
    row: dict = {"student_id": sid}

    # ── 1. Attendance ──────────────────────────────────────────────────────────
    for w in [7, 14, 30]:
        wdf = _window(sdf, snapshot_day, w)
        row[f"attend_rate_{w}d"]   = wdf["attend_rate"].mean()   if len(wdf) else 0.0
        row[f"attend_count_{w}d"]  = wdf["attended_any"].sum()   if len(wdf) else 0

    row["attend_delta_7_14"]   = _pct_drop(
        row["attend_rate_14d"], row["attend_rate_7d"]
    )
    row["completion_rate_14d"] = _window(sdf, snapshot_day, 14)["completion_rate"].mean()

    # ── 2. Test Performance ────────────────────────────────────────────────────
    for w in [7, 14, 30]:
        wdf  = _window(sdf, snapshot_day, w)
        scores = wdf["test_score_mean"].dropna()
        row[f"test_score_avg_{w}d"]  = scores.mean()   if len(scores) else np.nan
        row[f"test_attempts_{w}d"]   = wdf["test_attempted_any"].sum() if len(wdf) else 0

    row["score_slope_14d"]    = _score_slope(_window(sdf, snapshot_day, 14)["test_score_mean"])
    row["score_delta_14d"]    = _safe_score_delta(sdf, snapshot_day)

    # ── 3. Session Behavior ────────────────────────────────────────────────────
    active_days = sdf[sdf["day"] <= snapshot_day]["day"]
    if len(active_days):
        row["days_since_last_session"] = snapshot_day - active_days.max()
    else:
        row["days_since_last_session"] = 30  # penalize absent students
    row["session_freq_7d"]  = _window(sdf, snapshot_day, 7)["attended_any"].sum()
    row["session_freq_14d"] = _window(sdf, snapshot_day, 14)["attended_any"].sum()

    # ── 4. Assignment Completion ───────────────────────────────────────────────
    for w in [7, 14]:
        wdf = _window(sdf, snapshot_day, w)
        row[f"assignment_rate_{w}d"] = wdf["assignment_done"].mean() if len(wdf) else 0.0

    # ── 5. Doubt Activity ─────────────────────────────────────────────────────
    row["doubt_count_7d"]    = _window(sdf, snapshot_day, 7)["doubt_asked"].sum()
    row["doubt_count_14d"]   = _window(sdf, snapshot_day, 14)["doubt_asked"].sum()
    row["doubt_delta_7_14"]  = _pct_drop(
        max(row["doubt_count_14d"], 1e-6),
        max(row["doubt_count_7d"],  1e-6)
    )

    # ── 6. Revision Behavior ───────────────────────────────────────────────────
    row["revision_count_14d"] = _window(sdf, snapshot_day, 14)["revision_done"].sum()
    row["revision_count_7d"]  = _window(sdf, snapshot_day, 7)["revision_done"].sum()

    # ── 7. App Engagement ─────────────────────────────────────────────────────
    row["app_minutes_7d"]  = _window(sdf, snapshot_day, 7)["app_minutes"].mean()
    row["app_minutes_14d"] = _window(sdf, snapshot_day, 14)["app_minutes"].mean()
    row["app_delta_7_14"]  = _pct_drop(row["app_minutes_14d"] or 1e-6,
                                        row["app_minutes_7d"] or 1e-6)

    # ── 8. Peer / Batch Comparison ────────────────────────────────────────────
    row["attend_vs_batch"]   = row["attend_rate_14d"] - batch_avg["attend_14d"]
    row["score_vs_batch"]    = (row["test_score_avg_14d"] or 0) - batch_avg["score_14d"]
    row["progress_vs_batch"] = row["completion_rate_14d"] - batch_avg["progress_14d"]
    row["app_vs_batch"]      = row["app_minutes_14d"] - batch_avg["app_min_14d"]

    # ── 9. Subject-Level Signals ──────────────────────────────────────────────
    for subj, sdf_subj in subject_dfs.items():
        subj_lc = subj.lower()
        ssdf = sdf_subj[sdf_subj["student_id"] == sid].sort_values("day")

        attend_col = f"{subj_lc}_attend"
        score_col  = f"{subj_lc}_score"

        att_14 = _window(ssdf, snapshot_day, 14)[attend_col].mean() if len(ssdf) else 0.0
        att_7  = _window(ssdf, snapshot_day, 7)[attend_col].mean()  if len(ssdf) else 0.0
        sc_14  = _window(ssdf, snapshot_day, 14)[score_col].dropna()
        sc_7   = _window(ssdf, snapshot_day, 7)[score_col].dropna()

        row[f"{subj_lc}_attend_14d"]      = att_14
        row[f"{subj_lc}_attend_drop_pct"] = _pct_drop(att_14, att_7)
        row[f"{subj_lc}_score_14d"]       = sc_14.mean() if len(sc_14) else np.nan
        row[f"{subj_lc}_score_7d"]        = sc_7.mean()  if len(sc_7)  else np.nan
        row[f"{subj_lc}_score_drop_pct"]  = _pct_drop(
            sc_14.mean() if len(sc_14) else 0,
            sc_7.mean()  if len(sc_7)  else 0,
        )

    return row


def _safe_score_delta(sdf: pd.DataFrame, snapshot_day: int) -> float:
    """Score in last 7d minus score in prior 7d (day-14 to day-7)."""
    recent = _window(sdf, snapshot_day, 7)["test_score_mean"].dropna()
    older  = sdf[(sdf["day"] > snapshot_day - 14) &
                 (sdf["day"] <= snapshot_day - 7)]["test_score_mean"].dropna()
    if len(recent) == 0 or len(older) == 0:
        return 0.0
    return float(recent.mean() - older.mean())


def fill_missing_features(feat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute NaNs:
      - Score-related columns → fill with column median
      - Others → fill with 0
    """
    score_cols = [c for c in feat_df.columns if "score" in c]
    feat_df[score_cols] = feat_df[score_cols].fillna(feat_df[score_cols].median())
    feat_df = feat_df.fillna(0)
    return feat_df


def get_feature_columns(feat_df: pd.DataFrame) -> list[str]:
    """Return all feature column names (everything except student_id)."""
    return [c for c in feat_df.columns if c != "student_id"]
