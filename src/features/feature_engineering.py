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
    """Build one row per student using daily events through snapshot_day."""
    history = daily_df.loc[daily_df["day"].le(snapshot_day)]
    features = pd.DataFrame(index=pd.Index(history["student_id"].unique(), name="student_id"))
    windows = {days: _window(history, snapshot_day, days) for days in (7, 14, 30)}
    for days, window in windows.items():
        grouped = window.groupby("student_id", sort=False)
        features[f"attend_rate_{days}d"] = grouped["attend_rate"].mean().reindex(features.index, fill_value=0)
        features[f"attend_count_{days}d"] = grouped["attended_any"].sum().reindex(features.index, fill_value=0)
        features[f"test_score_avg_{days}d"] = grouped["test_score_mean"].mean()
        features[f"test_attempts_{days}d"] = grouped["test_attempted_any"].sum().reindex(features.index, fill_value=0)
        if days != 30:
            features[f"session_freq_{days}d"] = features[f"attend_count_{days}d"]
            features[f"assignment_rate_{days}d"] = grouped["assignment_done"].mean().reindex(features.index, fill_value=0)
            features[f"doubt_count_{days}d"] = grouped["doubt_asked"].sum().reindex(features.index, fill_value=0)
            features[f"revision_count_{days}d"] = grouped["revision_done"].sum().reindex(features.index, fill_value=0)
            features[f"app_minutes_{days}d"] = grouped["app_minutes"].mean()

    features["attend_delta_7_14"] = _positive_drop(features["attend_rate_14d"], features["attend_rate_7d"])
    features["doubt_delta_7_14"] = _positive_drop(features["doubt_count_14d"], features["doubt_count_7d"])
    features["app_delta_7_14"] = _positive_drop(features["app_minutes_14d"], features["app_minutes_7d"])

    recent = windows[14]
    features["completion_rate_14d"] = recent.groupby("student_id", sort=False)["completion_rate"].mean()
    features["score_slope_14d"] = _grouped_score_slope(recent).reindex(features.index, fill_value=0)
    previous = recent.loc[recent["day"].le(snapshot_day - 7)]
    previous_scores = previous.groupby("student_id", sort=False)["test_score_mean"].mean()
    features["score_delta_14d"] = (features["test_score_avg_7d"] - previous_scores).fillna(0)

    active = history.loc[history["attended_any"].gt(0), ["student_id", "day"]]
    last_session = active.groupby("student_id", sort=False)["day"].max()
    features["days_since_last_session"] = (snapshot_day - last_session).reindex(features.index, fill_value=30)

    # Historical cohort comparisons are available to the rules engine.
    for suffix, student_col, daily_col in (
        ("attend", "attend_rate_14d", "attend_rate"),
        ("score", "test_score_avg_14d", "test_score_mean"),
        ("progress", "completion_rate_14d", "completion_rate"),
        ("app", "app_minutes_14d", "app_minutes"),
    ):
        features[f"{suffix}_vs_batch"] = features[student_col] - recent[daily_col].mean()

    for subject, daily in subject_dfs.items():
        name = subject.lower()
        columns = [f"{name}_attend", f"{name}_score"]
        fortnight = _window(daily, snapshot_day, 14).groupby("student_id", sort=False)[columns].mean()
        week = _window(daily, snapshot_day, 7).groupby("student_id", sort=False)[columns].mean()
        features[f"{name}_attend_14d"] = fortnight[columns[0]]
        features[f"{name}_attend_drop_pct"] = _positive_drop(fortnight[columns[0]], week[columns[0]])
        features[f"{name}_score_14d"] = fortnight[columns[1]]
        features[f"{name}_score_7d"] = week[columns[1]]
        features[f"{name}_score_drop_pct"] = _positive_drop(fortnight[columns[1]], week[columns[1]])

    logger.info("Built %d feature rows with %d features", len(features), features.shape[1])
    return features.reset_index()


def _positive_drop(before: pd.Series, after: pd.Series) -> pd.Series:
    return ((before - after) / before.where(before >= 1e-6)).clip(lower=0)


def _grouped_score_slope(daily: pd.DataFrame) -> pd.Series:
    """Use the same OLS slope as _score_slope over each student's observed scores."""
    scores = (daily[["student_id", "day", "test_score_mean"]]
              .dropna(subset=["test_score_mean"]).sort_values("day"))
    position = scores.groupby("student_id", sort=False).cumcount()
    stats = scores.assign(weighted_score=position * scores["test_score_mean"]).groupby(
        "student_id", sort=False
    ).agg(count=("test_score_mean", "size"), total=("test_score_mean", "sum"),
          weighted=("weighted_score", "sum"))
    count = stats["count"]
    numerator = 12 * (stats["weighted"] - (count - 1) * stats["total"] / 2)
    denominator = count * (count ** 2 - 1)
    return (numerator / denominator.where(count > 1)).fillna(0)




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
