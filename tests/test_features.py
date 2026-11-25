"""
Tests — Feature Engineering
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import (
    _window, _score_slope, _pct_drop, fill_missing_features, get_feature_columns,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_daily_df():
    """10 days of data for 2 students."""
    rows = []
    for sid in [1, 2]:
        for day in range(1, 11):
            rows.append({
                "student_id":       sid,
                "day":              day,
                "attended_any":     1 if day % 2 == 0 else 0,
                "attend_rate":      0.8 if sid == 1 else 0.3,
                "completion_rate":  0.9,
                "test_attempted_any": 1,
                "test_score_mean":  70 + day * (1 if sid == 1 else -2),
                "assignment_done":  0.75,
                "doubt_asked":      1 if day == 5 else 0,
                "revision_done":    0,
                "app_minutes":      60.0,
            })
    return pd.DataFrame(rows)


# ── Unit Tests ────────────────────────────────────────────────────────────────

def test_window_filters_correctly(sample_daily_df):
    sid_df = sample_daily_df[sample_daily_df["student_id"] == 1]
    result = _window(sid_df, day=7, window_days=7)
    assert result["day"].min() > 0    # no day-0
    assert result["day"].max() <= 7   # at most snapshot_day


def test_window_empty_for_far_day(sample_daily_df):
    sid_df = sample_daily_df[sample_daily_df["student_id"] == 1]
    result = _window(sid_df, day=100, window_days=3)
    # All days in sample are 1-10, so day 97-100 → empty
    assert len(result) == 0


def test_score_slope_positive():
    scores = pd.Series([50, 55, 60, 65, 70])
    slope = _score_slope(scores)
    assert slope > 0


def test_score_slope_negative():
    scores = pd.Series([70, 65, 60, 55, 50])
    slope = _score_slope(scores)
    assert slope < 0


def test_score_slope_single_value():
    scores = pd.Series([60.0])
    slope = _score_slope(scores)
    assert slope == 0.0


def test_pct_drop_positive():
    # 80 → 50: 37.5% drop
    drop = _pct_drop(0.80, 0.50)
    assert abs(drop - 0.375) < 1e-6


def test_pct_drop_zero_baseline():
    drop = _pct_drop(0.0, 0.5)
    assert drop == 0.0


def test_pct_drop_no_drop():
    drop = _pct_drop(0.6, 0.7)  # improved — no drop
    assert drop == 0.0


def test_fill_missing_features():
    df = pd.DataFrame({
        "student_id": [1, 2],
        "score_col":  [np.nan, 80.0],
        "other_col":  [np.nan, 5.0],
    })
    filled = fill_missing_features(df)
    assert filled["score_col"].isna().sum() == 0
    assert filled["other_col"].isna().sum() == 0


def test_get_feature_columns_excludes_id():
    df = pd.DataFrame({"student_id": [1], "feat_a": [0.5], "feat_b": [1.0]})
    cols = get_feature_columns(df)
    assert "student_id" not in cols
    assert "feat_a" in cols
    assert "feat_b" in cols


def test_first_label_day_is_excluded_from_pipeline_features(tmp_path):
    from src.features.feature_engineering import build_features
    from src.ingestion.data_loader import load_raw_data, pivot_to_daily

    events_path = tmp_path / "events.csv"
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame({
        "student_id": [1, 1, 1], "day": [73, 74, 75],
        "batch_id": "BATCH_001", "city": "Delhi", "course": "JEE Mains",
        "subject": "Physics", "attended_lecture": [1, 0, 1],
        "completion_rate": [1.0, 0.0, 1.0], "test_attempted": [1, 0, 1],
        "test_score": [70.0, np.nan, 100.0], "assignment_done": [1, 0, 1],
        "doubt_asked": 0, "revision_done": 0, "app_minutes": 60.0,
    }).to_csv(events_path, index=False)
    pd.DataFrame({"student_id": [1], "disengaged_in_14d": [1]}).to_csv(labels_path, index=False)

    events, _ = load_raw_data(events_path, labels_path, before_day=75)
    features = build_features(pivot_to_daily(events), {}, snapshot_day=74).iloc[0]

    assert events["day"].tolist() == [73, 74]
    assert features["attend_rate_7d"] == 0.5
    assert features["test_score_avg_7d"] == 70.0
    assert features["days_since_last_session"] == 1
