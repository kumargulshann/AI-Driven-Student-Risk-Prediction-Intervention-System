"""
Data Ingestion & Validation Layer
===================================
Loads raw student event data, validates schema, and returns
a clean merged DataFrame ready for feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

REQUIRED_EVENT_COLS = [
    "student_id", "batch_id", "city", "course", "day", "subject",
    "attended_lecture", "completion_rate", "test_attempted", "test_score",
    "assignment_done", "doubt_asked", "revision_done", "app_minutes",
]
REQUIRED_LABEL_COLS = ["student_id", "disengaged_in_14d"]


def load_raw_data(
    events_path: str | Path | None = None,
    labels_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw CSV files and return (events_df, labels_df)."""
    raw_dir    = ROOT / "data" / "raw"
    events_path = Path(events_path) if events_path else raw_dir / "student_events.csv"
    labels_path = Path(labels_path) if labels_path else raw_dir / "student_labels.csv"

    if not events_path.exists():
        raise FileNotFoundError(
            f"Events file not found: {events_path}\n"
            "Run:  python data/synthetic_generator.py"
        )

    events_df = pd.read_csv(events_path, low_memory=False)
    labels_df = pd.read_csv(labels_path)

    _validate_schema(events_df, REQUIRED_EVENT_COLS, "events")
    _validate_schema(labels_df, REQUIRED_LABEL_COLS, "labels")

    logger.info(
        "Loaded %d event rows for %d students",
        len(events_df), events_df["student_id"].nunique(),
    )
    return events_df, labels_df


def _validate_schema(df: pd.DataFrame, required_cols: list[str], name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{name}] Missing columns: {missing}")


def get_student_metadata(events_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per student with static metadata."""
    return (
        events_df.drop_duplicates(subset="student_id")
        [["student_id", "batch_id", "city", "course"]]
        .reset_index(drop=True)
    )


def get_events_up_to_day(events_df: pd.DataFrame, snapshot_day: int) -> pd.DataFrame:
    """Filter events to only those on or before the snapshot day."""
    return events_df[events_df["day"] <= snapshot_day].copy()


def pivot_to_daily(events_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-subject rows into a single daily row per student.
    Returns columns: student_id, day, plus aggregated signal columns.
    """
    agg = (
        events_df
        .groupby(["student_id", "day"])
        .agg(
            attended_any        = ("attended_lecture",  "max"),
            attend_rate         = ("attended_lecture",  "mean"),
            completion_rate     = ("completion_rate",   "mean"),
            test_attempted_any  = ("test_attempted",    "max"),
            test_score_mean     = ("test_score",        "mean"),
            assignment_done     = ("assignment_done",   "mean"),
            doubt_asked         = ("doubt_asked",       "sum"),
            revision_done       = ("revision_done",     "sum"),
            app_minutes         = ("app_minutes",       "max"),
        )
        .reset_index()
    )
    return agg


def pivot_to_daily_per_subject(events_df: pd.DataFrame, subject: str) -> pd.DataFrame:
    """Return daily aggregates for a single subject."""
    df = events_df[events_df["subject"] == subject].copy()
    agg = (
        df.groupby(["student_id", "day"])
        .agg(
            attend   = ("attended_lecture", "mean"),
            score    = ("test_score",       "mean"),
        )
        .reset_index()
        .rename(columns={
            "attend": f"{subject.lower()}_attend",
            "score":  f"{subject.lower()}_score",
        })
    )
    return agg
