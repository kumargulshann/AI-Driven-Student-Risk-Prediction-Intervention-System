"""
Synthetic Student Behavioral Data Generator
============================================
Generates realistic student behavioral data for 5,000 students
over 90 days across PW-style subjects (Physics, Chemistry, Maths, Biology).

Output: data/raw/student_events.csv + data/raw/student_labels.csv
"""

import numpy as np
import pandas as pd
import os
import yaml
from pathlib import Path

# ── reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)

ROOT = Path(__file__).parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

N_STUDENTS   = CFG["data"]["synthetic_students"]
SIM_DAYS     = CFG["data"]["simulation_days"]
SUBJECTS     = CFG["data"]["subjects"]
SNAPSHOT_DAY = CFG["data"]["snapshot_day"]

# ── Student Archetype Definitions ──────────────────────────────────────────────
# Each archetype describes a learning trajectory pattern
ARCHETYPES = {
    "star":           {"weight": 0.20, "disengage_prob": 0.03},
    "consistent":     {"weight": 0.30, "disengage_prob": 0.08},
    "at_risk_early":  {"weight": 0.15, "disengage_prob": 0.72},
    "gradual_drop":   {"weight": 0.20, "disengage_prob": 0.55},
    "irregular":      {"weight": 0.10, "disengage_prob": 0.40},
    "recovered":      {"weight": 0.05, "disengage_prob": 0.12},
}

def sample_archetype() -> str:
    names = list(ARCHETYPES.keys())
    weights = [ARCHETYPES[a]["weight"] for a in names]
    return np.random.choice(names, p=weights)


def base_engagement(archetype: str, day: int) -> float:
    """Return a 0-1 base engagement level given archetype and day in cycle."""
    if archetype == "star":
        return np.clip(np.random.normal(0.90, 0.05), 0.70, 1.0)
    elif archetype == "consistent":
        return np.clip(np.random.normal(0.75, 0.08), 0.55, 0.95)
    elif archetype == "at_risk_early":
        # Starts ok, drops steeply after day 15
        decay = max(0, (day - 15) / 60)
        return np.clip(np.random.normal(0.70 - 0.55 * decay, 0.10), 0.05, 0.95)
    elif archetype == "gradual_drop":
        decay = day / SIM_DAYS
        return np.clip(np.random.normal(0.80 - 0.45 * decay, 0.08), 0.05, 0.95)
    elif archetype == "irregular":
        # Bursty — alternates high and low weeks
        cycle = np.sin(day / 7 * np.pi)
        return np.clip(np.random.normal(0.60 + 0.20 * cycle, 0.15), 0.10, 0.95)
    elif archetype == "recovered":
        # Drops then recovers
        midpoint = SIM_DAYS // 2
        if day < midpoint:
            decay = day / midpoint
            return np.clip(np.random.normal(0.75 - 0.40 * decay, 0.10), 0.05, 0.95)
        else:
            recovery = (day - midpoint) / midpoint
            return np.clip(np.random.normal(0.40 + 0.40 * recovery, 0.10), 0.10, 0.95)
    return 0.5


def generate_student_record(student_id: int) -> list[dict]:
    archetype    = sample_archetype()
    batch_id     = f"BATCH_{np.random.randint(1, 21):03d}"
    city         = np.random.choice(["Delhi", "Mumbai", "Bangalore", "Hyderabad",
                                     "Pune", "Jaipur", "Lucknow", "Patna", "Kolkata"])
    course       = np.random.choice(["JEE Advanced", "JEE Mains", "NEET", "Foundation"])
    join_day     = np.random.randint(0, 10)
    target_score = np.random.uniform(60, 100)  # aspirational %
    records      = []

    # subject-level base bias
    subj_bias = {s: np.random.normal(0, 0.08) for s in SUBJECTS}

    for day in range(join_day, SIM_DAYS):
        eng = base_engagement(archetype, day - join_day)

        for subject in SUBJECTS:
            s_eng = np.clip(eng + subj_bias[subject], 0.0, 1.0)

            # derive signals from engagement level
            did_attend       = int(np.random.random() < s_eng)
            completion_rate  = np.clip(np.random.normal(s_eng, 0.10), 0, 1) if did_attend else 0.0
            test_attempted   = int(np.random.random() < s_eng * 0.7)
            raw_score        = np.clip(np.random.normal(target_score * s_eng / 100, 10), 0, 100) \
                               if test_attempted else np.nan
            assignment_done  = int(np.random.random() < s_eng * 0.85)
            doubt_asked      = int(np.random.random() < s_eng * 0.3)
            revision_done    = int(np.random.random() < s_eng * 0.5)

            records.append({
                "student_id":        student_id,
                "batch_id":          batch_id,
                "city":              city,
                "course":            course,
                "archetype":         archetype,  # dropped before modeling
                "day":               day,
                "subject":           subject,
                "attended_lecture":  did_attend,
                "completion_rate":   round(completion_rate, 3),
                "test_attempted":    test_attempted,
                "test_score":        round(raw_score, 2) if not np.isnan(raw_score) else np.nan,
                "assignment_done":   assignment_done,
                "doubt_asked":       doubt_asked,
                "revision_done":     revision_done,
            })

        # App-level daily signal (once per day, not per subject)
        if subject == SUBJECTS[-1]:  # append once using last subject iteration
            records[-1]["app_minutes"] = round(
                np.clip(np.random.normal(eng * 90, 20), 0, 180), 1
            )

    return records, archetype


def compute_label(student_df: pd.DataFrame, snapshot_day: int, window: int = 14) -> int:
    """
    Ground-truth disengagement label.
    1 = student significantly disengages in the [snapshot_day, snapshot_day+window] window.
    Criterion: average attendance in the post-window drops >40% vs prior 14 days.
    """
    pre  = student_df[(student_df["day"] >= snapshot_day - 14) & (student_df["day"] < snapshot_day)]
    post = student_df[(student_df["day"] >= snapshot_day) & (student_df["day"] < snapshot_day + window)]
    if len(pre) == 0 or len(post) == 0:
        return 0
    pre_attend  = pre["attended_lecture"].mean()
    post_attend = post["attended_lecture"].mean()
    drop = (pre_attend - post_attend) / (pre_attend + 1e-9)
    return int(drop > 0.40)


def main():
    out_dir = ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    labels      = []

    print(f"Generating {N_STUDENTS} student records over {SIM_DAYS} days …")

    for sid in range(1, N_STUDENTS + 1):
        records, archetype = generate_student_record(sid)
        all_records.extend(records)

        student_df = pd.DataFrame(records)
        label      = compute_label(student_df, SNAPSHOT_DAY)
        labels.append({
            "student_id":         sid,
            "archetype":          archetype,
            "disengaged_in_14d":  label,
        })

        if sid % 500 == 0:
            print(f"  … {sid}/{N_STUDENTS} students done")

    events_df = pd.DataFrame(all_records)
    # fill app_minutes forward (it was only written on last-subject rows)
    events_df["app_minutes"] = events_df.groupby(["student_id", "day"])["app_minutes"].transform("max")

    labels_df = pd.DataFrame(labels)

    events_path = out_dir / "student_events.csv"
    labels_path = out_dir / "student_labels.csv"
    events_df.to_csv(events_path, index=False)
    labels_df.to_csv(labels_path, index=False)

    pos_rate = labels_df["disengaged_in_14d"].mean()
    print(f"\n✅ Done!")
    print(f"   Events  : {len(events_df):,} rows  →  {events_path}")
    print(f"   Labels  : {len(labels_df):,} rows  →  {labels_path}")
    print(f"   Disengagement rate: {pos_rate:.1%}  ({int(pos_rate*N_STUDENTS)} at-risk students)")


if __name__ == "__main__":
    main()
