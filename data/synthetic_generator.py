"""Generate reproducible synthetic event and label CSVs in bounded batches."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
EVENTS = "student_events.csv"
LABELS = "student_labels.csv"
MANIFEST = "synthetic_config.json"
BATCH_SIZE = 1_000
HORIZON = 14

ARCHETYPES = ("star", "consistent", "at_risk_early", "gradual_drop", "irregular", "recovered")
WEIGHTS = (0.20, 0.30, 0.15, 0.20, 0.10, 0.05)
CITIES = ("Delhi", "Mumbai", "Bangalore", "Hyderabad", "Pune", "Jaipur", "Lucknow", "Patna", "Kolkata")
COURSES = ("JEE Advanced", "JEE Mains", "NEET", "Foundation")
BATCHES = tuple(f"BATCH_{i:03d}" for i in range(1, 21))


def synthetic_settings(config: dict) -> dict:
    return {
        key: config[key]
        for key in ("synthetic_students", "simulation_days", "snapshot_day", "subjects")
    }


def _engagement(rng, archetypes, age, days):
    midpoint = max(1, days // 2)
    means = np.vstack((
        np.broadcast_to(0.90, age.shape),
        np.broadcast_to(0.75, age.shape),
        0.70 - 0.55 * np.maximum(0, (age - 15) / 60),
        0.80 - 0.45 * age / days,
        0.60 + 0.20 * np.sin(age / 7 * np.pi),
        np.where(age < midpoint, 0.75 - 0.40 * age / midpoint,
                 0.40 + 0.40 * (age - midpoint) / midpoint),
    )).reshape(len(ARCHETYPES), *age.shape)
    students = np.arange(len(archetypes))
    mean = means[archetypes, students, :]
    std = np.array((0.05, 0.08, 0.10, 0.08, 0.15, 0.10))[archetypes, None]
    lower = np.array((0.70, 0.55, 0.05, 0.05, 0.10, 0.05))[archetypes, None]
    lower = np.where((archetypes[:, None] == 5) & (age >= midpoint), 0.10, lower)
    upper = np.where(archetypes[:, None] == 0, 1.0, 0.95)
    return np.clip(rng.normal(mean, std), lower, upper)


def _generate_batch(rng, ids, days, subjects, snapshot):
    count = len(ids)
    subject_count = len(subjects)
    shape = (count, days, subject_count)
    day = np.arange(days)
    joining = rng.integers(0, min(10, snapshot - HORIZON + 1), count)
    age = day[None, :] - joining[:, None]
    archetype = rng.choice(len(ARCHETYPES), count, p=WEIGHTS)
    engagement = _engagement(rng, archetype, age, days)
    subject_bias = rng.normal(0, 0.08, size=(count, 1, subject_count))
    subject_engagement = np.clip(engagement[:, :, None] + subject_bias, 0, 1)

    attended = rng.random(shape) < subject_engagement
    attempted = rng.random(shape) < subject_engagement * 0.7
    completion = np.where(attended, np.clip(rng.normal(subject_engagement, 0.10), 0, 1), 0)
    target = rng.uniform(60, 100, size=(count, 1, 1))
    scores = np.where(attempted, np.clip(rng.normal(target * subject_engagement, 10), 0, 100), np.nan)
    app_minutes = np.clip(rng.normal(engagement * 90, 20), 0, 180).round(1)

    student_idx, day_idx, subject_idx = np.nonzero(np.broadcast_to((age >= 0)[:, :, None], shape))
    index = student_idx, day_idx, subject_idx
    archetype_names = np.asarray(ARCHETYPES)[archetype]
    events = pd.DataFrame({
        "student_id": ids[student_idx],
        "batch_id": rng.choice(BATCHES, count)[student_idx],
        "city": rng.choice(CITIES, count)[student_idx],
        "course": rng.choice(COURSES, count)[student_idx],
        "archetype": archetype_names[student_idx],
        "day": day_idx,
        "subject": np.asarray(subjects)[subject_idx],
        "attended_lecture": attended[index].astype("uint8"),
        "completion_rate": completion[index].round(3),
        "test_attempted": attempted[index].astype("uint8"),
        "test_score": scores[index].round(2),
        "assignment_done": (rng.random(shape)[index] < subject_engagement[index] * 0.85).astype("uint8"),
        "doubt_asked": (rng.random(shape)[index] < subject_engagement[index] * 0.3).astype("uint8"),
        "revision_done": (rng.random(shape)[index] < subject_engagement[index] * 0.5).astype("uint8"),
        "app_minutes": app_minutes[student_idx, day_idx],
    })
    pre = attended[:, snapshot - HORIZON:snapshot].mean(axis=(1, 2))
    post = attended[:, snapshot:snapshot + HORIZON].mean(axis=(1, 2))
    labels = pd.DataFrame({
        "student_id": ids,
        "archetype": archetype_names,
        "disengaged_in_14d": ((pre - post) / (pre + 1e-9) > 0.40).astype("uint8"),
    })
    return events, labels


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as config_file:
        settings = synthetic_settings(yaml.safe_load(config_file)["data"])
    students = settings["synthetic_students"]
    days = settings["simulation_days"]
    snapshot = settings["snapshot_day"]
    subjects = settings["subjects"]
    if students < 1 or not HORIZON <= snapshot <= days - HORIZON or not subjects:
        raise ValueError("Synthetic generation requires students, subjects and full pre/post label windows.")

    rng = np.random.default_rng(42)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    event_count = positive_count = 0
    print(f"Generating {students:,} student records over {days} days", flush=True)
    with TemporaryDirectory(dir=RAW_DIR) as staging:
        staging = Path(staging)
        for offset in range(0, students, BATCH_SIZE):
            ids = np.arange(offset + 1, min(students, offset + BATCH_SIZE) + 1, dtype=np.int32)
            events, labels = _generate_batch(rng, ids, days, subjects, snapshot)
            events.to_csv(staging / EVENTS, index=False, mode="a", header=offset == 0)
            labels.to_csv(staging / LABELS, index=False, mode="a", header=offset == 0)
            event_count += len(events)
            positive_count += int(labels["disengaged_in_14d"].sum())

        for filename in (EVENTS, LABELS):
            os.replace(staging / filename, RAW_DIR / filename)
        (RAW_DIR / MANIFEST).write_text(json.dumps(settings), encoding="utf-8")

    print(f"Generated {event_count:,} events and {students:,} labels ({positive_count:,} positives).", flush=True)


if __name__ == "__main__":
    main()
