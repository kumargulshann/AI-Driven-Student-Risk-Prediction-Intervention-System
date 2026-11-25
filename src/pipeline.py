"""
End-to-End Pipeline Orchestrator
==================================
Runs the full student early-warning pipeline:

  1.  Load raw data (or generate synthetic data if missing)
  2.  Feature engineering
  3.  Train model (or load from registry if already trained)
  4.  Score all students
  5.  Generate SHAP explanations
  6.  Generate intervention recommendations
  7.  Simulate outcomes (demo mode)
  8.  Save enriched risk report to data/processed/risk_report.csv

Usage
-----
  python src/pipeline.py              # full pipeline, trains new model
  python src/pipeline.py --no-train  # load existing model, re-score only
"""

import sys
import argparse
import logging
import time
import yaml
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ── configure root logger ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

SNAPSHOT_DAY = CFG["data"]["snapshot_day"]
SUBJECTS     = CFG["data"]["subjects"]
PROCESSED_DIR = ROOT / "data" / "processed"


def _ensure_data_exists() -> bool:
    """Generate missing or stale synthetic data while preserving external inputs."""
    from data.synthetic_generator import (
        EVENTS, LABELS, MANIFEST, RAW_DIR, main as generate, synthetic_settings,
    )

    events_path, labels_path = RAW_DIR / EVENTS, RAW_DIR / LABELS
    manifest = RAW_DIR / MANIFEST
    if not events_path.exists() and not labels_path.exists():
        generate()
        return True
    if manifest.exists():
        if (not events_path.exists() or not labels_path.exists()
                or json.loads(manifest.read_text(encoding="utf-8")) != synthetic_settings(CFG["data"])):
            generate()
        return True
    if events_path.exists() and labels_path.exists():
        # The original generator left its archetype marker in both CSVs.
        if all("archetype" in pd.read_csv(path, nrows=0).columns for path in (events_path, labels_path)):
            generate()
            return True
        logger.info("Using existing input files in %s", RAW_DIR)
        return False
    raise FileNotFoundError(f"Both {EVENTS} and {LABELS} are required in {RAW_DIR}")


def run_pipeline(train_model: bool = True) -> dict:
    """
    Execute the full pipeline.

    Parameters
    ----------
    train_model : if True, retrain; if False, load existing model artifacts.

    Returns
    -------
    dict with keys: risk_report, model_metrics, outcome_metrics
    """
    t0 = time.time()

    # ── Step 1: Data ──────────────────────────────────────────────────────────
    is_synthetic = _ensure_data_exists()

    from src.ingestion.data_loader import (
        load_raw_data, get_student_metadata,
        pivot_to_daily, pivot_to_daily_per_subject,
    )
    logger.info("Step 1/8 — Loading data …")
    events_df, labels_df = load_raw_data(before_day=SNAPSHOT_DAY)
    meta_df = get_student_metadata(events_df)

    daily_df = pivot_to_daily(events_df)

    subject_dfs = {
        subj: pivot_to_daily_per_subject(events_df, subj)
        for subj in SUBJECTS
    }
    del events_df

    # ── Step 2: Feature Engineering ───────────────────────────────────────────
    from src.features.feature_engineering import (
        build_features, fill_missing_features, get_feature_columns,
    )
    logger.info("Step 2/8 — Engineering features …")
    feat_df = build_features(daily_df, subject_dfs, SNAPSHOT_DAY - 1)
    del daily_df, subject_dfs
    feat_df = fill_missing_features(feat_df)
    feature_cols = get_feature_columns(feat_df)
    logger.info("  → %d features for %d students", len(feature_cols), len(feat_df))

    # ── Step 3: Model ─────────────────────────────────────────────────────────
    from src.models.model_registry import save_model, load_model, model_exists
    from src.models.risk_model import train_model as _train, predict_risk, cross_validate_model

    if train_model or not model_exists():
        logger.info("Step 3/8 — Training risk model …")
        model, scaler, metrics, X_test, y_test = _train(feat_df, labels_df, feature_cols)
        save_model(model, scaler, feature_cols, metrics)

        logger.info("Step 3b/8 — Cross-validation …")
        cv_metrics = cross_validate_model(feat_df, labels_df, feature_cols)
        metrics.update(cv_metrics)
    else:
        logger.info("Step 3/8 — Loading existing model from registry …")
        model, scaler, feature_cols, meta = load_model()
        metrics = meta["metrics"]

    # ── Step 4: Score Students ────────────────────────────────────────────────
    logger.info("Step 4/8 — Scoring all students …")
    risk_df = predict_risk(feat_df, feature_cols, model, scaler)
    logger.info(
        "  At-risk distribution: %s",
        risk_df["risk_tier"].value_counts().to_dict(),
    )

    # ── Step 5: SHAP Explanations ─────────────────────────────────────────────
    from src.explainability.shap_explainer import StudentShapExplainer

    logger.info("Step 5/8 — Computing SHAP explanations …")
    alerts = risk_df.loc[risk_df["risk_tier"].isin(["High", "Critical"])]
    indexed_features = feat_df.set_index("student_id")
    alert_features = indexed_features.loc[alerts["student_id"]]
    explainer = StudentShapExplainer(model, feature_cols)
    shap_values = explainer.explain_batch(
        scaler.transform(alert_features[feature_cols].to_numpy(dtype="float32"))
    ) if len(alerts) else []
    logger.info("  Computed SHAP for %d at-risk students", len(alerts))

    from src.interventions.recommendation_engine import generate_intervention

    logger.info("Step 6/8 — Generating interventions …")
    intervention_records = []
    for row, (_, feat_row), sv in zip(alerts.itertuples(index=False), alert_features.iterrows(), shap_values):
        sid = int(row.student_id)
        factors = explainer.factors_from_values(sv, feat_row)
        result = generate_intervention(
            student_id=sid,
            risk_score=float(row.risk_score),
            risk_tier=str(row.risk_tier),
            features=feat_row.fillna(0).to_dict(),
            shap_factors=factors,
            use_llm=True,
        )
        intervention_records.append({
            "student_id":        sid,
            "intervention_type": result.intervention_type,
            "urgency_hours":     result.urgency_hours,
            "interventions":     " | ".join(result.interventions),
            "top_reasons":       " | ".join(result.top_reasons),
            "llm_narrative":     result.llm_narrative,
        })

    interventions_df = pd.DataFrame(intervention_records, columns=[
        "student_id", "intervention_type", "urgency_hours", "interventions", "top_reasons", "llm_narrative",
    ])

    # ── Step 7: Simulate Outcomes ─────────────────────────────────────────────
    from src.tracking.outcome_tracker import simulate_outcomes, compute_outcome_metrics

    logger.info("Step 7/8 — Preparing outcome data …")
    if is_synthetic:
        simulate_outcomes(risk_df)
    outcome_metrics = compute_outcome_metrics()

    # ── Step 8: Save Risk Report ──────────────────────────────────────────────
    logger.info("Step 8/8 — Saving risk report …")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    report_df = (
        risk_df
        .merge(meta_df, on="student_id", how="left", validate="one_to_one")
        .merge(labels_df.reindex(columns=["student_id", "disengaged_in_14d", "archetype"]),
               on="student_id", how="left", validate="one_to_one")
        .merge(interventions_df, on="student_id", how="left", validate="one_to_one")
    )

    report_path = PROCESSED_DIR / "risk_report.csv"
    report_df.to_csv(report_path, index=False)

    elapsed = time.time() - t0
    logger.info("✅  Pipeline complete in %.1fs | Report: %s", elapsed, report_path)
    logger.info("📊  Model AUC-ROC: %.3f | CV-AUC: %.3f ± %.3f",
                metrics.get("auc_roc", 0),
                metrics.get("cv_auc_mean", 0),
                metrics.get("cv_auc_std", 0))
    logger.info("🎯  Recovery rate @14d: %.1f%% | Score improvement: +%.1f pts",
                outcome_metrics.get("recovery_rate_14d", 0) * 100,
                outcome_metrics.get("avg_score_improvement", 0))

    return {
        "risk_report":     report_df,
        "model_metrics":   metrics,
        "outcome_metrics": outcome_metrics,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PW Student Early-Warning Pipeline")
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip model training — load existing model artifacts",
    )
    args = parser.parse_args()
    run_pipeline(train_model=not args.no_train)
