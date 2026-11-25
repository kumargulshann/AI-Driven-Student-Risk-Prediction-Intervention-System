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
from tempfile import NamedTemporaryFile
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ── configure root logger ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

with (ROOT / "config.yaml").open(encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

SNAPSHOT_DAY = CFG["data"]["snapshot_day"]
SUBJECTS     = CFG["data"]["subjects"]
PROCESSED_DIR = ROOT / "data" / "processed"
ASSETS_DIR = ROOT / "assets"
SHAP_BATCH_SIZE = 2_000


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
    # Peer comparisons serve the intervention rules. The model uses individual
    # history only, so held-out students cannot influence fitted inputs.
    feature_cols = [col for col in get_feature_columns(feat_df) if not col.endswith("_vs_batch")]
    logger.info("  → %d features for %d students", len(feature_cols), len(feat_df))

    # ── Step 3: Model ─────────────────────────────────────────────────────────
    from src.models.model_registry import save_model, load_model, model_exists
    from src.models.risk_model import train_model as _train, predict_risk, cross_validate_model

    retrain = train_model or not model_exists()
    if retrain:
        logger.info("Step 3/8 — Training risk model …")
        model, scaler, metrics, X_test, _ = _train(
            feat_df, labels_df, feature_cols, calibration_plot=ASSETS_DIR / "calibration_curve.png",
        )

        logger.info("Step 3b/8 — Cross-validation …")
        cv_metrics = cross_validate_model(feat_df, labels_df, feature_cols)
        metrics.update(cv_metrics)
        save_model(model, scaler, feature_cols, metrics)
    else:
        logger.info("Step 3/8 — Loading existing model from registry …")
        model, scaler, feature_cols, meta = load_model()
        metrics = meta["metrics"]
        # Legacy artifacts were trained with median-imputed features.
        if "brier_after" not in metrics:
            feat_df = fill_missing_features(feat_df)

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
    explainer = StudentShapExplainer(model, feature_cols)
    if retrain:
        from src.models.evaluation_plots import save_shap_summary

        save_shap_summary(
            explainer, X_test, ASSETS_DIR / "shap_summary.png", seed=CFG["model"]["random_state"],
        )

    # Compute bounded batches once. Keep human-readable reasons on all reports
    # and retain full factor records only for students with interventions.
    alerts = risk_df.loc[risk_df["risk_tier"].isin(["High", "Critical"])]
    alert_ids = set(alerts["student_id"])
    indexed_features = feat_df.set_index("student_id")
    reasons_by_id = {}
    alert_factors = {}
    for offset in range(0, len(feat_df), SHAP_BATCH_SIZE):
        batch = indexed_features.iloc[offset:offset + SHAP_BATCH_SIZE]
        scaled = scaler.transform(batch[feature_cols].to_numpy(dtype="float32"))
        for (sid, row), values in zip(batch.iterrows(), explainer.explain_batch(scaled)):
            factors = explainer.factors_from_values(values, row)
            reasons_by_id[sid] = " | ".join(factor["narrative"] for factor in factors)
            if sid in alert_ids:
                alert_factors[sid] = factors
    risk_df["top_reasons"] = risk_df["student_id"].map(reasons_by_id)
    logger.info("  Computed SHAP for %d students", len(reasons_by_id))

    from src.interventions.recommendation_engine import generate_intervention

    logger.info("Step 6/8 — Generating interventions for %d students …", len(alerts))
    intervention_records = []
    for row in alerts.itertuples(index=False):
        sid = int(row.student_id)
        feat_row = indexed_features.loc[sid]
        result = generate_intervention(
            student_id=sid,
            risk_score=float(row.risk_score),
            risk_tier=str(row.risk_tier),
            features=feat_row.fillna(0).to_dict(),
            shap_factors=alert_factors[sid],
            use_llm=True,
        )
        intervention_records.append({
            "student_id":        sid,
            "intervention_type": result.intervention_type,
            "urgency_hours":     result.urgency_hours,
            "interventions":     " | ".join(result.interventions),
            "llm_narrative":     result.llm_narrative,
        })

    interventions_df = pd.DataFrame(intervention_records, columns=[
        "student_id", "intervention_type", "urgency_hours", "interventions", "llm_narrative",
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
    with NamedTemporaryFile(dir=PROCESSED_DIR, suffix=".tmp", delete=False) as staging:
        report_tmp = Path(staging.name)
    try:
        report_df.to_csv(report_tmp, index=False)
        report_tmp.replace(report_path)
    finally:
        report_tmp.unlink(missing_ok=True)

    elapsed = time.time() - t0
    logger.info("✅  Pipeline complete in %.1fs | Report: %s", elapsed, report_path)
    if "cv_auc_mean" in metrics:
        logger.info(
            "Held-out AUC: %.6f | CV AUC: %.6f ± %.6f",
            metrics["auc_roc"], metrics["cv_auc_mean"], metrics["cv_auc_std"],
        )
    if is_synthetic and "recovery_rate_14d" in outcome_metrics:
        logger.info("Simulated recovery @14d: %.1f%% | Simulated score change: %+.1f pts",
                    outcome_metrics["recovery_rate_14d"] * 100, outcome_metrics["avg_score_improvement"])

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
