"""Train and evaluate the calibrated LightGBM student risk model."""

import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
with (ROOT / "config.yaml").open(encoding="utf-8") as config_file:
    CFG = yaml.safe_load(config_file)

MODEL_CFG = CFG["model"]
TIER_CFG = CFG["risk_tiers"]
THRESHOLD = 0.50


def assign_risk_tier(score: float) -> str:
    if score < TIER_CFG["low"][1]:
        return "Low"
    if score < TIER_CFG["medium"][1]:
        return "Medium"
    if score < TIER_CFG["high"][1]:
        return "High"
    return "Critical"


def _split_data(feat_df, labels_df, feature_cols):
    merged = feat_df.merge(
        labels_df[["student_id", "disengaged_in_14d"]], on="student_id", validate="one_to_one",
    )
    X = merged[feature_cols].to_numpy(dtype=np.float32)
    y = merged["disengaged_in_14d"].to_numpy()
    return train_test_split(
        X, y, stratify=y, test_size=MODEL_CFG["test_size"], random_state=MODEL_CFG["random_state"],
    )


def _fit_calibrated(X, y):
    base = lgb.LGBMClassifier(
        n_estimators=MODEL_CFG["n_estimators"],
        learning_rate=MODEL_CFG["learning_rate"],
        max_depth=MODEL_CFG["max_depth"],
        num_leaves=MODEL_CFG["num_leaves"],
        min_child_samples=MODEL_CFG["min_child_samples"],
        scale_pos_weight=MODEL_CFG["scale_pos_weight"],
        random_state=MODEL_CFG["random_state"],
        n_jobs=-1,
        verbose=-1,
    )
    # Learn the sigmoid from out-of-fold probabilities, then fit the base model on all training rows.
    return CalibratedClassifierCV(base, method="sigmoid", cv=3, ensemble=False).fit(X, y)


def train_model(
    feat_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    calibration_plot: Path | None = None,
) -> tuple:
    """Fit on the training split and measure both predictions on the untouched test split."""
    X_train, X_test, y_train, y_test = _split_data(feat_df, labels_df, feature_cols)
    # Keep the registry's transformer interface. Trees can work with the original units and NaNs.
    scaler = StandardScaler(with_mean=False, with_std=False).fit(X_train)
    model = _fit_calibrated(X_train, y_train)
    base = model.calibrated_classifiers_[0].estimator
    raw_prob = base.predict_proba(X_test)[:, 1]
    calibrated_prob = model.predict_proba(X_test)[:, 1]
    prediction = calibrated_prob >= THRESHOLD

    metrics = {
        "auc_roc": float(roc_auc_score(y_test, calibrated_prob)),
        "avg_precision": float(average_precision_score(y_test, calibrated_prob)),
        "precision_at_50": float(precision_score(y_test, prediction, zero_division=0)),
        "recall_at_50": float(recall_score(y_test, prediction)),
        "f1_at_50": float(f1_score(y_test, prediction)),
        "brier_before": float(brier_score_loss(y_test, raw_prob)),
        "brier_after": float(brier_score_loss(y_test, calibrated_prob)),
        "test_size": int(len(y_test)),
        "positive_rate": float(y_test.mean()),
    }
    logger.info(
        "Held-out test AUC: %.6f | Precision @ 0.50: %.6f | Recall @ 0.50: %.6f",
        metrics["auc_roc"], metrics["precision_at_50"], metrics["recall_at_50"],
    )
    logger.info(
        "Brier score before Platt: %.6f | After Platt: %.6f",
        metrics["brier_before"], metrics["brier_after"],
    )
    logger.info("\n%s", classification_report(
        y_test, prediction, target_names=["Engaged", "At-Risk"], zero_division=0,
    ))
    if calibration_plot is not None:
        from src.models.evaluation_plots import save_calibration_plot

        save_calibration_plot(y_test, raw_prob, calibrated_prob, calibration_plot)

    return model, scaler, metrics, X_test, y_test


def predict_risk(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    model,
    scaler,
) -> pd.DataFrame:
    """Return student scores and risk tiers, ordered by the model's probability."""
    X = scaler.transform(feat_df[feature_cols].to_numpy(dtype=np.float32))
    probabilities = model.predict_proba(X)[:, 1]
    result = pd.DataFrame({
        "student_id": feat_df["student_id"].to_numpy(),
        "risk_score": probabilities,
    })
    result["risk_tier"] = result["risk_score"].map(assign_risk_tier)
    result = result.sort_values("risk_score", ascending=False, kind="stable").reset_index(drop=True)
    result["risk_score"] = result["risk_score"].round(4)
    return result


def cross_validate_model(
    feat_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
) -> dict:
    """Evaluate the full model on development folds, leaving the test split untouched."""
    X, _, y, _ = _split_data(feat_df, labels_df, feature_cols)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=MODEL_CFG["random_state"])
    aucs, average_precisions, f1_scores = [], [], []
    for fold, (train, valid) in enumerate(folds.split(X, y), 1):
        model = _fit_calibrated(X[train], y[train])
        y_valid = y[valid]
        probability = model.predict_proba(X[valid])[:, 1]
        aucs.append(roc_auc_score(y_valid, probability))
        average_precisions.append(average_precision_score(y_valid, probability))
        f1_scores.append(f1_score(y_valid, probability >= THRESHOLD, zero_division=0))
        logger.info("CV fold %d/%d | AUC: %.6f", fold, n_splits, aucs[-1])

    mean, std = float(np.mean(aucs)), float(np.std(aucs))
    logger.info("%d-fold stratified CV AUC: %.6f ± %.6f", n_splits, mean, std)
    return {
        "cv_auc_mean": mean,
        "cv_auc_std": std,
        "cv_ap_mean": float(np.mean(average_precisions)),
        "cv_f1_mean": float(np.mean(f1_scores)),
    }
