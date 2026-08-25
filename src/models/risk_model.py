"""
Risk Model — LightGBM + Calibration
=====================================
Trains a binary classifier to predict 14-day disengagement.
Outputs calibrated probabilities and risk tiers.

Key design decisions:
- LightGBM: handles NaN natively, fast, good on tabular data
- CalibratedClassifierCV: Platt scaling for reliable probabilities
- scale_pos_weight: handles class imbalance
- Stratified K-Fold evaluation with AUC-ROC + Precision/Recall
"""

import numpy as np
import pandas as pd
import yaml
import logging
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

MODEL_CFG = CFG["model"]
TIER_CFG  = CFG["risk_tiers"]


# ── Risk Tier Assignment ───────────────────────────────────────────────────────

def assign_risk_tier(score: float) -> str:
    if score < TIER_CFG["low"][1]:
        return "Low"
    elif score < TIER_CFG["medium"][1]:
        return "Medium"
    elif score < TIER_CFG["high"][1]:
        return "High"
    return "Critical"


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    feat_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple:
    """
    Train and calibrate the risk model.

    Returns
    -------
    calibrated_model : sklearn pipeline
    scaler           : fitted StandardScaler
    metrics          : dict of evaluation metrics
    X_test, y_test   : held-out test set
    """
    merged = feat_df.merge(labels_df[["student_id", "disengaged_in_14d"]], on="student_id")
    X = merged[feature_cols].values.astype(np.float32)
    y = merged["disengaged_in_14d"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=MODEL_CFG["test_size"],
        random_state=MODEL_CFG["random_state"],
        stratify=y,
    )

    # Scale features (improves calibration)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Base LightGBM
    base_lgbm = lgb.LGBMClassifier(
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

    # Calibrate with Platt scaling (sigmoid)
    calibrated = CalibratedClassifierCV(base_lgbm, method="sigmoid", cv=3)
    calibrated.fit(X_train_s, y_train)

    # Evaluate
    y_prob = calibrated.predict_proba(X_test_s)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    metrics = {
        "auc_roc":          float(roc_auc_score(y_test, y_prob)),
        "avg_precision":    float(average_precision_score(y_test, y_prob)),
        "precision_at_50":  float(precision_score(y_test, y_pred, zero_division=0)),
        "recall_at_50":     float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_at_50":         float(f1_score(y_test, y_pred, zero_division=0)),
        "test_size":        int(len(y_test)),
        "positive_rate":    float(y_test.mean()),
    }

    logger.info(
        "Model trained | AUC-ROC: %.3f | AP: %.3f | F1: %.3f",
        metrics["auc_roc"], metrics["avg_precision"], metrics["f1_at_50"],
    )
    _log_classification_report(y_test, y_pred)

    return calibrated, scaler, metrics, X_test_s, y_test


def _log_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, target_names=["Engaged", "At-Risk"])
    logger.info("\n%s", report)


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_risk(
    feat_df: pd.DataFrame,
    feature_cols: list[str],
    model,
    scaler,
) -> pd.DataFrame:
    """
    Score all students.

    Returns DataFrame with: student_id, risk_score, risk_tier.
    """
    X = feat_df[feature_cols].values.astype(np.float32)
    X_s = scaler.transform(X)
    probs = model.predict_proba(X_s)[:, 1]

    result = pd.DataFrame({
        "student_id": feat_df["student_id"].values,
        "risk_score": probs,
    })
    result["risk_tier"] = result["risk_score"].apply(assign_risk_tier)
    result["risk_score"] = result["risk_score"].round(4)
    return result.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ── Cross-Validation Summary ──────────────────────────────────────────────────

def cross_validate_model(
    feat_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    feature_cols: list[str],
    n_splits: int = 5,
) -> dict:
    """Run stratified K-fold CV and return mean/std of key metrics."""
    merged = feat_df.merge(labels_df[["student_id", "disengaged_in_14d"]], on="student_id")
    X = merged[feature_cols].values.astype(np.float32)
    y = merged["disengaged_in_14d"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs, aps, f1s = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)

        m = lgb.LGBMClassifier(
            n_estimators=200,
            scale_pos_weight=MODEL_CFG["scale_pos_weight"],
            random_state=42, verbose=-1, n_jobs=-1,
        )
        m.fit(X_tr_s, y_tr)
        prob = m.predict_proba(X_va_s)[:, 1]
        pred = (prob >= 0.5).astype(int)

        aucs.append(roc_auc_score(y_va, prob))
        aps.append(average_precision_score(y_va, prob))
        f1s.append(f1_score(y_va, pred, zero_division=0))
        logger.info("Fold %d | AUC: %.3f | AP: %.3f | F1: %.3f", fold, aucs[-1], aps[-1], f1s[-1])

    return {
        "cv_auc_mean": float(np.mean(aucs)),
        "cv_auc_std":  float(np.std(aucs)),
        "cv_ap_mean":  float(np.mean(aps)),
        "cv_f1_mean":  float(np.mean(f1s)),
    }
