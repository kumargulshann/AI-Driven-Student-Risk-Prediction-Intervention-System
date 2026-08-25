"""
Model Registry — Save & Load Artifacts
========================================
Handles serialization of the trained model, scaler, feature columns,
and evaluation metrics using joblib.
"""

import joblib
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

ARTIFACTS_DIR = ROOT / CFG["model"]["artifacts_dir"]


def save_model(model, scaler, feature_cols: list[str], metrics: dict) -> Path:
    """Persist all model artifacts to disk. Returns the artifact directory path."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model,        ARTIFACTS_DIR / "risk_model.pkl")
    joblib.dump(scaler,       ARTIFACTS_DIR / "scaler.pkl")
    joblib.dump(feature_cols, ARTIFACTS_DIR / "feature_cols.pkl")

    meta = {
        "trained_at": datetime.now().isoformat(),
        "metrics":    metrics,
        "n_features": len(feature_cols),
    }
    with open(ARTIFACTS_DIR / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Artifacts saved to %s", ARTIFACTS_DIR)
    return ARTIFACTS_DIR


def load_model():
    """Load trained model artifacts from disk. Returns (model, scaler, feature_cols, metadata)."""
    model_path = ARTIFACTS_DIR / "risk_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained model found at {model_path}.\n"
            "Run: python src/pipeline.py"
        )
    model        = joblib.load(ARTIFACTS_DIR / "risk_model.pkl")
    scaler       = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
    feature_cols = joblib.load(ARTIFACTS_DIR / "feature_cols.pkl")

    with open(ARTIFACTS_DIR / "metadata.json") as f:
        metadata = json.load(f)

    logger.info("Model loaded | AUC-ROC: %.3f", metadata["metrics"].get("auc_roc", 0))
    return model, scaler, feature_cols, metadata


def model_exists() -> bool:
    return (ARTIFACTS_DIR / "risk_model.pkl").exists()
