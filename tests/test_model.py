"""
Tests — Risk Model
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.risk_model import assign_risk_tier


# ── Risk Tier Tests ───────────────────────────────────────────────────────────

def test_tier_low():
    assert assign_risk_tier(0.10) == "Low"
    assert assign_risk_tier(0.39) == "Low"


def test_tier_medium():
    assert assign_risk_tier(0.40) == "Medium"
    assert assign_risk_tier(0.64) == "Medium"


def test_tier_high():
    assert assign_risk_tier(0.65) == "High"
    assert assign_risk_tier(0.79) == "High"


def test_tier_critical():
    assert assign_risk_tier(0.80) == "Critical"
    assert assign_risk_tier(0.99) == "Critical"
    assert assign_risk_tier(1.00) == "Critical"


def test_tier_boundary_values():
    # Exact boundary: 0.40 should be Medium, not Low
    assert assign_risk_tier(0.40) != "Low"
    # Exact boundary: 0.65 should be High, not Medium
    assert assign_risk_tier(0.65) != "Medium"


# ── Model Output Shape Tests ──────────────────────────────────────────────────

def test_predict_risk_returns_correct_columns():
    """Smoke test: train a tiny model and check output schema."""
    from sklearn.calibration import CalibratedClassifierCV
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from src.models.risk_model import predict_risk

    np.random.seed(0)
    n = 200
    feat_df = pd.DataFrame({
        "student_id": np.arange(n),
        "feat1": np.random.rand(n),
        "feat2": np.random.rand(n),
        "feat3": np.random.rand(n),
    })
    labels  = np.random.randint(0, 2, size=n)
    feature_cols = ["feat1", "feat2", "feat3"]

    X = feat_df[feature_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    base = lgb.LGBMClassifier(n_estimators=5, verbose=-1)
    model = CalibratedClassifierCV(base, cv=2)
    model.fit(X_s, labels)

    result = predict_risk(feat_df, feature_cols, model, scaler)

    assert "student_id" in result.columns
    assert "risk_score" in result.columns
    assert "risk_tier" in result.columns
    assert len(result) == n
    assert result["risk_score"].between(0, 1).all()
    assert set(result["risk_tier"].unique()).issubset({"Low", "Medium", "High", "Critical"})
