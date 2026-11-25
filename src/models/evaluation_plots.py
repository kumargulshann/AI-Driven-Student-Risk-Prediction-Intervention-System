"""Save reproducible evaluation figures without a graphical display."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.calibration import CalibrationDisplay


def save_calibration_plot(y_test, raw_prob, calibrated_prob, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    CalibrationDisplay.from_predictions(
        y_test, raw_prob, n_bins=10, strategy="quantile", name="Raw LightGBM", ax=ax,
    )
    CalibrationDisplay.from_predictions(
        y_test, calibrated_prob, n_bins=10, strategy="quantile", name="After Platt calibration", ax=ax,
        ref_line=False,
    )
    ax.set_title("Calibration on the held-out test set")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_shap_summary(explainer, X_test, path: Path, seed: int) -> None:
    import shap

    sample_size = min(1_000, len(X_test))
    rows = np.random.default_rng(seed).choice(len(X_test), size=sample_size, replace=False)
    sample = X_test[rows]
    values = explainer.explain_batch(sample)
    path.parent.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(values, sample, feature_names=explainer.feature_cols, show=False)
    fig = plt.gcf()
    plt.xlabel("SHAP contribution to raw LightGBM margin")
    plt.title("Held-out student sample")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
