# src/bootstrap.py
"""
Bootstrap confidence-interval utilities.

Usage
-----
    from src.bootstrap import predict_interval

    models = joblib.load("model/bootstrap_models.joblib")
    result = predict_interval(models, X_input, alpha=0.1)
    # {"prediction": 387.2, "p10": 361.0, "p90": 415.4}
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BOOTSTRAP_PATH = Path(__file__).resolve().parent.parent / "model" / "bootstrap_models.joblib"
PIPELINE_PATH = Path(__file__).resolve().parent.parent / "model" / "ev_range_pipeline.joblib"

_pipeline: object = None
_bootstrap_models: list | None = None


def _load_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(PIPELINE_PATH)
    return _pipeline


def _load_bootstrap_models():
    global _bootstrap_models
    if _bootstrap_models is None:
        _bootstrap_models = joblib.load(BOOTSTRAP_PATH)
    return _bootstrap_models


def predict_interval(
    X: pd.DataFrame,
    alpha: float = 0.1,
    pipeline=None,
    bootstrap_models=None,
) -> dict:
    """
    Generate point prediction and bootstrap confidence interval.

    Parameters
    ----------
    X : pd.DataFrame
        Input features (1 or more rows).  Must contain all 21 feature columns.
    alpha : float
        Tail probability on each side.  Default 0.1 → p10–p90 interval.
    pipeline : sklearn Pipeline or None
        If None, loads from ``model/ev_range_pipeline.joblib``.
    bootstrap_models : list or None
        If None, loads from ``model/bootstrap_models.joblib``.

    Returns
    -------
    dict with keys:
        ``prediction``  — point estimate from the main pipeline (float)
        ``p_lower``     — lower percentile prediction (float)
        ``p_upper``     — upper percentile prediction (float)
        ``interval_pct``— e.g. "p10–p90"
    """
    if pipeline is None:
        pipeline = _load_pipeline()
    if bootstrap_models is None:
        bootstrap_models = _load_bootstrap_models()

    point_pred = pipeline.predict(X)

    boot_preds = np.array([m.predict(X) for m in bootstrap_models])  # (N_BOOT, n_rows)

    lower = np.percentile(boot_preds, alpha * 100, axis=0)
    upper = np.percentile(boot_preds, (1 - alpha) * 100, axis=0)

    if len(point_pred) == 1:
        return {
            "prediction":   round(float(point_pred[0]), 2),
            "p_lower":      round(float(lower[0]), 2),
            "p_upper":      round(float(upper[0]), 2),
            "interval_pct": f"p{int(alpha*100)}–p{int((1-alpha)*100)}",
        }

    return {
        "prediction":   [round(float(v), 2) for v in point_pred],
        "p_lower":      [round(float(v), 2) for v in lower],
        "p_upper":      [round(float(v), 2) for v in upper],
        "interval_pct": f"p{int(alpha*100)}–p{int((1-alpha)*100)}",
    }
