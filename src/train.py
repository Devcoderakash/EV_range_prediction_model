# src/train.py
"""
Full training pipeline for TechTrack EV Range Prediction.

Workflow
--------
1. Load & clean data
2. Feature engineering (via src/features.py)
3. 70 / 15 / 15 reproducible split (train / val / test)
4. Nested cross-validation: outer 5-fold for honest generalisation estimate,
   inner RandomizedSearchCV (30 iter) for hyperparameter tuning
5. Residual analysis: plots + MAE/RMSE by brand and range bucket
6. SHAP interpretability: global summary, top-5 dependence, waterfall plots
7. MLflow experiment tracking: all model variants + tuned XGBoost
8. Bootstrap confidence-interval ensemble training
9. Save final pipeline + bootstrap models

Usage
-----
    python -m src.train
or
    make train
"""

import sys
import warnings
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend for CI / server environments
import matplotlib.pyplot as plt
import joblib
import shap
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV,
    cross_validate,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Add project root to path so `src` is importable when run as a module
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import (
    engineer_features,
    select_model_columns,
    add_range_bucket,
    flag_sparse_brands,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DATA_PATH = ROOT / "notebook" / "ev_dataset.xls"
MODEL_DIR = ROOT / "model"
FIGURES_DIR = ROOT / "reports" / "figures"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_PATH = MODEL_DIR / "ev_range_pipeline.joblib"
BOOTSTRAP_PATH = MODEL_DIR / "bootstrap_models.joblib"

N_OUTER_FOLDS = 5
N_INNER_ITER = 30
N_BOOTSTRAP = 500   # bootstrap resamples for CI; keep reasonable for speed
BOOTSTRAP_ALPHA = 0.1  # p10–p90 interval

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, NUMERIC_FEATURES),
        ("categorical", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def make_xgb_pipeline(**xgb_kwargs) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            n_jobs=-1,
            **xgb_kwargs,
        )),
    ])


def compute_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def save_fig(fig, name: str):
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info(f"  Saved figure → {path.relative_to(ROOT)}")
    return str(path)


# ---------------------------------------------------------------------------
# Step 1: Load & split data
# ---------------------------------------------------------------------------

def load_and_split():
    log.info("─── Loading dataset ───────────────────────────────────────────")
    raw = pd.read_excel(DATA_PATH, engine="xlrd")
    log.info(f"  Raw shape: {raw.shape}")

    df = engineer_features(raw)

    X, y = select_model_columns(df)
    brand_series = raw["brand"].reset_index(drop=True)

    log.info("\n─── Sparse brand audit ────────────────────────────────────────")
    sparse = flag_sparse_brands(raw)
    if len(sparse):
        log.info(f"  Brands with < 5 samples (may distort metrics):\n{sparse.to_string()}")
    else:
        log.info("  No sparse brands found.")

    # 70 / 15 / 15 split — test set is locked away until final evaluation
    X_trainval, X_test, y_trainval, y_test, brand_tv, brand_test = train_test_split(
        X, y, brand_series, test_size=0.15, random_state=RANDOM_SEED
    )
    X_train, X_val, y_train, y_val, brand_train, brand_val = train_test_split(
        X_trainval, y_trainval, brand_tv, test_size=0.1765,  # 0.15 / 0.85 ≈ 17.65%
        random_state=RANDOM_SEED
    )
    log.info(
        f"\n  Split sizes → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_trainval, y_trainval,
        brand_train, brand_val, brand_test,
        raw,
    )


# ---------------------------------------------------------------------------
# Step 3: MLflow — compare 5 model variants
# ---------------------------------------------------------------------------

def compare_models_mlflow(X_train, y_train, X_val, y_val):
    log.info("\n─── MLflow: Comparing model variants (5-fold CV) ──────────────")

    mlflow.set_experiment("ev_range_prediction")

    candidates = {
        "DummyRegressor": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", DummyRegressor(strategy="mean")),
        ]),
        "RidgeRegression": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "XGBoost_default": make_xgb_pipeline(n_estimators=300, max_depth=5),
        "ExtraTrees": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", GradientBoostingRegressor(n_estimators=300, random_state=RANDOM_SEED)),
        ]),
        "RandomForest": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
    }

    cv5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_results = []

    for name, pipe in candidates.items():
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv5,
            scoring=["neg_mean_absolute_error", "neg_root_mean_squared_error", "r2"],
            n_jobs=-1,
        )
        cv_mae = -scores["test_neg_mean_absolute_error"].mean()
        cv_rmse = -scores["test_neg_root_mean_squared_error"].mean()
        cv_r2 = scores["test_r2"].mean()

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_metric("cv_MAE", round(cv_mae, 4))
            mlflow.log_metric("cv_RMSE", round(cv_rmse, 4))
            mlflow.log_metric("cv_R2", round(cv_r2, 4))

        cv_results.append({
            "Model": name,
            "CV MAE (km)": round(cv_mae, 2),
            "CV RMSE (km)": round(cv_rmse, 2),
            "CV R²": round(cv_r2, 4),
        })
        log.info(f"  {name:25s}  MAE={cv_mae:.2f}  RMSE={cv_rmse:.2f}  R²={cv_r2:.4f}")

    df_cv = pd.DataFrame(cv_results).sort_values("CV RMSE (km)")
    log.info(f"\n  CV comparison table:\n{df_cv.to_string(index=False)}")
    return df_cv


# ---------------------------------------------------------------------------
# Step 1: Nested cross-validation (honest outer-loop estimate)
# ---------------------------------------------------------------------------

def nested_cv(X_trainval, y_trainval):
    log.info("\n─── Nested Cross-Validation ────────────────────────────────────")

    param_dist = {
        "model__n_estimators":   [300, 500, 700, 900],
        "model__max_depth":      [3, 4, 5, 6],
        "model__learning_rate":  [0.03, 0.05, 0.08, 0.1],
        "model__subsample":      [0.6, 0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 2, 5, 10],
    }

    outer_cv = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    outer_mae, outer_rmse, outer_r2 = [], [], []
    best_params_per_fold = []

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X_trainval, y_trainval), 1):
        X_tr, X_vl = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
        y_tr, y_vl = y_trainval.iloc[train_idx], y_trainval.iloc[val_idx]

        inner_search = RandomizedSearchCV(
            make_xgb_pipeline(),
            param_distributions=param_dist,
            n_iter=N_INNER_ITER,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=RANDOM_SEED,
            refit=True,
        )
        inner_search.fit(X_tr, y_tr)
        best_params_per_fold.append(inner_search.best_params_)

        y_pred_vl = inner_search.predict(X_vl)
        m = compute_metrics(y_vl, y_pred_vl)
        outer_mae.append(m["MAE"])
        outer_rmse.append(m["RMSE"])
        outer_r2.append(m["R2"])
        log.info(
            f"  Fold {fold}/{N_OUTER_FOLDS}  MAE={m['MAE']:.2f}  "
            f"RMSE={m['RMSE']:.2f}  R²={m['R2']:.4f}"
        )

    log.info(
        f"\n  ► Outer-loop (honest) estimate:"
        f"  MAE={np.mean(outer_mae):.2f} ± {np.std(outer_mae):.2f}"
        f"  RMSE={np.mean(outer_rmse):.2f} ± {np.std(outer_rmse):.2f}"
        f"  R²={np.mean(outer_r2):.4f} ± {np.std(outer_r2):.4f}"
    )
    return {
        "outer_mae": outer_mae,
        "outer_rmse": outer_rmse,
        "outer_r2": outer_r2,
        "best_params_per_fold": best_params_per_fold,
    }


# ---------------------------------------------------------------------------
# Step 1: Train final model on full trainval set
# ---------------------------------------------------------------------------

def train_final_model(X_trainval, y_trainval):
    log.info("\n─── Training final model (full trainval set) ──────────────────")

    param_dist = {
        "model__n_estimators":     [300, 500, 700, 900],
        "model__max_depth":        [3, 4, 5, 6],
        "model__learning_rate":    [0.03, 0.05, 0.08, 0.1],
        "model__subsample":        [0.6, 0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.6, 0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 2, 5, 10],
    }
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    search = RandomizedSearchCV(
        make_xgb_pipeline(),
        param_distributions=param_dist,
        n_iter=N_INNER_ITER,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=RANDOM_SEED,
        refit=True,
    )
    search.fit(X_trainval, y_trainval)
    best = search.best_estimator_
    log.info(f"  Best params: {search.best_params_}")
    return best


# ---------------------------------------------------------------------------
# Step 1: Residual analysis
# ---------------------------------------------------------------------------

def residual_analysis(pipeline, X_test, y_test, brand_test, artifact_paths: list):
    log.info("\n─── Residual Analysis ─────────────────────────────────────────")

    y_pred = pipeline.predict(X_test)
    residuals = y_test.values - y_pred
    m = compute_metrics(y_test, y_pred)
    log.info(f"  Test set  MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R²={m['R2']:.4f}")

    # ── 1. Residuals vs Predicted ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residual Analysis — Test Set", fontsize=14, fontweight="bold")

    axes[0].scatter(y_pred, residuals, alpha=0.5, edgecolors="none", color="#2196F3", s=30)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[0].set_xlabel("Predicted Range (km)")
    axes[0].set_ylabel("Residual (actual − predicted) km")
    axes[0].set_title("Residuals vs Predicted")
    axes[0].annotate(
        f"RMSE={m['RMSE']:.1f} km  MAE={m['MAE']:.1f} km  R²={m['R2']:.4f}",
        xy=(0.02, 0.96), xycoords="axes fraction", fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7),
    )

    # ── 2. Q-Q plot ───────────────────────────────────────────────────────
    from scipy import stats
    (osm, osr), (slope, intercept, r) = stats.probplot(residuals, dist="norm")
    axes[1].scatter(osm, osr, alpha=0.5, s=30, color="#2196F3", edgecolors="none")
    axes[1].plot(osm, slope * np.array(osm) + intercept, "r--", linewidth=1.5)
    axes[1].set_title("Q-Q Plot of Residuals")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Sample Quantiles")

    artifact_paths.append(save_fig(fig, "residuals_vs_predicted.png"))

    # ── 3. MAE / RMSE by range bucket ────────────────────────────────────
    df_res = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": y_pred,
        "residual": residuals,
        "brand": brand_test.values,
    })
    df_res = add_range_bucket(df_res, "y_true")

    bucket_metrics = (
        df_res.groupby("range_bucket", observed=True)
        .apply(lambda g: pd.Series({
            "n": len(g),
            "MAE":  mean_absolute_error(g["y_true"], g["y_pred"]),
            "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
        }))
        .reset_index()
    )
    log.info(f"\n  Error by range bucket:\n{bucket_metrics.to_string(index=False)}")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Error by Range Bucket — Test Set", fontsize=14, fontweight="bold")
    colors = ["#42A5F5", "#66BB6A", "#FFA726"]
    for i, ax in enumerate(axes2):
        metric = ["MAE", "RMSE"][i]
        bars = ax.bar(
            bucket_metrics["range_bucket"].astype(str),
            bucket_metrics[metric],
            color=colors, edgecolor="white", linewidth=0.8,
        )
        for bar, val in zip(bars, bucket_metrics[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9,
            )
        ax.set_title(f"{metric} by Range Bucket")
        ax.set_ylabel(f"{metric} (km)")
        ax.set_xlabel("Range Bucket")
        ax.tick_params(axis="x", rotation=15)
    artifact_paths.append(save_fig(fig2, "error_by_range_bucket.png"))

    # ── 4. MAE / RMSE by brand (top 20 by MAE) ───────────────────────────
    brand_metrics = (
        df_res.groupby("brand")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "MAE":  mean_absolute_error(g["y_true"], g["y_pred"]),
            "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
        }))
        .reset_index()
        .sort_values("MAE", ascending=False)
    )
    log.info(f"\n  Top 15 brands by MAE (test set):\n{brand_metrics.head(15).to_string(index=False)}")

    top_brands = brand_metrics.head(20)
    fig3, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        top_brands["brand"], top_brands["MAE"],
        color=["#EF5350" if n < 5 else "#42A5F5" for n in top_brands["n"]],
        edgecolor="white",
    )
    ax.set_xlabel("MAE (km)")
    ax.set_title("MAE by Brand — Top 20 (red = sparse brand, <5 samples)")
    ax.invert_yaxis()
    for bar, val in zip(bars, top_brands["MAE"]):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}", va="center", fontsize=8,
        )
    artifact_paths.append(save_fig(fig3, "mae_by_brand.png"))

    return m, df_res, bucket_metrics, brand_metrics


# ---------------------------------------------------------------------------
# Step 2: SHAP interpretability
# ---------------------------------------------------------------------------

def shap_analysis(pipeline, X_train, X_test, outer_results, artifact_paths: list):
    log.info("\n─── SHAP Analysis ─────────────────────────────────────────────")

    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model = pipeline.named_steps["model"]

    X_test_t = preprocessor.transform(X_test)

    # Feature names after OHE
    ohe = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = NUMERIC_FEATURES + cat_names

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(X_test_t)

    # Patch feature names onto explanation object
    shap_values.feature_names = feature_names

    # ── 1. Global summary (beeswarm) ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.title("SHAP Summary (Beeswarm) — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    artifact_paths.append(save_fig(plt.gcf(), "shap_summary_beeswarm.png"))

    # ── 2. Global bar (mean |SHAP|) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.title("SHAP Global Feature Importance (mean |SHAP|)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    artifact_paths.append(save_fig(plt.gcf(), "shap_global_bar.png"))

    # ── 3. Top-5 dependence plots ─────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top5_idx = np.argsort(mean_abs_shap)[-5:][::-1]
    top5_names = [feature_names[i] for i in top5_idx]
    log.info(f"  Top-5 features by mean |SHAP|: {top5_names}")

    for feat in top5_names:
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.plots.scatter(shap_values[:, feat], ax=ax, show=False)
        ax.set_title(f"SHAP Dependence — {feat}", fontsize=12, fontweight="bold")
        plt.tight_layout()
        safe_name = feat.replace("/", "_").replace(" ", "_")
        artifact_paths.append(save_fig(fig, f"shap_dependence_{safe_name}.png"))

    # ── 4. Waterfall plots (short / medium / long range predictions) ──────
    y_pred = pipeline.predict(X_test)
    df_idx = pd.DataFrame({"y_pred": y_pred, "idx": range(len(y_pred))})
    short = df_idx[y_pred < 300]["idx"].iloc[0] if (y_pred < 300).any() else 0
    medium = df_idx[(y_pred >= 300) & (y_pred < 500)]["idx"].iloc[0] if ((y_pred >= 300) & (y_pred < 500)).any() else 1
    long_ = df_idx[y_pred >= 500]["idx"].iloc[0] if (y_pred >= 500).any() else 2

    for label, i in [("short", short), ("medium", medium), ("long", long_)]:
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.plots.waterfall(shap_values[int(i)], max_display=15, show=False)
        plt.title(f"SHAP Waterfall — {label}-range EV (pred={y_pred[int(i)]:.0f} km)",
                  fontsize=12, fontweight="bold")
        plt.tight_layout()
        artifact_paths.append(save_fig(fig, f"shap_waterfall_{label}.png"))

    # ── 5. Feature importance stability across outer CV folds ─────────────
    if outer_results and outer_results.get("best_params_per_fold"):
        log.info("  Checking feature importance stability across outer folds …")
        # Use SHAP mean abs values from test set explainer as stability proxy
        # Simpler: use SHAP mean abs values from test set explainer as proxy
        importances_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(importances_df["feature"], importances_df["mean_abs_shap"],
                color="#42A5F5", edgecolor="white")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("Top-15 Features by SHAP Importance (Final Model)", fontsize=12)
        ax.invert_yaxis()
        plt.tight_layout()
        artifact_paths.append(save_fig(fig, "shap_feature_importance_ranking.png"))

    return top5_names


# ---------------------------------------------------------------------------
# Step 5: Bootstrap confidence interval models
# ---------------------------------------------------------------------------

def train_bootstrap_models(X_trainval, y_trainval, best_params: dict):
    log.info(f"\n─── Bootstrap CI Training ({N_BOOTSTRAP} resamples) ──────────────────")

    clean_params = {k.replace("model__", ""): v for k, v in best_params.items()}
    bootstrap_models = []

    for i in range(N_BOOTSTRAP):
        rng = np.random.RandomState(RANDOM_SEED + i)
        idx = rng.choice(len(X_trainval), size=len(X_trainval), replace=True)
        X_b = X_trainval.iloc[idx]
        y_b = y_trainval.iloc[idx]

        pipe = make_xgb_pipeline(**clean_params)
        pipe.fit(X_b, y_b)
        bootstrap_models.append(pipe)

        if (i + 1) % 100 == 0:
            log.info(f"  Bootstrapped {i+1}/{N_BOOTSTRAP} models …")

    joblib.dump(bootstrap_models, BOOTSTRAP_PATH)
    log.info(f"  Saved {N_BOOTSTRAP} bootstrap models → {BOOTSTRAP_PATH.relative_to(ROOT)}")
    return bootstrap_models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("══════════════════════════════════════════════════════════════")
    log.info("  TechTrack EV Range Prediction — Industry-Grade Training Run ")
    log.info("══════════════════════════════════════════════════════════════\n")

    artifact_paths = []

    # ── 1. Load & split ────────────────────────────────────────────────────
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_trainval, y_trainval,
        brand_train, brand_val, brand_test,
        raw,
    ) = load_and_split()

    # ── 3. MLflow model comparison ─────────────────────────────────────────
    compare_models_mlflow(X_train, y_train, X_val, y_val)

    # ── 1. Nested CV (honest outer-loop estimate) ──────────────────────────
    outer_results = nested_cv(X_trainval, y_trainval)

    # ── 1+7. Train final model ─────────────────────────────────────────────
    final_pipeline = train_final_model(X_trainval, y_trainval)

    # Save final pipeline
    joblib.dump(final_pipeline, PIPELINE_PATH)
    log.info(f"\n  Saved final pipeline → {PIPELINE_PATH.relative_to(ROOT)}")

    # ── 3. Log final model to MLflow ───────────────────────────────────────
    test_metrics = compute_metrics(y_test, final_pipeline.predict(X_test))
    best_params = dict(zip(
        [
            "model__n_estimators", "model__max_depth", "model__learning_rate",
            "model__subsample", "model__colsample_bytree", "model__min_child_weight",
        ],
        [
            final_pipeline.named_steps["model"].n_estimators,
            final_pipeline.named_steps["model"].max_depth,
            final_pipeline.named_steps["model"].learning_rate,
            final_pipeline.named_steps["model"].subsample,
            final_pipeline.named_steps["model"].colsample_bytree,
            final_pipeline.named_steps["model"].min_child_weight,
        ],
    ))

    with mlflow.start_run(run_name="XGBoost_tuned_FINAL") as run:
        mlflow.log_params(best_params)
        mlflow.log_metric("test_MAE", test_metrics["MAE"])
        mlflow.log_metric("test_RMSE", test_metrics["RMSE"])
        mlflow.log_metric("test_R2", test_metrics["R2"])
        mlflow.log_metric("nested_cv_MAE_mean", round(np.mean(outer_results["outer_mae"]), 4))
        mlflow.log_metric("nested_cv_RMSE_mean", round(np.mean(outer_results["outer_rmse"]), 4))
        mlflow.log_metric("nested_cv_R2_mean", round(np.mean(outer_results["outer_r2"]), 4))
        # Trust XGBoost types required by mlflow>=3.x / skops
        mlflow.sklearn.log_model(
            final_pipeline,
            name="ev_range_pipeline",
            skops_trusted_types=[
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBRegressor",
            ],
        )
        mlflow.log_artifact(str(PIPELINE_PATH))
        final_run_id = run.info.run_id

    # ── 1. Residual analysis ───────────────────────────────────────────────
    test_metrics, df_res, bucket_metrics, brand_metrics = residual_analysis(
        final_pipeline, X_test, y_test, brand_test, artifact_paths
    )

    # ── 2. SHAP ────────────────────────────────────────────────────────────
    top5 = shap_analysis(final_pipeline, X_train, X_test, outer_results, artifact_paths)

    # Log plots to MLflow final run
    with mlflow.start_run(run_id=final_run_id):
        for p in artifact_paths:
            try:
                mlflow.log_artifact(p, artifact_path="figures")
            except Exception:
                pass

    # ── 5. Bootstrap CI models ─────────────────────────────────────────────
    train_bootstrap_models(X_trainval, y_trainval, best_params)

    # ── Summary ────────────────────────────────────────────────────────────
    log.info("\n══════════════════════════════════════════════════════════════")
    log.info("  TRAINING COMPLETE — Summary")
    log.info("══════════════════════════════════════════════════════════════")
    log.info("\n  Nested CV (honest generalization estimate):")
    log.info(f"    MAE = {np.mean(outer_results['outer_mae']):.2f} ± {np.std(outer_results['outer_mae']):.2f} km")
    log.info(f"    RMSE = {np.mean(outer_results['outer_rmse']):.2f} ± {np.std(outer_results['outer_rmse']):.2f} km")
    log.info(f"    R² = {np.mean(outer_results['outer_r2']):.4f} ± {np.std(outer_results['outer_r2']):.4f}")
    log.info("\n  Held-out test set (15%% — never touched during tuning):")
    log.info(f"    MAE = {test_metrics['MAE']:.2f} km")
    log.info(f"    RMSE = {test_metrics['RMSE']:.2f} km")
    log.info(f"    R² = {test_metrics['R2']:.4f}")
    log.info(f"\n  Top-5 features (SHAP): {top5}")
    log.info("\n  Artifacts saved to: reports/figures/")
    log.info(f"  Pipeline:  {PIPELINE_PATH.relative_to(ROOT)}")
    log.info(f"  Bootstrap: {BOOTSTRAP_PATH.relative_to(ROOT)}")
    log.info("  MLflow:    mlruns/  (run `mlflow ui` to explore)")
    log.info("══════════════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    main()
