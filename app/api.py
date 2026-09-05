# app/api.py
"""
FastAPI service for TechTrack EV Range Prediction.

Endpoints
---------
GET  /health   — liveness probe
POST /predict  — returns predicted range + p10/p90 confidence interval
POST /explain  — returns SHAP values for a given input

Usage
-----
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
import joblib
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.schemas import EVInput, PredictionResponse, HealthResponse, ExplainResponse
from src.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
from src.bootstrap import predict_interval

log = logging.getLogger("uvicorn.error")

PIPELINE_PATH = ROOT / "model" / "ev_range_pipeline.joblib"
BOOTSTRAP_PATH = ROOT / "model" / "bootstrap_models.joblib"

# ── Shared state loaded at startup ────────────────────────────────────────────
_state: dict = {
    "pipeline": None,
    "bootstrap_models": None,
    "explainer": None,
    "feature_names": None,
}


def _input_to_df(ev: EVInput) -> pd.DataFrame:
    """Convert an EVInput Pydantic model into a single-row feature DataFrame."""
    d = ev.model_dump()

    # Derive engineered features from raw dimensions
    length = float(d["length_mm"])
    width = float(d["width_mm"])
    height = float(d["height_mm"])
    batt = float(d["battery_capacity_kWh"])

    footprint = (length * width) / 1_000_000
    vol_proxy = (length * width * height) / 1_000_000_000
    batt_per_fp = batt / footprint if footprint > 0 else 0.0
    torque_per_b = (float(d["torque_nm"]) / batt) if d["torque_nm"] and batt > 0 else 0.0
    batt_per_vol = (batt / vol_proxy) if vol_proxy > 0 else 0.0

    row = {
        "top_speed_kmh":            d["top_speed_kmh"],
        "battery_capacity_kWh":     d["battery_capacity_kWh"],
        "number_of_cells":          d["number_of_cells"],
        "torque_nm":                d["torque_nm"],
        "acceleration_0_100_s":     d["acceleration_0_100_s"],
        "fast_charging_power_kw_dc": d["fast_charging_power_kw_dc"],
        "towing_capacity_kg":       d["towing_capacity_kg"],
        "cargo_volume_l":           d["cargo_volume_l"],
        "seats":                    d["seats"],
        "length_mm":                d["length_mm"],
        "width_mm":                 d["width_mm"],
        "height_mm":                d["height_mm"],
        "footprint_m2":             footprint,
        "volume_proxy_m3":          vol_proxy,
        "battery_per_footprint":    batt_per_fp,
        "torque_per_battery":       torque_per_b,
        "battery_per_volume":       batt_per_vol,
        "fast_charge_port":         d["fast_charge_port"],
        "drivetrain":               d["drivetrain"],
        "segment":                  d["segment"],
        "car_body_type":            d["car_body_type"],
    }
    return pd.DataFrame([row])[ALL_FEATURES]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup; free on shutdown."""
    log.info("Loading pipeline …")
    _state["pipeline"] = joblib.load(PIPELINE_PATH)
    log.info(f"  Pipeline loaded from {PIPELINE_PATH}")

    if BOOTSTRAP_PATH.exists():
        log.info(f"Loading {BOOTSTRAP_PATH.name} …")
        _state["bootstrap_models"] = joblib.load(BOOTSTRAP_PATH)
        log.info(f"  {len(_state['bootstrap_models'])} bootstrap models loaded")
    else:
        log.warning(f"  Bootstrap models not found at {BOOTSTRAP_PATH} — intervals unavailable")

    # Build SHAP explainer once
    pipeline = _state["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    xgb_model = pipeline.named_steps["model"]
    ohe = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
    cat_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    _state["feature_names"] = NUMERIC_FEATURES + cat_names
    _state["explainer"] = shap.TreeExplainer(xgb_model)
    log.info("  SHAP TreeExplainer ready")

    yield

    _state.clear()
    log.info("Shutdown: state cleared")


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TechTrack EV Range Prediction API",
    description=(
        "Predicts the driving range (km) of an electric vehicle from its "
        "static manufacturer specifications, with bootstrap p10–p90 confidence intervals."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Liveness / readiness probe."""
    return HealthResponse(
        status="ok",
        model_loaded=_state.get("pipeline") is not None,
        bootstrap_models_loaded=_state.get("bootstrap_models") is not None,
        pipeline_path=str(PIPELINE_PATH),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(ev: EVInput):
    """
    Predict EV driving range (km) with p10–p90 confidence interval.

    - Input: 21 EV specification fields (see schema)
    - Output: point prediction + bootstrap confidence interval
    """
    if _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        X = _input_to_df(ev)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature construction failed: {e}")

    try:
        result = predict_interval(
            X,
            alpha=0.1,
            pipeline=_state["pipeline"],
            bootstrap_models=_state["bootstrap_models"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        prediction=result["prediction"],
        p_lower=result["p_lower"],
        p_upper=result["p_upper"],
        interval_pct=result["interval_pct"],
        units="km",
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Inference"])
def explain(ev: EVInput):
    """
    Return SHAP values for a given EV specification (for Streamlit UI).
    """
    if _state["pipeline"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if _state["explainer"] is None:
        raise HTTPException(status_code=503, detail="Explainer not initialized")

    try:
        X = _input_to_df(ev)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature construction failed: {e}")

    pipeline = _state["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    explainer = _state["explainer"]
    feature_names = _state["feature_names"]

    X_t = preprocessor.transform(X)
    sv = explainer(X_t)

    point_pred = float(pipeline.predict(X)[0])

    return ExplainResponse(
        feature_names=feature_names,
        shap_values=[round(float(v), 4) for v in sv.values[0]],
        base_value=round(float(sv.base_values[0]), 4),
        prediction=round(point_pred, 2),
    )
