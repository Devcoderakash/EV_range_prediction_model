# app/schemas.py
"""
Pydantic models for FastAPI request / response validation.
"""

from typing import Literal

from pydantic import BaseModel, Field


class EVInput(BaseModel):
    """Input specification for one electric vehicle."""

    # ── Numeric features ──────────────────────────────────────────────────
    top_speed_kmh: float = Field(..., ge=50, le=500, description="Maximum speed (km/h)")
    battery_capacity_kWh: float = Field(..., ge=5, le=250, description="Usable battery capacity (kWh)")
    number_of_cells: float | None = Field(None, ge=1, le=10000, description="Total battery cells")
    torque_nm: float | None = Field(None, ge=0, le=3000, description="Peak motor torque (Nm)")
    acceleration_0_100_s: float = Field(..., ge=1.0, le=30.0, description="0–100 km/h time (s)")
    fast_charging_power_kw_dc: float | None = Field(None, ge=0, le=1000, description="DC fast-charge power (kW)")
    towing_capacity_kg: float | None = Field(None, ge=0, le=5000, description="Maximum towing capacity (kg)")
    cargo_volume_l: float | None = Field(None, ge=0, le=5000, description="Cargo/boot volume (litres)")
    seats: int = Field(..., ge=1, le=9, description="Number of seats")
    length_mm: int = Field(..., ge=2000, le=7000, description="Vehicle length (mm)")
    width_mm: int = Field(..., ge=1400, le=3000, description="Vehicle width (mm)")
    height_mm: int = Field(..., ge=1000, le=3000, description="Vehicle height (mm)")

    # ── Categorical features ──────────────────────────────────────────────
    fast_charge_port: Literal["CCS", "CHAdeMO", "Type 2", "Tesla", "None"] = Field(
        ..., description="DC charging standard"
    )
    drivetrain: Literal["AWD", "FWD", "RWD"] = Field(..., description="Drivetrain type")
    segment: str = Field(..., description="EU vehicle segment (e.g. 'D - Large')")
    car_body_type: str = Field(..., description="Body style (e.g. 'SUV', 'Sedan')")

    model_config = {"json_schema_extra": {
        "example": {
            "top_speed_kmh": 180,
            "battery_capacity_kWh": 77.4,
            "number_of_cells": 288,
            "torque_nm": 350.0,
            "acceleration_0_100_s": 7.4,
            "fast_charging_power_kw_dc": 100.0,
            "towing_capacity_kg": 0.0,
            "cargo_volume_l": 385,
            "seats": 5,
            "length_mm": 4180,
            "width_mm": 1800,
            "height_mm": 1445,
            "fast_charge_port": "CCS",
            "drivetrain": "RWD",
            "segment": "D - Large",
            "car_body_type": "Hatchback",
        }
    }}


class PredictionResponse(BaseModel):
    """Response from /predict endpoint."""
    prediction: float = Field(..., description="Predicted range in km")
    p_lower: float = Field(..., description="Lower bound of confidence interval (km)")
    p_upper: float = Field(..., description="Upper bound of confidence interval (km)")
    interval_pct: str = Field(..., description="Interval description, e.g. 'p10–p90'")
    units: str = "km"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    bootstrap_models_loaded: bool
    pipeline_path: str


class ExplainResponse(BaseModel):
    """Response from /explain endpoint — SHAP values for one input."""
    feature_names: list[str]
    shap_values: list[float]
    base_value: float
    prediction: float
