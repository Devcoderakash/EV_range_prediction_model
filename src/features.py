# src/features.py
"""
Pure feature-engineering functions for the EV range prediction project.

All functions here are stateless and side-effect free so they can be
independently unit-tested with fixed inputs/outputs.
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "top_speed_kmh",
    "battery_capacity_kWh",
    "number_of_cells",
    "torque_nm",
    "acceleration_0_100_s",
    "fast_charging_power_kw_dc",
    "towing_capacity_kg",
    "cargo_volume_l",
    "seats",
    "length_mm",
    "width_mm",
    "height_mm",
    "footprint_m2",
    "volume_proxy_m3",
    "battery_per_footprint",
    "torque_per_battery",
    "battery_per_volume",
]

CATEGORICAL_FEATURES = [
    "fast_charge_port",
    "drivetrain",
    "segment",
    "car_body_type",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET = "range_km"

# Columns to drop before modelling (leakage or non-informative)
DROP_COLS = [
    "efficiency_wh_per_km",  # target leakage: range ≈ battery/efficiency
    "brand",
    "model",
    "source_url",
    "battery_type",
]

# Range buckets for error breakdown
RANGE_BUCKET_BINS = [0, 300, 500, float("inf")]
RANGE_BUCKET_LABELS = ["short (<300 km)", "medium (300–500 km)", "long (>500 km)"]

# Minimum samples per brand before flagging as sparse
SPARSE_BRAND_THRESHOLD = 5


# ---------------------------------------------------------------------------
# Individual engineered-feature functions (each testable in isolation)
# ---------------------------------------------------------------------------

def compute_footprint_m2(length_mm: pd.Series, width_mm: pd.Series) -> pd.Series:
    """Ground footprint of the vehicle in m²."""
    return (length_mm * width_mm) / 1_000_000


def compute_volume_proxy_m3(
    length_mm: pd.Series, width_mm: pd.Series, height_mm: pd.Series
) -> pd.Series:
    """Rough proxy for vehicle size/mass in m³."""
    return (length_mm * width_mm * height_mm) / 1_000_000_000


def compute_battery_per_footprint(
    battery_capacity_kWh: pd.Series, footprint_m2: pd.Series
) -> pd.Series:
    """Battery energy density relative to vehicle footprint (kWh / m²)."""
    return battery_capacity_kWh / footprint_m2


def compute_torque_per_battery(
    torque_nm: pd.Series, battery_capacity_kWh: pd.Series
) -> pd.Series:
    """Motor torque relative to battery capacity (Nm / kWh)."""
    return torque_nm / battery_capacity_kWh


def compute_battery_per_volume(
    battery_capacity_kWh: pd.Series, volume_proxy_m3: pd.Series
) -> pd.Series:
    """Battery energy density relative to vehicle volume (kWh / m³)."""
    return battery_capacity_kWh / volume_proxy_m3


# ---------------------------------------------------------------------------
# Cargo volume coercion (handles mixed-type column in raw data)
# ---------------------------------------------------------------------------

def coerce_cargo_volume(series: pd.Series) -> pd.Series:
    """
    Coerce ``cargo_volume_l`` to numeric, setting non-numeric values to NaN.
    The raw XLS column contains mixed string/numeric entries.
    """
    return pd.to_numeric(series, errors="coerce")


# ---------------------------------------------------------------------------
# Main pipeline: raw DataFrame → feature-engineered DataFrame
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps to a raw EV dataset DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame loaded from ``ev_dataset.xls``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing ``ALL_FEATURES`` columns plus ``TARGET``.
        Metadata columns (brand, model, source_url, etc.) are **not** dropped
        here so callers can still use them for analysis; call
        ``select_model_columns`` to get only the modelling subset.
    """
    out = df.copy()

    # Coerce mixed-type cargo volume
    out["cargo_volume_l"] = coerce_cargo_volume(out["cargo_volume_l"])

    # Derived features
    out["footprint_m2"] = compute_footprint_m2(out["length_mm"], out["width_mm"])
    out["volume_proxy_m3"] = compute_volume_proxy_m3(
        out["length_mm"], out["width_mm"], out["height_mm"]
    )
    out["battery_per_footprint"] = compute_battery_per_footprint(
        out["battery_capacity_kWh"], out["footprint_m2"]
    )
    out["torque_per_battery"] = compute_torque_per_battery(
        out["torque_nm"], out["battery_capacity_kWh"]
    )
    out["battery_per_volume"] = compute_battery_per_volume(
        out["battery_capacity_kWh"], out["volume_proxy_m3"]
    )

    return out


def select_model_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return (X, y) ready for the sklearn pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Output of ``engineer_features``.

    Returns
    -------
    X : pd.DataFrame  — shape (n, 21)
    y : pd.Series     — target ``range_km``
    """
    X = df[ALL_FEATURES].copy()
    y = df[TARGET].copy()
    return X, y


# ---------------------------------------------------------------------------
# Bucket / analysis utilities
# ---------------------------------------------------------------------------

def add_range_bucket(df: pd.DataFrame, range_col: str = TARGET) -> pd.DataFrame:
    """Add a ``range_bucket`` column to *df* based on ``range_col``."""
    out = df.copy()
    out["range_bucket"] = pd.cut(
        out[range_col],
        bins=RANGE_BUCKET_BINS,
        labels=RANGE_BUCKET_LABELS,
        right=True,
    )
    return out


def flag_sparse_brands(
    df: pd.DataFrame, threshold: int = SPARSE_BRAND_THRESHOLD
) -> pd.Series:
    """
    Return a boolean Series (index = brand name) where True means the brand
    has fewer than *threshold* samples in *df*.
    """
    counts = df["brand"].value_counts()
    return counts[counts < threshold]
