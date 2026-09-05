# tests/test_features.py
"""
Unit tests for every function in src/features.py.
All tests use fixed, deterministic inputs so results are reproducible.
"""

import pytest
import numpy as np
import pandas as pd

from src.features import (
    compute_footprint_m2,
    compute_volume_proxy_m3,
    compute_battery_per_footprint,
    compute_torque_per_battery,
    compute_battery_per_volume,
    coerce_cargo_volume,
    engineer_features,
    select_model_columns,
    add_range_bucket,
    flag_sparse_brands,
    ALL_FEATURES,
    TARGET,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_series():
    """Reusable numeric Series for dimension tests."""
    return pd.Series([4000.0, 3000.0, 5000.0])


@pytest.fixture
def minimal_raw_df():
    """Minimal raw DataFrame mimicking ev_dataset.xls structure."""
    return pd.DataFrame({
        "brand": ["Tesla", "Tesla", "Audi", "BMW"],
        "model": ["Model 3", "Model Y", "e-tron", "i4"],
        "top_speed_kmh": [225, 217, 200, 215],
        "battery_capacity_kWh": [75.0, 82.0, 95.0, 83.9],
        "battery_type": ["Lithium-ion"] * 4,
        "number_of_cells": [4416.0, 4416.0, None, 288.0],
        "torque_nm": [450.0, 500.0, 664.0, 430.0],
        "efficiency_wh_per_km": [150, 155, 230, 165],
        "range_km": [500, 500, 374, 590],
        "acceleration_0_100_s": [4.4, 5.0, 5.7, 5.1],
        "fast_charging_power_kw_dc": [250.0, 250.0, 150.0, 205.0],
        "fast_charge_port": ["CCS", "CCS", "CCS", "CCS"],
        "towing_capacity_kg": [0.0, 1600.0, 1800.0, 0.0],
        "cargo_volume_l": ["425", 854, "660", 470],  # mixed types on purpose
        "seats": [5, 5, 5, 5],
        "drivetrain": ["RWD", "AWD", "AWD", "RWD"],
        "segment": ["D - Large", "D - Large", "D - Large", "D - Large"],
        "length_mm": [4694, 4751, 4901, 4783],
        "width_mm": [1849, 1921, 1935, 1852],
        "height_mm": [1443, 1624, 1616, 1448],
        "car_body_type": ["Sedan", "SUV", "SUV", "Sedan"],
        "source_url": ["https://example.com"] * 4,
    })


# ── compute_footprint_m2 ──────────────────────────────────────────────────────

class TestComputeFootprintM2:
    def test_known_value(self):
        length_s = pd.Series([4000.0])
        w = pd.Series([2000.0])
        result = compute_footprint_m2(length_s, w)
        assert pytest.approx(result.iloc[0], rel=1e-6) == 8.0  # 4000*2000/1e6

    def test_series_length_preserved(self, sample_series):
        result = compute_footprint_m2(sample_series, sample_series)
        assert len(result) == len(sample_series)

    def test_all_positive(self, sample_series):
        result = compute_footprint_m2(sample_series, sample_series)
        assert (result > 0).all()

    def test_zero_dimension_gives_zero(self):
        length_s = pd.Series([0.0])
        w = pd.Series([2000.0])
        assert compute_footprint_m2(length_s, w).iloc[0] == 0.0

    def test_realistic_range(self):
        """Real EVs: length ~3500–5500 mm, width ~1600–2100 mm → footprint ~5.6–11.6 m²"""
        length_s = pd.Series([3500.0, 4500.0, 5500.0])
        w = pd.Series([1600.0, 1850.0, 2100.0])
        result = compute_footprint_m2(length_s, w)
        assert (result >= 5.0).all()
        assert (result <= 12.0).all()


# ── compute_volume_proxy_m3 ───────────────────────────────────────────────────

class TestComputeVolumeProxyM3:
    def test_known_value(self):
        length_s = pd.Series([4000.0])
        w = pd.Series([2000.0])
        h = pd.Series([1500.0])
        result = compute_volume_proxy_m3(length_s, w, h)
        assert pytest.approx(result.iloc[0], rel=1e-6) == 12.0  # 4*2*1.5 m³

    def test_series_length_preserved(self, sample_series):
        result = compute_volume_proxy_m3(sample_series, sample_series, sample_series)
        assert len(result) == len(sample_series)

    def test_all_positive(self, sample_series):
        assert (compute_volume_proxy_m3(sample_series, sample_series, sample_series) > 0).all()


# ── compute_battery_per_footprint ────────────────────────────────────────────

class TestComputeBatteryPerFootprint:
    def test_known_value(self):
        batt = pd.Series([100.0])
        fp = pd.Series([10.0])
        assert pytest.approx(compute_battery_per_footprint(batt, fp).iloc[0]) == 10.0

    def test_proportional_to_battery(self):
        batt = pd.Series([50.0, 100.0])
        fp = pd.Series([8.0, 8.0])
        result = compute_battery_per_footprint(batt, fp)
        assert result.iloc[1] == pytest.approx(result.iloc[0] * 2)

    def test_inversely_proportional_to_footprint(self):
        batt = pd.Series([80.0, 80.0])
        fp = pd.Series([8.0, 16.0])
        result = compute_battery_per_footprint(batt, fp)
        assert result.iloc[0] == pytest.approx(result.iloc[1] * 2)


# ── compute_torque_per_battery ───────────────────────────────────────────────

class TestComputeTorquePerBattery:
    def test_known_value(self):
        torque = pd.Series([400.0])
        batt = pd.Series([80.0])
        assert pytest.approx(compute_torque_per_battery(torque, batt).iloc[0]) == 5.0

    def test_zero_torque_gives_zero(self):
        torque = pd.Series([0.0])
        batt = pd.Series([80.0])
        assert compute_torque_per_battery(torque, batt).iloc[0] == 0.0


# ── compute_battery_per_volume ───────────────────────────────────────────────

class TestComputeBatteryPerVolume:
    def test_known_value(self):
        batt = pd.Series([120.0])
        vol = pd.Series([12.0])
        assert pytest.approx(compute_battery_per_volume(batt, vol).iloc[0]) == 10.0


# ── coerce_cargo_volume ──────────────────────────────────────────────────────

class TestCoerceCargoVolume:
    def test_numeric_string_converted(self):
        s = pd.Series(["425", "854", "660"])
        result = coerce_cargo_volume(s)
        # pd.to_numeric returns int64 for clean integer strings, float64 when NaN present
        assert np.issubdtype(result.dtype, np.number)
        assert result.iloc[0] == 425

    def test_non_numeric_becomes_nan(self):
        s = pd.Series(["425", "n/a", "unknown"])
        result = coerce_cargo_volume(s)
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

    def test_numeric_values_unchanged(self):
        s = pd.Series([500.0, 700.0])
        result = coerce_cargo_volume(s)
        assert result.iloc[0] == 500.0

    def test_mixed_int_and_string(self):
        s = pd.Series([400, "600", None, "bad"])
        result = coerce_cargo_volume(s)
        assert result.iloc[0] == 400.0
        assert result.iloc[1] == 600.0
        assert pd.isna(result.iloc[2])
        assert pd.isna(result.iloc[3])


# ── engineer_features ────────────────────────────────────────────────────────

class TestEngineerFeatures:
    def test_output_contains_engineered_columns(self, minimal_raw_df):
        out = engineer_features(minimal_raw_df)
        for col in ["footprint_m2", "volume_proxy_m3", "battery_per_footprint",
                    "torque_per_battery", "battery_per_volume"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_original_columns_preserved(self, minimal_raw_df):
        out = engineer_features(minimal_raw_df)
        for col in minimal_raw_df.columns:
            assert col in out.columns

    def test_cargo_volume_coerced(self, minimal_raw_df):
        out = engineer_features(minimal_raw_df)
        # coerce_cargo_volume returns numeric dtype (int or float depending on content)
        assert np.issubdtype(out["cargo_volume_l"].dtype, np.number)

    def test_footprint_values_reasonable(self, minimal_raw_df):
        out = engineer_features(minimal_raw_df)
        assert (out["footprint_m2"] > 5).all()
        assert (out["footprint_m2"] < 15).all()

    def test_no_rows_dropped(self, minimal_raw_df):
        out = engineer_features(minimal_raw_df)
        assert len(out) == len(minimal_raw_df)


# ── select_model_columns ──────────────────────────────────────────────────────

class TestSelectModelColumns:
    def test_returns_correct_feature_count(self, minimal_raw_df):
        df = engineer_features(minimal_raw_df)
        X, y = select_model_columns(df)
        assert X.shape[1] == len(ALL_FEATURES)

    def test_returns_correct_target(self, minimal_raw_df):
        df = engineer_features(minimal_raw_df)
        X, y = select_model_columns(df)
        assert y.name == TARGET
        assert list(y) == list(minimal_raw_df[TARGET])

    def test_no_leakage_columns(self, minimal_raw_df):
        df = engineer_features(minimal_raw_df)
        X, _ = select_model_columns(df)
        leakage = ["efficiency_wh_per_km", "brand", "model", "source_url", "battery_type"]
        for col in leakage:
            assert col not in X.columns, f"Leakage column found: {col}"


# ── add_range_bucket ─────────────────────────────────────────────────────────

class TestAddRangeBucket:
    def test_buckets_assigned_correctly(self):
        df = pd.DataFrame({"range_km": [200, 400, 600]})
        out = add_range_bucket(df)
        buckets = out["range_bucket"].astype(str).tolist()
        assert "short" in buckets[0]
        assert "medium" in buckets[1]
        assert "long" in buckets[2]

    def test_no_nulls_in_bucket_for_valid_range(self):
        df = pd.DataFrame({"range_km": [150, 350, 700]})
        out = add_range_bucket(df)
        assert out["range_bucket"].notna().all()

    def test_boundary_300_goes_to_short(self):
        """With right=True, the bin (0, 300] includes 300 → short bucket."""
        df = pd.DataFrame({"range_km": [300]})
        out = add_range_bucket(df)
        assert "short" in str(out["range_bucket"].iloc[0])

    def test_boundary_301_goes_to_medium(self):
        df = pd.DataFrame({"range_km": [301]})
        out = add_range_bucket(df)
        assert "medium" in str(out["range_bucket"].iloc[0])


# ── flag_sparse_brands ───────────────────────────────────────────────────────

class TestFlagSparseBrands:
    def test_detects_sparse_brand(self):
        df = pd.DataFrame({
            "brand": ["Tesla"] * 10 + ["RareBrand"] * 2 + ["Audi"] * 6
        })
        sparse = flag_sparse_brands(df, threshold=5)
        assert "RareBrand" in sparse.index

    def test_non_sparse_brand_not_flagged(self):
        df = pd.DataFrame({"brand": ["Tesla"] * 10 + ["Audi"] * 6})
        sparse = flag_sparse_brands(df, threshold=5)
        assert "Tesla" not in sparse.index
        assert "Audi" not in sparse.index

    def test_empty_result_when_all_sufficient(self):
        df = pd.DataFrame({"brand": ["A"] * 10 + ["B"] * 8})
        sparse = flag_sparse_brands(df, threshold=5)
        assert len(sparse) == 0

    def test_custom_threshold(self):
        df = pd.DataFrame({"brand": ["X"] * 8 + ["Y"] * 3})
        sparse = flag_sparse_brands(df, threshold=10)
        assert "X" in sparse.index
        assert "Y" in sparse.index
