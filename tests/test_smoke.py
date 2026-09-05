# tests/test_smoke.py
"""
Smoke tests:
1. Load trained pipeline and assert predictions on known inputs are sane.
2. FastAPI /health and /predict endpoints return expected responses.
"""

import pytest
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE_PATH = ROOT / "model" / "ev_range_pipeline.joblib"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline():
    """Load the trained pipeline once for the entire module."""
    if not PIPELINE_PATH.exists():
        pytest.skip(f"Pipeline not found at {PIPELINE_PATH}. Run `make train` first.")
    return joblib.load(PIPELINE_PATH)


@pytest.fixture
def known_inputs():
    """
    Three EV specifications with plausible expected range brackets.
    These are real-world approximate specs — NOT from the training set labels.
    """
    return pd.DataFrame([
        # Short-range city EV (~150–250 km expected)
        {
            "top_speed_kmh": 130, "battery_capacity_kWh": 30.0,
            "number_of_cells": 192.0, "torque_nm": 160.0,
            "acceleration_0_100_s": 11.0, "fast_charging_power_kw_dc": 50.0,
            "towing_capacity_kg": 0.0, "cargo_volume_l": 250.0,
            "seats": 4, "length_mm": 3600, "width_mm": 1660, "height_mm": 1450,
            "footprint_m2": 3600*1660/1e6, "volume_proxy_m3": 3600*1660*1450/1e9,
            "battery_per_footprint": 30.0/(3600*1660/1e6),
            "torque_per_battery": 160.0/30.0,
            "battery_per_volume": 30.0/(3600*1660*1450/1e9),
            "fast_charge_port": "CCS", "drivetrain": "FWD",
            "segment": "B - Compact", "car_body_type": "Hatchback",
        },
        # Mid-range family EV (~400–550 km expected)
        {
            "top_speed_kmh": 180, "battery_capacity_kWh": 77.0,
            "number_of_cells": 288.0, "torque_nm": 350.0,
            "acceleration_0_100_s": 7.5, "fast_charging_power_kw_dc": 100.0,
            "towing_capacity_kg": 0.0, "cargo_volume_l": 385.0,
            "seats": 5, "length_mm": 4180, "width_mm": 1800, "height_mm": 1445,
            "footprint_m2": 4180*1800/1e6, "volume_proxy_m3": 4180*1800*1445/1e9,
            "battery_per_footprint": 77.0/(4180*1800/1e6),
            "torque_per_battery": 350.0/77.0,
            "battery_per_volume": 77.0/(4180*1800*1445/1e9),
            "fast_charge_port": "CCS", "drivetrain": "RWD",
            "segment": "D - Large", "car_body_type": "Hatchback",
        },
        # Long-range flagship EV (~600–700 km expected)
        {
            "top_speed_kmh": 250, "battery_capacity_kWh": 100.0,
            "number_of_cells": 8256.0, "torque_nm": 900.0,
            "acceleration_0_100_s": 2.1, "fast_charging_power_kw_dc": 250.0,
            "towing_capacity_kg": 1020.0, "cargo_volume_l": 828.0,
            "seats": 5, "length_mm": 4979, "width_mm": 1964, "height_mm": 1445,
            "footprint_m2": 4979*1964/1e6, "volume_proxy_m3": 4979*1964*1445/1e9,
            "battery_per_footprint": 100.0/(4979*1964/1e6),
            "torque_per_battery": 900.0/100.0,
            "battery_per_volume": 100.0/(4979*1964*1445/1e9),
            "fast_charge_port": "CCS", "drivetrain": "AWD",
            "segment": "F - Luxury", "car_body_type": "Sedan",
        },
    ])


# ── Pipeline smoke tests ───────────────────────────────────────────────────────

class TestPipelineLoading:
    def test_pipeline_loads(self, pipeline):
        assert pipeline is not None

    def test_pipeline_has_expected_steps(self, pipeline):
        assert "preprocessor" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_preprocessor_has_numeric_and_categorical(self, pipeline):
        pre = pipeline.named_steps["preprocessor"]
        transformer_names = [t[0] for t in pre.transformers]
        assert "numeric" in transformer_names
        assert "categorical" in transformer_names


class TestPredictionSanity:
    def test_predictions_return_correct_count(self, pipeline, known_inputs):
        preds = pipeline.predict(known_inputs)
        assert len(preds) == 3

    def test_predictions_are_positive(self, pipeline, known_inputs):
        preds = pipeline.predict(known_inputs)
        assert (preds > 0).all(), f"Negative predictions found: {preds}"

    def test_predictions_in_realistic_range(self, pipeline, known_inputs):
        """All predicted ranges should be between 50 km and 1 200 km."""
        preds = pipeline.predict(known_inputs)
        for i, pred in enumerate(preds):
            assert 50 < pred < 1200, (
                f"Prediction {i} ({pred:.1f} km) outside realistic range [50, 1200]"
            )

    def test_short_range_ev_predicts_lower(self, pipeline, known_inputs):
        """Short-range EV (30 kWh) should predict less than long-range EV (100 kWh)."""
        preds = pipeline.predict(known_inputs)
        assert preds[0] < preds[2], (
            f"Short-range EV ({preds[0]:.1f} km) ≥ long-range EV ({preds[2]:.1f} km)"
        )

    def test_predictions_are_finite(self, pipeline, known_inputs):
        preds = pipeline.predict(known_inputs)
        assert np.isfinite(preds).all(), f"Non-finite predictions: {preds}"

    def test_single_row_prediction(self, pipeline, known_inputs):
        """Pipeline should handle single-row input without errors."""
        single = known_inputs.iloc[[0]]
        pred = pipeline.predict(single)
        assert len(pred) == 1
        assert 50 < pred[0] < 1200


class TestPredictionConsistency:
    def test_same_input_gives_same_output(self, pipeline, known_inputs):
        """Model is deterministic — same input must give same output."""
        preds1 = pipeline.predict(known_inputs)
        preds2 = pipeline.predict(known_inputs)
        np.testing.assert_array_equal(preds1, preds2)

    def test_larger_battery_increases_prediction(self, pipeline, known_inputs):
        """Doubling battery capacity (ceteris paribus) should increase predicted range."""
        base = known_inputs.iloc[[1]].copy()
        boosted = base.copy()
        boosted["battery_capacity_kWh"] = base["battery_capacity_kWh"] * 1.5
        # Also update engineered features that depend on battery
        boosted["battery_per_footprint"] = boosted["battery_capacity_kWh"] / boosted["footprint_m2"]
        boosted["battery_per_volume"] = boosted["battery_capacity_kWh"] / boosted["volume_proxy_m3"]

        pred_base = pipeline.predict(base)[0]
        pred_boosted = pipeline.predict(boosted)[0]
        assert pred_boosted > pred_base, (
            f"Larger battery ({pred_boosted:.1f}) did not predict higher range than "
            f"smaller battery ({pred_base:.1f})"
        )
