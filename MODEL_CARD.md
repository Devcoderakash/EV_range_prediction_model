# Model Card — TechTrack EV Range Prediction

**Model version:** 2.0.0  
**Last updated:** September 2026  
**Framework:** scikit-learn Pipeline · XGBoost Regressor  
**Language:** Python 3  
**Artifact:** `model/ev_range_pipeline.joblib`

---

## 1. Intended Use

### Primary Use Case
Predict the **manufacturer-rated driving range (km)** of an electric vehicle from its published static specifications, **without** requiring real-world telemetry.

**Intended users:** EV researchers, fleet procurement teams, automotive analysts, and enthusiasts comparing published specs.

### In-Scope
- Comparing expected range across EV models based on published manufacturer data
- Feature-level sensitivity analysis (which specs most influence range)
- Educational demonstration of ML-based regression on tabular automotive data

### Out-of-Scope
- Real-time range prediction during a trip
- Predictions for EVs with specs far outside the training distribution (e.g., future solid-state 400 kWh packs)
- Warranty, safety, or insurance decisions
- Predicting range under specific real-world conditions (see Limitations)

---

## 2. Training Data

| Property | Value |
|---|---|
| Source | `notebook/ev_dataset.xls` (scraped from ev-database.org) |
| Size | 478 EV models |
| Unique brands | 59 |
| Unique models | 477 |
| Coverage | Primarily European-market EVs, 2015–2024 |
| Target | `range_km` — manufacturer-rated range (WLTP or equivalent) |
| Split | 70% train / 15% validation / 15% held-out test |
| Random seed | 42 (fixed for reproducibility) |

### Features Used (21)
17 numeric + 4 categorical. See `README.md` for the full feature list.

### Excluded Due to Data Leakage
`efficiency_wh_per_km` — algebraically related to `range_km` and `battery_capacity_kWh`.

---

## 3. Model Architecture

```
Pipeline:
  ColumnTransformer:
    numeric  → SimpleImputer(median)
    categorical → SimpleImputer(most_frequent) → OneHotEncoder
  XGBRegressor:
    objective: reg:squarederror
    n_estimators: [tuned via RandomizedSearchCV]
    max_depth: [tuned]
    learning_rate: [tuned]
    subsample: [tuned]
    colsample_bytree: [tuned]
    random_state: 42
```

---

## 4. Performance

### Validation Method
Nested cross-validation: 5-fold outer loop (honest generalization estimate) with 30-iteration `RandomizedSearchCV` inner loop.

> The outer-loop score is reported as the honest generalization estimate. The single-split test set score is a secondary check.

### Aggregate Performance (held-out test set, 15% of data)

| Metric | Score |
|---|---|
| **MAE** | ~10–12 km |
| **RMSE** | ~13–15 km |
| **R²** | ~0.98 |

### Performance by Range Bucket

| Bucket | MAE (km) | Notes |
|---|---|---|
| Short (<300 km) | Higher | Fewer examples, specs less differentiated |
| Medium (300–500 km) | Lowest | Densest part of training distribution |
| Long (>500 km) | Moderate | Fewer high-range models in dataset |

### Performance by Brand
See `reports/figures/mae_by_brand.png` (generated at training time).

> [!WARNING]
> Brands with fewer than **5 samples** in the dataset (flagged during training) may have inflated or deflated apparent accuracy due to small sample size. Predictions for these brands should be interpreted with extra caution.

---

## 5. Limitations

### What the Model Does NOT Account For

| Factor | Impact | Notes |
|---|---|---|
| **Driving speed** | High | Highway driving at 130 km/h vs city driving at 50 km/h can change range by 30–50% |
| **Ambient temperature** | High | Cold weather (below 0°C) can reduce range by 20–40% |
| **HVAC load** | Medium | Heating/cooling draws significantly from battery |
| **Terrain** | Medium | Hilly routes reduce range; regenerative braking partially compensates |
| **Battery degradation (SoH)** | High | A 3-year-old battery may deliver only 85–90% of original range |
| **Traffic and stop-start** | Low-Medium | Affects energy consumption profile |
| **Driver behavior** | High | Aggressive acceleration and high speeds substantially reduce range |
| **Tyre pressure** | Low | Underinflation increases rolling resistance |

### Data-Level Limitations

- **Manufacturer figures** — `range_km` is the rated (WLTP/EPA) range, which consistently overestimates real-world range by 10–20%.
- **Data vintage** — EVs released after the dataset cutoff (2024+) are not in the training distribution.
- **European-market bias** — Non-European market variants (different battery packs, software configs) may differ.
- **Sparse brands** — Brands with < 5 models in the dataset have unreliable predictions.

### Extrapolation Risk
Predictions for EVs outside the training distribution (e.g., battery > 150 kWh, range > 800 km) may be unreliable. The model has not seen these values during training.

---

## 6. Known Failure Modes

| Failure Mode | Description |
|---|---|
| **Small city EV over-prediction** | Very small EVs with unusual battery chemistry may be predicted higher than actual |
| **Hypercars / performance EVs** | High torque, low-efficiency configurations may not have enough training examples |
| **Trucks and large pickups** | Very few examples in training set; predictions unreliable |
| **Bidirectional charging** (V2G/V2H) | Not encoded in features; may affect efficiency estimates |

---

## 7. Confidence Intervals

The `/predict` API endpoint returns **bootstrap p10–p90 intervals** based on 500 resampled models trained on the same training set.

These intervals capture **model uncertainty** from training variance, not aleatoric uncertainty from real-world driving conditions. The true uncertainty (including real-world factors) is substantially wider.

---

## 8. Ethical Considerations

- This model should **not** be used for warranty or safety-critical decisions.
- Range anxiety is a real concern for EV adoption; predictions should always be accompanied by the confidence interval and a reminder about real-world factors.
- The dataset reflects market EVs that are predominantly mid-to-high price segment; low-cost EV segments may be underrepresented.

---

## 9. Intended Next Steps (Future Work)

- Integrate real-world owner-reported range data (e.g., Spritmonitor, EV owner surveys) to complement manufacturer figures
- Add temperature-adjusted predictions using climate data
- Periodic retraining as new EV models enter the market
- Formal quantile regression for more calibrated intervals

---

## 10. Citation / Attribution

Dataset source: [ev-database.org](https://ev-database.org)  
Project: TechTrack EV Range Prediction Challenge  
