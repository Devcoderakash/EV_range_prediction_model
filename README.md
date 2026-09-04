# ⚡ TechTrack EV Range Prediction

> Predicting electric vehicle driving range from static specifications using machine learning regression.

---

## 📌 Project Overview

This project tackles the **TechTrack EV Range Prediction** challenge: given a set of static electric vehicle specifications, predict the vehicle's driving range in kilometres (`range_km`).

The prediction is **specification-based** — it uses only manufacturer-published attributes such as battery capacity, dimensions, motor torque, and charging configuration. It does **not** incorporate real-time telemetry such as traffic, weather, HVAC load, current State of Charge (SoC), or State of Health (SoH).

---

## 🎯 Problem Statement

**Task:** Regression — predict `range_km` from available EV specifications.

Given a dataset of EV models characterised by their published hardware and design specifications, the goal is to train a model that generalises well to unseen vehicles and produces accurate range estimates in kilometres.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source file | `ev_dataset.xls` |
| Total rows | **478** EV models |
| Total columns | 22 |
| Unique brands | **59** |
| Granularity | One row per EV model |
| Target variable | `range_km` |
| Missing target values | 0 |

**Input specifications used by the model (21 features after engineering):**

| Feature | Type | Description |
|---|---|---|
| `top_speed_kmh` | Numeric | Maximum speed (km/h) |
| `battery_capacity_kWh` | Numeric | Usable battery capacity (kWh) |
| `number_of_cells` | Numeric | Total number of battery cells |
| `torque_nm` | Numeric | Peak motor torque (Nm) |
| `acceleration_0_100_s` | Numeric | 0–100 km/h time (seconds) |
| `fast_charging_power_kw_dc` | Numeric | DC fast-charge power (kW) |
| `towing_capacity_kg` | Numeric | Maximum towing capacity (kg) |
| `cargo_volume_l` | Numeric | Cargo/boot volume (litres) |
| `seats` | Numeric | Seating capacity |
| `length_mm` | Numeric | Vehicle length (mm) |
| `width_mm` | Numeric | Vehicle width (mm) |
| `height_mm` | Numeric | Vehicle height (mm) |
| `fast_charge_port` | Categorical | DC charging standard (`CCS`, `CHAdeMO`) |
| `drivetrain` | Categorical | `AWD`, `FWD`, or `RWD` |
| `segment` | Categorical | EU vehicle segment (e.g. `D - Large`) |
| `car_body_type` | Categorical | Body style (e.g. `SUV`, `Sedan`) |
| `footprint_m2` | Engineered | Vehicle ground footprint (m²) |
| `volume_proxy_m3` | Engineered | Vehicle size proxy (m³) |
| `battery_per_footprint` | Engineered | Battery capacity relative to footprint |
| `torque_per_battery` | Engineered | Motor torque relative to battery capacity |
| `battery_per_volume` | Engineered | Battery capacity relative to volume proxy |

---

## 🧹 Data Cleaning

The following cleaning steps were performed on the raw dataset:

- **Duplicate check** — no duplicate rows were found (0 duplicates).
- **Cargo volume coercion** — `cargo_volume_l` contained mixed-type entries; non-numeric values were coerced to `NaN` (4 affected rows).
- **Missing-value audit** — columns with missing values identified:

  | Column | Missing Count |
  |---|---|
  | `number_of_cells` | 202 |
  | `towing_capacity_kg` | 26 |
  | `torque_nm` | 7 |
  | `cargo_volume_l` | 4 |
  | `fast_charging_power_kw_dc` | 1 |

- **No rows dropped** — missing values are handled downstream by the pipeline's imputer steps rather than by row deletion.
- **Columns excluded from modelling:** `efficiency_wh_per_km`, `brand`, `model`, `source_url`, `battery_type`, and the target `range_km` itself.

---

## 📈 Exploratory Data Analysis

EDA performed in the notebook covers:

- **Target distribution** — distribution and summary statistics of `range_km`.
- **Numerical feature relationships** — scatter plots and correlation analysis between numeric predictors and `range_km`.
- **Categorical analysis** — value counts and unique-value inspection for `brand`, `model`, `fast_charge_port`, `drivetrain`, `segment`, and `car_body_type` (59 unique brands, 477 unique models).
- **Missing-value analysis** — bar charts and summary tables for columns with missing data.
- **Engineered feature relationships** — scatter plots of `battery_per_footprint`, `battery_per_volume`, and `torque_per_battery` against `range_km`.
- **Correlation heatmap** — Pearson correlation between numeric features and the target.

---

## 🧠 Feature Engineering

Five domain-inspired features were derived from the raw specifications:

| Feature | Formula | Rationale |
|---|---|---|
| `footprint_m2` | `(length_mm × width_mm) / 1,000,000` | Approximates vehicle ground footprint |
| `volume_proxy_m3` | `(length_mm × width_mm × height_mm) / 1,000,000,000` | Rough proxy for vehicle size/mass |
| `battery_per_footprint` | `battery_capacity_kWh / footprint_m2` | Battery density relative to vehicle footprint |
| `torque_per_battery` | `torque_nm / battery_capacity_kWh` | Motor effort relative to battery capacity |
| `battery_per_volume` | `battery_capacity_kWh / volume_proxy_m3` | Battery density relative to vehicle volume |

**Ablation study result** (confirmed from notebook output):

| Feature Set | MAE (km) | RMSE (km) | R² |
|---|---|---|---|
| Raw Features only | 11.39 | 15.53 | 0.98 |
| + Engineered Features | 10.54 | 13.67 | 0.98 |

> **Note on `efficiency_wh_per_km`:** This column was explicitly excluded from the predictive model. Because `range_km ≈ battery_capacity_kWh × 1000 / efficiency_wh_per_km`, including it would introduce **target leakage** — the model would be learning a near-perfect algebraic identity rather than genuine predictive relationships.

---

## 🤖 Machine Learning Models

### Baseline

A **DummyRegressor** (mean prediction) was used to establish a naive performance floor.

### Linear Model

| Model | Notes |
|---|---|
| Ridge Regression | `alpha=10.0`; serves as an interpretable linear baseline |

**Ridge test performance:** MAE 17.11 km · RMSE 21.46 km · R² 0.9565

### Ensemble Comparison — 5-Fold Cross Validation

| Model | CV MAE (km) | CV RMSE (km) | CV R² |
|---|---|---|---|
| XGBoost | 13.55 | 18.98 | 0.97 |
| Extra Trees | 15.37 | 21.41 | 0.96 |
| Gradient Boosting | 16.62 | 22.14 | 0.95 |
| Random Forest | 17.12 | 23.59 | 0.95 |

**XGBoost was selected** as the best-performing model based on CV RMSE.

### Hyperparameter Tuning

`RandomizedSearchCV` (30 iterations, 5-fold CV, scoring: RMSE) was applied to XGBoost.

**Best parameters found:**

```
n_estimators=700, max_depth=3, learning_rate=0.08,
subsample=0.7, colsample_bytree=0.7, min_child_weight=2
```

---

## 📏 Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average absolute deviation in km |
| **RMSE** | Root Mean Squared Error — penalises large errors more heavily |
| **R²** | Coefficient of determination — proportion of variance explained |

### Final Model Performance on Held-Out Test Set (20%)

| Metric | Value |
|---|---|
| **MAE** | **10.54 km** |
| **RMSE** | **13.67 km** |
| **R²** | **0.9823** |

*Verified: loaded model from `model/ev_range_pipeline.joblib` reproduces identical RMSE (13.6725).*

---

## 🔄 Reproducible Pipeline

The trained preprocessing and model are packaged together in a single **scikit-learn Pipeline** saved via Joblib:

```
model/ev_range_pipeline.joblib
```

The pipeline contains:
- **ColumnTransformer** with:
  - `numeric` branch: `SimpleImputer` (median) on 17 numeric features
  - `categorical` branch: `SimpleImputer` (most frequent) → `OneHotEncoder` on 4 categorical features
- **XGBRegressor** with tuned hyperparameters

Loading and predicting:

```python
import joblib
import pandas as pd

model = joblib.load("model/ev_range_pipeline.joblib")
prediction = model.predict(input_df)  # input_df must include all 21 features
```

---

## 🖥️ Interactive Demo

A **Streamlit** application (`app/app.py`) is planned as part of this project. When complete, it will allow users to enter EV specifications and receive a predicted driving range instantly.

> `app/app.py` is a target deliverable. `streamlit` is included in `requirements.txt`.

To run the application once available:

```bash
streamlit run app/app.py
```

---

## 📁 Project Structure

```
TechTrack_EV_Range/
├── notebook/
│   └── EV_Range_Prediction.ipynb   # Full ML workflow
├── model/
│   └── ev_range_pipeline.joblib    # Trained pipeline (preprocessing + XGBoost)
├── app/
│   └── app.py                      # Streamlit interactive demo (planned)
├── report/
│   └── technical_report.pdf        # Technical report
├── requirements.txt                # Python dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd TechTrack_EV_Range

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the environment
# macOS / Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app/app.py
```

---

## 📓 Run the Notebook

```bash
jupyter notebook notebook/EV_Range_Prediction.ipynb
```

Run all cells top-to-bottom. The notebook covers:

1. Data loading and cleaning
2. Exploratory data analysis
3. Feature engineering
4. Train/test split (80/20)
5. Preprocessing pipeline construction
6. Baseline and linear modelling
7. Ensemble model cross-validation comparison
8. XGBoost hyperparameter tuning (RandomizedSearchCV)
9. Final model evaluation on held-out test set
10. Feature importance analysis
11. Model serialisation and verification

---

## 📄 Technical Report

```
report/technical_report.pdf
```

The technical report is included in the `report/` directory.

---

## ⚠️ Limitations

- **Static specifications only** — real-world range is influenced by factors not captured in manufacturer specs, including:
  - Driving style and speed profile
  - Ambient temperature and weather conditions
  - HVAC (heating/cooling) load
  - Terrain and elevation changes
  - Battery degradation (State of Health)
  - Traffic conditions and stop-start driving
- **Manufacturer data accuracy** — the model inherits any inaccuracies in source specification data.
- **Extrapolation risk** — predictions for EVs outside the training distribution may be unreliable.
- **No real-time adaptation** — the model does not update based on live vehicle data or telematics.

---

## 🚀 Future Improvements

The following are suggestions for **future work** and do not reflect existing functionality:

- Build and deploy `app/app.py` as an interactive Streamlit range predictor
- Integrate real-world owner-reported range data to complement manufacturer figures
- Add SHAP explainability to the app
- Explore range-by-temperature modelling using climate data
- Add confidence intervals via quantile regression or conformal prediction
- Periodic retraining as new EV models enter the market

---

## 👨‍💻 Project

**TechTrack EV Range Prediction**  
Built as part of the TechTrack machine learning challenge.

> Model: XGBoost Regressor · Framework: scikit-learn Pipeline · Language: Python 3
