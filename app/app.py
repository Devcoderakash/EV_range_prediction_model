# app/app.py
"""
Streamlit demo UI — thin frontend for the FastAPI EV Range Prediction service.

Calls:
  POST http://localhost:8000/predict   → range + CI
  POST http://localhost:8000/explain   → SHAP values

Run with:
    streamlit run app/app.py
(The FastAPI service must be running on port 8000 first)
"""

import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ TechTrack EV Range Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.metric-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #1b2a3b 100%);
    border: 1px solid #2196F3;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(33,150,243,0.2);
}
.metric-card h1 { color: #42A5F5; font-size: 2.8rem; margin: 0; font-weight: 700; }
.metric-card p  { color: #90CAF9; margin: 4px 0 0; font-size: 0.9rem; }

.ci-bar-label { color: #B0BEC5; font-size: 0.82rem; text-align: center; margin-top: 4px; }

.sidebar-section {
    background: #0d1b2a;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    border-left: 3px solid #2196F3;
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar — input controls ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ EV Specifications")
    st.caption("Adjust specs below and click **Predict** to see the range estimate.")
    st.divider()

    st.markdown("### 🔋 Battery & Performance")
    battery = st.slider("Battery Capacity (kWh)", 10.0, 200.0, 77.0, step=0.5)
    top_speed = st.slider("Top Speed (km/h)", 80, 400, 180)
    torque = st.slider("Motor Torque (Nm)", 50, 2000, 350)
    accel = st.slider("0–100 km/h Time (s)", 1.0, 25.0, 7.5, step=0.1)
    cells = st.number_input("Number of Battery Cells", min_value=0, max_value=10000, value=288)
    fast_charge_kw = st.slider("DC Fast Charge Power (kW)", 0, 800, 100)

    st.markdown("### 📐 Dimensions")
    length = st.number_input("Length (mm)", 2500, 6000, 4180)
    width = st.number_input("Width (mm)", 1400, 2500, 1800)
    height = st.number_input("Height (mm)", 1000, 2500, 1445)

    st.markdown("### 🚗 Configuration")
    drivetrain = st.selectbox("Drivetrain", ["RWD", "AWD", "FWD"])
    fast_charge_port = st.selectbox("Fast Charge Port", ["CCS", "CHAdeMO", "Type 2", "Tesla", "None"])
    segment = st.selectbox("Segment", [
        "A - Mini", "B - Compact", "C - Medium", "D - Large",
        "E - Executive", "F - Luxury", "J - SUV", "M - Minivan", "S - Sport",
    ], index=3)
    body_type = st.selectbox("Body Type", [
        "Hatchback", "Sedan", "SUV", "Crossover", "MPV",
        "Pickup Truck", "Coupe", "Convertible", "Station Wagon",
    ])
    seats = st.slider("Seats", 2, 9, 5)
    towing = st.number_input("Towing Capacity (kg)", 0, 4000, 0)
    cargo = st.number_input("Cargo Volume (L)", 0, 4000, 385)

    predict_btn = st.button("⚡ Predict Range", type="primary", use_container_width=True)


# ── Main panel ─────────────────────────────────────────────────────────────────
st.title("⚡ TechTrack EV Range Predictor")
st.caption(
    "Predicts EV driving range from manufacturer specs using XGBoost + bootstrap confidence intervals. "
    "Powered by a FastAPI backend."
)

if predict_btn:
    # Build request payload
    footprint = (length * width) / 1_000_000
    vol_proxy = (length * width * height) / 1_000_000_000
    batt_fp = battery / footprint if footprint > 0 else 0
    torque_b = torque / battery if battery > 0 else 0
    batt_vol = battery / vol_proxy if vol_proxy > 0 else 0

    payload = {
        "top_speed_kmh":             top_speed,
        "battery_capacity_kWh":      battery,
        "number_of_cells":           float(cells) if cells > 0 else None,
        "torque_nm":                 float(torque),
        "acceleration_0_100_s":      accel,
        "fast_charging_power_kw_dc": float(fast_charge_kw),
        "towing_capacity_kg":        float(towing),
        "cargo_volume_l":            float(cargo),
        "seats":                     seats,
        "length_mm":                 length,
        "width_mm":                  width,
        "height_mm":                 height,
        "fast_charge_port":          fast_charge_port,
        "drivetrain":                drivetrain,
        "segment":                   segment,
        "car_body_type":             body_type,
    }

    col1, col2 = st.columns([1, 2])

    with col1:
        # ── Predict ──────────────────────────────────────────────────────────
        with st.spinner("Calling prediction API …"):
            try:
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
                r.raise_for_status()
                pred_data = r.json()
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to API at `localhost:8000`. Run `make api` first.")
                st.stop()
            except Exception as e:
                st.error(f"❌ API error: {e}")
                st.stop()

        pred = pred_data["prediction"]
        p_low = pred_data["p_lower"]
        p_hi = pred_data["p_upper"]
        ci = pred_data["interval_pct"]

        # ── Main metric card ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="metric-card">
            <h1>{pred:.0f} km</h1>
            <p>Predicted Range · {ci} interval: {p_low:.0f} – {p_hi:.0f} km</p>
        </div>
        """, unsafe_allow_html=True)

        # ── CI bar chart ───────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 1.4))
        fig.patch.set_facecolor("#0d1b2a")
        ax.set_facecolor("#0d1b2a")

        ax.barh([0], [p_hi - p_low], left=p_low, height=0.4,
                color="#1565C0", alpha=0.6, label="p10–p90 interval")
        ax.plot([pred], [0], "o", color="#42A5F5", markersize=10, zorder=5, label="Point estimate")

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_yticks([])
        ax.set_xlabel("Predicted Range (km)", color="#90CAF9", fontsize=9)
        ax.tick_params(colors="#90CAF9")
        ax.legend(fontsize=8, labelcolor="#90CAF9", framealpha=0)

        x_pad = (p_hi - p_low) * 0.3
        ax.set_xlim(max(0, p_low - x_pad), p_hi + x_pad)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── Spec summary ──────────────────────────────────────────────────
        with st.expander("📋 Input Spec Summary"):
            spec_df = pd.DataFrame({
                "Specification": [
                    "Battery", "Top Speed", "Torque", "0–100 km/h",
                    "DC Fast Charge", "Drivetrain", "Segment",
                ],
                "Value": [
                    f"{battery} kWh", f"{top_speed} km/h", f"{torque} Nm",
                    f"{accel} s", f"{fast_charge_kw} kW", drivetrain, segment,
                ],
            })
            st.dataframe(spec_df, hide_index=True, use_container_width=True)

    with col2:
        # ── SHAP explanation ──────────────────────────────────────────────
        st.subheader("🔍 Why this prediction? (SHAP)")
        with st.spinner("Fetching SHAP explanation …"):
            try:
                r2 = requests.post(f"{API_URL}/explain", json=payload, timeout=20)
                r2.raise_for_status()
                explain_data = r2.json()
            except Exception as e:
                st.warning(f"SHAP explanation unavailable: {e}")
                explain_data = None

        if explain_data:
            feat_names = explain_data["feature_names"]
            shap_vals = np.array(explain_data["shap_values"])
            base_val = explain_data["base_value"]

            # Top 15 features by |SHAP|
            top_n = 15
            order = np.argsort(np.abs(shap_vals))[-top_n:][::-1]
            top_feats = [feat_names[i] for i in order]
            top_shap = [shap_vals[i] for i in order]

            colors = ["#EF5350" if v > 0 else "#42A5F5" for v in top_shap]

            fig2, ax2 = plt.subplots(figsize=(9, 5))
            fig2.patch.set_facecolor("#0d1b2a")
            ax2.set_facecolor("#0d1b2a")

            y_pos = np.arange(len(top_feats))
            bars = ax2.barh(y_pos, top_shap, color=colors, edgecolor="none", height=0.65)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(top_feats, fontsize=9, color="#E0E0E0")
            ax2.invert_yaxis()
            ax2.axvline(0, color="#90CAF9", linewidth=0.8, linestyle="--")
            ax2.set_xlabel("SHAP value (impact on predicted range)", color="#90CAF9", fontsize=9)
            ax2.set_title(
                f"SHAP Waterfall — base={base_val:.0f} km → pred={explain_data['prediction']:.0f} km",
                color="#E0E0E0", fontsize=10, fontweight="bold",
            )
            for spine in ax2.spines.values():
                spine.set_edgecolor("#2a3a4a")
            ax2.tick_params(colors="#90CAF9")

            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

            st.caption(
                "🔴 Red bars = features pushing range **higher** | "
                "🔵 Blue bars = features pushing range **lower**"
            )

else:
    # ── Placeholder state ─────────────────────────────────────────────────────
    st.info("👈 Configure EV specifications in the sidebar and click **Predict Range**.")

    with st.expander("ℹ️ About this app"):
        st.markdown("""
        This tool predicts EV driving range from **static manufacturer specifications** only.

        **What it models:** Battery capacity, dimensions, drivetrain, motor torque, charging, etc.

        **What it does NOT model:** Weather, terrain, driving style, battery degradation (SoH), HVAC load.

        **Model:** XGBoost Regressor · **R² ≈ 0.982** on held-out test set.

        **Confidence interval:** Bootstrap p10–p90 from 500 resampled models.
        """)

# ── Sidebar footer — model card ───────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("### 📋 Model Card Summary")
    st.markdown("""
    <div class="sidebar-section">
    <b>Model:</b> XGBoost Regressor<br>
    <b>Data:</b> 478 EV models, 59 brands<br>
    <b>Test R²:</b> ~0.982<br>
    <b>Test RMSE:</b> ~13.7 km<br>
    <b>CI Method:</b> Bootstrap (500 resamples)
    </div>
    """, unsafe_allow_html=True)

    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success("✅ API connected")
    except Exception:
        st.error("❌ API offline — run `make api`")
