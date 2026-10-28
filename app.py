from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from src.forecasting import make_supervised, FEATURES, TARGET

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "load_forecasting_model.joblib"
DATA_PATH = ROOT / "data" / "sample_hourly_load.csv"
OUT = ROOT / "outputs"

st.set_page_config(page_title="AI Electrical Load Forecasting", page_icon="⚡", layout="wide")
st.title("⚡ AI-Based Electrical Load Forecasting System")
st.caption("Random Forest | One-hour-ahead forecast | Chronological test evaluation")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python train_model.py --sample` or `python train_model.py --uci` first.")
    st.stop()

artifact = joblib.load(MODEL_PATH)
model = artifact["model"]
st.info(f"Model data source: **{artifact.get('data_source', 'unknown')}**. If this says synthetic demo data, train with `python train_model.py --uci` before reporting real-data results.")

metrics_path = OUT / "metrics.json"
if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    st.subheader("Test-set performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (kW)", f"{metrics['MAE_kW']:.3f}")
    c2.metric("RMSE (kW)", f"{metrics['RMSE_kW']:.3f}")
    c3.metric("R²", f"{metrics['R2']:.3f}")
    c4.metric("Test rows", f"{metrics['test_rows']:,}")
    st.caption(f"Test period: {metrics['test_start']} to {metrics['test_end']} | Baseline MAE: {metrics['persistence_baseline_MAE_kW']:.3f} kW")

st.subheader("Forecast from recent hourly readings")
pred_path = OUT / "test_predictions.csv"
if pred_path.exists():
    history = pd.read_csv(pred_path, parse_dates=["forecast_origin"])
    history = history.dropna(subset=["actual_next_hour_kw"]).sort_values("forecast_origin")
    if len(history):
        st.write("Choose a historical forecast origin from the held-out test period. This demonstrates the model's forecast at that point in time.")
        pos = st.slider("Test row", min_value=0, max_value=len(history)-1, value=max(0, len(history)-1))
        row = history.iloc[pos]
        a,b,c = st.columns(3)
        a.metric("Model forecast (kW)", f"{row['predicted_next_hour_kw']:.3f}")
        b.metric("Actual next-hour load (kW)", f"{row['actual_next_hour_kw']:.3f}")
        c.metric("Absolute error (kW)", f"{abs(row['predicted_next_hour_kw']-row['actual_next_hour_kw']):.3f}")
        st.caption(f"Forecast origin: {row['forecast_origin']}. Target is the following hour.")
        chart = history.set_index("forecast_origin")[["actual_next_hour_kw", "predicted_next_hour_kw"]].rename(
            columns={"actual_next_hour_kw":"Actual", "predicted_next_hour_kw":"Random Forest"}
        )
        st.line_chart(chart.tail(24*7))
else:
    st.warning("Prediction output not found. Run training again.")

st.subheader("Training graphs")
for filename, title in [
    ("load_profile.png", "Daily average load profile"),
    ("actual_vs_predicted.png", "Actual versus predicted"),
    ("feature_importance.png", "Feature importance"),
]:
    path = OUT / filename
    if path.exists():
        st.markdown(f"**{title}**")
        st.image(str(path), use_container_width=True)

with st.expander("About this project"):
    st.markdown("""
    - **Goal:** predict average active power for the next hour.
    - **Features:** hour/day/month, recent load lags, and trailing rolling means.
    - **Split:** first 80% of time-ordered examples for training, last 20% for testing.
    - **Metrics:** MAE, RMSE and R²; a persistence baseline is also shown.
    - **Warning:** a forecast is not an electrical protection or safety system.
    """)
