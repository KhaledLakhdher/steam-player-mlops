"""Streamlit dashboard — the clickable, portfolio-facing view of the system.

Run:  streamlit run dashboard.py
Shows a live prediction widget, the current drift status (PSI vs the frozen reference), and the
temporal-replay chart. Reuses the exact same src/ modules as training and serving.
"""
from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

import config
from src.data import generate_synthetic_panel
from src.features import build_feature_row, build_training_frame
from src.monitoring import psi_report

st.set_page_config(page_title="Steam Player Forecaster", page_icon="🎮", layout="wide")


@st.cache_resource
def load_model():
    path = config.MODELS_DIR / "model.joblib"
    return joblib.load(path) if path.exists() else None


@st.cache_data
def load_reference():
    path = config.MODELS_DIR / "reference.parquet"
    return pd.read_parquet(path) if path.exists() else None


st.title("🎮 Steam Player-Count Forecaster — MLOps")
st.caption("Predict next-day peak concurrent players · monitor drift · retrain on a champion/challenger gate")

model = load_model()
if model is None:
    st.error("No model found. Run `python -m src.train` first.")
    st.stop()

# ---- Live prediction -------------------------------------------------------
st.header("Live prediction")
c1, c2 = st.columns(2)
with c1:
    genre = st.selectbox("Genre", config.GENRES)
    dsr = st.number_input("Days since release", min_value=0, max_value=5000, value=420)
    dow = st.selectbox(
        "Day being predicted", list(range(7)),
        format_func=lambda d: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][d],
    )
with c2:
    recent_str = st.text_area(
        "Last 7 daily peak players (oldest → newest, comma-separated)",
        "12000, 12500, 11800, 13000, 12700, 12400, 12900",
    )

try:
    recent = [float(x) for x in recent_str.replace("\n", "").split(",") if x.strip()]
    X = build_feature_row(recent, dow, dsr, genre)
    pred = int(max(0, round(model.predict(X)[0])))
    st.metric("Predicted next-day peak players", f"{pred:,}")
except Exception as e:  # noqa: BLE001
    st.warning(f"Enter at least 7 comma-separated numbers. ({e})")

# ---- Drift monitor ---------------------------------------------------------
st.header("Drift monitor")
reference = load_reference()
if reference is None:
    st.info("No reference yet — run `python -m src.train`.")
else:
    frame = build_training_frame(generate_synthetic_panel(concept_drift_date="2024-01-01"))
    current = frame[frame["date"] > frame["date"].max() - pd.DateOffset(days=30)]
    report = psi_report(reference, current)
    st.subheader("🔴 DRIFT DETECTED" if report["drift"] else "🟢 No significant drift")
    st.write(f"Max PSI **{report['max_psi']:.3f}** vs threshold {config.PSI_THRESHOLD}")
    st.bar_chart(pd.Series(report["per_feature"], name="PSI"))

# ---- Temporal replay -------------------------------------------------------
st.header("Temporal replay — drift & retraining")
chart = config.REPORTS_DIR / "replay.png"
if chart.exists():
    st.image(str(chart))
    st.caption("Stable → concept drift spikes PSI → gated champion/challenger retrain → recovery.")
else:
    st.info("Run `python -m src.replay` to generate the chart.")
