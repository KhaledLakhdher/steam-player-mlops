"""Temporal-replay simulation — the centerpiece artifact.

Trains a champion on the pre-drift past, then walks forward month by month simulating
production. Each month it (1) measures the live champion's error, (2) tests for input drift
(PSI vs a frozen reference), and (3) when drift fires, trains a challenger and PROMOTES it only
if it beats the champion on held-out data (the champion/challenger gate). The resulting chart
shows the whole MLOps story in one image: error degrades under drift, the alarm fires,
retraining runs, error recovers.

Run:  python -m src.replay      (writes reports/replay.png)
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import config
from src.data import generate_synthetic_panel
from src.features import build_training_frame
from src.monitoring import psi_report

DRIFT_DATE = "2024-01-01"         # concept drift (weekend effect flips) starts here
INITIAL_TRAIN_END = "2023-10-01"  # champion trained on data strictly before this
GATE_MARGIN = 0.02                # challenger must beat champion by >2% relative MAPE
REFERENCE_WINDOW_MONTHS = 3       # drift reference = a recent trailing window (comparable in time)


def _window(frame, end_ts, months=REFERENCE_WINDOW_MONTHS):
    """Recent trailing window [end_ts - months, end_ts) — a time-comparable drift reference."""
    start_ts = end_ts - pd.DateOffset(months=months)
    return frame[(frame["date"] >= start_ts) & (frame["date"] < end_ts)]


def _fit(frame_slice):
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=config.SEED)
    m.fit(frame_slice[config.FEATURE_COLUMNS], frame_slice[config.TARGET])
    return m


def _mape(model, X, y):
    pred = model.predict(X)
    mask = y.values > 100
    return float(np.mean(np.abs((y.values[mask] - pred[mask]) / y.values[mask])) * 100)


def run_replay():
    # 1. Data with an injected concept-drift event.
    panel = generate_synthetic_panel(concept_drift_date=DRIFT_DATE)
    frame = build_training_frame(panel)

    # 2. Initial champion (full pre-drift history) + a recent-window drift reference.
    init_end = pd.Timestamp(INITIAL_TRAIN_END)
    champion = _fit(frame[frame["date"] < init_end])
    reference = _window(frame, init_end)[config.FEATURE_COLUMNS]

    # 3. Walk forward month by month, simulating production.
    months = pd.period_range(INITIAL_TRAIN_END, frame["date"].max(), freq="M")
    records = []
    for p in months:
        mask = (frame["date"] >= p.start_time) & (frame["date"] <= p.end_time)
        month_df = frame[mask]
        if month_df.empty:
            continue
        Xm, ym = month_df[config.FEATURE_COLUMNS], month_df[config.TARGET]

        champ_mape = _mape(champion, Xm, ym)                 # error of the model live this month
        report = psi_report(reference, month_df)
        retrained = False

        if report["drift"]:
            # Challenger trained on everything BEFORE this month (no leakage); gate on this month.
            challenger = _fit(frame[frame["date"] < p.start_time])
            chall_mape = _mape(challenger, Xm, ym)
            if chall_mape < champ_mape * (1 - GATE_MARGIN):
                champion = challenger                        # promote
                reference = _window(frame, p.start_time)[config.FEATURE_COLUMNS]  # reset to recent baseline
                retrained = True

        records.append({
            "month": str(p),
            "champion_mape": round(champ_mape, 2),
            "max_psi": round(report["max_psi"], 3),
            "drift": report["drift"],
            "retrained": retrained,
        })

    rep = pd.DataFrame(records)
    _plot(rep)
    return rep


def _plot(rep):
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    x = list(range(len(rep)))
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.plot(x, rep["champion_mape"], "-o", color="#c0392b", label="Champion MAPE (%)")
    ax1.set_ylabel("MAPE (%)", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(rep["month"], rotation=45, ha="right", fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, rep["max_psi"], "--s", color="#2980b9", label="Max PSI (drift)")
    ax2.axhline(config.PSI_THRESHOLD, color="#2980b9", ls=":", lw=1, alpha=0.7)
    ax2.set_ylabel("Max PSI", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")

    for i, r in rep.reset_index(drop=True).iterrows():
        if r["retrained"]:
            ax1.axvline(i, color="green", ls="-", lw=1.5, alpha=0.6)
            ax1.annotate("retrain", (i, ax1.get_ylim()[1]), color="green",
                         fontsize=8, ha="center", va="top")

    ax1.set_title("Temporal replay — drift detection & champion/challenger retraining")
    fig.tight_layout()
    out = config.REPORTS_DIR / "replay.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved chart to {out}")


if __name__ == "__main__":
    rep = run_replay()
    with pd.option_context("display.max_rows", None, "display.width", 120):
        print(rep.to_string(index=False))
    print(
        f"\nMonths: {len(rep)} | drift months: {int(rep['drift'].sum())} | "
        f"retrains: {int(rep['retrained'].sum())}"
    )
