"""Standalone drift check for the scheduled monitoring job.

Compares recent production data against the frozen reference saved at training time and reports
PSI per feature. Exits non-zero when drift is detected, so a CI/cron job can react (alert, or
trigger a retrain).

Run:  python -m src.check_drift
"""
from __future__ import annotations

import sys

import pandas as pd

import config
from src.data import generate_synthetic_panel
from src.features import build_training_frame
from src.monitoring import psi_report

RECENT_DAYS = 30


def load_reference() -> pd.DataFrame:
    path = config.MODELS_DIR / "reference.parquet"
    if not path.exists():
        raise SystemExit("No reference found. Run `python -m src.train` first.")
    return pd.read_parquet(path)


def recent_window() -> pd.DataFrame:
    # In production this reads your latest COLLECTED data (Week 4's Steam-API collector).
    # For the demo we synthesize a recent, drifted window so the check has something to fire on.
    frame = build_training_frame(generate_synthetic_panel(concept_drift_date="2024-01-01"))
    cutoff = frame["date"].max() - pd.DateOffset(days=RECENT_DAYS)
    return frame[frame["date"] > cutoff]


def main() -> None:
    report = psi_report(load_reference(), recent_window())
    print("Drift report (PSI vs frozen reference):")
    for feature, value in report["per_feature"].items():
        print(f"  {feature:22s} {value:.3f}")
    print(f"  max PSI: {report['max_psi']:.3f}  (threshold {config.PSI_THRESHOLD})")
    if report["drift"]:
        print("DRIFT DETECTED -> a retrain should run.")
        sys.exit(1)
    print("No significant drift.")
    sys.exit(0)


if __name__ == "__main__":
    main()
