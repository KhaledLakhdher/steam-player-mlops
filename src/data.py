"""Data layer.

Loads real Steam history if a CSV is present in data/raw, otherwise generates realistic
synthetic panel data. The synthetic generator mimics the real distribution — game lifecycles
(launch ramp + decay), weekly seasonality, update spikes, and noise — so the ENTIRE pipeline
runs end-to-end for free, offline, today. Swap in a real Kaggle "Steam concurrent players"
CSV later via load_panel() with zero changes to the rest of the code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def generate_synthetic_panel(
    n_games: int = config.N_GAMES,
    start: str = config.START_DATE,
    end: str = config.END_DATE,
    seed: int = config.SEED,
    concept_drift_date: str | None = None,
) -> pd.DataFrame:
    """Generate a (games x days) panel of daily peak concurrent players.

    If `concept_drift_date` is set, a regime change is injected on/after that date: the weekend
    effect FLIPS (weekends were busier, become quieter — a concept drift the input->output
    relationship) and a broad level surge shifts the input distributions (so PSI detects it).
    Off by default, so Week 1/2 training data is unchanged; the replay simulation turns it on.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)
    dow = np.array([d.weekday() for d in dates])          # 0=Mon .. 6=Sun

    post = (
        np.asarray(dates >= pd.Timestamp(concept_drift_date))
        if concept_drift_date is not None
        else np.zeros(n_days, dtype=bool)
    )
    weekend_boost = np.where(dow >= 5, np.where(post, 0.75, 1.35), 1.0)   # weekend effect flips
    surge = np.where(post, 2.5, 1.0)                                       # level surge -> distribution shift

    rows = []
    for g in range(n_games):
        genre = config.GENRES[int(rng.integers(len(config.GENRES)))]
        # Base peak popularity spans small indies to blockbusters (heavy-tailed).
        base = float(np.exp(rng.normal(9.5, 1.3)))        # e^9.5 ≈ 13k
        # Release may predate the window (mature game) or fall inside it (fresh launch).
        release_offset = int(rng.integers(-800, n_days - 30))
        days_since_release = np.arange(n_days) - release_offset

        decay = rng.uniform(0.05, 0.35)                   # mild per-year decay (fairly stationary)
        lifecycle = np.exp(-decay * np.clip(days_since_release, 0, None) / 365.0)
        ramp = np.clip(days_since_release / 14.0, 0.0, 1.0)   # 2-week launch ramp
        level = base * lifecycle * ramp

        # Update/sale spikes: occasional multi-day boosts.
        spike = np.ones(n_days)
        for _ in range(int(rng.integers(2, 8))):
            s = int(rng.integers(0, n_days))
            width = int(rng.integers(3, 21))
            spike[s:min(n_days, s + width)] *= rng.uniform(1.3, 3.0)

        noise = rng.lognormal(0.0, 0.08, n_days)          # ~8% multiplicative noise
        players = np.clip(level * weekend_boost * spike * noise * surge, 20, None)

        mask = days_since_release >= 0                    # keep on/after release only
        rows.append(pd.DataFrame({
            "date": dates[mask],
            "game": f"game_{g:03d}",
            "genre": genre,
            "days_since_release": days_since_release[mask],
            "players": players[mask].round().astype(int),
        }))

    panel = pd.concat(rows, ignore_index=True)
    return panel.sort_values(["game", "date"]).reset_index(drop=True)


def load_panel() -> pd.DataFrame:
    """Load real data if a CSV is present in data/raw, else generate synthetic data.

    A real Kaggle export should provide columns:
        date, game, genre, days_since_release, players
    Adapt the read below to whatever column names the dataset uses.
    """
    csvs = sorted(config.RAW_DIR.glob("*.csv"))
    if csvs:
        df = pd.read_csv(csvs[0], parse_dates=["date"])
        return df.sort_values(["game", "date"]).reset_index(drop=True)
    return generate_synthetic_panel()


if __name__ == "__main__":
    p = load_panel()
    print(p.head())
    print(f"\n{len(p):,} rows | {p['game'].nunique()} games | "
          f"{p['date'].min().date()} -> {p['date'].max().date()}")
