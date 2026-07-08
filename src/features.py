"""Feature engineering.

Kept in ONE place so training and serving build byte-for-byte identical features — the most
common source of silent train/serve skew bugs. Both paths produce columns == config.FEATURE_COLUMNS.

Framing: at the end of day t we know players[t] and all history; we predict players[t+1].
So features use info up to day t, plus the calendar of the predicted day (t+1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config


def _add_genre_onehot(df: pd.DataFrame) -> pd.DataFrame:
    for g in config.GENRES:
        df[f"genre_{g}"] = (df["genre"] == g).astype(int)
    return df


def build_training_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """From a raw panel, build the per-row feature matrix + target (next-day players)."""
    df = panel.sort_values(["game", "date"]).copy()
    grp = df.groupby("game")

    df["players_lag_1"] = df["players"]                                   # latest known (day t)
    df["players_lag_7"] = grp["players"].shift(6)                        # ~7 days back
    df["players_roll_mean_7"] = grp["players"].transform(lambda s: s.rolling(7).mean())

    next_date = df["date"] + np.timedelta64(1, "D")                      # calendar of predicted day
    df["day_of_week"] = next_date.dt.weekday
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["days_since_release"] = df["days_since_release"] + 1              # at the predicted day

    df[config.TARGET] = grp["players"].shift(-1)                         # players[t+1]

    df = _add_genre_onehot(df)
    df = df.dropna(subset=config.FEATURE_COLUMNS + [config.TARGET]).reset_index(drop=True)
    return df


def build_feature_row(
    recent_players: list[float],
    day_of_week: int,
    days_since_release: int,
    genre: str,
) -> pd.DataFrame:
    """Build a single feature row from a game's recent daily history (used by /predict).

    recent_players: at least 7 daily peaks, oldest -> newest (the last value is 'today', day t).
    day_of_week / days_since_release: values for the day being predicted (t+1).
    """
    if len(recent_players) < 7:
        raise ValueError("need at least 7 days of recent_players (oldest -> newest)")
    recent = np.asarray(recent_players, dtype=float)
    row = {
        "players_lag_1": recent[-1],
        "players_lag_7": recent[-7],
        "players_roll_mean_7": recent[-7:].mean(),
        "day_of_week": int(day_of_week),
        "is_weekend": int(day_of_week >= 5),
        "days_since_release": int(days_since_release),
    }
    for g in config.GENRES:
        row[f"genre_{g}"] = int(genre == g)
    return pd.DataFrame([row], columns=config.FEATURE_COLUMNS)
