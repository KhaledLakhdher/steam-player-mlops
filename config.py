"""Central configuration for the Steam player-count MLOps project.

Everything the pipeline needs to agree on lives here — paths, the games/date window,
and (critically) the FEATURE_COLUMNS, so that training and serving build identical features.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MLFLOW_DB = ROOT / "mlflow.db"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"   # DB backend (file store is retired in MLflow 3)

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, MLFLOW_ARTIFACTS, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Data / domain settings ---
SEED = 42
N_GAMES = 50
START_DATE = "2022-01-01"
END_DATE = "2024-12-31"
GENRES = ["shooter", "moba", "rpg", "strategy", "survival", "sandbox"]

# --- Modeling ---
TARGET = "players_next"
TEST_FRACTION = 0.2          # last 20% of the timeline is the test set (TIME-BASED split)
MLFLOW_EXPERIMENT = "steam-player-forecasting"
REGISTERED_MODEL = "steam-player-forecaster"
MODEL_ALIAS = "champion"     # the alias the API serves (Week 3 gates promotion to it)

# --- Monitoring / drift ---
# Player-level features only. days_since_release is a monotonic counter that always "drifts"
# against a historical reference, so it's excluded from drift detection.
DRIFT_FEATURES = ["players_lag_1", "players_lag_7", "players_roll_mean_7"]
PSI_THRESHOLD = 0.2          # max-feature PSI above this = significant drift

# Feature columns the model trains on — defined in ONE place so train + serve never drift apart.
NUMERIC_FEATURES = [
    "players_lag_1",          # most recent known daily peak (day t)
    "players_lag_7",          # daily peak ~7 days back
    "players_roll_mean_7",    # mean of the last 7 known days
    "day_of_week",            # weekday of the day being predicted (t+1)
    "is_weekend",
    "days_since_release",     # at the predicted day
]
GENRE_FEATURES = [f"genre_{g}" for g in GENRES]
FEATURE_COLUMNS = NUMERIC_FEATURES + GENRE_FEATURES
