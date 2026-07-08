"""Train a global next-day player-count model and log everything to MLflow.

Run from the project root:  python -m src.train
"""
from __future__ import annotations

import json

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import config
from src.data import load_panel
from src.features import build_training_frame
from src.validation import validate_panel


def time_based_split(df):
    """Split chronologically. NEVER random-split time series — it leaks the future."""
    cutoff = df["date"].quantile(1 - config.TEST_FRACTION)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    return train, test, cutoff


def main():
    print("Loading panel data ...")
    panel = load_panel()
    panel = validate_panel(panel)                             # fail loudly on bad data
    print(f"  {len(panel):,} rows | {panel['game'].nunique()} games | schema OK")

    print("Building features ...")
    frame = build_training_frame(panel)
    train, test, cutoff = time_based_split(frame)
    print(f"  train={len(train):,}  test={len(test):,}  cutoff={cutoff.date()}")

    X_train, y_train = train[config.FEATURE_COLUMNS], train[config.TARGET]
    X_test, y_test = test[config.FEATURE_COLUMNS], test[config.TARGET]

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(
            config.MLFLOW_EXPERIMENT, artifact_location=config.MLFLOW_ARTIFACTS.as_uri()
        )
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    params = dict(max_iter=400, learning_rate=0.05, random_state=config.SEED)

    with mlflow.start_run() as run:
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_train, y_train)

        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        m = y_test.values > 100                                   # MAPE only on non-tiny targets
        mape = float(np.mean(np.abs((y_test.values[m] - pred[m]) / y_test.values[m])) * 100)

        mlflow.log_params(params)
        mlflow.log_param("n_games", int(panel["game"].nunique()))
        mlflow.log_param("n_train_rows", int(len(train)))
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)
        # Log AND register the model as a new version in the MLflow Model Registry.
        info = mlflow.sklearn.log_model(
            sk_model=model, name="model", registered_model_name=config.REGISTERED_MODEL
        )
        version = info.registered_model_version
        # Point the serving alias at this version (Week 3 will gate this on beating the champion).
        MlflowClient().set_registered_model_alias(
            config.REGISTERED_MODEL, config.MODEL_ALIAS, version
        )

        # Local copy too — the API uses this only as a fallback if the registry is unavailable.
        joblib.dump(model, config.MODELS_DIR / "model.joblib")
        (config.MODELS_DIR / "features.json").write_text(json.dumps(config.FEATURE_COLUMNS, indent=2))
        # Freeze the training feature distribution as the drift-monitoring reference.
        train[config.FEATURE_COLUMNS].to_parquet(config.MODELS_DIR / "reference.parquet")

        print("\n=== Results (test set) ===")
        print(f"  RMSE: {rmse:,.0f} players")
        print(f"  MAE : {mae:,.0f} players")
        print(f"  MAPE: {mape:.1f}%")
        print(f"\nRegistered '{config.REGISTERED_MODEL}' v{version}  ->  alias @{config.MODEL_ALIAS}")
        print(f"MLflow run {run.info.run_id}")
        print("Browse:  mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
