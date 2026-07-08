import config
from src.data import generate_synthetic_panel
from src.features import build_feature_row, build_training_frame


def test_feature_columns_and_no_nan():
    frame = build_training_frame(generate_synthetic_panel(n_games=5))
    assert list(frame[config.FEATURE_COLUMNS].columns) == config.FEATURE_COLUMNS
    assert not frame[config.FEATURE_COLUMNS].isna().any().any()


def test_train_serve_feature_consistency():
    """The serving path (build_feature_row) must produce the same features as training."""
    panel = generate_synthetic_panel(n_games=1)
    frame = build_training_frame(panel)
    row = frame.iloc[10]

    game = panel[panel["game"] == row["game"]].sort_values("date").reset_index(drop=True)
    idx = int(game.index[game["date"] == row["date"]][0])
    recent = game.loc[idx - 6:idx, "players"].tolist()      # 7 days up to and including day t
    assert len(recent) == 7

    served = build_feature_row(
        recent, int(row["day_of_week"]), int(row["days_since_release"]), row["genre"]
    )
    for col in ["players_lag_1", "players_lag_7", "players_roll_mean_7"]:
        assert abs(served[col].iloc[0] - row[col]) < 1e-6
