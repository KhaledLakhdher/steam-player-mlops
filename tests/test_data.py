import pytest

from src.data import generate_synthetic_panel
from src.validation import validate_panel


def test_panel_is_valid():
    panel = generate_synthetic_panel(n_games=5)
    validated = validate_panel(panel)
    assert len(validated) > 0
    assert {"date", "game", "genre", "players", "days_since_release"}.issubset(validated.columns)


def test_validation_rejects_negative_players():
    panel = generate_synthetic_panel(n_games=3)
    panel.loc[panel.index[0], "players"] = -5      # invalid: players must be > 0
    with pytest.raises(Exception):
        validate_panel(panel)


def test_validation_rejects_unknown_genre():
    panel = generate_synthetic_panel(n_games=3)
    panel.loc[panel.index[0], "genre"] = "not-a-real-genre"
    with pytest.raises(Exception):
        validate_panel(panel)
