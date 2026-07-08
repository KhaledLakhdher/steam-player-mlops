"""End-to-end check of the serving layer without starting a server.

Run from the project root:  python scripts/smoke_test.py
Using TestClient as a context manager runs the startup event, which loads the model.
"""
from fastapi.testclient import TestClient

from api.main import app

with TestClient(app) as client:
    print("health:", client.get("/health").json())

    payload = {
        "recent_players": [12000, 12500, 11800, 13000, 12700, 12400, 12900],
        "day_of_week": 5,
        "days_since_release": 420,
        "genre": "shooter",
    }
    r = client.post("/predict", json=payload)
    print("predict:", r.status_code, r.json())

    # Guardrail: <7 days of history must be rejected with 422.
    bad = client.post("/predict", json={**payload, "recent_players": [1, 2, 3]})
    print("short-history (expect 422):", bad.status_code)
