"""Live Steam Web API collector — the 'living system' data ingestion.

Polls current concurrent players for a basket of games via Steam's free, keyless endpoint
(ISteamUserStats/GetNumberOfCurrentPlayers) and appends to a Parquet store, de-duplicated by
(date, appid). Designed to run on a GitHub Actions cron (.github/workflows/collect.yml) so data
keeps flowing after you deploy. Falls back to a synthetic value per game when offline, so it
runs anywhere (CI, sandbox, your laptop with no network).

Run:  python -m src.collect          (live, with offline fallback per game)
      python -m src.collect --offline (force synthetic values)
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request

import numpy as np
import pandas as pd

import config

# A small basket of well-known appids. Extend to 40-60 across lifecycle stages (see roadmap).
GAMES = {
    "730": "Counter-Strike 2",
    "570": "Dota 2",
    "440": "Team Fortress 2",
    "578080": "PUBG: BATTLEGROUNDS",
    "271590": "Grand Theft Auto V",
    "1172470": "Apex Legends",
    "252490": "Rust",
    "1085660": "Destiny 2",
    "359550": "Rainbow Six Siege",
    "304930": "Unturned",
    "236390": "War Thunder",
    "550": "Left 4 Dead 2",
}

STORE = config.DATA_DIR / "collected.parquet"
ENDPOINT = "https://api.steampowered.com/ISteamUserStats/GetNumberOfCurrentPlayers/v1/?appid={appid}"


def fetch_players(appid: str) -> int:
    with urllib.request.urlopen(ENDPOINT.format(appid=appid), timeout=10) as resp:
        data = json.load(resp)
    return int(data["response"]["player_count"])


def collect(offline: bool = False) -> pd.DataFrame:
    today = dt.date.today().isoformat()
    rows = []
    for appid, name in GAMES.items():
        try:
            if offline:
                raise RuntimeError("offline mode")
            players = fetch_players(appid)
            source = "steam-api"
        except Exception:  # noqa: BLE001 — network down / rate limited: fall back to synthetic
            seed = abs(hash((appid, today))) % (2**32)
            players = int(np.random.default_rng(seed).lognormal(10, 1))
            source = "synthetic-fallback"
        rows.append({"date": today, "appid": appid, "game": name, "players": players, "source": source})

    new = pd.DataFrame(rows)
    if STORE.exists():
        combined = (
            pd.concat([pd.read_parquet(STORE), new])
            .drop_duplicates(["date", "appid"], keep="last")
            .reset_index(drop=True)
        )
    else:
        combined = new
    combined.to_parquet(STORE, index=False)
    return new


def main() -> None:
    new = collect(offline="--offline" in sys.argv)
    print(new.to_string(index=False))
    print(f"\nAppended {len(new)} rows -> {STORE}")


if __name__ == "__main__":
    main()
