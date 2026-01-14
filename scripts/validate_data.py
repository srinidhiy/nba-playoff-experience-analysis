"""Validate raw and processed data completeness."""

from __future__ import annotations

import json
import re
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
NBA_DIR = RAW_DIR / "nba_api"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

REQUIRED_FILES = {
    "team_regular.csv",
    "team_regular_advanced.csv",
    "team_playoffs.csv",
    "team_playoffs_advanced.csv",
    "player_regular.csv",
    "player_playoffs.csv",
    "rosters.csv",
    "standings.csv",
    "series_results.csv",
}


def list_seasons() -> list[str]:
    seasons = []
    if NBA_DIR.exists():
        for path in NBA_DIR.iterdir():
            if path.is_dir() and re.match(r"^\d{4}-\d{2}$", path.name):
                seasons.append(path.name)
    config_path = NBA_DIR / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        seasons.extend(config.get("seasons", []))
    return sorted(set(seasons))


def main() -> None:
    seasons = list_seasons()
    if not seasons:
        print("No seasons found under data/raw/nba_api.")
        return

    missing = {}
    for season in seasons:
        season_dir = NBA_DIR / season
        season_missing = [
            name for name in sorted(REQUIRED_FILES) if not (season_dir / name).exists()
        ]
        if season_missing:
            missing[season] = season_missing

    processed_missing = []
    for name in ["team_features.csv", "player_experience.csv"]:
        if not (PROCESSED_DIR / name).exists():
            processed_missing.append(name)

    print("Seasons detected:", ", ".join(seasons))
    if missing:
        print("\nMissing raw files:")
        for season, files in missing.items():
            print(f"- {season}: {', '.join(files)}")
    else:
        print("\nAll required raw files present.")

    if processed_missing:
        print("\nMissing processed outputs:")
        print(", ".join(processed_missing))
    else:
        print("\nProcessed outputs present.")


if __name__ == "__main__":
    main()
