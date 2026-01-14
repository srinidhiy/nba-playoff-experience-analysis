"""Summarize playoff series coverage and round counts."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "nba_api"


def list_seasons() -> list[str]:
    seasons = []
    if RAW_DIR.exists():
        for path in RAW_DIR.iterdir():
            if path.is_dir() and re.match(r"^\d{4}-\d{2}$", path.name):
                seasons.append(path.name)
    return sorted(seasons)


def main() -> None:
    seasons = list_seasons()
    if not seasons:
        print("No seasons found under data/raw/nba_api.")
        return

    summary_rows = []
    for season in seasons:
        path = RAW_DIR / season / "series_results.csv"
        if not path.exists():
            summary_rows.append(
                {
                    "season": season,
                    "teams": 0,
                    "series": 0,
                    "round_1": 0,
                    "round_2": 0,
                    "round_3": 0,
                    "round_4": 0,
                }
            )
            continue

        df = pd.read_csv(path)
        teams = df["team_id"].nunique()
        series = df["series_id"].nunique()
        rounds = df.groupby("round")["series_id"].nunique().to_dict()
        summary_rows.append(
            {
                "season": season,
                "teams": teams,
                "series": series,
                "round_1": rounds.get(1, 0),
                "round_2": rounds.get(2, 0),
                "round_3": rounds.get(3, 0),
                "round_4": rounds.get(4, 0),
            }
        )

    summary = pd.DataFrame(summary_rows)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
