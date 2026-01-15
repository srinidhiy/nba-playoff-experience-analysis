"""Deep investigation into 2019-2024 accuracy drop.

Analyzes:
- COVID 2020 bubble effects
- Play-in tournament impact (started 2021)
- Increased player movement / super team effects
- Year-by-year accuracy breakdown
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"

ROUND_LABELS = {
    0: "No Playoffs",
    1: "First Round",
    2: "Second Round",
    3: "Conference Finals",
    4: "Finals",
}

# Era boundaries
COVID_YEAR = 2020
PLAY_IN_START = 2021
MODERN_ERA_START = 2019


def load_features() -> pd.DataFrame:
    path = PROCESSED_DIR / "team_features.csv"
    if not path.exists():
        raise FileNotFoundError("Run feature_build.py first.")
    return pd.read_csv(path)


def load_model(era: str | None = None) -> tuple:
    """Load model bundle, optionally era-specific."""
    if era:
        path = ARTIFACTS_DIR / f"playoff_round_model_{era}.joblib"
    else:
        path = ARTIFACTS_DIR / "playoff_round_model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_columns"]


def analyze_year(
    df: pd.DataFrame,
    year: int,
    model,
    feature_columns: list[str],
) -> dict:
    """Analyze a single year's predictions."""
    year_df = df[df["season_start"] == year].dropna(subset=feature_columns)
    if year_df.empty:
        return {"year": year, "count": 0}

    x = year_df[feature_columns]
    y_true = year_df["playoff_round_reached"].astype(int)
    y_pred = model.predict(x).astype(int)

    accuracy = float((y_pred == y_true).mean())
    within_one = float((abs(y_pred - y_true) <= 1).mean())

    # Count upsets (lower seed goes further than predicted)
    year_df = year_df.copy()
    year_df["predicted"] = y_pred
    year_df["actual"] = y_true
    year_df["error"] = y_true - y_pred

    # Overperformers: actual > predicted
    overperformers = year_df[year_df["actual"] > year_df["predicted"]]
    # Underperformers: actual < predicted
    underperformers = year_df[year_df["actual"] < year_df["predicted"]]

    return {
        "year": int(year),
        "count": int(len(year_df)),
        "accuracy": accuracy,
        "within_one_round": within_one,
        "overperformers": int(len(overperformers)),
        "underperformers": int(len(underperformers)),
        "exact_matches": int((y_pred == y_true).sum()),
        "avg_error": float((y_true - y_pred).mean()),
        "abs_avg_error": float(abs(y_true - y_pred).mean()),
    }


def analyze_seed_performance(df: pd.DataFrame, year: int) -> dict:
    """Analyze how well seeding predicts outcomes."""
    year_df = df[df["season_start"] == year].copy()
    if year_df.empty or "seed" not in year_df.columns:
        return {}

    # Correlation between seed and playoff round
    year_df = year_df.dropna(subset=["seed", "playoff_round_reached"])
    if len(year_df) < 5:
        return {}

    # Lower seed should correlate with higher round reached
    corr = year_df["seed"].corr(year_df["playoff_round_reached"])
    return {
        "year": int(year),
        "seed_round_correlation": float(corr) if pd.notna(corr) else 0.0,
        "avg_seed_round_1": float(
            year_df[year_df["playoff_round_reached"] == 1]["seed"].mean()
        )
        if not year_df[year_df["playoff_round_reached"] == 1].empty
        else 0.0,
        "avg_seed_finals": float(
            year_df[year_df["playoff_round_reached"] == 4]["seed"].mean()
        )
        if not year_df[year_df["playoff_round_reached"] == 4].empty
        else 0.0,
    }


def main() -> None:
    df = load_features()
    df = df[df["playoff_round_reached_source"] == "nba_api_series"].copy()
    df["season_start"] = df["season"].str[:4].astype(int)

    try:
        model, feature_columns = load_model()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Analyze years from 2015 to latest
    latest = df["season_start"].max()
    years = range(2015, latest + 1)

    year_results = []
    seed_results = []
    for year in years:
        result = analyze_year(df, year, model, feature_columns)
        if result.get("count", 0) > 0:
            year_results.append(result)
            seed_result = analyze_seed_performance(df, year)
            if seed_result:
                seed_results.append(seed_result)

    year_df = pd.DataFrame(year_results)

    # Compute era aggregates
    pre_2019 = year_df[year_df["year"] < MODERN_ERA_START]
    post_2019 = year_df[year_df["year"] >= MODERN_ERA_START]
    covid_only = year_df[year_df["year"] == COVID_YEAR]
    play_in_era = year_df[year_df["year"] >= PLAY_IN_START]

    era_summary = {
        "pre_2019": {
            "years": list(pre_2019["year"]),
            "avg_accuracy": float(pre_2019["accuracy"].mean())
            if not pre_2019.empty
            else 0.0,
            "avg_within_one": float(pre_2019["within_one_round"].mean())
            if not pre_2019.empty
            else 0.0,
        },
        "post_2019": {
            "years": list(post_2019["year"]),
            "avg_accuracy": float(post_2019["accuracy"].mean())
            if not post_2019.empty
            else 0.0,
            "avg_within_one": float(post_2019["within_one_round"].mean())
            if not post_2019.empty
            else 0.0,
        },
        "covid_2020": {
            "accuracy": float(covid_only["accuracy"].iloc[0])
            if not covid_only.empty
            else 0.0,
            "within_one": float(covid_only["within_one_round"].iloc[0])
            if not covid_only.empty
            else 0.0,
            "note": "Bubble playoffs in Orlando, no home court advantage",
        },
        "play_in_era_2021_plus": {
            "years": list(play_in_era["year"]),
            "avg_accuracy": float(play_in_era["accuracy"].mean())
            if not play_in_era.empty
            else 0.0,
            "avg_within_one": float(play_in_era["within_one_round"].mean())
            if not play_in_era.empty
            else 0.0,
            "note": "Play-in tournament added, affecting 7-10 seeds",
        },
    }

    # Roster continuity comparison
    if "roster_continuity" in df.columns:
        pre_continuity = df[df["season_start"] < MODERN_ERA_START][
            "roster_continuity"
        ].mean()
        post_continuity = df[df["season_start"] >= MODERN_ERA_START][
            "roster_continuity"
        ].mean()
        era_summary["roster_continuity"] = {
            "pre_2019_avg": float(pre_continuity) if pd.notna(pre_continuity) else 0.0,
            "post_2019_avg": float(post_continuity)
            if pd.notna(post_continuity)
            else 0.0,
            "note": "Lower continuity = more player movement",
        }

    # Experience comparison
    if "avg_playoff_games_prior" in df.columns:
        pre_exp = df[df["season_start"] < MODERN_ERA_START][
            "avg_playoff_games_prior"
        ].mean()
        post_exp = df[df["season_start"] >= MODERN_ERA_START][
            "avg_playoff_games_prior"
        ].mean()
        era_summary["avg_playoff_experience"] = {
            "pre_2019_avg": float(pre_exp) if pd.notna(pre_exp) else 0.0,
            "post_2019_avg": float(post_exp) if pd.notna(post_exp) else 0.0,
        }

    # Save results
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "analysis_description": "Investigation into 2019-2024 accuracy drop",
        "year_by_year": year_results,
        "era_summary": era_summary,
        "seed_analysis": seed_results,
        "key_findings": [],
    }

    # Derive key findings
    findings = []
    if era_summary["pre_2019"]["avg_accuracy"] > era_summary["post_2019"]["avg_accuracy"]:
        drop = (
            era_summary["pre_2019"]["avg_accuracy"]
            - era_summary["post_2019"]["avg_accuracy"]
        )
        findings.append(
            f"Accuracy dropped {drop:.1%} from pre-2019 to post-2019 era"
        )

    if (
        era_summary.get("roster_continuity", {}).get("pre_2019_avg", 0)
        > era_summary.get("roster_continuity", {}).get("post_2019_avg", 0)
    ):
        findings.append(
            "Roster continuity decreased post-2019, indicating more player movement"
        )

    if covid_only.empty is False and len(covid_only) > 0:
        covid_acc = float(covid_only["accuracy"].iloc[0])
        if covid_acc < era_summary["post_2019"]["avg_accuracy"]:
            findings.append(
                f"COVID bubble year (2020) had lower accuracy ({covid_acc:.1%}) than post-2019 average"
            )

    output["key_findings"] = findings

    (ARTIFACTS_DIR / "era_investigation.json").write_text(
        json.dumps(output, indent=2)
    )

    # Print summary
    print("=" * 60)
    print("ERA INVESTIGATION: 2019-2024 Accuracy Drop Analysis")
    print("=" * 60)
    print("\nYear-by-year accuracy:")
    for r in year_results:
        marker = ""
        if r["year"] == COVID_YEAR:
            marker = " [COVID BUBBLE]"
        elif r["year"] >= PLAY_IN_START:
            marker = " [PLAY-IN ERA]"
        print(
            f"  {r['year']}{marker}: accuracy={r['accuracy']:.3f}, "
            f"within_one={r['within_one_round']:.3f}, "
            f"over/under={r['overperformers']}/{r['underperformers']}"
        )

    print("\nEra comparison:")
    print(
        f"  Pre-2019:  avg accuracy={era_summary['pre_2019']['avg_accuracy']:.3f}, "
        f"within_one={era_summary['pre_2019']['avg_within_one']:.3f}"
    )
    print(
        f"  Post-2019: avg accuracy={era_summary['post_2019']['avg_accuracy']:.3f}, "
        f"within_one={era_summary['post_2019']['avg_within_one']:.3f}"
    )

    if "roster_continuity" in era_summary:
        rc = era_summary["roster_continuity"]
        print(f"\nRoster continuity:")
        print(f"  Pre-2019:  {rc['pre_2019_avg']:.3f}")
        print(f"  Post-2019: {rc['post_2019_avg']:.3f}")

    print("\nKey findings:")
    for finding in findings:
        print(f"  - {finding}")

    print(f"\nFull analysis saved to {ARTIFACTS_DIR / 'era_investigation.json'}")


if __name__ == "__main__":
    main()
