"""Generate a concise markdown summary from artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parents[1] / "artifacts"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    comparison = load_json(ARTIFACTS_DIR / "model_comparison.json")
    impact = load_json(ARTIFACTS_DIR / "feature_impact.json")
    season_metrics = load_json(ARTIFACTS_DIR / "playoff_round_metrics.json")

    lines = [
        "# Analysis Summary",
        "",
        "## Key takeaways",
    ]

    takeaways = []
    if comparison.get("results"):
        best = None
        for name, result in comparison["results"].items():
            candidate = (result.get("accuracy", 0.0), name, result)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best:
            accuracy, name, result = best
            within_one = result.get("within_one_round", 0.0)
            takeaways.append(
                f"- Best overall model: `{name}` (accuracy {accuracy:.3f}, within one round {within_one:.3f})."
            )
        exp = comparison["results"].get("experience_only")
        conf = comparison["results"].get("confounders_only")
        if exp and conf:
            delta = exp.get("accuracy", 0.0) - conf.get("accuracy", 0.0)
            takeaways.append(
                f"- Experience-only vs confounders-only accuracy gap: {delta:.3f}."
            )
    if impact.get("impact"):
        top = max(impact["impact"], key=lambda row: row.get("mean_abs_coef", 0.0))
        takeaways.append(
            f"- Top driver by mean |coef|: `{top.get('feature')}` (mean |coef| {top.get('mean_abs_coef', 0.0):.3f})."
        )

    if not takeaways:
        takeaways.append(
            "- Run the analysis scripts to populate this section with results."
        )

    lines.extend(takeaways)
    lines.append("")
    lines.append("## Model comparison (last 10 seasons)")
    lines.append("")

    if comparison.get("results"):
        table_rows = []
        for name, result in comparison["results"].items():
            table_rows.append(
                {
                    "Model": name,
                    "Accuracy": f"{result['accuracy']:.3f}",
                    "Within 1 Round": f"{result['within_one_round']:.3f}",
                    "Features": len(result.get("features", [])),
                }
            )
        table = pd.DataFrame(table_rows).sort_values("Model")
        lines.append(table.to_markdown(index=False))
    else:
        lines.append("No comparison results found. Run `python scripts/model_compare.py`.")

    lines.extend(["", "## Per-season accuracy (last 10 seasons)"])
    summary = season_metrics.get("season_summary")
    if summary:
        rows = [
            {
                "Season": season,
                "Accuracy": f"{values['accuracy']:.3f}",
                "Within 1 Round": f"{values['within_one_round']:.3f}",
                "Support": values.get("support", 0),
            }
            for season, values in sorted(summary.items())
        ]
        table = pd.DataFrame(rows)
        lines.append(table.to_markdown(index=False))

        era_a = [year for year in summary.keys() if 2015 <= int(year) <= 2018]
        era_b = [year for year in summary.keys() if int(year) >= 2019]
        if era_a or era_b:
            lines.extend(["", "## Era segmentation"])
            if era_a:
                avg_a = sum(summary[str(year)]["accuracy"] for year in era_a) / len(era_a)
                within_a = sum(summary[str(year)]["within_one_round"] for year in era_a) / len(era_a)
                lines.append(
                    f"- 2015-2018 avg accuracy {avg_a:.3f}, within one round {within_a:.3f}."
                )
            if era_b:
                avg_b = sum(summary[str(year)]["accuracy"] for year in era_b) / len(era_b)
                within_b = sum(summary[str(year)]["within_one_round"] for year in era_b) / len(era_b)
                lines.append(
                    f"- 2019-2024 avg accuracy {avg_b:.3f}, within one round {within_b:.3f}."
                )
    else:
        lines.append("No per-season summary found. Run `python scripts/train_model.py`.")

    lines.extend(["", "## Feature impact (mean |coef|)"])
    if impact.get("impact"):
        impact_rows = impact["impact"]
        top = pd.DataFrame(impact_rows).sort_values("mean_abs_coef", ascending=False).head(10)
        top["mean_abs_coef"] = top["mean_abs_coef"].map(lambda x: f"{x:.3f}")
        lines.append(top[["feature", "mean_abs_coef"]].to_markdown(index=False))
    else:
        lines.append("No feature impact results found. Run `python scripts/feature_impact.py`.")

    output = "\n".join(lines) + "\n"
    (REPORTS_DIR / "summary.md").write_text(output)
    print(f"Wrote {REPORTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
