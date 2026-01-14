# Repository Guidelines

## Overview
This repo builds an NBA playoff experience analysis pipeline plus a FastAPI + React web app. The data pipeline pulls from `swar/nba_api` with Basketball-Reference fallbacks and produces team-level features for modeling.

## Project Structure & Module Organization
- `backend/` FastAPI service for predictions and data lookups.
- `frontend/` React + Vite UI for team search and visual summaries.
- `scripts/` ETL and feature build utilities.
- `data/` local-only storage (`raw/` for pulls, `processed/` for features).

## Build, Test, and Development Commands
- `python -m pip install -r backend/requirements.txt` to set up backend deps.
- `uvicorn app.main:app --reload` (from `backend/`) to run the API.
- `npm install` then `npm run dev` (from `frontend/`) to run the UI.
- `python scripts/etl_pull.py --start-year 1996` to pull raw data.
- `python scripts/feature_build.py` to generate `data/processed/team_features.csv`.
- `python scripts/validate_data.py` to verify raw/processed data completeness.
- `python scripts/series_sanity.py` to summarize playoff series coverage and rounds.
- `python scripts/train_model.py` to train the playoff round model.
- `python scripts/score_model.py --team-id 1610612738 --season 2023-24` to score a team-season.
- `python scripts/model_compare.py` to compare experience vs confounder vs full models.
- `python scripts/feature_impact.py` to report model feature impact.
- `python scripts/report_summary.py` to generate `reports/summary.md`.

## Coding Style & Naming Conventions
- Python: 4-space indentation, snake_case for files/functions.
- JavaScript/React: 2-space indentation, PascalCase for components.
- Keep filenames descriptive (e.g., `feature_build.py`, `teams.py`).

## Testing Guidelines
- No tests are set up yet. When added, prefer `pytest` for backend and `vitest` or `jest` for frontend.
- Name tests `test_*.py` and keep them near the related module.

## Commit & Pull Request Guidelines
- Use clear, scoped commits like `pipeline: add season feature build`.
- PRs should include:
  - A concise summary and rationale.
  - Links to issues/tasks when relevant.
  - Screenshots for UI changes or sample outputs for data changes.

## Configuration & Security Notes
- Avoid committing raw data or large processed files; keep them in `data/` and rely on `.gitignore`.
- Document API keys or external data requirements in `README.md` when introduced.
