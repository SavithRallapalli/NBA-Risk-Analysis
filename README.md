# NBA Player Salary Prediction

A compact, reproducible Python project that asks: **how well do a player's previous-season box-score production and availability explain their salary in the following season?** It merges public player-season statistics with salary snapshots, engineers per-game and availability features, compares a median-salary baseline with a regularized regression model, and saves evaluation tables and plots.

The project also includes an expanded historical analysis of salary relative to the cap, prior-season player impact, and statistically similar player cohorts. The original three-season Ridge model remains as a simple baseline; use the expanded analysis for the deeper investigation.

## Data

- Player stats: [BoxScore Lab downloads](https://www.boxscorelab.com/downloads/), which says the figures are sourced from NBA Stats and publishes the files under CC BY 4.0. This project downloads the 2022-23, 2023-24, and 2024-25 player-season CSVs.
- Salaries: [Mr-Bridge NBA Salary Cap & Contracts dataset](https://huggingface.co/datasets/Mr-Bridge/nba-salary-cap-contracts-2016-2026), specifically `player_salaries.csv`. The dataset documents HoopsHype public salary data and notes that these are contract/cap-hit snapshots, not final league accounting.

The model pairs a player's stats from season *t* with their salary in season *t+1*. It therefore uses two training target seasons (2023-24 and 2024-25) and holds out 2025-26 as a later-season test. Player names are normalized for joining; ambiguous duplicates are reduced to one player-season row. This is a small observational dataset, not a causal measure of player value.

## Features and evaluation

Features use prior-season per-game production (points, rebounds, assists, steals, blocks, turnovers, shooting percentages, and minutes per game) plus games played and games-played share. A `StandardScaler` and median imputation are fit inside a scikit-learn pipeline, so test-season data does not inform preprocessing. The model is `Ridge` regression. The baseline predicts the median training salary for every test player. Evaluation reports MAE, RMSE, and R². The project also writes actual-vs-predicted salary, residual, and feature-coefficient plots.

For the practical comparison, the pipeline calculates `recorded salary − model estimate` for each held-out player, exports the 10 largest gaps in each direction, and plots the five largest in each direction. A positive gap means recorded salary is above the model estimate; a negative gap means it is below. These are review candidates, not labels of overpayment or underpayment.

Salary is influenced by contract timing, age, role, team context, awards, injuries, salary-cap growth, and negotiation rules that are not fully represented here. The salary source is a snapshot; name-based joins can miss or mis-match players; and only three preceding-stat seasons are used. Results should be interpreted as a baseline demonstration, not a reliable contract valuation tool.

## Run

Python 3.10+ is recommended.

```bash
cd nba_salary_project
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_project.py
```

Run the expanded historical analysis separately:

```bash
python src/run_value_analysis.py
```

This queries the public [`nba_data` DuckDB catalog](https://github.com/llimllib/nba_data) for historical stats, salaries, and OpenRAPM ratings, then downloads the salary-cap table. It caches the joined data and cap table under `data/raw/`; use `--refresh` to retrieve updated data. Network access is required. The expanded pipeline selects between Ridge and Random Forest using rolling season-based validation, calibrates an approximate prediction range on the previous season, predicts the latest completed salary season, and compares players within learned statistical-role clusters and experience bands.

Raw downloads are stored under `data/raw/`; merged modeling data and outputs are stored under `data/processed/` and `outputs/`. Re-running reuses cached downloads and regenerates results. Pass `--refresh` to fetch the latest source files. Network access is required for the initial download and for `--refresh`.

## Run results

Run on November 4, 2026, using 1,197 joined player-season rows: 783 training rows from salary seasons 2023-24 and 2024-25, and 414 held-out rows from salary season 2025-26. Salary rows are summed across teams for players traded mid-season. The held-out metrics were:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge regression | $5.98M | $8.02M | 0.669 |
| Training median baseline | $9.63M | $15.22M | -0.192 |

These are outputs of this specific historical snapshot and join, not a promise of future performance. The raw salary snapshot was accessed on the run date; upstream data can change, so a fresh download may produce different values.

## Expanded value analysis

The historical analysis uses **4,578 player-season observations** across salary seasons 2015–16 through 2025–26. It normalizes salary by each season's cap, joins on stable player IDs, and includes box-score stats, age, experience, and OpenRAPM impact ratings. On the 2025–26 holdout (434 players), rolling validation selected Random Forest: **$3.92M MAE, $5.49M RMSE, and 0.844 R² on salary-cap share**, compared with **$9.66M MAE and $15.05M RMSE** for the training-median baseline. The approximate 90% interval was calibrated on one prior season; **92.6%** of holdout salaries fell inside it. These figures describe one data snapshot and do not establish fair market value.

Six unsupervised clusters summarize statistical roles; they are not official positions. OpenRAPM WOSA is included as a separate impact lens and is not converted into dollars. The analysis does not claim a causal return on investment.

## Outputs

- `outputs/metrics.csv` — model and median-baseline metrics, with test-season/sample information.
- `outputs/test_predictions.csv` — held-out predictions and residuals.
- `outputs/players_for_review.csv` and `outputs/practical_findings.md` — examples of the largest gaps between observed salary and model estimate, framed as review prompts.
- `outputs/player_salary_gaps.png` — bar chart of the five largest gaps in each direction.
- `outputs/actual_vs_predicted.png`, `outputs/residuals.png`, `outputs/feature_coefficients.png` — diagnostic visuals.
- `outputs/value_analysis/` — historical model comparison, cap-adjusted player review, prediction bands, learned role groups, feature importance, and a copy-ready AI research prompt with model evidence cards for checking player-specific context. The prompt is a research handoff; it does not call a generative AI service.

No model scores are stated here in advance; run the pipeline to generate the results from the downloaded source files.

## Attribution

Please retain attribution to BoxScore Lab, Mr-Bridge, `nba_data`, and OpenRAPM when sharing derived data or outputs. The expanded analysis source notes and data-method details are linked in `outputs/value_analysis/value_analysis.md`.
