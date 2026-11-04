# NBA-Risk-Analysis

An NBA project on player salaries, roster construction, and team value.

The repo started with a 2022 salary-versus-performance analysis. I’ve kept those notebooks and added a second project that uses several seasons of data to estimate player salaries one season ahead.

## Season-ahead salary model

The main question is simple: **how much of a player's next-season salary can we explain with their recent performance, availability, age, experience, and estimated on-court impact?**

The model predicts salary as a share of the salary cap, which makes comparisons across seasons more useful. It also groups players with similar statistical profiles and experience, then flags salaries outside an approximate prediction range. A flag means “take a closer look”; it does not mean a player is overpaid or underpaid.

### Results

The dataset has **4,578 player-season records** from salary seasons 2015–16 through 2025–26. I used earlier seasons for rolling, time-based validation and held out 2025–26 (434 players) for the final comparison. Random Forest performed best in validation.

| 2025–26 holdout | Random Forest | Median-salary baseline |
|---|---:|---:|
| MAE | $3.92M | $9.66M |
| RMSE | $5.49M | $15.05M |
| R² on salary-cap share | 0.844 | -0.173 |

The model’s average absolute error was **59% lower** than the baseline. **92.6%** of held-out salaries fell within the approximate range, calibrated on the previous season. That range is useful for screening, but one test season cannot establish whether a contract is fair.

### Run it

From the repository root, create an environment and install the packages:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_value_analysis.py
```

The first run needs internet access to retrieve the historical stats, salary, impact, and salary-cap data. Downloads are cached under `data/raw/`; use `--refresh` to update them. Run the original three-season Ridge model with `python src/run_project.py`.

### Useful files

- `outputs/final_consulting_writeup.md` — short summary of the findings.
- `outputs/value_analysis/value_analysis.md` — method, detailed results, examples, and limitations.
- `outputs/value_analysis/holdout_metrics.csv` — comparison of the tested models.
- `outputs/value_analysis/player_value_review.csv` — salary estimates, peer comparisons, and uncertainty ranges for the test season.
- `outputs/value_analysis/review_candidates.csv` — players outside the approximate range.
- `outputs/value_analysis/role_clusters.csv` and `feature_importance.csv` — player groups and model features.
- `outputs/value_analysis/ai_research_brief.md` — model evidence and a prompt for checking player context with linked sources. It does not call an AI service.

## Original 2022 analysis

The original notebooks estimate a simple performance score, adjust it for games played, and compare it with salary. The score weights are exploratory choices, not official measures of player value.

- `notebooks/01_data_collection.ipynb` downloads season stats.
- `notebooks/02_exploratory_analysis.ipynb` builds the availability-adjusted score.
- `notebooks/03_salary_data.ipynb` joins salary and stats, then ranks players by the score and runs a simple regression.
- `data/merged_nba_salary_stats_2022.csv` and `data/nba_contract_valuation_2022.csv` contain the original analysis data.

## Data and limitations

The expanded model uses NBA player stats and Basketball-Reference salary records distributed through [`nba_data`](https://github.com/llimllib/nba_data), OpenRAPM impact ratings, and salary-cap figures from the [Mr-Bridge dataset](https://huggingface.co/datasets/Mr-Bridge/nba-salary-cap-contracts-2016-2026). See the [data dictionary](https://github.com/llimllib/nba_data/blob/main/docs/DATA_DICTIONARY.md) and detailed report for source notes.

The model does not include every factor that affects a contract, such as options, guaranteed money, years remaining, awards, injuries, or team needs. Its salary estimate is a statistical benchmark, not a counterfactual market price. Keep source attribution when sharing the data or results.
