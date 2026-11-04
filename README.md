# NBA-Risk-Analysis

Risk management techniques on NBA roster construction and team valuation. This repository contains an earlier 2022 salary/contract analysis and an expanded, season-ahead player compensation and on-court impact project.

## Expanded player compensation model

The expanded analysis asks whether a player's next-season salary looks unusual given prior-season production, availability, age, experience, and an adjusted plus-minus impact estimate. It compares salary as a share of the cap, forms statistical-role cohorts, and flags large gaps with an approximate prediction range. These are prompts for review, not conclusions that a contract is fair or unfair.

### Results

The historical dataset contains **4,578 player-season observations** spanning salary seasons 2015–16 through 2025–26. Rolling season-based validation selected a Random Forest model. On the 2025–26 holdout of 434 players, it achieved **$3.92M MAE**, **$5.49M RMSE**, and **0.844 R² on salary-cap share**, compared with **$9.66M MAE** for the training-median baseline. An approximate interval calibrated on one preceding season contained **92.6%** of the holdout salaries. Results describe this data snapshot; they do not establish fair market value.

Player statistics and salaries are joined using stable player IDs. Salary is normalized by the season's cap. The analysis uses prior-season NBA stats and OpenRAPM impact ratings, and groups players by learned statistical profiles and experience bands. The historical catalog and field definitions are documented by [`nba_data`](https://github.com/llimllib/nba_data) and its [data dictionary](https://github.com/llimllib/nba_data/blob/main/docs/DATA_DICTIONARY.md).

### Run the expanded analysis

Python 3.10+ is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/run_value_analysis.py
```

The first run needs network access to query the public historical NBA catalog and retrieve salary-cap data. It caches downloaded source data under `data/raw/`; pass `--refresh` to retrieve an updated snapshot. The original three-season Ridge baseline can be run with `python src/run_project.py`.

### Key outputs

- `outputs/value_analysis/value_analysis.md` — detailed methodology, findings, examples, and limitations.
- `outputs/value_analysis/holdout_metrics.csv` — model comparison on the latest completed season.
- `outputs/value_analysis/player_value_review.csv` and `review_candidates.csv` — player-level comparisons and interval flags.
- `outputs/value_analysis/role_clusters.csv` and `feature_importance.csv` — learned cohorts and model features.
- `outputs/value_analysis/ai_research_brief.md` — model evidence cards and a copy-ready prompt for source-backed context research. It does not call an AI service automatically.
- `outputs/final_consulting_writeup.md` — concise consulting-style summary.

## Original NBA risk analysis materials

The repository also retains the original 2022 contract-valuation datasets and notebooks:

- `data/merged_nba_salary_stats_2022.csv`
- `data/nba_contract_valuation_2022.csv`
- `notebooks/01_data_collection.ipynb`
- `notebooks/02_exploratory_analysis.ipynb`
- `notebooks/03_salary_data.ipynb`

## Data, attribution, and limitations

The expanded model draws historical box-score and salary data from `nba_data`, impact ratings from OpenRAPM, and salary-cap figures from the Mr-Bridge cap dataset. See the detailed report for exact source notes and limitations. The model omits many contract and team factors, including options, guaranteed money, years remaining, injuries, awards, and team needs. Its prediction range is approximate and calibrated on one season; a salary gap is not proof of overpayment or underpayment.

Please retain attribution to the data publishers when sharing derived outputs.
