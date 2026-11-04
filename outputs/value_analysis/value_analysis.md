# NBA Player Compensation and On-Court Impact Analysis

## Executive summary

I matched **4,578 player-seasons** from 2015–16 through 2025–26. The model uses prior-season stats and impact ratings to estimate next-season salary as a share of the cap. I held out the latest completed season for evaluation.

I compared Ridge and Random Forest using rolling season splits. **Random Forest** had the lower validation error. On the 2025–26 holdout, its MAE was **$3.92M**, versus **$9.66M** for the training-median baseline—a **59.4% reduction**. R² on salary-cap share was **0.844**. The approximate prediction range is **±6.38 percentage points of the cap** (about $9.9M that season).

## Holdout model comparison

| Model | MAE | RMSE | MAE (% of cap) | R² (cap share) |
|---|---:|---:|---:|---:|
| Ridge (L2) | $5.19M | $6.90M | 3.35 pp | 0.754 |
| Random forest | $3.92M | $5.49M | 2.53 pp | 0.844 |
| Training median baseline | $9.66M | $15.05M | 6.25 pp | -0.173 |

The range covered **92.6%** of the 434 holdout players; 32 fell outside it. That is close to the 90% target, but the range is based on one holdout season and one calibration season.

## What went into the model

Inputs include prior-season age, experience, games played, minutes, scoring and shooting rates, and OpenRAPM impact ratings. The target is next-season salary as a share of the cap. Salary entries from multiple teams are added together for traded players.

K-means groups players into six statistical profiles using usage, assists, rebounding, shooting efficiency, minutes, and impact. Each player is compared with the same profile and experience band when there are enough peers. Smaller groups fall back to a broader comparison.

## Review candidates

A player is flagged when recorded salary falls outside the approximate range calibrated on the previous season. The flag shows how salary compares with the model; it does not say whether a contract is fair.

| Player | Salary season | Salary | Model estimate | Peer median | Prior WOSA | Interval flag |
|---|---:|---:|---:|---:|---:|---|
| Khris Middleton | 2025-26 | $33.3M | $11.5M | 8.2% cap | -0.7 | above approximate range |
| Lauri Markkanen | 2025-26 | $46.4M | $25.8M | 10.5% cap | -0.5 | above approximate range |
| Paul George | 2025-26 | $51.7M | $35.5M | 31.5% cap | -0.2 | above approximate range |
| Terry Rozier | 2025-26 | $26.6M | $12.8M | 8.2% cap | -4.2 | above approximate range |
| Jrue Holiday | 2025-26 | $32.4M | $19.2M | 11.9% cap | -0.8 | above approximate range |
| Trae Young | 2025-26 | $46.4M | $34.6M | 24.3% cap | 3.1 | above approximate range |
| Jalen Suggs | 2025-26 | $35.0M | $23.3M | 9.7% cap | -0.1 | above approximate range |
| Marcus Smart | 2025-26 | $19.9M | $8.3M | 8.2% cap | -0.1 | above approximate range |

## What to do with the flags

Use the list to decide what to investigate next. Check contract length, guaranteed money, options, years of service, awards, injuries, and team circumstances. The report does not fetch or verify those details. The AI research brief gives a web-enabled assistant the model results and asks it to cite its sources.

OpenRAPM's `wins over starter average` (WOSA) is shown as a separate impact measure. It estimates regular-season wins above an average starter. I have not converted wins into dollars, so the report does not claim a financial return on salary.

## Limits to keep in mind

- The model estimates salary as a share of the cap; it does not estimate a fair-market price.
- It does not include contract options, guaranteed money, years remaining, team needs, awards, or detailed injury history.
- The six groups describe statistical profiles; they are not official positions or scouting labels.
- The prediction range was calibrated on one season. Results may shift as the league and salary cap change.
- OpenRAPM is an estimate, and its historical ratings can be updated by the publisher.
- Salary records can include amounts owed by multiple teams after a trade or buyout. They may differ from a player's contract cash flow.

## Source notes

- Historical NBA box-score and advanced player stats: NBA Stats, distributed through [nba_data](https://github.com/llimllib/nba_data).
- Player salaries: Basketball-Reference team salary tables, distributed through [nba_data](https://github.com/llimllib/nba_data).
- Impact: [OpenRAPM](https://openrapm.com/), described in the [nba_data dictionary](https://github.com/llimllib/nba_data/blob/main/docs/DATA_DICTIONARY.md#v2-openrapm-data-nbaopenrapm).
- Salary cap: [Mr-Bridge cap and apron dataset](https://huggingface.co/datasets/Mr-Bridge/nba-salary-cap-contracts-2016-2026), cross-checked to NBA official releases per its source notes.
