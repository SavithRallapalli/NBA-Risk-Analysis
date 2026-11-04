# NBA Player Compensation and On-Court Impact Analysis

## Executive summary

This analysis expands salary prediction into a context-aware screening tool. It combines **4,578 player-season observations** across salary seasons 2015-16 through 2025-26, normalizes salary by the season's salary cap, adds a published adjusted plus-minus impact metric, groups players by statistical role, and estimates a broad prediction range. The latest completed salary season is held out for evaluation.

Rolling time-based validation selected **Random forest**. On the 2025-26 holdout, the model's MAE was **$3.92M** versus **$9.66M** for a training-median baseline (59.4% lower). R² on salary-cap share was **0.844**. The approximate 90% prediction band is **±6.38 percentage points of the cap** (about $9.9M in 2025-26 dollars).

## Holdout model comparison

| Model | MAE | RMSE | MAE (% of cap) | R² (cap share) |
|---|---:|---:|---:|---:|
| Ridge (L2) | $5.19M | $6.90M | 3.35 pp | 0.754 |
| Random forest | $3.92M | $5.49M | 2.53 pp | 0.844 |
| Training median baseline | $9.66M | $15.05M | 6.25 pp | -0.173 |

Of the 434 held-out players, **92.6%** fell inside the approximate band and 32 fell outside. Coverage is close to the nominal 90%, but it is measured on one test season after calibrating on one earlier season; it is not a formal guarantee under future season shifts.

## What the model compares

For each player, the model uses prior-season age and experience, availability, minutes, scoring and efficiency rates, and OpenRAPM offensive, defensive, and overall ratings. It predicts next-season salary as a share of the salary cap, so an identical contract percentage is comparable across seasons with different cap levels. Salary records are summed across team rows for a player-season.

Six unsupervised clusters summarize statistical roles from usage, playmaking, rebounding, shooting efficiency, minutes, and impact rating. Salary comparisons use players in the same role cluster and experience band when the peer sample is large enough; otherwise the comparison falls back to the role cluster or the training-set median.

## Review candidates

A player is flagged only when observed salary falls outside an approximate range calibrated on the preceding salary season. The direction indicates where salary sits relative to the performance-based estimate; it is not a verdict on contract fairness.

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

## Interpretation and next action

The tool narrows the question from ‘Does salary look different from a basic statistical forecast?’ to ‘How unusual is salary among players with a similar statistical profile and experience, and how uncertain is that estimate?’ A next review should check contract years, guaranteed money, options, service-time rules, awards, injuries, and team circumstances using linked primary sources. This repository creates the evidence fields and candidate list; it does not automatically retrieve or verify contract/news context, and no generative-AI conclusion is presented as fact.

OpenRAPM's `wins over starter average` (WOSA) is also included as a separate impact lens. It estimates regular-season wins added over an average starter; it is not converted into dollars here. A salary-to-WOSA ratio is not a true return-on-investment measure without an explicit, validated value-per-win assumption.

## Limits

- The outcome is salary as a share of the cap, not a counterfactual fair-market salary.
- Salary caps, contract status, options, years remaining, team needs, market size, awards, and injury detail are not fully modeled.
- The role groups are statistical clusters, not official positions or scouting archetypes.
- The latest completed season is one temporal holdout; the calibration band comes from one preceding season and is approximate under year-to-year distribution shifts.
- OpenRAPM is a published adjusted plus-minus estimate and carries model uncertainty. Its historical ratings are periodically recomputed by the publisher.
- Salary data is sourced from Basketball-Reference team salary tables via the catalog. Payroll accounting can include salary obligations to former teams after trades/buyouts and can differ from contract cash flow.

## Source notes

- Historical NBA box-score and advanced player stats: NBA Stats, distributed through [nba_data](https://github.com/llimllib/nba_data).
- Player salaries: Basketball-Reference team salary tables, distributed through [nba_data](https://github.com/llimllib/nba_data).
- Impact: [OpenRAPM](https://openrapm.com/), described in the [nba_data dictionary](https://github.com/llimllib/nba_data/blob/main/docs/DATA_DICTIONARY.md#v2-openrapm-data-nbaopenrapm).
- Salary cap: [Mr-Bridge cap and apron dataset](https://huggingface.co/datasets/Mr-Bridge/nba-salary-cap-contracts-2016-2026), cross-checked to NBA official releases per its source notes.
