# NBA Player Compensation and On-Court Impact
## Practical findings

### Summary

This project estimates a player's next-season salary from prior-season performance, availability, age, experience, and an adjusted plus-minus rating. Salary is measured as a share of the cap, so amounts from different seasons are easier to compare.

The dataset contains **4,578 player-seasons** from 2015–16 through 2025–26. On the 2025–26 holdout of **434 players**, the Random Forest model had **$3.92M MAE** and **0.844 R² on salary-cap share**. Its MAE was **59% lower** than the $9.66M training-median baseline. A prediction range calibrated on the prior season contained **92.6%** of the holdout salaries; 32 players fell outside it.

### Model comparison

| Model | MAE | RMSE | MAE as cap share | R² on cap share |
|---|---:|---:|---:|---:|
| Random Forest | $3.92M | $5.49M | 2.53 percentage points | 0.844 |
| Ridge regression | $5.19M | $6.90M | 3.35 percentage points | 0.754 |
| Median baseline | $9.66M | $15.05M | 6.25 percentage points | −0.173 |

Random Forest was selected using rolling, season-based validation. The final results come from a separate later-season holdout.

### What the flags show

The review list includes players whose recorded salary falls outside the approximate range. For example, **Paul George** and **Khris Middleton** were above the model range; **Cam Thomas** and **Ivica Zubac** were below it. The model also compares each player with peers from a similar statistical group and experience band.

These are cases to investigate, not overpaid or underpaid labels. A salary gap could reflect contract length, options, guaranteed money, service time, awards, injuries, or team needs. The model does not fully account for those factors.

![Players outside the approximate model range](value_analysis/review_candidate_gaps.png)

### How to use the results

Use the model to narrow the review list, then check the contract and player context. OpenRAPM's wins-over-starter-average measure provides a separate view of prior-season impact; this project does not convert that measure into dollars or claim a return on salary.

The repository includes a copy-ready prompt for a web-enabled AI assistant to research contract context and cite its sources. It does not make those searches or generate a player judgment automatically.

### Limits

- Salary prediction is not the same as estimating a fair-market contract.
- The learned statistical groups are not official positions or scouting labels.
- The prediction range uses one calibration season, so it may not hold after league-wide changes.
- Salary records may include obligations across multiple teams after a trade or buyout.
- OpenRAPM and the model both contain estimation uncertainty.

**Takeaway:** the model is useful for screening and peer comparison. Use it to decide what to investigate next, not as a standalone contract recommendation.
