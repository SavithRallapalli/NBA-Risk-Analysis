# AI-assisted context research brief

This file contains model outputs only. Use it as input to a web-enabled research assistant to find possible explanations for salary gaps. The model did not retrieve contract news or decide that any salary is fair or unfair.

## Research prompt

For each player below, investigate public context that could explain why recorded 2025-26 salary differs from this project's estimate. Search for and cite direct, reliable sources: NBA/team announcements, the NBA/NBPA CBA or official salary-cap material, and reputable reporting for contract details, options, extensions, awards, availability, role, and team changes. Distinguish verified facts from hypotheses. Explain what the model omitted that could account for the gap. Do not call a player overpaid/underpaid or infer fair value. If a fact cannot be verified with a link, say so. Keep the answer to a short evidence table with columns: player, verified context, source link, how it could relate to the model gap, confidence/unknowns.

## Model evidence cards

Interpret gap as recorded salary minus model estimate. Positive means recorded salary is above the estimate; negative means below. The estimate predicts salary share of cap from prior-season stats, an impact rating, age, and experience. The interval is calibrated on one earlier season and is approximate.

| Player | Salary | Model estimate | Gap | Approximate model range | Peer median share | Prior-season WOSA | Prior RAPM | Flag |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Cam Thomas | $6.8M | $34.3M | $-27.5M | $24.5M–$44.2M | 9.7% cap | -0.39 | 0.02 | below approximate range |
| Khris Middleton | $33.3M | $11.5M | $+21.8M | $1.6M–$21.4M | 8.2% cap | -0.65 | -0.17 | above approximate range |
| Lauri Markkanen | $46.4M | $25.8M | $+20.6M | $15.9M–$35.7M | 10.5% cap | -0.52 | 0.39 | above approximate range |
| Ivica Zubac | $18.9M | $37.1M | $-18.2M | $27.2M–$46.9M | 6.0% cap | 4.29 | 3.14 | below approximate range |
| Austin Reaves | $13.9M | $31.4M | $-17.5M | $21.6M–$41.3M | 9.7% cap | 2.13 | 1.47 | below approximate range |
| Kelly Oubre Jr. | $8.4M | $25.4M | $-17.0M | $15.6M–$35.3M | 11.9% cap | -1.76 | -0.37 | below approximate range |
| Russell Westbrook | $2.3M | $18.6M | $-16.3M | $8.8M–$28.5M | 8.2% cap | -2.83 | -1.68 | below approximate range |
| Paul George | $51.7M | $35.5M | $+16.1M | $25.7M–$45.4M | 31.5% cap | -0.18 | 0.66 | above approximate range |
