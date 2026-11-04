# Practical player comparisons

These comparisons use the held-out 2025-26 salary season. A positive gap means recorded salary was above the model estimate; a negative gap means it was below. The model only uses prior-season box-score performance and availability. These are prompts for investigation, not conclusions about player value or whether a contract is fair.

## Salary above model estimate

| Player | Recorded salary | Model estimate | Difference |
|---|---:|---:|---:|
| Paul George | $51.7M | $20.3M | +$31.3M |
| Stephen Curry | $59.6M | $30.5M | +$29.1M |
| Jaylen Brown | $53.1M | $27.2M | +$26.0M |
| Lauri Markkanen | $46.4M | $21.9M | +$24.5M |
| Joel Embiid | $55.2M | $31.8M | +$23.4M |

## Salary below model estimate

| Player | Recorded salary | Model estimate | Difference |
|---|---:|---:|---:|
| Victor Wembanyama | $13.4M | $36.0M | −$22.6M |
| Cam Thomas | $6.9M | $28.7M | −$21.8M |
| Jalen Williams | $6.6M | $27.6M | −$21.1M |
| Paolo Banchero | $15.3M | $33.4M | −$18.0M |
| Keyonte George | $4.3M | $20.0M | −$15.7M |

## How to use this shortlist

The largest positive gaps include established high-salary players whose contract levels are not fully explained by these box-score inputs. The largest negative gaps include several younger players on comparatively low recorded salaries despite strong prior-season statistical signals. Contract status, years of service, extensions, team context, injuries, awards, role, and salary-cap rules can explain much of this difference; they are not adequately controlled for in this model. A sensible next step is to add those variables and compare players within similar experience and contract groups before drawing any business conclusion.

The 2025-26 salary source can contain more than one team row for a player traded during the season. The pipeline sums those team amounts before comparing a player-season, avoiding duplicate entries and partial salary comparisons.
