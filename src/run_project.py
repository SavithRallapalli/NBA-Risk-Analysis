from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
STATS_URL = "https://www.boxscorelab.com/downloads/{season}-player-season-totals.csv"
SALARY_URL = "https://huggingface.co/datasets/Mr-Bridge/nba-salary-cap-contracts-2016-2026/resolve/main/player_salaries.csv"
STATS_SEASONS = ["2022-23", "2023-24", "2024-25"]


def download(url: str, path: Path, refresh: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size and not refresh:
        return
    response = requests.get(url, timeout=(15, 90), headers={"User-Agent": "Mozilla/5.0 nba-salary-prediction-project/1.0"})
    response.raise_for_status()
    path.write_bytes(response.content)


def player_key(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = re.sub(r"[^a-z0-9]+", "", value.lower())
    return value


def fetch_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    RAW.mkdir(parents=True, exist_ok=True)
    refresh = "--refresh" in sys.argv
    salary_path = RAW / "player_salaries.csv"
    download(SALARY_URL, salary_path, refresh)
    salaries = pd.read_csv(salary_path)
    stats_frames = []
    for season in STATS_SEASONS:
        path = RAW / f"{season}-player-season-totals.csv"
        download(STATS_URL.format(season=season), path, refresh)
        frame = pd.read_csv(path)
        frame["season"] = season
        stats_frames.append(frame)
    return pd.concat(stats_frames, ignore_index=True), salaries


def make_model_data(stats: pd.DataFrame, salaries: pd.DataFrame) -> pd.DataFrame:
    stats = stats.copy()
    salaries = salaries.copy()
    stats["player_key"] = stats.player_name.map(player_key)
    salaries["player_key"] = salaries.player.map(player_key)
    stats["season_start"] = stats.season.str[:4].astype(int)
    salaries["season_start"] = salaries.season.str[:4].astype(int)

    # Keep one row if a player appears more than once in the stats file.
    stats = stats.sort_values(["games_played", "minutes"], ascending=False).drop_duplicates(
        ["player_key", "season"], keep="first"
    )
    stats["minutes_per_game"] = stats.minutes / stats.games_played.replace(0, np.nan)
    stats["games_played_share"] = stats.games_played / 82.0
    features = ["points_pg", "rebounds_pg", "assists_pg", "steals", "blocks", "turnovers",
                "field_goal_pct", "three_point_pct", "free_throw_pct", "minutes_per_game",
                "games_played", "games_played_share"]
    # Convert these season totals to per-game rates before fitting the model.
    for col in ["steals", "blocks", "turnovers"]:
        stats[col] = stats[col] / stats.games_played.replace(0, np.nan)
    stats["target_season_start"] = stats.season_start + 1
    salary = salaries.copy()
    for flag in ["terminated", "two_way"]:
        if flag in salary.columns:
            salary = salary[~salary[flag].astype(str).str.lower().isin(["true", "1"])]
    # Add team salary rows together so a trade does not create duplicate players.
    salary = (salary.groupby(["player_key", "season_start", "season"], as_index=False)
              .agg(salary=("salary", "sum"), player=("player", "first")))
    merged = stats.merge(salary, left_on=["player_key", "target_season_start"],
                         right_on=["player_key", "season_start"], how="inner",
                         suffixes=("_stats", "_salary"))
    merged = merged.rename(columns={"season_salary": "target_season", "player_name": "player_name"})
    merged = merged.dropna(subset=["salary"])
    merged = merged[(merged.salary > 0) & merged[features].notna().any(axis=1)].copy()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    merged.to_csv(PROCESSED / "model_dataset.csv", index=False)
    return merged


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stats, salaries = fetch_data()
    data = make_model_data(stats, salaries)
    feature_cols = ["points_pg", "rebounds_pg", "assists_pg", "steals", "blocks", "turnovers",
                    "field_goal_pct", "three_point_pct", "free_throw_pct", "minutes_per_game",
                    "games_played", "games_played_share"]
    # Train on earlier salary seasons and hold out the latest complete season.
    test_season = int(data.target_season_start.max())
    train = data[data.target_season_start < test_season]
    test = data[data.target_season_start == test_season]
    if train.empty or test.empty:
        raise ValueError("Need at least two salary seasons after joining prior-season stats.")
    X_train, y_train = train[feature_cols], train.salary.astype(float)
    X_test, y_test = test[feature_cols], test.salary.astype(float)
    model = Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    baseline_pred = np.repeat(float(y_train.median()), len(y_test))

    rows = []
    for label, prediction in [("Ridge regression", pred), ("Training median baseline", baseline_pred)]:
        rows.append({"model": label, "test_salary_season": test_season,
                     "train_rows": len(train), "test_rows": len(test),
                     "MAE": mean_absolute_error(y_test, prediction),
                     "RMSE": mean_squared_error(y_test, prediction) ** 0.5,
                     "R2": r2_score(y_test, prediction)})
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    result = test[["player", "target_season", "salary"]].copy()
    result["predicted_salary"] = pred
    result["baseline_salary"] = baseline_pred
    result["residual"] = result.salary - result.predicted_salary
    result.to_csv(OUT / "test_predictions.csv", index=False)

    # Large gaps are worth checking; they do not prove a contract is mispriced.
    result["gap"] = result.salary - result.predicted_salary
    review = pd.concat([result.nlargest(10, "gap"), result.nsmallest(10, "gap")])
    review["comparison"] = np.where(review.gap > 0, "Salary above model estimate", "Salary below model estimate")
    review[["comparison", "player", "target_season", "salary", "predicted_salary", "gap"]].to_csv(
        OUT / "players_for_review.csv", index=False)
    # Show the five largest gaps above and below the model estimate.
    plot_rows = pd.concat([result.nlargest(5, "gap"), result.nsmallest(5, "gap")]).copy()
    plot_rows = plot_rows.sort_values("gap")
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#dc2626" if value > 0 else "#2563eb" for value in plot_rows.gap]
    ax.barh(plot_rows.player, plot_rows.gap / 1_000_000, color=colors)
    ax.axvline(0, color="#555", linewidth=1)
    ax.set(xlabel="Recorded salary minus model estimate ($ millions)",
           title="Largest held-out salary gaps: review candidates")
    fig.tight_layout(); fig.savefig(OUT / "player_salary_gaps.png", dpi=160); plt.close(fig)
    above = result.nlargest(5, "gap")
    below = result.nsmallest(5, "gap")
    def money(value: float) -> str:
        return f"${value / 1_000_000:.1f}M"
    lines = [
        "# Three-season baseline: player salary comparisons",
        "",
        "This table compares the model with recorded salaries in the held-out 2025–26 season. A positive gap means salary was higher than the estimate; a negative gap means it was lower. The model uses prior-season box-score stats and availability, so the gaps point to questions for further review—not conclusions about contract value.",
        "",
        "## Salary above model estimate",
        "",
        "| Player | Recorded salary | Model estimate | Difference |",
        "|---|---:|---:|---:|",
    ]
    lines += [f"| {r.player} | {money(r.salary)} | {money(r.predicted_salary)} | +{money(r.gap)} |" for r in above.itertuples()]
    lines += ["", "## Salary below model estimate", "", "| Player | Recorded salary | Model estimate | Difference |", "|---|---:|---:|---:|"]
    lines += [f"| {r.player} | {money(r.salary)} | {money(r.predicted_salary)} | −{money(abs(r.gap))} |" for r in below.itertuples()]
    lines += [
        "",
        "## How to use this shortlist",
        "",
        "Some established players had salaries well above the estimate; some younger players had salaries well below it despite strong prior stats. Contract status, years of service, extensions, team context, injuries, awards, role, and salary-cap rules may explain the gaps. This model does not account for those factors. The next step is to add them and compare players with similar contracts and experience.",
        "",
        "Some traded players have salary records for more than one team. The pipeline adds those amounts together before making the comparison.",
    ]
    (OUT / "practical_findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_test, pred, alpha=.65, color="#2563eb")
    lim = [0, max(float(y_test.max()), float(np.max(pred))) * 1.05]
    ax.plot(lim, lim, "--", color="#555", linewidth=1)
    ax.set(xlim=lim, ylim=lim, xlabel="Actual salary ($)", ylabel="Predicted salary ($)", title=f"Held-out {test_season}-{str(test_season+1)[-2:]} salaries")
    fig.tight_layout(); fig.savefig(OUT / "actual_vs_predicted.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(pred, result.residual, alpha=.65, color="#0f766e")
    ax.axhline(0, color="#555", linestyle="--", linewidth=1)
    ax.set(xlabel="Predicted salary ($)", ylabel="Residual: actual − predicted ($)", title="Held-out residuals")
    fig.tight_layout(); fig.savefig(OUT / "residuals.png", dpi=160); plt.close(fig)

    coef = pd.Series(model.named_steps["ridge"].coef_, index=feature_cols).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    coef.plot.barh(ax=ax, color=["#dc2626" if x < 0 else "#2563eb" for x in coef])
    ax.set(xlabel="Ridge coefficient (salary dollars per standardized feature)", title="Model coefficients")
    fig.tight_layout(); fig.savefig(OUT / "feature_coefficients.png", dpi=160); plt.close(fig)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nMatched player-seasons: {len(data)}; training seasons: {sorted(train.target_season.unique())}; holdout: {test_season}")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
