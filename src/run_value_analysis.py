"""Historical, salary-cap-normalized player compensation/value analysis."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "value_analysis"
CATALOG_URL = "https://basketball-data.billmill.org/nba/nba.duckdb"
CAP_URL = "https://huggingface.co/datasets/Mr-Bridge/nba-salary-cap-contracts-2016-2026/resolve/main/cap_tax_aprons.csv"
JOINED_PATH = RAW / "historical_value_dataset.csv"
CAP_PATH = RAW / "cap_tax_aprons.csv"
FEATURES = [
    "age", "experience_seasons", "games_played", "minutes_per_game",
    "points_per_game", "rebounds_per_game", "assists_per_game", "turnovers_per_game",
    "steals_per_game", "blocks_per_game", "true_shooting_pct", "usage_pct",
    "assist_pct", "rebound_pct", "net_rating", "rapm_rating", "rapm_offense",
    "rapm_defense", "wins_over_starter_average", "rapm_possessions",
]
ROLE_FEATURES = ["usage_pct", "assist_pct", "rebound_pct", "true_shooting_pct", "minutes_per_game", "rapm_rating"]
CAT_FEATURES = ["role_cluster", "experience_band"]


def fetch_source_data(refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    RAW.mkdir(parents=True, exist_ok=True)
    if refresh or not JOINED_PATH.exists():
        con = duckdb.connect()
        con.execute(f"ATTACH '{CATALOG_URL}' AS nba")
        query = """
        WITH salary AS (
            SELECT season, player_id, SUM(salary)::DOUBLE AS salary
            FROM nba.bbref_salaries
            WHERE season BETWEEN 2016 AND 2026 AND player_id IS NOT NULL AND salary > 0
            GROUP BY season, player_id
        )
        SELECT
            s.season AS stats_season,
            sal.season AS target_season,
            p.player_id,
            p.name AS player,
            p.first_season,
            s.age,
            s.gp AS games_played,
            s.min AS minutes_per_game,
            s.pts AS points_per_game,
            s.reb AS rebounds_per_game,
            s.ast AS assists_per_game,
            s.tov AS turnovers_per_game,
            s.stl AS steals_per_game,
            s.blk AS blocks_per_game,
            s.ts_pct AS true_shooting_pct,
            s.usg_pct AS usage_pct,
            s.ast_pct AS assist_pct,
            s.reb_pct AS rebound_pct,
            s.net_rating,
            r.rating AS rapm_rating,
            r.o_rating AS rapm_offense,
            r.d_rating AS rapm_defense,
            r.wosa AS wins_over_starter_average,
            r.possessions AS rapm_possessions,
            sal.salary
        FROM nba.player_season_stats_per_game s
        JOIN nba.players p USING (player_id)
        LEFT JOIN nba.openrapm r ON r.player_id = s.player_id AND r.season = s.season
        JOIN salary sal ON sal.player_id = s.player_id AND sal.season = s.season + 1
        WHERE s.season_type = 'regular_season'
          AND s.season BETWEEN 2015 AND 2025
          AND s.gp > 0
        """
        data = con.sql(query).df()
        con.close()
        if data.empty:
            raise RuntimeError("No rows returned from the historical NBA data catalog.")
        data["experience_seasons"] = (data.stats_season - data.first_season).clip(lower=0)
        data.to_csv(JOINED_PATH, index=False)
    else:
        data = pd.read_csv(JOINED_PATH)

    if refresh or not CAP_PATH.exists():
        response = requests.get(CAP_URL, timeout=(15, 90), headers={"User-Agent": "Mozilla/5.0 nba-value-analysis/1.0"})
        response.raise_for_status()
        CAP_PATH.write_bytes(response.content)
    cap = pd.read_csv(CAP_PATH)
    cap = cap[["season_year", "season", "salary_cap"]].copy()
    # Salary file uses NBA season end year; cap file keys by season start year.
    cap["target_season"] = cap.season_year + 1
    cap["salary_cap"] = pd.to_numeric(cap.salary_cap, errors="coerce")
    data = data.merge(cap[["target_season", "salary_cap"]], on="target_season", how="inner")
    data = data.dropna(subset=["salary", "salary_cap"])
    # 2026-27 is in progress as of this project's data snapshot; test on the
    # latest completed salary season rather than partial current-season stats.
    data = data[data.target_season <= 2026].copy()
    data["salary_cap_share"] = data.salary / data.salary_cap
    data = data[data.salary_cap_share.between(0, 1)].copy()
    return data, cap


def add_experience_band(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["experience_band"] = pd.cut(
        out.experience_seasons, bins=[-1, 3, 7, np.inf],
        labels=["0-3 years", "4-7 years", "8+ years"],
    ).astype("object")
    return out


def role_model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clusters", KMeans(n_clusters=6, random_state=42, n_init=20)),
    ])


def attach_roles(train: pd.DataFrame, apply: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Pipeline]:
    clusterer = role_model()
    clusterer.fit(train[ROLE_FEATURES])
    train_out, apply_out = add_experience_band(train), add_experience_band(apply)
    train_out["role_cluster"] = clusterer.predict(train[ROLE_FEATURES]).astype(str)
    apply_out["role_cluster"] = clusterer.predict(apply[ROLE_FEATURES]).astype(str)
    return train_out, apply_out, clusterer


def make_model(name: str) -> Pipeline:
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipe, FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ])
    estimator = Ridge(alpha=25.0) if name == "Ridge (L2)" else RandomForestRegressor(
        n_estimators=350, min_samples_leaf=8, max_features=0.8,
        random_state=42, n_jobs=1,
    )
    return Pipeline([("features", preprocess), ("model", estimator)])


def time_series_model_selection(data: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    seasons = np.array(sorted(data.target_season.unique()))
    splitter = TimeSeriesSplit(n_splits=4)
    rows = []
    for name in ["Ridge (L2)", "Random forest"]:
        fold_errors = []
        for train_idx, valid_idx in splitter.split(seasons):
            train_seasons, valid_seasons = set(seasons[train_idx]), set(seasons[valid_idx])
            tr, va = data[data.target_season.isin(train_seasons)], data[data.target_season.isin(valid_seasons)]
            if tr.empty or va.empty:
                continue
            tr_r, va_r, _ = attach_roles(tr, va)
            model = make_model(name)
            model.fit(tr_r[FEATURES + CAT_FEATURES], tr_r.salary_cap_share)
            fold_errors.append(mean_absolute_error(va_r.salary_cap_share, model.predict(va_r[FEATURES + CAT_FEATURES])))
        rows.append({"model": name, "rolling_cv_folds": len(fold_errors),
                     "mean_cv_MAE_cap_share": float(np.mean(fold_errors)) if fold_errors else np.nan})
    cv = pd.DataFrame(rows).sort_values("mean_cv_MAE_cap_share")
    return str(cv.iloc[0].model), cv


def metrics(y_share: pd.Series, pred_share: np.ndarray, cap: pd.Series, model: str, train_rows: int) -> dict:
    actual_dollars = y_share.to_numpy() * cap.to_numpy()
    pred_dollars = pred_share * cap.to_numpy()
    return {
        "model": model,
        "train_rows": train_rows,
        "test_rows": len(y_share),
        "MAE_salary_dollars": mean_absolute_error(actual_dollars, pred_dollars),
        "RMSE_salary_dollars": mean_squared_error(actual_dollars, pred_dollars) ** 0.5,
        "MAE_cap_share_percentage_points": 100 * mean_absolute_error(y_share, pred_share),
        "R2_cap_share": r2_score(y_share, pred_share),
    }


def cohort_median(train: pd.DataFrame, row: pd.Series) -> tuple[float, int, str]:
    cohort = train[(train.role_cluster == row.role_cluster) & (train.experience_band == row.experience_band)]
    label = f"role {row.role_cluster}, {row.experience_band}"
    if len(cohort) < 20:
        cohort = train[train.role_cluster == row.role_cluster]
        label = f"role {row.role_cluster} (all experience bands)"
    if len(cohort) < 20:
        return float(train.salary_cap_share.median()), len(train), "all training players"
    return float(cohort.salary_cap_share.median()), len(cohort), label


def cluster_descriptions(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, g in train.groupby("role_cluster"):
        means = g[ROLE_FEATURES].median(numeric_only=True)
        desc = []
        for feature, label in [("usage_pct", "usage"), ("assist_pct", "playmaking"), ("rebound_pct", "rebounding"), ("true_shooting_pct", "shooting efficiency")]:
            # Use within-training quartiles to describe a cluster's profile plainly.
            q75 = train[feature].quantile(.75)
            q25 = train[feature].quantile(.25)
            if means[feature] >= q75:
                desc.append(f"high {label}")
            elif means[feature] <= q25:
                desc.append(f"lower {label}")
        rows.append({"role_cluster": str(cluster_id), "players": len(g),
                     "cluster_description": ", ".join(desc) or "mixed statistical profile",
                     **{f"median_{k}": v for k, v in means.items()}})
    return pd.DataFrame(rows).sort_values("role_cluster")


def write_report(metrics_df: pd.DataFrame, selected: str, cv: pd.DataFrame, reviews: pd.DataFrame,
                 data: pd.DataFrame, calibration_band: float, cap_value: float) -> None:
    selected_metrics = metrics_df[metrics_df.model == selected].iloc[0]
    baseline = metrics_df[metrics_df.model == "Training median baseline"].iloc[0]
    reduction = 100 * (1 - selected_metrics.MAE_salary_dollars / baseline.MAE_salary_dollars)
    flagged = reviews[reviews.interval_status != "within approximate range"].head(8)
    coverage = 100 * (reviews.interval_status == "within approximate range").mean()
    first_start = int(data.target_season.min()) - 1
    last_end = int(data.target_season.max())
    lines = [
        "# NBA Player Compensation and On-Court Impact Analysis",
        "",
        "## Executive summary",
        "",
        f"This analysis expands salary prediction into a context-aware screening tool. It combines **{len(data):,} player-season observations** across salary seasons {first_start}-{str(first_start+1)[-2:]} through {last_end-1}-{str(last_end)[-2:]}, normalizes salary by the season's salary cap, adds a published adjusted plus-minus impact metric, groups players by statistical role, and estimates a broad prediction range. The latest completed salary season is held out for evaluation.",
        "",
        f"Rolling time-based validation selected **{selected}**. On the 2025-26 holdout, the model's MAE was **${selected_metrics.MAE_salary_dollars/1e6:.2f}M** versus **${baseline.MAE_salary_dollars/1e6:.2f}M** for a training-median baseline ({reduction:.1f}% lower). R² on salary-cap share was **{selected_metrics.R2_cap_share:.3f}**. The approximate 90% prediction band is **±{calibration_band*100:.2f} percentage points of the cap** (about ${calibration_band*cap_value/1e6:.1f}M in 2025-26 dollars).",
        "",
        "## Holdout model comparison",
        "",
        "| Model | MAE | RMSE | MAE (% of cap) | R² (cap share) |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in metrics_df.itertuples():
        lines.append(f"| {metric.model} | ${metric.MAE_salary_dollars/1e6:.2f}M | ${metric.RMSE_salary_dollars/1e6:.2f}M | {metric.MAE_cap_share_percentage_points:.2f} pp | {metric.R2_cap_share:.3f} |")
    lines += [
        "",
        f"Of the {len(reviews)} held-out players, **{coverage:.1f}%** fell inside the approximate band and {len(reviews)-int((reviews.interval_status == 'within approximate range').sum())} fell outside. Coverage is close to the nominal 90%, but it is measured on one test season after calibrating on one earlier season; it is not a formal guarantee under future season shifts.",
        "",
        "## What the model compares",
        "",
        "For each player, the model uses prior-season age and experience, availability, minutes, scoring and efficiency rates, and OpenRAPM offensive, defensive, and overall ratings. It predicts next-season salary as a share of the salary cap, so an identical contract percentage is comparable across seasons with different cap levels. Salary records are summed across team rows for a player-season.",
        "",
        "Six unsupervised clusters summarize statistical roles from usage, playmaking, rebounding, shooting efficiency, minutes, and impact rating. Salary comparisons use players in the same role cluster and experience band when the peer sample is large enough; otherwise the comparison falls back to the role cluster or the training-set median.",
        "",
        "## Review candidates",
        "",
        "A player is flagged only when observed salary falls outside an approximate range calibrated on the preceding salary season. The direction indicates where salary sits relative to the performance-based estimate; it is not a verdict on contract fairness.",
        "",
        "| Player | Salary season | Salary | Model estimate | Peer median | Prior WOSA | Interval flag |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in flagged.itertuples():
        lines.append(f"| {r.player} | {r.target_season-1}-{str(r.target_season)[-2:]} | ${r.salary/1e6:.1f}M | ${r.predicted_salary/1e6:.1f}M | {r.peer_salary_share*100:.1f}% cap | {r.wins_over_starter_average:.1f} | {r.interval_status} |")
    if len(flagged) == 0:
        lines.append("| No held-out player fell outside the approximate prediction band. | — | — | — | — | — | — |")
    lines += [
        "",
        "## Interpretation and next action",
        "",
        "The tool narrows the question from ‘Does salary look different from a basic statistical forecast?’ to ‘How unusual is salary among players with a similar statistical profile and experience, and how uncertain is that estimate?’ A next review should check contract years, guaranteed money, options, service-time rules, awards, injuries, and team circumstances using linked primary sources. This repository creates the evidence fields and candidate list; it does not automatically retrieve or verify contract/news context, and no generative-AI conclusion is presented as fact.",
        "",
        "OpenRAPM's `wins over starter average` (WOSA) is also included as a separate impact lens. It estimates regular-season wins added over an average starter; it is not converted into dollars here. A salary-to-WOSA ratio is not a true return-on-investment measure without an explicit, validated value-per-win assumption.",
        "",
        "## Limits",
        "",
        "- The outcome is salary as a share of the cap, not a counterfactual fair-market salary.",
        "- Salary caps, contract status, options, years remaining, team needs, market size, awards, and injury detail are not fully modeled.",
        "- The role groups are statistical clusters, not official positions or scouting archetypes.",
        "- The latest completed season is one temporal holdout; the calibration band comes from one preceding season and is approximate under year-to-year distribution shifts.",
        "- OpenRAPM is a published adjusted plus-minus estimate and carries model uncertainty. Its historical ratings are periodically recomputed by the publisher.",
        "- Salary data is sourced from Basketball-Reference team salary tables via the catalog. Payroll accounting can include salary obligations to former teams after trades/buyouts and can differ from contract cash flow.",
        "",
        "## Source notes",
        "",
        "- Historical NBA box-score and advanced player stats: NBA Stats, distributed through [nba_data](https://github.com/llimllib/nba_data).",
        "- Player salaries: Basketball-Reference team salary tables, distributed through [nba_data](https://github.com/llimllib/nba_data).",
        "- Impact: [OpenRAPM](https://openrapm.com/), described in the [nba_data dictionary](https://github.com/llimllib/nba_data/blob/main/docs/DATA_DICTIONARY.md#v2-openrapm-data-nbaopenrapm).",
        "- Salary cap: [Mr-Bridge cap and apron dataset](https://huggingface.co/datasets/Mr-Bridge/nba-salary-cap-contracts-2016-2026), cross-checked to NBA official releases per its source notes.",
    ]
    (OUT / "value_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ai_research_brief(candidates: pd.DataFrame) -> None:
    selected = candidates.head(8)
    lines = [
        "# AI-assisted context research brief",
        "",
        "This file contains model outputs only. Use it as input to a web-enabled research assistant to find possible explanations for salary gaps. The model did not retrieve contract news or decide that any salary is fair or unfair.",
        "",
        "## Research prompt",
        "",
        "For each player below, investigate public context that could explain why recorded 2025-26 salary differs from this project's estimate. Search for and cite direct, reliable sources: NBA/team announcements, the NBA/NBPA CBA or official salary-cap material, and reputable reporting for contract details, options, extensions, awards, availability, role, and team changes. Distinguish verified facts from hypotheses. Explain what the model omitted that could account for the gap. Do not call a player overpaid/underpaid or infer fair value. If a fact cannot be verified with a link, say so. Keep the answer to a short evidence table with columns: player, verified context, source link, how it could relate to the model gap, confidence/unknowns.",
        "",
        "## Model evidence cards",
        "",
        "Interpret gap as recorded salary minus model estimate. Positive means recorded salary is above the estimate; negative means below. The estimate predicts salary share of cap from prior-season stats, an impact rating, age, and experience. The interval is calibrated on one earlier season and is approximate.",
        "",
        "| Player | Salary | Model estimate | Gap | Approximate model range | Peer median share | Prior-season WOSA | Prior RAPM | Flag |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in selected.itertuples():
        lines.append(
            f"| {row.player} | ${row.salary/1e6:.1f}M | ${row.predicted_salary/1e6:.1f}M | ${row.salary_gap/1e6:+.1f}M | ${row.lower_salary_estimate/1e6:.1f}M–${row.upper_salary_estimate/1e6:.1f}M | {row.peer_salary_share*100:.1f}% cap | {row.wins_over_starter_average:.2f} | {row.rapm_rating:.2f} | {row.interval_status} |"
        )
    (OUT / "ai_research_brief.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data, cap = fetch_source_data("--refresh" in sys.argv)
    data = data.sort_values(["target_season", "player_id"]).reset_index(drop=True)
    test_season = int(data.target_season.max())
    calibration_season = test_season - 1
    if test_season != 2026:
        print(f"Using latest salary season in source data: {test_season-1}-{str(test_season)[-2:]}")
    fit_pool = data[data.target_season < calibration_season].copy()
    calibration = data[data.target_season == calibration_season].copy()
    final_train = data[data.target_season < test_season].copy()
    test = data[data.target_season == test_season].copy()
    if len(fit_pool) == 0 or len(calibration) == 0 or len(test) == 0:
        raise RuntimeError("Need training, calibration, and later-season test observations.")

    selected, cv = time_series_model_selection(fit_pool)
    cv.to_csv(OUT / "rolling_validation.csv", index=False)

    # One-season calibration of absolute prediction errors; interpret as an
    # approximate range because temporal shifts weaken exchangeability.
    cal_train_r, cal_r, _ = attach_roles(fit_pool, calibration)
    cal_model = make_model(selected)
    cal_model.fit(cal_train_r[FEATURES + CAT_FEATURES], cal_train_r.salary_cap_share)
    cal_pred = np.clip(cal_model.predict(cal_r[FEATURES + CAT_FEATURES]), 0, 0.5)
    band = float(np.quantile(np.abs(cal_r.salary_cap_share.to_numpy() - cal_pred), .90, method="higher"))

    # Final train and later-season test: all preprocessing and role clustering
    # are fitted only on the earlier player-seasons.
    train_r, test_r, clusterer = attach_roles(final_train, test)
    test_models = {}
    metrics_rows = []
    for name in ["Ridge (L2)", "Random forest"]:
        model = make_model(name)
        model.fit(train_r[FEATURES + CAT_FEATURES], train_r.salary_cap_share)
        pred = np.clip(model.predict(test_r[FEATURES + CAT_FEATURES]), 0, 0.5)
        test_models[name] = (model, pred)
        metrics_rows.append(metrics(test_r.salary_cap_share, pred, test_r.salary_cap, name, len(train_r)))
    baseline_pred = np.repeat(train_r.salary_cap_share.median(), len(test_r))
    metrics_rows.append(metrics(test_r.salary_cap_share, baseline_pred, test_r.salary_cap, "Training median baseline", len(train_r)))
    metric_df = pd.DataFrame(metrics_rows)
    metric_df.to_csv(OUT / "holdout_metrics.csv", index=False)
    model, pred_share = test_models[selected]

    review = test_r[["player", "player_id", "target_season", "salary", "salary_cap", "salary_cap_share",
                     "wins_over_starter_average", "rapm_rating", "rapm_possessions", "role_cluster", "experience_band"]].copy()
    review["predicted_cap_share"] = pred_share
    review["predicted_salary"] = pred_share * review.salary_cap
    review["salary_gap"] = review.salary - review.predicted_salary
    review["gap_cap_percentage_points"] = (review.salary_cap_share - review.predicted_cap_share) * 100
    review["lower_salary_estimate"] = np.maximum(0, pred_share - band) * review.salary_cap
    review["upper_salary_estimate"] = (pred_share + band) * review.salary_cap
    peer_values = [cohort_median(train_r, row) for _, row in review.iterrows()]
    review["peer_salary_share"] = [v[0] for v in peer_values]
    review["peer_count"] = [v[1] for v in peer_values]
    review["peer_definition"] = [v[2] for v in peer_values]
    review["peer_salary_estimate"] = review.peer_salary_share * review.salary_cap
    review["impact_per_cap_point"] = review.wins_over_starter_average / (review.salary_cap_share * 100)
    review["interval_status"] = np.select(
        [review.salary < review.lower_salary_estimate, review.salary > review.upper_salary_estimate],
        ["below approximate range", "above approximate range"],
        default="within approximate range",
    )
    review = review.sort_values("salary_gap", ascending=False)
    review.to_csv(OUT / "player_value_review.csv", index=False)
    # Only clear interval exceptions are termed candidates; keep largest cases.
    candidates = review[review.interval_status != "within approximate range"].copy()
    candidates["review_distance"] = np.where(
        candidates.interval_status == "above approximate range",
        candidates.salary - candidates.upper_salary_estimate,
        candidates.lower_salary_estimate - candidates.salary,
    )
    candidates = candidates.sort_values("review_distance", ascending=False)
    candidates.to_csv(OUT / "review_candidates.csv", index=False)

    # Learned feature importance from the selected model.
    pre = model.named_steps["features"]
    names = pre.get_feature_names_out()
    estimator = model.named_steps["model"]
    importance = getattr(estimator, "feature_importances_", None)
    if importance is None:
        importance = np.abs(estimator.coef_)
    pd.DataFrame({"feature": names, "importance": importance}).sort_values(
        "importance", ascending=False).to_csv(OUT / "feature_importance.csv", index=False)
    cluster_descriptions(train_r).to_csv(OUT / "role_clusters.csv", index=False)

    # The top prior-impact/cap-spend rates are descriptive screening values,
    # requiring positive WOSA and at least 1,000 RAPM possessions.
    impact = review[(review.wins_over_starter_average > 0) & (review.rapm_possessions >= 1000)].copy()
    impact.nlargest(20, "impact_per_cap_point").to_csv(OUT / "impact_per_cap_shortlist.csv", index=False)

    # Visual checks and compact review charts.
    cap_test = float(test_r.salary_cap.iloc[0])
    fig, ax = plt.subplots(figsize=(7, 6))
    actual = review.salary_cap_share * 100
    estimate = review.predicted_cap_share * 100
    ax.scatter(actual, estimate, alpha=.55, s=20, c=np.where(review.interval_status == "within approximate range", "#2563eb", "#dc2626"))
    lim = [0, max(float(actual.max()), float(estimate.max())) * 1.05]
    ax.plot(lim, lim, "--", color="#555", linewidth=1)
    ax.set(xlim=lim, ylim=lim, xlabel="Recorded salary (% of cap)", ylabel="Predicted salary (% of cap)", title=f"Season-ahead salary share: {test_season-1}-{str(test_season)[-2:]} holdout")
    fig.tight_layout(); fig.savefig(OUT / "holdout_cap_share.png", dpi=160); plt.close(fig)

    chart = candidates.nlargest(12, "review_distance").drop_duplicates("player_id")
    if not chart.empty:
        chart = chart.sort_values("salary_gap")
        fig, ax = plt.subplots(figsize=(9, max(5, .34 * len(chart))))
        ax.barh(chart.player, chart.gap_cap_percentage_points,
                color=["#2563eb" if x < 0 else "#dc2626" for x in chart.gap_cap_percentage_points])
        ax.axvline(0, color="#555", linewidth=1)
        ax.set(xlabel="Recorded salary minus model estimate (percentage points of cap)", title="Players outside the approximate prediction range")
        fig.tight_layout(); fig.savefig(OUT / "review_candidate_gaps.png", dpi=160); plt.close(fig)

    write_report(metric_df, selected, cv, review, data, band, cap_test)
    write_ai_research_brief(candidates)
    print("Selected by expanding-window validation:", selected)
    print("\nHoldout metrics (2025-26):")
    print(metric_df.to_string(index=False))
    print(f"\nCalibration band: ±{band*100:.2f} cap percentage points; candidates: {len(candidates)}")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
