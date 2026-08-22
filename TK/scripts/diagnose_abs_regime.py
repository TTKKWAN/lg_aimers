"""ABS 전후 데이터 구조를 진단해 stable/ABS expert 피처 후보를 만든다.

이 스크립트는 모델 제출물을 변경하지 않는다. 대규모 분류기·클러스터링을 포함하므로
저장소 지침에 따라 Google Colab에서 실행한다.

실행:
    python3 scripts/diagnose_abs_regime.py \
        --train open/data/train.csv \
        --output artifacts/abs_diagnostics

산출물:
    season_month_distribution.csv   시즌·월별 피처 분포
    feature_shift.csv                pre-2024 vs 2024 표준화 차이와 PSI
    adversarial_scores.csv           2024 구분 분류기 AUC
    adversarial_importance.csv       분류기별 gain 중요도
    pitcher_season_clusters.csv      투수-시즌 군집
    pitcher_month_clusters.csv       투수-월 군집
    cluster_summary.csv              군집별 프로필
    monthly_change_points.csv        월별 최적 단일 변화점
    feature_routing_candidate.csv    stable/ABS expert 라우팅 초안
    summary.json                     핵심 실행 메타

NaN 행 삭제나 NaN 진단은 의도적으로 수행하지 않는다. 분류·클러스터 입력의 결측은
학습 가능하도록 train 중앙값으로만 대체하며, 결측률 자체는 진단 피처로 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET = "control_success"
ABS_SEASON = 2024

SENSITIVE = [
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
]

STABLE = [
    "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before",
    "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "base_state", "home_win_expectancy", "away_win_expectancy", "li",
    "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id",
    "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n",
]

UNCERTAIN = [
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

ID_COLS = {"row_id", "pitcher_id", "batter_id"}
DIRECT_TIME_COLS = {"season"}


@dataclass
class ClusterResult:
    assignments: pd.DataFrame
    summary: pd.DataFrame
    k: int
    silhouette: float


def existing(columns, names):
    available = set(columns)
    return [c for c in names if c in available]


def psi(pre, post, bins=10):
    """pre 분위수 구간을 고정해 population stability index를 계산한다."""
    a = pd.to_numeric(pre, errors="coerce").dropna().to_numpy(float)
    b = pd.to_numeric(post, errors="coerce").dropna().to_numpy(float)
    if len(a) < 20 or len(b) < 20:
        return np.nan
    edges = np.unique(np.nanquantile(a, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ah = np.histogram(a, bins=edges)[0].astype(float)
    bh = np.histogram(b, bins=edges)[0].astype(float)
    ap = np.clip(ah / max(ah.sum(), 1), 1e-6, None)
    bp = np.clip(bh / max(bh.sum(), 1), 1e-6, None)
    return float(np.sum((bp - ap) * np.log(bp / ap)))


def distribution_tables(df, output):
    metrics = existing(df.columns, SENSITIVE + UNCERTAIN + [TARGET])
    rows = []
    grouped = df.groupby(["season", "game_month"], observed=True, sort=True)
    for (season, month), part in grouped:
        for col in metrics:
            x = pd.to_numeric(part[col], errors="coerce")
            rows.append({
                "season": int(season), "game_month": int(month), "feature": col,
                "n_rows": int(len(part)), "n_observed": int(x.notna().sum()),
                "mean": float(x.mean()), "std": float(x.std()),
                "q10": float(x.quantile(.10)), "q25": float(x.quantile(.25)),
                "median": float(x.median()), "q75": float(x.quantile(.75)),
                "q90": float(x.quantile(.90)),
            })
    pd.DataFrame(rows).to_csv(
        os.path.join(output, "season_month_distribution.csv"), index=False)

    pre, post = df[df["season"] < ABS_SEASON], df[df["season"] == ABS_SEASON]
    shifts = []
    for col in metrics:
        a = pd.to_numeric(pre[col], errors="coerce")
        b = pd.to_numeric(post[col], errors="coerce")
        pooled = float(np.sqrt((a.var() + b.var()) / 2))
        delta = float(b.mean() - a.mean())
        shifts.append({
            "feature": col,
            "pre_mean": float(a.mean()), "abs_mean": float(b.mean()),
            "mean_delta": delta,
            "standardized_delta": delta / pooled if pooled > 0 else 0.0,
            "psi": psi(a, b),
        })
    shift = pd.DataFrame(shifts).sort_values(
        ["psi", "standardized_delta"], ascending=[False, False])
    shift.to_csv(os.path.join(output, "feature_shift.csv"), index=False)
    return shift


def encode_frame(df, columns):
    out = pd.DataFrame(index=df.index)
    for col in columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            out[col] = pd.to_numeric(s, errors="coerce").astype("float32")
        else:
            codes, _ = pd.factorize(s.fillna("__NA__").astype(str), sort=True)
            out[col] = codes.astype("float32")
    return out


def balanced_regime_sample(df, max_per_class, seed):
    pre = df[df["season"] < ABS_SEASON]
    post = df[df["season"] == ABS_SEASON]
    n = min(len(pre), len(post), max_per_class)
    return pd.concat([
        pre.sample(n=n, random_state=seed),
        post.sample(n=n, random_state=seed),
    ], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def adversarial_validation(df, output, max_per_class, seed):
    sample = balanced_regime_sample(df, max_per_class, seed)
    all_no_ids = [
        c for c in df.columns
        if c not in ID_COLS | DIRECT_TIME_COLS | {TARGET}
    ]
    variants = {
        "sensitive_only": existing(df.columns, SENSITIVE),
        "stable_only": existing(df.columns, STABLE),
        "uncertain_only": existing(df.columns, UNCERTAIN),
        "all_no_ids_no_season": all_no_ids,
    }
    y = (sample["season"] == ABS_SEASON).astype(np.int8)
    scores, importances = [], []
    for name, cols in variants.items():
        if not cols:
            continue
        x = encode_frame(sample, cols)
        train_idx, valid_idx = train_test_split(
            np.arange(len(x)), test_size=.25, stratify=y, random_state=seed)
        medians = x.iloc[train_idx].median()
        x_train = x.iloc[train_idx].fillna(medians)
        x_valid = x.iloc[valid_idx].fillna(medians)
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=400, learning_rate=.04,
            num_leaves=31, min_child_samples=100, reg_lambda=2.0,
            colsample_bytree=.85, subsample=.8, subsample_freq=1,
            random_state=seed, verbosity=-1,
        )
        model.fit(x_train, y.iloc[train_idx])
        pred = model.predict_proba(x_valid)[:, 1]
        auc = float(roc_auc_score(y.iloc[valid_idx], pred))
        scores.append({
            "variant": name, "auc": auc, "n_features": len(cols),
            "n_train": len(train_idx), "n_valid": len(valid_idx),
        })
        gain = model.booster_.feature_importance(importance_type="gain")
        gain_total = max(float(gain.sum()), 1.0)
        for col, value in zip(cols, gain):
            importances.append({
                "variant": name, "feature": col, "gain": float(value),
                "gain_share": float(value / gain_total),
            })
    score_df = pd.DataFrame(scores).sort_values("auc", ascending=False)
    imp_df = pd.DataFrame(importances).sort_values(
        ["variant", "gain_share"], ascending=[True, False])
    score_df.to_csv(os.path.join(output, "adversarial_scores.csv"), index=False)
    imp_df.to_csv(os.path.join(output, "adversarial_importance.csv"), index=False)
    return score_df, imp_df


def aggregate_pitchers(df, keys, min_rows):
    rates = existing(df.columns, SENSITIVE + UNCERTAIN)
    agg = df.groupby(keys, observed=True, sort=False).agg(
        n_rows=(TARGET, "size"), target_rate=(TARGET, "mean"),
    ).reset_index()
    means = df.groupby(keys, observed=True, sort=False)[rates].mean().reset_index()
    agg = agg.merge(means, on=keys, how="left")
    return agg[agg["n_rows"] >= min_rows].reset_index(drop=True), rates


def cluster_profiles(frame, keys, features, seed):
    x = frame[features].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median())
    scaled = StandardScaler().fit_transform(x)
    sample_idx = np.arange(len(frame))
    if len(sample_idx) > 10000:
        sample_idx = np.random.default_rng(seed).choice(
            sample_idx, size=10000, replace=False)
    candidates = []
    for k in range(2, min(7, len(frame))):
        model = MiniBatchKMeans(
            n_clusters=k, random_state=seed, n_init=10, batch_size=2048)
        labels = model.fit_predict(scaled)
        score = float(silhouette_score(scaled[sample_idx], labels[sample_idx]))
        candidates.append((score, k, labels))
    score, k, labels = max(candidates, key=lambda z: z[0])
    assigned = frame.copy()
    assigned["cluster"] = labels
    summary = assigned.groupby("cluster", observed=True).agg(
        n_profiles=(keys[0], "size"), mean_rows=("n_rows", "mean"),
        mean_target=("target_rate", "mean"),
    ).reset_index()
    profile = assigned.groupby("cluster", observed=True)[features].mean().reset_index()
    summary = summary.merge(profile, on="cluster", how="left")
    summary["k"] = k
    summary["silhouette"] = score
    return ClusterResult(assigned, summary, k, score)


def run_clustering(df, output, seed, min_season_rows, min_month_rows):
    summaries = []
    season, season_features = aggregate_pitchers(
        df, ["pitcher_id", "season"], min_season_rows)
    season_result = cluster_profiles(
        season, ["pitcher_id", "season"], season_features, seed)
    season_result.assignments.to_csv(
        os.path.join(output, "pitcher_season_clusters.csv"), index=False)
    s = season_result.summary.copy()
    s.insert(0, "level", "pitcher_season")
    summaries.append(s)

    month, month_features = aggregate_pitchers(
        df, ["pitcher_id", "season", "game_month"], min_month_rows)
    month_result = cluster_profiles(
        month, ["pitcher_id", "season", "game_month"], month_features, seed)
    month_result.assignments.to_csv(
        os.path.join(output, "pitcher_month_clusters.csv"), index=False)
    m = month_result.summary.copy()
    m.insert(0, "level", "pitcher_month")
    summaries.append(m)
    pd.concat(summaries, ignore_index=True).to_csv(
        os.path.join(output, "cluster_summary.csv"), index=False)
    return season_result, month_result


def best_single_change(values, min_segment=5):
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 2 * min_segment:
        return None
    y = pd.Series(y).interpolate(limit_direction="both").to_numpy()
    base = float(np.sum((y - y.mean()) ** 2))
    best = None
    for split in range(min_segment, len(y) - min_segment + 1):
        left, right = y[:split], y[split:]
        loss = float(np.sum((left - left.mean()) ** 2)
                     + np.sum((right - right.mean()) ** 2))
        gain = base - loss
        if best is None or gain > best[0]:
            best = (gain, split, loss)
    return best, base


def change_points(df, output):
    metrics = existing(df.columns, SENSITIVE + UNCERTAIN + [TARGET])
    monthly = df.groupby(["season", "game_month"], observed=True)[metrics].mean()
    monthly = monthly.sort_index().reset_index()
    rows = []
    for col in metrics:
        result = best_single_change(monthly[col].to_numpy())
        if result is None:
            continue
        (gain, split, loss), base = result
        point = monthly.iloc[split]
        rows.append({
            "feature": col, "change_season": int(point["season"]),
            "change_month": int(point["game_month"]),
            "sse_gain": gain, "relative_sse_gain": gain / base if base > 0 else 0.0,
            "left_mean": float(monthly[col].iloc[:split].mean()),
            "right_mean": float(monthly[col].iloc[split:].mean()),
        })
    result = pd.DataFrame(rows).sort_values("relative_sse_gain", ascending=False)
    result.to_csv(os.path.join(output, "monthly_change_points.csv"), index=False)
    return result


def routing_candidate(df, shift, importance, output):
    shift_map = shift.set_index("feature").to_dict("index")
    full_imp = importance[importance["variant"] == "all_no_ids_no_season"]
    imp_map = full_imp.set_index("feature")["gain_share"].to_dict()
    rows = []
    for col in df.columns:
        if col in ID_COLS | DIRECT_TIME_COLS | {TARGET}:
            continue
        if col in SENSITIVE:
            route, reason = "abs_expert_only_candidate", "ABS-sensitive rate family"
        elif col in STABLE:
            route, reason = "stable_backbone_candidate", "pre-pitch context/workload"
        elif col in UNCERTAIN:
            route, reason = "ablation_required", "pitch-mix may reflect adaptation"
        else:
            route, reason = "review_required", "not preclassified"
        stat = shift_map.get(col, {})
        rows.append({
            "feature": col, "candidate_route": route, "reason": reason,
            "psi": stat.get("psi", np.nan),
            "standardized_delta": stat.get("standardized_delta", np.nan),
            "adversarial_gain_share": imp_map.get(col, 0.0),
        })
    result = pd.DataFrame(rows).sort_values(
        ["candidate_route", "adversarial_gain_share"], ascending=[True, False])
    result.to_csv(os.path.join(output, "feature_routing_candidate.csv"), index=False)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="open/data/train.csv")
    parser.add_argument("--output", default="artifacts/abs_diagnostics")
    parser.add_argument("--max-per-class", type=int, default=250000)
    parser.add_argument("--min-season-rows", type=int, default=300)
    parser.add_argument("--min-month-rows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(args.train, encoding="utf-8-sig")
    required = {"season", "game_month", "pitcher_id", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼 없음: {sorted(missing)}")

    shift = distribution_tables(df, args.output)
    scores, importance = adversarial_validation(
        df, args.output, args.max_per_class, args.seed)
    season_clusters, month_clusters = run_clustering(
        df, args.output, args.seed, args.min_season_rows, args.min_month_rows)
    changes = change_points(df, args.output)
    routing = routing_candidate(df, shift, importance, args.output)

    summary = {
        "rows": int(len(df)),
        "seasons": sorted(int(x) for x in df["season"].unique()),
        "abs_season": ABS_SEASON,
        "adversarial_auc": dict(zip(scores["variant"], scores["auc"])),
        "pitcher_season_cluster_k": season_clusters.k,
        "pitcher_season_silhouette": season_clusters.silhouette,
        "pitcher_month_cluster_k": month_clusters.k,
        "pitcher_month_silhouette": month_clusters.silhouette,
        "top_change_points": changes.head(10).to_dict("records"),
        "routing_counts": routing["candidate_route"].value_counts().to_dict(),
        "nan_diagnostics_included": False,
    }
    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
