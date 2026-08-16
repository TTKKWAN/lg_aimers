"""제구 성공의 상황적 메커니즘과 중복 피처 제거를 함께 진단한다.

사회과학적 해석 프레임:
  ability   : 투수의 장기 제구 능력 / 최근 폼
  strategy  : 카운트에 따른 위험 선택(3볼에는 존 승부, 2스트라이크에는 유인구 여지)
  pressure  : 접전·득점권·높은 LI에서의 수행 변화
  fatigue   : 후반/연장 이닝이라는 제한적인 피로 대리변수
  matchup   : 타자 위협도·좌우 상성·경험 차이
  redundancy: 같은 경기 상태를 여러 방식으로 표현해 생기는 과적합

주의: 관찰 데이터의 조건부 차이는 인과효과가 아니다. 아래 기술통계는
pitcher-season 평균을 제거한 within-group association이고, 채택 여부는 반드시
forward-chaining 3개 시즌 폴드의 예측 성능으로만 결정한다.

사용법:
  python3 scripts/diag_social_features.py quick
  python3 scripts/diag_social_features.py confirm social drop_raw_rates

quick은 LightGBM 1개로 넓게 스크리닝하고, confirm은 지정 후보를 3개 이질
LightGBM 앙상블로 baseline과 다시 비교한다.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, RATE_N_PAIRS, PREV_SPECS,
                       add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                       make_lgbm_model, bss)

DATA_DIR = "./open/data"
PRED_DIR = "./experiments/preds"
VAL_SEASONS = [2022, 2023, 2024]
K = 50

LGBM_SPECS = [
    dict(seed=99, learning_rate=0.03, num_leaves=63, min_child_samples=30,
         colsample_bytree=0.8, subsample=0.8),
    dict(seed=2718, learning_rate=0.05, num_leaves=31, min_child_samples=50,
         colsample_bytree=0.7, subsample=0.7),
    dict(seed=31415, learning_rate=0.02, num_leaves=127, min_child_samples=20,
         colsample_bytree=0.9, subsample=0.9),
]

RAW_RATE_COLS = [r for r, _ in RATE_N_PAIRS] + [c for c, _, _ in PREV_SPECS]
REDUNDANT_STATE_COLS = [
    # 아래 정보는 우측 주석의 보존 컬럼으로 거의 동일하게 표현된다.
    "run_top_before", "run_bot_before",       # run_total + score difference 보존
    "score_diff_home",                         # score_diff_pitcher_team 보존
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
                                                # base_state + 파생 risp 보존
    "away_win_expectancy",                     # home_win_expectancy 보존
]
TEAM_CALENDAR_COLS = ["game_dayofweek", "pitcher_team_id", "batter_team_id"]


def log(*args):
    print(*args, flush=True)


def add_social_features(df):
    """한 행만으로 계산 가능한 메커니즘 기반 상호작용 피처."""
    li_log = np.log1p(df["li"].clip(lower=0))
    risp = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype(np.int8)
    close = (df["score_diff_pitcher_team"].abs() <= 1).astype(np.int8)
    late = (df["inning"] >= 7).astype(np.int8)
    p = df[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate"]].to_numpy(dtype=float)
    p = np.clip(p, 1e-12, 1.0)

    out = {
        # strategy: 실제 타석 투구수는 아니지만 현재 카운트의 깊이를 나타내는 사전정보
        "pa_depth_proxy": df["balls_before"] + df["strikes_before"],
        "deep_count": ((df["balls_before"] + df["strikes_before"]) >= 3).astype(np.int8),
        "count_tension": df["balls_before"] * df["strikes_before"],
        # pressure: LI를 그대로 복제하지 않고 이론상 의미 있는 비선형/상호작용만 추가
        "log_li": li_log,
        "high_leverage": (df["li"] >= 2.0).astype(np.int8),
        "traffic_pressure": df["num_runners_on"] * li_log,
        "risp_leverage": risp * li_log,
        "close_leverage": close * li_log,
        "win_uncertainty": 1.0 - (df["home_win_expectancy"] - 50.0).abs() / 50.0,
        # fatigue/score state: inning은 투수 개인 투구수가 아니므로 약한 proxy로만 해석
        "late_inning": late,
        "extra_inning": (df["inning"] >= 10).astype(np.int8),
        "late_close": late * close,
        "late_high_leverage": late * (df["li"] >= 2.0).astype(np.int8),
        "score_margin_abs": df["score_diff_pitcher_team"].abs(),
        "pitcher_leading": (df["score_diff_pitcher_team"] > 0).astype(np.int8),
        "pitcher_trailing": (df["score_diff_pitcher_team"] < 0).astype(np.int8),
        "blowout": (df["score_diff_pitcher_team"].abs() >= 5).astype(np.int8),
        # matchup/experience
        "experience_gap": np.log1p(df["asof_pitcher_n"]) - np.log1p(df["asof_batter_n"]),
        "pitcher_cold_start": (df["asof_pitcher_n"] == 0).astype(np.int8),
        "batter_cold_start": (df["asof_batter_n"] == 0).astype(np.int8),
        "middle_matchup_gap": df["asof_batter_middle_rate"] - df["asof_pitcher_middle_rate"],
        # repertoire complexity: 세 비율의 합이 1이 아니어도 상대적 다양성으로 작동
        "pitchmix_entropy": -(p * np.log(p)).sum(axis=1),
        "pitchmix_concentration": (p ** 2).sum(axis=1),
        # form stability: level보다 최근 경기 간 변화/불안정성에 초점
        "success_trend_1v3": (df["asof_pitcher_prev1_game_success_rate"]
                              - df["asof_pitcher_prev3_game_success_rate"]),
        "success_trend_3v5": (df["asof_pitcher_prev3_game_success_rate"]
                              - df["asof_pitcher_prev5_game_success_rate"]),
        "middle_trend_1v3": (df["asof_pitcher_prev1_game_middle_rate"]
                             - df["asof_pitcher_prev3_game_middle_rate"]),
        "middle_trend_3v5": (df["asof_pitcher_prev3_game_middle_rate"]
                             - df["asof_pitcher_prev5_game_middle_rate"]),
    }
    success_recent = df[["asof_pitcher_prev1_game_success_rate",
                         "asof_pitcher_prev3_game_success_rate",
                         "asof_pitcher_prev5_game_success_rate"]]
    middle_recent = df[["asof_pitcher_prev1_game_middle_rate",
                        "asof_pitcher_prev3_game_middle_rate",
                        "asof_pitcher_prev5_game_middle_rate"]]
    out["success_form_volatility"] = success_recent.std(axis=1)
    out["middle_form_volatility"] = middle_recent.std(axis=1)
    return pd.DataFrame(out, index=df.index)


SOCIAL_COLS = [
    "pa_depth_proxy", "deep_count", "count_tension", "log_li", "high_leverage",
    "traffic_pressure", "risp_leverage", "close_leverage", "win_uncertainty",
    "late_inning", "extra_inning", "late_close", "late_high_leverage",
    "score_margin_abs", "pitcher_leading", "pitcher_trailing", "blowout",
    "experience_gap", "pitcher_cold_start", "batter_cold_start",
    "middle_matchup_gap", "pitchmix_entropy", "pitchmix_concentration",
    "success_trend_1v3", "success_trend_3v5", "middle_trend_1v3",
    "middle_trend_3v5", "success_form_volatility", "middle_form_volatility",
]

STRATEGY_COLS = ["pa_depth_proxy", "deep_count", "count_tension"]
PRESSURE_COLS = ["log_li", "high_leverage", "traffic_pressure", "risp_leverage",
                 "close_leverage", "win_uncertainty"]
FATIGUE_SCORE_COLS = ["late_inning", "extra_inning", "late_close",
                      "late_high_leverage", "score_margin_abs", "pitcher_leading",
                      "pitcher_trailing", "blowout"]
MATCHUP_COLS = ["experience_gap", "pitcher_cold_start", "batter_cold_start",
                "middle_matchup_gap"]
PITCHMIX_COLS = ["pitchmix_entropy", "pitchmix_concentration"]
FORM_COLS = ["success_trend_1v3", "success_trend_3v5", "middle_trend_1v3",
             "middle_trend_3v5", "success_form_volatility", "middle_form_volatility"]

CONFIGS = {
    "baseline": dict(social_cols=[], drop=[]),
    "strategy": dict(social_cols=STRATEGY_COLS, drop=[]),
    "pressure": dict(social_cols=PRESSURE_COLS, drop=[]),
    "fatigue_score": dict(social_cols=FATIGUE_SCORE_COLS, drop=[]),
    "matchup": dict(social_cols=MATCHUP_COLS, drop=[]),
    "pitchmix": dict(social_cols=PITCHMIX_COLS, drop=[]),
    "form": dict(social_cols=FORM_COLS, drop=[]),
    "social": dict(social_cols=SOCIAL_COLS, drop=[]),
    "drop_raw_rates": dict(social_cols=[], drop=RAW_RATE_COLS),
    "drop_redundant_state": dict(social_cols=[], drop=REDUNDANT_STATE_COLS),
    "drop_team_calendar": dict(social_cols=[], drop=TEAM_CALENDAR_COLS),
    "drop_batter_id": dict(social_cols=[], drop=["batter_id"]),
    "social_drop_raw": dict(social_cols=SOCIAL_COLS, drop=RAW_RATE_COLS),
    "social_parsimonious": dict(social_cols=SOCIAL_COLS,
                                 drop=RAW_RATE_COLS + REDUNDANT_STATE_COLS + TEAM_CALENDAR_COLS),
}


def describe_associations(raw):
    """투수-시즌 평균 능력을 제거한 조건부 연관성을 출력한다."""
    y = raw[TARGET].astype(float)
    g = raw.groupby(["season", "pitcher_id"])[TARGET]
    group_sum = g.transform("sum").astype(float)
    group_n = g.transform("size").astype(float)
    peer_mean = (group_sum - y) / (group_n - 1).clip(lower=1)
    resid = y - peer_mean
    factors = {
        "3-ball (zone obligation)": raw["balls_before"] == 3,
        "2-strike (waste/chase freedom)": raw["strikes_before"] == 2,
        "full count": (raw["balls_before"] == 3) & (raw["strikes_before"] == 2),
        "RISP": (raw["runner_on_2b"] == 1) | (raw["runner_on_3b"] == 1),
        "high leverage (LI>=2)": raw["li"] >= 2.0,
        "close game (|diff|<=1)": raw["score_diff_pitcher_team"].abs() <= 1,
        "late inning (>=7)": raw["inning"] >= 7,
        "late & close": ((raw["inning"] >= 7)
                         & (raw["score_diff_pitcher_team"].abs() <= 1)),
        "pitcher trailing": raw["score_diff_pitcher_team"] < 0,
        "blowout (|diff|>=5)": raw["score_diff_pitcher_team"].abs() >= 5,
        "platoon same hand": raw["pitcher_hand"] == raw["batter_hand"],
        "final/postseason game type": raw["game_type"] == "F",
    }
    log("\n" + "=" * 88)
    log("기술통계 — 투수×시즌 평균을 제거한 within-pitcher association")
    log("(pp = 성공확률 percentage point; 인과효과로 해석하지 않음)")
    log("=" * 88)
    for name, mask in factors.items():
        mask = mask.to_numpy(dtype=bool)
        raw_diff = (y[mask].mean() - y[~mask].mean()) * 100
        within_diff = (resid[mask].mean() - resid[~mask].mean()) * 100
        log(f"{name:34s} n={mask.sum():>8,}  raw={raw_diff:+7.3f}pp  "
            f"within pitcher-season={within_diff:+7.3f}pp")


def num_cols_for(raw_num, cfg):
    cols = list(raw_num) + list(DERIVED_COLS) + shrinkage_cols()
    cols += cfg["social_cols"]
    drop = set(cfg["drop"])
    return [c for c in cols if c not in drop]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if mode not in ("quick", "confirm"):
        raise SystemExit("mode must be quick or confirm")

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    derived = add_derived(raw)
    social = add_social_features(raw)
    base_df = pd.concat([raw[raw_features], derived, social], axis=1)
    describe_associations(raw)

    if mode == "quick":
        requested = sys.argv[2:]
        unknown = [x for x in requested if x not in CONFIGS]
        if unknown:
            raise SystemExit(f"unknown configs: {unknown}; choices={list(CONFIGS)}")
        names = ["baseline"] + [x for x in requested if x != "baseline"] if requested else list(CONFIGS)
        specs = LGBM_SPECS[:1]
    else:
        requested = sys.argv[2:]
        unknown = [x for x in requested if x not in CONFIGS]
        if unknown:
            raise SystemExit(f"unknown configs: {unknown}; choices={list(CONFIGS)}")
        names = ["baseline"] + [x for x in requested if x != "baseline"]
        if len(names) == 1:
            raise SystemExit("confirm 뒤에 후보 config를 하나 이상 지정하세요")
        specs = LGBM_SPECS

    log(f"\nmode={mode} configs={names} members={len(specs)} folds={VAL_SEASONS}")
    losses, scores = {}, {}
    os.makedirs(PRED_DIR, exist_ok=True)

    for val_season in VAL_SEASONS:
        tr_m = (seasons < val_season).to_numpy()
        va_m = (seasons == val_season).to_numpy()
        y_tr, y_va = y[tr_m], y[va_m]
        yv = y_va.to_numpy(dtype=float)
        prior = fit_prior(raw.loc[tr_m])
        sh = add_shrinkage(base_df, prior, K)
        X = pd.concat([base_df, sh], axis=1)
        log(f"\n[fold={val_season}] train={tr_m.sum():,} val={va_m.sum():,}")

        for name in names:
            cfg = CONFIGS[name]
            num_cols = num_cols_for(raw_num, cfg)
            cols = CAT_COLS + num_cols
            ps = []
            log(f"  -- {name} features={len(cols)} drop={len(cfg['drop'])}")
            for i, spec0 in enumerate(specs):
                spec = dict(spec0)
                seed = spec.pop("seed")
                t = time.time()
                model = make_lgbm_model(num_cols, seed=seed, **spec)
                model.fit(X.loc[tr_m, cols], y_tr)
                p = model.predict_proba(X.loc[va_m, cols])[:, 1]
                ps.append(p)
                _, single, _ = bss(yv, p)
                log(f"     member={i+1}/{len(specs)} seed={seed} score={single:8.2f} "
                    f"time={time.time()-t:.0f}s")
                del model
                gc.collect()
            p = np.mean(ps, axis=0)
            _, score, base = bss(yv, p)
            losses[(name, val_season)] = ((p - yv) ** 2, base)
            scores[(name, val_season)] = score
            np.savez_compressed(f"{PRED_DIR}/social_{mode}_{name}_{val_season}.npz",
                                p=p, y=yv, n=len(specs))
            log(f"     >> ensemble={score:.2f}")
        del X, sh
        gc.collect()

    log("\n" + "=" * 88)
    log(f"{mode.upper()} 요약 — baseline 대비 동일 행 paired BSS 차이")
    log("=" * 88)
    for name in names:
        fold_scores = [scores[(name, f)] for f in VAL_SEASONS]
        line = f"{name:24s} folds=" + "/".join(f"{s:7.2f}" for s in fold_scores)
        line += f" mean={np.mean(fold_scores):7.2f}"
        if name != "baseline":
            fold_gain = []
            fold_se = []
            for f in VAL_SEASONS:
                loss0, base = losses[("baseline", f)]
                loss1, _ = losses[(name, f)]
                d = (loss0 - loss1) / base * 100000
                fold_gain.append(d.mean())
                fold_se.append(d.std(ddof=1) / np.sqrt(len(d)))
            line += "  gains=" + "/".join(f"{d:+6.2f}" for d in fold_gain)
            line += f" mean_gain={np.mean(fold_gain):+7.2f}"
            line += "  SE=" + "/".join(f"{s:.1f}" for s in fold_se)
        log(line)


if __name__ == "__main__":
    main()
