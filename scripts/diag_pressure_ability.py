"""투수 능력/신뢰도와 현재 압박 상황의 최소 상호작용을 스크리닝한다.

기존 ``diag_social_features.py``가 단순 pressure/form 파생을, 그리고
``diag_pitcher_change_state.py``가 최근 궤적 상태를 이미 시험해 기각했으므로 이
스크립트는 그 피처를 반복하지 않는다. v4 고정 EB baseline에 11개의 해석 가능한
``현재 row-local 상황 x EB 능력`` 상호작용만 더한다. 같은 폴드에서 작은 context
전용 LGBM도 학습해 baseline과 90/10, 80/20, 70/30 혼합을 비교한다.

모든 피처는 평가 행 하나와 학습 fold에서 고정한 EB prior만으로 계산된다.
production/submit 경로는 수정하지 않는다.

사용법:
  python3 scripts/diag_pressure_ability.py
"""
import gc
import os
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                       add_shrinkage, shrinkage_cols, fit_prior,
                       make_lgbm_model, bss)

DATA_DIR = "./open/data"
PRED_DIR = "./experiments/preds"
VAL_SEASONS = [2022, 2023, 2024]
K = 50

BASE_SPEC = dict(seed=99, learning_rate=0.03, num_leaves=63,
                 min_child_samples=30, colsample_bytree=0.8, subsample=0.8)
CONTEXT_SPEC = dict(seed=8049, learning_rate=0.03, num_leaves=31,
                    min_child_samples=100, colsample_bytree=0.85,
                    subsample=0.8, reg_lambda=2.0)


def log(*args):
    print(*args, flush=True)


def add_pressure_ability(df, prior):
    """11개 최소 상호작용. 능력은 fold prior 대비 centered EB로 표현한다."""
    full = df["is_full_count"].to_numpy(dtype=float)
    three = df["is_three_ball"].to_numpy(dtype=float)
    risp = df["risp"].to_numpy(dtype=float)
    close = df["close_game"].to_numpy(dtype=float)
    late = (df["inning"].to_numpy(dtype=float) >= 7).astype(float)
    log_li = np.log1p(np.clip(df["li"].to_numpy(dtype=float), 0, None))
    high_li = (df["li"].to_numpy(dtype=float) >= 2.0).astype(float)

    success = (df["sh_asof_pitcher_success_rate"].to_numpy(dtype=float)
               - prior["asof_pitcher_success_rate"])
    ball = (df["sh_asof_pitcher_ball_rate"].to_numpy(dtype=float)
            - prior["asof_pitcher_ball_rate"])
    reliability = df["rel_asof_pitcher_n"].to_numpy(dtype=float)
    recent3 = df[
        "dev_asof_pitcher_prev3_game_success_rate"
    ].to_numpy(dtype=float)

    out = {
        "pa_full_x_success_skill": full * success,
        "pa_full_x_ball_skill": full * ball,
        "pa_three_ball_x_ball_skill": three * ball,
        "pa_risp_x_success_skill": risp * success,
        "pa_risp_x_ball_skill": risp * ball,
        "pa_logli_x_success_skill": log_li * success,
        "pa_logli_x_ball_skill": log_li * ball,
        "pa_late_x_success_skill": late * success,
        "pa_close_x_success_skill": close * success,
        "pa_logli_x_reliability": log_li * reliability,
        "pa_highli_x_recent3_dev": high_li * recent3,
    }
    return pd.DataFrame(out, index=df.index)


INTERACTION_COLS = [
    "pa_full_x_success_skill", "pa_full_x_ball_skill",
    "pa_three_ball_x_ball_skill", "pa_risp_x_success_skill",
    "pa_risp_x_ball_skill", "pa_logli_x_success_skill",
    "pa_logli_x_ball_skill", "pa_late_x_success_skill",
    "pa_close_x_success_skill", "pa_logli_x_reliability",
    "pa_highli_x_recent3_dev",
]

# 작은 보조 모델은 선수/팀 ID와 타자 이력을 제외하고, 현재 상황과 투수의 안정화된
# 능력/최근 폼만 본다. baseline의 축소판이 아니라 context expert가 되도록 제한한다.
CONTEXT_RAW = [
    "season", "game_month", "inning", "balls_before", "strikes_before",
    "outs_before", "score_diff_pitcher_team", "run_total_before",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
]
CONTEXT_EB = [
    "sh_asof_pitcher_success_rate", "sh_asof_pitcher_middle_rate",
    "sh_asof_pitcher_ball_rate", "sh_asof_pitcher_reverse_rate",
    "rel_asof_pitcher_n", "log_asof_pitcher_n",
    "sh_asof_pitcher_prev1_game_success_rate",
    "sh_asof_pitcher_prev3_game_success_rate",
    "sh_asof_pitcher_prev5_game_success_rate",
    "dev_asof_pitcher_prev1_game_success_rate",
    "dev_asof_pitcher_prev3_game_success_rate",
    "dev_asof_pitcher_prev5_game_success_rate",
]
CONTEXT_DERIVED = [
    "count_diff", "is_two_strike", "is_three_ball", "is_full_count", "risp",
    "pitcher_command_gap", "pitcher_recent_trend", "close_game",
]


def score_line(y, p):
    br, score, base = bss(y, p)
    return br, score, base


def paired(y, p0, p1, base):
    d = ((p0 - y) ** 2 - (p1 - y) ** 2) / base * 100000
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def fit_predict(X, y_tr, tr_m, va_m, num_cols, spec):
    spec = dict(spec)
    seed = spec.pop("seed")
    model = make_lgbm_model(num_cols, seed=seed, **spec)
    model.fit(X.loc[tr_m, CAT_COLS + num_cols], y_tr)
    p = model.predict_proba(X.loc[va_m, CAT_COLS + num_cols])[:, 1]
    del model
    gc.collect()
    return p


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y = raw[TARGET]
    seasons = raw["season"]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    base_num = raw_num + list(DERIVED_COLS) + shrinkage_cols()
    context_num = CONTEXT_RAW + CONTEXT_EB + CONTEXT_DERIVED + INTERACTION_COLS

    os.makedirs(PRED_DIR, exist_ok=True)
    results = {}
    log("fixed-EB v4 single-LGBM screen; folds=2022/2023/2024")
    log(f"interactions={len(INTERACTION_COLS)} context_num={len(context_num)}")

    for val_season in VAL_SEASONS:
        tr_m = (seasons < val_season).to_numpy()
        va_m = (seasons == val_season).to_numpy()
        y_tr = y[tr_m]
        yv = y[va_m].to_numpy(dtype=float)
        prior = fit_prior(raw.loc[tr_m])
        sh = add_shrinkage(base_df, prior, K)
        X0 = pd.concat([base_df, sh], axis=1)
        interactions = add_pressure_ability(X0, prior)
        X = pd.concat([X0, interactions], axis=1)

        log(f"\n[fold={val_season}] train={tr_m.sum():,} val={va_m.sum():,}")
        t = time.time()
        p_base = fit_predict(X, y_tr, tr_m, va_m, base_num, BASE_SPEC)
        log(f"  baseline trained {time.time()-t:.0f}s")
        t = time.time()
        p_inter = fit_predict(X, y_tr, tr_m, va_m,
                              base_num + INTERACTION_COLS, BASE_SPEC)
        log(f"  +interactions trained {time.time()-t:.0f}s")
        t = time.time()
        p_context = fit_predict(X, y_tr, tr_m, va_m, context_num, CONTEXT_SPEC)
        log(f"  context expert trained {time.time()-t:.0f}s")

        candidates = {
            "baseline": p_base,
            "interactions": p_inter,
            "context_only": p_context,
            "base90_context10": 0.90 * p_base + 0.10 * p_context,
            "base80_context20": 0.80 * p_base + 0.20 * p_context,
            "base70_context30": 0.70 * p_base + 0.30 * p_context,
        }
        br0, sc0, base = score_line(yv, p_base)
        for name, p in candidates.items():
            br, sc, _ = score_line(yv, p)
            gain, se = paired(yv, p_base, p, base)
            results[(name, val_season)] = (br, sc, gain, se)
            log(f"  {name:20s} brier={br:.8f} BSS={sc:8.2f} "
                f"gain={gain:+7.2f} SE={se:.2f}")
        np.savez_compressed(
            f"{PRED_DIR}/pressure_ability_{val_season}.npz",
            y=yv, p_baseline=p_base, p_interactions=p_inter,
            p_context=p_context, interaction_cols=np.asarray(INTERACTION_COLS),
            context_num_cols=np.asarray(context_num))
        del X, X0, sh, interactions
        gc.collect()

    log("\n" + "=" * 105)
    log("SUMMARY — baseline 대비 동일 행 paired BSS gain (양수=개선)")
    log("=" * 105)
    names = ["baseline", "interactions", "context_only",
             "base90_context10", "base80_context20", "base70_context30"]
    for name in names:
        rows = [results[(name, s)] for s in VAL_SEASONS]
        log(f"{name:20s} Brier=" + "/".join(f"{r[0]:.8f}" for r in rows)
            + " BSS=" + "/".join(f"{r[1]:.2f}" for r in rows)
            + " gain=" + "/".join(f"{r[2]:+.2f}" for r in rows)
            + f" mean_gain={np.mean([r[2] for r in rows]):+.2f}"
            + " SE=" + "/".join(f"{r[3]:.2f}" for r in rows))
    log("\ninteraction columns:")
    for c in INTERACTION_COLS:
        log(f"- {c}")


if __name__ == "__main__":
    main()
