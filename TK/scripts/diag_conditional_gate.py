"""Strict-past OOF 조건부 mixture-of-experts gate를 검증한다.

현재 v9은 모든 행에서 v8 base 40% + CatBoost expert 60%를 고정 사용한다. 이
스크립트는 각 outer validation 시즌보다 엄격히 과거인 시즌의 OOF 예측만으로
작은 ridge gate를 학습한다. gate는 ``p_base + w(x) * (p_cat - p_base)`` 형태이고
``w(x)``는 [0, 1]로 제한한다.

기존 2022~2024 OOF 캐시 외에 2021 OOF를 production과 같은 HGB8/LGBM3/CatBoost2로
한 번 생성한다. 따라서 outer 2022도 2021 OOF만으로 gate를 학습하며 자기 시즌
정답이나 미래 시즌 정답을 보지 않는다.
"""
import gc
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                      add_shrinkage, shrinkage_cols, fit_prior,
                      make_model, make_lgbm_model,
                      CATBOOST_CAT_COLS, add_catboost_context)
from diag_season_to_date import (DATA_DIR, PRED_DIR, K, GROUPS,
                                 _previous_lookup, _season_block, paired)
from diag_season_to_date_followup import exact_cols
from diag_current_season_family import (HGB_SPECS, LGBM_SPECS, mean_predictions,
                                        extrapolated_rate,
                                        recenter_like_production)
from diag_catboost_expert import fit_predict_catboost


OOF_SEASONS = [2021, 2022, 2023, 2024]
VAL_SEASONS = [2022, 2023, 2024]
CACHE_2021 = f"{PRED_DIR}/conditional_gate_oof_2021.npz"
LAMBDAS = [0.0, 1e-6, 1e-5, 1e-4]


def log(*args):
    print(*args, flush=True)


def _prepare_fold_matrix(raw, raw_features, raw_num, val_season):
    """production과 같은 base/CatBoost 입력을 한 outer fold에 만든다."""
    y, seasons = raw[TARGET], raw["season"]
    tr = (seasons < val_season).to_numpy()
    va = seasons.eq(val_season).to_numpy()
    last_season = int(seasons.loc[tr].max())
    lm = tr & seasons.eq(last_season).to_numpy()
    prior = fit_prior(raw.loc[tr])
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    X0 = pd.concat([base_df, add_shrinkage(base_df, prior, K)], axis=1)
    blocks = {}
    for group in ("pitcher", "batter"):
        previous = _previous_lookup(raw, tr, GROUPS[group])
        blocks[group], _ = _season_block(
            X0, previous, GROUPS[group], prior, zero_baseline=True)
    current16 = (exact_cols(blocks["pitcher"], "pitcher",
                            "asof_pitcher_success_rate")
                 + exact_cols(blocks["batter"], "batter",
                              "asof_batter_success_rate"))
    X = pd.concat([X0, blocks["pitcher"], blocks["batter"],
                   add_catboost_context(raw)], axis=1)
    base_num = raw_num + list(DERIVED_COLS) + shrinkage_cols()
    numeric_ids = {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
    expert_num = [c for c in raw_num if c not in numeric_ids]
    expert_num += list(DERIVED_COLS) + shrinkage_cols()
    expert_cols = list(CATBOOST_CAT_COLS) + expert_num + current16
    return X, base_num, current16, expert_cols, tr, va, lm, prior


def build_2021_cache(raw, raw_features, raw_num):
    if os.path.exists(CACHE_2021):
        z = np.load(CACHE_2021)
        required = {"y", "p_base_raw", "last_base_raw", "p_cat_raw", "last_cat_raw"}
        if required.issubset(z.files):
            log(f"reuse {CACHE_2021}")
            return
        raise RuntimeError(f"incomplete cache: {CACHE_2021}")

    val_season = 2021
    y, seasons = raw[TARGET], raw["season"]
    (X, base_num, current16, expert_cols, tr, va, lm,
     prior) = _prepare_fold_matrix(raw, raw_features, raw_num, val_season)
    log(f"\n[build OOF {val_season}] train={tr.sum():,} val={va.sum():,}")
    log("  HGB8")
    p_hgb, last_hgb = mean_predictions(
        make_model, HGB_SPECS, base_num, X, tr, va, lm, y)
    log("  current-season LGBM3")
    p_lgbm, last_lgbm = mean_predictions(
        make_lgbm_model, LGBM_SPECS, base_num + current16, X, tr, va, lm, y)
    p_base = 0.25 * p_hgb + 0.75 * p_lgbm
    last_base = 0.25 * last_hgb + 0.75 * last_lgbm

    p_cat, last_cat = [], []
    for seed in (2026, 2718):
        log(f"  CatBoost seed={seed}")
        model, pv, pl = fit_predict_catboost(
            X, expert_cols, tr, va, lm, y, seed=seed)
        p_cat.append(pv); last_cat.append(pl)
        del model
        gc.collect()
    r_extrap = extrapolated_rate(y, seasons, tr, val_season)
    np.savez_compressed(
        CACHE_2021,
        y=y.loc[va].to_numpy(dtype=float),
        p_base_raw=p_base, last_base_raw=last_base,
        p_cat_raw=np.mean(p_cat, axis=0),
        last_cat_raw=np.mean(last_cat, axis=0),
        r_extrap=np.asarray(r_extrap), last_season=np.asarray(2020),
    )
    log(f"saved {CACHE_2021}")
    del X, p_hgb, last_hgb, p_lgbm, last_lgbm, p_cat, last_cat
    gc.collect()


def load_predictions(season):
    if season == 2021:
        z = np.load(CACHE_2021)
        return {k: z[k] for k in z.files}
    fam = np.load(f"{PRED_DIR}/current_season_family_{season}.npz")
    cat = np.load(f"{PRED_DIR}/catboost_confirm_{season}.npz")
    if "last_expert2" not in cat.files:
        raise RuntimeError(
            "CatBoost confirm cache lacks last_expert2; rerun diag_catboost_confirm.py")
    return dict(
        y=cat["y"],
        p_base_raw=(0.25 * fam["p_hgb8"] + 0.75 * fam["p_current16"]),
        last_base_raw=(0.25 * fam["last_hgb8"] + 0.75 * fam["last_current16"]),
        p_cat_raw=cat["p_expert2"],
        last_cat_raw=cat["last_expert2"],
        r_extrap=fam["r_extrap"],
        last_season=fam["last_season"],
    )


def current_reliability(raw, train_mask, spec):
    """현재 시즌 workload reliability를 row-local lookup 계약으로 복원한다."""
    previous = _previous_lookup(raw, train_mask, spec)
    ncol = spec["n"]
    current = raw[ncol].to_numpy(dtype=float)
    prev = previous[f"prev_{ncol}"].to_numpy(dtype=float)
    known = np.isfinite(prev)
    prev = np.where(known, prev, 0.0)
    delta = current - prev
    valid = np.isfinite(current) & (delta > 0)
    rel = np.where(valid, delta / (delta + K), 0.0)
    cold = (~known).astype(float)
    return rel, cold


def context_frames(raw, val_season):
    seasons = raw["season"]
    tr = (seasons < val_season).to_numpy()
    last_season = int(seasons.loc[tr].max())
    p_rel, p_cold = current_reliability(raw, tr, GROUPS["pitcher"])
    b_rel, b_cold = current_reliability(raw, tr, GROUPS["batter"])

    balls = raw["balls_before"].fillna(-1).astype(int)
    strikes = raw["strikes_before"].fillna(-1).astype(int)
    base = pd.DataFrame(index=raw.index)
    base["global"] = 1.0
    base["is_F"] = raw["game_type"].eq("F").astype(float)
    base["pitcher_ahead"] = strikes.gt(balls).astype(float)
    base["batter_ahead"] = balls.gt(strikes).astype(float)
    base["two_strike"] = strikes.eq(2).astype(float)
    base["three_ball"] = balls.eq(3).astype(float)
    base["full_count"] = (balls.eq(3) & strikes.eq(2)).astype(float)
    base["early_count"] = (balls + strikes <= 1).astype(float)
    base["pitcher_rel"] = p_rel
    base["batter_rel"] = b_rel
    base["pitcher_cold"] = p_cold
    base["batter_cold"] = b_cold
    base["F_x_pitcher_ahead"] = base["is_F"] * base["pitcher_ahead"]
    base["F_x_batter_ahead"] = base["is_F"] * base["batter_ahead"]
    base["F_x_full"] = base["is_F"] * base["full_count"]
    base["pitcher_rel_x_two"] = base["pitcher_rel"] * base["two_strike"]
    base["pitcher_rel_x_three"] = base["pitcher_rel"] * base["three_ball"]
    for b in range(4):
        for s in range(3):
            base[f"count_{b}_{s}"] = (balls.eq(b) & strikes.eq(s)).astype(float)
    return (base.loc[seasons.eq(val_season)].reset_index(drop=True),
            base.loc[seasons.eq(last_season)].reset_index(drop=True))


COARSE = [
    "global", "is_F", "pitcher_ahead", "batter_ahead", "two_strike",
    "three_ball", "full_count", "early_count", "pitcher_rel", "batter_rel",
    "pitcher_cold", "batter_cold",
]
COUNT = (["global", "is_F", "pitcher_rel", "batter_rel",
          "pitcher_cold", "batter_cold"]
         + [f"count_{b}_{s}" for b in range(4) for s in range(3)])
MECHANISM = COARSE + [
    "F_x_pitcher_ahead", "F_x_batter_ahead", "F_x_full",
    "pitcher_rel_x_two", "pitcher_rel_x_three",
]
CONFIGS = {"global": ["global"], "coarse": COARSE,
           "count": COUNT, "mechanism": MECHANISM}


def fit_gate(train_rows, cols, lam):
    y = np.concatenate([r["y"] for r in train_rows])
    p0 = np.concatenate([r["p_base_raw"] for r in train_rows])
    pe = np.concatenate([r["p_cat_raw"] for r in train_rows])
    z = np.concatenate([r["context"][cols].to_numpy(dtype=float)
                        for r in train_rows], axis=0)
    delta = pe - p0
    design = z * delta[:, None]
    model = Ridge(alpha=lam * len(y), fit_intercept=False)
    model.fit(design, y - p0)
    return model.coef_.astype(float)


def gated_prediction(p0, pe, context, cols, coef):
    z = context[cols].to_numpy(dtype=float)
    weight = np.clip(z @ coef, 0.0, 1.0)
    return p0 + weight * (pe - p0), weight


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    build_2021_cache(raw, raw_features, raw_num)

    folds = {}
    for season in OOF_SEASONS:
        pred = load_predictions(season)
        cv, cl = context_frames(raw, season)
        if not (len(cv) == len(pred["y"]) and
                len(cl) == len(pred["last_base_raw"])):
            raise RuntimeError(f"context/prediction length mismatch season={season}")
        pred["context"], pred["last_context"] = cv, cl
        folds[season] = pred

    rows = []
    log("\n" + "=" * 110)
    log("STRICT-PAST CONDITIONAL GATE vs fixed v9 (CatBoost weight=0.60)")
    for config, cols in CONFIGS.items():
        for lam in LAMBDAS:
            gains, ses = [], []
            for season in VAL_SEASONS:
                train_rows = [folds[s] for s in OOF_SEASONS if s < season]
                coef = fit_gate(train_rows, cols, lam)
                fd = folds[season]
                fixed_raw = 0.4 * fd["p_base_raw"] + 0.6 * fd["p_cat_raw"]
                fixed_last = (0.4 * fd["last_base_raw"]
                              + 0.6 * fd["last_cat_raw"])
                fixed, fixed_shift, _, _ = recenter_like_production(
                    fixed_raw, fixed_last, float(fd["r_extrap"]), 0.5)
                gated_raw, weights = gated_prediction(
                    fd["p_base_raw"], fd["p_cat_raw"], fd["context"], cols, coef)
                gated_last, _ = gated_prediction(
                    fd["last_base_raw"], fd["last_cat_raw"],
                    fd["last_context"], cols, coef)
                gated, gate_shift, _, _ = recenter_like_production(
                    gated_raw, gated_last, float(fd["r_extrap"]), 0.5)
                yv = fd["y"].astype(float)
                base = yv.mean() * (1 - yv.mean())
                gain, se = paired(yv, fixed, gated, base)
                gains.append(gain); ses.append(se)
                rows.append(dict(config=config, lam=lam, season=season,
                                 gain=gain, se=se, weight_mean=weights.mean(),
                                 weight_q10=np.quantile(weights, .1),
                                 weight_q90=np.quantile(weights, .9),
                                 fixed_shift=fixed_shift, gate_shift=gate_shift))
            log(f"{config:10s} lambda={lam:7g} gain="
                + "/".join(f"{x:+7.2f}" for x in gains)
                + f" mean={np.mean(gains):+7.2f} SE="
                + "/".join(f"{x:.2f}" for x in ses))

    result = pd.DataFrame(rows)
    out = f"{PRED_DIR}/conditional_gate_summary.csv"
    result.to_csv(out, index=False)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
