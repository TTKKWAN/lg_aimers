"""CatBoost 선수×상황 categorical expert를 v8 production 기준으로 검증한다.

선수 ID는 수치형 서열로 넣지 않고 명목형 lookup key로만 사용한다. CatBoost의
ordered target statistics가 pitcher/batter와 count/base/pressure의 저차 조합을
축소 학습한다. test/validation 행끼리 통계를 만들지 않는다.
"""
import gc
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                      add_shrinkage, shrinkage_cols, fit_prior,
                      CATBOOST_CAT_COLS as CAT_EXPERT_COLS,
                      add_catboost_context as add_cat_context)
from diag_season_to_date import (DATA_DIR, PRED_DIR, VAL_SEASONS, K, GROUPS,
                                 _previous_lookup, _season_block, paired)
from diag_season_to_date_followup import exact_cols
from diag_current_season_family import recenter_like_production, extrapolated_rate


FAMILY_WEIGHT = 0.75
BLEND_WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.30]


def log(*args):
    print(*args, flush=True)


def fit_predict_catboost(X, features, train_mask, val_mask, last_mask, y, seed=2026,
                         cat_cols=None):
    if cat_cols is None:
        cat_cols = CAT_EXPERT_COLS
    model = CatBoostClassifier(
        loss_function="Logloss", eval_metric="BrierScore",
        iterations=350, depth=7, learning_rate=0.05, l2_leaf_reg=10.0,
        random_seed=seed, random_strength=1.0,
        bootstrap_type="Bayesian", bagging_temperature=1.0,
        one_hot_max_size=12, max_ctr_complexity=2,
        boosting_type="Plain", thread_count=4,
        allow_writing_files=False, verbose=100,
    )
    t = time.time()
    model.fit(X.loc[train_mask, features], y.loc[train_mask],
              cat_features=cat_cols)
    pv = model.predict_proba(X.loc[val_mask, features])[:, 1]
    pl = model.predict_proba(X.loc[last_mask, features])[:, 1]
    log(f"  CatBoost seed={seed} fit+predict {time.time()-t:.0f}s")
    return model, pv, pl


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    # ID와 팀 ID는 오직 categorical key로만 사용한다.
    numeric_id_cols = {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
    expert_base_num = [c for c in raw_num if c not in numeric_id_cols]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    expert_base_num += list(DERIVED_COLS) + shrinkage_cols()
    fold_data = {}
    os.makedirs(PRED_DIR, exist_ok=True)

    for val_season in VAL_SEASONS:
        train_mask = (seasons < val_season).to_numpy()
        val_mask = (seasons == val_season).to_numpy()
        last_train = int(seasons.loc[train_mask].max())
        last_mask = train_mask & seasons.eq(last_train).to_numpy()
        yv = y.loc[val_mask].to_numpy(dtype=float)
        prior = fit_prior(raw.loc[train_mask])
        X0 = pd.concat([base_df, add_shrinkage(base_df, prior, K)], axis=1)
        blocks = {}
        for group in ("pitcher", "batter"):
            previous = _previous_lookup(raw, train_mask, GROUPS[group])
            blocks[group], _ = _season_block(X0, previous, GROUPS[group], prior,
                                              zero_baseline=True)
        current16 = (exact_cols(blocks["pitcher"], "pitcher", "asof_pitcher_success_rate")
                     + exact_cols(blocks["batter"], "batter", "asof_batter_success_rate"))
        cat_context = add_cat_context(raw)
        X = pd.concat([X0, blocks["pitcher"], blocks["batter"], cat_context], axis=1)
        features = CAT_EXPERT_COLS + expert_base_num + current16

        cache = np.load(f"{PRED_DIR}/current_season_family_{val_season}.npz")
        p_hgb, last_hgb = cache["p_hgb8"], cache["last_hgb8"]
        p_lgbm, last_lgbm = cache["p_current16"], cache["last_current16"]
        p0_raw = (1 - FAMILY_WEIGHT) * p_hgb + FAMILY_WEIGHT * p_lgbm
        p0_last = (1 - FAMILY_WEIGHT) * last_hgb + FAMILY_WEIGHT * last_lgbm
        r_extrap = extrapolated_rate(y, seasons, train_mask, val_season)
        p0, _, _, _ = recenter_like_production(p0_raw, p0_last, r_extrap, 0.5)
        model, pe, last_e = fit_predict_catboost(
            X, features, train_mask, val_mask, last_mask, y)
        base = yv.mean() * (1 - yv.mean())
        log(f"\n[fold={val_season}] expert-only")
        eg, ese = paired(yv, p0_raw, pe, base)
        log(f"  raw vs v8={eg:+.2f} SE={ese:.2f}")
        saved = dict(y=yv, p_baseline=p0, p_baseline_raw=p0_raw,
                     p_expert=pe, last_expert=last_e)
        fold_data[val_season] = dict(y=yv, p0=p0, p0_raw=p0_raw,
                                     p0_last=p0_last, pe=pe, last_e=last_e,
                                     r_extrap=r_extrap)
        for w in BLEND_WEIGHTS:
            raw_pred = (1 - w) * p0_raw + w * pe
            last_pred = (1 - w) * p0_last + w * last_e
            pred, shift, _, _ = recenter_like_production(raw_pred, last_pred,
                                                          r_extrap, 0.5)
            gain, se = paired(yv, p0, pred, base)
            log(f"  expert w={w:.2f}: recenter gain={gain:+.2f} SE={se:.2f} "
                f"shift={shift:+.5f}")
            saved[f"p_w{int(w*100):02d}"] = pred
        np.savez_compressed(f"{PRED_DIR}/catboost_expert_{val_season}.npz", **saved)
        del model, X, X0, blocks, cat_context
        gc.collect()

    log("\nSUMMARY vs v8, production-style recenter_f=0.5")
    for w in BLEND_WEIGHTS:
        vals = []
        for season, fd in fold_data.items():
            raw_pred = (1 - w) * fd["p0_raw"] + w * fd["pe"]
            last_pred = (1 - w) * fd["p0_last"] + w * fd["last_e"]
            pred, _, _, _ = recenter_like_production(raw_pred, last_pred,
                                                       fd["r_extrap"], 0.5)
            base = fd["y"].mean() * (1 - fd["y"].mean())
            gain, se = paired(fd["y"], fd["p0"], pred, base)
            vals.append(gain)
            log(f"  w={w:.2f} {season}: {gain:+.2f} (SE {se:.2f})")
        log(f"  w={w:.2f} mean={np.mean(vals):+.2f}")


if __name__ == "__main__":
    main()
