"""검증된 CatBoost categorical expert 2개를 현재 v8 번들에 추가한다.

기존 HGB8/current-season LGBM3는 재학습하지 않고 보존한다. 전체 2019~2024로
CatBoost 2시드를 학습한 뒤 v8 40% + CatBoost 60%를 혼합하고, 혼합된 2024
학습 예측으로 production과 동일한 recenter_f=0.5 logit shift를 다시 고정한다.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                      add_shrinkage, shrinkage_cols,
                      add_season_success_train_features, season_success_cols,
                      CATBOOST_CAT_COLS as CAT_EXPERT_COLS,
                      add_catboost_context as add_cat_context)


DATA_DIR = "./open/data"
BUNDLE_PATH = "./open/baseline_submit/model/bundle.pkl"
TEMP_PATH = "./open/baseline_submit/model/bundle_catboost_building.pkl"
CATBOOST_WEIGHT = 0.60
SEEDS = [2026, 2718]


def log(*args):
    print(*args, flush=True)


def make_catboost(seed):
    return CatBoostClassifier(
        loss_function="Logloss", eval_metric="BrierScore",
        iterations=350, depth=7, learning_rate=0.05, l2_leaf_reg=10.0,
        random_seed=seed, random_strength=1.0,
        bootstrap_type="Bayesian", bagging_temperature=1.0,
        one_hot_max_size=12, max_ctr_complexity=2,
        boosting_type="Plain", thread_count=4,
        allow_writing_files=False, verbose=100,
    )


def solve_logit_shift(p_last, target):
    q = np.clip(p_last, 1e-9, 1 - 1e-9)
    logits = np.log(q / (1 - q))
    lo, hi = -6.0, 6.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(logits + mid)))) < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def main():
    bundle = joblib.load(BUNDLE_PATH)
    meta = bundle.get("meta", {})
    if not (meta.get("season_success") and
            abs(float(bundle.get("lgbm_family_weight")) - 0.75) < 1e-12):
        raise RuntimeError("현재 bundle이 검증된 v8 season-success/HGB25-LGBM75가 아님")
    if bundle.get("catboost_members"):
        raise RuntimeError("현재 bundle에 이미 CatBoost expert가 들어 있음")

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y = raw[TARGET]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    sh = add_shrinkage(base_df, bundle["prior"], bundle["k"])
    Xbase = pd.concat([base_df, sh], axis=1)
    train_for_season = pd.concat([Xbase, y.rename(TARGET)], axis=1)
    season = add_season_success_train_features(
        train_for_season, bundle["prior"], bundle["k"])
    Xbase = pd.concat([Xbase, season], axis=1)

    raw_num = [c for c in raw_features if c not in CAT_COLS]
    numeric_ids = {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
    expert_base_num = [c for c in raw_num if c not in numeric_ids]
    expert_base_num += list(DERIVED_COLS) + shrinkage_cols()
    expert_features = CAT_EXPERT_COLS + expert_base_num + season_success_cols()
    Xcat = pd.concat([Xbase, add_cat_context(raw)], axis=1)
    log(f"CatBoost full train rows={len(raw):,} features={len(expert_features)}")

    catboost_members = []
    last_mask = raw["season"].eq(int(meta["last_season"])).to_numpy()
    last_cat_preds = []
    for i, seed in enumerate(SEEDS):
        t = time.time()
        model = make_catboost(seed)
        model.fit(Xcat[expert_features], y, cat_features=CAT_EXPERT_COLS)
        last_cat_preds.append(model.predict_proba(
            Xcat.loc[last_mask, expert_features])[:, 1])
        catboost_members.append(model)
        log(f"  CatBoost {i+1}/{len(SEEDS)} seed={seed} 완료 {time.time()-t:.0f}s")

    members = bundle["members"]
    member_num_cols = bundle["member_num_cols"]
    last_member_preds = [
        m.predict_proba(Xbase.loc[last_mask, bundle["cat_cols"] + cols])[:, 1]
        for m, cols in zip(members, member_num_cols)
    ]
    n_hgb, n_lgbm = int(meta["n_hgb"]), int(meta["n_lgbm"])
    p_hgb = np.mean(last_member_preds[:n_hgb], axis=0)
    p_lgbm = np.mean(last_member_preds[n_hgb:n_hgb + n_lgbm], axis=0)
    p_v8 = 0.25 * p_hgb + 0.75 * p_lgbm
    p_cat = np.mean(last_cat_preds, axis=0)
    p_last = (1 - CATBOOST_WEIGHT) * p_v8 + CATBOOST_WEIGHT * p_cat

    natural = float(p_last.mean())
    r_extrap = float(meta["r_extrap"])
    target = natural + 0.5 * (r_extrap - natural)
    shift = solve_logit_shift(p_last, target)
    log(f"2024 mixed mean={natural:.8f}, recenter target={target:.8f}, shift={shift:+.8f}")

    bundle["catboost_members"] = catboost_members
    bundle["catboost_feature_cols"] = expert_features
    bundle["catboost_cat_cols"] = list(CAT_EXPERT_COLS)
    bundle["catboost_weight"] = CATBOOST_WEIGHT
    bundle["logit_shift"] = shift
    bundle["meta"] = dict(meta, version="v9_catboost_expert",
                           n_catboost=len(catboost_members),
                           catboost_weight=CATBOOST_WEIGHT,
                           p_last_mean=natural,
                           n_catboost_features=len(expert_features))
    joblib.dump(bundle, TEMP_PATH, compress=3)
    # 완전한 직렬화/역직렬화를 확인한 뒤에만 production 번들을 교체한다.
    check = joblib.load(TEMP_PATH)
    if len(check.get("catboost_members", [])) != len(SEEDS):
        raise RuntimeError("CatBoost 번들 직렬화 검증 실패")
    os.replace(TEMP_PATH, BUNDLE_PATH)
    log(f"저장 완료: {BUNDLE_PATH} ({os.path.getsize(BUNDLE_PATH)/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
