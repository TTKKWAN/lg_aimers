"""v9에 검증된 current-season all4 command CatBoost2를 50% 추가한다.

입력 v9 production 번들은 변경하지 않는다. 전체 2019~2024 데이터로 command
expert 두 시드를 학습하고 candidate 번들에만 저장한다. 최종 혼합은
v8 40% + [기존 CatBoost2 50% + command CatBoost2 50%] 60%이며, 혼합된
2024 학습 예측으로 production 동일 recenter_f=0.5 shift를 다시 고정한다.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd

from build_catboost_final import make_catboost, solve_logit_shift
from pipeline import (ID, TARGET, add_derived, add_shrinkage,
                      add_season_success_train_features,
                      add_season_command_train_features,
                      fit_season_command_lookup, season_command_cols,
                      add_catboost_context)


DATA_DIR = "./open/data"
SOURCE_PATH = "./open/baseline_submit/model/bundle.pkl"
OUTPUT_PATH = "./open/baseline_submit/model/bundle_v10_candidate.pkl"
SEEDS = [2026, 2718]
COMMAND_WEIGHT = 0.50


def log(*args):
    print(*args, flush=True)


def load_bundle(path, attempts=5):
    for attempt in range(attempts):
        try:
            return joblib.load(path)
        except TimeoutError:
            if attempt + 1 == attempts:
                raise
            time.sleep(2)


def main():
    bundle = load_bundle(SOURCE_PATH)
    meta = bundle.get("meta", {})
    if meta.get("version") != "v9_catboost_expert":
        raise RuntimeError(f"v9 production 번들이 아님: {meta.get('version')}")
    if len(bundle.get("catboost_members") or []) != 2:
        raise RuntimeError("기존 CatBoost2 계약이 없음")
    if bundle.get("catboost_command_members"):
        raise RuntimeError("입력 번들에 이미 command expert가 있음")

    test_cols = pd.read_csv(
        f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y = raw[TARGET]
    base = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    sh = add_shrinkage(base, bundle["prior"], bundle["k"])
    x0 = pd.concat([base, sh], axis=1)
    train_with_target = pd.concat([x0, y.rename(TARGET)], axis=1)
    success = add_season_success_train_features(
        train_with_target, bundle["prior"], bundle["k"])
    command = add_season_command_train_features(
        train_with_target, bundle["prior"], bundle["k"])
    xbase = pd.concat([x0, success], axis=1)
    xcat = pd.concat([xbase, command, add_catboost_context(raw)], axis=1)

    command_cols = season_command_cols()
    features = list(bundle["catboost_feature_cols"]) + command_cols
    if len(features) != len(set(features)):
        raise RuntimeError("command feature 중복")
    last_mask = raw["season"].eq(int(meta["last_season"])).to_numpy()
    log(f"command CatBoost full train rows={len(raw):,} features={len(features)}")

    command_members, last_command = [], []
    for i, seed in enumerate(SEEDS):
        started = time.time()
        model = make_catboost(seed)
        model.fit(xcat[features], y, cat_features=bundle["catboost_cat_cols"])
        last_command.append(model.predict_proba(xcat.loc[last_mask, features])[:, 1])
        command_members.append(model)
        log(f"  command {i+1}/{len(SEEDS)} seed={seed} 완료 "
            f"{time.time()-started:.0f}s")

    last_member = [
        model.predict_proba(
            xbase.loc[last_mask, bundle["cat_cols"] + cols])[:, 1]
        for model, cols in zip(bundle["members"], bundle["member_num_cols"])
    ]
    n_hgb, n_lgbm = int(meta["n_hgb"]), int(meta["n_lgbm"])
    p_hgb = np.mean(last_member[:n_hgb], axis=0)
    p_lgbm = np.mean(last_member[n_hgb:n_hgb + n_lgbm], axis=0)
    p_v8 = ((1.0 - float(bundle["lgbm_family_weight"])) * p_hgb
            + float(bundle["lgbm_family_weight"]) * p_lgbm)
    p_old = np.mean([
        model.predict_proba(
            xcat.loc[last_mask, bundle["catboost_feature_cols"]])[:, 1]
        for model in bundle["catboost_members"]
    ], axis=0)
    p_command = np.mean(last_command, axis=0)
    p_cat = (1.0 - COMMAND_WEIGHT) * p_old + COMMAND_WEIGHT * p_command
    p_last = ((1.0 - float(bundle["catboost_weight"])) * p_v8
              + float(bundle["catboost_weight"]) * p_cat)

    natural = float(p_last.mean())
    extrapolated = float(meta["r_extrap"])
    target = natural + 0.5 * (extrapolated - natural)
    shift = solve_logit_shift(p_last, target)
    log(f"2024 mixed mean={natural:.8f}, recenter target={target:.8f}, "
        f"shift={shift:+.8f}")

    bundle["season_command_lookup"] = fit_season_command_lookup(raw)
    bundle["catboost_command_members"] = command_members
    bundle["catboost_command_feature_cols"] = features
    bundle["catboost_command_weight"] = COMMAND_WEIGHT
    bundle["logit_shift"] = shift
    bundle["meta"] = dict(
        meta, version="v10_command_all4_50", n_catboost_command=len(SEEDS),
        catboost_command_weight=COMMAND_WEIGHT,
        n_catboost_command_features=len(features), p_last_mean=natural,
    )
    temp = OUTPUT_PATH + ".building"
    joblib.dump(bundle, temp, compress=3)
    check = load_bundle(temp)
    if len(check.get("catboost_command_members") or []) != len(SEEDS):
        raise RuntimeError("candidate 직렬화 검증 실패")
    os.replace(temp, OUTPUT_PATH)
    log(f"candidate 저장: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
