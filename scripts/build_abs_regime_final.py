"""ABS regime expert를 command 번들에 추가한다.

기존 v10 호환 경로는 검증된 10%를 유지한다. 공통 regime v13 후보는 2024 월별
시간창 pooled 검증의 보수적 고정값 25%를 사용한다.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from build_catboost_final import solve_logit_shift
from pipeline import (
    ID, TARGET, CATBOOST_CAT_COLS, ABS_REGIME_RATES, abs_regime_cols,
    add_abs_regime_features, add_catboost_context, add_derived, add_shrinkage,
    add_season_command_train_features, add_season_success_train_features,
    fit_abs_regime_centers, add_era_features, add_regime_current_features,
    regime_current_cols, ERA_RECENT_SKILL_COLS,
)

DATA_DIR = "./open/data"
SOURCE_PATH = "./open/baseline_submit/model/bundle.pkl"
OUTPUT_PATH = "./open/baseline_submit/model/bundle_v13_abs_candidate.pkl"
SEEDS = [4242, 5151]
LEGACY_ABS_WEIGHT = 0.10
SHARED_REGIME_ABS_WEIGHT = 0.25
RECENT_COLS = [
    "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
]
CONTEXT_NUM = [
    "game_month", "inning", "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_pitcher_team", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li", "count_diff",
    "is_two_strike", "is_three_ball", "is_full_count", "risp",
    "platoon_match", "close_game",
]


def log(*args):
    print(*args, flush=True)


def make_model(seed):
    return CatBoostClassifier(
        loss_function="Logloss", eval_metric="BrierScore",
        iterations=350, depth=7, learning_rate=0.05, l2_leaf_reg=15.0,
        random_seed=seed, random_strength=1.0,
        bootstrap_type="Bayesian", bagging_temperature=1.0,
        one_hot_max_size=12, max_ctr_complexity=2,
        boosting_type="Plain", thread_count=4,
        allow_writing_files=False, verbose=100,
    )


def main():
    bundle = joblib.load(SOURCE_PATH)
    source_version = bundle.get("meta", {}).get("version")
    if source_version not in {"v10_command_all4_50", "v13_shared_regime_command"}:
        raise RuntimeError(f"unsupported command bundle: {source_version}")
    abs_weight = (SHARED_REGIME_ABS_WEIGHT
                  if bundle.get("meta", {}).get("shared_regime")
                  else LEGACY_ABS_WEIGHT)
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=raw_features + [TARGET])
    y = raw[TARGET]
    base = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    x0 = pd.concat([base, add_shrinkage(base, bundle["prior"], bundle["k"])], axis=1)
    if bundle.get("era_specs"):
        x0 = pd.concat([
            x0, add_era_features(raw, bundle["era_specs"], bundle["k"])
        ], axis=1)
    train_target = pd.concat([x0, y.rename(TARGET)], axis=1)
    success = add_season_success_train_features(train_target, bundle["prior"], bundle["k"])
    command = add_season_command_train_features(train_target, bundle["prior"], bundle["k"])
    x = pd.concat([x0, success, command, add_catboost_context(raw)], axis=1)
    if bundle.get("era_specs"):
        x = pd.concat([
            x, add_regime_current_features(x, bundle["era_specs"], bundle["k"])
        ], axis=1)

    last = raw["season"].eq(2024).to_numpy()
    mature = last & raw["game_month"].between(5, 9).to_numpy()
    centers = fit_abs_regime_centers(x, mature)
    x = pd.concat([x, add_abs_regime_features(x, centers, bundle["k"])], axis=1)
    if bundle.get("era_specs"):
        current_cols = [c for c in regime_current_cols(ABS_REGIME_RATES)
                        if "_dev_" not in c]
        recent_cols = list(ERA_RECENT_SKILL_COLS)
    else:
        current_cols = ([f"std_{r}" for r in ABS_REGIME_RATES]
                        + [f"std_sh_{r}" for r in ABS_REGIME_RATES])
        recent_cols = RECENT_COLS
    features = (list(CATBOOST_CAT_COLS) + CONTEXT_NUM + current_cols
                + recent_cols + abs_regime_cols())

    months = raw.loc[last, "game_month"].to_numpy()
    sample_weight = np.select([months <= 3, months == 4], [.20, .45], default=1.0)
    members, last_abs = [], []
    for seed in SEEDS:
        started = time.time()
        model = make_model(seed)
        model.fit(x.loc[last, features], y.loc[last], cat_features=CATBOOST_CAT_COLS,
                  sample_weight=sample_weight)
        last_abs.append(model.predict_proba(x.loc[last, features])[:, 1])
        members.append(model)
        log(f"ABS expert seed={seed} done in {time.time()-started:.1f}s")

    # v10의 재중심화 전 raw 예측을 정확히 재구성한다.
    base_preds = [
        m.predict_proba(x.loc[last, bundle["cat_cols"] + cols])[:, 1]
        for m, cols in zip(bundle["members"], bundle["member_num_cols"])
    ]
    nh, nl = int(bundle["meta"]["n_hgb"]), int(bundle["meta"]["n_lgbm"])
    pv8 = ((1.0 - bundle["lgbm_family_weight"]) * np.mean(base_preds[:nh], axis=0)
           + bundle["lgbm_family_weight"] * np.mean(base_preds[nh:nh + nl], axis=0))
    pcat = np.mean([
        m.predict_proba(x.loc[last, bundle["catboost_feature_cols"]])[:, 1]
        for m in bundle["catboost_members"]], axis=0)
    pcommand = np.mean([
        m.predict_proba(x.loc[last, bundle["catboost_command_feature_cols"]])[:, 1]
        for m in bundle["catboost_command_members"]], axis=0)
    pcat_family = ((1.0 - bundle["catboost_command_weight"]) * pcat
                   + bundle["catboost_command_weight"] * pcommand)
    pv10 = ((1.0 - bundle["catboost_weight"]) * pv8
            + bundle["catboost_weight"] * pcat_family)
    pabs = np.mean(last_abs, axis=0)
    plast = (1.0 - abs_weight) * pv10 + abs_weight * pabs
    natural = float(plast.mean())
    target = natural + .5 * (float(bundle["meta"]["r_extrap"]) - natural)
    shift = solve_logit_shift(plast, target)
    log(f"v11 2024 natural={natural:.8f} target={target:.8f} shift={shift:+.8f}")

    bundle["abs_regime_members"] = members
    bundle["abs_regime_feature_cols"] = features
    bundle["abs_regime_cat_cols"] = list(CATBOOST_CAT_COLS)
    bundle["abs_regime_centers"] = centers
    bundle["abs_regime_weight"] = abs_weight
    bundle["logit_shift"] = shift
    bundle["meta"] = dict(
        bundle["meta"],
        version=("v13_shared_regime_abs25" if bundle["meta"].get("shared_regime")
                 else "v11_abs_regime_10"),
                          n_abs_regime=len(members), abs_regime_weight=abs_weight,
                          n_abs_regime_features=len(features), p_last_mean=natural)
    temp = OUTPUT_PATH + ".building"
    joblib.dump(bundle, temp, compress=3)
    check = joblib.load(temp)
    if len(check.get("abs_regime_members", [])) != len(SEEDS):
        raise RuntimeError("v11 serialization check failed")
    os.replace(temp, OUTPUT_PATH)
    log(f"saved {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
