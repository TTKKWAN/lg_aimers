"""현 시즌 strike/ball/middle/reverse command profile을 CatBoost에만 추가한다.

v9 CatBoost seed=2026 expert를 기준으로 outcome별 소규모 블록을 먼저 스크리닝한다.
각 rate 블록은 current-season raw/EB/career-dev 3개뿐이며, workload/known/invalid는
기존 current-season success 8개에 이미 포함되어 중복하지 않는다. 모든 이전 시즌
endpoint는 fold-train에서 고정되며 validation 행끼리는 참조하지 않는다.
"""
import gc
import os
import sys

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, DERIVED_COLS, add_derived, add_shrinkage,
                      shrinkage_cols, fit_prior, CATBOOST_CAT_COLS,
                      add_catboost_context)
from diag_season_to_date import (DATA_DIR, PRED_DIR, VAL_SEASONS, K, GROUPS,
                                 _previous_lookup, _season_block, paired)
from diag_season_to_date_followup import exact_cols
from diag_current_season_family import recenter_like_production
from diag_catboost_expert import fit_predict_catboost


RATE_NAMES = {
    "strike": "asof_pitcher_strike_rate",
    "ball": "asof_pitcher_ball_rate",
    "middle": "asof_pitcher_middle_rate",
    "reverse": "asof_pitcher_reverse_rate",
}
CONFIGS = {
    "strike": ["strike"],
    "ball": ["ball"],
    "middle": ["middle"],
    "reverse": ["reverse"],
    "strike_ball": ["strike", "ball"],
    "middle_reverse": ["middle", "reverse"],
    "all4": ["strike", "ball", "middle", "reverse"],
}


def log(*args):
    print(*args, flush=True)


def rate_cols(names):
    cols = []
    for name in names:
        rate = RATE_NAMES[name]
        cols.extend([f"std_{rate}", f"std_sh_{rate}", f"std_dev_{rate}"])
    return cols


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    raw_num = [c for c in raw_features
               if c not in ["top_bottom", "game_type", "base_state"]]
    numeric_ids = {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
    expert_num = [c for c in raw_num if c not in numeric_ids]
    expert_num += list(DERIVED_COLS) + shrinkage_cols()
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    os.makedirs(PRED_DIR, exist_ok=True)
    summary = []

    run_seasons = [int(x) for x in sys.argv[1:]] or list(VAL_SEASONS)
    for val_season in run_seasons:
        tr = (seasons < val_season).to_numpy()
        va = seasons.eq(val_season).to_numpy()
        last_season = int(seasons.loc[tr].max())
        lm = tr & seasons.eq(last_season).to_numpy()
        prior = fit_prior(raw.loc[tr])
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
        base_features = list(CATBOOST_CAT_COLS) + expert_num + current16
        X = pd.concat([X0, blocks["pitcher"], blocks["batter"],
                       add_catboost_context(raw)], axis=1)

        with np.load(f"{PRED_DIR}/current_season_family_{val_season}.npz") as z:
            p_v8 = 0.25 * z["p_hgb8"] + 0.75 * z["p_current16"]
            last_v8 = 0.25 * z["last_hgb8"] + 0.75 * z["last_current16"]
            r_extrap = float(z["r_extrap"])
        with np.load(f"{PRED_DIR}/catboost_expert_{val_season}.npz") as z:
            p_cat0 = z["p_expert"]
            last_cat0 = z["last_expert"]
            yv = z["y"].astype(float)
        p0_raw = 0.4 * p_v8 + 0.6 * p_cat0
        last0_raw = 0.4 * last_v8 + 0.6 * last_cat0
        p0, _, _, _ = recenter_like_production(
            p0_raw, last0_raw, r_extrap, 0.5)
        brier_base = yv.mean() * (1 - yv.mean())
        saved = dict(y=yv, p_reference=p0, p_reference_raw=p0_raw,
                     last_reference_raw=last0_raw)

        log(f"\n[fold={val_season}] CatBoost seed=2026 command blocks")
        # 앞선 두 폴드에서 이미 유의하게 실패한 후보는 최신 폴드 계산을 생략한다.
        # 단독 2024 재실행은 두 폴드 모두 살아남은 후보만 확인한다.
        active_configs = (CONFIGS if len(run_seasons) > 1 or val_season != 2024
                          else {k: CONFIGS[k] for k in ("middle_reverse", "all4")})
        for name, rates in active_configs.items():
            extra = rate_cols(rates)
            features = base_features + extra
            model, pe, le = fit_predict_catboost(
                X, features, tr, va, lm, y, seed=2026)
            p_raw = 0.4 * p_v8 + 0.6 * pe
            last_raw = 0.4 * last_v8 + 0.6 * le
            pred, shift, _, _ = recenter_like_production(
                p_raw, last_raw, r_extrap, 0.5)
            gain, se = paired(yv, p0, pred, brier_base)
            summary.append(dict(config=name, season=val_season, gain=gain,
                                se=se, shift=shift, n_extra=len(extra)))
            saved[f"p_{name}"] = pred
            saved[f"p_{name}_raw"] = p_raw
            saved[f"last_{name}_raw"] = last_raw
            log(f"  {name:15s} extra={len(extra):2d} gain={gain:+8.2f} "
                f"SE={se:.2f} shift={shift:+.5f}")
            del model
            gc.collect()
        np.savez_compressed(
            f"{PRED_DIR}/catboost_command_profile_{val_season}.npz", **saved)
        del X, X0, blocks
        gc.collect()

    result = pd.DataFrame(summary)
    suffix = "" if run_seasons == list(VAL_SEASONS) else "_" + "_".join(map(str, run_seasons))
    result.to_csv(f"{PRED_DIR}/catboost_command_profile_summary{suffix}.csv", index=False)
    log("\nSUMMARY seed=2026 vs seed=2026 v9-style reference")
    for name in result["config"].unique():
        d = result[result["config"].eq(name)]
        log(f"{name:15s} gain=" + "/".join(f"{x:+.2f}" for x in d["gain"])
            + f" mean={d['gain'].mean():+.2f} SE="
            + "/".join(f"{x:.2f}" for x in d["se"]))


if __name__ == "__main__":
    main()
