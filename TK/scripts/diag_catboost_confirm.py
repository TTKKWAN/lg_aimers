"""두 번째 CatBoost 시드를 추가해 categorical expert의 재현성을 확인한다."""
import gc
import os

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, DERIVED_COLS, add_derived, add_shrinkage,
                      shrinkage_cols, fit_prior)
from diag_season_to_date import (DATA_DIR, PRED_DIR, VAL_SEASONS, K, GROUPS,
                                 _previous_lookup, _season_block, paired)
from diag_season_to_date_followup import exact_cols
from diag_current_season_family import recenter_like_production, extrapolated_rate
from diag_catboost_expert import (CAT_EXPERT_COLS, add_cat_context,
                                  fit_predict_catboost)


WEIGHTS = [0.30, 0.40, 0.50, 0.60, 0.75]


def log(*args):
    print(*args, flush=True)


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    raw_num = [c for c in raw_features if c not in ["top_bottom", "game_type", "base_state"]]
    numeric_ids = {"pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"}
    expert_base_num = [c for c in raw_num if c not in numeric_ids]
    expert_base_num += list(DERIVED_COLS) + shrinkage_cols()
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    fold_data = {}

    for val_season in VAL_SEASONS:
        tr = (seasons < val_season).to_numpy()
        va = (seasons == val_season).to_numpy()
        last_season = int(seasons.loc[tr].max())
        lm = tr & seasons.eq(last_season).to_numpy()
        prior = fit_prior(raw.loc[tr])
        X0 = pd.concat([base_df, add_shrinkage(base_df, prior, K)], axis=1)
        blocks = {}
        for group in ("pitcher", "batter"):
            previous = _previous_lookup(raw, tr, GROUPS[group])
            blocks[group], _ = _season_block(X0, previous, GROUPS[group], prior,
                                              zero_baseline=True)
        current16 = (exact_cols(blocks["pitcher"], "pitcher", "asof_pitcher_success_rate")
                     + exact_cols(blocks["batter"], "batter", "asof_batter_success_rate"))
        X = pd.concat([X0, blocks["pitcher"], blocks["batter"], add_cat_context(raw)], axis=1)
        features = CAT_EXPERT_COLS + expert_base_num + current16
        # 큰 모델 학습이 끝난 직후 cloud-backed 캐시를 처음 열면 macOS에서 간헐적으로
        # I/O timeout이 발생할 수 있어, 필요한 캐시는 학습 전에 메모리에 올린다.
        with np.load(f"{PRED_DIR}/catboost_expert_{val_season}.npz") as cache:
            z = {key: cache[key] for key in cache.files}
        with np.load(f"{PRED_DIR}/current_season_family_{val_season}.npz") as cache:
            c = {key: cache[key] for key in cache.files}
        log(f"\n[fold={val_season}] second seed")
        model, p2, last2 = fit_predict_catboost(X, features, tr, va, lm, y, seed=2718)

        p_expert = 0.5 * (z["p_expert"] + p2)
        last_expert = 0.5 * (z["last_expert"] + last2)
        p0_raw = z["p_baseline_raw"]
        p0 = z["p_baseline"]
        last0 = 0.25 * c["last_hgb8"] + 0.75 * c["last_current16"]
        r_extrap = extrapolated_rate(y, seasons, tr, val_season)
        fold_data[val_season] = dict(y=z["y"], p0=p0, p0_raw=p0_raw,
                                     last0=last0, pe=p_expert, le=last_expert,
                                     r_extrap=r_extrap)
        np.savez_compressed(f"{PRED_DIR}/catboost_confirm_{val_season}.npz",
                            y=z["y"], p_baseline=p0, p_baseline_raw=p0_raw,
                            p_expert_seed2026=z["p_expert"], p_expert_seed2718=p2,
                            p_expert2=p_expert,
                            last_expert_seed2026=z["last_expert"],
                            last_expert_seed2718=last2,
                            last_expert2=last_expert)
        del model, X, X0, blocks
        gc.collect()

    log("\nSUMMARY two-seed CatBoost mean vs v8 + recenter_f=0.5")
    for w in WEIGHTS:
        vals = []
        for season, fd in fold_data.items():
            raw_pred = (1 - w) * fd["p0_raw"] + w * fd["pe"]
            last_pred = (1 - w) * fd["last0"] + w * fd["le"]
            pred, shift, _, _ = recenter_like_production(raw_pred, last_pred,
                                                          fd["r_extrap"], 0.5)
            base = fd["y"].mean() * (1 - fd["y"].mean())
            gain, se = paired(fd["y"], fd["p0"], pred, base)
            vals.append(gain)
            log(f"  w={w:.2f} {season}: {gain:+.2f} (SE {se:.2f}) shift={shift:+.5f}")
        log(f"  w={w:.2f} mean={np.mean(vals):+.2f}")


if __name__ == "__main__":
    main()
