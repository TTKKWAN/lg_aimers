"""middle/reverse 및 all4 command profile의 CatBoost 두 번째 시드를 확인한다."""
import gc
import os

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
from diag_catboost_command_profile import CONFIGS, rate_cols


CANDIDATES = ["middle_reverse", "all4"]
REPLACE_WEIGHTS = [0.5, 1.0]


def log(*args):
    print(*args, flush=True)


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
    results = []

    for val_season in VAL_SEASONS:
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
        with np.load(f"{PRED_DIR}/catboost_confirm_{val_season}.npz") as z:
            yv = z["y"].astype(float)
            p_cat0 = z["p_expert2"]
            last_cat0 = z["last_expert2"]
        screen_path = (f"{PRED_DIR}/catboost_command_profile_{val_season}.npz")
        with np.load(screen_path) as z:
            screen = {key: z[key] for key in z.files}

        ref_raw = 0.4 * p_v8 + 0.6 * p_cat0
        ref_last = 0.4 * last_v8 + 0.6 * last_cat0
        ref, _, _, _ = recenter_like_production(
            ref_raw, ref_last, r_extrap, 0.5)
        base = yv.mean() * (1 - yv.mean())
        saved = dict(y=yv, p_reference=ref)
        log(f"\n[fold={val_season}] seed=2718 command confirmation")

        for name in CANDIDATES:
            features = base_features + rate_cols(CONFIGS[name])
            model, p2, l2 = fit_predict_catboost(
                X, features, tr, va, lm, y, seed=2718)
            # screen cache는 v8 40% + candidate-seed1 60%이므로 expert를 역산한다.
            p1 = (screen[f"p_{name}_raw"] - 0.4 * p_v8) / 0.6
            l1 = (screen[f"last_{name}_raw"] - 0.4 * last_v8) / 0.6
            pc = 0.5 * (p1 + p2)
            lc = 0.5 * (l1 + l2)
            for replace in REPLACE_WEIGHTS:
                expert = (1 - replace) * p_cat0 + replace * pc
                last_expert = (1 - replace) * last_cat0 + replace * lc
                raw_pred = 0.4 * p_v8 + 0.6 * expert
                raw_last = 0.4 * last_v8 + 0.6 * last_expert
                pred, shift, _, _ = recenter_like_production(
                    raw_pred, raw_last, r_extrap, 0.5)
                gain, se = paired(yv, ref, pred, base)
                results.append(dict(config=name, replace=replace,
                                    season=val_season, gain=gain, se=se,
                                    shift=shift))
                saved[f"p_{name}_r{int(replace*100)}"] = pred
                log(f"  {name:15s} replace={replace:.2f} gain={gain:+8.2f} "
                    f"SE={se:.2f} shift={shift:+.5f}")
            saved[f"p_{name}_expert2"] = pc
            saved[f"last_{name}_expert2"] = lc
            del model
            gc.collect()
        np.savez_compressed(
            f"{PRED_DIR}/catboost_command_confirm_{val_season}.npz", **saved)
        del X, X0, blocks, screen
        gc.collect()

    result = pd.DataFrame(results)
    result.to_csv(f"{PRED_DIR}/catboost_command_confirm_summary.csv", index=False)
    log("\nSUMMARY two-seed command expert vs current v9")
    for name in CANDIDATES:
        for replace in REPLACE_WEIGHTS:
            d = result[result["config"].eq(name) & result["replace"].eq(replace)]
            log(f"{name:15s} replace={replace:.2f} gain="
                + "/".join(f"{x:+.2f}" for x in d["gain"])
                + f" mean={d['gain'].mean():+.2f} SE="
                + "/".join(f"{x:.2f}" for x in d["se"]))


if __name__ == "__main__":
    main()
