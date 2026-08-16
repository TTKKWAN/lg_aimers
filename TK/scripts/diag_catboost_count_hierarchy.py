"""count hierarchy와 명시적 선수×전략 category를 command CatBoost에 추가한다.

stage2에서 채택한 all4 command expert를 seed=2026 기준선으로 두고, 정확한 count
category 외에 표본을 공유하는 전략 계층과 pitcher/batter×전략 cross를 비교한다.
"""
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
from diag_catboost_command_profile import CONFIGS as COMMAND_CONFIGS, rate_cols


HIERARCHY_COLS = [
    "count_advantage_cat", "count_strategy_cat", "count_depth_cat",
]
PITCHER_CROSS_COLS = [
    "pitcher_strategy_cat", "pitcher_advantage_cat",
]
PLAYER_CROSS_COLS = [
    "batter_strategy_cat", "hand_strategy_cat",
]
CONFIGS = {
    "hierarchy": HIERARCHY_COLS,
    "pitcher_cross": HIERARCHY_COLS + PITCHER_CROSS_COLS,
    "player_cross": HIERARCHY_COLS + PITCHER_CROSS_COLS + PLAYER_CROSS_COLS,
}


def log(*args):
    print(*args, flush=True)


def add_count_hierarchy(df):
    balls = df["balls_before"].fillna(-1).astype(int)
    strikes = df["strikes_before"].fillna(-1).astype(int)
    advantage = np.select(
        [strikes > balls, balls > strikes],
        ["pitcher_ahead", "batter_ahead"], default="even")
    strategy = np.select(
        [(balls == 3) & (strikes == 2), balls == 3,
         (strikes == 2) & (balls < 3)],
        ["full", "must_strike", "chase_freedom"], default="neutral")
    depth = np.select(
        [balls + strikes <= 1, balls + strikes >= 4],
        ["early", "deep"], default="middle")
    pitcher = df["pitcher_id"].fillna("__NA__").astype(str)
    batter = df["batter_id"].fillna("__NA__").astype(str)
    ph = df["pitcher_hand"].fillna("__NA__").astype(str)
    bh = df["batter_hand"].fillna("__NA__").astype(str)
    out = pd.DataFrame(index=df.index)
    out["count_advantage_cat"] = advantage.astype(str)
    out["count_strategy_cat"] = strategy.astype(str)
    out["count_depth_cat"] = depth.astype(str)
    out["pitcher_strategy_cat"] = pitcher + "|" + out["count_strategy_cat"]
    out["pitcher_advantage_cat"] = pitcher + "|" + out["count_advantage_cat"]
    out["batter_strategy_cat"] = batter + "|" + out["count_strategy_cat"]
    out["hand_strategy_cat"] = ph + "_" + bh + "|" + out["count_strategy_cat"]
    return out


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
    hierarchy = add_count_hierarchy(raw)
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
        all4 = rate_cols(COMMAND_CONFIGS["all4"])
        base_features = list(CATBOOST_CAT_COLS) + expert_num + current16 + all4
        X = pd.concat([X0, blocks["pitcher"], blocks["batter"],
                       add_catboost_context(raw), hierarchy], axis=1)

        with np.load(f"{PRED_DIR}/current_season_family_{val_season}.npz") as z:
            p_v8 = 0.25 * z["p_hgb8"] + 0.75 * z["p_current16"]
            last_v8 = 0.25 * z["last_hgb8"] + 0.75 * z["last_current16"]
            r_extrap = float(z["r_extrap"])
        with np.load(f"{PRED_DIR}/catboost_expert_{val_season}.npz") as z:
            p_base_cat1, last_base_cat1 = z["p_expert"], z["last_expert"]
            yv = z["y"].astype(float)
        with np.load(f"{PRED_DIR}/catboost_command_profile_{val_season}.npz") as z:
            p_all4_cat1 = (z["p_all4_raw"] - 0.4 * p_v8) / 0.6
            last_all4_cat1 = (z["last_all4_raw"] - 0.4 * last_v8) / 0.6

        ref_cat = 0.5 * p_base_cat1 + 0.5 * p_all4_cat1
        ref_last_cat = 0.5 * last_base_cat1 + 0.5 * last_all4_cat1
        ref_raw = 0.4 * p_v8 + 0.6 * ref_cat
        ref_last = 0.4 * last_v8 + 0.6 * ref_last_cat
        ref, _, _, _ = recenter_like_production(
            ref_raw, ref_last, r_extrap, 0.5)
        base = yv.mean() * (1 - yv.mean())
        saved = dict(y=yv, p_reference=ref)
        log(f"\n[fold={val_season}] count hierarchy seed=2026")

        for name, new_cats in CONFIGS.items():
            features = list(CATBOOST_CAT_COLS) + new_cats + expert_num + current16 + all4
            model, pc, lc = fit_predict_catboost(
                X, features, tr, va, lm, y, seed=2026,
                cat_cols=list(CATBOOST_CAT_COLS) + new_cats)
            # stage2의 command 절반만 새 hierarchy command로 교체한다.
            candidate_cat = 0.5 * p_base_cat1 + 0.5 * pc
            candidate_last_cat = 0.5 * last_base_cat1 + 0.5 * lc
            raw_pred = 0.4 * p_v8 + 0.6 * candidate_cat
            raw_last = 0.4 * last_v8 + 0.6 * candidate_last_cat
            pred, shift, _, _ = recenter_like_production(
                raw_pred, raw_last, r_extrap, 0.5)
            gain, se = paired(yv, ref, pred, base)
            results.append(dict(config=name, season=val_season,
                                gain=gain, se=se, shift=shift))
            saved[f"p_{name}"] = pred
            saved[f"p_{name}_expert"] = pc
            saved[f"last_{name}_expert"] = lc
            log(f"  {name:15s} cats={len(new_cats)} gain={gain:+8.2f} "
                f"SE={se:.2f} shift={shift:+.5f}")
            del model
            gc.collect()
        np.savez_compressed(
            f"{PRED_DIR}/catboost_count_hierarchy_{val_season}.npz", **saved)
        del X, X0, blocks
        gc.collect()

    result = pd.DataFrame(results)
    result.to_csv(f"{PRED_DIR}/catboost_count_hierarchy_summary.csv", index=False)
    log("\nSUMMARY vs stage2 all4-50 seed1 baseline")
    for name in CONFIGS:
        d = result[result["config"].eq(name)]
        log(f"{name:15s} gain=" + "/".join(f"{x:+.2f}" for x in d["gain"])
            + f" mean={d['gain'].mean():+.2f} SE="
            + "/".join(f"{x:.2f}" for x in d["se"]))


if __name__ == "__main__":
    main()
