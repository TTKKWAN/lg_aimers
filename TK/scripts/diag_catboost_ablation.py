"""stage2 all4 command CatBoost의 피처 그룹 ablation을 forward 순차 검사한다."""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, DERIVED_COLS, RATE_N_PAIRS, PREV_SPECS,
                      add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                      CATBOOST_CAT_COLS, add_catboost_context)
from diag_season_to_date import (DATA_DIR, PRED_DIR, VAL_SEASONS, K, GROUPS,
                                 _previous_lookup, _season_block, paired)
from diag_season_to_date_followup import exact_cols
from diag_current_season_family import recenter_like_production
from diag_catboost_expert import fit_predict_catboost
from diag_catboost_command_profile import CONFIGS as COMMAND_CONFIGS, rate_cols


RAW_RATE_COLS = [r for r, _ in RATE_N_PAIRS] + [c for c, _, _ in PREV_SPECS]
ABLATIONS = {
    "drop_raw_rates": set(RAW_RATE_COLS),
    "drop_redundant_state": {
        "run_top_before", "run_bot_before", "score_diff_home",
        "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
        "away_win_expectancy",
    },
    "drop_calendar": {"game_month", "game_dayofweek"},
    "drop_team_cats": {"pitcher_team_id_cat", "batter_team_id_cat"},
    "drop_game_type": {"game_type_cat"},
    "drop_season": {"season"},
    "drop_win_context": {"home_win_expectancy", "away_win_expectancy", "li"},
}


def log(*args):
    print(*args, flush=True)


def load_npz(path, attempts=4):
    """macOS cloud-backed 파일의 간헐적 Errno 60을 짧게 재시도한다."""
    for attempt in range(attempts):
        try:
            with np.load(path) as z:
                return {key: z[key] for key in z.files}
        except TimeoutError:
            if attempt + 1 == attempts:
                raise
            time.sleep(1.0)


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
    start_season = int(sys.argv[1]) if len(sys.argv) > 1 else VAL_SEASONS[0]
    active = sys.argv[2:] if len(sys.argv) > 2 else list(ABLATIONS)
    unknown = [name for name in active if name not in ABLATIONS]
    if unknown:
        raise SystemExit(f"unknown ablations: {unknown}")
    results = []

    for val_season in [s for s in VAL_SEASONS if s >= start_season]:
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
        full_features = list(CATBOOST_CAT_COLS) + expert_num + current16 + all4
        X = pd.concat([X0, blocks["pitcher"], blocks["batter"],
                       add_catboost_context(raw)], axis=1)

        fam = load_npz(f"{PRED_DIR}/current_season_family_{val_season}.npz")
        cat = load_npz(f"{PRED_DIR}/catboost_expert_{val_season}.npz")
        command = load_npz(f"{PRED_DIR}/catboost_command_profile_{val_season}.npz")
        p_v8 = 0.25 * fam["p_hgb8"] + 0.75 * fam["p_current16"]
        last_v8 = 0.25 * fam["last_hgb8"] + 0.75 * fam["last_current16"]
        r_extrap = float(fam["r_extrap"])
        p_base_cat1, last_base_cat1 = cat["p_expert"], cat["last_expert"]
        yv = cat["y"].astype(float)
        p_all4_cat1 = (command["p_all4_raw"] - 0.4 * p_v8) / 0.6
        last_all4_cat1 = (command["last_all4_raw"] - 0.4 * last_v8) / 0.6

        ref_cat = 0.5 * p_base_cat1 + 0.5 * p_all4_cat1
        ref_last_cat = 0.5 * last_base_cat1 + 0.5 * last_all4_cat1
        ref_raw = 0.4 * p_v8 + 0.6 * ref_cat
        ref_last = 0.4 * last_v8 + 0.6 * ref_last_cat
        ref, _, _, _ = recenter_like_production(ref_raw, ref_last, r_extrap, 0.5)
        base = yv.mean() * (1 - yv.mean())
        saved = dict(y=yv, p_reference=ref)
        log(f"\n[fold={val_season}] active ablations={active}")

        fold_result = {}
        for name in active:
            dropped = ABLATIONS[name]
            features = [c for c in full_features if c not in dropped]
            cat_cols = [c for c in CATBOOST_CAT_COLS if c in features]
            model, pc, lc = fit_predict_catboost(
                X, features, tr, va, lm, y, seed=2026, cat_cols=cat_cols)
            candidate_cat = 0.5 * p_base_cat1 + 0.5 * pc
            candidate_last_cat = 0.5 * last_base_cat1 + 0.5 * lc
            raw_pred = 0.4 * p_v8 + 0.6 * candidate_cat
            raw_last = 0.4 * last_v8 + 0.6 * candidate_last_cat
            pred, shift, _, _ = recenter_like_production(
                raw_pred, raw_last, r_extrap, 0.5)
            gain, se = paired(yv, ref, pred, base)
            fold_result[name] = (gain, se)
            results.append(dict(config=name, season=val_season, gain=gain,
                                se=se, shift=shift, n_features=len(features)))
            saved[f"p_{name}"] = pred
            saved[f"p_{name}_expert"] = pc
            saved[f"last_{name}_expert"] = lc
            log(f"  {name:22s} features={len(features):3d} gain={gain:+8.2f} "
                f"SE={se:.2f}")
            del model
            gc.collect()
        np.savez_compressed(f"{PRED_DIR}/catboost_ablation_{val_season}.npz", **saved)

        # 한 forward 폴드에서 1SE보다 크게 손해면 세 폴드 안정 후보가 될 가능성이
        # 낮아 다음 시즌 계산을 하지 않는다. 약한 음수/양수는 계속 확인한다.
        active = [name for name in active
                  if fold_result[name][0] >= -fold_result[name][1]]
        log(f"  survivors -> {active}")
        if not active:
            break
        del X, X0, blocks
        gc.collect()

    result = pd.DataFrame(results)
    suffix = "" if start_season == VAL_SEASONS[0] else f"_{start_season}"
    result.to_csv(f"{PRED_DIR}/catboost_ablation_summary{suffix}.csv", index=False)
    log("\nSUMMARY sequential ablation vs stage2 all4-50 seed1")
    for name in ABLATIONS:
        d = result[result["config"].eq(name)]
        if len(d):
            log(f"{name:22s} gain=" + "/".join(f"{x:+.2f}" for x in d["gain"])
                + " SE=" + "/".join(f"{x:.2f}" for x in d["se"]))


if __name__ == "__main__":
    main()
