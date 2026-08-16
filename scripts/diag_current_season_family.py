"""현 시즌 입력 표현과 production family 가중치를 함께 검증한다.

기존 season-to-date 확인은 저장 HGB5 OOF와 raw 확률만 비교했다. 이 스크립트는
production과 같은 HGB8 + LGBM3를 forward 2022/2023/2024에서 다시 학습하고,
각 family 가중치마다 학습 마지막 시즌 예측으로 recenter_f=0.5 logit shift를
고정한 뒤 검증 시즌에 적용한다.

LGBM 입력 후보:
  current16       : v7의 투수/타자 현 시즌 success/workload 16개
  success_predev  : current16의 dev를 현 누적 career가 아니라 시즌 시작 전 career와 비교
  expanded_predev : pitcher outcome, pitchmix, batter middle까지 같은 방식으로 확장

모든 lookup은 fold-train의 과거 시즌 endpoint만 사용한다. 검증 행끼리는 참조하지
않으므로 제출의 row-local 계약과 같다.
"""
import gc
import os
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                       add_shrinkage, shrinkage_cols, fit_prior,
                       make_model, make_lgbm_model, bss)
from diag_season_to_date import (DATA_DIR, PRED_DIR, VAL_SEASONS, K, GROUPS,
                                 _previous_lookup, _season_block, paired)
from diag_season_to_date_followup import exact_cols


HGB_SPECS = [
    dict(seed=42, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=30, max_features=1.0),
    dict(seed=7, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, max_features=0.7),
    dict(seed=2024, learning_rate=0.02, max_leaf_nodes=95, min_samples_leaf=20, max_features=0.8),
    dict(seed=1, learning_rate=0.04, max_leaf_nodes=63, min_samples_leaf=100, max_features=0.6),
    dict(seed=12345, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=40, max_features=0.9),
    dict(seed=99, learning_rate=0.06, max_leaf_nodes=45, min_samples_leaf=60, max_features=0.7),
    dict(seed=2718, learning_rate=0.025, max_leaf_nodes=80, min_samples_leaf=25, max_features=0.85),
    dict(seed=31415, learning_rate=0.045, max_leaf_nodes=50, min_samples_leaf=80, max_features=0.75),
]

LGBM_SPECS = [
    dict(seed=99, learning_rate=0.03, num_leaves=63, min_child_samples=30,
         colsample_bytree=0.8, subsample=0.8),
    dict(seed=2718, learning_rate=0.05, num_leaves=31, min_child_samples=50,
         colsample_bytree=0.7, subsample=0.7),
    dict(seed=31415, learning_rate=0.02, num_leaves=127, min_child_samples=20,
         colsample_bytree=0.9, subsample=0.9),
]

WEIGHTS = [3 / 11, 0.35, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 1.00]


def log(*args):
    print(*args, flush=True)


def preseason_dev(block, previous, spec, prior):
    """현 시즌 EB와 시즌 시작 직전 career EB의 차이.

    lookup이 없는 신규 선수는 이전 표본 0, prior 중심으로 시작한다. success가 아닌
    outcome/pitchmix endpoint는 대회 입력만으로 알 수 없는 마지막 1구를 제외한
    스냅샷이며, diag_season_to_date의 기존 안전 계약과 동일하다.
    """
    out = {}
    for rate in spec["rates"]:
        pn = previous[f"prev_n_{rate}"].to_numpy(dtype=float)
        pc = previous[f"prev_num_{rate}"].to_numpy(dtype=float)
        known = np.isfinite(pn) & np.isfinite(pc)
        pn = np.where(known, pn, 0.0)
        pc = np.where(known, pc, 0.0)
        pre_sh = (pc + K * prior[rate]) / (pn + K)
        season_sh = block[f"std_sh_{rate}"].to_numpy(dtype=float)
        out[f"std_pre_dev_{rate}"] = season_sh - pre_sh
    return pd.DataFrame(out, index=block.index, dtype="float32")


def replace_dev(cols, rates):
    old = {f"std_dev_{rate}" for rate in rates}
    return [c for c in cols if c not in old] + [f"std_pre_dev_{rate}" for rate in rates]


def extrapolated_rate(y, seasons, train_mask, val_season):
    sr = y.loc[train_mask].groupby(seasons.loc[train_mask]).mean()
    recent = sorted(sr.index)[-min(3, len(sr)):]
    if len(recent) >= 2:
        slope, intercept = np.polyfit(np.asarray(recent, float),
                                      np.asarray([sr[s] for s in recent], float), 1)
        return float(intercept + slope * val_season)
    return float(sr.iloc[-1])


def recenter_like_production(p_val, p_last, r_extrap, fraction=0.5):
    """build_final.py와 같은 in-sample last-season 기반 고정 logit shift."""
    natural = float(np.mean(p_last))
    target = natural + fraction * (r_extrap - natural)
    q_last = np.clip(p_last, 1e-9, 1 - 1e-9)
    lg_last = np.log(q_last / (1 - q_last))
    lo, hi = -6.0, 6.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(lg_last + mid)))) < target:
            lo = mid
        else:
            hi = mid
    shift = float((lo + hi) / 2)
    q = np.clip(p_val, 1e-9, 1 - 1e-9)
    pred = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + shift)))
    return pred, shift, natural, target


def mean_predictions(builder, specs, num_cols, X, train_mask, val_mask, last_mask, y):
    cols = CAT_COLS + num_cols
    pv, pl = [], []
    for i, spec0 in enumerate(specs):
        spec = dict(spec0)
        seed = spec.pop("seed")
        t = time.time()
        model = builder(num_cols, seed=seed, **spec)
        model.fit(X.loc[train_mask, cols], y.loc[train_mask])
        pv.append(model.predict_proba(X.loc[val_mask, cols])[:, 1])
        pl.append(model.predict_proba(X.loc[last_mask, cols])[:, 1])
        log(f"    member {i+1}/{len(specs)} seed={seed} {time.time()-t:.0f}s")
        del model
        gc.collect()
    return np.mean(pv, axis=0), np.mean(pl, axis=0)


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    base_num = raw_num + list(DERIVED_COLS) + shrinkage_cols()
    os.makedirs(PRED_DIR, exist_ok=True)
    fold_data = {}

    for val_season in VAL_SEASONS:
        train_mask = (seasons < val_season).to_numpy()
        val_mask = (seasons == val_season).to_numpy()
        last_train_season = int(seasons.loc[train_mask].max())
        last_mask = train_mask & seasons.eq(last_train_season).to_numpy()
        yv = y.loc[val_mask].to_numpy(dtype=float)
        prior = fit_prior(raw.loc[train_mask])
        sh = add_shrinkage(base_df, prior, K)
        X0 = pd.concat([base_df, sh], axis=1)

        blocks, predevs = {}, {}
        for name, spec in GROUPS.items():
            previous = _previous_lookup(raw, train_mask, spec)
            block, _ = _season_block(X0, previous, spec, prior, zero_baseline=True)
            blocks[name] = block
            predevs[name] = preseason_dev(block, previous, spec, prior)
            del previous
        X = pd.concat([X0] + list(blocks.values()) + list(predevs.values()), axis=1)

        pitcher_success = exact_cols(blocks["pitcher"], "pitcher",
                                     "asof_pitcher_success_rate")
        batter_success = exact_cols(blocks["batter"], "batter",
                                    "asof_batter_success_rate")
        current16 = pitcher_success + batter_success
        success_rates = ["asof_pitcher_success_rate", "asof_batter_success_rate"]
        success_predev = replace_dev(current16, success_rates)
        expanded = (list(blocks["pitcher"].columns)
                    + list(blocks["pitchmix"].columns)
                    + list(blocks["batter"].columns))
        all_rates = sum((spec["rates"] for spec in GROUPS.values()), [])
        expanded_predev = replace_dev(expanded, all_rates)
        configs = {
            "current16": current16,
            "success_predev": success_predev,
            "expanded_predev": expanded_predev,
        }

        log(f"\n[fold={val_season}] train={train_mask.sum():,} val={val_mask.sum():,} "
            f"last_train={last_train_season} rows={last_mask.sum():,}")
        log("  HGB8")
        p_hgb, last_hgb = mean_predictions(make_model, HGB_SPECS, base_num, X,
                                             train_mask, val_mask, last_mask, y)
        predictions = {}
        for name, new_cols in configs.items():
            log(f"  LGBM3 {name} features={len(new_cols)}")
            predictions[name] = mean_predictions(
                make_lgbm_model, LGBM_SPECS, base_num + new_cols, X,
                train_mask, val_mask, last_mask, y)

        r_extrap = extrapolated_rate(y, seasons, train_mask, val_season)
        fold_data[val_season] = dict(y=yv, p_hgb=p_hgb, last_hgb=last_hgb,
                                     r_extrap=r_extrap, predictions=predictions)
        saved = dict(y=yv, p_hgb8=p_hgb, last_hgb8=last_hgb,
                     r_extrap=np.asarray(r_extrap), last_season=last_train_season)
        for name, (pv, pl) in predictions.items():
            saved[f"p_{name}"] = pv
            saved[f"last_{name}"] = pl
        np.savez_compressed(f"{PRED_DIR}/current_season_family_{val_season}.npz", **saved)
        del X, X0, sh, blocks, predevs, predictions, p_hgb, last_hgb
        gc.collect()

    baseline_weight = 3 / 11
    baseline = {}
    for season, fd in fold_data.items():
        pv, pl = fd["predictions"]["current16"]
        p = (1 - baseline_weight) * fd["p_hgb"] + baseline_weight * pv
        last = (1 - baseline_weight) * fd["last_hgb"] + baseline_weight * pl
        pr, shift, natural, target = recenter_like_production(
            p, last, fd["r_extrap"], 0.5)
        baseline[season] = (p, pr)
        log(f"baseline fold={season} natural={natural:.6f} target={target:.6f} "
            f"r_extrap={fd['r_extrap']:.6f} shift={shift:+.6f}")

    log("\n" + "=" * 120)
    log("SUMMARY — baseline=current16 HGB8:LGBM3=8:3; raw와 production-style recenter_f=0.5")
    for name in ["current16", "success_predev", "expanded_predev"]:
        log(f"\n[{name}]")
        for w in WEIGHTS:
            raw_gain, raw_se, rc_gain, rc_se = [], [], [], []
            shifts = []
            for season, fd in fold_data.items():
                yv = fd["y"]
                pv, pl = fd["predictions"][name]
                p = (1 - w) * fd["p_hgb"] + w * pv
                last = (1 - w) * fd["last_hgb"] + w * pl
                pr, shift, _, _ = recenter_like_production(p, last, fd["r_extrap"], 0.5)
                base = yv.mean() * (1 - yv.mean())
                g, se = paired(yv, baseline[season][0], p, base)
                gr, ser = paired(yv, baseline[season][1], pr, base)
                raw_gain.append(g); raw_se.append(se)
                rc_gain.append(gr); rc_se.append(ser); shifts.append(shift)
            log(f"  LGBM w={w:7.4f} raw=" + "/".join(f"{x:+7.2f}" for x in raw_gain)
                + f" mean={np.mean(raw_gain):+7.2f}"
                + " | rc=" + "/".join(f"{x:+7.2f}" for x in rc_gain)
                + f" mean={np.mean(rc_gain):+7.2f}"
                + " rcSE=" + "/".join(f"{x:.2f}" for x in rc_se)
                + " shift=" + "/".join(f"{x:+.4f}" for x in shifts))


if __name__ == "__main__":
    main()
