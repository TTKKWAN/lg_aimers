"""작은 context expert를 3개 이질 LGBM으로 확인하고 v4 계열 OOF와 혼합한다.

기준선은 저장된 fixed-EB ``ens_hgb_*``(5멤버)와 ``ens_lgbm_*``(3멤버)를
멤버 수 5:3으로 평균한 예측이다. context expert는 선수/팀 ID와 타자 이력을 빼고
현재 상황, 투수 EB 능력, 최소 능력x압박 상호작용만 사용한다. 생산 코드는 건드리지
않는다.
"""
import gc
import os
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, add_derived, add_shrinkage,
                       fit_prior, make_lgbm_model, bss, add_pressure_ability,
                       PRESSURE_ABILITY_COLS, CONTEXT_NUM_COLS)

DATA_DIR = "./open/data"
PRED_DIR = "./experiments/preds"
VAL_SEASONS = [2022, 2023, 2024]
K = 50

CONTEXT_SPECS = [
    dict(seed=8049, learning_rate=0.03, num_leaves=31,
         min_child_samples=100, colsample_bytree=0.85, subsample=0.8,
         reg_lambda=2.0),
    dict(seed=2718, learning_rate=0.05, num_leaves=15,
         min_child_samples=200, colsample_bytree=0.70, subsample=0.7,
         reg_lambda=3.0),
    dict(seed=31415, learning_rate=0.02, num_leaves=63,
         min_child_samples=75, colsample_bytree=0.95, subsample=0.9,
         reg_lambda=1.0),
]


def log(*args):
    print(*args, flush=True)


def paired(y, p0, p1, base):
    d = ((p0 - y) ** 2 - (p1 - y) ** 2) / base * 100000
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def convex_weight(y, p_base, p_context):
    """Brier를 최소화하는 p=(1-w)*base+w*context의 닫힌형 OOF 해."""
    d = p_context - p_base
    denom = float(np.dot(d, d))
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(y - p_base, d) / denom, 0.0, 1.0))


def load_baseline(season, y_expected):
    zh = np.load(f"{PRED_DIR}/ens_hgb_{season}.npz")
    zl = np.load(f"{PRED_DIR}/ens_lgbm_{season}.npz")
    yh, yl = zh["y"], zl["y"]
    if not (np.array_equal(yh, yl) and np.array_equal(yh, y_expected)):
        raise ValueError(f"saved baseline row/target mismatch for fold {season}")
    if int(zh["n"]) != 5 or int(zl["n"]) != 3:
        raise ValueError(f"unexpected member count: hgb={zh['n']} lgbm={zl['n']}")
    return (5.0 * zh["p"] + 3.0 * zl["p"]) / 8.0


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y = raw[TARGET]
    seasons = raw["season"]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    context_num = list(CONTEXT_NUM_COLS)
    os.makedirs(PRED_DIR, exist_ok=True)

    fold_data = {}
    log("confirm: fixed-EB HGB5+LGBM3 baseline vs 3-member context expert")
    log(f"context_features={len(CAT_COLS) + len(context_num)} members={len(CONTEXT_SPECS)}")
    for val_season in VAL_SEASONS:
        tr_m = (seasons < val_season).to_numpy()
        va_m = (seasons == val_season).to_numpy()
        y_tr = y[tr_m]
        yv = y[va_m].to_numpy(dtype=float)
        p_base = load_baseline(val_season, yv)
        prior = fit_prior(raw.loc[tr_m])
        sh = add_shrinkage(base_df, prior, K)
        X0 = pd.concat([base_df, sh], axis=1)
        interactions = add_pressure_ability(X0, prior)
        X = pd.concat([X0, interactions], axis=1)

        log(f"\n[fold={val_season}] train={tr_m.sum():,} val={va_m.sum():,}")
        members = []
        for i, spec0 in enumerate(CONTEXT_SPECS):
            spec = dict(spec0)
            seed = spec.pop("seed")
            model = make_lgbm_model(context_num, seed=seed, **spec)
            t = time.time()
            model.fit(X.loc[tr_m, CAT_COLS + context_num], y_tr)
            p = model.predict_proba(X.loc[va_m, CAT_COLS + context_num])[:, 1]
            members.append(p)
            _, score, _ = bss(yv, p)
            log(f"  member={i+1}/3 seed={seed} leaves={spec['num_leaves']} "
                f"BSS={score:.2f} time={time.time()-t:.0f}s")
            del model
            gc.collect()
        p_context = np.mean(members, axis=0)
        oracle_w = convex_weight(yv, p_base, p_context)
        fold_data[val_season] = dict(y=yv, p_base=p_base,
                                     p_context=p_context, oracle_w=oracle_w)
        np.savez_compressed(
            f"{PRED_DIR}/pressure_context_confirm_{val_season}.npz",
            y=yv, p_baseline=p_base, p_context=p_context,
            p_context_members=np.asarray(members), oracle_weight=oracle_w,
            context_num_cols=np.asarray(context_num))
        log(f"  fold oracle context weight={oracle_w:.6f}")
        del X, X0, sh, interactions, members
        gc.collect()

    y_pool = np.concatenate([fold_data[s]["y"] for s in VAL_SEASONS])
    b_pool = np.concatenate([fold_data[s]["p_base"] for s in VAL_SEASONS])
    c_pool = np.concatenate([fold_data[s]["p_context"] for s in VAL_SEASONS])
    pooled_w = convex_weight(y_pool, b_pool, c_pool)
    log(f"\npooled OOF convex context weight={pooled_w:.6f}")

    weights = [("baseline", 0.0), ("base90_context10", 0.10),
               ("base80_context20", 0.20), ("base70_context30", 0.30),
               ("pooled_convex", pooled_w), ("context_only", 1.0)]
    log("\n" + "=" * 112)
    log("CONFIRM SUMMARY — HGB5+LGBM3 fixed-EB baseline 대비 paired BSS gain")
    log("=" * 112)
    for name, weight in weights:
        rows = []
        for season in VAL_SEASONS:
            fd = fold_data[season]
            pred = (1.0 - weight) * fd["p_base"] + weight * fd["p_context"]
            br, score, base = bss(fd["y"], pred)
            gain, se = paired(fd["y"], fd["p_base"], pred, base)
            rows.append((br, score, gain, se))
        log(f"{name:20s} w={weight:.6f} "
            + "Brier=" + "/".join(f"{r[0]:.8f}" for r in rows)
            + " BSS=" + "/".join(f"{r[1]:.2f}" for r in rows)
            + " gain=" + "/".join(f"{r[2]:+.2f}" for r in rows)
            + f" mean_gain={np.mean([r[2] for r in rows]):+.2f}"
            + " SE=" + "/".join(f"{r[3]:.2f}" for r in rows))
    log("fold oracle weights=" + "/".join(
        f"{s}:{fold_data[s]['oracle_w']:.6f}" for s in VAL_SEASONS))


if __name__ == "__main__":
    main()
