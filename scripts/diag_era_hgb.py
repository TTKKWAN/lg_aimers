"""era_all_replace_static을 기존 HGB 5-member 저장 예측과 비교한다.

baseline HGB 예측은 experiments/preds/ens_hgb_{season}.npz를 재사용하고,
동일한 5개 HGB 설정으로 시대보정 후보만 학습한다.
"""
import gc
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, RATE_N_PAIRS,
                       add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                       make_model, bss)
from diag_era_features import fit_era_prior, add_era_features, skill_cols, ALL_RATES

DATA_DIR = "./open/data"
PRED_DIR = "./experiments/preds"
VAL_SEASONS = [2022, 2023, 2024]
K = 50

HGB_SPECS = [
    dict(seed=42, learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=30, max_features=1.0),
    dict(seed=7, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, max_features=0.7),
    dict(seed=2024, learning_rate=0.02, max_leaf_nodes=95, min_samples_leaf=20, max_features=0.8),
    dict(seed=1, learning_rate=0.04, max_leaf_nodes=63, min_samples_leaf=100, max_features=0.6),
    dict(seed=12345, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=40, max_features=0.9),
]


def log(*args):
    print(*args, flush=True)


test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
raw_features = [c for c in test_cols if c != ID]
raw_num = [c for c in raw_features if c not in CAT_COLS]
raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                  usecols=raw_features + [TARGET])
y, seasons = raw[TARGET], raw["season"]
base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
static_sh_rate_cols = {f"sh_{r}" for r, _ in RATE_N_PAIRS}
num_cols = [c for c in list(raw_num) + list(DERIVED_COLS) + shrinkage_cols()
            if c not in static_sh_rate_cols]
num_cols += skill_cols(ALL_RATES)
cols = CAT_COLS + num_cols

fold_gains = []
for val_season in VAL_SEASONS:
    tr_m = (seasons < val_season).to_numpy()
    va_m = (seasons == val_season).to_numpy()
    y_tr, y_va = y[tr_m], y[va_m]
    yv = y_va.to_numpy(dtype=float)
    baseline = np.load(f"{PRED_DIR}/ens_hgb_{val_season}.npz")
    assert np.array_equal(baseline["y"], yv) and int(baseline["n"]) == len(HGB_SPECS)

    static_sh = add_shrinkage(base_df, fit_prior(raw.loc[tr_m]), K)
    era = add_era_features(raw, fit_era_prior(raw.loc[tr_m]))
    X = pd.concat([base_df, static_sh, era], axis=1)
    log(f"\n[fold={val_season}] train={tr_m.sum():,} val={va_m.sum():,}")
    ps = []
    for i, spec0 in enumerate(HGB_SPECS):
        spec = dict(spec0)
        seed = spec.pop("seed")
        t = time.time()
        model = make_model(num_cols, seed=seed, **spec)
        model.fit(X.loc[tr_m, cols], y_tr)
        p = model.predict_proba(X.loc[va_m, cols])[:, 1]
        ps.append(p)
        _, score, _ = bss(yv, p)
        log(f"  member={i+1}/{len(HGB_SPECS)} seed={seed} score={score:.2f} time={time.time()-t:.0f}s")
        del model
        gc.collect()
    p_era = np.mean(ps, axis=0)
    p_base = baseline["p"]
    _, s_base, base = bss(yv, p_base)
    _, s_era, _ = bss(yv, p_era)
    d = ((p_base - yv) ** 2 - (p_era - yv) ** 2) / base * 100000
    gain, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))
    fold_gains.append(gain)
    log(f">> baseline_hgb5={s_base:.2f} era_hgb5={s_era:.2f} gain={gain:+.2f} SE={se:.1f}")
    np.savez_compressed(f"{PRED_DIR}/era_hgb_{val_season}.npz", p=p_era, y=yv, n=len(ps))
    del X, static_sh, era, ps
    gc.collect()

log(f"\nfold gains={fold_gains} mean={np.mean(fold_gains):+.2f}")
