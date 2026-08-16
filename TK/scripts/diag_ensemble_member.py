"""HGB 또는 LightGBM 멤버 앙상블 학습 후 예측 평균만 저장 — 별도 프로세스로 실행.

diag_lgbm_ensemble.py를 한 프로세스에서 HGB+LightGBM을 섞어 돌렸더니 fold가
진행될수록 멤버 하나 학습에 40초 -> 수천 초까지 느려지는 현상이 있었음
(OpenMP 스레드풀 충돌 또는 누적 메모리 압박으로 추정). 원인 격리를 위해
모델 계열별로 완전히 별도 프로세스에서 실행하고, 예측 평균만 .npz로 저장한 뒤
가벼운 별도 스크립트(diag_lgbm_compare.py)에서 비교한다.

사용법: python3 scripts/diag_ensemble_member.py hgb|lgbm
"""
import gc
import sys
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS,
                       add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                       make_model, make_lgbm_model, bss)

DATA_DIR = "./open/data"
OUT_DIR = "./experiments/preds"
VAL_SEASONS = [2022, 2023, 2024]
K = 50

MODEL_TYPE = sys.argv[1]
assert MODEL_TYPE in ("hgb", "lgbm")

HGB_SPECS = [
    dict(seed=42,    learning_rate=0.03,  max_leaf_nodes=63,  min_samples_leaf=30,  max_features=1.0),
    dict(seed=7,     learning_rate=0.05,  max_leaf_nodes=31,  min_samples_leaf=50,  max_features=0.7),
    dict(seed=2024,  learning_rate=0.02,  max_leaf_nodes=95,  min_samples_leaf=20,  max_features=0.8),
    dict(seed=1,     learning_rate=0.04,  max_leaf_nodes=63,  min_samples_leaf=100, max_features=0.6),
    dict(seed=12345, learning_rate=0.03,  max_leaf_nodes=127, min_samples_leaf=40,  max_features=0.9),
]

LGBM_SPECS = [
    dict(seed=99,    learning_rate=0.03, num_leaves=63,  min_child_samples=30,  colsample_bytree=0.8, subsample=0.8),
    dict(seed=2718,  learning_rate=0.05, num_leaves=31,  min_child_samples=50,  colsample_bytree=0.7, subsample=0.7),
    dict(seed=31415, learning_rate=0.02, num_leaves=127, min_child_samples=20,  colsample_bytree=0.9, subsample=0.9),
]

SPECS = HGB_SPECS if MODEL_TYPE == "hgb" else LGBM_SPECS
BUILDER = make_model if MODEL_TYPE == "hgb" else make_lgbm_model


def log(*a):
    print(*a, flush=True)


test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW_FEATURES = [c for c in test_cols if c != ID]
RAW_NUM = [c for c in RAW_FEATURES if c not in CAT_COLS]
raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                   usecols=RAW_FEATURES + [TARGET])
y, seasons = raw[TARGET], raw["season"]
base_df = pd.concat([raw[RAW_FEATURES], add_derived(raw)], axis=1)

for val_season in VAL_SEASONS:
    log(f"\n{'='*70}\n[{MODEL_TYPE}] fold val_season={val_season}\n{'='*70}")
    tr_m = (seasons < val_season).to_numpy()
    va_m = (seasons == val_season).to_numpy()
    y_tr, y_va = y[tr_m], y[va_m]
    yv = y_va.to_numpy(dtype=float)

    prior = fit_prior(raw.loc[tr_m])
    sh = add_shrinkage(base_df, prior, K)
    num_cols = list(RAW_NUM) + list(DERIVED_COLS) + shrinkage_cols()
    cols = CAT_COLS + num_cols
    Xtr = pd.concat([base_df.loc[tr_m], sh.loc[tr_m]], axis=1)[cols]
    Xva = pd.concat([base_df.loc[va_m], sh.loc[va_m]], axis=1)[cols]
    del sh
    gc.collect()

    ps = []
    for i, spec in enumerate(SPECS):
        spec = dict(spec)
        seed = spec.pop("seed")
        t = time.time()
        m = BUILDER(num_cols, seed=seed, **spec)
        m.fit(Xtr, y_tr)
        p = m.predict_proba(Xva)[:, 1]
        ps.append(p)
        _, s1, _ = bss(yv, p)
        log(f"  member {i+1}/{len(SPECS)} single={s1:7.2f} ({time.time()-t:.0f}s)")
        del m
        gc.collect()

    p_mean = np.vstack(ps).mean(axis=0)
    _, s, _ = bss(yv, p_mean)
    log(f">> [{MODEL_TYPE}] fold {val_season} ensemble({len(SPECS)})={s:.2f}")
    np.savez_compressed(f"{OUT_DIR}/ens_{MODEL_TYPE}_{val_season}.npz",
                         p=p_mean, y=yv, n=len(SPECS))

    del Xtr, Xva, ps, p_mean
    gc.collect()

log(f"\n[{MODEL_TYPE}] 전체 폴드 완료")
