"""LightGBM 이질성 추가 진단 — 여러 forward-chaining 폴드로 검증.

현재 v3 앙상블은 8개 멤버가 전부 HistGradientBoosting이라 모델 계열 다양성이
없다. LightGBM 멤버를 섞으면 트리 성장 방식(leaf-wise vs level-wise)이 달라
이질성이 늘고 앙상블 이득이 커질 수 있다는 가설을 검증한다.

recenter_f/trackman 진단에서 배운 교훈 그대로: 단일 폴드가 아니라 여러
forward-chaining 폴드(2022, 2023, 2024)에서 일관되게 이득이 나는지 확인.
"""
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS,
                       add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                       make_model, make_lgbm_model, bss)

DATA_DIR = "./open/data"
VAL_SEASONS = [2022, 2023, 2024]
K = 50

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


def log(*a):
    print(*a, flush=True)


test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW_FEATURES = [c for c in test_cols if c != ID]
RAW_NUM = [c for c in RAW_FEATURES if c not in CAT_COLS]
raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                   usecols=RAW_FEATURES + [TARGET])
y, seasons = raw[TARGET], raw["season"]
base_df = pd.concat([raw[RAW_FEATURES], add_derived(raw)], axis=1)

fold_results = []
for val_season in VAL_SEASONS:
    log(f"\n{'='*78}\nfold val_season={val_season}\n{'='*78}")
    tr_m = (seasons < val_season).to_numpy()
    va_m = (seasons == val_season).to_numpy()
    y_tr, y_va = y[tr_m], y[va_m]
    yv = y_va.to_numpy(dtype=float)

    prior = fit_prior(raw.loc[tr_m])
    sh = add_shrinkage(base_df, prior, K)
    num_cols = list(RAW_NUM) + list(DERIVED_COLS) + shrinkage_cols()
    cols = CAT_COLS + num_cols
    Xtr = pd.concat([base_df.loc[tr_m], sh.loc[tr_m]], axis=1)
    Xva = pd.concat([base_df.loc[va_m], sh.loc[va_m]], axis=1)

    ps_hgb, ps_lgbm = [], []
    for i, spec in enumerate(HGB_SPECS):
        spec = dict(spec)
        seed = spec.pop("seed")
        t = time.time()
        m = make_model(num_cols, seed=seed, **spec)
        m.fit(Xtr[cols], y_tr)
        p = m.predict_proba(Xva[cols])[:, 1]
        ps_hgb.append(p)
        log(f"  [hgb] member {i+1}/{len(HGB_SPECS)} ({time.time()-t:.0f}s)")

    for i, spec in enumerate(LGBM_SPECS):
        spec = dict(spec)
        seed = spec.pop("seed")
        t = time.time()
        m = make_lgbm_model(num_cols, seed=seed, **spec)
        m.fit(Xtr[cols], y_tr)
        p = m.predict_proba(Xva[cols])[:, 1]
        ps_lgbm.append(p)
        _, s1, _ = bss(yv, p)
        log(f"  [lgbm] member {i+1}/{len(LGBM_SPECS)} single={s1:7.2f} ({time.time()-t:.0f}s)")

    p_hgb_only = np.vstack(ps_hgb).mean(axis=0)
    p_mixed = np.vstack(ps_hgb + ps_lgbm).mean(axis=0)

    _, s_hgb, base = bss(yv, p_hgb_only)
    _, s_mix, _ = bss(yv, p_mixed)
    sq_hgb = (p_hgb_only - yv) ** 2
    sq_mix = (p_mixed - yv) ** 2
    diff = (sq_hgb - sq_mix) / base * 100000
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    log(f"\n>> fold {val_season}: hgb{len(HGB_SPECS)}={s_hgb:.2f}  "
        f"hgb{len(HGB_SPECS)}+lgbm{len(LGBM_SPECS)}={s_mix:.2f}  "
        f"diff={diff.mean():+.2f} (SE={se:.1f})")
    fold_results.append(dict(val_season=val_season, hgb=s_hgb, mixed=s_mix,
                              diff=diff.mean(), se=se))

log("\n" + "=" * 78)
log(f"요약 (hgb{len(HGB_SPECS)} vs hgb{len(HGB_SPECS)}+lgbm{len(LGBM_SPECS)}, raw 점수 — 재중심화 없음)")
log("=" * 78)
for r in fold_results:
    log(f"  fold {r['val_season']}: hgb_only={r['hgb']:7.2f}  mixed={r['mixed']:7.2f}  "
        f"diff={r['diff']:+7.2f} (SE={r['se']:.1f})")
diffs = np.array([r["diff"] for r in fold_results])
log(f"\n폴드 평균 diff = {diffs.mean():+.2f} (폴드 std={diffs.std(ddof=1):.2f}, n={len(diffs)})")
