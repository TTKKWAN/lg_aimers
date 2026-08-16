"""최종 후보 확정 실험 — fold 2024 (2025와 가장 유사한 '연속 시즌' 상황).

지금까지 확정된 사실:
  - season 피처 유지, 전체 시즌 학습, 최근시즌 가중 없음  (diag_level.py)
  - a 보정은 시즌 간 전이 안 됨 -> 사용하지 않음          (diag_calibration.py)
  - 시드 앙상블 + shrinkage 피처 + 추세 외삽 재중심화가 이득
남은 질문:
  1) 타깃 인코딩(TE)이 shrinkage 위에 추가 이득을 주는가
  2) shrinkage 강도 k=50 vs 200
  3) 앙상블은 몇 개에서 포화되는가 / 이질 앙상블이 동질보다 나은가
"""
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, TE_FEATURE_COLS,
                       add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                       TargetEncoder, make_model, bss)

DATA_DIR = "./open/data"
VAL_SEASON = 2024
N_MEMBERS = 8


def log(*a):
    print(*a, flush=True)


test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW_FEATURES = [c for c in test_cols if c != ID]
RAW_NUM = [c for c in RAW_FEATURES if c not in CAT_COLS]
raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                   usecols=RAW_FEATURES + [TARGET])
y, seasons = raw[TARGET], raw["season"]
base_df = pd.concat([raw[RAW_FEATURES], add_derived(raw)], axis=1)
season_rate = y.groupby(seasons).mean()

tr_m = (seasons < VAL_SEASON).to_numpy()
va_m = (seasons == VAL_SEASON).to_numpy()
y_tr, y_va = y[tr_m], y[va_m]
yv = y_va.to_numpy().astype(float)
r_true = yv.mean()

tr_seasons = sorted(s for s in season_rate.index if s < VAL_SEASON)
b, a0 = np.polyfit(np.array(tr_seasons[-3:], float),
                    np.array([season_rate[s] for s in tr_seasons[-3:]], float), 1)
r_hat = float(a0 + b * VAL_SEASON)
log(f"fold {VAL_SEASON}: 실제 r={r_true:.4f}  외삽 r={r_hat:.4f}")

prior = fit_prior(raw.loc[tr_m])
SH = {k: add_shrinkage(base_df, prior, k) for k in (50, 200)}
te = TargetEncoder(smooth=200.0)
TE_TR = te.fit_transform_oof(raw.loc[tr_m], y_tr, n_splits=5, seed=0)
TE_VA = te.transform(raw.loc[va_m])

SETS = {
    "shrink_k50":     dict(k=50,  te=False),
    "shrink_k200":    dict(k=200, te=False),
    "shrink_k50_te":  dict(k=50,  te=True),
}

HETERO = [
    dict(seed=42,    learning_rate=0.03,  max_leaf_nodes=63,  min_samples_leaf=30,  max_features=1.0),
    dict(seed=7,     learning_rate=0.05,  max_leaf_nodes=31,  min_samples_leaf=50,  max_features=0.7),
    dict(seed=2024,  learning_rate=0.02,  max_leaf_nodes=95,  min_samples_leaf=20,  max_features=0.8),
    dict(seed=1,     learning_rate=0.04,  max_leaf_nodes=63,  min_samples_leaf=100, max_features=0.6),
    dict(seed=12345, learning_rate=0.03,  max_leaf_nodes=127, min_samples_leaf=40,  max_features=0.9),
    dict(seed=99,    learning_rate=0.06,  max_leaf_nodes=45,  min_samples_leaf=60,  max_features=0.7),
    dict(seed=2718,  learning_rate=0.025, max_leaf_nodes=80,  min_samples_leaf=25,  max_features=0.85),
    dict(seed=31415, learning_rate=0.045, max_leaf_nodes=50,  min_samples_leaf=80,  max_features=0.75),
][:N_MEMBERS]


def recenter(p, target):
    q = np.clip(p, 1e-9, 1 - 1e-9)
    lg = np.log(q / (1 - q))
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(lg + mid)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + np.exp(-(lg + (lo + hi) / 2)))


results = {}
for sname, cfg in SETS.items():
    num_cols = list(RAW_NUM) + list(DERIVED_COLS) + shrinkage_cols()
    Xtr = pd.concat([base_df.loc[tr_m], SH[cfg["k"]].loc[tr_m]], axis=1)
    Xva = pd.concat([base_df.loc[va_m], SH[cfg["k"]].loc[va_m]], axis=1)
    if cfg["te"]:
        num_cols += TE_FEATURE_COLS
        Xtr = pd.concat([Xtr, TE_TR], axis=1)
        Xva = pd.concat([Xva, TE_VA], axis=1)
    cols = CAT_COLS + num_cols
    log(f"\n=== {sname}  (features={len(cols)}) ===")

    ps = []
    for i, spec in enumerate(HETERO):
        spec = dict(spec)
        seed = spec.pop("seed")
        t = time.time()
        m = make_model(num_cols, seed=seed, **spec)
        m.fit(Xtr[cols], y_tr)
        p = m.predict_proba(Xva[cols])[:, 1]
        ps.append(p)
        _, s1, _ = bss(yv, p)
        log(f"  member {i+1}/{N_MEMBERS} single={s1:7.2f}  ({time.time()-t:.0f}s)")
    P = np.vstack(ps)

    curve = []
    for n in range(1, N_MEMBERS + 1):
        _, s, _ = bss(yv, P[:n].mean(axis=0))
        curve.append(s)
    log("  누적 앙상블: " + " ".join(f"{n+1}:{s:.0f}" for n, s in enumerate(curve)))

    p_ens = P.mean(axis=0)
    _, s_raw, base = bss(yv, p_ens)
    _, s_rc, _ = bss(yv, recenter(p_ens, r_hat))
    _, s_or, _ = bss(yv, recenter(p_ens, r_true))
    log(f"  >> raw={s_raw:7.2f}  외삽재중심화={s_rc:7.2f}  oracle재중심화={s_or:7.2f}  "
        f"mean(p)={p_ens.mean():.4f}")
    results[sname] = dict(raw=s_raw, rc=s_rc, oracle=s_or, curve=curve,
                           p=p_ens, base=base)
    np.savez_compressed(f"preds_final_{sname}_{VAL_SEASON}.npz", p=p_ens, y=yv)

log("\n" + "=" * 78)
log("요약 (fold 2024)")
log("=" * 78)
for sname, d in results.items():
    log(f"{sname:15s} raw={d['raw']:7.2f}  외삽재중심화={d['rc']:7.2f}  "
        f"oracle={d['oracle']:7.2f}   앙상블 1개->{N_MEMBERS}개: "
        f"{d['curve'][0]:.0f} -> {d['curve'][-1]:.0f}")

ref = "shrink_k50"
log(f"\npaired 비교 (기준={ref}, 재중심화 후)")
for sname, d in results.items():
    if sname == ref:
        continue
    lr = (recenter(results[ref]["p"], r_hat) - yv) ** 2
    ln = (recenter(d["p"], r_hat) - yv) ** 2
    diff = (lr - ln) / d["base"] * 100000
    log(f"  {sname:15s} {diff.mean():+7.2f}점 "
        f"(SE={diff.std(ddof=1)/np.sqrt(len(diff)):.1f})")
