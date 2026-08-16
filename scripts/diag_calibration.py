"""진단: 보정 기울기 a* 측정 — 지금 모델이 얼마나 과신하고 있는가.

diag_level.py 에서 2023 폴드는 oracle 재중심화를 해도 0점이었다.
평균을 r 로 맞춘 뒤의 Brier 는

    Brier - base = E[(p-r)^2] * (1 - 2*a*),   a* = E[(p-r)(y-r)] / E[(p-r)^2]

이므로 a* < 0.5 면 모델이 상수 예측보다 해롭다. a* 는 '예측을 얼마나 믿어도 되는가'
의 척도이고, p' = r + a(p-r) 로 축소하면 최적 a=a* 에서

    Brier - base = -a*^2 * E[(p-r)^2]   (항상 <= 0, 즉 반드시 이득)

이 된다. 여기서는 폴드별 a* 를 재고, a 를 **다른 폴드에서 추정해** 적용했을 때도
이득이 남는지(일반화 여부) 확인한다. 예측값은 npz 로 저장해 뒤 단계에서 재사용.
"""
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, shrinkage_cols,
                       add_derived, add_shrinkage, fit_prior, make_model, bss)

DATA_DIR = "./open/data"
FOLDS = [2023, 2024]
SEEDS = [42, 7, 2024]


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

SETS = {"v2": dict(k=None), "shrink_k50": dict(k=50)}


def extrapolate_rate(train_seasons, target, n_last=3):
    s = sorted(train_seasons)[-n_last:]
    b, a = np.polyfit(np.array(s, float), np.array([season_rate[v] for v in s], float), 1)
    return float(a + b * target)


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


store = {}
for val_season in FOLDS:
    tr_m = (seasons < val_season).to_numpy()
    va_m = (seasons == val_season).to_numpy()
    y_tr, y_va = y[tr_m], y[va_m]
    yv = y_va.to_numpy().astype(float)
    prior = fit_prior(raw.loc[tr_m])
    sh = add_shrinkage(base_df, prior, 50)
    log(f"\n[fold {val_season}] train={tr_m.sum():,} val={va_m.sum():,}")

    for sname, cfg in SETS.items():
        num_cols = list(RAW_NUM) + list(DERIVED_COLS)
        Xtr, Xva = base_df.loc[tr_m], base_df.loc[va_m]
        if cfg["k"] is not None:
            num_cols += shrinkage_cols()
            Xtr = pd.concat([Xtr, sh.loc[tr_m]], axis=1)
            Xva = pd.concat([Xva, sh.loc[va_m]], axis=1)
        cols = CAT_COLS + num_cols
        ps = []
        for seed in SEEDS:
            t = time.time()
            m = make_model(num_cols, seed=seed)
            m.fit(Xtr[cols], y_tr)
            ps.append(m.predict_proba(Xva[cols])[:, 1])
            log(f"   {sname} seed={seed} ({time.time()-t:.0f}s)")
        p = np.mean(ps, axis=0)
        store[(sname, val_season)] = p
        np.savez_compressed(f"preds_{sname}_{val_season}.npz", p=p, y=yv)

log("\n" + "=" * 88)
log("보정 기울기 a* 와 보정 후 점수")
log("=" * 88)

alpha = {}
for sname in SETS:
    for f in FOLDS:
        p = store[(sname, f)]
        yv = y[(seasons == f).to_numpy()].to_numpy().astype(float)
        r_true = yv.mean()
        pc = recenter(p, r_true)          # 먼저 수준을 맞춘 뒤 기울기를 잰다
        d = pc - r_true
        a = float((d * (yv - r_true)).sum() / (d * d).sum())
        alpha[(sname, f)] = a
        spread = float((d * d).mean())
        _, s_raw, base = bss(yv, p)
        _, s_rc, _ = bss(yv, pc)
        _, s_a, _ = bss(yv, np.clip(r_true + a * d, 1e-6, 1 - 1e-6))
        log(f"{sname:11s} fold{f}: a*={a:6.3f}  spread E[(p-r)^2]={spread:.6f}  "
            f"raw={s_raw:7.2f}  재중심화={s_rc:7.2f}  +a*축소={s_a:7.2f}")
        log(f"{'':11s}          이론 최대이득 = a*^2*spread/base*1e5 = "
            f"{100000*a*a*spread/base:7.2f}점")

log("\n--- a 를 '다른 폴드'에서 추정해 적용 (일반화 확인) ---")
for sname in SETS:
    f_fit, f_test = FOLDS[0], FOLDS[1]
    a_prev = alpha[(sname, f_fit)]
    p = store[(sname, f_test)]
    yv = y[(seasons == f_test).to_numpy()].to_numpy().astype(float)
    r_true = yv.mean()
    r_hat = extrapolate_rate([s for s in season_rate.index if s < f_test], f_test)
    for rname, r_use in [("oracle r", r_true), ("외삽 r", r_hat)]:
        pc = recenter(p, r_use)
        for aname, a in [("a=1(보정없음)", 1.0), (f"a={a_prev:.3f}(전폴드)", a_prev),
                          ("a=0.5(보수적)", 0.5)]:
            q = np.clip(r_use + a * (pc - r_use), 1e-6, 1 - 1e-6)
            _, s, _ = bss(yv, q)
            log(f"  {sname:11s} {rname:9s} {aname:16s} -> fold{f_test} score={s:8.2f}")
