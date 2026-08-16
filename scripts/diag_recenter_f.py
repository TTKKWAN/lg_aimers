"""recenter_f 스윕 진단 — fold 2024, shrink_k50 + 이질앙상블8 고정.

목적: v3에서 recenter_f=0.5로 채택한 게 로컬 폴드 기준 최적점인지 확인.
f in {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5} 각각에 대해:
  natural = 앙상블 자연평균, target = natural + f*(r_hat - natural)
  p_ens를 target으로 재중심화한 뒤 Brier Skill Score 계산.
r_hat(외삽 정답률)과 r_true(2024 실제 정답률)를 모두 비교 기준으로 보여줘서,
f를 늘릴수록 oracle(r_true 완전반영)에 가까워지는지 아니면 지나쳐버리는지 확인한다.
"""
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS,
                       add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                       make_model, bss)

DATA_DIR = "./open/data"
VAL_SEASON = 2024
N_MEMBERS = 8
K = 50
F_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


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
log(f"fold {VAL_SEASON}: 실제 r={r_true:.4f}  외삽 r_hat={r_hat:.4f}")

prior = fit_prior(raw.loc[tr_m])
sh = add_shrinkage(base_df, prior, K)
num_cols = list(RAW_NUM) + list(DERIVED_COLS) + shrinkage_cols()
Xtr = pd.concat([base_df.loc[tr_m], sh.loc[tr_m]], axis=1)
Xva = pd.concat([base_df.loc[va_m], sh.loc[va_m]], axis=1)
cols = CAT_COLS + num_cols

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

p_ens = np.vstack(ps).mean(axis=0)
natural = float(p_ens.mean())
log(f"\n앙상블 자연평균={natural:.4f}")


def recenter_to(p, target):
    q = np.clip(p, 1e-9, 1 - 1e-9)
    lg = np.log(q / (1 - q))
    lo, hi = -6.0, 6.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(lg + mid)))).mean() < target:
            lo = mid
        else:
            hi = mid
    shift = (lo + hi) / 2
    return 1 / (1 + np.exp(-(lg + shift))), shift


log("\n" + "=" * 78)
log(f"{'f':>6} {'target':>8} {'score':>9} {'diff_vs_f0.5':>14} {'SE':>7}")
log("=" * 78)

base_scores = {}
_, s_raw, base = bss(yv, p_ens)
sq_f05 = None
for f in F_GRID:
    target = natural + f * (r_hat - natural)
    p_rc, shift = recenter_to(p_ens, target)
    _, s, base = bss(yv, p_rc)
    sq = (p_rc - yv) ** 2
    base_scores[f] = (s, sq, target, shift)
    if abs(f - 0.5) < 1e-9:
        sq_f05 = sq

for f in F_GRID:
    s, sq, target, shift = base_scores[f]
    if sq_f05 is not None and f != 0.5:
        diff = (sq_f05 - sq) / base * 100000
        log(f"{f:6.2f} {target:8.4f} {s:9.2f} {diff.mean():+13.2f} {diff.std(ddof=1)/np.sqrt(len(diff)):7.1f}")
    else:
        log(f"{f:6.2f} {target:8.4f} {s:9.2f} {'(기준)':>13} {'':>7}")

_, s_oracle, _ = bss(yv, recenter_to(p_ens, r_true)[0])
log(f"\nraw(f=0)={s_raw:.2f}  oracle(r_true 완전반영)={s_oracle:.2f}  r_true={r_true:.4f}  r_hat={r_hat:.4f}")
