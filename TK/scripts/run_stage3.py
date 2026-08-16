"""Stage 3 — 앙상블 구조와 사후 보정.

diag_seed_variance.py 에서 시드 5개 앙상블이 단일 모델 평균보다 +105점이었다.
여기서는 그 이득을 더 밀어붙인다:
  A) 앙상블 크기 곡선 — 멤버를 늘릴수록 언제 포화되는가
  B) 동질(같은 config, 시드만 다름) vs 이질(하이퍼파라미터+max_features 다양화) 앙상블
  C) 사후 축소 보정 p' = r + a(p - r)
     최적 a는 (y-r)을 (p-r)에 회귀한 기울기. a<1 이면 모델이 과신하고 있다는 뜻.
     a는 반드시 **다른 폴드**에서 적합해 다음 폴드에 적용해 일반화 여부를 확인한다.

BEST_SET / BEST_DECAY 는 run_experiments.py(STAGE 1~2) 결과로 채운다.
"""
import sys
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, TE_FEATURE_COLS,
                       add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                       TargetEncoder, make_model, bss, season_weights)

DATA_DIR = "./open/data"
FOLDS = [2023, 2024]
N_MEMBERS = 8

# ---- STAGE 1~2 결과로 설정 (기본값은 v2 유지) ----
BEST_SET = dict(k=50, te=False)
BEST_DECAY = 1.0
if len(sys.argv) > 1:      # 사용법: python3 run_stage3.py <k|none> <te0/1> <decay>
    BEST_SET = dict(k=(None if sys.argv[1] == "none" else int(sys.argv[1])),
                     te=bool(int(sys.argv[2])))
    BEST_DECAY = float(sys.argv[3])


def log(*a):
    print(*a, flush=True)


log(f"설정: k={BEST_SET['k']} te={BEST_SET['te']} decay={BEST_DECAY} members={N_MEMBERS}")

test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW_FEATURES = [c for c in test_cols if c != ID]
RAW_NUM = [c for c in RAW_FEATURES if c not in CAT_COLS]
raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                   usecols=RAW_FEATURES + [TARGET])
y, seasons = raw[TARGET], raw["season"]
base_df = pd.concat([raw[RAW_FEATURES], add_derived(raw)], axis=1)

NUM_COLS = list(RAW_NUM) + list(DERIVED_COLS)
if BEST_SET["k"] is not None:
    NUM_COLS += shrinkage_cols()
if BEST_SET["te"]:
    NUM_COLS += TE_FEATURE_COLS
COLS = CAT_COLS + NUM_COLS

# 동질 앙상블: 시드만 변경 / 이질 앙상블: 하이퍼파라미터도 함께 변경
HOMO = [dict(seed=s) for s in [42, 7, 2024, 1, 12345, 99, 2718, 31415][:N_MEMBERS]]
HETERO = [
    dict(seed=42,    learning_rate=0.03, max_leaf_nodes=63,  min_samples_leaf=30,  max_features=1.0),
    dict(seed=7,     learning_rate=0.05, max_leaf_nodes=31,  min_samples_leaf=50,  max_features=0.7),
    dict(seed=2024,  learning_rate=0.02, max_leaf_nodes=95,  min_samples_leaf=20,  max_features=0.8),
    dict(seed=1,     learning_rate=0.04, max_leaf_nodes=63,  min_samples_leaf=100, max_features=0.6),
    dict(seed=12345, learning_rate=0.03, max_leaf_nodes=127, min_samples_leaf=40,  max_features=0.9),
    dict(seed=99,    learning_rate=0.06, max_leaf_nodes=45,  min_samples_leaf=60,  max_features=0.7),
    dict(seed=2718,  learning_rate=0.025,max_leaf_nodes=80,  min_samples_leaf=25,  max_features=0.85),
    dict(seed=31415, learning_rate=0.045,max_leaf_nodes=50,  min_samples_leaf=80,  max_features=0.75),
][:N_MEMBERS]


def build_fold(val_season):
    tr_m = (seasons < val_season).to_numpy()
    va_m = (seasons == val_season).to_numpy()
    prior = fit_prior(raw.loc[tr_m])
    y_tr, y_va = y[tr_m], y[va_m]
    ptr, pva = [base_df.loc[tr_m]], [base_df.loc[va_m]]
    if BEST_SET["k"] is not None:
        sh = add_shrinkage(base_df, prior, BEST_SET["k"])
        ptr.append(sh.loc[tr_m]); pva.append(sh.loc[va_m])
    if BEST_SET["te"]:
        te = TargetEncoder(smooth=200.0)
        ptr.append(te.fit_transform_oof(raw.loc[tr_m], y_tr, n_splits=5, seed=0))
        pva.append(te.transform(raw.loc[va_m]))
    w = None if BEST_DECAY == 1.0 else season_weights(seasons[tr_m], BEST_DECAY)
    return (pd.concat(ptr, axis=1), y_tr, pd.concat(pva, axis=1), y_va, w)


def train_members(specs, X_tr, y_tr, X_va, w, label):
    ps = []
    for i, spec in enumerate(specs):
        spec = dict(spec)
        seed = spec.pop("seed")
        t = time.time()
        m = make_model(NUM_COLS, seed=seed, **spec)
        if w is None:
            m.fit(X_tr[COLS], y_tr)
        else:
            m.fit(X_tr[COLS], y_tr, clf__sample_weight=w)
        ps.append(m.predict_proba(X_va[COLS])[:, 1])
        log(f"      [{label}] member {i+1}/{len(specs)} ({time.time()-t:.0f}s)")
    return np.vstack(ps)


def curve(P, y_va, label):
    """멤버를 1개씩 누적하며 앙상블 점수 변화."""
    scores = []
    for n in range(1, len(P) + 1):
        _, s, _ = bss(y_va, P[:n].mean(axis=0))
        scores.append(s)
    log(f"    {label} 누적곡선: " + " ".join(f"{n+1}:{s:.0f}" for n, s in enumerate(scores)))
    return scores


store = {}
for val_season in FOLDS:
    log(f"\n[fold val={val_season}]")
    X_tr, y_tr, X_va, y_va, w = build_fold(val_season)
    log(f"  train={len(X_tr):,} val={len(X_va):,} features={len(COLS)}")
    P_homo = train_members(HOMO, X_tr, y_tr, X_va, w, "homo")
    P_het = train_members(HETERO, X_tr, y_tr, X_va, w, "hetero")
    c_homo = curve(P_homo, y_va, "동질 ")
    c_het = curve(P_het, y_va, "이질 ")
    P_all = np.vstack([P_homo, P_het])
    _, s_all, base = bss(y_va, P_all.mean(axis=0))
    log(f"    동질+이질 전체({len(P_all)}개) = {s_all:.2f}")
    store[val_season] = dict(y=y_va.to_numpy(), base=base,
                              p_homo=P_homo.mean(axis=0),
                              p_het=P_het.mean(axis=0),
                              p_all=P_all.mean(axis=0),
                              c_homo=c_homo, c_het=c_het, s_all=s_all)

log("\n" + "=" * 78)
log("STAGE 3 요약")
log("=" * 78)
for f in FOLDS:
    d = store[f]
    log(f"fold {f}: 동질{N_MEMBERS}={d['c_homo'][-1]:7.2f}  이질{N_MEMBERS}={d['c_het'][-1]:7.2f}  "
        f"전체{2*N_MEMBERS}={d['s_all']:7.2f}")

# ---- C) 사후 축소 보정: 앞 폴드에서 a를 적합해 뒤 폴드에 적용 ----
log("\n--- 사후 보정 p' = r + a(p - r) ---")


def fit_alpha(p, y, r):
    d = p - r
    return float(((d * (y - r)).sum()) / ((d * d).sum()))


f_fit, f_test = FOLDS[0], FOLDS[-1]
for key in ["p_homo", "p_het", "p_all"]:
    dfit, dtest = store[f_fit], store[f_test]
    r_fit = dfit["y"].mean()
    a = fit_alpha(dfit[key], dfit["y"], r_fit)
    # 적용 시 r 은 학습 데이터에서 얻은 값을 쓴다 (평가 r 은 비공개)
    r_apply = float(y[(seasons < f_test).to_numpy()].mean())
    p_cal = r_apply + a * (dtest[key] - r_apply)
    p_cal = np.clip(p_cal, 1e-6, 1 - 1e-6)
    _, s_raw, _ = bss(dtest["y"], dtest[key])
    _, s_cal, _ = bss(dtest["y"], p_cal)
    log(f"  {key:7s} a(fold{f_fit})={a:.4f}  fold{f_test}: 원본={s_raw:7.2f} -> 보정={s_cal:7.2f} "
        f"({s_cal - s_raw:+.2f})")
