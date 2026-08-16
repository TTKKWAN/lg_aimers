"""Stage 1~2: 평가 프로토콜을 고친 뒤 피처/가중치 개선을 검증.

diag_seed_variance.py 결과로 드러난 사실:
  - 같은 config도 시드만 바꾸면 score가 417~561 (std=63, range=144) 로 흔들린다.
  - 두 모델 점수차가 ~37점 미만이면 통계적으로 구분 불가.
  - 시드 5개 평균 앙상블은 단일 모델 평균보다 +105점.
=> 단일 모델끼리 비교하면 노이즈에 속는다. 따라서 이 스크립트의 모든 비교는
   **시드 앙상블 vs 시드 앙상블** 로 하고, 같은 (폴드) 위에서 paired 차이와
   그 표준오차를 함께 보고한다.

평가 프로토콜: forward-chaining 시즌 폴드 (과거로 학습 -> 미래 시즌 검증).
실제 평가가 2025(미래)이므로 랜덤 분할은 쓰지 않는다.
"""
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, TE_FEATURE_COLS,
                       add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                       TargetEncoder, make_model, bss, season_weights)

DATA_DIR = "./open/data"
FOLDS = [2023, 2024]          # 2025 예측이 목표이므로 최근 폴드 위주
SEEDS = [42, 7, 2024]         # 폴드당 3시드 앙상블


def log(*a):
    print(*a, flush=True)


log("데이터 로드...")
test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW_FEATURES = [c for c in test_cols if c != ID]
RAW_NUM = [c for c in RAW_FEATURES if c not in CAT_COLS]

raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                   usecols=RAW_FEATURES + [TARGET])
y = raw[TARGET]
seasons = raw["season"]
base_df = pd.concat([raw[RAW_FEATURES], add_derived(raw)], axis=1)
log(f"  raw={raw.shape}  base={base_df.shape}")

FEATURE_SETS = {
    "v2_current":    dict(k=None, te=False),
    "shrink_k50":    dict(k=50,  te=False),
    "shrink_k200":   dict(k=200, te=False),
    "shrink_k50_te": dict(k=50,  te=True),
}
REF = "v2_current"


def num_cols_for(cfg):
    cols = list(RAW_NUM) + list(DERIVED_COLS)
    if cfg["k"] is not None:
        cols += shrinkage_cols()
    if cfg["te"]:
        cols += TE_FEATURE_COLS
    return cols


def build_fold(val_season, cfgs_needed):
    """폴드별 학습/검증 행렬. prior와 TE 맵은 그 폴드의 train에서만 산출."""
    tr_m = (seasons < val_season).to_numpy()
    va_m = (seasons == val_season).to_numpy()
    prior = fit_prior(raw.loc[tr_m])
    y_tr, y_va = y[tr_m], y[va_m]

    shr = {k: add_shrinkage(base_df, prior, k)
           for k in {c["k"] for c in cfgs_needed if c["k"] is not None}}
    te_tr = te_va = None
    if any(c["te"] for c in cfgs_needed):
        te = TargetEncoder(smooth=200.0)
        te_tr = te.fit_transform_oof(raw.loc[tr_m], y_tr, n_splits=5, seed=0)
        te_va = te.transform(raw.loc[va_m])
    return tr_m, va_m, y_tr, y_va, shr, te_tr, te_va


def assemble(cfg, tr_m, va_m, shr, te_tr, te_va):
    ptr, pva = [base_df.loc[tr_m]], [base_df.loc[va_m]]
    if cfg["k"] is not None:
        ptr.append(shr[cfg["k"]].loc[tr_m]); pva.append(shr[cfg["k"]].loc[va_m])
    if cfg["te"]:
        ptr.append(te_tr); pva.append(te_va)
    return pd.concat(ptr, axis=1), pd.concat(pva, axis=1)


def run_ensemble(X_tr, y_tr, X_va, y_va, num_cols, weights=None, tag=""):
    """시드 앙상블 학습 -> 평균 확률 반환."""
    cols = CAT_COLS + num_cols
    ps, singles = [], []
    for seed in SEEDS:
        t = time.time()
        m = make_model(num_cols, seed=seed)
        if weights is None:
            m.fit(X_tr[cols], y_tr)
        else:
            m.fit(X_tr[cols], y_tr, clf__sample_weight=weights)
        p = m.predict_proba(X_va[cols])[:, 1]
        ps.append(p)
        _, s, _ = bss(y_va, p)
        singles.append(s)
        log(f"      seed={seed:<5d} single={s:8.2f}  ({time.time()-t:.0f}s)")
    p_ens = np.mean(ps, axis=0)
    _, s_ens, base = bss(y_va, p_ens)
    log(f"    {tag} 단일평균={np.mean(singles):7.2f}  앙상블={s_ens:7.2f}")
    return p_ens, s_ens, base, float(np.mean(singles))


# ============================================================== STAGE 1
log("\n" + "=" * 78)
log("STAGE 1 — 피처 세트 비교 (시즌 폴드 x 시드앙상블)")
log("=" * 78)

ens_loss = {}   # (set, fold) -> per-sample squared loss of ensemble
ens_score = {}
single_mean = {}

for val_season in FOLDS:
    log(f"\n[fold val={val_season}]")
    tr_m, va_m, y_tr, y_va, shr, te_tr, te_va = build_fold(
        val_season, list(FEATURE_SETS.values()))
    log(f"  train={tr_m.sum():,}  val={va_m.sum():,}")
    yv = y_va.to_numpy()

    for name, cfg in FEATURE_SETS.items():
        log(f"  -- {name}")
        X_tr, X_va = assemble(cfg, tr_m, va_m, shr, te_tr, te_va)
        p, s, base, sm = run_ensemble(X_tr, y_tr, X_va, y_va,
                                       num_cols_for(cfg), tag=name)
        ens_loss[(name, val_season)] = ((p - yv) ** 2, base)
        ens_score[(name, val_season)] = s
        single_mean[(name, val_season)] = sm

log("\n" + "-" * 78)
log("STAGE 1 요약 (앙상블 기준, baseline 대비 paired 차이)")
log("-" * 78)
means = {}
for name in FEATURE_SETS:
    sc = [ens_score[(name, f)] for f in FOLDS]
    sm = [single_mean[(name, f)] for f in FOLDS]
    means[name] = float(np.mean(sc))
    line = (f"{name:16s} 앙상블평균={np.mean(sc):8.2f}  "
            f"(단일평균={np.mean(sm):7.2f}, 앙상블이득={np.mean(sc)-np.mean(sm):+6.1f})")
    if name != REF:
        ds = []
        for f in FOLDS:
            lr, base = ens_loss[(REF, f)]
            ln, _ = ens_loss[(name, f)]
            ds.append((lr - ln) / base * 100000)
        d = np.concatenate(ds)
        gain, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        line += f"  | vs {REF}: {gain:+7.2f}점 (SE={se:.1f}, {gain/se:+.1f}σ)"
    log(line)

best_set = max(means, key=means.get)
log(f"\nSTAGE 1 최고: {best_set}")

# ============================================================== STAGE 2
log("\n" + "=" * 78)
log(f"STAGE 2 — 시즌 가중치 스윕 (피처={best_set})")
log("=" * 78)

cfg = FEATURE_SETS[best_set]
DECAYS = [1.0, 0.9, 0.8, 0.65]
s2_loss, s2_score = {}, {}

for val_season in FOLDS:
    log(f"\n[fold val={val_season}]")
    tr_m, va_m, y_tr, y_va, shr, te_tr, te_va = build_fold(val_season, [cfg])
    X_tr, X_va = assemble(cfg, tr_m, va_m, shr, te_tr, te_va)
    yv = y_va.to_numpy()
    sw = seasons[tr_m]

    for decay in DECAYS:
        if decay == 1.0 and (best_set, val_season) in ens_loss:
            s2_loss[(decay, val_season)] = ens_loss[(best_set, val_season)]
            s2_score[(decay, val_season)] = ens_score[(best_set, val_season)]
            log(f"  -- decay={decay} (STAGE1 결과 재사용: {ens_score[(best_set, val_season)]:.2f})")
            continue
        log(f"  -- decay={decay}")
        w = None if decay == 1.0 else season_weights(sw, decay)
        p, s, base, _ = run_ensemble(X_tr, y_tr, X_va, y_va, num_cols_for(cfg),
                                      weights=w, tag=f"decay={decay}")
        s2_loss[(decay, val_season)] = ((p - yv) ** 2, base)
        s2_score[(decay, val_season)] = s

log("\n" + "-" * 78)
log("STAGE 2 요약 (decay=1.0 대비 paired 차이)")
log("-" * 78)
for decay in DECAYS:
    sc = [s2_score[(decay, f)] for f in FOLDS]
    line = f"decay={decay:<5} 앙상블평균={np.mean(sc):8.2f}"
    if decay != 1.0:
        ds = []
        for f in FOLDS:
            lr, base = s2_loss[(1.0, f)]
            ln, _ = s2_loss[(decay, f)]
            ds.append((lr - ln) / base * 100000)
        d = np.concatenate(ds)
        gain, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
        line += f"  | vs decay=1.0: {gain:+7.2f}점 (SE={se:.1f}, {gain/se:+.1f}σ)"
    log(line)

best_decay = max(DECAYS, key=lambda d: np.mean([s2_score[(d, f)] for f in FOLDS]))
log(f"\n>>> 최종 선택: feature_set={best_set}, decay={best_decay}")
