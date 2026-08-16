"""진단: 시즌 간 base rate 드리프트(level 문제)가 점수를 얼마나 갉아먹는가.

시즌별 control_success 평균이 단조 감소한다:
  2019 .5647 / 2020 .5327 / 2021 .5328 / 2022 .5289 / 2023 .5000 / 2024 .4861
과거로 학습해 미래를 예측하면 예측의 '평균 수준'이 실제보다 높게 잡히고,
Brier에서 이 편향은 bias^2 만큼 그대로 손해다. 신호(~500점)보다 훨씬 크다.

측정 내용:
  1) 모델의 평균 예측치 vs 실제 검증 시즌 정답률 (편향의 크기)
  2) raw score
  3) oracle 재중심화 — 검증 시즌의 실제 r 을 알 때 (달성 가능 상한)
  4) 현실적 재중심화 — 학습 시즌들의 추세를 외삽해 목표 시즌 r 을 추정 (규칙상 합법)
  => 이 차이가 곧 '순수 판별력(discrimination)' 과 '수준 보정' 의 분해다.

변형:
  A season 피처 포함(현재 방식)  B season 피처 제외
  C 최근 시즌 가중(decay .7)     D 최근 2시즌만 학습
"""
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                       make_model, bss, season_weights)

DATA_DIR = "./open/data"
FOLDS = [2023, 2024]
SEED = 42


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
log("시즌별 정답률:\n" + season_rate.round(4).to_string())


def shift_to_mean(p, target):
    """로짓에 상수를 더해 평균 예측치를 target 에 맞춘다 (0~1 유지)."""
    lg = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    lo, hi = -6.0, 6.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(lg + mid)))).mean() < target:
            lo = mid
        else:
            hi = mid
    return 1 / (1 + np.exp(-(lg + (lo + hi) / 2)))


def extrapolate_rate(train_seasons, target_season, n_last=3):
    """학습 시즌들의 정답률에 직선을 적합해 목표 시즌 값을 외삽 (train 정보만 사용)."""
    s = sorted(train_seasons)[-n_last:]
    xs = np.array(s, dtype=float)
    ys = np.array([season_rate[v] for v in s], dtype=float)
    b, a = np.polyfit(xs, ys, 1)
    return float(a + b * target_season)


VARIANTS = {
    "A_season_included": dict(drop_season=False, decay=1.0, last_n=None),
    "B_season_dropped":  dict(drop_season=True,  decay=1.0, last_n=None),
    "C_recency_decay.7": dict(drop_season=False, decay=0.7, last_n=None),
    "D_last2_seasons":   dict(drop_season=False, decay=1.0, last_n=2),
}

for val_season in FOLDS:
    tr_all = sorted(s for s in season_rate.index if s < val_season)
    r_true = float(season_rate[val_season])
    r_hat = extrapolate_rate(tr_all, val_season, n_last=3)
    r_trainmean = float(y[(seasons < val_season).to_numpy()].mean())
    log(f"\n{'='*78}\n[fold val={val_season}]  실제 r={r_true:.4f}  "
        f"학습평균={r_trainmean:.4f}  외삽추정={r_hat:.4f} (오차 {r_hat-r_true:+.4f})")

    va_m = (seasons == val_season).to_numpy()
    y_va = y[va_m]
    yv = y_va.to_numpy()

    for name, v in VARIANTS.items():
        tr_m = (seasons < val_season).to_numpy()
        if v["last_n"]:
            keep = set(tr_all[-v["last_n"]:])
            tr_m &= seasons.isin(keep).to_numpy()
        num_cols = list(RAW_NUM) + list(DERIVED_COLS)
        if v["drop_season"]:
            num_cols = [c for c in num_cols if c != "season"]
        cols = CAT_COLS + num_cols

        w = None if v["decay"] == 1.0 else season_weights(seasons[tr_m], v["decay"])
        t = time.time()
        m = make_model(num_cols, seed=SEED)
        if w is None:
            m.fit(base_df.loc[tr_m, cols], y[tr_m])
        else:
            m.fit(base_df.loc[tr_m, cols], y[tr_m], clf__sample_weight=w)
        p = m.predict_proba(base_df.loc[va_m, cols])[:, 1]

        _, s_raw, _ = bss(y_va, p)
        _, s_oracle, _ = bss(y_va, shift_to_mean(p, r_true))
        _, s_extrap, _ = bss(y_va, shift_to_mean(p, r_hat))
        log(f"  {name:18s} n_train={tr_m.sum():>9,}  mean(p)={p.mean():.4f} "
            f"(편향 {p.mean()-r_true:+.4f})")
        log(f"  {'':18s}   raw={s_raw:8.2f}   외삽보정={s_extrap:8.2f}   "
            f"oracle보정={s_oracle:8.2f}   ({time.time()-t:.0f}s)")
