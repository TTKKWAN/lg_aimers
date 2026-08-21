"""최종 제출 모델 빌드 — 전체 시즌(2019~2024)으로 이질 앙상블 학습 후 번들 저장.

번들에는 추론에 필요한 모든 것을 담는다:
  members     : 학습된 파이프라인 리스트
  prior       : EB 축소용 리그 평균 (학습 데이터에서만 산출)
  k           : 축소 강도
  num_cols    : 추론 피처의 전체 수치형 컬럼 순서(union)
  member_num_cols: 각 멤버가 실제 학습한 수치형 컬럼 순서
  era_specs   : 모든 모델이 공유하는 시즌별 중심과 미래 외삽식
  context_members/context_num_cols/context_weight: 압박 context 보조 앙상블 계약
  te_maps     : 타깃 인코딩 룩업 (사용 시)
  recenter_to : 추론 시 맞출 평균 예측치 (None 이면 재중심화 안 함)

사용법:  python3 build_final.py [k] [te0|te1] [n_hgb] [recenter_fraction] [n_lgbm] [era0|era1(공통 regime)] [context0|context1] [season0|season1] [lgbm_family_weight]
"""
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, TE_FEATURE_COLS, TE_COLS,
                       RATE_N_PAIRS, ERA_SKILL_COLS, add_derived, add_shrinkage,
                       shrinkage_cols, fit_prior, fit_era_prior, add_era_features,
                       regime_base_num_cols, regime_season_success_cols,
                       add_regime_current_features,
                       PRESSURE_ABILITY_COLS, CONTEXT_NUM_COLS,
                       add_pressure_ability, TargetEncoder, make_model,
                       make_lgbm_model, make_context_lgbm_model, bss,
                       season_success_cols, add_season_success_train_features,
                       fit_season_success_lookup)

DATA_DIR = "./open/data"
OUT = "./open/baseline_submit/model/bundle.pkl"

K = int(sys.argv[1]) if len(sys.argv) > 1 else 50
USE_TE = (sys.argv[2] == "te1") if len(sys.argv) > 2 else False
N_MEMBERS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
# 재중심화 비율: 0=안 함(모델 자연 평균 유지), 1=추세 외삽값까지 완전 이동
RECENTER_F = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
N_LGBM = int(sys.argv[5]) if len(sys.argv) > 5 else 0
# 기존 CLI 위치/표현(era1)은 유지하되 v13에서는 모든 모델의 공통 전처리다.
USE_SHARED_REGIME = (sys.argv[6] == "era1") if len(sys.argv) > 6 else False
USE_CONTEXT = (sys.argv[7] == "context1") if len(sys.argv) > 7 else False
USE_SEASON_SUCCESS = (sys.argv[8] == "season1") if len(sys.argv) > 8 else False
LGBM_FAMILY_WEIGHT = float(sys.argv[9]) if len(sys.argv) > 9 else None
CONTEXT_WEIGHT = 0.20 if USE_CONTEXT else 0.0

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

# diag_ensemble_member.py에서 3폴드(2022/2023/2024) 전부 유의미한 이득을 확인한 구성
LGBM_HETERO = [
    dict(seed=99,    learning_rate=0.03, num_leaves=63,  min_child_samples=30,  colsample_bytree=0.8, subsample=0.8),
    dict(seed=2718,  learning_rate=0.05, num_leaves=31,  min_child_samples=50,  colsample_bytree=0.7, subsample=0.7),
    dict(seed=31415, learning_rate=0.02, num_leaves=127, min_child_samples=20,  colsample_bytree=0.9, subsample=0.9),
][:N_LGBM]

# fixed-EB context screen/confirm에서 사용한 3개 구성 그대로 고정한다.
CONTEXT_HETERO = [
    dict(seed=8049, learning_rate=0.03, num_leaves=31,
         min_child_samples=100, colsample_bytree=0.85, subsample=0.8,
         reg_lambda=2.0),
    dict(seed=2718, learning_rate=0.05, num_leaves=15,
         min_child_samples=200, colsample_bytree=0.70, subsample=0.7,
         reg_lambda=3.0),
    dict(seed=31415, learning_rate=0.02, num_leaves=63,
         min_child_samples=75, colsample_bytree=0.95, subsample=0.9,
         reg_lambda=1.0),
] if USE_CONTEXT else []


def log(*a):
    print(*a, flush=True)


log(f"설정: k={K} te={USE_TE} hgb_members={N_MEMBERS} lgbm_members={N_LGBM} "
    f"recenter_f={RECENTER_F} shared_regime={USE_SHARED_REGIME} "
    f"context_members={len(CONTEXT_HETERO)} context_weight={CONTEXT_WEIGHT:.2f} "
    f"season_success={USE_SEASON_SUCCESS} lgbm_family_weight={LGBM_FAMILY_WEIGHT}")
if LGBM_FAMILY_WEIGHT is not None and not (0.0 <= LGBM_FAMILY_WEIGHT <= 1.0):
    raise ValueError("lgbm_family_weight는 [0, 1]이어야 한다")
if USE_CONTEXT and USE_SHARED_REGIME:
    raise ValueError("공통 regime 전처리 후보는 context expert를 사용하지 않는다")
if USE_CONTEXT and USE_SEASON_SUCCESS:
    raise ValueError("v7 season-to-date는 context 제거 조건으로 검증됐다")
if USE_SEASON_SUCCESS and N_LGBM == 0:
    raise ValueError("season1 피처는 검증된 LightGBM 멤버가 하나 이상 필요하다")

test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW_FEATURES = [c for c in test_cols if c != ID]
RAW_NUM = [c for c in RAW_FEATURES if c not in CAT_COLS]
raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                   usecols=RAW_FEATURES + [TARGET])
y, seasons = raw[TARGET], raw["season"]
base_df = pd.concat([raw[RAW_FEATURES], add_derived(raw)], axis=1)
season_rate = y.groupby(seasons).mean()
log("시즌별 정답률:\n" + season_rate.round(4).to_string())

# ---- 2025 정답률 추정 (학습 시즌 추세 외삽 — train 정보만 사용하므로 규칙상 합법) ----
last3 = sorted(season_rate.index)[-3:]
b, a0 = np.polyfit(np.array(last3, float),
                    np.array([season_rate[s] for s in last3], float), 1)
TARGET_SEASON = int(max(season_rate.index)) + 1
r_extrap = float(a0 + b * TARGET_SEASON)
log(f"{TARGET_SEASON} 외삽 정답률 = {r_extrap:.4f} (최근 3시즌 직선 적합, 기울기 {b:+.4f}/년)")

# ---- 피처 구성 ----
prior = fit_prior(raw)
sh = add_shrinkage(base_df, prior, K)
X = pd.concat([base_df, sh], axis=1)
era_specs = fit_era_prior(raw) if USE_SHARED_REGIME else None
if era_specs is not None:
    X = pd.concat([X, add_era_features(raw, era_specs, K)], axis=1)
    static_num_cols = regime_base_num_cols(RAW_NUM)
else:
    static_num_cols = list(RAW_NUM) + list(DERIVED_COLS) + shrinkage_cols()
season_success_lookup = None
season_cols = []
if USE_SEASON_SUCCESS:
    season_cols = (regime_season_success_cols() if era_specs is not None
                   else season_success_cols())
    train_for_season = pd.concat([X, y.rename(TARGET)], axis=1)
    season_raw = add_season_success_train_features(train_for_season, prior, K)
    X = pd.concat([X, season_raw], axis=1)
    if era_specs is not None:
        X = pd.concat([X, add_regime_current_features(X, era_specs, K)], axis=1)
    season_success_lookup = fit_season_success_lookup(raw)
context_num_cols = list(CONTEXT_NUM_COLS) if USE_CONTEXT else None
if USE_CONTEXT:
    X = pd.concat([X, add_pressure_ability(X, prior)], axis=1)

te_maps = None
if USE_TE:
    te = TargetEncoder(smooth=200.0)
    X = pd.concat([X, te.fit_transform_oof(raw, y, n_splits=5, seed=0)], axis=1)
    static_num_cols += TE_FEATURE_COLS
    te_maps = dict(global_=te.global_, maps={c: te.maps_[c] for c in TE_COLS},
                    smooth=te.smooth, cols=TE_COLS)

lgbm_num_cols = list(static_num_cols)
lgbm_num_cols += season_cols
num_cols = list(static_num_cols)
for col in lgbm_num_cols:
    if col not in num_cols:
        num_cols.append(col)
if context_num_cols is not None:
    for col in context_num_cols:
        if col not in num_cols:
            num_cols.append(col)
log(f"학습 행렬: {X.shape}  union={len(CAT_COLS)+len(num_cols)} "
    f"hgb={len(CAT_COLS)+len(static_num_cols)} lgbm={len(CAT_COLS)+len(lgbm_num_cols)} "
    f"context={len(CAT_COLS)+len(context_num_cols) if context_num_cols else 0}")

members = []
member_num_cols = []
for i, spec in enumerate(HETERO):
    spec = dict(spec)
    seed = spec.pop("seed")
    t = time.time()
    m = make_model(static_num_cols, seed=seed, **spec)
    m.fit(X[CAT_COLS + static_num_cols], y)
    members.append(m)
    member_num_cols.append(list(static_num_cols))
    log(f"  [hgb] member {i+1}/{N_MEMBERS} 학습 완료 n_iter={m.named_steps['clf'].n_iter_} "
        f"({time.time()-t:.0f}s)")

for i, spec in enumerate(LGBM_HETERO):
    spec = dict(spec)
    seed = spec.pop("seed")
    t = time.time()
    m = make_lgbm_model(lgbm_num_cols, seed=seed, **spec)
    m.fit(X[CAT_COLS + lgbm_num_cols], y)
    members.append(m)
    member_num_cols.append(list(lgbm_num_cols))
    log(f"  [lgbm] member {i+1}/{N_LGBM} 학습 완료 ({time.time()-t:.0f}s)")

context_members = []
for i, spec in enumerate(CONTEXT_HETERO):
    spec = dict(spec)
    seed = spec.pop("seed")
    t = time.time()
    m = make_context_lgbm_model(context_num_cols, seed=seed, **spec)
    m.fit(X[CAT_COLS + context_num_cols], y)
    context_members.append(m)
    log(f"  [context] member {i+1}/{len(CONTEXT_HETERO)} 학습 완료 "
        f"({time.time()-t:.0f}s)")

if len(member_num_cols) != len(members):
    raise RuntimeError("members와 member_num_cols 길이 불일치")
if context_members and not context_num_cols:
    raise RuntimeError("context_members가 있지만 context_num_cols 계약이 없음")

# ---- 재중심화용 로짓 오프셋 delta 를 '학습 데이터에서' 미리 계산 ----
# 규칙 §5 상 추론 시 평가 데이터 전체의 평균을 구해 보정하는 것은 금지된다.
# 대신 여기서 상수 delta 를 구해 두고, 추론에서는 각 행에 독립적으로
# sigmoid(logit(p) + delta) 만 적용한다(= 행 단위 계산, 규칙 준수).
last_season = int(max(season_rate.index))
m_last = (seasons == last_season).to_numpy()
p_last_members = [
    m.predict_proba(X.loc[m_last, CAT_COLS + mcols])[:, 1]
    for m, mcols in zip(members, member_num_cols)
]
if LGBM_FAMILY_WEIGHT is not None and N_MEMBERS and N_LGBM:
    p_last_hgb = np.mean(p_last_members[:N_MEMBERS], axis=0)
    p_last_lgbm = np.mean(p_last_members[N_MEMBERS:], axis=0)
    p_last_base = ((1.0 - LGBM_FAMILY_WEIGHT) * p_last_hgb
                   + LGBM_FAMILY_WEIGHT * p_last_lgbm)
else:
    p_last_base = np.mean(p_last_members, axis=0)
if context_members:
    p_last_context = np.mean([
        m.predict_proba(X.loc[m_last, CAT_COLS + context_num_cols])[:, 1]
        for m in context_members
    ], axis=0)
    p_last = ((1.0 - CONTEXT_WEIGHT) * p_last_base
              + CONTEXT_WEIGHT * p_last_context)
else:
    p_last = p_last_base
log(f"마지막 학습 시즌({last_season}) 평균 예측={p_last.mean():.4f} "
    f"(실제 {season_rate[last_season]:.4f})")

logit_shift = 0.0
if RECENTER_F > 0:
    natural = float(p_last.mean())
    target = natural + RECENTER_F * (r_extrap - natural)
    lg = np.log(np.clip(p_last, 1e-9, 1 - 1e-9) / (1 - np.clip(p_last, 1e-9, 1 - 1e-9)))
    lo, hi = -6.0, 6.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(lg + mid)))).mean() < target:
            lo = mid
        else:
            hi = mid
    logit_shift = float((lo + hi) / 2)
    log(f"재중심화: {natural:.4f} -> {target:.4f} "
        f"(외삽 {r_extrap:.4f} 의 {RECENTER_F:.0%} 지점)  logit_shift={logit_shift:+.5f}")

bundle = dict(members=members, member_num_cols=member_num_cols,
               context_members=context_members,
               context_num_cols=context_num_cols,
               context_weight=CONTEXT_WEIGHT,
               prior=prior, era_specs=era_specs, k=K, num_cols=num_cols,
               season_success_lookup=season_success_lookup,
               lgbm_family_weight=LGBM_FAMILY_WEIGHT,
               cat_cols=CAT_COLS, te_maps=te_maps, logit_shift=logit_shift,
               meta=dict(version=("v13_shared_regime_base" if era_specs is not None
                                  else "legacy_base"),
                         n_members=len(members), n_hgb=N_MEMBERS, n_lgbm=N_LGBM,
                         n_context=len(context_members), context_weight=CONTEXT_WEIGHT,
                         r_extrap=r_extrap, target_season=TARGET_SEASON,
                         recenter_f=RECENTER_F, last_season=last_season,
                         p_last_mean=float(p_last.mean()),
                         era_lgbm=USE_SHARED_REGIME,
                         shared_regime=bool(era_specs is not None),
                         season_success=USE_SEASON_SUCCESS,
                         lgbm_family_weight=LGBM_FAMILY_WEIGHT,
                         n_season_features=len(season_cols)))

os.makedirs(os.path.dirname(OUT), exist_ok=True)
joblib.dump(bundle, OUT, compress=3)
log(f"저장 완료: {OUT}  ({os.path.getsize(OUT)/1e6:.1f}MB)")
