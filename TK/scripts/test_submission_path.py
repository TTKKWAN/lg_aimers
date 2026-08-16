"""제출 경로 스모크 테스트 — 작은 번들을 만들어 script.py 를 실제로 실행.

가장 중요한 검증: script.py 가 만드는 피처가 학습(pipeline.py)과 **완전히 동일**한가.
컬럼 순서나 값이 하나라도 어긋나면 리더보드에서 조용히 성능이 무너지므로
여기서 numerically identical 인지 직접 비교한다.
"""
import importlib.util
import os
import shutil
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, RATE_N_PAIRS,
                       ERA_SKILL_COLS,
                       PRESSURE_ABILITY_COLS, CONTEXT_NUM_COLS,
                       add_derived, add_shrinkage, add_pressure_ability,
                       shrinkage_cols, fit_prior, fit_era_prior, add_era_features,
                       make_model, make_lgbm_model,
                       make_context_lgbm_model, season_success_cols,
                       add_season_success_train_features, add_season_success_features,
                       fit_season_success_lookup, season_command_cols,
                       add_season_command_train_features,
                       add_season_command_features, fit_season_command_lookup)

SCRATCH = "/private/tmp/lgaimers_context_subtest"
DATA_DIR = "./open/data"
K = 50

shutil.rmtree(SCRATCH, ignore_errors=True)
os.makedirs(f"{SCRATCH}/model")
os.makedirs(f"{SCRATCH}/data")

test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW = [c for c in test_cols if c != ID]
RAW_NUM = [c for c in RAW if c not in CAT_COLS]
raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                   usecols=RAW + [TARGET], nrows=50000)
y = raw[TARGET]
base_df = pd.concat([raw[RAW], add_derived(raw)], axis=1)
prior = fit_prior(raw)
sh = add_shrinkage(base_df, prior, K)
static_num_cols = list(RAW_NUM) + list(DERIVED_COLS) + shrinkage_cols()
pressure = add_pressure_ability(pd.concat([base_df, sh], axis=1), prior)
context_num_cols = list(CONTEXT_NUM_COLS)
num_cols = static_num_cols + [c for c in PRESSURE_ABILITY_COLS
                              if c not in static_num_cols]
X = pd.concat([base_df, sh, pressure], axis=1)
cols = CAT_COLS + num_cols

members = [
    make_model(static_num_cols, seed=42, max_iter=15, early_stopping=False).fit(
        X[CAT_COLS + static_num_cols], y),
    make_lgbm_model(static_num_cols, seed=99, n_estimators=20).fit(
        X[CAT_COLS + static_num_cols], y),
]
member_num_cols = [static_num_cols, static_num_cols]
context_members = [
    make_context_lgbm_model(context_num_cols, seed=8049, n_estimators=20).fit(
        X[CAT_COLS + context_num_cols], y),
    make_context_lgbm_model(context_num_cols, seed=2718, num_leaves=15,
                            min_child_samples=200, n_estimators=20).fit(
        X[CAT_COLS + context_num_cols], y),
]
bundle = dict(members=members, member_num_cols=member_num_cols,
               context_members=context_members,
               context_num_cols=context_num_cols, context_weight=0.20,
               prior=prior, era_specs=None, k=K, num_cols=num_cols,
               cat_cols=CAT_COLS, te_maps=None, logit_shift=0.03,
               meta=dict(n_members=2, n_hgb=1, n_lgbm=1, era_lgbm=False,
                         n_context=2, context_weight=0.20))
joblib.dump(bundle, f"{SCRATCH}/model/bundle.pkl", compress=3)

shutil.copy("./open/baseline_submit/script.py", f"{SCRATCH}/script.py")
shutil.copy("./open/baseline_submit/requirements.txt", f"{SCRATCH}/requirements.txt")
shutil.copy(f"{DATA_DIR}/test.csv", f"{SCRATCH}/data/test.csv")
shutil.copy(f"{DATA_DIR}/sample_submission.csv", f"{SCRATCH}/data/sample_submission.csv")

# ---- script.py 를 평가 서버처럼 실행 ----
r = subprocess.run([sys.executable, "script.py"], cwd=SCRATCH,
                    capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("STDERR:", r.stderr)
    sys.exit(1)

out = pd.read_csv(f"{SCRATCH}/output/submission.csv")
print(out)
assert out[TARGET].between(0, 1).all(), "확률이 0~1 범위를 벗어남"

# ---- 피처 동등성 검증: script.py 의 build_features vs 학습 파이프라인 ----
spec = importlib.util.spec_from_file_location("subscript", f"{SCRATCH}/script.py")
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

test_df = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig")
X_script = sub.build_features(test_df, bundle)

# 학습 쪽 경로로 같은 test 행의 피처를 재현
tb = pd.concat([test_df[RAW], add_derived(test_df)], axis=1)
tsh = add_shrinkage(tb, prior, K)
tpressure = add_pressure_ability(pd.concat([tb, tsh], axis=1), prior)
X_train_path = pd.concat([tb, tsh, tpressure], axis=1)[cols]

assert list(X_script.columns) == list(X_train_path.columns), "컬럼 순서 불일치"
num_s = X_script[num_cols].to_numpy(dtype=float)
num_t = X_train_path[num_cols].to_numpy(dtype=float)
assert np.allclose(num_s, num_t, equal_nan=True), "수치형 피처 값 불일치"
assert (X_script[CAT_COLS].values == X_train_path[CAT_COLS].values).all(), "범주형 불일치"
print(f"\n피처 동등성 OK — {X_script.shape[1]}개 컬럼, 값/순서 완전 일치")

# context 전용 컬럼도 순서와 값이 정확한지 별도 확인한다.
assert list(X_script[CAT_COLS + context_num_cols].columns) == \
       list(X_train_path[CAT_COLS + context_num_cols].columns), \
       "context 컬럼 순서 불일치"
ctx_s = X_script[context_num_cols].to_numpy(dtype=float)
ctx_t = X_train_path[context_num_cols].to_numpy(dtype=float)
assert np.allclose(ctx_s, ctx_t, equal_nan=True), "context 피처 값 불일치"
print(f"context 피처 동등성 OK — {len(CAT_COLS)+len(context_num_cols)}개 컬럼")

# ---- logit_shift 가 실제로 적용되는지 확인 ----
p_base = np.mean([
    m.predict_proba(X_script[CAT_COLS + mcols])[:, 1]
    for m, mcols in zip(members, member_num_cols)
], axis=0)
p_context = np.mean([
    m.predict_proba(X_script[CAT_COLS + context_num_cols])[:, 1]
    for m in context_members
], axis=0)
p_raw = 0.80 * p_base + 0.20 * p_context
q = np.clip(p_raw, 1e-9, 1 - 1e-9)
expect = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + 0.03)))
assert np.allclose(out[TARGET].to_numpy(), expect, atol=1e-9), \
       "base/context 혼합 또는 logit_shift 적용 불일치"
print("base 80% + context 20% 혼합 및 logit_shift 적용 OK")

# ---- HGB/LGBM family 가중 평균 계약 ----
family_bundle = dict(bundle)
family_bundle["context_members"] = []
family_bundle["context_num_cols"] = None
family_bundle["context_weight"] = 0.0
family_bundle["num_cols"] = static_num_cols
family_bundle["lgbm_family_weight"] = 0.75
joblib.dump(family_bundle, f"{SCRATCH}/model/bundle.pkl", compress=3)
r_family = subprocess.run([sys.executable, "script.py"], cwd=SCRATCH,
                          capture_output=True, text=True)
if r_family.returncode != 0:
    print("FAMILY STDERR:", r_family.stderr)
    sys.exit(1)
out_family = pd.read_csv(f"{SCRATCH}/output/submission.csv")
p_hgb = members[0].predict_proba(X_script[CAT_COLS + static_num_cols])[:, 1]
p_lgbm = members[1].predict_proba(X_script[CAT_COLS + static_num_cols])[:, 1]
p_family = 0.25 * p_hgb + 0.75 * p_lgbm
q_family = np.clip(p_family, 1e-9, 1 - 1e-9)
expect_family = 1 / (1 + np.exp(-(np.log(q_family / (1 - q_family)) + 0.03)))
assert np.allclose(out_family[TARGET].to_numpy(), expect_family, atol=1e-9), \
       "HGB/LGBM family 가중 평균 불일치"
print("HGB 25% + LGBM 75% family 혼합 및 logit_shift 적용 OK")

# ---- context 키가 없는 v4/v5 이하 번들의 하위 호환도 유지되는지 확인 ----
legacy_bundle = dict(bundle)
for key in ("context_members", "context_num_cols", "context_weight",
            "member_num_cols"):
    legacy_bundle.pop(key, None)
legacy_bundle["num_cols"] = static_num_cols
joblib.dump(legacy_bundle, f"{SCRATCH}/model/bundle.pkl", compress=3)
r_legacy = subprocess.run([sys.executable, "script.py"], cwd=SCRATCH,
                          capture_output=True, text=True)
if r_legacy.returncode != 0:
    print("LEGACY STDERR:", r_legacy.stderr)
    sys.exit(1)
out_legacy = pd.read_csv(f"{SCRATCH}/output/submission.csv")
q_base = np.clip(p_base, 1e-9, 1 - 1e-9)
expect_legacy = 1 / (1 + np.exp(-(np.log(q_base / (1 - q_base)) + 0.03)))
assert np.allclose(out_legacy[TARGET].to_numpy(), expect_legacy, atol=1e-9), \
       "context 키 없는 legacy 번들 하위 호환 불일치"
print("context 키 없는 legacy 번들 하위 호환 OK")

# ---- v5 HGB-static/LGBM-era의 101-column union 계약도 별도 보존 검증 ----
era_specs = fit_era_prior(raw)
era = add_era_features(raw, era_specs, K)
static_sh_rate_cols = {f"sh_{rate}" for rate, _ in RATE_N_PAIRS}
era_num_cols = [c for c in static_num_cols if c not in static_sh_rate_cols]
era_num_cols += ERA_SKILL_COLS
era_union_num = static_num_cols + [c for c in ERA_SKILL_COLS
                                   if c not in static_num_cols]
X_era_train = pd.concat([base_df, sh, era], axis=1)
era_member = make_lgbm_model(era_num_cols, seed=123, n_estimators=20).fit(
    X_era_train[CAT_COLS + era_num_cols], y)
era_bundle = dict(
    members=[members[0], era_member],
    member_num_cols=[static_num_cols, era_num_cols],
    prior=prior, era_specs=era_specs, k=K, num_cols=era_union_num,
    cat_cols=CAT_COLS, te_maps=None, logit_shift=0.03,
    meta=dict(n_members=2, n_hgb=1, n_lgbm=1, era_lgbm=True),
)
joblib.dump(era_bundle, f"{SCRATCH}/model/bundle.pkl", compress=3)
r_era = subprocess.run([sys.executable, "script.py"], cwd=SCRATCH,
                       capture_output=True, text=True)
if r_era.returncode != 0:
    print("ERA STDERR:", r_era.stderr)
    sys.exit(1)
out_era = pd.read_csv(f"{SCRATCH}/output/submission.csv")
X_era_script = sub.build_features(test_df, era_bundle)
tera = add_era_features(test_df, era_specs, K)
X_era_expected = pd.concat([tb, tsh, tera], axis=1)[
    CAT_COLS + era_union_num]
assert list(X_era_script.columns) == list(X_era_expected.columns), \
       "era union 컬럼 순서 불일치"
assert np.allclose(X_era_script[era_union_num].to_numpy(dtype=float),
                   X_era_expected[era_union_num].to_numpy(dtype=float),
                   equal_nan=True), "era union 피처 값 불일치"
p_era_raw = np.mean([
    members[0].predict_proba(X_era_script[CAT_COLS + static_num_cols])[:, 1],
    era_member.predict_proba(X_era_script[CAT_COLS + era_num_cols])[:, 1],
], axis=0)
q_era = np.clip(p_era_raw, 1e-9, 1 - 1e-9)
expect_era = 1 / (1 + np.exp(-(np.log(q_era / (1 - q_era)) + 0.03)))
assert np.allclose(out_era[TARGET].to_numpy(), expect_era, atol=1e-9), \
       "v5 static/era 멤버별 선택 또는 logit_shift 불일치"
print(f"v5 static/era union 하위 호환 OK — {len(CAT_COLS)+len(era_union_num)}개 컬럼")

# ---- v7 exact season-to-date: 값/순서, row 독립성, unseen zero fallback ----
season_lookup = fit_season_success_lookup(raw)
# n=0/rate=NaN인 단일 시즌 endpoint도 현재 target을 더해 정확히 0/1이 되어야 한다.
corner = raw.iloc[[0]].copy()
corner["pitcher_id"] = -777001
corner["batter_id"] = -777002
corner["asof_pitcher_n"] = 0
corner["asof_pitcher_success_rate"] = np.nan
corner["asof_batter_n"] = 0
corner["asof_batter_success_rate"] = np.nan
corner[TARGET] = 1
corner_lookup = fit_season_success_lookup(corner)
assert corner_lookup["pitcher"]["end_count"][-777001] == 1
assert corner_lookup["batter"]["end_count"][-777002] == 1
season_cols = season_success_cols()
season_bundle = dict(
    members=members, member_num_cols=[static_num_cols, static_num_cols + season_cols],
    prior=prior, era_specs=None, season_success_lookup=season_lookup,
    k=K, num_cols=static_num_cols + season_cols, cat_cols=CAT_COLS,
    te_maps=None, logit_shift=0.0,
)
X_season_script = sub.build_features(test_df, season_bundle)
tseason = add_season_success_features(pd.concat([tb, tsh], axis=1), season_lookup, prior, K)
X_season_expected = pd.concat([tb, tsh, tseason], axis=1)[
    CAT_COLS + static_num_cols + season_cols]
assert np.allclose(X_season_script[static_num_cols + season_cols].to_numpy(float),
                   X_season_expected[static_num_cols + season_cols].to_numpy(float),
                   equal_nan=True), "v7 season-to-date 학습/제출 값 불일치"

# batch==single 및 shuffle 불변: 다른 test 행의 존재/순서가 한 행 피처에 영향 없음.
single = sub.build_features(test_df.iloc[[0]].copy(), season_bundle).reset_index(drop=True)
batch0 = X_season_script.iloc[[0]].reset_index(drop=True)
assert np.allclose(single[static_num_cols + season_cols].to_numpy(float),
                   batch0[static_num_cols + season_cols].to_numpy(float), equal_nan=True)
shuffled = test_df.sample(frac=1, random_state=17)
xs = sub.build_features(shuffled, season_bundle).loc[test_df.index]
assert np.allclose(xs[static_num_cols + season_cols].to_numpy(float),
                   X_season_script[static_num_cols + season_cols].to_numpy(float), equal_nan=True)

unseen = test_df.iloc[[0]].copy()
unseen["pitcher_id"] = -987654321
unseen["batter_id"] = -123456789
xu = sub.build_features(unseen, season_bundle)
assert xu["std_pitcher_known"].iloc[0] == 0 and xu["std_batter_known"].iloc[0] == 0
assert np.isclose(xu["std_pitcher_n"].iloc[0], unseen["asof_pitcher_n"].iloc[0])
assert np.isclose(xu["std_batter_n"].iloc[0], unseen["asof_batter_n"].iloc[0])

# 실제 107-column LGBM을 학습하고 script.py end-to-end 멤버 선택/평균도 검증한다.
train_with_target = pd.concat([base_df, sh, y.rename(TARGET)], axis=1)
season_train = add_season_success_train_features(train_with_target, prior, K)
X_season_train = pd.concat([base_df, sh, season_train], axis=1)
season_member = make_lgbm_model(static_num_cols + season_cols, seed=222,
                                n_estimators=20).fit(
    X_season_train[CAT_COLS + static_num_cols + season_cols], y)
season_bundle["members"] = [members[0], season_member]
joblib.dump(season_bundle, f"{SCRATCH}/model/bundle.pkl", compress=3)
r_season = subprocess.run([sys.executable, "script.py"], cwd=SCRATCH,
                          capture_output=True, text=True)
if r_season.returncode != 0:
    print("SEASON STDERR:", r_season.stderr)
    sys.exit(1)
out_season = pd.read_csv(f"{SCRATCH}/output/submission.csv")
p_season_expect = np.mean([
    members[0].predict_proba(X_season_script[CAT_COLS + static_num_cols])[:, 1],
    season_member.predict_proba(
        X_season_script[CAT_COLS + static_num_cols + season_cols])[:, 1],
], axis=0)
assert np.allclose(out_season[TARGET].to_numpy(), p_season_expect, atol=1e-9), \
       "v7 HGB91/LGBM107 멤버 선택 또는 평균 불일치"
print(f"v7 season-to-date 동등성/행독립/unseen/91+107 추론 OK — {len(season_cols)}개 추가")

# ---- v10 pitcher command profile: 값/순서, row 독립성, unseen zero fallback ----
command_lookup = fit_season_command_lookup(raw)
command_cols = season_command_cols()
command_bundle = dict(season_bundle)
command_bundle["season_command_lookup"] = command_lookup
command_bundle["catboost_command_feature_cols"] = command_cols
X_command_script = sub.build_features(test_df, command_bundle)
tcommand = add_season_command_features(
    pd.concat([tb, tsh], axis=1), command_lookup, prior, K)
assert np.allclose(
    X_command_script[command_cols].to_numpy(float),
    tcommand[command_cols].to_numpy(float), equal_nan=True), \
    "v10 command 학습/제출 값 불일치"

single = sub.build_features(
    test_df.iloc[[0]].copy(), command_bundle)[command_cols].reset_index(drop=True)
batch0 = X_command_script.iloc[[0]][command_cols].reset_index(drop=True)
assert np.allclose(single, batch0, equal_nan=True)
shuffled = test_df.sample(frac=1, random_state=23)
xs = sub.build_features(shuffled, command_bundle).loc[test_df.index, command_cols]
assert np.allclose(xs, X_command_script[command_cols], equal_nan=True)

unseen = test_df.iloc[[0]].copy()
unseen["pitcher_id"] = -987654322
xu = sub.build_features(unseen, command_bundle)
direct = add_season_command_features(
    pd.concat([unseen.drop(columns=[ID]), add_derived(unseen),
               add_shrinkage(pd.concat([unseen.drop(columns=[ID]),
                                        add_derived(unseen)], axis=1), prior, K)], axis=1),
    command_lookup, prior, K)
assert np.allclose(xu[command_cols], direct[command_cols], equal_nan=True)

train_command = add_season_command_train_features(train_with_target, prior, K)
assert list(train_command.columns) == command_cols
print(f"v10 command 동등성/행독립/unseen OK — {len(command_cols)}개 추가")
print("\nSUBMISSION PATH TEST PASSED")
