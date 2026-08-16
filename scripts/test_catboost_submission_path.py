"""현재 production 번들의 CatBoost feature/혼합/recenter 계약을 직접 검증한다."""
import importlib.util
import os

import joblib
import numpy as np
import pandas as pd

from pipeline import (CATBOOST_CAT_COLS, add_catboost_context,
                      add_season_command_features)


SCRIPT = "./open/baseline_submit/script.py"
BUNDLE = os.environ.get(
    "LGAIMERS_BUNDLE", "./open/baseline_submit/model/bundle.pkl")
TEST = "./open/data/test.csv"

spec = importlib.util.spec_from_file_location("production_script", SCRIPT)
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)

bundle = joblib.load(BUNDLE)
test = pd.read_csv(TEST, encoding="utf-8-sig")
X = production.build_features(test, bundle)
cat_train = add_catboost_context(test)
cat_submit = production.add_catboost_context(test)
assert list(cat_train.columns) == list(cat_submit.columns) == CATBOOST_CAT_COLS
assert (cat_train.to_numpy() == cat_submit.to_numpy()).all(), \
    "CatBoost 학습/제출 category 피처 불일치"

assert len(bundle.get("catboost_members", [])) == 2
assert float(bundle["catboost_weight"]) == 0.60
assert set(bundle["catboost_cat_cols"]).issubset(X.columns)
for col in bundle["catboost_cat_cols"]:
    assert X[col].dtype == object, f"CatBoost category가 문자열이 아님: {col}"

member_preds = [
    m.predict_proba(X[bundle["cat_cols"] + cols])[:, 1]
    for m, cols in zip(bundle["members"], bundle["member_num_cols"])
]
n_hgb = int(bundle["meta"]["n_hgb"])
p_hgb = np.mean(member_preds[:n_hgb], axis=0)
p_lgbm = np.mean(member_preds[n_hgb:], axis=0)
p_v8 = 0.25 * p_hgb + 0.75 * p_lgbm
p_cat = np.mean([
    m.predict_proba(X[bundle["catboost_feature_cols"]])[:, 1]
    for m in bundle["catboost_members"]
], axis=0)
command_members = bundle.get("catboost_command_members") or []
if command_members:
    assert len(command_members) == 2
    assert float(bundle["catboost_command_weight"]) == 0.50
    command_cols = bundle["catboost_command_feature_cols"]
    command_train = add_season_command_features(
        X, bundle["season_command_lookup"], bundle["prior"], bundle["k"])
    assert np.allclose(X[command_cols[-12:]].to_numpy(float),
                       command_train[command_cols[-12:]].to_numpy(float),
                       equal_nan=True), "command 학습/제출 피처 불일치"
    p_command = np.mean([
        m.predict_proba(X[command_cols])[:, 1] for m in command_members
    ], axis=0)
    w_command = float(bundle["catboost_command_weight"])
    p_cat = (1.0 - w_command) * p_cat + w_command * p_command
p_raw = 0.40 * p_v8 + 0.60 * p_cat
abs_members = bundle.get("abs_regime_members") or []
if abs_members:
    p_abs = np.mean([
        m.predict_proba(X[bundle["abs_regime_feature_cols"]])[:, 1]
        for m in abs_members
    ], axis=0)
    p_raw = ((1.0 - float(bundle["abs_regime_weight"])) * p_raw
             + float(bundle["abs_regime_weight"]) * p_abs)
if bundle.get("pitcher_chase_policy_lookup"):
    p_raw = production.apply_pitcher_chase_policy(
        test, p_raw, bundle["pitcher_chase_policy_lookup"],
        float(bundle["pitcher_chase_k_state"]),
        float(bundle["pitcher_chase_k_current"]),
        float(bundle["pitcher_chase_w_max"]))
q = np.clip(p_raw, 1e-9, 1 - 1e-9)
p_expected = 1 / (1 + np.exp(-(
    np.log(q / (1 - q)) + float(bundle["logit_shift"]))))

# main과 같은 유틸 경로도 결과가 보존되는지 확인한다.
sub = pd.read_csv("./open/data/sample_submission.csv", encoding="utf-8-sig")
merged = production.merge_predictions(sub, test["row_id"].tolist(), p_expected)
assert np.allclose(merged["control_success"].to_numpy(), p_expected, atol=1e-12)
assert np.isfinite(p_expected).all() and ((p_expected > 0) & (p_expected < 1)).all()
print(f"CATBOOST SUBMISSION PATH PASSED — rows={len(test)}, features={X.shape[1]}, "
      f"command_members={len(command_members)}, abs_members={len(abs_members)}, "
      f"pitcher_chase={bool(bundle.get('pitcher_chase_policy_lookup'))}, "
      f"raw_mean={p_raw.mean():.8f}, "
      f"final_mean={p_expected.mean():.8f}")
