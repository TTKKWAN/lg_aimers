"""현재 제출 번들의 평가 규모 피처 생성+추론 시간을 측정한다."""
import importlib.util
import os
import time

import joblib
import numpy as np
import pandas as pd

N = 245_789
SCRIPT = "./open/baseline_submit/script.py"
BUNDLE = os.environ.get(
    "LGAIMERS_BUNDLE", "./open/baseline_submit/model/bundle.pkl")
TEST = "./open/data/test.csv"

spec = importlib.util.spec_from_file_location("submission_script", SCRIPT)
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)
bundle = joblib.load(BUNDLE)
small = pd.read_csv(TEST, encoding="utf-8-sig")
test = pd.concat([small] * ((N + len(small) - 1) // len(small)), ignore_index=True).iloc[:N]

t0 = time.perf_counter()
X = sub.build_features(test, bundle)
t1 = time.perf_counter()
member_cols = bundle.get("member_num_cols") or [bundle["num_cols"]] * len(bundle["members"])
member_preds = [m.predict_proba(X[bundle["cat_cols"] + cols])[:, 1]
                for m, cols in zip(bundle["members"], member_cols)]
n_hgb = int(bundle.get("meta", {}).get("n_hgb", 0))
n_lgbm = int(bundle.get("meta", {}).get("n_lgbm", 0))
family_weight = bundle.get("lgbm_family_weight")
if family_weight is not None and n_hgb and n_lgbm:
    p = ((1 - family_weight) * np.mean(member_preds[:n_hgb], axis=0)
         + family_weight * np.mean(member_preds[n_hgb:], axis=0))
else:
    p = np.mean(member_preds, axis=0)
catboost_members = bundle.get("catboost_members") or []
if catboost_members:
    pc = np.mean([m.predict_proba(X[bundle["catboost_feature_cols"]])[:, 1]
                  for m in catboost_members], axis=0)
    command_members = bundle.get("catboost_command_members") or []
    if command_members:
        pcommand = np.mean([
            m.predict_proba(X[bundle["catboost_command_feature_cols"]])[:, 1]
            for m in command_members
        ], axis=0)
        wc = float(bundle["catboost_command_weight"])
        pc = (1 - wc) * pc + wc * pcommand
    w = float(bundle["catboost_weight"])
    p = (1 - w) * p + w * pc
abs_members = bundle.get("abs_regime_members") or []
if abs_members:
    pa = np.mean([
        m.predict_proba(X[bundle["abs_regime_feature_cols"]])[:, 1]
        for m in abs_members
    ], axis=0)
    wa = float(bundle["abs_regime_weight"])
    p = (1 - wa) * p + wa * pa
shift = float(bundle.get("logit_shift", 0.0))
if shift:
    q = np.clip(p, 1e-9, 1 - 1e-9)
    p = 1 / (1 + np.exp(-(np.log(q / (1 - q)) + shift)))
p = np.clip(p, 1e-6, 1 - 1e-6)
t2 = time.perf_counter()
print(f"rows={len(test)} features={X.shape[1]} members={len(bundle['members'])} "
      f"catboost={len(catboost_members)} "
      f"command={len(bundle.get('catboost_command_members') or [])} "
      f"abs_regime={len(abs_members)}")
print(f"feature_seconds={t1-t0:.2f}")
print(f"predict_seconds={t2-t1:.2f}")
print(f"total_seconds={t2-t0:.2f}")
print(f"pred_min={p.min():.8f} pred_max={p.max():.8f} pred_mean={p.mean():.8f}")
