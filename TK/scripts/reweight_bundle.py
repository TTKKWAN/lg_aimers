"""기존 v7 모델을 재학습하지 않고 family 가중치와 logit shift만 갱신한다."""
import os
import sys

import joblib
import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                       add_shrinkage, add_season_success_features)

MODEL = "./open/baseline_submit/model/bundle.pkl"
WEIGHT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.75
if not 0 <= WEIGHT <= 1:
    raise ValueError("family weight must be in [0,1]")

bundle = joblib.load(MODEL)
meta = bundle["meta"]
n_hgb, n_lgbm = int(meta["n_hgb"]), int(meta["n_lgbm"])
if n_hgb + n_lgbm != len(bundle["members"]):
    raise ValueError("현재 base-only v7 번들에서만 실행 가능")

test_cols = pd.read_csv("./open/data/test.csv", encoding="utf-8-sig", nrows=0).columns
raw_features = [c for c in test_cols if c != ID]
raw = pd.read_csv("./open/data/train.csv", encoding="utf-8-sig",
                  usecols=raw_features + [TARGET])
base = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
sh = add_shrinkage(base, bundle["prior"], bundle["k"])
parts = [base, sh]
feature_base = pd.concat(parts, axis=1)
if bundle.get("season_success_lookup"):
    parts.append(add_season_success_features(
        feature_base, bundle["season_success_lookup"], bundle["prior"], bundle["k"]))
X = pd.concat(parts, axis=1)[bundle["cat_cols"] + bundle["num_cols"]]

last_season = int(meta["last_season"])
mask = raw["season"].eq(last_season).to_numpy()
preds = [
    model.predict_proba(X.loc[mask, CAT_COLS + cols])[:, 1]
    for model, cols in zip(bundle["members"], bundle["member_num_cols"])
]
p_hgb = np.mean(preds[:n_hgb], axis=0)
p_lgbm = np.mean(preds[n_hgb:], axis=0)
p_last = (1 - WEIGHT) * p_hgb + WEIGHT * p_lgbm
natural = float(p_last.mean())
r_extrap = float(meta["r_extrap"])
fraction = float(meta["recenter_f"])
target = natural + fraction * (r_extrap - natural)
lg = np.log(np.clip(p_last, 1e-9, 1 - 1e-9) /
            (1 - np.clip(p_last, 1e-9, 1 - 1e-9)))
lo, hi = -6.0, 6.0
for _ in range(100):
    mid = (lo + hi) / 2
    if np.mean(1 / (1 + np.exp(-(lg + mid)))) < target:
        lo = mid
    else:
        hi = mid
shift = float((lo + hi) / 2)

bundle["lgbm_family_weight"] = WEIGHT
bundle["logit_shift"] = shift
bundle["meta"]["lgbm_family_weight"] = WEIGHT
bundle["meta"]["p_last_mean"] = natural
joblib.dump(bundle, MODEL, compress=3)
print(f"updated {MODEL}: HGB={1-WEIGHT:.2f} LGBM={WEIGHT:.2f} "
      f"natural={natural:.8f} target={target:.8f} shift={shift:+.8f} "
      f"size={os.path.getsize(MODEL)/1e6:.1f}MB")
