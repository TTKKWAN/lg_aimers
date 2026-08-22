"""학습/제출 경로의 투수별 chase postprocessor 동등성과 경계조건 검사."""
import importlib.util
import os

import joblib
import numpy as np
import pandas as pd

from pipeline import apply_pitcher_chase_policy

BUNDLE = os.environ.get(
    "LGAIMERS_BUNDLE", "./open/baseline_submit/model/bundle.pkl")


def main():
    spec = importlib.util.spec_from_file_location(
        "production", "./open/baseline_submit/script.py")
    production = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(production)
    bundle = joblib.load(BUNDLE)
    test = pd.read_csv("./open/data/test.csv")
    lookup = bundle["pitcher_chase_policy_lookup"]
    pitcher = next(iter(lookup["chase_rate"]))
    synthetic = test.iloc[[0]].copy()
    synthetic["pitcher_id"] = pitcher
    synthetic["balls_before"] = 0
    synthetic["strikes_before"] = 2
    total_n = float(lookup["end_n"][pitcher]) + 100.0
    total_count = (float(lookup["end_count"][pitcher])
                   + np.rint(100.0 * float(lookup["own_rate"][pitcher])))
    synthetic["asof_pitcher_n"] = total_n
    synthetic["asof_pitcher_success_rate"] = total_count / total_n
    test = pd.concat([test, synthetic], ignore_index=True)
    p0 = np.linspace(.25, .75, len(test))
    args = (bundle["pitcher_chase_policy_lookup"],
            bundle["pitcher_chase_k_state"], bundle["pitcher_chase_k_current"],
            bundle["pitcher_chase_w_max"])
    train_path = apply_pitcher_chase_policy(test, p0, *args)
    submit_path = production.apply_pitcher_chase_policy(test, p0, *args)
    assert np.allclose(train_path, submit_path, equal_nan=True)
    inactive = ~((test["strikes_before"] == 2) & (test["balls_before"] < 3))
    assert np.array_equal(train_path[inactive], p0[inactive])
    assert train_path[-1] != p0[-1], "known pitcher active chase row was not adjusted"
    assert np.isfinite(train_path).all()
    assert ((train_path >= 0) & (train_path <= 1)).all()
    single = apply_pitcher_chase_policy(test.iloc[[-1]], p0[[-1]], *args)
    assert np.allclose(single[0], train_path[-1])
    order = np.arange(len(test))[::-1]
    shuffled = apply_pitcher_chase_policy(test.iloc[order], p0[order], *args)
    assert np.allclose(shuffled[::-1], train_path)
    unseen = synthetic.copy()
    unseen["pitcher_id"] = -987654321
    assert apply_pitcher_chase_policy(unseen, np.array([.5]), *args)[0] == .5
    print(f"pitcher chase path equality OK: rows={len(test):,}, "
          f"changed={(train_path != p0).sum():,}, single/shuffle/unseen OK")


if __name__ == "__main__":
    main()
