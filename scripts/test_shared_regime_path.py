"""공통 ABS/시대 보정 피처의 학습·제출 경로 동등성 검사(모델 fit 없음)."""
import importlib.util

import numpy as np
import pandas as pd

from pipeline import (
    ID, TARGET, CAT_COLS, REGIME_RAW_COLS, SEASON_COMMAND_RATES,
    add_derived, add_shrinkage, fit_prior, fit_era_prior, add_era_features,
    regime_base_num_cols, regime_season_success_cols, regime_current_cols,
    add_regime_current_features, add_season_success_features,
    add_season_command_features, fit_season_success_lookup,
    fit_season_command_lookup,
)

DATA = "./open/data"
SCRIPT = "./open/baseline_submit/script.py"
K = 50


def main():
    spec = importlib.util.spec_from_file_location("submission_script", SCRIPT)
    submission = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submission)

    test_cols = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_cols = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_cols if c not in CAT_COLS]
    train = pd.read_csv(
        f"{DATA}/train.csv", encoding="utf-8-sig",
        usecols=raw_cols + [TARGET], nrows=50_000,
    )
    test = pd.read_csv(f"{DATA}/test.csv", encoding="utf-8-sig")

    prior = fit_prior(train)
    era_specs = fit_era_prior(train)
    success_lookup = fit_season_success_lookup(train)
    command_lookup = fit_season_command_lookup(train)
    base_cols = regime_base_num_cols(raw_num)
    success_cols = regime_season_success_cols()
    command_cols = regime_current_cols(SEASON_COMMAND_RATES)
    num_cols = base_cols + success_cols + command_cols
    bundle = {
        "prior": prior, "era_specs": era_specs, "k": K,
        "season_success_lookup": success_lookup,
        "season_command_lookup": command_lookup,
        "cat_cols": CAT_COLS, "num_cols": num_cols,
    }

    actual = submission.build_features(test, bundle)
    base = pd.concat([test[raw_cols], add_derived(test)], axis=1)
    sh = add_shrinkage(base, prior, K)
    x = pd.concat([base, sh], axis=1)
    success = add_season_success_features(x, success_lookup, prior, K)
    command = add_season_command_features(x, command_lookup, prior, K)
    era = add_era_features(test, era_specs, K)
    x = pd.concat([x, success, command, era], axis=1)
    current = add_regime_current_features(x, era_specs, K)
    expected = pd.concat([x, current], axis=1)[CAT_COLS + num_cols]

    assert list(actual.columns) == list(expected.columns)
    assert np.allclose(
        actual[num_cols].to_numpy(float), expected[num_cols].to_numpy(float),
        equal_nan=True,
    )
    assert not (set(base_cols) & REGIME_RAW_COLS), "절대 rate가 공통 입력에 남아 있음"
    assert not any(c.startswith("sh_asof_") and c.endswith("_rate")
                   for c in base_cols), "절대 EB rate가 공통 입력에 남아 있음"

    single = submission.build_features(test.iloc[[0]].copy(), bundle)
    assert np.allclose(
        single[num_cols].to_numpy(float), actual.iloc[[0]][num_cols].to_numpy(float),
        equal_nan=True,
    )
    shuffled = test.sample(frac=1, random_state=17)
    shuffled_x = submission.build_features(shuffled, bundle).loc[test.index]
    assert np.allclose(
        shuffled_x[num_cols].to_numpy(float), actual[num_cols].to_numpy(float),
        equal_nan=True,
    )
    print(
        f"SHARED REGIME PATH PASSED — rows={len(test)}, "
        f"base={len(base_cols)}, success={len(success_cols)}, "
        f"command={len(command_cols)}, total={actual.shape[1]}"
    )


if __name__ == "__main__":
    main()
