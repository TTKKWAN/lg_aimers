"""ABS 민감 rate를 완전히 차단한 stable LightGBM backbone forward 검증.

기존 모델/번들은 변경하지 않고 experiments/preds에 fold 예측과 요약만 저장한다.
사용자가 명시적으로 로컬 학습을 허용한 경우 또는 Colab에서 실행한다.

실행:
    python3 scripts/experiment_stable_backbone.py
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from pipeline import (
    CAT_COLS, DERIVED_COLS, ID, PREV_SPECS, RATE_N_PAIRS, TARGET,
    add_derived, add_shrinkage, fit_prior, make_lgbm_model, shrinkage_cols,
)


DATA = "./open/data"
OUT = "./experiments/preds"
K = 50
FOLDS = [2022, 2023, 2024]
SEEDS = [99, 2718, 31415]

ABS_DIRECT = [
    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_batter_success_rate",
    "asof_batter_middle_rate",
]
ABS_RECENT = [c for c, _, _ in PREV_SPECS]
ABS_RAW = set(ABS_DIRECT + ABS_RECENT)
ABS_DERIVED = {
    "pitcher_command_gap", "pitcher_recent_trend", "batter_pitcher_gap",
}
ABS_SHRINK = (
    {f"sh_{c}" for c in ABS_DIRECT}
    | {f"sh_{c}" for c in ABS_RECENT}
    | {f"dev_{c}" for c in ABS_RECENT}
)


def bss(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    base = y.mean() * (1.0 - y.mean())
    loss = (p - y) ** 2
    return float(100000.0 * (1.0 - loss.mean() / base)), loss, base


def paired_gain(y, reference, candidate):
    score_ref, loss_ref, base = bss(y, reference)
    score_new, loss_new, _ = bss(y, candidate)
    delta = loss_ref - loss_new
    gain = float(100000.0 * delta.mean() / base)
    se = float(100000.0 * delta.std(ddof=1) / np.sqrt(len(delta)) / base)
    return score_ref, score_new, gain, se


def main():
    os.makedirs(OUT, exist_ok=True)
    test_cols = pd.read_csv(
        f"{DATA}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(
        f"{DATA}/train.csv", encoding="utf-8-sig",
        usecols=raw_features + [TARGET])
    y = raw[TARGET].to_numpy(dtype=np.int8)
    base = pd.concat([raw[raw_features], add_derived(raw)], axis=1)

    baseline_num = list(raw_num) + list(DERIVED_COLS) + shrinkage_cols()
    stable_num = [c for c in raw_num if c not in ABS_RAW]
    stable_num += [c for c in DERIVED_COLS if c not in ABS_DERIVED]
    stable_num += [c for c in shrinkage_cols() if c not in ABS_SHRINK]
    if len(stable_num) != len(set(stable_num)):
        raise RuntimeError("stable feature 중복")
    leaked = ABS_RAW.intersection(stable_num) | ABS_DERIVED.intersection(stable_num)
    leaked |= ABS_SHRINK.intersection(stable_num)
    if leaked:
        raise RuntimeError(f"stable 경로에 ABS 민감 피처 잔존: {sorted(leaked)}")
    print(f"features: baseline={len(CAT_COLS)+len(baseline_num)} "
          f"stable={len(CAT_COLS)+len(stable_num)}", flush=True)

    summaries = []
    for fold in FOLDS:
        train = raw["season"].lt(fold).to_numpy()
        valid = raw["season"].eq(fold).to_numpy()
        prior = fit_prior(raw.loc[train])
        sh = add_shrinkage(base, prior, K)
        x = pd.concat([base, sh], axis=1)
        fold_preds = {"baseline": [], "stable": []}
        started = time.time()
        for seed in SEEDS:
            for variant, cols in (
                ("baseline", baseline_num), ("stable", stable_num)):
                model = make_lgbm_model(cols, seed=seed)
                model.fit(x.loc[train, CAT_COLS + cols], y[train])
                pred = model.predict_proba(
                    x.loc[valid, CAT_COLS + cols])[:, 1]
                fold_preds[variant].append(pred)
                print(f"fold={fold} seed={seed} {variant} done", flush=True)
        p0 = np.mean(fold_preds["baseline"], axis=0)
        ps = np.mean(fold_preds["stable"], axis=0)
        score0, scores, gain, se = paired_gain(y[valid], p0, ps)
        summaries.append({
            "fold": fold, "baseline_bss": score0, "stable_bss": scores,
            "stable_gain": gain, "paired_se": se,
            "z": gain / se if se > 0 else np.nan,
            "n_train": int(train.sum()), "n_valid": int(valid.sum()),
            "n_baseline_features": len(CAT_COLS) + len(baseline_num),
            "n_stable_features": len(CAT_COLS) + len(stable_num),
            "elapsed_sec": time.time() - started,
        })
        np.savez_compressed(
            f"{OUT}/stable_backbone_{fold}.npz", y=y[valid],
            p_baseline=p0, p_stable=ps, seeds=np.asarray(SEEDS),
            baseline_cols=np.asarray(CAT_COLS + baseline_num),
            stable_cols=np.asarray(CAT_COLS + stable_num),
        )
        print(pd.DataFrame(summaries).tail(1).to_string(index=False), flush=True)
        del x, sh

    summary = pd.DataFrame(summaries)
    summary.to_csv(f"{OUT}/stable_backbone_summary.csv", index=False)
    print("\n" + summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
