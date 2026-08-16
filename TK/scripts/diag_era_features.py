"""리그의 period effect와 선수의 상대 능력을 분리하는 era-adjusted 피처 실험.

현재 EB 축소 prior는 여러 시즌을 합친 고정 상수다. 그러나 성공률 수준이 시즌마다
크게 이동하므로, 동일한 과거 성공률도 어느 시즌의 리그 환경에서 나온 값인지에 따라
의미가 다르다. 각 rate의 시즌별 리그 중심을 학습 입력에서만 추정하고, 미래 시즌은
최근 3개 학습 시즌의 선형 추세로 외삽한다. 평가 행에는 미리 고정된 상수만 적용하므로
test 행 간 통계를 사용하지 않는다.

핵심 피처:
  era_skill = n/(n+k) * (rate - expected_league_rate_for_season)

즉 표본이 많은 선수만 그 시대의 리그 평균보다 얼마나 위/아래인지 강하게 믿는다.
관찰자료의 period effect 분해이지 인과효과 추정은 아니다.

사용법:
  python3 scripts/diag_era_features.py quick
  python3 scripts/diag_era_features.py confirm era_core era_all
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, RATE_N_PAIRS,
                       add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                       make_lgbm_model, bss)

DATA_DIR = "./open/data"
PRED_DIR = "./experiments/preds"
VAL_SEASONS = [2022, 2023, 2024]
K = 50

LGBM_SPECS = [
    dict(seed=99, learning_rate=0.03, num_leaves=63, min_child_samples=30,
         colsample_bytree=0.8, subsample=0.8),
    dict(seed=2718, learning_rate=0.05, num_leaves=31, min_child_samples=50,
         colsample_bytree=0.7, subsample=0.7),
    dict(seed=31415, learning_rate=0.02, num_leaves=127, min_child_samples=20,
         colsample_bytree=0.9, subsample=0.9),
]

CORE_RATES = [
    "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
    "asof_batter_success_rate", "asof_batter_middle_rate",
]
ALL_RATES = [r for r, _ in RATE_N_PAIRS]


def log(*args):
    print(*args, flush=True)


def fit_era_prior(train_df):
    """입력 X만으로 시즌별 중심과 미래 외삽값을 고정한다."""
    seasons = sorted(int(s) for s in train_df["season"].unique())
    specs = {}
    for rate, ncol in RATE_N_PAIRS:
        observed = {}
        for season in seasons:
            d = train_df.loc[train_df["season"] == season, [rate, ncol]]
            m = d[rate].notna() & (d[ncol] > 0)
            # '그 시즌에 마주치는 평균적 선수'를 뜻하도록 행 평균을 사용한다.
            # 누적 n으로 다시 가중하면 베테랑의 동일 이력값을 매 투구마다 과도하게
            # 반복 가중하는 문제가 있어, 기존 static prior와 다른 견고한 관점을 둔다.
            observed[season] = float(d.loc[m, rate].mean())
        recent = seasons[-min(3, len(seasons)):]
        if len(recent) >= 2:
            slope, intercept = np.polyfit(
                np.asarray(recent, dtype=float),
                np.asarray([observed[s] for s in recent], dtype=float), 1)
        else:
            slope, intercept = 0.0, observed[recent[0]]
        specs[rate] = dict(observed=observed, slope=float(slope), intercept=float(intercept))
    return specs


def expected_for_season(season_values, spec):
    seasons = np.asarray(season_values, dtype=int)
    out = spec["intercept"] + spec["slope"] * seasons.astype(float)
    for season, value in spec["observed"].items():
        out[seasons == season] = value
    return out


def add_era_features(df, specs):
    out = {}
    for rate, ncol in RATE_N_PAIRS:
        r = df[rate].to_numpy(dtype=float)
        n = df[ncol].to_numpy(dtype=float)
        era = expected_for_season(df["season"].to_numpy(), specs[rate])
        rf = np.where(np.isnan(r), era, r)
        nf = np.where(np.isnan(r), 0.0, n)
        reliability = nf / (nf + K)
        out[f"era_skill_{rate}"] = reliability * (rf - era)
        out[f"era_center_{rate}"] = era
    return pd.DataFrame(out, index=df.index)


def skill_cols(rates):
    return [f"era_skill_{r}" for r in rates]


def center_cols(rates):
    return [f"era_center_{r}" for r in rates]


CONFIGS = {
    "baseline": dict(add=[], drop=[]),
    "era_core": dict(add=skill_cols(CORE_RATES), drop=[]),
    "era_all": dict(add=skill_cols(ALL_RATES), drop=[]),
    "era_all_with_level": dict(add=skill_cols(ALL_RATES) + center_cols(ALL_RATES), drop=[]),
    "era_core_replace_static": dict(
        add=skill_cols(CORE_RATES),
        drop=[f"sh_{r}" for r in CORE_RATES]),
    "era_all_replace_static": dict(
        add=skill_cols(ALL_RATES),
        drop=[f"sh_{r}" for r in ALL_RATES]),
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if mode not in ("quick", "confirm"):
        raise SystemExit("mode must be quick or confirm")
    requested = sys.argv[2:]
    unknown = [x for x in requested if x not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs: {unknown}; choices={list(CONFIGS)}")
    names = ["baseline"] + [x for x in requested if x != "baseline"] if requested else list(CONFIGS)
    if mode == "confirm" and len(names) == 1:
        raise SystemExit("confirm 뒤에 후보 config를 하나 이상 지정하세요")
    model_specs = LGBM_SPECS[:1] if mode == "quick" else LGBM_SPECS

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    base_num = list(raw_num) + list(DERIVED_COLS) + shrinkage_cols()
    os.makedirs(PRED_DIR, exist_ok=True)
    losses, scores = {}, {}
    log(f"mode={mode} configs={names} members={len(model_specs)}")

    for val_season in VAL_SEASONS:
        tr_m = (seasons < val_season).to_numpy()
        va_m = (seasons == val_season).to_numpy()
        y_tr, y_va = y[tr_m], y[va_m]
        yv = y_va.to_numpy(dtype=float)

        static_prior = fit_prior(raw.loc[tr_m])
        static_sh = add_shrinkage(base_df, static_prior, K)
        era_specs = fit_era_prior(raw.loc[tr_m])
        era = add_era_features(raw, era_specs)
        X = pd.concat([base_df, static_sh, era], axis=1)
        log(f"\n[fold={val_season}] train={tr_m.sum():,} val={va_m.sum():,}")
        for r in CORE_RATES:
            spec = era_specs[r]
            predicted = spec["intercept"] + spec["slope"] * val_season
            log(f"  era {r}: predicted_{val_season}={predicted:.5f} slope={spec['slope']:+.5f}/yr")

        for name in names:
            cfg = CONFIGS[name]
            num_cols = [c for c in base_num + cfg["add"] if c not in set(cfg["drop"])]
            cols = CAT_COLS + num_cols
            ps = []
            log(f"  -- {name} features={len(cols)}")
            for i, spec0 in enumerate(model_specs):
                spec = dict(spec0)
                seed = spec.pop("seed")
                t = time.time()
                model = make_lgbm_model(num_cols, seed=seed, **spec)
                model.fit(X.loc[tr_m, cols], y_tr)
                p = model.predict_proba(X.loc[va_m, cols])[:, 1]
                ps.append(p)
                _, single, _ = bss(yv, p)
                log(f"     member={i+1}/{len(model_specs)} seed={seed} score={single:8.2f} "
                    f"time={time.time()-t:.0f}s")
                del model
                gc.collect()
            p = np.mean(ps, axis=0)
            _, score, base = bss(yv, p)
            losses[(name, val_season)] = ((p - yv) ** 2, base)
            scores[(name, val_season)] = score
            np.savez_compressed(f"{PRED_DIR}/era_{mode}_{name}_{val_season}.npz",
                                p=p, y=yv, n=len(model_specs))
            log(f"     >> ensemble={score:.2f}")
        del X, static_sh, era
        gc.collect()

    log("\n" + "=" * 92)
    log(f"{mode.upper()} 요약 — baseline 대비 동일 행 paired BSS 차이")
    log("=" * 92)
    for name in names:
        fold_scores = [scores[(name, f)] for f in VAL_SEASONS]
        line = f"{name:25s} folds=" + "/".join(f"{s:7.2f}" for s in fold_scores)
        line += f" mean={np.mean(fold_scores):7.2f}"
        if name != "baseline":
            fold_gain, fold_se = [], []
            for f in VAL_SEASONS:
                loss0, base = losses[("baseline", f)]
                loss1, _ = losses[(name, f)]
                d = (loss0 - loss1) / base * 100000
                fold_gain.append(d.mean())
                fold_se.append(d.std(ddof=1) / np.sqrt(len(d)))
            line += " gains=" + "/".join(f"{d:+7.2f}" for d in fold_gain)
            line += f" mean_gain={np.mean(fold_gain):+7.2f}"
            line += " SE=" + "/".join(f"{s:.1f}" for s in fold_se)
        log(line)


if __name__ == "__main__":
    main()
