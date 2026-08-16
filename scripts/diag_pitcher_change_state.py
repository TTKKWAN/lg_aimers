"""투수의 잠재적 변화 상태를 row-local 이력만으로 스크리닝한다.

가설:
  훈련·폼 수정·부상 회복 같은 행위는 직접 관찰할 수 없지만, 장기 EB 능력과
  직전 1/3/5경기 EB 능력 사이의 지속적 개선/악화, 가속/감속으로 흔적이 남을 수 있다.

기존 ``diag_social_features.py``의 FORM_COLS는 raw 최근값의 1v3/3v5와 표준편차를
이미 시험해 기각했다. 이 스크립트는 그 피처를 반복하지 않고 아래만 시험한다.

  state       : 장기 대비 EB 편차의 가중 요약, 1>3>5>장기 순서 상태, 곡률
  reliability : 관측 가능 비율 × 누적 투구 신뢰도로 state를 완만하게 게이팅

모든 피처는 test.csv 한 행의 제공된 asof 값과 X_train에서 고정한 EB prior만 쓴다.
test 행 간 통계, 순서, rolling/expanding 계산은 없다.

사용법:
  python3 scripts/diag_pitcher_change_state.py quick
  python3 scripts/diag_pitcher_change_state.py quick success_state all_state
  python3 scripts/diag_pitcher_change_state.py confirm all_state

quick은 LightGBM 1개, confirm은 3개 이질 LightGBM 앙상블이다. 산출물 이름은
``change_state_*``로 고정해 기존 social/direct-brier 예측과 충돌하지 않는다.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                       add_shrinkage, shrinkage_cols, fit_prior,
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


def log(*args):
    print(*args, flush=True)


def _state_block(df, kind):
    """success/middle 각각에 대해 EB축소된 변화의 모양과 신뢰도를 만든다."""
    long_col = f"sh_asof_pitcher_{kind}_rate"
    recent_cols = [f"sh_asof_pitcher_prev{n}_game_{kind}_rate" for n in (1, 3, 5)]
    miss_cols = [f"miss_asof_pitcher_prev{n}_game_{kind}_rate" for n in (1, 3, 5)]

    long = df[long_col].to_numpy(dtype=float)
    r1, r3, r5 = (df[c].to_numpy(dtype=float) for c in recent_cols)
    d1, d3, d5 = r1 - long, r3 - long, r5 - long
    short, medium = r1 - r3, r3 - r5

    # 최근일수록 더 큰 비중. 개별 d1/d3/d5 자체는 baseline shrinkage_cols의
    # dev_*에 이미 있으므로 여기서는 하나의 안정적인 상태 요약만 추가한다.
    weighted_dev = 0.50 * d1 + 0.30 * d3 + 0.20 * d5
    direction_balance = ((d1 > 0).astype(np.int8)
                         + (d3 > 0).astype(np.int8)
                         + (d5 > 0).astype(np.int8) - 1.5) / 1.5
    improving = ((r1 > r3) & (r3 > r5) & (r5 > long)).astype(np.int8)
    declining = ((r1 < r3) & (r3 < r5) & (r5 < long)).astype(np.int8)
    acceleration = short - medium

    observed_fraction = 1.0 - np.column_stack(
        [df[c].to_numpy(dtype=float) for c in miss_cols]).mean(axis=1)
    career_reliability = df["rel_asof_pitcher_n"].to_numpy(dtype=float)
    form_reliability = observed_fraction * career_reliability

    prefix = f"change_{kind}"
    state = {
        f"{prefix}_weighted_dev": weighted_dev,
        f"{prefix}_direction_balance": direction_balance,
        f"{prefix}_sustained_improvement": improving,
        f"{prefix}_sustained_decline": declining,
        f"{prefix}_acceleration": acceleration,
    }
    reliable = {
        f"{prefix}_observed_fraction": observed_fraction,
        f"{prefix}_form_reliability": form_reliability,
        f"{prefix}_reliable_weighted_dev": form_reliability * weighted_dev,
        f"{prefix}_reliable_acceleration": form_reliability * acceleration,
        f"{prefix}_reliable_improvement": form_reliability * improving,
        f"{prefix}_reliable_decline": form_reliability * declining,
    }
    return state, reliable


def add_change_features(df):
    """add_shrinkage 결과가 붙은 DataFrame에서 변화 상태를 계산한다."""
    success_state, success_rel = _state_block(df, "success")
    middle_state, middle_rel = _state_block(df, "middle")
    return pd.DataFrame({**success_state, **middle_state,
                         **success_rel, **middle_rel}, index=df.index)


SUCCESS_STATE_COLS = [
    "change_success_weighted_dev", "change_success_direction_balance",
    "change_success_sustained_improvement", "change_success_sustained_decline",
    "change_success_acceleration",
]
MIDDLE_STATE_COLS = [
    "change_middle_weighted_dev", "change_middle_direction_balance",
    "change_middle_sustained_improvement", "change_middle_sustained_decline",
    "change_middle_acceleration",
]
SUCCESS_RELIABILITY_COLS = [
    "change_success_observed_fraction", "change_success_form_reliability",
    "change_success_reliable_weighted_dev", "change_success_reliable_acceleration",
    "change_success_reliable_improvement", "change_success_reliable_decline",
]
MIDDLE_RELIABILITY_COLS = [
    "change_middle_observed_fraction", "change_middle_form_reliability",
    "change_middle_reliable_weighted_dev", "change_middle_reliable_acceleration",
    "change_middle_reliable_improvement", "change_middle_reliable_decline",
]

CONFIGS = {
    "baseline": [],
    "success_state": SUCCESS_STATE_COLS,
    "middle_state": MIDDLE_STATE_COLS,
    "both_state": SUCCESS_STATE_COLS + MIDDLE_STATE_COLS,
    "success_reliable": SUCCESS_STATE_COLS + SUCCESS_RELIABILITY_COLS,
    "all_state": (SUCCESS_STATE_COLS + MIDDLE_STATE_COLS
                  + SUCCESS_RELIABILITY_COLS + MIDDLE_RELIABILITY_COLS),
}


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "quick"
    if mode not in ("quick", "confirm"):
        raise SystemExit("mode must be quick or confirm")
    requested = sys.argv[2:]
    unknown = [x for x in requested if x not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs: {unknown}; choices={list(CONFIGS)}")
    if mode == "quick":
        names = ["baseline"] + [x for x in requested if x != "baseline"] \
            if requested else list(CONFIGS)
        specs = LGBM_SPECS[:1]
    else:
        names = ["baseline"] + [x for x in requested if x != "baseline"]
        if len(names) == 1:
            raise SystemExit("confirm 뒤에 후보 config를 하나 이상 지정하세요")
        specs = LGBM_SPECS

    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    base_num_cols = raw_num + list(DERIVED_COLS) + shrinkage_cols()

    log(f"mode={mode} configs={names} members={len(specs)} folds={VAL_SEASONS}")
    log("baseline already contains dev_prev{1,3,5}_vs_long for success/middle")
    losses, scores = {}, {}
    os.makedirs(PRED_DIR, exist_ok=True)

    for val_season in VAL_SEASONS:
        tr_m = (seasons < val_season).to_numpy()
        va_m = (seasons == val_season).to_numpy()
        y_tr, y_va = y[tr_m], y[va_m]
        yv = y_va.to_numpy(dtype=float)
        prior = fit_prior(raw.loc[tr_m])
        sh = add_shrinkage(base_df, prior, K)
        X0 = pd.concat([base_df, sh], axis=1)
        changes = add_change_features(X0)
        X = pd.concat([X0, changes], axis=1)
        log(f"\n[fold={val_season}] train={tr_m.sum():,} val={va_m.sum():,}")

        for name in names:
            new_cols = CONFIGS[name]
            num_cols = base_num_cols + new_cols
            cols = CAT_COLS + num_cols
            ps = []
            log(f"  -- {name} total_features={len(cols)} new={len(new_cols)}")
            for i, spec0 in enumerate(specs):
                spec = dict(spec0)
                seed = spec.pop("seed")
                t = time.time()
                model = make_lgbm_model(num_cols, seed=seed, **spec)
                model.fit(X.loc[tr_m, cols], y_tr)
                p = model.predict_proba(X.loc[va_m, cols])[:, 1]
                ps.append(p)
                _, member_score, _ = bss(yv, p)
                log(f"     member={i+1}/{len(specs)} seed={seed} "
                    f"score={member_score:8.2f} time={time.time()-t:.0f}s")
                del model
                gc.collect()
            p = np.mean(ps, axis=0)
            brier, score, base = bss(yv, p)
            losses[(name, val_season)] = ((p - yv) ** 2, base)
            scores[(name, val_season)] = score
            np.savez_compressed(
                f"{PRED_DIR}/change_state_{mode}_{name}_{val_season}.npz",
                p=p, y=yv, n=len(specs), new_cols=np.asarray(new_cols))
            log(f"     >> brier={brier:.8f} ensemble={score:.2f}")
        del X, X0, changes, sh
        gc.collect()

    log("\n" + "=" * 100)
    log(f"{mode.upper()} 요약 — baseline 대비 동일 행 paired BSS 차이")
    log("=" * 100)
    for name in names:
        fold_scores = [scores[(name, f)] for f in VAL_SEASONS]
        line = f"{name:20s} folds=" + "/".join(f"{s:8.2f}" for s in fold_scores)
        line += f" mean={np.mean(fold_scores):8.2f}"
        if name != "baseline":
            gains, ses = [], []
            for f in VAL_SEASONS:
                loss0, base = losses[("baseline", f)]
                loss1, _ = losses[(name, f)]
                paired = (loss0 - loss1) / base * 100000
                gains.append(paired.mean())
                ses.append(paired.std(ddof=1) / np.sqrt(len(paired)))
            line += " gains=" + "/".join(f"{d:+7.2f}" for d in gains)
            line += f" mean_gain={np.mean(gains):+8.2f}"
            line += " SE=" + "/".join(f"{s:.2f}" for s in ses)
        log(line)

    log("\n피처 목록")
    for name in names:
        if name != "baseline":
            log(f"  {name}: {CONFIGS[name]}")


if __name__ == "__main__":
    main()
