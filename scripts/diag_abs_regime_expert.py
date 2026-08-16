"""ABS 관측체계 전용 expert: 초기 적응 노이즈/지표 이동/선수 신호를 분리한다.

2024 전체 시즌을 과거 모델과 섞어 학습하지 않고, 같은 ABS 체계 안에서 앞선 달로
뒤의 달을 예측한다. 입력은 row-local 현재시즌 누적과 최근경기 기록뿐이며, 오염된
통산 rate는 사용하지 않는다. 2024 forward v10 예측을 고정 기준선으로 사용한다.
"""
import gc
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from pipeline import (
    ID, TARGET, CATBOOST_CAT_COLS, add_catboost_context, add_derived,
    add_shrinkage, add_season_command_train_features,
    add_season_success_train_features,
)

DATA_DIR = "./open/data"
PRED_DIR = "./experiments/preds"
BUNDLE_PATH = "./open/baseline_submit/model/bundle.pkl"
SEED = 4242

RATE_COLS = [
    "asof_pitcher_success_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate", "asof_batter_success_rate",
]
RECENT_COLS = [
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]
CONTEXT_NUM = [
    "game_month", "inning", "balls_before", "strikes_before", "outs_before",
    "run_total_before", "score_diff_pitcher_team", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
    "count_diff", "is_two_strike", "is_three_ball", "is_full_count", "risp",
    "platoon_match", "close_game",
]
WINDOWS = [(5, (6, 7)), (7, (8, 9))]
BLENDS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]


def log(*args):
    print(*args, flush=True)


def robust_features(x, center_mask, k=50.0):
    """현재 ABS 표본의 posterior signal과 Bernoulli noise를 명시적으로 분리."""
    out = {}
    centers = {}
    for rate in RATE_COLS:
        src = f"std_{rate}"
        sh = f"std_sh_{rate}"
        values = x.loc[center_mask, sh].to_numpy(dtype=float)
        center = float(np.nanmedian(values))
        scale = float(np.nanquantile(values, .75) - np.nanquantile(values, .25))
        scale = max(scale, 0.01)
        centers[rate] = (center, scale)
        ncol = "std_batter_n" if rate.startswith("asof_batter") else "std_pitcher_n"
        n = x[ncol].to_numpy(dtype=float)
        raw = x[src].to_numpy(dtype=float)
        eb = x[sh].to_numpy(dtype=float)
        rel = n / (n + k)
        out[f"abs_centered_{rate}"] = (eb - center) / scale
        out[f"abs_signal_{rate}"] = rel * (eb - center) / scale
        out[f"abs_noise_{rate}"] = np.sqrt(
            np.clip(raw * (1.0 - raw) / np.maximum(n, 1.0), 0.0, None))
    # 지표 이동과 진짜 제구를 모델이 별도 축으로 볼 수 있게 합성 좌표도 제공한다.
    out["abs_command_risk"] = (
        out["abs_centered_asof_pitcher_middle_rate"]
        + out["abs_centered_asof_pitcher_reverse_rate"]
        + out["abs_centered_asof_pitcher_ball_rate"]
        - out["abs_centered_asof_pitcher_strike_rate"]
    )
    out["abs_command_signal"] = (
        out["abs_signal_asof_pitcher_middle_rate"]
        + out["abs_signal_asof_pitcher_reverse_rate"]
        + out["abs_signal_asof_pitcher_ball_rate"]
        - out["abs_signal_asof_pitcher_strike_rate"]
    )
    return pd.DataFrame(out, index=x.index).astype("float32"), centers


def make_model(seed):
    return CatBoostClassifier(
        loss_function="Logloss", eval_metric="BrierScore",
        iterations=350, depth=7, learning_rate=0.05, l2_leaf_reg=15.0,
        random_seed=seed, random_strength=1.0,
        bootstrap_type="Bayesian", bagging_temperature=1.0,
        one_hot_max_size=12, max_ctr_complexity=2,
        boosting_type="Plain", thread_count=4,
        allow_writing_files=False, verbose=100,
    )


def paired(y, p0, p1):
    base = y.mean() * (1 - y.mean())
    d = ((p0 - y) ** 2 - (p1 - y) ** 2) / base * 100000
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main():
    bundle = joblib.load(BUNDLE_PATH)
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=raw_features + [TARGET])
    y = raw[TARGET]
    base = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    x0 = pd.concat([base, add_shrinkage(base, bundle["prior"], bundle["k"])], axis=1)
    train_target = pd.concat([x0, y.rename(TARGET)], axis=1)
    success = add_season_success_train_features(train_target, bundle["prior"], bundle["k"])
    command = add_season_command_train_features(train_target, bundle["prior"], bundle["k"])
    x = pd.concat([x0, success, command, add_catboost_context(raw)], axis=1)

    mask2024 = raw["season"].eq(2024).to_numpy()
    mature_center = mask2024 & raw["game_month"].between(5, 9).to_numpy()
    robust, centers = robust_features(x, mature_center, bundle["k"])
    x = pd.concat([x, robust], axis=1)
    current_cols = [f"std_{r}" for r in RATE_COLS] + [f"std_sh_{r}" for r in RATE_COLS]
    robust_cols = list(robust.columns)
    features = list(CATBOOST_CAT_COLS) + CONTEXT_NUM + current_cols + RECENT_COLS + robust_cols

    with np.load(f"{PRED_DIR}/catboost_command_confirm_2024.npz") as z:
        p_base_all = z["p_all4_r50"].astype(float)
        y_cache = z["y"].astype(float)
    if not np.array_equal(y.loc[mask2024].to_numpy(dtype=float), y_cache):
        raise RuntimeError("2024 cache row order mismatch")
    base_by_index = pd.Series(p_base_all, index=raw.index[mask2024])

    os.makedirs(PRED_DIR, exist_ok=True)
    rows = []
    saved = {"centers": np.array([centers[r] for r in RATE_COLS], dtype=float),
             "rate_cols": np.asarray(RATE_COLS)}
    for train_end, val_months in WINDOWS:
        val = mask2024 & raw["game_month"].isin(val_months).to_numpy()
        variants = {
            "all_abs": mask2024 & (raw["game_month"].to_numpy() <= train_end),
            "mature_only": mask2024 & raw["game_month"].between(5, train_end).to_numpy(),
        }
        for name, tr in variants.items():
            model = make_model(SEED)
            # 3~4월은 적응 노이즈로 가정하되 all_abs 비교에서는 완전히 버리지 않는다.
            weights = np.ones(int(tr.sum()), dtype=float)
            if name == "all_abs":
                tm = raw.loc[tr, "game_month"].to_numpy()
                weights = np.select([tm <= 3, tm == 4], [.20, .45], default=1.0)
            started = time.time()
            model.fit(x.loc[tr, features], y.loc[tr], cat_features=CATBOOST_CAT_COLS,
                      sample_weight=weights)
            pe = model.predict_proba(x.loc[val, features])[:, 1]
            yv = y.loc[val].to_numpy(dtype=float)
            p0 = base_by_index.loc[raw.index[val]].to_numpy(dtype=float)
            g, se = paired(yv, p0, pe)
            log(f"window <=M{train_end} -> {val_months}, {name}: rows={tr.sum():,} "
                f"expert gain={g:+.2f} SE={se:.2f}")
            saved[f"y_{train_end}_{name}"] = yv
            saved[f"p0_{train_end}_{name}"] = p0
            saved[f"pe_{train_end}_{name}"] = pe
            for w in BLENDS:
                p = (1 - w) * p0 + w * pe
                gain, blend_se = paired(yv, p0, p)
                rows.append(dict(train_end=train_end, val_months=str(val_months),
                                 variant=name, weight=w, gain=gain, se=blend_se,
                                 n_train=int(tr.sum()), n_val=int(val.sum())))
                log(f"  blend {w:.2f}: {gain:+.2f} (SE {blend_se:.2f})")
            del model
            gc.collect()
    out = pd.DataFrame(rows)
    out.to_csv(f"{PRED_DIR}/abs_regime_expert_summary.csv", index=False)
    np.savez_compressed(f"{PRED_DIR}/abs_regime_expert_windows.npz", **saved)
    log("\nMEAN GAIN BY VARIANT/WEIGHT")
    print(out.groupby(["variant", "weight"]).gain.mean().unstack().round(2).to_string())
    log("saved summary and predictions")


if __name__ == "__main__":
    main()
