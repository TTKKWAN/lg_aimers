"""HistGradientBoosting 후보 비교 — season 2024 홀드아웃 기준.

베이스라인(RandomForest, max_depth=10, min_samples_leaf=200)의 한계:
  - 얕은 트리(100개) + bagging -> 약한 신호 학습에 취약
  - 결측치를 median으로 대치 -> "cold-start(표본 0)"라는 정보 자체가 사라짐
  - 스트라이크존 관련 파생 피처 없음

여기서는:
  1. HistGradientBoostingClassifier로 교체 (scikit-learn 내장, 서버에 이미 설치되어
     있어 requirements.txt에 새 패키지를 추가할 필요가 없음 -> 오프라인 설치 위험 없음).
  2. 결측치를 대치하지 않고 그대로 통과시켜 HGB의 네이티브 NaN 분기를 활용.
  3. 카운트/주자/투수-타자 조합 파생 피처 추가 (행 단위 계산, 규정 §5 준수).
  4. pitcher_id/batter_id를 포함/제외한 버전을 비교 (raw ID가 노이즈인지 신호인지 검증).
"""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

DATA_DIR = "./open/data"
ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]
ID_COLS = ["pitcher_id", "batter_id"]


def add_derived(df):
    df = df.copy()
    df["count_diff"] = df["strikes_before"] - df["balls_before"]
    df["is_two_strike"] = (df["strikes_before"] == 2).astype(int)
    df["is_three_ball"] = (df["balls_before"] == 3).astype(int)
    df["is_full_count"] = df["is_two_strike"] & df["is_three_ball"]
    df["risp"] = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype(int)
    df["platoon_match"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
    df["pitcher_command_gap"] = df["asof_pitcher_success_rate"] - df["asof_pitcher_middle_rate"]
    df["pitcher_recent_trend"] = (df["asof_pitcher_prev1_game_success_rate"]
                                   - df["asof_pitcher_prev5_game_success_rate"])
    df["batter_pitcher_gap"] = df["asof_batter_success_rate"] - df["asof_pitcher_success_rate"]
    df["close_game"] = (df["score_diff_pitcher_team"].abs() <= 1).astype(int)
    return df


DERIVED_COLS = ["count_diff", "is_two_strike", "is_three_ball", "is_full_count",
                 "risp", "platoon_match", "pitcher_command_gap",
                 "pitcher_recent_trend", "batter_pitcher_gap", "close_game"]

test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW_FEATURES = [c for c in test_cols if c != ID]

train_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                         usecols=RAW_FEATURES + [TARGET])
print("raw:", train_raw.shape)
train = add_derived(train_raw)
print("with derived:", train.shape)

is_val = train["season"] == 2024
train_split, val_split = train.loc[~is_val], train.loc[is_val]
y_train, y_val = train_split[TARGET], val_split[TARGET]
print("train:", len(train_split), "val:", len(val_split))


def brier_skill_score(y_true, p):
    r = y_true.mean()
    brier = ((p - y_true) ** 2).mean()
    base = r * (1 - r)
    return brier, max(0, 100000 * (1 - brier / base))


def build_pipeline(num_cols, **hgb_kwargs):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])
    clf = HistGradientBoostingClassifier(
        loss="log_loss", random_state=42, **hgb_kwargs)
    return Pipeline([("pre", pre), ("clf", clf)])


NUM_WITH_IDS = [c for c in RAW_FEATURES if c not in CAT_COLS] + DERIVED_COLS
NUM_NO_IDS = [c for c in NUM_WITH_IDS if c not in ID_COLS]

configs = [
    dict(name="A_lr.1_leaf31_withIDs", num_cols=NUM_WITH_IDS,
         hgb=dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=50,
                   l2_regularization=0.0, max_iter=600, early_stopping=True,
                   validation_fraction=0.1, n_iter_no_change=20)),
    dict(name="B_lr.05_leaf63_withIDs", num_cols=NUM_WITH_IDS,
         hgb=dict(learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
                   l2_regularization=1.0, max_iter=800, early_stopping=True,
                   validation_fraction=0.1, n_iter_no_change=20)),
    dict(name="C_lr.1_leaf31_noIDs", num_cols=NUM_NO_IDS,
         hgb=dict(learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=50,
                   l2_regularization=0.0, max_iter=600, early_stopping=True,
                   validation_fraction=0.1, n_iter_no_change=20)),
]

results = []
for cfg in configs:
    t = time.time()
    pipe = build_pipeline(cfg["num_cols"], **cfg["hgb"])
    pipe.fit(train_split[CAT_COLS + cfg["num_cols"]], y_train)
    dt = time.time() - t
    n_iter = pipe.named_steps["clf"].n_iter_
    p = pipe.predict_proba(val_split[CAT_COLS + cfg["num_cols"]])[:, 1]
    brier, score = brier_skill_score(y_val, p)
    results.append((cfg["name"], dt, n_iter, brier, score))
    print(f"{cfg['name']:24s} fit={dt:6.1f}s n_iter={n_iter:4d} "
          f"brier={brier:.6f} score={score:.2f}")

print()
print("=== 요약 (baseline RF: brier=0.248767, score=416.18) ===")
for name, dt, n_iter, brier, score in results:
    print(f"{name:24s} score={score:8.2f}  brier={brier:.6f}  fit={dt:.1f}s  n_iter={n_iter}")
