"""1차 탐색에서 B(lr=.05, leaf=63, l2=1.0, IDs 포함)가 최고 -> 그 주변을 추가 탐색."""
import time

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from experiment_search import add_derived, DERIVED_COLS, CAT_COLS, ID as ID_COL, TARGET, brier_skill_score

DATA_DIR = "./open/data"
test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW_FEATURES = [c for c in test_cols if c != ID_COL]

train_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                         usecols=RAW_FEATURES + [TARGET])
train = add_derived(train_raw)
is_val = train["season"] == 2024
train_split, val_split = train.loc[~is_val], train.loc[is_val]
y_train, y_val = train_split[TARGET], val_split[TARGET]

NUM_COLS = [c for c in RAW_FEATURES if c not in CAT_COLS] + DERIVED_COLS


def build_pipeline(**hgb_kwargs):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ])
    clf = HistGradientBoostingClassifier(loss="log_loss", random_state=42, **hgb_kwargs)
    return Pipeline([("pre", pre), ("clf", clf)])


configs = [
    dict(name="D_lr.03_leaf63_minleaf30", hgb=dict(
        learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=30,
        l2_regularization=1.0, max_iter=1500, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=25)),
    dict(name="E_lr.05_leaf127_minleaf50", hgb=dict(
        learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=50,
        l2_regularization=1.0, max_iter=1000, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=25)),
    dict(name="F_lr.05_leaf63_minleaf50_l2.3", hgb=dict(
        learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=50,
        l2_regularization=3.0, max_iter=1000, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=25)),
    dict(name="G_lr.05_leaf63_minleaf100_l2.1", hgb=dict(
        learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=100,
        l2_regularization=1.0, max_iter=1000, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=25)),
]

results = []
for cfg in configs:
    t = time.time()
    pipe = build_pipeline(**cfg["hgb"])
    pipe.fit(train_split[CAT_COLS + NUM_COLS], y_train)
    dt = time.time() - t
    n_iter = pipe.named_steps["clf"].n_iter_
    p = pipe.predict_proba(val_split[CAT_COLS + NUM_COLS])[:, 1]
    brier, score = brier_skill_score(y_val, p)
    results.append((cfg["name"], dt, n_iter, brier, score))
    print(f"{cfg['name']:28s} fit={dt:6.1f}s n_iter={n_iter:4d} "
          f"brier={brier:.6f} score={score:.2f}")

print()
print("=== 요약 (baseline RF: score=416.18 | round1 best B: score=457.54) ===")
for name, dt, n_iter, brier, score in sorted(results, key=lambda r: -r[4]):
    print(f"{name:28s} score={score:8.2f}  brier={brier:.6f}  fit={dt:.1f}s  n_iter={n_iter}")
