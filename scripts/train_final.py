"""최종 모델 학습 — HistGradientBoostingClassifier (탐색 결과 config D 채택).

experiment_search.py / _search2 / _search3 로 season 2024 홀드아웃 기준 총 10개 설정을
비교한 결과, 아래 하이퍼파라미터(config D)가 최고 점수였음:

  baseline RF                         : brier=0.248767  score=416.18
  HGB lr=.1  leaf=31  minleaf=50      : brier=0.248786  score=408.54
  HGB lr=.05 leaf=63  minleaf=50      : brier=0.248664  score=457.54
  HGB lr=.03 leaf=63  minleaf=30 (D)  : brier=0.248405  score=561.34  <- 채택
  (lr을 더 낮추거나 min_leaf를 더 줄이면 오히려 악화 -> D 부근이 국소 최적)

이 스크립트는 config D로 전체 2019~2024 데이터를 재학습해 최종 제출용 모델을 저장한다.
"""
import os
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from experiment_search import add_derived, DERIVED_COLS, CAT_COLS, ID, TARGET, brier_skill_score

DATA_DIR = "./open/data"
MODEL_OUT = "./open/baseline_submit/model/hgb.pkl"

test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW_FEATURES = [c for c in test_cols if c != ID]
NUM_COLS = [c for c in RAW_FEATURES if c not in CAT_COLS] + DERIVED_COLS

train_raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                         usecols=RAW_FEATURES + [TARGET])
train = add_derived(train_raw)
print("train:", train.shape)

# 참고용: 2024 홀드아웃 재확인 (train_baseline.py, experiment_search*.py 와 동일한 분할)
is_val = train["season"] == 2024
X_val, y_val = train.loc[is_val, CAT_COLS + NUM_COLS], train.loc[is_val, TARGET]
X_tr, y_tr = train.loc[~is_val, CAT_COLS + NUM_COLS], train.loc[~is_val, TARGET]

HGB_KWARGS = dict(
    learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=30,
    l2_regularization=1.0, max_iter=1500, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=25, random_state=42,
)


def build_pipeline():
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ])
    clf = HistGradientBoostingClassifier(loss="log_loss", **HGB_KWARGS)
    return Pipeline([("pre", pre), ("clf", clf)])


t = time.time()
val_pipe = build_pipeline()
val_pipe.fit(X_tr, y_tr)
p = val_pipe.predict_proba(X_val)[:, 1]
brier, score = brier_skill_score(y_val, p)
print(f"[재확인] 2019-2023 학습 / 2024 검증 :: brier={brier:.6f} score={score:.2f} "
      f"({time.time()-t:.1f}s)")

t = time.time()
final_pipe = build_pipeline()
final_pipe.fit(train[CAT_COLS + NUM_COLS], train[TARGET])
print(f"전체(2019-2024) 재학습 완료 :: {time.time() - t:.1f}s "
      f"n_iter={final_pipe.named_steps['clf'].n_iter_}")

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
joblib.dump(final_pipe, MODEL_OUT, compress=3)
print("저장 완료:", MODEL_OUT, "size=%.1fMB" % (os.path.getsize(MODEL_OUT) / 1e6))
