"""진단: 모델 선택의 '노이즈 바닥'이 얼마인지 측정.

같은 config를 seed만 바꿔 여러 번 학습했을 때 홀드아웃 점수가 얼마나 흔들리는지 본다.
이 흔들림(seed 간 표준편차)보다 작은 점수 차이는 '개선'이 아니라 노이즈다.
experiment_search에서 D(561) vs E(508) vs B(458)을 비교했는데, 이 차이가 노이즈
바닥 안이면 지금까지의 하이퍼파라미터 선택 자체가 신뢰할 수 없다는 뜻.

추가로 paired difference(같은 검증 행에 대한 예측끼리 비교)의 표준오차도 함께 계산해
"어느 정도 차이부터 실제 신호인지" 임계값을 구한다.
"""
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from experiment_search import (add_derived, DERIVED_COLS, CAT_COLS, ID,
                                TARGET, brier_skill_score)

DATA_DIR = "./open/data"
test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW_FEATURES = [c for c in test_cols if c != ID]
NUM_COLS = [c for c in RAW_FEATURES if c not in CAT_COLS] + DERIVED_COLS

train = add_derived(pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                                 usecols=RAW_FEATURES + [TARGET]))
is_val = train["season"] == 2024
X_tr, y_tr = train.loc[~is_val, CAT_COLS + NUM_COLS], train.loc[~is_val, TARGET]
X_val, y_val = train.loc[is_val, CAT_COLS + NUM_COLS], train.loc[is_val, TARGET]
y_val_np = y_val.to_numpy()
r = y_val_np.mean()
base = r * (1 - r)
print(f"val n={len(y_val_np)}  r={r:.4f}  base={base:.6f}")

D = dict(learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=30,
         l2_regularization=1.0, max_iter=1500, early_stopping=True,
         validation_fraction=0.1, n_iter_no_change=25)


def fit_predict(seed):
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ])
    pipe = Pipeline([("pre", pre),
                      ("clf", HistGradientBoostingClassifier(
                          loss="log_loss", random_state=seed, **D))])
    pipe.fit(X_tr, y_tr)
    return pipe.predict_proba(X_val)[:, 1]


SEEDS = [42, 1, 7, 2024, 12345]
preds, scores = [], []
for s in SEEDS:
    t = time.time()
    p = fit_predict(s)
    brier, score = brier_skill_score(y_val, p)
    preds.append(p)
    scores.append(score)
    print(f"seed={s:6d}  brier={brier:.6f}  score={score:7.2f}  ({time.time()-t:.0f}s)")

scores = np.array(scores)
print()
print(f"seed 간 score: mean={scores.mean():.1f}  std={scores.std(ddof=1):.1f}  "
      f"min={scores.min():.1f}  max={scores.max():.1f}  range={scores.ptp():.1f}")

# paired 차이의 표준오차 -> "몇 점 차이부터 진짜인가"
P = np.vstack(preds)
loss = (P - y_val_np) ** 2          # (n_seeds, n_val)
d = loss[0] - loss[1]               # seed42 vs seed1 의 per-sample 손실 차
se_paired = d.std(ddof=1) / np.sqrt(len(d))
print(f"paired brier diff SE = {se_paired:.6f} "
      f"-> score 환산 약 {100000*se_paired/base:.1f}점")
print(f"즉 두 모델 점수차가 약 {2*100000*se_paired/base:.0f}점 미만이면 통계적으로 구분 불가(2SE)")

# 시드 앙상블(평균)의 효과
p_ens = P.mean(axis=0)
brier_e, score_e = brier_skill_score(y_val, p_ens)
print()
print(f"[시드 {len(SEEDS)}개 평균 앙상블] brier={brier_e:.6f} score={score_e:.2f} "
      f"(단일 모델 평균 {scores.mean():.1f} 대비 +{score_e - scores.mean():.1f})")
