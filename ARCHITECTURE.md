# 현재 코드 구조

저장소에는 v12 제출물을 다시 만들고 검증하는 경로만 유지한다.

```text
open/data/*.csv
       |
       v
scripts/pipeline.py
       |
       +--> build_final.py                    HGB8 + LightGBM3
       +--> build_catboost_final.py           CatBoost2
       +--> build_catboost_command_final.py   command CatBoost2
       +--> build_abs_regime_final.py         ABS expert2
       +--> build_pitcher_policy_final.py     pitcher chase lookup
       |
       v
open/baseline_submit/model/bundle.pkl
       |
       v
open/baseline_submit/script.py --> output/submission.csv
```

## 핵심 계약

- `scripts/pipeline.py`가 학습 피처 로직의 기준이다.
- 제출 ZIP에는 공용 모듈을 넣을 수 없으므로 `script.py`가 필요한 피처 로직을
  독립적으로 포함한다.
- 두 구현의 값과 컬럼 순서는 세 개의 `test_*submission_path.py`로 검증한다.
- 번들은 모델과 함께 prior, lookup, 피처 순서, 혼합 가중치, logit shift를 저장한다.
- 추론은 평가 행 하나와 학습 시점에 고정한 값만 사용하며 다른 평가 행을 참조하지 않는다.

세부 모델 계약은 [`METHOD.md`](METHOD.md), 대회 제약은 [`CLAUDE.md`](CLAUDE.md)를
기준으로 한다.
