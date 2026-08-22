# LG Aimers — KBO 제구 성공 확률 예측

이 문서는 프로젝트의 목표, 데이터 계약, 제출 규격과 현재 상태의 단일 진실 소스다.

## 1. 목표와 평가

- 2019~2024 투구 사전 정보로 각 투구의 `control_success=1` 확률을 예측한다.
- 현재 투구 결과, 실제 코스·구종, Trackman 실측값 등 사후 정보는 사용하지 않는다.
- 평가지표는 Brier Skill Score다.

```text
Brier = mean((p - y)^2)
Base  = mean(y) * (1 - mean(y))
Score = max(0, 100000 * (1 - Brier / Base))
```

## 2. 데이터 계약

- `open/data/train.csv`: 2019~2024 학습 데이터, `control_success` 포함
- `open/data/test.csv`: 로컬에는 형식 확인용 샘플, 서버에서 실제 평가 데이터로 교체
- `open/data/sample_submission.csv`: `row_id`, `control_success`
- `open/data/trackman_history.csv`: 공식 제공 보조 데이터이며 현재 방식은 사용하지 않음
- 학습 피처 목록은 항상 `test.csv` 컬럼을 기준으로 정한다.
- `asof_*` 컬럼은 투구 직전까지 계산된 공식 사전 정보이므로 사용할 수 있다.

## 3. 리크 및 대회 제약

- 평가 데이터의 다른 행, 전체 분포, 행 순서로 만든 통계는 금지한다.
- 모든 평가 피처는 현재 행과 학습 데이터에서 고정한 상수·lookup만으로 계산한다.
- 외부 데이터와 원격 API 모델은 사용하지 않는다.
- 현재 투구 이후 확정되는 정보는 입력에 포함하지 않는다.

## 4. 기존 실험 성과와 이식 상태

- 기존 실전 제출물은 `v13_shared_regime_chase`였다.
- 누적·최근·현 시즌 rate를 시즌 중심 대비 상대값으로 바꿔 모든 모델이 공유한다.
- HGB8 + LightGBM3 + CatBoost2 + command CatBoost2 + ABS CatBoost2를 사용한다.
- ABS expert 비중은 25%, 투수별 chase policy 최대 비중은 20%다.
- v13의 `logit_shift=-0.04985414`다.
- v13 실전 제출 점수는 **1038**로, 이전 확인 최고 v11의 1018보다 20점 높다.
- v12는 실전 미제출이므로 v13과 v12의 직접 리더보드 비교값은 없다.
- 이 이식용 작업본은 코드·데이터·실험 기록만 보관하며, 학습된 번들과
  기존 `submit.zip`은 포함하지 않는다. 새 서버에서 재생성한다.

v13 후보 계약은 `METHOD_V13_CANDIDATE.md`, v12 기준 계약은 `METHOD.md`다.

## 5. 저장소 구조

```text
AGENTS.md, CLAUDE.md, ARCHITECTURE.md, METHOD.md, README.md
scripts/                  v12 호환 + v13 학습·검증 코드
open/data/                공식 데이터(.gitignore 제외)
open/baseline_submit/     script.py, requirements.txt, model/(새 서버에서 생성)
backups/                  문서형 과거 실험 기록
```

기존 v13 제출 ZIP의 SHA-256 기록은
`66d4b618a9a0793667e2e932a67c6b92aa8e21fbc52f2be2d9b6cefe0380b5e8`다.
기존 내부 번들 SHA-256 기록은
`59beb89219190bd8cf2aad8b9dff1fcc76cb0739907de18194641260714d598d`다.
해당 바이너리 산출물은 이 작업본에 포함되지 않는다.

## 6. 학습과 검증

- 모델 학습·재학습·교차검증·대규모 실험은 로컬에서 실행하지 않는다.
- 사용자가 Google Colab에서 실행할 수 있는 완결된 코드와 순서를 제공한다.
- 로컬에서는 모델을 fit하지 않는 문법·정적·ZIP 구조 검증만 허용한다.
- 모델 비교는 forward-chaining 시즌 폴드, 다중 시드, paired 표준오차와 BSS로 한다.
- 공용 로직은 `scripts/pipeline.py`, 제출 추론은
  `open/baseline_submit/script.py`이며 두 경로의 동등성을 유지한다.

## 7. 제출 규격

`submit.zip` 최상위 구조는 다음 세 항목만 허용한다.

```text
model/
script.py
requirements.txt
```

서버는 `data/`와 `output/`을 추가한다. `script.py`는 `data/test.csv`를 읽어
`output/submission.csv`를 생성해야 한다. 제한은 Python 3.11, 6 vCPU, RAM 28GB,
설치 10분, 추론 10분, ZIP 10GB 이하이다.
