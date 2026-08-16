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

## 4. 현재 v12 방식

- 공통 fixed-EB 피처로 HGB 8개와 LightGBM 3개를 학습하고 25:75로 혼합한다.
- CatBoost 2개와 현 시즌 command CatBoost 2개를 반반 혼합한 뒤, base 40%와
  CatBoost family 60%를 혼합한다.
- 2024 ABS regime CatBoost 2개 평균을 10% 혼합한다.
- 2스트라이크·비풀카운트에서 같은 투수의 2024 chase 정책을 신뢰도에 따라 최대
  20% 적용한다. 투수 간 정보는 공유하지 않는다.
- 학습 데이터에서 산출한 `logit_shift=-0.05087341`을 행별로 적용한다.

전체 피처와 수식은 `METHOD.md`가 기준이다.

## 5. 저장소 구조

```text
AGENTS.md, CLAUDE.md, ARCHITECTURE.md, METHOD.md, README.md
scripts/                  현재 v12 학습·검증 코드
open/data/                공식 데이터(.gitignore 제외)
open/baseline_submit/     script.py, requirements.txt, model/bundle.pkl
backups/                  로컬 백업(.gitignore 제외)
submit.zip                현재 제출물(.gitignore 제외)
```

현재 모델 번들 메타 버전은 `v12_pitcher_chase_policy_20`이다. 현재 제출 ZIP의
SHA-256은 `d455adc05108e56eef1d128904270cecdb60ff933124b17e87cedefb1807e47d`다.

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
