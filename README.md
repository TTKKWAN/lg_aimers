# LG Aimers KBO 제구 성공 확률 예측

2019~2024년 투구 사전 정보만 사용해 `control_success` 확률을 예측하는 코드입니다.
현재 루트 `submit.zip`은 재학습·경로 검사를 통과한 v13 공통 ABS/시대 보정 실험
제출물입니다. 비교·복구용 `open/baseline_submit/model/bundle.pkl`은 v12를 유지합니다.
v13의 실전 제출 점수는 **1038**로, 이전 확인 최고 v11(1018)보다 20점 높습니다.
v12는 실전 미제출이라 v13과 직접 비교할 리더보드 점수는 없습니다.

## v12 비교 모델

- HGB 8개와 LightGBM 3개 앙상블
- 기존 CatBoost 2개와 현 시즌 command CatBoost 2개
- 2024 ABS regime CatBoost 2개를 10% 혼합
- 2스트라이크·비풀카운트에서 투수별 chase policy를 최대 20% 적용
- 학습 데이터에서 고정한 logit shift로 최종 확률 재중심화

## v13 공통 regime 후보

- 누적·최근·현 시즌 rate의 절대값을 모델 공통 입력에서 제거
- 학습 시즌별 리그 중심을 뺀 신뢰도 보정 상대능력으로 교체
- HGB, LightGBM, 일반/command/ABS CatBoost가 같은 보정 축을 사용
- 정답 `control_success`는 임의로 수정하지 않음
- 2024 ABS expert를 25% 혼합하고 chase policy를 최대 20% 적용

자세한 피처와 혼합식은 [`METHOD.md`](METHOD.md)를 참고하세요.

## 구조

```text
scripts/
  pipeline.py                          공용 피처·모델 로직
  build_final.py                       HGB8 + LightGBM3 번들 생성
  build_catboost_final.py              CatBoost2 추가
  build_catboost_command_final.py      command CatBoost2 추가
  build_abs_regime_final.py            ABS expert2 추가
  build_pitcher_policy_final.py        투수별 chase policy 추가
  test_submission_path.py              기본 학습/추론 피처 동등성 검사
  test_catboost_submission_path.py     전체 혼합·피처 계약 검사
  test_pitcher_policy_submission_path.py  chase policy 검사
  test_shared_regime_path.py              공통 regime 학습/추론 동등성 검사
  benchmark_submission.py              평가 규모 추론 시간 측정
open/baseline_submit/
  script.py                            제출 서버 추론 진입점
  requirements.txt                    제출 의존성
```

`open/data/*.csv`, 모델 번들, 제출 ZIP, 가상환경과 백업은 `.gitignore`로 제외됩니다.

## 실행 위치

```bash
python3 scripts/build_final.py 50 te0 8 0.5 3 era1 context0 season1 0.75
python3 scripts/build_catboost_final.py
python3 scripts/build_catboost_command_final.py
cp open/baseline_submit/model/bundle_v13_command_candidate.pkl open/baseline_submit/model/bundle.pkl
python3 scripts/build_abs_regime_final.py
cp open/baseline_submit/model/bundle_v13_abs_candidate.pkl open/baseline_submit/model/bundle.pkl
python3 scripts/build_pitcher_policy_final.py
```

최종 후보는 `bundle_v13_shared_regime_candidate.pkl`입니다. production 교체 전 아래
검사와 forward 비교를 수행합니다.

```bash
python3 scripts/test_submission_path.py
python3 scripts/test_shared_regime_path.py
LGAIMERS_BUNDLE=open/baseline_submit/model/bundle_v13_shared_regime_candidate.pkl python3 scripts/test_catboost_submission_path.py
LGAIMERS_BUNDLE=open/baseline_submit/model/bundle_v13_shared_regime_candidate.pkl python3 scripts/test_pitcher_policy_submission_path.py
python3 scripts/benchmark_submission.py
```

제출 ZIP의 최상위에는 `model/`, `script.py`, `requirements.txt`만 있어야 합니다.
