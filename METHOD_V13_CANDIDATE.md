# v13 후보 — 공통 ABS/시대 보정 전처리

2024 월별 시간창 pooled 검증을 바탕으로 공통-regime 후보의 ABS expert 비중은
보수적으로 25%를 사용한다. 기존 v12 production의 10% 계약은 변경하지 않는다.

전체 17개 모델 재학습과 제출 경로 검사는 완료됐다. 정확한 최종 앙상블 forward
비교 전 실험 제출 계약이며, 비교·복구용 v12 계약은 `METHOD.md`다.

## 실전 제출 결과

- v13 리더보드 점수: **1038**
- 이전 확인 최고 v11: 1018
- 확인 개선폭: **+20점**
- v12는 실전 미제출이므로 v13 대비 직접적인 리더보드 개선폭은 알 수 없다.

이 결과로 공통 시대 보정과 ABS expert 25%를 포함한 v13은 현재 실전 최고 모델로
기록한다. 다만 각 변경 요소의 기여도와 v12 대비 순수 개선폭은 별도 비교가 필요하다.

## 목적

v12의 ABS 블록은 2024 전용 병렬 expert였고 다른 15개 모델의 입력을 보정하지
않았다. v13은 스트라이크 존과 관측체계 변화의 영향을 받는 rate를 모든 모델 앞에서
동일한 상대 축으로 변환한다. 정답 라벨은 반사실을 알 수 없으므로 수정하지 않는다.

## 공통 변환

학습 데이터에서 rate별 시즌 중심과 최근 3시즌 선형 외삽식을 고정한다. 평가 시즌의
중심은 이 외삽식으로 한 행씩 계산하며 다른 평가 행을 참조하지 않는다.

```text
reliability = n / (n + 50)
regime_skill = reliability * (player_rate - season_center)
```

누적 rate 10개와 최근 1·3·5경기 rate 6개를 변환하고, command gap·최근 추세·
타자-투수 gap도 변환된 값으로 다시 만든다. 모델 공통 입력에서는 대응하는 절대
rate, 절대 EB rate와 기존 gap을 제거한다.

현 시즌 success와 command도 같은 방식으로 변환한다.

```text
current_skill = current_rate - season_center
current_shrunk_skill = current_reliability * current_skill
current_deviation = current_shrunk_skill - career_regime_skill
```

## 모델 구조

모델은 직렬이 아니라 계속 병렬이다. 공통 전처리 결과를 HGB, LightGBM, 일반
CatBoost, command CatBoost, ABS CatBoost가 각자 받아 정답을 직접 학습한다.
base/CatBoost 혼합과 chase 정책은 v12를 유지하고, ABS expert는 월별 시간창 검증에
따라 25%를 사용한다.

공통 base 입력은 범주형 포함 69개, LightGBM은 현 시즌 16개를 더한 85개다.
일반 CatBoost는 90개, command CatBoost는 102개다. ABS expert는 기존처럼 68개지만
현 시즌과 최근 rate 입력을 공통 regime 표현으로 교체한다.

## 승격 조건

- 2022·2023·2024 forward fold에서 v12 대비 paired BSS 비교
- 다중 seed 평균과 표준오차 확인
- 세 시즌 중 특정 한 시즌에만 의존하지 않는지 확인
- `test_shared_regime_path.py`와 제출 경로 검사 통과
- 비교·복구용 production `bundle.pkl`은 v12 유지
- 루트 `submit.zip`은 실험 제출용 v13이며 v12 ZIP은 backups에 보존
