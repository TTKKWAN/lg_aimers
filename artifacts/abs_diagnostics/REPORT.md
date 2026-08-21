# ABS regime 진단 결과

## 결론

- 2023→2024에서 `control_success`는 0.49996→0.48610으로 1.385%p 하락했다.
- 12개 볼-스트라이크 상태 모두 성공률이 하락해 특정 count 구성만의 변화로 보기 어렵다.
- 선수 지문을 차단한 pitcher-season 그룹 검증에서 최근 1·3·5경기
  success/middle 피처의 2024 구분 AUC는 `0.7369 ± 0.0429`였다.
- 같은 검증에서 통산 pitcher rate는 AUC `0.5964`, batter rate는 `0.5690`으로
  약한 변화였고, pitch mix는 `0.4643`으로 일반화되는 regime 신호가 없었다.
- 투수-시즌/월 클러스터의 silhouette는 각각 0.204/0.165로 약하다. 클러스터 ID를
  바로 모델 입력이나 gating에 쓰지 않는다.
- 2024 내부 변화점은 최근 success가 5월, 최근 middle이 6~7월에 집중됐다.
  3~4월만 적응기로 보는 현재 가중치는 별도 ablation이 필요하다.

## 주의: 행 단위 adversarial AUC

행 무작위 분할에서는 민감 피처 AUC 0.963, 전체 피처 AUC 0.997이 나왔다. 하지만
같은 선수의 반복 누적률이 학습/검증에 함께 들어가 선수 지문을 외울 수 있으므로 이
수치를 regime 크기로 해석하지 않는다. 최종 판단은 선수 그룹 분할 결과를 우선한다.

## 권장 피처 라우팅

### Stable backbone: 2019~2024

- 경기 상황, count, out, 주자, inning, score, LI
- 선수/팀 ID와 손
- workload `n`, `log1p(n)`, reliability
- pitch mix 및 관련 표본 수
- 아래 ABS 민감 rate와 그 EB/편차/상호작용 파생값은 차단

### ABS expert: 2024

- pitcher/batter success
- pitcher strike, ball, middle, reverse
- 최근 1·3·5경기 success/middle
- current-season success/command
- 위 rate의 EB, reliability, noise, context interaction

## 순차 모델 후보

1. **S1 stable ablation**: 현재 base 계열에서 민감 원본·파생 rate만 제거한다.
2. **S2 season weighting**: S1에 과거 시즌 감쇠 가중치를 추가한다.
3. **S3 ABS expert**: 2024 rate-aware expert를 병렬로 다시 학습한다.
4. **S4 maturity ablation**: 현행 월 가중치와 5월/7월 이후 강화 후보를 비교한다.
5. **S5 blend grid**: ABS expert 10/20/30/40%를 paired BSS로 비교한다.
6. 클러스터 기반 gating은 현재 근거 부족으로 보류한다.

## 후속 모델 실험 결과

민감 원본·EB·최근 편차·gap 파생값을 전부 차단한 56-feature stable LightGBM을
91-feature 기준 모델과 동일한 forward fold·3 seed로 비교했다.

| fold | stable gain | paired SE |
|---:|---:|---:|
| 2022 | -288.98 | 27.77 |
| 2023 | -699.03 | 28.24 |
| 2024 | -796.75 | 35.74 |

따라서 민감 rate의 완전 삭제와 stable+ABS 복구 구조는 폐기한다. 2024 월별 ABS
expert를 stable과 최대 80%까지 혼합해도 원래 backbone보다 뒤졌다.

반면 절대 rate를 시즌 상대값으로 교체한 era-normalized backbone과 ABS expert를
결합하면 6~7월 및 8~9월 두 검증창에서 모두 개선됐다. 두 창 152,809행 pooled 기준
25% 혼합은 `+138.09 ± 15.75 BSS`, 수치상 최적은 38%였다. 교차 실험 조합의 낙관을
피해 v13 후보는 보수적인 ABS 25%를 사용한다. v12 production 10%는 유지한다.

## 전체 후보 재학습

독립 스테이징 경로에서 전체 17개 모델을 재학습했다.

- HGB 8 + LightGBM 3 + CatBoost 2 + command CatBoost 2 + ABS CatBoost 2
- 공통 base 69개, LightGBM 85개, 일반 CatBoost 90개, command 102개
- 제출 union 129개
- ABS expert weight 25%
- 2024 mixed natural mean `0.48681939`
- logit shift `-0.04985414`
- chase lookup 투수 386명
- 최종 번들 SHA-256 `59beb89219190bd8cf2aad8b9dff1fcc76cb0739907de18194641260714d598d`

공통-regime 피처 동등성, 전체 CatBoost/ABS 혼합, chase single/shuffle/unseen 검사를
통과했다. production `bundle.pkl`과 `submit.zip`은 교체하지 않았다.

## 핵심 산출물

- `feature_shift_2023_vs_2024.csv`
- `count_conditioned_target_2023_vs_2024.csv`
- `adversarial_scores_2023_vs_2024.csv`
- `adversarial_grouped_player_profiles.csv`
- `change_points_within_2024.csv`
- `pitcher_season_clusters.csv`
- `pitcher_month_clusters.csv`
