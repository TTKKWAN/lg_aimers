# 제출 상태

마지막 갱신: 2026-08-10

## 현재 제출 후보

- 파일: 루트 `submit.zip`
- 버전: **v10 — 현 시즌 all4 command CatBoost 50% 추가**
- 리더보드 상태: **1010점 확인 — v9과 사실상 동일, 개선 없음**
- SHA-256: `095ff3b7ce401c296600b7a9eb4c71f437d158d1cafb6fd6242add9c0fd89fb8`
- ZIP 최상위 구조: `model/bundle.pkl`, `script.py`, `requirements.txt`
- 모델: v8 HGB8/current-season LGBM3 + 기존 CatBoost2 + command CatBoost2
- 혼합: **v8 40% + [기존 CatBoost2 50% + command CatBoost2 50%] 60%**
- command 입력: 기존 112개 + 현 시즌 strike/ball/middle/reverse 12개 = 124개
- 재중심화: `recenter_f=0.5`, `logit_shift=-0.05126671`
- 전체 학습 2024 혼합 자연평균: `0.48751748`
- f=0.5 목표 평균: `0.47484808`
- 번들: 49.8MB, 245,789행 모사 피처+추론: 102.64초

CatBoost 2시드 평균을 v8에 섞고 fold마다 production과 같은 재중심화를 적용한
v8 대비 paired gain은 다음과 같다. 75%에서는 최신 폴드가 꺾여 60%를 채택했다.

| CatBoost 비중 | 2022 | 2023 | 2024 | 평균 |
|---:|---:|---:|---:|---:|
| 30% | +47.47 | +164.17 | +37.76 | +83.13 |
| 50% | +65.72 | +259.30 | +49.55 | +124.86 |
| **60%** | **+70.84** | **+302.57** | **+51.44** | **+141.62** |
| 75% | +73.49 | +362.11 | +49.26 | +161.62 |

논문 기반 동적 잠재상태는 투수-only `-169.23/-83.26/-24.04`, 투수+타자
`-138.30/-89.78/-87.91`로 세 폴드 모두 손해여서 CatBoost와 결합하지 않았다.

v9 대비 command family의 forward paired gain은 다음과 같다.

| command 교체 비중 | 2022 | 2023 | 2024 |
|---:|---:|---:|---:|
| **50% all4** | **+17.80 (SE 2.77)** | **+18.12 (2.31)** | **+6.93 (2.30)** |
| 100% all4 | +25.80 (5.55) | +29.86 (4.62) | +7.09 (4.60) |

100%는 평균 이득이 크지만 최신 폴드 이득이 거의 같고 SE가 두 배라 50%를 채택했다.

## 확인 최고 및 안전 제출물

- 확인 최고: **v10, 리더보드 1010점** (v9과 사실상 동률)
- 안전 제출 파일: `backups/submit_v9_1000plus_backup.zip`
- SHA-256: `8ea71f1aa8c27d94e81fcfb552d43f89c088ed018d64515755306d195561eeb9`
- 구성: v8(HGB25% + current-season LGBM75%) 40% + CatBoost2 60%
- 이전 확인 최고: v7 938점 (`backups/submit_v7_938_backup.zip`)

현재 루트 `submit.zip`은 1010점이 확인된 v10이며 v9 안전 제출 파일과 다르다.
command 추가의 실측 개선은 없었던 것으로 판단하고, 필요하면 v9 백업을 복구 기준으로
사용한다.

## 최근 개인 pressure agent 실험

`pitcher_id`를 선수 dossier lookup 키로만 사용해 투수별 풀카운트·3볼·득점권·
high-LI·접전 등 압박 반응을 strict-past 시즌에서 계산했다. 선수별 반응은 강하게
축소하고 raw 숫자형 `pitcher_id`는 모델 입력에서 제거했다.

전체과거 고정 dossier의 가장 좋은 후보:

| 검증 시즌 | v8 대비 BSS gain |
|---:|---:|
| 2022 | +34.92 |
| 2023 | -27.28 |
| 2024 | +24.36 |

직전 1시즌만 사용하고 k=500으로 축소한 refinement:

| 검증 시즌 | v8 대비 BSS gain | paired SE |
|---:|---:|---:|
| 2022 | +6.59 | 9.30 |
| 2023 | -5.43 | 8.76 |
| 2024 | +18.92 | 7.72 |

두 방식 모두 forward 세 폴드 방향이 일치하지 않아 **미채택**이다. pressure agent는
현재 `submit.zip`에 포함하지 않았다.

## 검증 상태

- `scripts/test_submission_path.py` 통과
- HGB/LightGBM family 혼합 수치 검증 통과
- v7 91/107 컬럼 및 season lookup 계약 통과
- v10 command 12컬럼 학습/제출 동등성, batch/single·shuffle·unseen 검증 통과
- batch/single, shuffle, unseen fallback 검증 통과
- legacy/context/era 하위호환 검증 통과
- 실제 ZIP 재해제 clean-room 5행 추론 통과
- CatBoost family 50:50·v8 40%/family 60% 혼합·재중심화 직접 대조 통과
- `catboost==1.2.8` 로컬 직렬화/역직렬화 통과
- 245,789행 모사 추론 102.64초
- ZIP 무결성 검사 통과
- ZIP 최상위 3파일 구조 확인

상세 실험 근거는 `docs/EXPERIMENTS.md`, 현재 모델 계약은 `CLAUDE.md`를 기준으로
한다.
