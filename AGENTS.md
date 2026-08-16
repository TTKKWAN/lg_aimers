# Codex 프로젝트 시작 지침 — LG Aimers KBO 제구 성공 확률 예측

이 파일은 Codex가 이 저장소에서 작업할 때 가장 먼저 따르는 운영 지침이다.
상세한 프로젝트 사실의 단일 진실 소스는 루트의 [`CLAUDE.md`](CLAUDE.md)다.

## 세션 시작 규약

1. 작업에 착수하기 전에 반드시 `CLAUDE.md`를 처음부터 읽는다. 목표, 데이터 계약,
   리크 금지, 제출 규격, 현재 최고 모델과 실험 결론은 그 문서가 기준이다.
2. 코드 구조나 변경 영향이 필요한 작업이면 `ARCHITECTURE.md`를 읽고, 모델링 판단이면
   `METHOD.md`, `docs/EXPERIMENTS.md` 인덱스와 `docs/experiments/`의 최신 날짜 파일을
   확인한다.
3. 현재 상태를 짧게 요약하면: 2019~2024 투구 사전정보로 `control_success` 확률을
   예측하는 Brier Skill Score 대회다. 현재 `submit.zip`은 **미제출 v12 투수별
   chase-policy 후보**다. 리더보드 1018의 v11 ABS expert 위에, 투수끼리 공유하지
   않는 2024 개인 2스트라이크 정책을 최대 20% 신뢰도 혼합했다. v11은
   `backups/submit_v11_1018_backup.zip`에 보존돼 있다. 2024 ABS 현재시즌 관측을 지표 중심·신뢰 신호·표본오차로
   분해한 CatBoost2를 v10에 10% 혼합했다. 2024 월순서 검증 gain은
   +12.00/+12.32(SE 5.66/4.73), 245,789행 모사 추론은 107.66초다. 확인 최고인
   v10 대비 실측 개선은 +8에 그쳤다. **v10 리더보드 1010점**은
   `backups/submit_v10_1010_backup.zip`에 보존돼 있다.

## 학습 실행 규약 — 항상 적용

- **Codex는 이 저장소에서 모델 학습·재학습·교차검증·대규모 실험을 로컬로 실행하지
  않는다.** 백그라운드 실행도 시작하지 않는다.
- 학습이나 실험이 필요하면 사용자가 **Google Colab에서 직접 실행할 수 있는 완결된
  코드와 셀 실행 순서·명령어**를 제공한다. 필요한 입력 경로, 출력 아티팩트, 예상 로그,
  재개 방법까지 코드에 포함하고 사용자가 결과를 전달할 때까지 기다린다.
- Codex가 로컬에서 실행해도 되는 것은 코드 문법 검사, 정적 분석, 소규모 샘플을 이용한
  피처 동등성·제출 경로·ZIP 구조 검사처럼 **모델을 fit하지 않는 가벼운 검증**뿐이다.
- 기존 모델을 로드한 추론 검증도 장시간·대규모라면 실행하지 말고 Colab용 명령으로
  제공한다.
- 사용자가 해당 요청에서 명시적으로 “로컬에서 학습/실행해 달라”고 지시한 경우에만
  예외로 하며, 과거의 포괄적 요청이나 자동 계속 실행은 예외 근거로 삼지 않는다.

## 변경 안전 규약

- 모든 새 피처는 평가 `test.csv`의 **행 하나만**과 학습 데이터에서 미리 계산해 고정한
  상수만으로 계산되어야 한다. test 행 간 통계, 전체 test 분포, 순서 기반 피처는 금지다.
- 실험/학습 공용 로직의 기준은 `scripts/pipeline.py`다. 제출 추론은
  `open/baseline_submit/script.py`가 단독 실행하므로, 피처 생성·컬럼 순서·번들 계약을
  바꾸면 두 경로를 함께 갱신하고 반드시 `python3 scripts/test_submission_path.py`로
  동등성을 검증한다.
- 모델 비교는 단일 시드 또는 단일 실행 점수로 결론 내리지 않는다. forward-chaining
  시즌 폴드, 다중 시드 앙상블, paired 표준오차와 Brier Skill Score를 기준으로 판단한다.
- 새 실험의 결과(날짜, 변경, 로그 경로, 채택/기각 근거)는
  `docs/experiments/YYYY-MM-DD.md`에 append한다. 새 날짜면 파일을 만들고
  `docs/EXPERIMENTS.md` 인덱스에 링크를 추가한다. 장기 결론이 바뀐 경우에만
  `CLAUDE.md`의 요약도 갱신한다.
- 학습/진단 스크립트는 저장소 루트에서 `python3 scripts/<name>.py`로 실행한다.
  `scripts/`로 이동해 실행하지 않는다.

## 제출 변경 시 최소 검증

1. 시즌 홀드아웃으로 Brier Skill Score를 확인한다.
2. 전체 학습 데이터로 번들을 재생성한다.
3. `script.py`의 로컬/clean-room 추론과 `test_submission_path.py`를 통과시킨다.
4. `model/`, `script.py`, `requirements.txt`만 최상위에 둔 `submit.zip` 구조를 확인한다.

## 파일 배치

- 루트: `AGENTS.md`, `CLAUDE.md`, `ARCHITECTURE.md`, `METHOD.md`, 현재 `submit.zip`만
  프로젝트 문서·최종 제출물로 유지한다.
- 실행 코드: `scripts/`; 실행 로그: `logs/`; 재현 가능한 중간 예측: `experiments/preds/`;
  누적 기록과 원문 자료: `docs/`; 이전 제출본: `backups/`.

`CLAUDE.md`와 이 파일이 충돌하면, 대회 사실·제약·현재 모델에 대해서는
`CLAUDE.md`를 우선한다. 이 파일은 Codex의 시작 순서와 작업 안전장치만 정의한다.
