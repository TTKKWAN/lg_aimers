# LG Aimers — 투구 제구 성공 확률 예측 (KBO Control Success)

이 문서는 이 프로젝트 폴더 전체를 관통하는 **단일 진실 소스**다. 모델링 방법론(피처
엔지니어링, 알고리즘, 앙상블 등)은 대회 진행 중 계속 바뀔 수 있지만, 아래에 정리된
목표·데이터 계약·평가 방식·제출 규격·금지 사항은 바뀌지 않는다. 새로운 세션에서 이
프로젝트를 다시 열었을 때 이 파일 하나만 읽으면 전체 구조를 파악할 수 있어야 한다.

## 1. 목표 (무엇을 예측하는가)

- 문제: 야구(KBO) 투구 하나하나에 대해 **투구 직전 시점**까지 알 수 있는 정보만으로
  그 투구가 "제구 성공"일 확률(`control_success = 1`일 확률)을 예측하는 이진 분류
  확률 추정 문제.
- 제구 실패로 정의되는 3가지 케이스(그 외는 전부 성공):
  1) 스트라이크존 가운데 부근으로 들어간 공
  2) 스트라이크존에서 크게 벗어난 공
  3) 포수 요구 방향과 반대로 들어간 공
- **핵심 제약**: 입력 피처는 반드시 "해당 투구가 던져지기 전에 확인 가능한 정보"만
  사용해야 한다. 투구 결과, 실제 코스, 실제 구종, Trackman 실측값 등 사후 정보는
  절대 입력으로 쓸 수 없다 (§5 금지 사항 참고).
- 대회 단계: 온라인 해커톤(Phase 2) → 상위 약 100명이 오프라인 해커톤(Phase 3)
  진출. 지금 하는 작업은 Phase 2 리더보드 제출.

## 2. 저장소 구조

2026-08-06에 루트에 흩어져 있던 파일을 종류별 폴더로 재정리했다 (§2.1 규칙 참고).
스크립트는 항상 **저장소 루트에서** `python3 scripts/xxx.py` 형태로 실행한다 —
스크립트 안의 `./open/data`, `./open/baseline_submit/...` 등은 전부 실행 시점의
cwd(=루트) 기준 상대경로이기 때문에, `scripts/`로 `cd`해서 실행하면 깨진다.
`scripts/` 안의 파일끼리 하는 `from pipeline import ...`류 임포트는 문제없다
(파이썬이 스크립트 자신의 디렉터리를 `sys.path`에 넣어주므로).

```
lgaimers/
├── CLAUDE.md                     # 이 문서 (프로젝트 전체 가이드, 항상 루트)
├── AGENTS.md                     # Codex 세션 시작 지침 (CLAUDE.md 읽기를 강제, 항상 루트)
├── ARCHITECTURE.md                # 코드 구조 하이레벨 설명 (항상 루트)
├── METHOD.md                      # 방법론 설명 (항상 루트)
├── submit.zip                     # 리더보드에 그대로 업로드하는 최종 산출물 (항상 루트)
├── .venv/                        # 로컬 학습용 가상환경 (python3.11, 평가서버 핀버전)
├── scripts/                       # 모든 학습/실험/진단/검증 .py
│   ├── pipeline.py                 # v3 공용 코어: 피처/EB축소/TE/모델/평가 (다른 스크립트가 import)
│   ├── train_baseline.py           # v1 RandomForest 학습
│   ├── experiment_search.py        # v2 HGB 탐색 라운드1 (+ add_derived 정의 원본)
│   ├── experiment_search2.py       # v2 탐색 라운드2
│   ├── experiment_search3.py       # v2 탐색 라운드3
│   ├── train_final.py              # v2 최종 재학습 (폐기, hgb.pkl은 v3로 교체됨)
│   ├── build_final.py              # v3 최종 재학습 → model/bundle.pkl 저장
│   ├── run_experiments.py          # v3 실험 러너
│   ├── run_final_search.py         # v3 최종 후보(EB축소 k/TE유무) 비교, fold=2024
│   ├── run_stage3.py               # v3 단계별 러너
│   ├── diag_seed_variance.py       # 진단: 시드 노이즈 바닥 측정
│   ├── diag_level.py               # 진단: 시즌 base rate 드리프트(level 문제) 측정
│   ├── diag_calibration.py         # 진단: 보정 기울기 a*의 시즌간 전이 여부 측정
│   └── test_submission_path.py     # 학습↔script.py 피처 동등성 검증 (clean-room)
├── logs/                          # 위 스크립트들의 실행 로그 (.log, 스크립트와 동일 이름)
├── experiments/preds/              # 진단/탐색 스크립트가 저장한 예측값 (.npz)
├── notebooks/                     # organizer 제공 베이스라인 노트북 (.ipynb)
├── docs/                          # 대회 원문 문서 + 날짜별 실험 기록
│   ├── guide.txt                   # 대회 개요·리더보드·제출 규격 원문
│   ├── rule.txt                    # 대회 규칙 원문 (외부 데이터/모델 제한 등)
│   ├── data_guide.txt              # 데이터 컬럼 설명 메모
│   ├── EXPERIMENTS.md              # 날짜별 실험 파일 인덱스 + 리더보드 표
│   └── experiments/YYYY-MM-DD.md   # 해당 날짜의 실험 로그 (날짜 파일 안에서 append)
├── backups/                       # 이전 제출본/모델 백업 (예: submit_v2_backup.zip)
└── open/
    ├── data_description.md       # 데이터 컬럼 설명서 (원문, 상세는 §4 참고)
    ├── data/
    │   ├── train.csv             # 학습 데이터 1,475,092행 x 49컬럼 (2019~2024, control_success 포함)
    │   ├── test.csv               # 형식 확인용 5행 샘플 x 48컬럼 (실제 평가 시 245,789행으로 서버가 교체)
    │   ├── sample_submission.csv  # 제출 양식 (row_id, control_success)
    │   └── trackman_history.csv   # 2019~2024 Trackman 원시 로그 1,793,078행 x 30컬럼 (보조 데이터, 1:1 결합 아님)
    └── baseline_submit/           # 제출 가능한 산출물 원형 (submit.zip으로 압축해서 제출)
        ├── model/bundle.pkl        # 미제출 v12: v11 + 투수별 chase policy
        ├── model/bundle_v10_1010_backup.pkl  # 확인 최고 v10 복구본
        ├── model/bundle_f05_backup.pkl  # v3(HGB전용) f=0.5, 리더보드 898점 확인
        ├── model/bundle_f0_backup.pkl   # v3(HGB전용) f=0(재중심화 없음), 미제출
        ├── model/bundle_f1_backup.pkl   # v3(HGB전용) f=1.0, 리더보드 843점(기각)
        ├── model/bundle_v4_hgb8_lgbm3_f05.pkl  # 리더보드 905점 v4 원본(참고용)
        ├── model/bundle_v4_905_backup.pkl       # 리더보드 905점 확인 v4 번들
        ├── model/bundle_v5_era_lgbm_f05.pkl     # 리더보드 903점 v5 참고·백업 번들
        ├── script.py              # 평가 서버가 실행하는 추론 코드
        └── requirements.txt       # 추론에 필요한 패키지 버전 고정
```

### 2.1 파일 정리 규칙 (새 파일을 만들 때 항상 적용)

- **루트에는 아래만 둔다**: `AGENTS.md`(Codex 시작 지침), `CLAUDE.md`,
  `ARCHITECTURE.md`, `METHOD.md`(§1 단일 진실 소스급 문서), `submit.zip`(현재 제출 산출물), `.venv/`, `open/`(데이터+제출
  원형, organizer 제공 구조라 그대로 유지), `.claude/`/`.omc/` 같은 툴링 디렉터리.
  그 외 새 파일은 아래 규칙에 따라 하위 폴더로 넣는다.
- **`scripts/`**: 학습·실험·진단·검증용 `.py`는 전부 여기. 항상 저장소 루트에서
  `python3 scripts/파일명.py` 로 실행한다 (스크립트 내부 상대경로가 cwd=루트를
  가정함). 서로 import하는 스크립트(`pipeline.py` 등 공용 모듈 포함)는 반드시
  같은 `scripts/` 안에 있어야 한다 — 다른 폴더로 흩어지면 임포트가 깨진다.
- **`logs/`**: 위 스크립트를 `python3 scripts/x.py > logs/x.log` 형태로 실행한
  로그. 로그 파일명은 스크립트 파일명과 동일하게 맞춘다 (예: `train_final.py` →
  `logs/train_final.log`).
- **`experiments/preds/`**: 진단/탐색 스크립트가 저장하는 예측값 등 재현 가능한
  중간 산출물(`.npz`, `.csv` 등). 언제든 스크립트 재실행으로 다시 만들 수 있는
  것만 여기 둔다 — 코드가 참조하지 않는 캐시성 파일.
- **`notebooks/`**: `.ipynb`. organizer 제공 원본이나 탐색용 노트북.
- **`docs/`**: 대회 규정/데이터 설명 등 참고용 원문 텍스트 + 날짜별 실험 기록.
  `EXPERIMENTS.md`는 날짜 파일 링크와 리더보드 표만 유지하고, 상세 실험은
  `docs/experiments/YYYY-MM-DD.md`에 쓴다. §1의 "단일 진실 소스"인
  `CLAUDE.md`/`ARCHITECTURE.md`/
  `METHOD.md`는 예외로 루트에 둔다 — 이 셋은 **한 번 읽으면 현재 상태를 전부
  파악할 수 있는 크기로 유지**하는 게 원칙이라, "새 실험 결과를 어디에 적을까"
  고민될 때는 당일 날짜 파일에 append하고, 새 날짜면 인덱스 링크를 추가한다.
  CLAUDE.md는 그 결과로 "지금까지의 결론"이 바뀐 경우에만 요약을 갱신한다.
- **`backups/`**: 더 이상 현재 제출본이 아닌 과거 `submit*.zip`이나 모델
  아카이브. 최신 제출 파일은 항상 루트의 `submit.zip` 하나뿐이어야 한다 — 새
  버전을 만들면 기존 루트 `submit.zip`을 `backups/`로 옮기고 새 파일을 루트에
  `submit.zip`이라는 이름으로 놓을 것.
- **`open/`**: organizer가 정해준 구조이므로 내부 레이아웃(`data/`,
  `baseline_submit/`)은 바꾸지 않는다. 모델 아티팩트 파일명만 자유
  (`script.py`의 `MODEL_PATH`와 일치시킬 것, §9 참고).
- 새 파일을 어디에 둘지 애매하면: "실행 중인 코드가 이 파일을 상대경로로
  읽거나/쓰는가?" 먼저 자문 — 그렇다면 그 코드를 실행하는 위치(루트) 기준으로
  경로가 깨지지 않는지 확인 후 배치할 것.

## 3. 평가 지표

Brier Skill Score. 값이 클수록 좋음 (상한 100000, 하한 0에서 클리핑).

```
Brier   = mean((p_i - y_i)^2)
r       = mean(y_i)                 # 전체 평가 데이터의 실제 평균 제구 성공률(비공개)
Base    = r * (1 - r)               # "항상 r을 예측"하는 상수 모델의 Brier
Score   = max(0, 100000 * (1 - Brier / Base))
```

- 로컬 검증 시에는 `r`을 검증 세트의 실제 평균으로 근사해서 같은 공식을 쓴다
  (`scripts/train_baseline.py`의 검증 셀 참고).
- **확률 보정(calibration)이 성능에 직결된다.** 단순 정확도나 AUC가 아니라 예측
  확률값 자체의 정밀도가 점수를 좌우하므로, 방법론을 바꾸더라도 항상 Brier(또는
  log loss + isotonic/platt calibration)로 검증할 것.
- Public = 전체 평가 데이터 100%, Private = 종료 시점 Public과 동일 (즉 사실상
  풀 데이터 단일 스코어). 리더보드 셰이크업보다 **일반화 성능** 자체에 집중.

## 4. 데이터 계약

### 4.1 `train.csv` / `test.csv` (핵심 입력, 행 = 투구 1개)

두 파일은 입력 피처 구조가 동일하고, `train.csv`에만 정답 `control_success`
(1=성공, 0=실패)가 추가로 있다. **학습에 쓸 피처 목록은 항상 `test.csv`의 컬럼
기준으로 정해야 한다** — `train.csv`에만 있는 컬럼을 넣으면 평가 시점에 없어서
추론이 깨진다 (베이스라인 노트북이 이 방식을 그대로 씀).

컬럼 그룹:
- 식별자/경기 정보: `row_id`, `season`, `game_month`, `game_dayofweek`, `inning`,
  `top_bottom`(T/B), `game_type`
- 카운트/점수: `balls_before`, `strikes_before`, `outs_before`,
  `run_top_before`, `run_bot_before`, `run_total_before`,
  `score_diff_home`, `score_diff_pitcher_team`
- 주자/상황 중요도: `runner_on_1b/2b/3b`, `num_runners_on`,
  `base_state`(`___`,`1__`,`_2_`,`__3`,`12_`,`1_3`,`_23`,`123`),
  `home_win_expectancy`, `away_win_expectancy`, `li`
- 선수/팀: `pitcher_id`, `batter_id`, `pitcher_hand`, `batter_hand`,
  `pitcher_team_id`, `batter_team_id`
- **`asof_*` 과거 이력 피처 (투구 직전까지 사전 계산됨, 사용 가능)**:
  `asof_pitcher_n`, `asof_pitcher_success_rate`, `asof_pitcher_reverse_rate`,
  `asof_pitcher_middle_rate`, `asof_pitcher_ball_rate`, `asof_pitcher_strike_rate`,
  `asof_pitcher_prev{1,3,5}_game_success_rate`,
  `asof_pitcher_prev{1,3,5}_game_middle_rate`,
  `asof_batter_n`, `asof_batter_success_rate`, `asof_batter_middle_rate`,
  `asof_pitcher_pitchmix_n`, `asof_pitcher_fastball_rate`,
  `asof_pitcher_breaking_rate`, `asof_pitcher_offspeed_rate`
  - 표본 수가 0인 cold-start 케이스는 결측일 수 있음 → 대치/스무딩/폴백 전략은
    자유롭게 설계 가능 (베이스라인은 단순 median impute).

`test.csv`는 배포본에서는 형식 확인용 5행 샘플이고, **실제 리더보드 평가 시
평가 서버가 245,789행짜리 진짜 test.csv로 그 자리에서 교체**한다. 로컬 5행 샘플로
컬럼/타입만 확인하고, 실제 정확도 검증은 `train.csv`를 시즌 등으로 쪼갠 홀드아웃으로
해야 한다.

### 4.2 `trackman_history.csv` (보조 데이터)

2019~2024년 원시 Trackman 로그. `train.csv`/`test.csv`와 1:1로 join되는 테이블이
아니다 (`trackman_game_id` ≠ `row_id`). 구종·구속·회전수 등 원시 물리량이 있어
투수별/구종별 요약 통계를 만들어 `asof_*`류 피처를 보강하는 용도로만 쓸 수 있다.
2025년 데이터는 없음. 베이스라인은 이 파일을 아예 사용하지 않음 — 개선 여지가 큰
부분.

### 4.3 `sample_submission.csv`

컬럼: `row_id`, `control_success`(0~1 실수 확률). 제출 시 실제 평가 서버의
`test.csv`와 `row_id`가 정확히 일치해야 함.

## 5. 절대 금지 사항 (리크 방지)

- 현재 투구 이후 확정되는 모든 정보, 실제 코스/판정/결과/구종, 현재 투구의
  Trackman 실측값, 2025년 Trackman 데이터.
- **평가 데이터(`test.csv`) 내부의 다른 행을 이용한 어떤 통계도 금지**: 행별
  누적/빈도/분포 통계, target encoding, 행 순서 기반 rolling/expanding, 평가
  데이터 분포를 본 사후 보정. → 즉 test 시점 피처는 **그 행 하나만** 갖고
  독립적으로 계산 가능해야 한다. 조직 측이 제공한 `asof_*`는 이미 이 원칙을
  지켜 사전 계산된 것이므로 사용 가능.
- 외부 데이터 사용 금지 (공식 제공 데이터만: train/test/trackman_history).
- 사전학습 모델/가중치: 최소 비상업 허용 라이선스(MIT, Apache-2.0 등)로 공개된
  것만 가능. OpenAI API, Gemini API 등 원격 API 전용 모델은 사용 불가 — 모든 추론은
  로컬(제출된 코드)에서 재현 가능해야 함.
- 새로운 피처를 설계할 때마다 "이 값이 test.csv 한 행만 보고 계산 가능한가?"를
  항상 자문할 것. `trackman_history.csv`로 피처를 만들 때도 평가 시점 이후 정보가
  섞이지 않게 시즌/날짜 컷오프를 지킬 것.

## 6. 제출 규격 (코드 제출 대회)

`submit.zip`의 최상위 구조가 정확히 아래와 동일해야 함 (추가 최상위 폴더 있으면
설치 오류):

```
submit.zip
├── model/            # 모델 가중치/아티팩트
├── script.py         # 평가 서버가 실행하는 추론 진입점
└── requirements.txt  # pip install -r requirements.txt 로 설치 가능해야 함
```

- 평가 서버가 여기에 `data/`(읽기전용, 실제 test.csv+sample_submission.csv)와
  `output/`을 자동 추가함. `script.py`는 `./data/test.csv`를 읽고
  `./output/submission.csv`를 **반드시 그 파일명으로** 저장해야 함.
- 리소스/시간 제한: 6 vCPU, 28GB RAM, L4 GPU 22.4GiB VRAM, Ubuntu 22.04.5,
  Python 3.11.15, CUDA 12.8. 패키지 설치 ≤10분, 추론 실행 ≤10분(245,789 샘플
  기준), zip ≤10GB(압축 해제 후 ≤32GB). **인터넷 연결 불가** (패키지 설치 단계
  제외) — 추론 코드에서 외부 다운로드가 필요한 모델/토크나이저는 절대 쓰면 안 됨,
  모델 파일은 반드시 `model/`에 로컬로 포함.
- 평가 서버 기본 설치 패키지(버전 고정, 굳이 requirements.txt에 다시 안 적는 게
  안전): `torch==2.7.1+cu128`, `pandas==2.0.3`, `numpy==1.26.4`,
  `scipy==1.15.3`, `scikit-learn==1.8.0`, `joblib==1.5.3` 등. 이 목록에 있는
  패키지는 버전을 다르게 pin하면 설치 오류 위험 → 로컬 개발 venv도 이 버전에
  맞춰뒀음 (`.venv`, python3.11).
- 오류 2종: **설치 오류**(zip 구조/패키지 설치 실패 → 일일 제출 횟수에 안 들어감)
  vs **제출 오류**(script.py 실행 중 오류 → 일일 제출 횟수에 들어감). 즉 로컬에서
  zip 구조와 script.py 실행을 미리 검증해두는 게 제출 횟수를 아끼는 길.

## 7. 현재 모델 (baseline → current-season → CatBoost → ABS → 개인 chase policy)

**v1 — organizer 베이스라인 (RandomForest, 보관용 스크립트만 남음, 산출물은 교체됨)**
- 파이프라인: `ColumnTransformer(OrdinalEncoder(top_bottom/game_type/base_state) +
  SimpleImputer(median, 나머지 44개 수치형)) → RandomForestClassifier(
  n_estimators=100, max_depth=10, min_samples_leaf=200, random_state=42)`.
- 로컬 재현: `python3 scripts/train_baseline.py` (로그: `logs/train_baseline.log`).
- 검증(2024 홀드아웃): brier=0.248767, **Score=416.18**.

**v2 — HistGradientBoosting (과거 모델, `model/hgb.pkl`)**
- 왜 바꿨나: RF는 얕은 트리(depth=10) + bagging이라 약한 신호를 못 잡고,
  median 대치로 cold-start(표본 0) 정보가 사라짐. `HistGradientBoostingClassifier`는
  scikit-learn 1.8.0에 내장(서버 기본 설치 → `requirements.txt`에 새 패키지 불필요,
  오프라인 설치 리스크 없음)이면서 결측치를 네이티브로 분기 처리하고, boosting이라
  약한 신호를 순차적으로 보정하며 학습함.
- 입력에 파생 피처 10개 추가 (행 단위 계산, 다른 행/평가 데이터 분포 미참조 —
  §5 준수): `count_diff`, `is_two_strike`, `is_three_ball`, `is_full_count`,
  `risp`(득점권 주자), `platoon_match`(투타 손 매치업), `pitcher_command_gap`
  (투수 성공률-가운데비율 차), `pitcher_recent_trend`(최근1경기-최근5경기 성공률
  차), `batter_pitcher_gap`(타자 성공률-투수 성공률 차), `close_game`(1점차 이내).
  `open/baseline_submit/script.py`의 `add_derived()` = `scripts/train_final.py`의
  `add_derived()`와 반드시 동일하게 유지할 것 (하나 바꾸면 둘 다 바꾸기).
- 파이프라인: `ColumnTransformer(OrdinalEncoder(top_bottom/game_type/base_state) +
  passthrough(54개 수치형, NaN 유지)) → HistGradientBoostingClassifier(
  loss="log_loss", learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=30,
  l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
  n_iter_no_change=25, random_state=42)`.
- 로컬 재현: `python3 scripts/train_final.py` (로그: `logs/train_final.log`) — 전체
  2019~2024 재학습 후 `open/baseline_submit/model/hgb.pkl` 저장.
- 검증(2024 홀드아웃): brier=0.248405, **Score=561.34** (RF 대비 약 +35% 상대개선).
- 추론 속도 실측: 245,789행 `predict_proba` = 3.92초 (10분 제한 대비 여유 큼).
- `trackman_history.csv`: main 데이터의 `pitcher_id`/`batter_id`와 `trackman_history`의
  `pitcher_trackman_id`/`batter_trackman_id`는 값 범위가 완전히 달라 **겹치는 ID가
  0개** — 개체 단위(선수별) join 불가능함을 실측 확인 (2026-08-06). 팀 단위도 main은
  익명 숫자 코드, trackman은 `KIW_HER` 같은 실명 약어라 직접 매핑이 불명확 —
  추측으로 역매핑하는 건 익명화 취지에 반해 보류. 리그/시즌 단위 집계 정도만 안전한
  활용법이라 시도 안 함 (기대 이득 대비 낮은 우선순위로 판단, 재검토 여지는 있음).

**v3 — 이질 앙상블 + EB축소 + 재중심화 (과거 모델)**

v2까지는 **단일 모델**로 하이퍼파라미터를 비교했는데, 이게 근본적으로 잘못된
접근이었음이 진단 실험으로 드러났다 (자세한 수치는 §8 실험 로그 참고):
- 같은 설정도 `random_state`만 바꾸면 검증 점수가 417~561로 흔들림 (std=63).
  두 모델 점수차가 ~37점 미만이면 통계적으로 구분 불가 — v2에서 config를 고른
  근거 자체가 노이즈 위에 있었다는 뜻.
- `season` 평균 정답률이 2019(0.565)→2024(0.486)로 단조 하락. 과거로 학습해
  미래를 예측하면 이 하락만으로 검증 점수가 수백 점 깎일 수 있음 (극단적 폴드는
  oracle 보정을 해도 0점).

그래서 평가 방법론 자체를 바꿨다: **모든 비교를 다중 시드 앙상블끼리, forward-
chaining 시즌 폴드(과거→미래) 위에서, paired 표준오차와 함께** 한다
(`scripts/pipeline.py`, `scripts/run_*.py`). 이 위에서 확정된 구성:

- **피처**: v2의 파생 10개 + 원본 47개 + **경험적 베이즈(EB) 축소 피처 34개**.
  `asof_*_rate` 컬럼은 표본 수(`asof_pitcher_n` 등)가 작을수록 신뢰할 수 없는데
  (예: n=86짜리 0.29와 n=3465짜리 0.29는 다른 신뢰도), 원본 값 그대로 넣으면
  모델이 이 차이를 스스로 학습해야 한다. `shrunk = (n·rate + k·prior)/(n+k)`
  (k=50, prior=학습 데이터 리그 평균)로 표본이 적으면 리그 평균 쪽으로 미리
  당겨서 넣어준다. `pitcher_id`/`batter_id` OOF 타깃 인코딩도 시도했으나 이미
  있는 EB축소 피처와 중복돼 순수 노이즈만 더해 검증 -83점 → **기각**.
- **모델**: `HistGradientBoostingClassifier` 하이퍼파라미터를 다르게 한
  **8개 이질 앙상블**(`learning_rate` 0.02~0.06, `max_leaf_nodes` 31~127,
  `max_features` 0.6~1.0로 다양화) 평균. 단일 모델의 시드 노이즈(std=63)를
  죽이기 위한 조치 — 8개 평균이 raw 기준 500대 → 609로, 가장 큰 단일 개선책.
- **재중심화 (`logit_shift`)**: 학습 데이터 마지막 3시즌 추세를 선형 외삽해
  다음 시즌 기대 성공률을 추정하고, 그 방향으로 로짓을 상수만큼 이동. **이동
  강도는 0(안 함)~1(추세선까지 완전 반영) 사이 `recenter_f`로 조절** — 하락
  추세가 2025에도 이어졌다는 가정에 얼마나 베팅할지의 문제라 확신이 없어 다이얼로
  둠. 2026-08-06 기준 `recenter_f=0.5` 채택 (자연평균 0.4870 → 0.4746,
  `logit_shift=-0.05035`). **주의**: 같은 날 `diag_recenter_f.py`로 2024 폴드
  스윕 시 f=1.0이 f=0.5보다 로컬로 +21점 우위로 나와 실제로 f=1.0을 리더보드에
  제출까지 해봤으나 **898→843으로 -55점 손해**였음(§11) — 단일 폴드의 외삽이
  우연히 그 폴드 실제값과 근접했던 것일 뿐 2025 실측엔 전이되지 않음을 확인,
  f=0.5로 원복. `model/bundle_f1_backup.pkl`(f=1.0, 리더보드 843 확인)과
  `model/bundle_f0_backup.pkl`(f=0, 미제출)은 참고용으로 보관 중.
  **중요**: 이 상수는 전부 학습 데이터에서만 산출되고
  추론 시엔 각 행에 독립적으로 `sigmoid(logit(p)+shift)`만 적용 — 평가 데이터
  분포를 보는 게 아니므로 §5 규칙을 어기지 않는다 (test 행 간 통계 아님).
- 확률 보정(`p'=r+a(p-r)`, calibration slope)도 시도했으나 **폴드 간 α*가
  0.157↔0.883로 전혀 전이되지 않아** (2023 α로 2024를 보정하면 655→216으로
  폭락) 채택하지 않음.
- 로컬 재현: `python3 scripts/build_final.py 50 te0 8 0.5` → `model/bundle.pkl` 저장
  (모델 8개 + prior + 재중심화 상수를 자기완결적 번들로 직렬화, 30MB).
- 검증(2024 홀드아웃, 시드3×이질8 앙상블): raw=608.73, 외삽재중심화=678.22
  (oracle=679.35로 거의 근접 — 추세 외삽이 실제로 정확했음을 시사).
- 추론 속도 실측: 245,789행 × 8-멤버 앙상블 = 40.2초 (10분 제한 대비 여유 큼).
- `script.py`는 `pipeline.py`를 import할 수 없어(zip에 동봉 불가) 피처 생성
  로직을 그대로 복제해 인라인으로 갖고 있음 — `test_submission_path.py`로 두
  경로의 산출 피처가 값·순서까지 동일한지 검증 완료. 구조 설명은 `ARCHITECTURE.md`,
  방법론 전체 설명(입력→가공→모델→왜)은 `METHOD.md` 참고.

**v4 — LightGBM 이질성 추가 (직전 확정 모델, 리더보드 905점 확인)**

v3까지 앙상블 8개가 전부 `HistGradientBoostingClassifier`라 모델 계열 다양성이
없었음. `scripts/diag_ensemble_member.py`(모델 계열별 완전 분리 프로세스로 학습
후 `.npz`로 예측만 저장 — 초기에 같은 프로세스에서 HGB+LightGBM을 섞어 돌렸다가
알 수 없는 이유로 멤버 하나에 수천 초씩 걸리는 현상을 겪어서 원인 격리차 분리함,
나중에 컴퓨터 재부팅 타이밍이 원인이었을 가능성이 높다고 확인됨) +
`scripts/diag_lgbm_compare.py`(재학습 없이 npz만 로드해 비교)로 forward-chaining
3폴드(2022/2023/2024)에서 `LightGBMClassifier` 3개(`num_leaves`/`learning_rate`/
`colsample_bytree`/`subsample` 다양화)를 기존 HGB 5개 앙상블에 섞었더니 **3개
폴드 전부 유의미한 이득**(2022: +12.69, 2023: +58.59, 2024: +28.84, 전부
SE 3.6~4.0 → 3.5σ~15σ, 폴드평균 +33.37) — recenter_f/trackman과 달리 방향이
일관돼 신뢰도가 높다고 판단해 채택.

- `scripts/pipeline.py`에 `make_lgbm_model`/`LGBM_D` 추가 (HGB와 동일
  `ColumnTransformer` 전처리 재사용, `Pipeline` 인터페이스가 같아 `script.py`는
  `members` 리스트를 모델 타입 구분 없이 순회 — 수정 불필요).
- `scripts/build_final.py`에 5번째 CLI 인자 `n_lgbm` 추가, HGB 8개 + LightGBM
  3개(diag에서 검증된 것과 동일 하이퍼파라미터) = 총 11개 멤버를 한 번들에 저장.
- `requirements.txt`에 `lightgbm==4.7.0` 추가. **로컬 macOS는 `brew install
  libomp` 없이는 import 자체가 실패**(`OSError: libomp.dylib not loaded`) —
  Ubuntu 평가 서버에서는 v4 제출이 정상 설치·실행되어 리더보드 905점까지 확인됨.
- 로컬 재현: `python3 scripts/build_final.py 50 te0 8 0.5 3` → `model/bundle.pkl`
  저장 (11개 멤버, 37.0MB). `logit_shift=-0.05052`로 v3(f=0.5)의 -0.05035와
  거의 동일 — LightGBM 추가가 앙상블 자연평균에 미치는 영향은 미미함.
- 245,789행 규모 추론 시간 실측: **41.5초** (v3의 HGB 8-멤버 단독 40.2초와
  거의 동일, 10분 제한 대비 여유 큼). 이 결과로 "HGB+LightGBM을 같은 프로세스에서
  같이 돌리면 비정상적으로 느려질 수 있다"는 우려도 사실상 기각됨(정상 속도).
- clean-room 검증 통과 (5행 샘플 → `submission.csv` 정상 생성).
- 이전 v3(HGB전용, f=0.5, 리더보드 898점 확인) 백업:
  `backups/submit_v3_hgb_only_f05_backup.zip`,
  `open/baseline_submit/model/bundle_f05_backup.pkl`.
  새 번들 별도 보관: `model/bundle_v4_hgb8_lgbm3_f05.pkl`.
- **리더보드 실측 905점** — v3 HGB전용 898점 대비 +7점으로 최고 기록을 갱신했지만,
  로컬 3폴드 평균 +33.37점에 비해 실제 이득은 작았다. 모델 다양성의 방향은 맞되
  로컬 개선폭을 그대로 기대하면 안 된다는 추가 근거다.

**v5 — 모델별 시대보정 EB (과거 제출, 리더보드 903점 확인·보류)**

사회과학의 period effect 분해처럼 선수의 과거 비율에서 해당 시즌의 리그 중심을
분리했다. 학습 입력의 시즌별 중심만 사용하고, 미지의 다음 시즌 중심은 최근 3개
학습 시즌 추세로 미리 외삽해 번들에 고정한다. 평가 데이터의 다른 행이나 분포는
전혀 보지 않는다.

```
era_skill = n / (n + 50) * (asof_rate - expected_league_rate_for_season)
```

- HGB에는 효과가 중립이라 v4의 고정 prior EB 10개를 그대로 유지한다.
- LightGBM 3개만 고정 `sh_*rate` 10개를 `era_skill_*` 10개로 교체한다. 최근경기
  계층축소·신뢰도·원본 피처는 그대로 유지한다. 각 모델 입력은 여전히 91개이며,
  추론이 만드는 union만 101개다.
- 검증 HGB5+LightGBM3 혼합 기준 v4 대비 2022 +2.65, 2023 +2.19,
  2024 **+15.59(SE 3.46)**로 세 폴드 모두 양수. 전 모델에 시대보정을 강제하면
  HGB에서 손해라 모델 계열별 피처 계약을 번들 `member_num_cols`에 저장한다.
- 재현: `python3 scripts/build_final.py 50 te0 8 0.5 3 era1`.
- 전체 재학습 완료: HGB8+LightGBM3, 37.0MB, 자연평균 0.48709,
  `logit_shift=-0.05044`.
- `scripts/test_submission_path.py`에서 HGB/LightGBM의 서로 다른 피처 경로와
  제출 `script.py` 값·순서가 동일함을 확인. clean-room 5행 실행 통과.
- 245,789행 추론 **41.95초**, ZIP 무결성 및 최상위 3파일 구조 확인.
- 905점 v4는 `backups/submit_v4_905_backup.zip`과
  `model/bundle_v4_905_backup.pkl`에 복구 가능하게 보존.
- **리더보드 실측 903점** — v4의 905점보다 2점 낮아 시대보정은 채택 근거가
  부족하며 기각 또는 보류 상태다. 현재 v5 제출물은
  `backups/submit_v5_903_backup.zip`에도 동일 SHA-256으로 보존했으며, 최고 확인
  모델과 실제 제출 권장 파일은 계속 v4 905 백업이다.

**v6 — fixed-EB pressure-context expert 20% 혼합 (리더보드 905점, 백업 보존)**

투수의 능력·최근 상태가 풀카운트, 3볼, 득점권, LI, 후반·접전 상황과 다르게
상호작용할 수 있다는 가설을 시험했다. 11개 `능력×압박` 피처를 전체 모델에 직접
추가하면 forward 폴드 gain이 -1.29/+61.46/-3.95로 불안정해 기각했다. 대신 선수·팀
ID와 타자 이력을 제외하고 현재 상황, 투수의 fixed-EB 능력·최근 dev, 해당 상호작용만
보는 작은 context LGBM을 별도 expert로 두자 오차 다양성으로 안정적인 이득이 났다.

- 확인 기준선은 저장된 **HGB5+LGBM3 fixed-EB** 앙상블이며, context LGBM 3개를
  20% 섞었을 때 paired gain은 2022 **+9.46(SE 3.60)**, 2023
  **+36.34(3.77)**, 2024 **+30.55(4.15)**로 세 폴드 모두 양수였다.
- 30%도 세 폴드 양수였지만 fold별 oracle 비중이 0.213/0.526/0.377로 변했고
  2022의 여유가 줄어, 평균 최대화보다 안정성을 우선해 **base 80% + context 20%**를
  고정했다. context-only는 약하므로 보조 expert 이상의 비중을 주지 않는다.
- production은 **HGB8+LightGBM3 모두 v4 fixed EB**를 쓰는 base 11개와 context
  LGBM3를 각각 평균한 뒤 80/20 혼합한다. v5 시대보정은 섞지 않는다. base 멤버는
  각 91개, context는 49개 입력, union은 102개다.
- 전체 재학습 결과 혼합 자연평균 0.4871, `recenter_f=0.5`,
  `logit_shift=-0.05048`, 번들 40.4MB. 제출 경로의 context 값·순서·혼합,
  legacy 번들, v5 시대보정 번들 하위호환 테스트와 clean-room 5행 실행을 통과했다.
  245,789행 추론은 **47.09초**다.
- 루트 `submit.zip` SHA-256은
  `02f4e28112e93cb3878b7659ec3dc90fd443fa9c9efab3b1e9eb6b8a44816e22`, 최상위
  파일 3개 구조와 ZIP 무결성을 확인했다.
- **리더보드 실측 905점** — v4와 동률이며 context expert의 로컬 개선이 실제 점수
  개선으로 이어지지는 않았다. 검증 기준선 HGB5+LGBM3와 production HGB8+LGBM3가
  정확히 같지 않았다는 한계도 그대로 남는다. v6은
  `backups/submit_v6_905_backup.zip`에 보존했으며, 같은 점수라면 더 단순한 v4
  `backups/submit_v4_905_backup.zip`을 제출 안전판으로 유지한다.

**v7 — current-season `asof` success/workload 복원 (리더보드 938점 확인)**

현재 행의 누적 `asof_n`과 `round(n×success_rate)`에서 학습 데이터로 고정한 이전
시즌 종료 누적값을 빼, 투수와 타자의 현 시즌 workload·success를 복원한다. 이전
시즌 마지막 행의 정답까지 더해 success endpoint를 정확히 닫고, lookup이 없는 신규
ID는 0 baseline을 쓴다. test의 다른 행을 읽지 않는 row-local 변환이다.

- 투수·타자 각각 known/invalid/n/log_n/reliability/raw rate/EB rate/career 대비 dev,
  총 16개를 LightGBM에만 추가한다. HGB8은 검증된 fixed-EB 91개 입력을 유지하고
  LGBM3는 107개 입력을 쓴다. v6 context와 v5 era 피처는 제외한다.
- 저장 HGB5+LGBM3 기준선에서 기존 LGBM3를 새 LGBM3로 전면 교체한 paired gain은
  2022 **+57.05(SE 6.02)**, 2023 **+211.90(6.30)**, 2024
  **+58.15(5.91)**다. 교체 비중 25/50/75/100%에서 세 폴드가 모두 양수이고
  단조 개선돼 100% 교체를 채택했다.
- 전체 재학습 결과 11멤버 자연평균은 0.48714493,
  `recenter_f=0.5`, `logit_shift=-0.05056`, 번들 37.0MB다. 91/107 멤버별
  컬럼 선택, batch/single·shuffle 불변성, unseen zero fallback, n=0 endpoint,
  legacy 경로와 clean-room 5행을 통과했다. 245,789행 추론은 **40.23초**다.
- 루트 `submit.zip` SHA-256은
  `3a5322c71cc806d96bc5bce6c16a2ec6ace52fc98dc65da1ae731d01bfc0d599`다.
  **리더보드 실측 938점**으로 확인됐으며 v4/v6 905 대비 +33점이다. 확인본은
  `backups/submit_v7_938_backup.zip`에 보존했다.

**v8 — production family 재가중 (v9 이전 후보, 백업 보존)**

- v7의 학습된 HGB8과 season-LGBM3는 그대로 유지하고, 멤버 11개 단순평균
  (HGB 72.7%/LGBM 27.3%) 대신 family 내부 평균 후 **HGB 25% + LGBM 75%**로 섞는다.
- production 동일 HGB8+LGBM3 forward 재학습과 fold별 `recenter_f=0.5`를 포함한
  v7 대비 paired gain은 2022 **+17.48(SE 7.73)**, 2023 **+272.22(8.29)**,
  2024 **+53.93(8.20)**다. LGBM 100%는 2022가 -3.81로 꺾여 채택하지 않았다.
- 전체 학습 2024 family 자연평균 `0.49587768`, f=0.5 목표 `0.47902818`,
  `logit_shift=-0.06769224`. 제출 경로와 clean-room 검증을 통과했다.
- 루트 ZIP SHA-256은
  `d4270cb1921de22845c657d2f2cc8782197901a1cadff8eecff76f0b28745029`다.
  확인 최고와 안전판은 `backups/submit_v7_938_backup.zip`이다.

**v9 — CatBoost 선수×상황 categorical expert (리더보드 1000+ 확인, 안전판)**

- 선수·팀 ID를 수치 서열로 쓰지 않고 문자열 명목형 lookup key로 바꿨다. count,
  base, pressure, inning 범주와 최대 2차 조합을 CatBoost ordered target statistics로
  학습한다. 현 시즌 16개와 EB 입력도 함께 보는 별도 expert다.
- 동적 잠재상태는 v8 대비 제출형 gain이 투수-only
  **-169.23/-83.26/-24.04**, 투수+타자 **-138.30/-89.78/-87.91**이라 기각했고
  CatBoost와 결합하지 않았다.
- CatBoost 2시드 평균의 v8 대비 제출형 gain은 50% 혼합에서
  **+65.72/+259.30/+49.55**, 60%에서 **+70.84/+302.57/+51.44**, 75%에서
  **+73.49/+362.11/+49.26**이었다. 최신 폴드가 60% 이후 꺾여 **v8 40% +
  CatBoost 60%**를 채택했다.
- 전체 학습 혼합 2024 자연평균 `0.48756376`, f=0.5 목표 `0.47487122`,
  `logit_shift=-0.05136262`, 번들 43.6MB다. 245,789행 모사 추론은 36.00초다.
- 루트 `submit.zip` SHA-256은
  `8ea71f1aa8c27d94e81fcfb552d43f89c088ed018d64515755306d195561eeb9`다.
  v8은 `backups/submit_v8_unsubmitted_backup.zip`에 보존했다.
- **리더보드 실측 1000점 초과**로 현재 최고를 갱신했다(정확한 점수 미기록).
  확인본은 `backups/submit_v9_1000plus_backup.zip`에 byte-identical 보존했다.

**v10 — 현 시즌 all4 command CatBoost family (백업, 리더보드 1010)**

- v9 기존 CatBoost2에 투수의 현 시즌 strike/ball/middle/reverse raw·EB·통산 대비
  편차 12개를 추가한 command CatBoost2를 별도로 학습했다. test 행 하나의 누적값과
  전체 train에서 고정한 투수별 2024 endpoint lookup만 사용한다.
- v9 대비 command expert 50% 교체의 forward paired gain은 2022
  **+17.80(SE 2.77)**, 2023 **+18.12(2.31)**, 2024 **+6.93(2.30)**이다.
  100% 교체는 +25.80/+29.86/+7.09였지만 최신 폴드 이득이 거의 같고 SE가 두 배라
  기존 CatBoost2 50% + command CatBoost2 50%를 채택했다.
- 전체 식은 **v8 40% + CatBoost family 60%**를 유지한다. 전체 학습 2024 자연평균
  `0.48751748`, f=0.5 목표 `0.47484808`, `logit_shift=-0.05126671`, 번들
  49.8MB다. 245,789행 모사 추론은 102.64초다.
- 조건부 gate, count hierarchy/선수×전략 category, feature ablation/유형별 재중심화,
  저랭크 pitcher×batter factorization은 forward 방향 불일치로 모두 기각했다.
- 루트 `submit.zip` SHA-256은
  `095ff3b7ce401c296600b7a9eb4c71f437d158d1cafb6fd6242add9c0fd89fb8`다.
  학습↔제출 피처 동등성, 실제 혼합 수식, source/ZIP clean-room, ZIP 무결성 및 구조를
  모두 통과했다. **리더보드 실측은 1010점**이지만 사용자는 v9과 사실상 동일해
  유의미한 개선이 없다고 판단했다. 복구 안전판은 계속 v9 1000+ 백업이다.

**v11 — ABS 관측체계 분해 expert 10% (확인 최고 백업, 리더보드 1018)**

- 2024 ABS 도입으로 같은 누적 rate의 의미가 바뀐다는 가설을 반영했다. 오염된 통산
  rate를 ABS expert에서 제외하고, 현재시즌 success/strike/ball/middle/reverse와
  최근경기만 사용한다. 각 rate는 2024년 5~9월 train-fixed 중앙값/IQR 기준 상대값,
  `n/(n+50)` 신뢰 신호, Bernoulli 표본오차로 분해한다.
- 초기 적응 노이즈는 행 삭제 대신 3월 0.20, 4월 0.45, 이후 1.0의 학습 가중치로
  낮춘다. 2024 forward v10을 기준으로 앞선 월만 학습한 시간 검증에서 10% 혼합은
  3~5월→6~7월 **+12.00(SE 5.66)**, 3~7월→8~9월
  **+12.32(SE 4.73)**였다. expert 단독은 크게 나빠 보조 expert로만 쓴다.
- 2024 행만으로 CatBoost 2시드(4242/5151)를 학습했다. 최종식은
  **v10 raw 90% + ABS expert 10%**, 자연평균 `0.48732353`, f=0.5 목표
  `0.47475110`, `logit_shift=-0.05087341`, 번들 52.5MB다.
- ABS 20개 파생 피처의 학습↔제출 값·순서와 single/batch 행 독립성, 기존 전체
  제출경로, source/ZIP clean-room, ZIP 구조·무결성을 통과했다. 245,789행 모사
  피처+추론은 107.66초다. ZIP SHA-256은
  `85e3a67b6f883f1c8f7c362df98e444dd4c3f359ebbb3e49d3dea35da40e214f`다.
  리더보드 실측은 **1018점**으로 v10 대비 +8의 미세 개선이다. v10은
  `backups/submit_v10_1010_backup.zip`에 보존했다.

**v12 — 투수별 chase 개인 정책 (현재 `submit.zip`, 미제출 후보)**

- 투수끼리 정책이나 제구력을 공유하지 않는다. 최신 학습시즌(2024)의 각 투수
  2스트라이크·비풀카운트 성공률을 오직 그 투수 자신의 전체 성공률로 `k=100`
  축소한다. 현재 행에서 복원한 2025 같은 투수 성공률 변화만 개인 정책 logit에 더한다.
- 개인 전체·chase·현재시즌 표본 신뢰도의 기하평균에 최대 0.20을 곱해 v11 raw
  예측과 혼합한다. cold start/누적 불일치는 가중치 0이며 다른 test 행은 보지 않는다.
- v10 기준 여섯 시간 분할에서 chase-only 설정은 모두 양수였다. v11 ABS 예측 위에서도
  최대 0.20은 3~5월→6~7월 **+1.86(SE 2.28)**, 3~7월→8~9월
  **+2.09(SE 2.78)**로 방향이 유지됐다. 효과가 작으므로 미제출 후보로 구분한다.
- lookup 투수 386명, 번들 52.5MB. 학습↔제출 postprocessor 동등성, 전체 제출 경로,
  source/ZIP clean-room과 ZIP 무결성을 통과했다. ZIP SHA-256은
  `d455adc05108e56eef1d128904270cecdb60ff933124b17e87cedefb1807e47d`다.
  v11은 `backups/submit_v11_1018_backup.zip`에 보존했다.

**전처리 설계 원칙 (방법론이 바뀌어도 유지할 가치)**: 전처리 전부를 sklearn
`Pipeline` 안에 넣어 학습·추론 코드가 완전히 동일한 변환을 거치게 한다. `script.py`
는 파생 피처 계산(`add_derived`)과 컬럼 순서 맞추기만 하고, 인코딩·결측치 처리는
전부 파이프라인(모델 파일) 안에서 수행 — 전처리 불일치로 인한 제출 오류 방지.
v3부터는 여기에 더해 **모델 비교는 절대 단일 실행으로 하지 않는다** (다중 시드
앙상블끼리, paired SE와 함께) — 이 원칙이 v2 튜닝의 근거를 무효화시킨 교훈.

## 8. 방법론 확장 여지 (자유롭게 교체/실험하되, §5·§6 제약은 항상 지킬 것)

**실험 로그는 날짜별 [`docs/experiments/`](docs/experiments/) 파일에 있고,
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)는 인덱스와 리더보드 표다.** 새 실험은
당일 날짜 파일에 추가하고, 아래 "현재 결론"이 바뀐 경우에만 CLAUDE.md를 갱신한다.

**지금까지의 결론 (요약, 근거는 EXPERIMENTS.md)**:
- 모델 비교는 절대 단일 실행으로 하지 않는다 — 시드만 바꿔도 검증 점수가
  417~561까지 흔들려서(std=63) 단일 실행 비교는 노이즈에 속는다. 항상 다중
  시드 앙상블끼리, forward-chaining 시즌 폴드 위에서, paired SE와 함께 비교.
- `season`을 피처에서 빼면 절대 안 된다 (561→9.8로 폭락) — 시즌 간 정답률
  드리프트(2019 .565→2024 .486)를 모델이 이 컬럼으로 부분 자가보정 중이므로.
- EB축소(표본수 기반 리그평균 축소)는 확실한 이득, 타깃인코딩은 확실한 손해
  (-83점, 기존 EB축소 피처와 중복돼 노이즈만 더함) — 둘 다 재시도 불필요.
- 검증 전략: forward-chaining 시즌 폴드(과거→미래) + 항상 Brier Skill Score
  공식(§3)으로 로컬 점수 산출, 실측 리더보드와 대조(`docs/EXPERIMENTS.md`
  리더보드 표).
- `recenter_f`는 단일 폴드(2024) 로컬 스윕만으로 조정하면 안 된다 — f=1.0이
  로컬로 f=0.5보다 우위였지만 실제 리더보드에서는 898→843으로 손해였음
  (§7 v3, §11, `docs/EXPERIMENTS.md` 참고). calibration slope α*의 폴드 간
  비전이와 같은 패턴 — 재중심화 강도는 시즌마다 흔들리는 불안정한 양이라는
  두 번째 증거. 현재 f=0.5 유지.
- **`trackman_history.csv` 리그/시즌 집계 피처는 기각** — forward-chaining
  3폴드(2022/2023/2024)에서 전부 손해(평균 -43.86점, 폴드별 -20.85/-16.95/-93.80,
  전부 통계적으로 유의). 개체 단위 join이 막혀 있어 시즌 집계만 가능한데, 이미
  있는 `season` 피처와 정보가 중복되고 평가 시점 값은 결국 외삽이라 노이즈만
  더함 — `recenter_f`와 같은 "추세 외삽 의존" 계열의 두 번째 실패 사례.
  **이 데이터의 개체/팀 익명화 구조상 trackman_history는 활용 가치가 낮다는
  결론 — 재검토하려면 외삽이 필요 없는 형태(예: within-season 상대 순위)로
  접근을 바꿔야 함.**
- **LightGBM 이질성 추가는 채택** — forward-chaining 3폴드 전부 유의미한 이득
  (§7 v4, `docs/EXPERIMENTS.md` 참고), v4에서 리더보드 905점(+7)을 확인했다.
  새 상황·압박·피로 상호작용과 단순 컬럼 제거는 3폴드에서
  일관된 이득이 없어 기각. **LightGBM 멤버에만 시대보정 상대능력을 적용한 v5는
  리더보드 903점으로 v4의 905점보다 낮아 기각 또는 보류**(§7 v5).
- 최근 `asof` 변화 상태를 더 정교하게 요약한 피처와 투수 능력×압박 피처를 전체
  모델에 직접 추가하는 방식은 폴드 방향이 뒤집혀 기각. 반면 작은 context expert를
  기존 앙상블에 20% 섞는 방식은 확인 기준선(HGB5+LGBM3) 대비
  **+9.46/+36.34/+30.55점**으로 세 폴드 모두 개선돼 v6 후보로 구현했다. 다만
  production base가 HGB8+LGBM3라 정확히 같은 비교가 아니었고, 리더보드도 v4와
  같은 905점이어서 확정 개선으로 채택하지 않는다.
- Direct-Brier 회귀는 classifier 대비 -57.86/+17.26/-24.85로 방향 불일치라 기각.
  더 큰 모델이나 GPU가 필요하다는 증거는 아직 없다. 정형 데이터에서 작은 context
  expert의 다양성은 로컬에서 확인됐지만 v6 실측은 v4와 같은 905점이었다. 다음
  후보(XGBoost 등 추가 모델 계열, entity representation)는 이 실측 불일치를 고려해
  판단한다. 세부는
  `docs/EXPERIMENTS.md` 참고.
- **동적 잠재 상태는 기각, CatBoost categorical expert는 v9 후보로 채택** — 고정
  AR 유지율·선수별 변화분산 posterior는 세 폴드 모두 손해였다. 숫자 ID를 제거하고
  선수·상황을 명목형으로 처리한 CatBoost 2시드 expert는 v8과 60% 혼합했을 때
  +70.84/+302.57/+51.44로 모두 개선됐다.

## 9. 제출 워크플로 (매 제출마다 반복)

1. 방법론 변경 → `train.csv`로 학습, 시즌 홀드아웃으로 Brier Skill Score 검증.
2. 최종 모델을 전체 `train.csv`로 재학습 후 `open/baseline_submit/model/`에
   저장 (파일명 자유, `script.py`의 `MODEL_PATH`와 일치시킬 것).
3. `script.py`가 그 모델 파일을 로드해서 `test.csv`(5행 샘플) → `output/submission.csv`
   생성까지 로컬에서 한 번 실행되는지 확인. 컬럼 순서/개수가 학습 때와 다르면
   즉시 실패하므로 이 스텝을 절대 생략하지 말 것.
4. `requirements.txt`가 §6의 서버 기본 패키지와 충돌하지 않는지, 오프라인
   설치가 가능한지 확인.
5. `model/`, `script.py`, `requirements.txt`를 **추가 최상위 폴더 없이** zip으로
   묶어 `submit.zip` 생성 → 업로드.

## 10. 지금 제출 가능한 산출물

- `open/baseline_submit/`가 그대로 제출 가능한 형태로 검증되어 있음
  (§7 v12 `model/bundle.pkl` 포함 — v11 위에 투수별 chase policy 혼합 후 재중심화).
  `script.py`는 개인 chase, ABS 분해, command profile, CatBoost와 v7 season lookup 및 기존
  context/era/legacy 계약을 모두 지원한다.
- 루트의 **`submit.zip`이 리더보드에 그대로 업로드하면 되는 단일 파일**
  (model/bundle.pkl + script.py + requirements.txt, 최상위 폴더 없이 압축).
  clean-room 환경(별도 디렉토리에 압축 해제 → data/ 채워넣기 → script.py 실행)에서
  end-to-end 실행 검증 완료. 번들 52.5MB, ZIP SHA-256
  `d455adc05108e56eef1d128904270cecdb60ff933124b17e87cedefb1807e47d`, 3파일
  구조와 무결성을 확인했다. **현재 ZIP은 미제출 v12 후보**다. 확인 최고 v11(1018)은
  `backups/submit_v11_1018_backup.zip`, v10은
  `backups/submit_v10_1010_backup.zip`에 보존돼 있다.
- v6 context20 검증은 HGB5+LGBM3 기준선 대비 +9.46/+36.34/+30.55였고,
  production은 HGB8+LGBM3 base라 정확히 같은 비교가 아니며, 리더보드도 905점
  동률이라 확정 개선으로 취급하지 않는다. v5는 로컬 세 폴드 양수였어도 실측
  903으로 v4보다 낮았다. 최고 실측 905점의 v4는
  `backups/submit_v4_905_backup.zip`, v5 903은
  `backups/submit_v5_903_backup.zip`, v6 905는
  `backups/submit_v6_905_backup.zip`에 각각 보존되어 즉시 복구 가능.
- **백업**: `model/bundle_f05_backup.pkl`(v3, HGB전용 f=0.5, 리더보드 898점
  확인), `model/bundle_f0_backup.pkl`(f=0, 미제출), `model/bundle_f1_backup.pkl`
  (f=1.0, 리더보드 843점 확인 — 기각됨), `model/bundle_v4_hgb8_lgbm3_f05.pkl`,
  `model/bundle_v4_905_backup.pkl`, `model/bundle_v5_era_lgbm_f05.pkl` 보관 중.
- 이전 버전들도 백업돼 있음: `backups/submit_v2_backup.zip`(단일 HGB, hgb.pkl),
  `backups/submit_v3_f05_backup.zip`, `backups/submit_v3_hgb_only_f05_backup.zip`
  (둘 다 v3, f=0.5, 898점 — 동일 파일 다른 시점 백업),
  `backups/submit_v4_905_backup.zip`, `backups/submit_v5_903_backup.zip`,
  `backups/submit_v6_905_backup.zip`.
- 새 방법론을 시도할 때는 §9 워크플로 그대로 반복 — 특히 3번(로컬 clean-room
  실행 검증)을 건너뛰지 말 것. `script.py`의 피처 생성 로직(`add_derived`,
  `add_shrinkage`, `add_target_encoding`)을 학습 파이프라인(`scripts/pipeline.py`)과
  다르게 고치면 컬럼 불일치로 제출 오류가 나므로 항상 같이 수정하고,
  `scripts/test_submission_path.py`로 동등성을 재검증할 것.

## 11. 실제 리더보드 점수 기록

**전체 기록(날짜별 표, 로컬↔리더보드 비율 관찰)은
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)에 있다.** 여기는 최신 한 줄만:

- **최신 리더보드 실측(v11): 1018점. 현재 루트 v12는 미제출 후보**
  (v1=500 → v2=700 → v3 f=0.5=898 → v3 f=1.0=843(기각) → v4=905 →
  v5=903 → v6=905 → v7=938).
  LightGBM 추가는 방향은 맞았지만 +7점으로 개선폭은 작았다. 이후 시대보정을
  LightGBM에만 적용한 v5는 903점으로 -2점이어서 기각 또는 보류한다. context를
  추가한 v6도 905점 동률이었다. v7의 current-season success/workload 입력이
  938점으로 실제 개선됐다. 같은 모델을 HGB25/LGBM75로 재가중한 v8에 CatBoost
  expert를 추가한 v9이 1000점을 넘어 다시 최고를 갱신했다. 정확한 점수는 아직
  미기록이며, 제출 안전판은 `backups/submit_v9_1000plus_backup.zip`이다.
