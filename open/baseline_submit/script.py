# script.py — 제구 성공 확률 추론
#
# 규칙 준수 메모:
#   모든 피처는 그 행 하나만으로 계산된다. 평가 데이터의 다른 행이나 전체 분포를
#   참조하는 계산은 없다. 모델 번들에 들어 있는 prior(리그 평균), 타깃 인코딩 맵,
#   logit_shift 는 전부 학습 데이터(train.csv)에서만 산출된 상수이며 추론 시에는
#   고정값으로 행마다 독립 적용된다.
import os

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"

CAT_COLS = ["top_bottom", "game_type", "base_state"]

RATE_N_PAIRS = [
    ("asof_pitcher_success_rate", "asof_pitcher_n"),
    ("asof_pitcher_reverse_rate", "asof_pitcher_n"),
    ("asof_pitcher_middle_rate", "asof_pitcher_n"),
    ("asof_pitcher_ball_rate", "asof_pitcher_n"),
    ("asof_pitcher_strike_rate", "asof_pitcher_n"),
    ("asof_batter_success_rate", "asof_batter_n"),
    ("asof_batter_middle_rate", "asof_batter_n"),
    ("asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n"),
    ("asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n"),
]
N_COLS = ["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]
PREV_SPECS = [
    ("asof_pitcher_prev1_game_success_rate", 25, "asof_pitcher_success_rate"),
    ("asof_pitcher_prev3_game_success_rate", 75, "asof_pitcher_success_rate"),
    ("asof_pitcher_prev5_game_success_rate", 125, "asof_pitcher_success_rate"),
    ("asof_pitcher_prev1_game_middle_rate", 25, "asof_pitcher_middle_rate"),
    ("asof_pitcher_prev3_game_middle_rate", 75, "asof_pitcher_middle_rate"),
    ("asof_pitcher_prev5_game_middle_rate", 125, "asof_pitcher_middle_rate"),
]
ERA_SKILL_COLS = [f"era_skill_{rate}" for rate, _ in RATE_N_PAIRS]
ERA_RECENT_SKILL_COLS = [f"era_skill_{c}" for c, _, _ in PREV_SPECS]
ERA_GAP_COLS = [
    "era_pitcher_command_gap", "era_pitcher_recent_trend",
    "era_batter_pitcher_gap",
]
PRESSURE_ABILITY_COLS = [
    "pa_full_x_success_skill", "pa_full_x_ball_skill",
    "pa_three_ball_x_ball_skill", "pa_risp_x_success_skill",
    "pa_risp_x_ball_skill", "pa_logli_x_success_skill",
    "pa_logli_x_ball_skill", "pa_late_x_success_skill",
    "pa_close_x_success_skill", "pa_logli_x_reliability",
    "pa_highli_x_recent3_dev",
]
CATBOOST_CAT_COLS = [
    "pitcher_id_cat", "batter_id_cat", "pitcher_team_id_cat",
    "batter_team_id_cat", "pitcher_hand_cat", "batter_hand_cat",
    "top_bottom_cat", "game_type_cat", "base_state_cat",
    "count_state_cat", "pressure_state_cat", "inning_bucket_cat",
]


# =======================
# 피처 생성 (학습 때와 반드시 동일 — build_final.py / pipeline.py 와 일치)
# =======================

def add_derived(df):
    out = {}
    out["count_diff"] = df["strikes_before"] - df["balls_before"]
    out["is_two_strike"] = (df["strikes_before"] == 2).astype(np.int8)
    out["is_three_ball"] = (df["balls_before"] == 3).astype(np.int8)
    out["is_full_count"] = (out["is_two_strike"] & out["is_three_ball"]).astype(np.int8)
    out["risp"] = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype(np.int8)
    out["platoon_match"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.int8)
    out["pitcher_command_gap"] = df["asof_pitcher_success_rate"] - df["asof_pitcher_middle_rate"]
    out["pitcher_recent_trend"] = (df["asof_pitcher_prev1_game_success_rate"]
                                    - df["asof_pitcher_prev5_game_success_rate"])
    out["batter_pitcher_gap"] = df["asof_batter_success_rate"] - df["asof_pitcher_success_rate"]
    out["close_game"] = (df["score_diff_pitcher_team"].abs() <= 1).astype(np.int8)
    return pd.DataFrame(out, index=df.index)


def add_catboost_context(df):
    """ID를 서열 없는 명목형 lookup key로 쓰는 row-local CatBoost 입력."""
    out = pd.DataFrame(index=df.index)
    rename = {
        "pitcher_id": "pitcher_id_cat", "batter_id": "batter_id_cat",
        "pitcher_team_id": "pitcher_team_id_cat", "batter_team_id": "batter_team_id_cat",
        "pitcher_hand": "pitcher_hand_cat", "batter_hand": "batter_hand_cat",
        "top_bottom": "top_bottom_cat", "game_type": "game_type_cat",
        "base_state": "base_state_cat",
    }
    for source, target in rename.items():
        out[target] = df[source].fillna("__NA__").astype(str)
    out["count_state_cat"] = (df["balls_before"].fillna(-1).astype(int).astype(str)
                              + "_" + df["strikes_before"].fillna(-1).astype(int).astype(str))
    risp = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype(int)
    high_li = (df["li"].fillna(0) >= 2.0).astype(int)
    close = (df["score_diff_pitcher_team"].abs() <= 1).astype(int)
    full = ((df["balls_before"] == 3) & (df["strikes_before"] == 2)).astype(int)
    out["pressure_state_cat"] = ("f" + full.astype(str) + "r" + risp.astype(str)
                                 + "l" + high_li.astype(str) + "c" + close.astype(str))
    inning = df["inning"].fillna(-1).to_numpy(dtype=float)
    out["inning_bucket_cat"] = np.select(
        [inning <= 3, inning <= 6, inning <= 9],
        ["early", "middle", "late"], default="extra")
    return out


def add_shrinkage(df, prior, k):
    """경험적 베이즈 축소: (n*rate + k*prior)/(n+k). prior 는 학습 데이터 상수."""
    out = {}
    shrunk = {}
    for rate, ncol in RATE_N_PAIRS:
        r = df[rate].to_numpy(dtype=float)
        n = df[ncol].to_numpy(dtype=float)
        p0 = prior[rate]
        rf = np.where(np.isnan(r), p0, r)
        nf = np.where(np.isnan(r), 0.0, n)
        sh = (nf * rf + k * p0) / (nf + k)
        out[f"sh_{rate}"] = sh
        shrunk[rate] = sh
    for ncol in N_COLS:
        n = df[ncol].to_numpy(dtype=float)
        out[f"log_{ncol}"] = np.log1p(n)
        out[f"rel_{ncol}"] = n / (n + k)
    for col, pseudo_n, base_rate in PREV_SPECS:
        r = df[col].to_numpy(dtype=float)
        base = shrunk[base_rate]
        miss = np.isnan(r)
        rf = np.where(miss, base, r)
        nf = np.where(miss, 0.0, float(pseudo_n))
        sh = (nf * rf + k * base) / (nf + k)
        out[f"sh_{col}"] = sh
        out[f"miss_{col}"] = miss.astype(np.int8)
        out[f"dev_{col}"] = sh - base
    return pd.DataFrame(out, index=df.index)


def add_pressure_ability(df, prior):
    """fixed-EB 능력과 현재 압박의 row-local 상호작용(학습 경로와 동일)."""
    full = df["is_full_count"].to_numpy(dtype=float)
    three = df["is_three_ball"].to_numpy(dtype=float)
    risp = df["risp"].to_numpy(dtype=float)
    close = df["close_game"].to_numpy(dtype=float)
    late = (df["inning"].to_numpy(dtype=float) >= 7).astype(float)
    log_li = np.log1p(np.clip(df["li"].to_numpy(dtype=float), 0, None))
    high_li = (df["li"].to_numpy(dtype=float) >= 2.0).astype(float)
    success = (df["sh_asof_pitcher_success_rate"].to_numpy(dtype=float)
               - prior["asof_pitcher_success_rate"])
    ball = (df["sh_asof_pitcher_ball_rate"].to_numpy(dtype=float)
            - prior["asof_pitcher_ball_rate"])
    reliability = df["rel_asof_pitcher_n"].to_numpy(dtype=float)
    recent3 = df[
        "dev_asof_pitcher_prev3_game_success_rate"
    ].to_numpy(dtype=float)
    out = {
        "pa_full_x_success_skill": full * success,
        "pa_full_x_ball_skill": full * ball,
        "pa_three_ball_x_ball_skill": three * ball,
        "pa_risp_x_success_skill": risp * success,
        "pa_risp_x_ball_skill": risp * ball,
        "pa_logli_x_success_skill": log_li * success,
        "pa_logli_x_ball_skill": log_li * ball,
        "pa_late_x_success_skill": late * success,
        "pa_close_x_success_skill": close * success,
        "pa_logli_x_reliability": log_li * reliability,
        "pa_highli_x_recent3_dev": high_li * recent3,
    }
    return pd.DataFrame(out, index=df.index)


def expected_era_rate(seasons, spec):
    seasons = np.asarray(seasons, dtype=int)
    out = spec["intercept"] + spec["slope"] * seasons.astype(float)
    for season, value in spec["observed"].items():
        out[seasons == int(season)] = value
    return out


def add_era_features(df, era_specs, k):
    """번들에 고정된 시즌 중심만으로 공통 regime 상대능력을 계산한다."""
    out = {}
    for rate, ncol in RATE_N_PAIRS:
        r = df[rate].to_numpy(dtype=float)
        n = df[ncol].to_numpy(dtype=float)
        era = expected_era_rate(df["season"].to_numpy(), era_specs[rate])
        rf = np.where(np.isnan(r), era, r)
        nf = np.where(np.isnan(r), 0.0, n)
        out[f"era_skill_{rate}"] = (nf / (nf + k)) * (rf - era)
    for col, pseudo_n, base_rate in PREV_SPECS:
        r = df[col].to_numpy(dtype=float)
        era = expected_era_rate(df["season"].to_numpy(), era_specs[base_rate])
        valid = np.isfinite(r)
        out[f"era_skill_{col}"] = np.where(
            valid, (pseudo_n / (pseudo_n + k)) * (r - era), 0.0)
    out["era_pitcher_command_gap"] = (
        out["era_skill_asof_pitcher_success_rate"]
        - out["era_skill_asof_pitcher_middle_rate"])
    out["era_pitcher_recent_trend"] = (
        out["era_skill_asof_pitcher_prev1_game_success_rate"]
        - out["era_skill_asof_pitcher_prev5_game_success_rate"])
    out["era_batter_pitcher_gap"] = (
        out["era_skill_asof_batter_success_rate"]
        - out["era_skill_asof_pitcher_success_rate"])
    return pd.DataFrame(out, index=df.index)


def add_regime_current_features(df, era_specs, k):
    """현 시즌 success/command를 해당 시즌 중심 대비 상대값으로 변환한다."""
    out = {}
    rates = ["asof_pitcher_success_rate", "asof_batter_success_rate"] + \
            SEASON_COMMAND_RATES
    seasons = df["season"].to_numpy()
    for rate in rates:
        raw_col = f"std_{rate}"
        if raw_col not in df:
            continue
        group = "batter" if rate.startswith("asof_batter") else "pitcher"
        n = df[f"std_{group}_n"].to_numpy(dtype=float)
        raw = df[raw_col].to_numpy(dtype=float)
        era = expected_era_rate(seasons, era_specs[rate])
        rel = n / (n + k)
        skill = raw - era
        sh_skill = rel * skill
        out[f"era_std_{rate}"] = skill
        out[f"era_std_sh_{rate}"] = sh_skill
        out[f"era_std_dev_{rate}"] = (
            sh_skill - df[f"era_skill_{rate}"].to_numpy(dtype=float))
    return pd.DataFrame(out, index=df.index)


def add_target_encoding(df, te_maps):
    out = {}
    for c in te_maps["cols"]:
        out[f"te_{c}"] = df[c].map(te_maps["maps"][c]).fillna(te_maps["global_"]).to_numpy()
    return pd.DataFrame(out, index=df.index)


def add_season_success_features(df, lookup, prior, k):
    """현재 행과 번들에 고정된 과거 시즌 종료 lookup만 사용한다."""
    specs = {
        "pitcher": ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
        "batter": ("batter_id", "asof_batter_n", "asof_batter_success_rate"),
    }
    out = {}
    for group, (entity, ncol, rate) in specs.items():
        ids = df[entity]
        pn = ids.map(lookup[group]["end_n"])
        pc = ids.map(lookup[group]["end_count"])
        known = pn.notna().to_numpy()
        prev_n = pn.fillna(0).to_numpy(dtype=float)
        prev_count = pc.fillna(0).to_numpy(dtype=float)
        current_n = df[ncol].to_numpy(dtype=float)
        current_count = np.rint(current_n * df[rate].to_numpy(dtype=float))
        delta_n, delta_count = current_n - prev_n, current_count - prev_count
        valid = (np.isfinite(delta_n) & (delta_n > 0) & np.isfinite(delta_count)
                 & (delta_count >= 0) & (delta_count <= delta_n))
        safe_n = np.where(valid, delta_n, 1.0)
        r = np.where(valid, delta_count / safe_n, np.nan)
        sh = np.where(valid, (delta_count + k * prior[rate]) / (safe_n + k), np.nan)
        out[f"std_{group}_known"] = known.astype(np.int8)
        out[f"std_{group}_invalid_n"] = (np.isfinite(delta_n) & (delta_n < 0)).astype(np.int8)
        out[f"std_{group}_n"] = np.where(valid, delta_n, np.nan)
        out[f"std_{group}_log_n"] = np.where(valid, np.log1p(delta_n), np.nan)
        out[f"std_{group}_rel_n"] = np.where(valid, delta_n / (delta_n + k), np.nan)
        out[f"std_{rate}"] = r
        out[f"std_sh_{rate}"] = sh
        out[f"std_dev_{rate}"] = sh - df[f"sh_{rate}"].to_numpy(dtype=float)
    return pd.DataFrame(out, index=df.index)


SEASON_COMMAND_RATES = [
    "asof_pitcher_strike_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
]
ABS_REGIME_RATES = [
    "asof_pitcher_success_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate", "asof_batter_success_rate",
]


def add_season_command_features(df, lookup, prior, k):
    """현재 행 하나와 학습 데이터에 고정된 pitcher command endpoint만 사용한다."""
    out = {}
    ids = df["pitcher_id"]
    current_n = df["asof_pitcher_n"].to_numpy(dtype=float)
    for rate in SEASON_COMMAND_RATES:
        prev_n = ids.map(lookup[rate]["end_n"]).fillna(0).to_numpy(dtype=float)
        prev_count = ids.map(lookup[rate]["end_count"]).fillna(0).to_numpy(dtype=float)
        delta_n = current_n - prev_n
        current_count = np.rint(current_n * df[rate].to_numpy(dtype=float))
        delta_count = current_count - prev_count
        valid = (np.isfinite(delta_n) & (delta_n > 0) & np.isfinite(delta_count)
                 & (delta_count >= 0) & (delta_count <= delta_n))
        safe_n = np.where(valid, delta_n, 1.0)
        clipped = np.clip(delta_count, 0.0, np.where(valid, delta_n, 0.0))
        season_rate = np.where(valid, clipped / safe_n, np.nan)
        shrunk = np.where(
            valid, (clipped + k * prior[rate]) / (safe_n + k), np.nan)
        out[f"std_{rate}"] = season_rate
        out[f"std_sh_{rate}"] = shrunk
        out[f"std_dev_{rate}"] = shrunk - df[f"sh_{rate}"].to_numpy(dtype=float)
    return pd.DataFrame(out, index=df.index)


def add_abs_regime_features(df, centers, k):
    """현재시즌 ABS 관측을 고정 중심, 신뢰 신호, 표본오차로 분해."""
    out = {}
    for rate in ABS_REGIME_RATES:
        center, scale = centers[rate]
        ncol = "std_batter_n" if rate.startswith("asof_batter") else "std_pitcher_n"
        n = df[ncol].to_numpy(dtype=float)
        raw = df[f"std_{rate}"].to_numpy(dtype=float)
        eb = df[f"std_sh_{rate}"].to_numpy(dtype=float)
        rel = n / (n + k)
        out[f"abs_centered_{rate}"] = (eb - center) / scale
        out[f"abs_signal_{rate}"] = rel * (eb - center) / scale
        out[f"abs_noise_{rate}"] = np.sqrt(
            np.clip(raw * (1.0 - raw) / np.maximum(n, 1.0), 0.0, None))
    out["abs_command_risk"] = (
        out["abs_centered_asof_pitcher_middle_rate"]
        + out["abs_centered_asof_pitcher_reverse_rate"]
        + out["abs_centered_asof_pitcher_ball_rate"]
        - out["abs_centered_asof_pitcher_strike_rate"])
    out["abs_command_signal"] = (
        out["abs_signal_asof_pitcher_middle_rate"]
        + out["abs_signal_asof_pitcher_reverse_rate"]
        + out["abs_signal_asof_pitcher_ball_rate"]
        - out["abs_signal_asof_pitcher_strike_rate"])
    return pd.DataFrame(out, index=df.index).astype("float32")


def apply_pitcher_chase_policy(df, preds, lookup, k_state=100.0,
                               k_current=50.0, w_max=0.20):
    """같은 투수의 최신 학습시즌 chase 정책과 현재행 누적값만 사용한다."""
    p0 = np.asarray(preds, dtype=float)
    ids = df["pitcher_id"]
    own_n = ids.map(lookup["own_n"]).to_numpy(dtype=float)
    own_rate = ids.map(lookup["own_rate"]).to_numpy(dtype=float)
    chase_n = ids.map(lookup["chase_n"]).to_numpy(dtype=float)
    chase_rate = ids.map(lookup["chase_rate"]).to_numpy(dtype=float)
    prev_n = ids.map(lookup["end_n"]).to_numpy(dtype=float)
    prev_count = ids.map(lookup["end_count"]).to_numpy(dtype=float)
    total_n = df["asof_pitcher_n"].to_numpy(dtype=float)
    total_rate = df["asof_pitcher_success_rate"].to_numpy(dtype=float)
    current_n = total_n - prev_n
    current_count = np.rint(total_n * total_rate) - prev_count
    current_valid = (np.isfinite(current_n) & (current_n > 0)
                     & np.isfinite(current_count) & (current_count >= 0)
                     & (current_count <= current_n))
    current_rate = np.where(
        current_valid, (current_count + k_current * own_rate)
        / (current_n + k_current), np.nan)
    valid = (df["strikes_before"].eq(2).to_numpy()
             & df["balls_before"].lt(3).to_numpy()
             & np.isfinite(own_n) & np.isfinite(own_rate)
             & np.isfinite(chase_n) & np.isfinite(chase_rate)
             & np.isfinite(current_rate))
    q_policy = np.clip(chase_rate, 1e-5, 1 - 1e-5)
    q_current = np.clip(current_rate, 1e-5, 1 - 1e-5)
    q_own = np.clip(own_rate, 1e-5, 1 - 1e-5)
    personal_logit = (np.log(q_policy / (1 - q_policy))
                      + np.log(q_current / (1 - q_current))
                      - np.log(q_own / (1 - q_own)))
    personal = 1.0 / (1.0 + np.exp(-np.clip(personal_logit, -30, 30)))
    reliability = np.where(
        valid,
        np.sqrt((own_n / (own_n + 200.0))
                * (chase_n / (chase_n + k_state))
                * (current_n / (current_n + k_current))), 0.0)
    weight = w_max * reliability
    return (1.0 - weight) * p0 + weight * np.where(valid, personal, p0)


def build_features(df, bundle):
    parts = [df.drop(columns=[ID_COL]), add_derived(df),
             add_shrinkage(df, bundle["prior"], bundle["k"])]
    base = pd.concat(parts, axis=1)
    if bundle.get("season_success_lookup"):
        parts.append(add_season_success_features(
            base, bundle["season_success_lookup"], bundle["prior"], bundle["k"]))
    if bundle.get("season_command_lookup"):
        parts.append(add_season_command_features(
            base, bundle["season_command_lookup"], bundle["prior"], bundle["k"]))
    if bundle.get("context_num_cols"):
        parts.append(add_pressure_ability(base, bundle["prior"]))
    if bundle.get("te_maps"):
        parts.append(add_target_encoding(df, bundle["te_maps"]))
    if bundle.get("era_specs"):
        parts.append(add_era_features(df, bundle["era_specs"], bundle["k"]))
        current = pd.concat(parts, axis=1)
        regime_current = add_regime_current_features(
            current, bundle["era_specs"], bundle["k"])
        if len(regime_current.columns):
            parts.append(regime_current)
    if bundle.get("catboost_members"):
        parts.append(add_catboost_context(df))
    if bundle.get("abs_regime_centers"):
        # success/command current-season 블록을 먼저 만든 뒤 같은 행에서만 계산한다.
        current = pd.concat(parts, axis=1)
        parts.append(add_abs_regime_features(
            current, bundle["abs_regime_centers"], bundle["k"]))
    X = pd.concat(parts, axis=1)
    required = list(bundle["cat_cols"]) + list(bundle["num_cols"])
    for col in bundle.get("catboost_feature_cols") or []:
        if col not in required:
            required.append(col)
    for col in bundle.get("catboost_command_feature_cols") or []:
        if col not in required:
            required.append(col)
    for col in bundle.get("abs_regime_feature_cols") or []:
        if col not in required:
            required.append(col)
    return X[required]


# =======================
# 데이터 로드 / 제출 파일 유틸
# =======================

def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼이 ({ID_COL}, {TARGET_COL})이 아님: "
                          f"{list(df.columns)}")
    return df


def merge_predictions(sub, ids, preds):
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        p = pred_map.get(rid)
        if p is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(p)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


# =======================
# main
# =======================

def main():
    TEST_DIR, MODEL_DIR, OUT_DIR = "./data", "./model", "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.csv")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "bundle.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    print("Load model bundle...")
    bundle = joblib.load(MODEL_PATH)
    members = bundle["members"]
    print(f" OK. members={len(members)} features={len(bundle['num_cols'])+len(bundle['cat_cols'])} "
          f"k={bundle['k']} logit_shift={bundle.get('logit_shift', 0.0):+.5f}")

    print("Load test data...")
    test = load_test(TEST_PATH)
    sub = load_sample_submission(SAMPLE_SUB_PATH)
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].tolist()
    X = build_features(test, bundle)
    print(f" features={X.shape[1]}")

    print(f"Inference ({len(members)} members)...")
    if len(X):
        member_num_cols = bundle.get("member_num_cols")
        if member_num_cols is None:  # v4 이하 번들 하위 호환
            member_num_cols = [bundle["num_cols"]] * len(members)
        if len(member_num_cols) != len(members):
            raise ValueError("bundle 계약 오류: members와 member_num_cols 길이 불일치")
        member_preds = [
            m.predict_proba(X[bundle["cat_cols"] + mcols])[:, 1]
            for m, mcols in zip(members, member_num_cols)
        ]
        family_weight = bundle.get("lgbm_family_weight")
        n_hgb = int(bundle.get("meta", {}).get("n_hgb", 0))
        n_lgbm = int(bundle.get("meta", {}).get("n_lgbm", 0))
        if family_weight is not None and n_hgb > 0 and n_lgbm > 0:
            family_weight = float(family_weight)
            if not (0.0 <= family_weight <= 1.0):
                raise ValueError("bundle 계약 오류: lgbm_family_weight는 [0, 1]이어야 함")
            if n_hgb + n_lgbm != len(member_preds):
                raise ValueError("bundle 계약 오류: family 멤버 수 합계 불일치")
            hgb_preds = np.mean(member_preds[:n_hgb], axis=0)
            lgbm_preds = np.mean(member_preds[n_hgb:], axis=0)
            base_preds = ((1.0 - family_weight) * hgb_preds
                          + family_weight * lgbm_preds)
        else:
            base_preds = np.mean(member_preds, axis=0)
        context_members = bundle.get("context_members") or []
        if context_members:
            context_num_cols = bundle.get("context_num_cols")
            if not context_num_cols:
                raise ValueError("bundle 계약 오류: context_members는 있으나 context_num_cols가 없음")
            context_weight = float(bundle.get("context_weight", 0.20))
            if not (0.0 <= context_weight <= 1.0):
                raise ValueError("bundle 계약 오류: context_weight는 [0, 1]이어야 함")
            context_preds = np.mean([
                m.predict_proba(X[bundle["cat_cols"] + context_num_cols])[:, 1]
                for m in context_members
            ], axis=0)
            preds = ((1.0 - context_weight) * base_preds
                     + context_weight * context_preds)
        else:
            preds = base_preds
        catboost_members = bundle.get("catboost_members") or []
        if catboost_members:
            catboost_feature_cols = bundle.get("catboost_feature_cols")
            if not catboost_feature_cols:
                raise ValueError("bundle 계약 오류: CatBoost 멤버는 있으나 feature 계약이 없음")
            catboost_weight = float(bundle.get("catboost_weight", 0.0))
            if not (0.0 <= catboost_weight <= 1.0):
                raise ValueError("bundle 계약 오류: catboost_weight는 [0, 1]이어야 함")
            catboost_preds = np.mean([
                m.predict_proba(X[catboost_feature_cols])[:, 1]
                for m in catboost_members
            ], axis=0)
            command_members = bundle.get("catboost_command_members") or []
            if command_members:
                command_cols = bundle.get("catboost_command_feature_cols")
                if not command_cols:
                    raise ValueError("bundle 계약 오류: command CatBoost feature 계약이 없음")
                command_weight = float(bundle.get("catboost_command_weight", 0.5))
                if not (0.0 <= command_weight <= 1.0):
                    raise ValueError("bundle 계약 오류: catboost_command_weight는 [0, 1]이어야 함")
                command_preds = np.mean([
                    m.predict_proba(X[command_cols])[:, 1]
                    for m in command_members
                ], axis=0)
                catboost_preds = ((1.0 - command_weight) * catboost_preds
                                  + command_weight * command_preds)
            preds = ((1.0 - catboost_weight) * preds
                     + catboost_weight * catboost_preds)
        abs_members = bundle.get("abs_regime_members") or []
        if abs_members:
            abs_cols = bundle.get("abs_regime_feature_cols")
            if not abs_cols:
                raise ValueError("bundle 계약 오류: ABS expert feature 계약이 없음")
            abs_weight = float(bundle.get("abs_regime_weight", 0.10))
            if not (0.0 <= abs_weight <= 1.0):
                raise ValueError("bundle 계약 오류: abs_regime_weight는 [0, 1]이어야 함")
            abs_preds = np.mean([
                m.predict_proba(X[abs_cols])[:, 1] for m in abs_members
            ], axis=0)
            preds = (1.0 - abs_weight) * preds + abs_weight * abs_preds
        policy_lookup = bundle.get("pitcher_chase_policy_lookup")
        if policy_lookup:
            preds = apply_pitcher_chase_policy(
                test, preds, policy_lookup,
                float(bundle.get("pitcher_chase_k_state", 100.0)),
                float(bundle.get("pitcher_chase_k_current", 50.0)),
                float(bundle.get("pitcher_chase_w_max", 0.20)))
        shift = float(bundle.get("logit_shift", 0.0))
        if shift != 0.0:
            # 행 단위 단조 변환 — 학습 데이터에서 구한 상수만 사용한다
            q = np.clip(preds, 1e-9, 1 - 1e-9)
            preds = 1.0 / (1.0 + np.exp(-(np.log(q / (1 - q)) + shift)))
        preds = np.clip(preds, 1e-6, 1 - 1e-6)
    else:
        preds = []
    print(f" preds={len(preds)}")

    print("Build submission...")
    sub = merge_predictions(sub, ids, preds)
    save_submission(OUT_PATH, sub)
    print(f"Saved: {OUT_PATH} (rows={len(sub)})")


if __name__ == "__main__":
    main()
