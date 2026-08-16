"""공용 피처/모델/평가 파이프라인.

설계 원칙 (대회 규칙 §5 준수):
  모든 피처는 **행 단위(row-local)** 로 계산 가능해야 한다. 즉 test의 어떤 행도
  다른 test 행을 참조하지 않는다. 여기서 쓰는 "학습 데이터에서 구한 상수/룩업"
  (리그 평균 prior, pitcher_id별 타깃 인코딩 맵)은 train.csv에서만 만들어지고
  추론 시에는 고정된 값으로 적용되므로 규칙상 허용된다.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

ID = "row_id"
TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]

# (비율 컬럼, 그 비율의 표본 수 컬럼) — EB 축소에 사용
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

# 직전 N경기 비율 — 실제 표본 수가 없어 경기당 대략 투구 수를 유사표본수로 사용,
# 리그 평균이 아니라 "그 투수의 통산(축소된) 비율" 쪽으로 당긴다 (계층적 축소).
PREV_SPECS = [
    ("asof_pitcher_prev1_game_success_rate", 25, "asof_pitcher_success_rate"),
    ("asof_pitcher_prev3_game_success_rate", 75, "asof_pitcher_success_rate"),
    ("asof_pitcher_prev5_game_success_rate", 125, "asof_pitcher_success_rate"),
    ("asof_pitcher_prev1_game_middle_rate", 25, "asof_pitcher_middle_rate"),
    ("asof_pitcher_prev3_game_middle_rate", 75, "asof_pitcher_middle_rate"),
    ("asof_pitcher_prev5_game_middle_rate", 125, "asof_pitcher_middle_rate"),
]

TE_COLS = ["pitcher_id", "batter_id"]

# 시대(period) 효과를 분리할 장기 비율. 기존 고정 prior EB 대신 일부 모델에서만
# n/(n+k) * (rate - 해당 시즌 리그 중심)을 사용한다.
ERA_RATES = [r for r, _ in RATE_N_PAIRS]
ERA_SKILL_COLS = [f"era_skill_{r}" for r in ERA_RATES]


# ---------------------------------------------------------------- 파생 피처
def add_derived(df):
    """v2에서 쓰던 행 단위 파생 피처."""
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


DERIVED_COLS = ["count_diff", "is_two_strike", "is_three_ball", "is_full_count",
                 "risp", "platoon_match", "pitcher_command_gap",
                 "pitcher_recent_trend", "batter_pitcher_gap", "close_game"]


# --------------------------------------- CatBoost 선수×상황 명목형 category
CATBOOST_CAT_COLS = [
    "pitcher_id_cat", "batter_id_cat", "pitcher_team_id_cat",
    "batter_team_id_cat", "pitcher_hand_cat", "batter_hand_cat",
    "top_bottom_cat", "game_type_cat", "base_state_cat",
    "count_state_cat", "pressure_state_cat", "inning_bucket_cat",
]


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


# ------------------------------------------ 투수 능력 x 현재 압박 / context expert
# diag_pressure_context_confirm.py에서 fixed-EB forward 3폴드 모두 개선된 작은 보조
# 모델의 계약. 아래 값은 모두 현재 행과 학습 fold에서 고정한 EB prior만 사용한다.
PRESSURE_ABILITY_COLS = [
    "pa_full_x_success_skill", "pa_full_x_ball_skill",
    "pa_three_ball_x_ball_skill", "pa_risp_x_success_skill",
    "pa_risp_x_ball_skill", "pa_logli_x_success_skill",
    "pa_logli_x_ball_skill", "pa_late_x_success_skill",
    "pa_close_x_success_skill", "pa_logli_x_reliability",
    "pa_highli_x_recent3_dev",
]

CONTEXT_RAW_COLS = [
    "season", "game_month", "inning", "balls_before", "strikes_before",
    "outs_before", "score_diff_pitcher_team", "run_total_before",
    "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
    "home_win_expectancy", "away_win_expectancy", "li",
]
CONTEXT_EB_COLS = [
    "sh_asof_pitcher_success_rate", "sh_asof_pitcher_middle_rate",
    "sh_asof_pitcher_ball_rate", "sh_asof_pitcher_reverse_rate",
    "rel_asof_pitcher_n", "log_asof_pitcher_n",
    "sh_asof_pitcher_prev1_game_success_rate",
    "sh_asof_pitcher_prev3_game_success_rate",
    "sh_asof_pitcher_prev5_game_success_rate",
    "dev_asof_pitcher_prev1_game_success_rate",
    "dev_asof_pitcher_prev3_game_success_rate",
    "dev_asof_pitcher_prev5_game_success_rate",
]
CONTEXT_DERIVED_COLS = [
    "count_diff", "is_two_strike", "is_three_ball", "is_full_count", "risp",
    "pitcher_command_gap", "pitcher_recent_trend", "close_game",
]
CONTEXT_NUM_COLS = (CONTEXT_RAW_COLS + CONTEXT_EB_COLS
                    + CONTEXT_DERIVED_COLS + PRESSURE_ABILITY_COLS)


def add_pressure_ability(df, prior):
    """fixed-EB 능력과 현재 압박의 11개 row-local 상호작용."""
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


# ------------------------------------------------------- 경험적 베이즈 축소
def fit_prior(train_df):
    """리그 평균 prior — 표본 수로 가중한 모집단 비율. 학습 데이터에서만 계산."""
    prior = {}
    for rate, ncol in RATE_N_PAIRS:
        r, n = train_df[rate], train_df[ncol].astype(float)
        m = r.notna() & (n > 0)
        prior[rate] = float((r[m] * n[m]).sum() / n[m].sum())
    return prior


def add_shrinkage(df, prior, k):
    """(n*rate + k*prior) / (n+k). n이 작으면 리그 평균으로 당겨진다."""
    out = {}
    shrunk_cache = {}
    for rate, ncol in RATE_N_PAIRS:
        r = df[rate].to_numpy(dtype=float)
        n = df[ncol].to_numpy(dtype=float)
        p0 = prior[rate]
        rf = np.where(np.isnan(r), p0, r)
        nf = np.where(np.isnan(r), 0.0, n)
        sh = (nf * rf + k * p0) / (nf + k)
        out[f"sh_{rate}"] = sh
        shrunk_cache[rate] = sh

    for ncol in N_COLS:
        n = df[ncol].to_numpy(dtype=float)
        out[f"log_{ncol}"] = np.log1p(n)
        out[f"rel_{ncol}"] = n / (n + k)          # 신뢰도 0~1

    # 직전 N경기 -> 그 투수의 통산 축소값 쪽으로 계층적 축소
    for col, pseudo_n, base_rate in PREV_SPECS:
        r = df[col].to_numpy(dtype=float)
        base = shrunk_cache[base_rate]
        miss = np.isnan(r)
        rf = np.where(miss, base, r)
        nf = np.where(miss, 0.0, float(pseudo_n))
        sh = (nf * rf + k * base) / (nf + k)
        out[f"sh_{col}"] = sh
        out[f"miss_{col}"] = miss.astype(np.int8)
        out[f"dev_{col}"] = sh - base             # 통산 대비 최근 폼 편차

    return pd.DataFrame(out, index=df.index)


def shrinkage_cols(k=None):
    cols = [f"sh_{r}" for r, _ in RATE_N_PAIRS]
    for n in N_COLS:
        cols += [f"log_{n}", f"rel_{n}"]
    for col, _, _ in PREV_SPECS:
        cols += [f"sh_{col}", f"miss_{col}", f"dev_{col}"]
    return cols


# -------------------------------------- 투수별 2스트라이크 chase 개인 정책
def fit_pitcher_chase_policy_lookup(train_df, k_state=100.0):
    """최신 시즌에서 투수 자신의 chase 성향만 학습해 고정 lookup으로 만든다.

    다른 투수의 평균이나 유사도는 전혀 쓰지 않는다. chase 성공률은 오직 같은
    투수의 해당 시즌 전체 성공률 쪽으로 축소한다.
    """
    season = int(train_df["season"].max())
    h = train_df.loc[train_df["season"].eq(season)].copy()
    overall = h.groupby("pitcher_id")[TARGET].agg(["sum", "size"])
    overall["rate"] = overall["sum"] / overall["size"]
    chase_mask = h["strikes_before"].eq(2) & h["balls_before"].lt(3)
    chase = h.loc[chase_mask].groupby("pitcher_id")[TARGET].agg(["sum", "size"])
    chase = chase.join(overall["rate"].rename("own_rate"), how="left")
    chase["policy_rate"] = ((chase["sum"] + k_state * chase["own_rate"])
                            / (chase["size"] + k_state))

    ep = _season_success_endpoints(
        h, "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate")
    ep = ep.sort_values("asof_pitcher_n").drop_duplicates("pitcher_id", keep="last")
    return {
        "season": season,
        "end_n": dict(zip(ep["pitcher_id"], ep["end_n"])),
        "end_count": dict(zip(ep["pitcher_id"], ep["end_count"])),
        "own_n": overall["size"].to_dict(),
        "own_rate": overall["rate"].to_dict(),
        "chase_n": chase["size"].to_dict(),
        "chase_rate": chase["policy_rate"].to_dict(),
    }


def apply_pitcher_chase_policy(df, preds, lookup, k_state=100.0,
                               k_current=50.0, w_max=0.20):
    """행 하나의 2025 누적값으로 같은 투수의 chase 예측만 보수적으로 보정."""
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


# ----------------------------------------------------------- 시대보정 상대능력
def fit_era_prior(train_df):
    """X_train만으로 시즌별 rate 중심과 미래 외삽식을 고정한다."""
    seasons = sorted(int(s) for s in train_df["season"].unique())
    specs = {}
    for rate, ncol in RATE_N_PAIRS:
        observed = {}
        for season in seasons:
            d = train_df.loc[train_df["season"] == season, [rate, ncol]]
            mask = d[rate].notna() & (d[ncol] > 0)
            observed[season] = float(d.loc[mask, rate].mean())
        recent = seasons[-min(3, len(seasons)):]
        if len(recent) >= 2:
            slope, intercept = np.polyfit(
                np.asarray(recent, dtype=float),
                np.asarray([observed[s] for s in recent], dtype=float), 1)
        else:
            slope, intercept = 0.0, observed[recent[0]]
        specs[rate] = dict(observed=observed, slope=float(slope),
                           intercept=float(intercept))
    return specs


def expected_era_rate(seasons, spec):
    seasons = np.asarray(seasons, dtype=int)
    out = spec["intercept"] + spec["slope"] * seasons.astype(float)
    for season, value in spec["observed"].items():
        out[seasons == int(season)] = value
    return out


def add_era_features(df, era_specs, k):
    """리그의 시대 수준을 뺀, 신뢰도 보정 개인 상대능력."""
    out = {}
    for rate, ncol in RATE_N_PAIRS:
        r = df[rate].to_numpy(dtype=float)
        n = df[ncol].to_numpy(dtype=float)
        era = expected_era_rate(df["season"].to_numpy(), era_specs[rate])
        rf = np.where(np.isnan(r), era, r)
        nf = np.where(np.isnan(r), 0.0, n)
        out[f"era_skill_{rate}"] = (nf / (nf + k)) * (rf - era)
    return pd.DataFrame(out, index=df.index)


# ------------------------------------------------------------ 타깃 인코딩
class TargetEncoder:
    """스무딩된 타깃 인코딩. 학습 데이터로만 맵을 만들고 추론 시 룩업으로 적용."""

    def __init__(self, cols=TE_COLS, smooth=200.0):
        self.cols, self.smooth = cols, smooth

    def fit(self, df, y):
        self.global_ = float(y.mean())
        self.maps_ = {}
        for c in self.cols:
            g = y.groupby(df[c].to_numpy()).agg(["sum", "count"])
            enc = (g["sum"] + self.smooth * self.global_) / (g["count"] + self.smooth)
            self.maps_[c] = enc
        return self

    def transform(self, df):
        out = {}
        for c in self.cols:
            out[f"te_{c}"] = df[c].map(self.maps_[c]).fillna(self.global_).to_numpy()
        return pd.DataFrame(out, index=df.index)

    def fit_transform_oof(self, df, y, n_splits=5, seed=0):
        """학습 행은 자기 자신의 정답이 새지 않도록 OOF로 인코딩."""
        self.fit(df, y)
        out = {f"te_{c}": np.full(len(df), self.global_) for c in self.cols}
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        idx = np.arange(len(df))
        for tr, va in kf.split(idx):
            sub = TargetEncoder(self.cols, self.smooth).fit(df.iloc[tr], y.iloc[tr])
            enc = sub.transform(df.iloc[va])
            for c in self.cols:
                out[f"te_{c}"][va] = enc[f"te_{c}"].to_numpy()
        return pd.DataFrame(out, index=df.index)


TE_FEATURE_COLS = [f"te_{c}" for c in TE_COLS]


# ----------------------------------- 누적 asof -> current-season success/workload
SEASON_SUCCESS_SPECS = {
    "pitcher": ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate"),
    "batter": ("batter_id", "asof_batter_n", "asof_batter_success_rate"),
}


def season_success_cols():
    cols = []
    for group, (_, _, rate) in SEASON_SUCCESS_SPECS.items():
        cols += [f"std_{group}_known", f"std_{group}_invalid_n", f"std_{group}_n",
                 f"std_{group}_log_n", f"std_{group}_rel_n", f"std_{rate}",
                 f"std_sh_{rate}", f"std_dev_{rate}"]
    return cols


def _season_success_endpoints(train_df, entity, ncol, rate):
    d = train_df[[entity, "season", ncol, rate, TARGET]]
    idx = d.loc[d[ncol].notna()].groupby([entity, "season"], sort=False)[ncol].idxmax()
    ep = d.loc[idx].copy()
    n = ep[ncol].to_numpy(dtype=float)
    ep["end_n"] = n + 1.0
    rate_values = ep[rate].to_numpy(dtype=float)
    before_count = np.where(n == 0, 0.0, np.rint(n * rate_values))
    ep["end_count"] = before_count + ep[TARGET].to_numpy(dtype=float)
    return ep


def fit_season_success_lookup(train_df):
    """최신 학습 시즌 종료 누적값. 추론 번들에 고정하는 entity lookup."""
    lookups = {}
    for group, (entity, ncol, rate) in SEASON_SUCCESS_SPECS.items():
        ep = _season_success_endpoints(train_df, entity, ncol, rate)
        latest = ep.sort_values("season").drop_duplicates(entity, keep="last")
        lookups[group] = {
            "end_n": dict(zip(latest[entity], latest["end_n"])),
            "end_count": dict(zip(latest[entity], latest["end_count"])),
        }
    return lookups


def _add_season_success_from_previous(df, previous, prior, k):
    out = {}
    for group, (_, ncol, rate) in SEASON_SUCCESS_SPECS.items():
        prev_n, prev_count, known = previous[group]
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


def add_season_success_train_features(df, prior, k):
    """학습 각 행에는 그 season보다 엄격히 과거인 endpoint만 사용한다."""
    previous = {}
    for group, (entity, ncol, rate) in SEASON_SUCCESS_SPECS.items():
        ep = _season_success_endpoints(df, entity, ncol, rate)
        pn = np.zeros(len(df), dtype=float)
        pc = np.zeros(len(df), dtype=float)
        known = np.zeros(len(df), dtype=bool)
        for season in sorted(df["season"].unique()):
            rows = df["season"].eq(season)
            p = ep.loc[ep["season"] < season].sort_values("season")
            p = p.drop_duplicates(entity, keep="last").set_index(entity)
            ids = df.loc[rows, entity]
            mapped_n, mapped_c = ids.map(p["end_n"]), ids.map(p["end_count"])
            has = mapped_n.notna().to_numpy()
            pn[rows] = mapped_n.fillna(0).to_numpy()
            pc[rows] = mapped_c.fillna(0).to_numpy()
            known[rows] = has
        previous[group] = (pn, pc, known)
    return _add_season_success_from_previous(df, previous, prior, k)


def add_season_success_features(df, lookup, prior, k):
    """현재 행 + train-fixed lookup만으로 평가 season-to-date를 계산한다."""
    previous = {}
    for group, (entity, _, _) in SEASON_SUCCESS_SPECS.items():
        ids = df[entity]
        n = ids.map(lookup[group]["end_n"])
        c = ids.map(lookup[group]["end_count"])
        known = n.notna().to_numpy()
        previous[group] = (n.fillna(0).to_numpy(dtype=float),
                           c.fillna(0).to_numpy(dtype=float), known)
    return _add_season_success_from_previous(df, previous, prior, k)


# ------------------------ 누적 asof -> current-season pitcher command outcomes
SEASON_COMMAND_RATES = [
    "asof_pitcher_strike_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate",
]


def season_command_cols():
    """기존 success workload 블록에 덧붙이는 command rate 12개."""
    return [name for rate in SEASON_COMMAND_RATES
            for name in (f"std_{rate}", f"std_sh_{rate}", f"std_dev_{rate}")]


def _season_command_endpoints(train_df):
    """entity-season 마지막 행 직전의 command 누적 count를 보존한다.

    현재 투구의 command outcome은 target으로 복원할 수 없으므로 success와 달리
    마지막 행의 +1을 포함하지 않는다. 이는 forward 진단의 strict-past 계약과 같다.
    """
    cols = ["pitcher_id", "season", "asof_pitcher_n"] + SEASON_COMMAND_RATES
    d = train_df[cols]
    idx = d.loc[d["asof_pitcher_n"].notna()].groupby(
        ["pitcher_id", "season"], sort=False)["asof_pitcher_n"].idxmax()
    ep = d.loc[idx].copy()
    n = ep["asof_pitcher_n"].to_numpy(dtype=float)
    for rate in SEASON_COMMAND_RATES:
        ep[f"end_n_{rate}"] = n
        ep[f"end_count_{rate}"] = np.rint(n * ep[rate].to_numpy(dtype=float))
    return ep


def fit_season_command_lookup(train_df):
    """최신 학습 시즌 종료 command 누적값을 pitcher별로 고정한다."""
    ep = _season_command_endpoints(train_df)
    latest = ep.sort_values("season").drop_duplicates("pitcher_id", keep="last")
    return {rate: {
        "end_n": dict(zip(latest["pitcher_id"], latest[f"end_n_{rate}"])),
        "end_count": dict(zip(latest["pitcher_id"], latest[f"end_count_{rate}"])),
    } for rate in SEASON_COMMAND_RATES}


def _add_season_command_from_previous(df, previous, prior, k):
    out = {}
    current_n = df["asof_pitcher_n"].to_numpy(dtype=float)
    for rate in SEASON_COMMAND_RATES:
        prev_n, prev_count = previous[rate]
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


def add_season_command_train_features(df, prior, k):
    """학습 행에는 그 season보다 엄격히 과거인 pitcher endpoint만 사용한다."""
    ep = _season_command_endpoints(df)
    previous = {rate: (np.zeros(len(df)), np.zeros(len(df)))
                for rate in SEASON_COMMAND_RATES}
    for season in sorted(df["season"].unique()):
        rows = df["season"].eq(season)
        p = ep.loc[ep["season"] < season].sort_values("season")
        p = p.drop_duplicates("pitcher_id", keep="last").set_index("pitcher_id")
        ids = df.loc[rows, "pitcher_id"]
        for rate in SEASON_COMMAND_RATES:
            pn, pc = previous[rate]
            pn[rows] = ids.map(p[f"end_n_{rate}"]).fillna(0).to_numpy(dtype=float)
            pc[rows] = ids.map(p[f"end_count_{rate}"]).fillna(0).to_numpy(dtype=float)
    return _add_season_command_from_previous(df, previous, prior, k)


def add_season_command_features(df, lookup, prior, k):
    """현재 행 하나와 train-fixed lookup만으로 command profile을 계산한다."""
    ids = df["pitcher_id"]
    previous = {}
    for rate in SEASON_COMMAND_RATES:
        previous[rate] = (
            ids.map(lookup[rate]["end_n"]).fillna(0).to_numpy(dtype=float),
            ids.map(lookup[rate]["end_count"]).fillna(0).to_numpy(dtype=float),
        )
    return _add_season_command_from_previous(df, previous, prior, k)


# ---------------------------- ABS 관측체계: 측정 중심 / 표본 노이즈 / 선수 신호 분리
ABS_REGIME_RATES = [
    "asof_pitcher_success_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_ball_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate", "asof_batter_success_rate",
]


def fit_abs_regime_centers(df, mask):
    """성숙한 ABS 구간의 중앙값/IQR. 최종 번들에 고정하는 train-only 상수."""
    centers = {}
    for rate in ABS_REGIME_RATES:
        values = df.loc[mask, f"std_sh_{rate}"].to_numpy(dtype=float)
        center = float(np.nanmedian(values))
        scale = max(float(np.nanquantile(values, .75) - np.nanquantile(values, .25)), .01)
        centers[rate] = (center, scale)
    return centers


def abs_regime_cols():
    cols = []
    for rate in ABS_REGIME_RATES:
        cols += [f"abs_centered_{rate}", f"abs_signal_{rate}", f"abs_noise_{rate}"]
    return cols + ["abs_command_risk", "abs_command_signal"]


def add_abs_regime_features(df, centers, k):
    """row-local 현재시즌 관측을 지표 중심, 신뢰 신호, 표본오차로 분해."""
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


# ------------------------------------------------------------------- 모델
HGB_D = dict(learning_rate=0.03, max_leaf_nodes=63, min_samples_leaf=30,
              l2_regularization=1.0, max_iter=1500, early_stopping=True,
              validation_fraction=0.1, n_iter_no_change=25)


def make_model(num_cols, seed=42, **overrides):
    params = {**HGB_D, **overrides}
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])
    clf = HistGradientBoostingClassifier(loss="log_loss", random_state=seed, **params)
    return Pipeline([("pre", pre), ("clf", clf)])


LGBM_D = dict(learning_rate=0.03, num_leaves=63, min_child_samples=30,
              reg_lambda=1.0, n_estimators=600, subsample_freq=1)


def make_lgbm_model(num_cols, seed=42, **overrides):
    """HGB 이질앙상블에 섞을 LightGBM 멤버. early_stopping은 fit 시 eval_set으로 별도 처리."""
    import lightgbm as lgb
    params = {**LGBM_D, **overrides}
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])
    clf = lgb.LGBMClassifier(objective="binary", random_state=seed, verbosity=-1, **params)
    return Pipeline([("pre", pre), ("clf", clf)])


CONTEXT_LGBM_D = dict(learning_rate=0.03, num_leaves=31,
                      min_child_samples=100, colsample_bytree=0.85,
                      subsample=0.8, reg_lambda=2.0)


def make_context_lgbm_model(num_cols=None, seed=8049, **overrides):
    """압박 context expert용 작은 LGBM. 기본 컬럼 계약도 함께 고정한다."""
    if num_cols is None:
        num_cols = CONTEXT_NUM_COLS
    params = {**CONTEXT_LGBM_D, **overrides}
    return make_lgbm_model(num_cols, seed=seed, **params)


# -------------------------------------------------------------------- 평가
def bss(y, p):
    y = np.asarray(y, dtype=float)
    r = y.mean()
    base = r * (1 - r)
    brier = float(((p - y) ** 2).mean())
    return brier, max(0.0, 100000 * (1 - brier / base)), base


def season_weights(seasons, decay):
    """최신 시즌에 가중치를 더 준다. decay=1.0 이면 균등."""
    seasons = np.asarray(seasons, dtype=float)
    return decay ** (seasons.max() - seasons)
