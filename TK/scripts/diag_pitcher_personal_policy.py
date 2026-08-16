"""투수끼리 연결하지 않는 개인별 intent-policy + current execution 진단.

각 투수의 count-intent 성공률은 그 투수 자신의 전체 성공률로만 축소한다.
현재 시즌 변화도 해당 투수의 직전 시즌과만 비교한다. 다른 투수 평균/embedding/ID
유사성은 쓰지 않으며, cold start는 개인 보정 weight=0으로 기존 모델에 돌아간다.

Colab 실행:
  python3 scripts/diag_pitcher_personal_policy.py
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = "./open/data"
PRED_DIR = "./experiments/preds"
FOLDS = [2023, 2024]
K_STATES = [25.0, 50.0, 100.0]
K_CURRENT = [25.0, 50.0, 100.0]
W_MAX = [0.10, 0.20, 0.30, 0.40]
STATES = ["full", "must_strike", "chase", "pitcher_ahead", "batter_ahead", "neutral"]


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def logit(p):
    q = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    return np.log(q / (1 - q))


def intent_state(df):
    b = df["balls_before"].to_numpy()
    s = df["strikes_before"].to_numpy()
    return np.select(
        [(b == 3) & (s == 2), (b == 3) & (s < 2),
         (s == 2) & (b < 3), s > b, b > s],
        STATES[:-1], default="neutral")


def endpoints(df):
    """각 투수-시즌 마지막 투구 직후 success 누적 endpoint."""
    d = df[["pitcher_id", "season", "asof_pitcher_n",
            "asof_pitcher_success_rate", "control_success"]]
    valid = d["asof_pitcher_n"].notna()
    idx = d.loc[valid].groupby(
        ["pitcher_id", "season"], sort=False)["asof_pitcher_n"].idxmax()
    ep = d.loc[idx].copy()
    n = ep["asof_pitcher_n"].to_numpy(dtype=float)
    ep["end_n"] = n + 1.0
    ep["end_success"] = (
        np.rint(n * ep["asof_pitcher_success_rate"].to_numpy(dtype=float))
        + ep["control_success"].to_numpy(dtype=float))
    ep["end_rate"] = ep["end_success"] / ep["end_n"]
    return ep


def previous_endpoint_for_rows(raw, row_mask):
    ep = endpoints(raw)
    rows = raw.loc[row_mask]
    out = pd.DataFrame(index=rows.index, columns=["prev_n", "prev_success", "prev_rate"],
                       dtype=float)
    for season in sorted(rows["season"].unique()):
        m = rows["season"].eq(season)
        previous = ep.loc[ep["season"] < season].sort_values("season")
        previous = previous.drop_duplicates("pitcher_id", keep="last").set_index("pitcher_id")
        ids = rows.loc[m, "pitcher_id"]
        out.loc[m, "prev_n"] = ids.map(previous["end_n"]).to_numpy()
        out.loc[m, "prev_success"] = ids.map(previous["end_success"]).to_numpy()
        out.loc[m, "prev_rate"] = ids.map(previous["end_rate"]).to_numpy()
    return out


def personal_profile(history, target_rows, k_state):
    """history의 각 투수 상태율을 오직 같은 투수 전체율 쪽으로 축소."""
    h = history.copy()
    h["intent"] = intent_state(h)
    overall = h.groupby("pitcher_id")["control_success"].agg(["sum", "size"])
    overall["rate"] = overall["sum"] / overall["size"]
    state = h.groupby(["pitcher_id", "intent"])["control_success"].agg(["sum", "size"])
    state = state.join(overall["rate"].rename("own_rate"), on="pitcher_id")
    state["policy_rate"] = (
        state["sum"] + k_state * state["own_rate"]) / (state["size"] + k_state)
    keys = pd.MultiIndex.from_arrays(
        [target_rows["pitcher_id"].to_numpy(), intent_state(target_rows)])
    policy = pd.Series(keys.map(state["policy_rate"]), index=target_rows.index, dtype=float)
    state_n = pd.Series(keys.map(state["size"]), index=target_rows.index, dtype=float)
    own_rate = target_rows["pitcher_id"].map(overall["rate"]).astype(float)
    own_n = target_rows["pitcher_id"].map(overall["size"]).astype(float)
    return policy, state_n, own_rate, own_n


def current_own_rate(raw, row_mask, own_prior, k_current):
    """현재행 누적에서 직전 endpoint를 빼고 같은 투수 prior와 결합."""
    rows = raw.loc[row_mask]
    prev = previous_endpoint_for_rows(raw, row_mask)
    n = rows["asof_pitcher_n"].to_numpy(dtype=float)
    count = np.rint(n * rows["asof_pitcher_success_rate"].to_numpy(dtype=float))
    dn = n - prev["prev_n"].to_numpy(dtype=float)
    dc = count - prev["prev_success"].to_numpy(dtype=float)
    valid = np.isfinite(dn) & (dn > 0) & np.isfinite(dc) & (dc >= 0) & (dc <= dn)
    prior = own_prior.to_numpy(dtype=float)
    posterior = np.where(valid, (dc + k_current * prior) / (dn + k_current), np.nan)
    return posterior, np.where(valid, dn, 0.0)


def paired(y, p0, p1):
    base = y.mean() * (1 - y.mean())
    d = ((p0 - y) ** 2 - (p1 - y) ** 2) / base * 100000
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def cache(season):
    with np.load(f"{PRED_DIR}/catboost_command_confirm_{season}.npz") as z:
        return z["y"].astype(float), z["p_all4_r50"].astype(float)


def evaluate(raw, history_mask, val_mask, p0, yv, label, results):
    history = raw.loc[history_mask]
    rows = raw.loc[val_mask]
    for ks in K_STATES:
        policy, state_n, own_rate, own_n = personal_profile(history, rows, ks)
        for kc in K_CURRENT:
            current, current_n = current_own_rate(raw, val_mask, own_rate, kc)
            # 개인별 상태 policy에 같은 투수의 current overall 변화만 이식한다.
            personal = sigmoid(logit(policy) + logit(current) - logit(own_rate))
            valid = np.isfinite(personal) & np.isfinite(state_n) & np.isfinite(own_n)
            reliability = np.where(
                valid,
                np.sqrt((own_n / (own_n + 200.0))
                        * (state_n / (state_n + ks))
                        * (current_n / (current_n + kc))),
                0.0)
            for wm in W_MAX:
                w = wm * reliability
                pred = (1.0 - w) * p0 + w * np.where(valid, personal, p0)
                gain, se = paired(yv, p0, pred)
                results.append(dict(split=label, k_state=ks, k_current=kc,
                                    w_max=wm, gain=gain, se=se,
                                    mean_weight=float(w.mean()),
                                    coverage=float(valid.mean())))
                print(f"{label:18s} ks={ks:3.0f} kc={kc:3.0f} wm={wm:.2f} "
                      f"gain={gain:+8.2f} SE={se:.2f} "
                      f"w={w.mean():.3f} cover={valid.mean():.3%}", flush=True)


def main():
    cols = ["season", "game_month", "pitcher_id", "balls_before", "strikes_before",
            "asof_pitcher_n", "asof_pitcher_success_rate", "control_success"]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=cols)
    results = []
    # 완결된 직전 시즌 개인 policy -> 다음 시즌. 2024 fold는 ABS 단절 스트레스 테스트.
    for season in FOLDS:
        val = raw["season"].eq(season).to_numpy()
        hist = raw["season"].eq(season - 1).to_numpy()
        yv, p0 = cache(season)
        if not np.array_equal(raw.loc[val, "control_success"].to_numpy(float), yv):
            raise RuntimeError(f"cache mismatch {season}")
        evaluate(raw, hist, val, p0, yv, f"prev->{season}", results)

    # 2025 production과 동일한 ABS 체계 내 가용성: 앞선 2024월 policy -> 후기 월.
    y24, p24 = cache(2024)
    idx24 = raw.index[raw["season"].eq(2024)]
    p24_series = pd.Series(p24, index=idx24)
    for end, months in [(5, (6, 7)), (7, (8, 9))]:
        hist = raw["season"].eq(2024).to_numpy() & (raw["game_month"].to_numpy() <= end)
        val = raw["season"].eq(2024).to_numpy() & raw["game_month"].isin(months).to_numpy()
        rows = raw.index[val]
        evaluate(raw, hist, val, p24_series.loc[rows].to_numpy(),
                 raw.loc[val, "control_success"].to_numpy(float),
                 f"2024M{end}->{months}", results)

    out = pd.DataFrame(results)
    os.makedirs(PRED_DIR, exist_ok=True)
    out.to_csv(f"{PRED_DIR}/pitcher_personal_policy_summary.csv", index=False)
    print("\nMEAN GAIN BY CONFIG")
    print(out.groupby(["k_state", "k_current", "w_max"]).gain.mean()
          .sort_values(ascending=False).head(30).to_string())


if __name__ == "__main__":
    main()
