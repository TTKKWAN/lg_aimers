"""야구 inductive bias를 강제한 intent × execution OOF logit-offset 진단.

주의: 2026-08-14 사용자 피드백으로 투수 간 전역 execution 계수를 공유하는 이 설계는
실행 전에 보류했다. 현재 우선 실험은 ``diag_pitcher_personal_policy.py``다.

포수/투수의 실제 요구 코스는 보이지 않으므로 count에서 6개 coarse intent state를
정한다. 기존 v10 OOF 확률을 다시 학습하지 않고, 투수 실행 축이 intent에 따라
달라지는 저차원 correction만 ridge-IRLS로 학습한다. ID와 test 행간 통계는 쓰지 않는다.

로컬 실행 금지 규약상 기본 사용처는 Colab이다.
  python3 scripts/diag_intent_execution_offset.py
"""
import os

import numpy as np
import pandas as pd

DATA_DIR = "./open/data"
PRED_DIR = "./experiments/preds"
OUTER = [2023, 2024]
OOF_SEASONS = [2022, 2023, 2024]
LAMBDAS = [300.0, 1000.0, 3000.0, 10000.0, 30000.0]
SCALES = [0.25, 0.50, 0.75, 1.00]
K = 200.0

RATES = {
    "success": "asof_pitcher_success_rate",
    "strike": "asof_pitcher_strike_rate",
    "ball": "asof_pitcher_ball_rate",
    "middle": "asof_pitcher_middle_rate",
    "reverse": "asof_pitcher_reverse_rate",
}
PITCHMIX = [
    "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
RECENT = [
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
]
STATES = ["full", "must_strike", "chase", "pitcher_ahead", "batter_ahead", "neutral"]


def logit(p):
    q = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def intent_state(df):
    b = df["balls_before"].to_numpy()
    s = df["strikes_before"].to_numpy()
    return np.select(
        [(b == 3) & (s == 2), (b == 3) & (s < 2),
         (s == 2) & (b < 3), s > b, b > s],
        STATES[:-1], default="neutral")


def fit_priors(df):
    out = {}
    for name, col in RATES.items():
        out[name] = float(np.average(
            df[col].fillna(df[col].mean()),
            weights=np.maximum(df["asof_pitcher_n"].fillna(0), 1)))
    for col in PITCHMIX:
        out[col] = float(df[col].mean())
    return out


def raw_axes(df, priors):
    n = df["asof_pitcher_n"].fillna(0).to_numpy(dtype=float)
    rel = n / (n + K)
    out = {}
    for name, col in RATES.items():
        r = df[col].fillna(priors[name]).to_numpy(dtype=float)
        out[name] = (n * r + K * priors[name]) / (n + K)
    # 실행능력과 위치 선택을 분리한다. sign은 모델에 강제하지 않고 계층 축만 고정한다.
    out["edge_proxy"] = out["strike"] - out["middle"]
    out["miss_proxy"] = out["ball"] + out["reverse"]
    out["reliability"] = rel
    for col in PITCHMIX:
        out[col.replace("asof_pitcher_", "mix_")] = (
            rel * df[col].fillna(priors[col]).to_numpy(dtype=float)
            + (1 - rel) * priors[col])
    f = np.clip(out["mix_fastball_rate"], 1e-6, 1)
    br = np.clip(out["mix_breaking_rate"], 1e-6, 1)
    off = np.clip(out["mix_offspeed_rate"], 1e-6, 1)
    total = np.maximum(f + br + off, 1e-6)
    q = np.column_stack([f / total, br / total, off / total])
    out["mix_entropy"] = -(q * np.log(q)).sum(axis=1)
    career_success, career_middle = out["success"], out["middle"]
    for col in RECENT[:3]:
        out[col.replace("asof_pitcher_", "recent_")] = (
            df[col].fillna(career_success).to_numpy(dtype=float) - career_success)
    for col in RECENT[3:]:
        out[col.replace("asof_pitcher_", "recent_")] = (
            df[col].fillna(career_middle).to_numpy(dtype=float) - career_middle)
    return pd.DataFrame(out, index=df.index)


def fit_transform_spec(axes, train_mask):
    mu = axes.loc[train_mask].mean()
    sd = axes.loc[train_mask].std().clip(lower=1e-3)
    return {"mu": mu.to_dict(), "sd": sd.to_dict()}


def build_design(df, axes, spec, train_mask):
    names = list(axes.columns)
    a = np.column_stack([
        (axes[c].to_numpy(dtype=float) - spec["mu"][c]) / spec["sd"][c]
        for c in names])
    state = intent_state(df)
    train_state = state[train_mask]
    freq = {s: max(float(np.mean(train_state == s)), 1e-6) for s in STATES}
    blocks, columns, penalty_group = [], [], []

    # 전역 실행축: 카운트를 넘어 유지되는 command/control 성분.
    blocks.append(a)
    columns += [f"global:{c}" for c in names]
    penalty_group += [1.0] * len(names)

    # intent main effect는 v10에 이미 있으므로 강하게 축소한다.
    main = np.column_stack([(state == s).astype(float) - freq[s] for s in STATES])
    blocks.append(main)
    columns += [f"intent:{s}" for s in STATES]
    penalty_group += [4.0] * len(STATES)

    # 핵심 inductive bias: 같은 실행축도 요구 위치에 따라 다른 의미를 갖는다.
    execution = [names.index(c) for c in
                 ["success", "strike", "ball", "middle", "reverse",
                  "edge_proxy", "miss_proxy", "reliability"]]
    for s in STATES:
        centered = ((state == s).astype(float) - freq[s])[:, None]
        blocks.append(centered * a[:, execution])
        columns += [f"intent:{s}*{names[j]}" for j in execution]
        penalty_group += [6.0] * len(execution)

    # 구종 구성은 의도를 실행할 수 있는 수단이므로 더 강하게 축소한 상호작용만 허용.
    mix = [i for i, c in enumerate(names) if c.startswith("mix_")]
    for s in STATES:
        centered = ((state == s).astype(float) - freq[s])[:, None]
        blocks.append(centered * a[:, mix])
        columns += [f"arsenal:{s}*{names[j]}" for j in mix]
        penalty_group += [12.0] * len(mix)

    return np.column_stack(blocks).astype("float64"), columns, np.asarray(penalty_group)


def fit_offset_irls(z, y, offset, lam, penalty_group, max_iter=30):
    beta = np.zeros(z.shape[1], dtype=float)
    penalty = lam * penalty_group
    for _ in range(max_iter):
        eta = offset + z @ beta
        p = sigmoid(eta)
        w = np.clip(p * (1 - p), 1e-5, None)
        grad = z.T @ (y - p) - penalty * beta
        h = (z.T * w) @ z + np.diag(penalty)
        step = np.linalg.solve(h, grad)
        beta += step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def paired(y, p0, p1):
    base = y.mean() * (1 - y.mean())
    d = ((p0 - y) ** 2 - (p1 - y) ** 2) / base * 100000
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def load_oof(season):
    with np.load(f"{PRED_DIR}/catboost_command_confirm_{season}.npz") as z:
        return z["y"].astype(float), z["p_all4_r50"].astype(float)


def main():
    needed = ["season", "balls_before", "strikes_before", "asof_pitcher_n",
              "control_success"] + list(RATES.values()) + PITCHMIX + RECENT
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", usecols=needed)
    caches = {s: load_oof(s) for s in OOF_SEASONS}
    rows = []
    saved = {}
    for outer in OUTER:
        train_seasons = [s for s in OOF_SEASONS if s < outer]
        tr = raw["season"].isin(train_seasons).to_numpy()
        va = raw["season"].eq(outer).to_numpy()
        ytr = np.concatenate([caches[s][0] for s in train_seasons])
        ptr = np.concatenate([caches[s][1] for s in train_seasons])
        yv, p0 = caches[outer]
        if not np.array_equal(raw.loc[tr, "control_success"].to_numpy(float), ytr):
            raise RuntimeError(f"train cache order mismatch outer={outer}")
        if not np.array_equal(raw.loc[va, "control_success"].to_numpy(float), yv):
            raise RuntimeError(f"val cache order mismatch outer={outer}")
        priors = fit_priors(raw.loc[tr])
        axes = raw_axes(raw, priors)
        spec = fit_transform_spec(axes, tr)
        z, names, pg = build_design(raw, axes, spec, tr)
        log0_tr, log0_va = logit(ptr), logit(p0)
        for lam in LAMBDAS:
            beta = fit_offset_irls(z[tr], ytr, log0_tr, lam, pg)
            correction = z[va] @ beta
            for scale in SCALES:
                p = sigmoid(log0_va + scale * correction)
                gain, se = paired(yv, p0, p)
                rows.append(dict(outer=outer, train_seasons=str(train_seasons),
                                 lam=lam, scale=scale, gain=gain, se=se))
                print(f"outer={outer} lambda={lam:7.0f} scale={scale:.2f} "
                      f"gain={gain:+8.2f} SE={se:.2f}", flush=True)
            saved[f"beta_{outer}_{int(lam)}"] = beta
        saved[f"y_{outer}"] = yv
        saved[f"p0_{outer}"] = p0
        saved[f"columns_{outer}"] = np.asarray(names)
    out = pd.DataFrame(rows)
    os.makedirs(PRED_DIR, exist_ok=True)
    out.to_csv(f"{PRED_DIR}/intent_execution_offset_summary.csv", index=False)
    np.savez_compressed(f"{PRED_DIR}/intent_execution_offset.npz", **saved)
    print("\nGAIN TABLE")
    print(out.pivot_table(index=["lam", "scale"], columns="outer", values="gain")
          .round(2).to_string())


if __name__ == "__main__":
    main()
