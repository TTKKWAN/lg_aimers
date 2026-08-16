"""개인 intent 중 실제 전이성이 확인된 chase/full만 제한적으로 검증."""
import numpy as np
import pandas as pd

import diag_pitcher_personal_policy as core

K_STATES = [100.0, 200.0, 400.0]
K_CURRENT = [25.0, 50.0, 100.0]
W_MAX = [0.10, 0.20, 0.30, 0.40, 0.60]
ACTIVE = {"chase": {"chase"}, "chase_full": {"chase", "full"}}


def evaluate(raw, history_mask, val_mask, p0, yv, label, results):
    history, rows = raw.loc[history_mask], raw.loc[val_mask]
    row_state = core.intent_state(rows)
    for ks in K_STATES:
        policy, state_n, own_rate, own_n = core.personal_profile(history, rows, ks)
        for kc in K_CURRENT:
            current, current_n = core.current_own_rate(raw, val_mask, own_rate, kc)
            personal = core.sigmoid(core.logit(policy) + core.logit(current) - core.logit(own_rate))
            valid0 = np.isfinite(personal) & np.isfinite(state_n) & np.isfinite(own_n)
            rel0 = np.where(
                valid0,
                np.sqrt((own_n / (own_n + 200.0))
                        * (state_n / (state_n + ks))
                        * (current_n / (current_n + kc))), 0.0)
            for mode, states in ACTIVE.items():
                valid = valid0 & np.isin(row_state, list(states))
                rel = np.where(valid, rel0, 0.0)
                for wm in W_MAX:
                    w = wm * rel
                    pred = (1 - w) * p0 + w * np.where(valid, personal, p0)
                    gain, se = core.paired(yv, p0, pred)
                    results.append(dict(split=label, mode=mode, k_state=ks,
                                        k_current=kc, w_max=wm, gain=gain, se=se,
                                        mean_weight=float(w.mean()),
                                        coverage=float(valid.mean())))
                    print(f"{label:18s} {mode:10s} ks={ks:3.0f} kc={kc:3.0f} "
                          f"wm={wm:.2f} gain={gain:+7.2f} SE={se:.2f} "
                          f"w={w.mean():.4f} cover={valid.mean():.2%}", flush=True)


def main():
    cols = ["season", "game_month", "pitcher_id", "balls_before", "strikes_before",
            "asof_pitcher_n", "asof_pitcher_success_rate", "control_success"]
    raw = pd.read_csv(f"{core.DATA_DIR}/train.csv", usecols=cols)
    results = []
    for season in core.FOLDS:
        val = raw["season"].eq(season).to_numpy()
        hist = raw["season"].eq(season - 1).to_numpy()
        yv, p0 = core.cache(season)
        evaluate(raw, hist, val, p0, yv, f"prev->{season}", results)
    y24, p24 = core.cache(2024)
    idx24 = raw.index[raw["season"].eq(2024)]
    p24 = pd.Series(p24, index=idx24)
    for start, end, months in [(3, 5, (6, 7)), (3, 7, (8, 9)),
                               (5, 7, (8, 9)), (5, 8, (9,))]:
        hist = (raw["season"].eq(2024).to_numpy()
                & raw["game_month"].between(start, end).to_numpy())
        val = (raw["season"].eq(2024).to_numpy()
               & raw["game_month"].isin(months).to_numpy())
        idx = raw.index[val]
        evaluate(raw, hist, val, p24.loc[idx].to_numpy(),
                 raw.loc[val, "control_success"].to_numpy(float),
                 f"M{start}-{end}->{months}", results)
    out = pd.DataFrame(results)
    out.to_csv(f"{core.PRED_DIR}/pitcher_personal_policy_followup_summary.csv", index=False)
    pivot = out.pivot_table(index=["mode", "k_state", "k_current", "w_max"],
                            columns="split", values="gain")
    pivot["min_gain"] = pivot.min(axis=1)
    pivot["mean_gain"] = pivot.drop(columns="min_gain").mean(axis=1)
    print("\nROBUST CONFIGS")
    print(pivot.sort_values(["min_gain", "mean_gain"], ascending=False).head(30).round(2))


if __name__ == "__main__":
    main()
