"""저장된 ABS forward 예측 위에 투수별 chase policy를 겹쳐 중복을 진단한다."""
import numpy as np
import pandas as pd

import diag_pitcher_personal_policy as core


def main():
    cols = ["season", "game_month", "pitcher_id", "balls_before", "strikes_before",
            "asof_pitcher_n", "asof_pitcher_success_rate", "control_success"]
    raw = pd.read_csv(f"{core.DATA_DIR}/train.csv", usecols=cols)
    z = np.load(f"{core.PRED_DIR}/abs_regime_expert_windows.npz")
    rows = []
    for end, months in [(5, (6, 7)), (7, (8, 9))]:
        history = (raw["season"].eq(2024).to_numpy()
                   & raw["game_month"].between(3, end).to_numpy())
        val = (raw["season"].eq(2024).to_numpy()
               & raw["game_month"].isin(months).to_numpy())
        target = raw.loc[val]
        policy, state_n, own_rate, own_n = core.personal_profile(
            raw.loc[history], target, 100.0)
        current, current_n = core.current_own_rate(raw, val, own_rate, 50.0)
        personal = core.sigmoid(core.logit(policy) + core.logit(current)
                                - core.logit(own_rate))
        active = core.intent_state(target) == "chase"
        valid = (active & np.isfinite(personal) & np.isfinite(state_n)
                 & np.isfinite(own_n))
        reliability = np.where(
            valid,
            np.sqrt((own_n / (own_n + 200.0))
                    * (state_n / (state_n + 100.0))
                    * (current_n / (current_n + 50.0))), 0.0)
        y = z[f"y_{end}_all_abs"].astype(float)
        p0 = z[f"p0_{end}_all_abs"].astype(float)
        pe = z[f"pe_{end}_all_abs"].astype(float)
        p11 = .90 * p0 + .10 * pe
        for wm in (.10, .20, .30, .40):
            w = wm * reliability
            pp = (1 - w) * p11 + w * np.where(valid, personal, p11)
            gain, se = core.paired(y, p11, pp)
            rows.append(dict(train_end=end, val_months=str(months), w_max=wm,
                             gain=gain, se=se, mean_weight=float(w.mean()),
                             coverage=float(valid.mean())))
            print(f"M3-{end}->{months} wm={wm:.2f} gain={gain:+.2f} "
                  f"SE={se:.2f} mean_w={w.mean():.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(f"{core.PRED_DIR}/pitcher_policy_on_abs_summary.csv", index=False)


if __name__ == "__main__":
    main()
