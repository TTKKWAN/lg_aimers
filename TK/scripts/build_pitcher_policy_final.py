"""v11 위에 투수별 chase 정책 lookup을 추가한 v12 후보 번들을 만든다."""
import os

import joblib
import pandas as pd

from pipeline import TARGET, fit_pitcher_chase_policy_lookup

SOURCE = "./open/baseline_submit/model/bundle.pkl"
OUTPUT = "./open/baseline_submit/model/bundle_v12_pitcher_chase_candidate.pkl"
K_STATE = 100.0
K_CURRENT = 50.0
W_MAX = 0.20


def main():
    bundle = joblib.load(SOURCE)
    if bundle.get("meta", {}).get("version") != "v11_abs_regime_10":
        raise RuntimeError("source bundle is not v11_abs_regime_10")
    cols = ["season", "pitcher_id", "balls_before", "strikes_before",
            "asof_pitcher_n", "asof_pitcher_success_rate", TARGET]
    raw = pd.read_csv("./open/data/train.csv", usecols=cols)
    lookup = fit_pitcher_chase_policy_lookup(raw, K_STATE)
    bundle["pitcher_chase_policy_lookup"] = lookup
    bundle["pitcher_chase_k_state"] = K_STATE
    bundle["pitcher_chase_k_current"] = K_CURRENT
    bundle["pitcher_chase_w_max"] = W_MAX
    bundle["meta"] = dict(
        bundle["meta"], version="v12_pitcher_chase_policy_20",
        pitcher_chase_history_season=lookup["season"],
        pitcher_chase_pitchers=len(lookup["chase_rate"]),
        pitcher_chase_k_state=K_STATE,
        pitcher_chase_k_current=K_CURRENT,
        pitcher_chase_w_max=W_MAX,
    )
    temp = OUTPUT + ".building"
    joblib.dump(bundle, temp, compress=3)
    check = joblib.load(temp)
    if not check.get("pitcher_chase_policy_lookup"):
        raise RuntimeError("v12 serialization check failed")
    os.replace(temp, OUTPUT)
    print(f"saved {OUTPUT} ({os.path.getsize(OUTPUT)/1e6:.1f} MB), "
          f"pitchers={len(lookup['chase_rate'])}")


if __name__ == "__main__":
    main()
