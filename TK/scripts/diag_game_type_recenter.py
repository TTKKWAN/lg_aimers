"""stage2 all4-50 후보에 game_type별 train-fixed 재중심화를 진단한다."""
import numpy as np
import pandas as pd

from pipeline import TARGET
from diag_season_to_date import DATA_DIR, PRED_DIR, VAL_SEASONS, paired


def solve_shift(p, target):
    q = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    logits = np.log(q / (1 - q))
    lo, hi = -6.0, 6.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(logits + mid)))) < target:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def apply_shift(p, shift):
    q = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    return 1 / (1 + np.exp(-(np.log(q / (1 - q)) + shift)))


def group_target(raw, train_mask, group, val_season, mode):
    d = raw.loc[train_mask & raw["game_type"].eq(group),
                ["season", TARGET]]
    rates = d.groupby("season")[TARGET].mean()
    if mode == "last":
        return float(rates.iloc[-1])
    recent = rates.iloc[-min(3, len(rates)):]
    if len(recent) < 2:
        return float(recent.iloc[-1])
    slope, intercept = np.polyfit(recent.index.to_numpy(float),
                                  recent.to_numpy(float), 1)
    return float(intercept + slope * val_season)


def main():
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=["season", "game_type", TARGET])
    rows = []
    for val_season in VAL_SEASONS:
        tr = raw["season"].lt(val_season).to_numpy()
        va = raw["season"].eq(val_season).to_numpy()
        last_season = int(raw.loc[tr, "season"].max())
        last_mask = raw["season"].eq(last_season).to_numpy()
        gv = raw.loc[va, "game_type"].to_numpy()
        gl = raw.loc[last_mask, "game_type"].to_numpy()

        fam = np.load(f"{PRED_DIR}/current_season_family_{val_season}.npz")
        cat = np.load(f"{PRED_DIR}/catboost_confirm_{val_season}.npz")
        command = np.load(f"{PRED_DIR}/catboost_command_confirm_{val_season}.npz")
        p_v8 = 0.25 * fam["p_hgb8"] + 0.75 * fam["p_current16"]
        l_v8 = 0.25 * fam["last_hgb8"] + 0.75 * fam["last_current16"]
        p_cat = 0.5 * cat["p_expert2"] + 0.5 * command["p_all4_expert2"]
        l_cat = 0.5 * cat["last_expert2"] + 0.5 * command["last_all4_expert2"]
        p_raw = 0.4 * p_v8 + 0.6 * p_cat
        l_raw = 0.4 * l_v8 + 0.6 * l_cat
        ref = command["p_all4_r50"]
        yv = command["y"].astype(float)
        base = yv.mean() * (1 - yv.mean())

        for mode in ("last", "linear"):
            for fraction in (0.25, 0.5):
                pred = np.empty_like(p_raw)
                shifts = {}
                targets = {}
                for group in ("F", "R"):
                    last_g = gl == group
                    val_g = gv == group
                    natural = float(l_raw[last_g].mean())
                    target0 = group_target(raw, tr, group, val_season, mode)
                    target = natural + fraction * (target0 - natural)
                    shift = solve_shift(l_raw[last_g], target)
                    pred[val_g] = apply_shift(p_raw[val_g], shift)
                    shifts[group], targets[group] = shift, target0
                gain, se = paired(yv, ref, pred, base)
                rows.append(dict(season=val_season, mode=mode, fraction=fraction,
                                 gain=gain, se=se, shift_F=shifts["F"],
                                 shift_R=shifts["R"], target_F=targets["F"],
                                 target_R=targets["R"]))

    result = pd.DataFrame(rows)
    result.to_csv(f"{PRED_DIR}/game_type_recenter_summary.csv", index=False)
    print("game_type-specific recenter vs stage2 global recenter_f=0.5", flush=True)
    for mode in ("last", "linear"):
        for fraction in (0.25, 0.5):
            d = result[result["mode"].eq(mode) & result["fraction"].eq(fraction)]
            print(f"{mode:6s} f={fraction:.2f} gain="
                  + "/".join(f"{x:+.2f}" for x in d["gain"])
                  + " SE=" + "/".join(f"{x:.2f}" for x in d["se"]), flush=True)


if __name__ == "__main__":
    main()
