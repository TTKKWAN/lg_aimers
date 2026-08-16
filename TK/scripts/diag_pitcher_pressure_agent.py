"""투수별 pressure-response dossier를 strict-past 방식으로 검증한다.

pitcher_id는 개인 반응 lookup의 키로만 사용한다. 각 투수-시즌 평균을 제거한 잔차에서
상황 안/밖 차이를 구하고, 리그 공통 상황효과를 뺀 뒤 effective sample size로 강하게
축소한다. 학습 행과 검증 행 모두 해당 season보다 엄격히 과거인 시즌만 참조한다.

최종 비교는 저장된 production 동일 HGB8 OOF와 family weight LGBM=0.75,
fold별 recenter_f=0.5까지 포함한다.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                       add_shrinkage, shrinkage_cols, fit_prior,
                       make_lgbm_model, season_success_cols,
                       add_season_success_train_features)
from diag_season_to_date import DATA_DIR, PRED_DIR, VAL_SEASONS, paired
from diag_current_season_family import recenter_like_production

K = 50
FAMILY_WEIGHT = 0.75
LGBM_SPECS = [
    dict(seed=99, learning_rate=0.03, num_leaves=63, min_child_samples=30,
         colsample_bytree=0.8, subsample=0.8),
    dict(seed=2718, learning_rate=0.05, num_leaves=31, min_child_samples=50,
         colsample_bytree=0.7, subsample=0.7),
    dict(seed=31415, learning_rate=0.02, num_leaves=127, min_child_samples=20,
         colsample_bytree=0.9, subsample=0.9),
]


def log(*args):
    print(*args, flush=True)


def context_frame(df):
    full = (df["balls_before"].eq(3) & df["strikes_before"].eq(2))
    risp = df["runner_on_2b"].eq(1) | df["runner_on_3b"].eq(1)
    close = df["score_diff_pitcher_team"].abs().le(1)
    late = df["inning"].ge(7)
    return pd.DataFrame({
        "full": full,
        "three_ball": df["balls_before"].eq(3),
        "two_strike": df["strikes_before"].eq(2),
        "risp": risp,
        "high_li": df["li"].ge(2.0),
        "close": close,
        "trailing": df["score_diff_pitcher_team"].lt(0),
        "late_close": late & close,
        "two_out_risp": df["outs_before"].eq(2) & risp,
        "bases_loaded": (df["runner_on_1b"].eq(1) & df["runner_on_2b"].eq(1)
                         & df["runner_on_3b"].eq(1)),
    }, index=df.index).astype(np.int8)


CORE = ["full", "three_ball", "risp", "high_li", "close"]
EXTENDED = list(context_frame(pd.DataFrame({
    "balls_before": [0], "strikes_before": [0], "runner_on_1b": [0],
    "runner_on_2b": [0], "runner_on_3b": [0], "score_diff_pitcher_team": [0],
    "inning": [1], "li": [0.0], "outs_before": [0],
})).columns)


def pressure_features(raw, allowed_mask, names, shrink_k, history_seasons=None):
    """각 행 season보다 엄격히 과거인 선수별 상황 반응을 고정 lookup으로 적용."""
    ctx = context_frame(raw)
    out = pd.DataFrame(index=raw.index)
    for name in names:
        out[f"agent_{name}_profile"] = 0.0
        out[f"agent_{name}_rel"] = 0.0
        out[f"agent_{name}_active"] = 0.0

    for target_season in sorted(raw["season"].unique()):
        target = raw["season"].eq(target_season)
        source = allowed_mask & raw["season"].lt(target_season).to_numpy()
        if history_seasons is not None:
            source &= raw["season"].ge(target_season - history_seasons).to_numpy()
        if not source.any():
            continue
        d = raw.loc[source, ["pitcher_id", "season", TARGET]].copy()
        # 시즌 수준과 해당 시즌의 투수 기본 능력을 제거한다. 모두 과거 시즌 정답이다.
        d["resid"] = d[TARGET] - d.groupby(["pitcher_id", "season"])[TARGET].transform("mean")
        ids = raw.loc[target, "pitcher_id"]
        for name in names:
            flag = ctx.loc[source, name].to_numpy(dtype=bool)
            tmp = d[["pitcher_id", "resid"]].copy()
            tmp["flag"] = flag
            agg = tmp.groupby(["pitcher_id", "flag"])["resid"].agg(["sum", "count"])
            wide_sum = agg["sum"].unstack(fill_value=0.0)
            wide_n = agg["count"].unstack(fill_value=0.0)
            for col in (False, True):
                if col not in wide_sum:
                    wide_sum[col] = 0.0
                    wide_n[col] = 0.0
            m0 = wide_sum[False] / wide_n[False].replace(0, np.nan)
            m1 = wide_sum[True] / wide_n[True].replace(0, np.nan)
            delta = (m1 - m0).fillna(0.0)
            n0, n1 = wide_n[False], wide_n[True]
            neff = (n0 * n1 / (n0 + n1).replace(0, np.nan)).fillna(0.0)
            # 리그 공통 상황효과는 raw context가 담당하므로 개인 편차만 남긴다.
            g0 = float(tmp.loc[~tmp["flag"], "resid"].mean())
            g1 = float(tmp.loc[tmp["flag"], "resid"].mean())
            global_delta = 0.0 if not np.isfinite(g0 + g1) else g1 - g0
            rel = neff / (neff + shrink_k)
            profile = rel * (delta - global_delta)
            mapped_profile = ids.map(profile).fillna(0.0).to_numpy(dtype=float)
            mapped_rel = ids.map(rel).fillna(0.0).to_numpy(dtype=float)
            active = ctx.loc[target, name].to_numpy(dtype=float)
            out.loc[target, f"agent_{name}_profile"] = mapped_profile
            out.loc[target, f"agent_{name}_rel"] = mapped_rel
            out.loc[target, f"agent_{name}_active"] = mapped_profile * active
    return out.astype("float32")


def predict_lgbm3(X, y, train_mask, val_mask, last_mask, num_cols):
    pv, pl = [], []
    cols = CAT_COLS + num_cols
    for i, spec0 in enumerate(LGBM_SPECS):
        spec = dict(spec0); seed = spec.pop("seed")
        t = time.time()
        model = make_lgbm_model(num_cols, seed=seed, **spec)
        model.fit(X.loc[train_mask, cols], y.loc[train_mask])
        pv.append(model.predict_proba(X.loc[val_mask, cols])[:, 1])
        pl.append(model.predict_proba(X.loc[last_mask, cols])[:, 1])
        log(f"    member={i+1} seed={seed} {time.time()-t:.0f}s")
        del model; gc.collect()
    return np.mean(pv, axis=0), np.mean(pl, axis=0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "allpast"
    if mode not in ("allpast", "last1"):
        raise SystemExit("mode must be allpast or last1")
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    base = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    base_num = raw_num + list(DERIVED_COLS) + shrinkage_cols()
    season_cols = season_success_cols()
    results = {}

    for val_season in VAL_SEASONS:
        train_mask = seasons.lt(val_season).to_numpy()
        val_mask = seasons.eq(val_season).to_numpy()
        last_season = int(seasons.loc[train_mask].max())
        last_mask = train_mask & seasons.eq(last_season).to_numpy()
        prior = fit_prior(raw.loc[train_mask])
        sh = add_shrinkage(base, prior, K)
        X0 = pd.concat([base, sh], axis=1)
        train_for_season = pd.concat([X0, y.rename(TARGET)], axis=1)
        std = add_season_success_train_features(train_for_season, prior, K)
        X0 = pd.concat([X0, std], axis=1)

        if mode == "allpast":
            specs = [
                ("core_k200", CORE, 200.0, False, None),
                ("extended_k200", EXTENDED, 200.0, False, None),
                ("extended_k500", EXTENDED, 500.0, False, None),
                ("extended_k200_no_pid", EXTENDED, 200.0, True, None),
            ]
        else:
            specs = [("extended_last1_k500_no_pid", EXTENDED, 500.0, True, 1)]
        configs = {}
        for label, names, sk, drop_id, history in specs:
            pf = pressure_features(raw, train_mask, names, sk, history)
            cols = list(pf.columns)
            num = base_num + season_cols + cols
            if drop_id:
                num = [c for c in num if c != "pitcher_id"]
            configs[label] = (pf, num)

        cache = np.load(f"{PRED_DIR}/current_season_family_{val_season}.npz")
        yv = cache["y"]
        p_hgb, last_hgb = cache["p_hgb8"], cache["last_hgb8"]
        p_cur, last_cur = cache["p_current16"], cache["last_current16"]
        r_extrap = float(cache["r_extrap"])
        p0 = (1-FAMILY_WEIGHT)*p_hgb + FAMILY_WEIGHT*p_cur
        l0 = (1-FAMILY_WEIGHT)*last_hgb + FAMILY_WEIGHT*last_cur
        p0r = recenter_like_production(p0, l0, r_extrap, .5)[0]
        base_brier = yv.mean() * (1-yv.mean())

        log(f"\n[fold={val_season}] train={train_mask.sum():,} val={val_mask.sum():,}")
        saved = dict(y=yv, p_baseline=p0r)
        for label, (pf, num) in configs.items():
            log(f"  {label} agent_features={len(pf.columns)} model_num={len(num)}")
            X = pd.concat([X0, pf], axis=1)
            pv, pl = predict_lgbm3(X, y, train_mask, val_mask, last_mask, num)
            p = (1-FAMILY_WEIGHT)*p_hgb + FAMILY_WEIGHT*pv
            last = (1-FAMILY_WEIGHT)*last_hgb + FAMILY_WEIGHT*pl
            pr = recenter_like_production(p, last, r_extrap, .5)[0]
            gain, se = paired(yv, p0r, pr, base_brier)
            results[(label, val_season)] = (gain, se)
            saved[f"p_{label}"] = pr
            log(f"    recentered gain={gain:+.2f} SE={se:.2f}")
            del X, pv, pl, p, last, pr
        np.savez_compressed(
            f"{PRED_DIR}/pitcher_pressure_agent_{mode}_{val_season}.npz", **saved)
        del X0, sh, std, configs
        gc.collect()

    log("\n" + "="*100)
    log("SUMMARY — v8 current16 HGB25/LGBM75 + fold recenter_f=0.5 대비")
    labels = (["core_k200", "extended_k200", "extended_k500",
               "extended_k200_no_pid"] if mode == "allpast"
              else ["extended_last1_k500_no_pid"])
    for label in labels:
        rows = [results[(label, s)] for s in VAL_SEASONS]
        log(f"{label:24s} gain=" + "/".join(f"{r[0]:+.2f}" for r in rows)
            + f" mean={np.mean([r[0] for r in rows]):+.2f} SE="
            + "/".join(f"{r[1]:.2f}" for r in rows))


if __name__ == "__main__":
    main()
