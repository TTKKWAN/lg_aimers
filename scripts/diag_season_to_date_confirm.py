"""정확 season-to-date pitcher+batter success/workload를 LGBM3로 확인한다.

저장된 fixed-EB HGB5/LGBM3 OOF를 재사용해, 기존 LGBM3를 새 season LGBM3로
0/25/50/75/100% 교체했을 때 전체 5:3 앙상블의 paired gain을 평가한다.
"""
import gc
import os
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                       add_shrinkage, shrinkage_cols, fit_prior,
                       make_lgbm_model, bss)
from diag_season_to_date import (DATA_DIR, PRED_DIR, VAL_SEASONS, K, GROUPS,
                                 _previous_lookup, _season_block, paired)
from diag_season_to_date_followup import exact_cols

SPECS = [
    dict(seed=99, learning_rate=0.03, num_leaves=63, min_child_samples=30,
         colsample_bytree=0.8, subsample=0.8),
    dict(seed=2718, learning_rate=0.05, num_leaves=31, min_child_samples=50,
         colsample_bytree=0.7, subsample=0.7),
    dict(seed=31415, learning_rate=0.02, num_leaves=127, min_child_samples=20,
         colsample_bytree=0.9, subsample=0.9),
]


def log(*args):
    print(*args, flush=True)


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    base_num = raw_num + list(DERIVED_COLS) + shrinkage_cols()
    results = {}

    for val_season in VAL_SEASONS:
        tr_m = (seasons < val_season).to_numpy()
        va_m = (seasons == val_season).to_numpy()
        y_tr, yv = y.loc[tr_m], y.loc[va_m].to_numpy(dtype=float)
        prior = fit_prior(raw.loc[tr_m])
        sh = add_shrinkage(base_df, prior, K)
        X0 = pd.concat([base_df, sh], axis=1)
        pprev = _previous_lookup(raw, tr_m, GROUPS["pitcher"])
        bprev = _previous_lookup(raw, tr_m, GROUPS["batter"])
        pb, _ = _season_block(X0, pprev, GROUPS["pitcher"], prior, True)
        bb, _ = _season_block(X0, bprev, GROUPS["batter"], prior, True)
        pc = exact_cols(pb, "pitcher", "asof_pitcher_success_rate")
        bc = exact_cols(bb, "batter", "asof_batter_success_rate")
        new_cols = pc + bc
        X = pd.concat([X0, pb, bb], axis=1)
        cols = CAT_COLS + base_num + new_cols

        ps = []
        log(f"\n[fold={val_season}] train={tr_m.sum():,} val={va_m.sum():,} new={len(new_cols)}")
        for i, spec0 in enumerate(SPECS):
            spec = dict(spec0); seed = spec.pop("seed")
            t = time.time()
            model = make_lgbm_model(base_num + new_cols, seed=seed, **spec)
            model.fit(X.loc[tr_m, cols], y_tr)
            p = model.predict_proba(X.loc[va_m, cols])[:, 1]
            ps.append(p)
            _, sc, _ = bss(yv, p)
            log(f"  member={i+1} seed={seed} BSS={sc:.2f} time={time.time()-t:.0f}s")
            del model
            gc.collect()
        p_season = np.mean(ps, axis=0)

        zh = np.load(f"{PRED_DIR}/ens_hgb_{val_season}.npz")
        zl = np.load(f"{PRED_DIR}/ens_lgbm_{val_season}.npz")
        if not (np.array_equal(zh["y"], yv) and np.array_equal(zl["y"], yv)):
            raise RuntimeError("saved ensemble y mismatch")
        p_hgb, p_lgbm = zh["p"], zl["p"]
        p_base = (5.0 * p_hgb + 3.0 * p_lgbm) / 8.0
        _, sc_l, base = bss(yv, p_lgbm)
        br_s, sc_s, _ = bss(yv, p_season)
        gl, sel = paired(yv, p_lgbm, p_season, base)
        log(f"  LGBM3 original BSS={sc_l:.2f}; season Brier={br_s:.8f} "
            f"BSS={sc_s:.2f}; direct gain={gl:+.2f} SE={sel:.2f}")

        for w in [0.0, 0.25, 0.50, 0.75, 1.0]:
            p_lmix = (1.0 - w) * p_lgbm + w * p_season
            p = (5.0 * p_hgb + 3.0 * p_lmix) / 8.0
            br, sc, full_base = bss(yv, p)
            gain, se = paired(yv, p_base, p, full_base)
            results[(w, val_season)] = (br, sc, gain, se)
            log(f"  season_lgbm_weight={w:.2f} full Brier={br:.8f} BSS={sc:.2f} "
                f"gain={gain:+.2f} SE={se:.2f}")
        np.savez_compressed(
            f"{PRED_DIR}/season_to_date_confirm_{val_season}.npz",
            y=yv, p_season_lgbm3=p_season, p_old_lgbm3=p_lgbm,
            p_hgb5=p_hgb, new_cols=np.asarray(new_cols), n=3)
        del X, X0, sh, pb, bb, pprev, bprev, ps
        gc.collect()

    log("\n" + "=" * 105)
    log("SUMMARY — HGB5 + LGBM3 전체 앙상블, old LGBM -> season LGBM 교체 비중")
    for w in [0.0, 0.25, 0.50, 0.75, 1.0]:
        rows = [results[(w, s)] for s in VAL_SEASONS]
        log(f"weight={w:.2f} Brier=" + "/".join(f"{r[0]:.8f}" for r in rows)
            + " gain=" + "/".join(f"{r[2]:+.2f}" for r in rows)
            + f" mean_gain={np.mean([r[2] for r in rows]):+.2f}"
            + " SE=" + "/".join(f"{r[3]:.2f}" for r in rows))


if __name__ == "__main__":
    main()
