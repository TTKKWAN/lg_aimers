"""season-to-date quick screen의 정확 복원 후보만 좁혀 재비교한다.

비교:
  - 이전 시즌 lookup이 있는 행만 쓰는 pitcher success/workload (기존 quick 예측)
  - 신규 ID의 정확한 zero baseline까지 살린 pitcher success/workload
  - 위 각각에 exact batter success/workload 추가

신규 ID는 각 entity의 데이터 최초 행 asof_n=0임을 별도 감사로 확인했다. 따라서
lookup 부재 시 zero baseline은 추측이 아니라 제공 누적값의 시작 계약을 복원한다.
"""
import gc
import os
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                       add_shrinkage, shrinkage_cols, fit_prior,
                       make_lgbm_model, bss)
from diag_season_to_date import (DATA_DIR, PRED_DIR, VAL_SEASONS, K, SPEC,
                                 GROUPS, _previous_lookup, _season_block, paired)


def log(*args):
    print(*args, flush=True)


def exact_cols(block, group, rate):
    return [c for c in block.columns if c in {
        f"std_{group}_known", f"std_{group}_invalid_n", f"std_{group}_n",
        f"std_{group}_log_n", f"std_{group}_rel_n", f"std_{rate}",
        f"std_sh_{rate}", f"std_dev_{rate}"}]


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    base_num = raw_num + list(DERIVED_COLS) + shrinkage_cols()
    os.makedirs(PRED_DIR, exist_ok=True)
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
        p_known, _ = _season_block(X0, pprev, GROUPS["pitcher"], prior, False)
        p_zero, _ = _season_block(X0, pprev, GROUPS["pitcher"], prior, True)
        b_known, _ = _season_block(X0, bprev, GROUPS["batter"], prior, False)
        b_zero, _ = _season_block(X0, bprev, GROUPS["batter"], prior, True)
        pc = exact_cols(p_known, "pitcher", "asof_pitcher_success_rate")
        bc = exact_cols(b_known, "batter", "asof_batter_success_rate")
        # 동일 이름의 zero/known block을 한 X에 둘 수 없어 config별 X를 명시한다.
        configs = {
            "pitcher_zero": (pd.concat([X0, p_zero], axis=1), pc),
            "pitcher_known_batter_known": (pd.concat([X0, p_known, b_known], axis=1), pc + bc),
            "pitcher_zero_batter_zero": (pd.concat([X0, p_zero, b_zero], axis=1), pc + bc),
        }
        z = np.load(f"{PRED_DIR}/season_to_date_quick_{val_season}.npz")
        if not np.array_equal(z["y"], yv):
            raise RuntimeError("cached quick y mismatch")
        p_base, p_pk = z["p_baseline"], z["p_pitcher_success"]
        br0, sc0, base = bss(yv, p_base)
        brk, sck, _ = bss(yv, p_pk)
        gk, sek = paired(yv, p_base, p_pk, base)
        results[("baseline", val_season)] = (br0, sc0, 0., 0.)
        results[("pitcher_known", val_season)] = (brk, sck, gk, sek)
        saved = dict(y=yv, p_baseline=p_base, p_pitcher_known=p_pk)
        log(f"\n[fold={val_season}] baseline={sc0:.2f} pitcher_known gain={gk:+.2f}")
        log(f"  zero fallback rows pitcher={p_known['std_pitcher_known'][va_m].eq(0).mean():.3%} "
            f"batter={b_known['std_batter_known'][va_m].eq(0).mean():.3%}")
        for name, (X, new_cols) in configs.items():
            spec = dict(SPEC); seed = spec.pop("seed")
            t = time.time()
            model = make_lgbm_model(base_num + new_cols, seed=seed, **spec)
            cols = CAT_COLS + base_num + new_cols
            model.fit(X.loc[tr_m, cols], y_tr)
            p = model.predict_proba(X.loc[va_m, cols])[:, 1]
            br, sc, _ = bss(yv, p)
            gain, se = paired(yv, p_base, p, base)
            results[(name, val_season)] = (br, sc, gain, se)
            saved[f"p_{name}"] = p
            log(f"  {name:28s} brier={br:.8f} BSS={sc:8.2f} "
                f"gain={gain:+7.2f} SE={se:.2f} time={time.time()-t:.0f}s")
            del model, X
            gc.collect()
        np.savez_compressed(f"{PRED_DIR}/season_to_date_followup_{val_season}.npz", **saved)
        del X0, sh, pprev, bprev, p_known, p_zero, b_known, b_zero, configs
        gc.collect()

    log("\n" + "=" * 105)
    for name in ["baseline", "pitcher_known", "pitcher_zero",
                 "pitcher_known_batter_known", "pitcher_zero_batter_zero"]:
        rows = [results[(name, s)] for s in VAL_SEASONS]
        log(f"{name:28s} Brier=" + "/".join(f"{r[0]:.8f}" for r in rows)
            + " gain=" + "/".join(f"{r[2]:+.2f}" for r in rows)
            + f" mean_gain={np.mean([r[2] for r in rows]):+.2f}"
            + " SE=" + "/".join(f"{r[3]:.2f}" for r in rows))


if __name__ == "__main__":
    main()
