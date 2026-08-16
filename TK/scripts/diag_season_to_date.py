"""누적 asof에서 current-season-to-date 상태를 복원해 스크리닝한다.

현재 행의 누적 ``n * rate``에서, 학습 fold로만 고정한 해당 선수의 과거 시즌
종료 누적 스냅샷을 뺀다. 검증/test의 다른 행은 참조하지 않으므로 추론 시에는
현재 행 하나 + train-fixed lookup만 필요하다.

단계(모두 v4 fixed-EB 입력에 add-only):
  baseline -> pitcher success/workload -> pitcher outcomes -> pitchmix -> batter

quick은 LightGBM 1개 스크린이다. 기존 동일 설정 baseline 예측이 있으면 재사용한다.
production/submit 경로는 수정하지 않는다.

사용법:
  python3 scripts/diag_season_to_date.py
"""
import gc
import os
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                       add_shrinkage, shrinkage_cols, fit_prior,
                       make_lgbm_model, bss)

DATA_DIR = "./open/data"
PRED_DIR = "./experiments/preds"
VAL_SEASONS = [2022, 2023, 2024]
K = 50.0
SPEC = dict(seed=99, learning_rate=0.03, num_leaves=63,
            min_child_samples=30, colsample_bytree=0.8, subsample=0.8)

GROUPS = {
    "pitcher": dict(
        entity="pitcher_id", n="asof_pitcher_n",
        rates=["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
               "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
               "asof_pitcher_strike_rate"]),
    "pitchmix": dict(
        entity="pitcher_id", n="asof_pitcher_pitchmix_n",
        rates=["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
               "asof_pitcher_offspeed_rate"]),
    "batter": dict(
        entity="batter_id", n="asof_batter_n",
        rates=["asof_batter_success_rate", "asof_batter_middle_rate"]),
}


def log(*args):
    print(*args, flush=True)


def _endpoints(train_only, spec):
    """각 entity-season의 마지막 투구 직후 누적 스냅샷을 복원한다.

    최대 asof_n 행의 현재 target까지는 과거 시즌의 정답이므로 success numerator와
    end_n은 정확히 +1 할 수 있다. 다른 outcome/pitchmix numerator는 현재 투구의
    세부 유형이 없어 직전 값(오차 최대 1구)을 쓴다.
    """
    entity, ncol, rates = spec["entity"], spec["n"], spec["rates"]
    cols = [entity, "season", ncol, TARGET] + rates
    d = train_only[cols]
    valid = d[ncol].notna()
    idx = d.loc[valid].groupby([entity, "season"], sort=False)[ncol].idxmax()
    ep = d.loc[idx].copy()
    observed_n = ep[ncol].to_numpy(dtype=float)
    ep[f"end_{ncol}"] = observed_n + 1.0
    for rate in rates:
        # 제공 비율은 누적 정수 count / n을 충분한 자릿수로 저장하므로 반올림해
        # 부동소수점 오차를 제거한다.
        numerator = np.rint(observed_n * ep[rate].to_numpy(dtype=float))
        if rate.endswith("_success_rate"):
            numerator = numerator + ep[TARGET].to_numpy(dtype=float)
            ep[f"end_n_{rate}"] = observed_n + 1.0
        else:
            # 마지막 과거 투구의 세부 유형은 없으므로 numerator/denominator를 모두
            # 그 투구 직전으로 둔다. 결과 window가 과거 마지막 1구를 포함하지만
            # count/rate 자체는 정확하고, 검증 첫 행으로 역산하는 누수도 없다.
            ep[f"end_n_{rate}"] = observed_n
        ep[f"end_num_{rate}"] = numerator
    return ep


def _previous_lookup(raw, train_mask, spec):
    """각 row season보다 엄격히 과거인 최신 endpoint를 entity별로 매핑한다."""
    entity, ncol, rates = spec["entity"], spec["n"], spec["rates"]
    ep = _endpoints(raw.loc[train_mask], spec)
    out = pd.DataFrame(index=raw.index)
    out[f"prev_{ncol}"] = np.nan
    for rate in rates:
        out[f"prev_num_{rate}"] = np.nan
        out[f"prev_n_{rate}"] = np.nan

    for season in sorted(raw["season"].unique()):
        rows = raw["season"].eq(season)
        prior = ep.loc[ep["season"] < season].sort_values("season")
        prior = prior.drop_duplicates(entity, keep="last").set_index(entity)
        ids = raw.loc[rows, entity]
        out.loc[rows, f"prev_{ncol}"] = ids.map(prior[f"end_{ncol}"]).to_numpy()
        for rate in rates:
            out.loc[rows, f"prev_num_{rate}"] = ids.map(
                prior[f"end_num_{rate}"]).to_numpy()
            out.loc[rows, f"prev_n_{rate}"] = ids.map(
                prior[f"end_n_{rate}"]).to_numpy()
    return out


def _season_block(raw, previous, spec, prior, zero_baseline=False):
    """누적 count 차분으로 season-to-date rate와 EB/dev를 만든다."""
    group = next(k for k, v in GROUPS.items() if v is spec)
    ncol, rates = spec["n"], spec["rates"]
    current_n = raw[ncol].to_numpy(dtype=float)
    prev_n = previous[f"prev_{ncol}"].to_numpy(dtype=float)
    known = np.isfinite(prev_n)
    if zero_baseline:
        prev_n = np.where(known, prev_n, 0.0)
    baseline_available = known | zero_baseline
    delta_n = current_n - prev_n
    valid_n = baseline_available & np.isfinite(current_n) & (delta_n > 0)
    invalid_n = baseline_available & np.isfinite(current_n) & (delta_n < -1e-6)

    out = {
        f"std_{group}_known": known.astype(np.int8),
        f"std_{group}_invalid_n": invalid_n.astype(np.int8),
        f"std_{group}_n": np.where(valid_n, delta_n, np.nan),
        f"std_{group}_log_n": np.where(valid_n, np.log1p(delta_n), np.nan),
        f"std_{group}_rel_n": np.where(valid_n, delta_n / (delta_n + K), np.nan),
    }
    diag = dict(known=known, valid_n=valid_n, invalid_n=invalid_n,
                delta_n=delta_n, invalid_count=np.zeros(len(raw), dtype=bool),
                invalid_by_rate={})

    for rate in rates:
        current_rate = raw[rate].to_numpy(dtype=float)
        prev_count = previous[f"prev_num_{rate}"].to_numpy(dtype=float)
        prev_rate_n = previous[f"prev_n_{rate}"].to_numpy(dtype=float)
        if zero_baseline:
            prev_count = np.where(known, prev_count, 0.0)
            prev_rate_n = np.where(known, prev_rate_n, 0.0)
        rate_delta_n = current_n - prev_rate_n
        current_count = np.rint(current_n * current_rate)
        delta_count = current_count - prev_count
        tol = 1e-7
        valid_rate_n = baseline_available & np.isfinite(rate_delta_n) & (rate_delta_n > 0)
        invalid_count = (valid_rate_n & np.isfinite(delta_count)
                         & ((delta_count < -tol) | (delta_count > rate_delta_n + tol)))
        valid = (valid_rate_n & np.isfinite(current_rate) & np.isfinite(prev_count)
                 & np.isfinite(delta_count) & ~invalid_count)
        clipped_count = np.clip(delta_count, 0.0, np.where(valid_rate_n, rate_delta_n, 0.0))
        safe_n = np.where(valid, rate_delta_n, 1.0)
        season_rate = np.where(valid, clipped_count / safe_n, np.nan)
        sh = np.where(valid, (clipped_count + K * prior[rate]) / (safe_n + K), np.nan)
        # 기존 fixed-EB career representation과의 차이가 선수 내 변화량이다.
        career_col = f"sh_{rate}"
        out[f"std_{rate}"] = season_rate
        out[f"std_sh_{rate}"] = sh
        out[f"std_dev_{rate}"] = sh - raw[career_col].to_numpy(dtype=float)
        diag["invalid_count"] |= invalid_count
        diag["invalid_by_rate"][rate] = invalid_count

    return pd.DataFrame(out, index=raw.index).astype({
        c: "float32" for c in out if not c.endswith(("_known", "_invalid_n"))
    }), diag


def paired(y, p0, p1, base):
    d = ((p0 - y) ** 2 - (p1 - y) ** 2) / base * 100000
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    base_num = raw_num + list(DERIVED_COLS) + shrinkage_cols()
    os.makedirs(PRED_DIR, exist_ok=True)
    results = {}

    log("season-to-date quick screen: v4 fixed-EB + add-only; LGBM seed=99")
    log("snapshots are built from fold-train rows only; previous season is strict (< row season)")
    for val_season in VAL_SEASONS:
        tr_m = (seasons < val_season).to_numpy()
        va_m = (seasons == val_season).to_numpy()
        y_tr = y.loc[tr_m]
        yv = y.loc[va_m].to_numpy(dtype=float)
        prior = fit_prior(raw.loc[tr_m])
        sh = add_shrinkage(base_df, prior, K)
        X0 = pd.concat([base_df, sh], axis=1)

        blocks, diagnostics = {}, {}
        for name, spec in GROUPS.items():
            prev = _previous_lookup(raw, tr_m, spec)
            block, diag = _season_block(X0, prev, spec, prior)
            blocks[name], diagnostics[name] = block, diag
            del prev
        X = pd.concat([X0] + list(blocks.values()), axis=1)

        pitcher_cols = list(blocks["pitcher"].columns)
        pitchmix_cols = list(blocks["pitchmix"].columns)
        batter_cols = list(blocks["batter"].columns)
        success_cols = [c for c in pitcher_cols if c in {
            "std_pitcher_known", "std_pitcher_invalid_n", "std_pitcher_n",
            "std_pitcher_log_n", "std_pitcher_rel_n",
            "std_asof_pitcher_success_rate", "std_sh_asof_pitcher_success_rate",
            "std_dev_asof_pitcher_success_rate"}]
        outcome_cols = [c for c in pitcher_cols if c not in success_cols]
        configs = {
            "pitcher_success": success_cols,
            "pitcher_outcomes": success_cols + outcome_cols,
            "plus_pitchmix": success_cols + outcome_cols + pitchmix_cols,
            "plus_batter": success_cols + outcome_cols + pitchmix_cols + batter_cols,
        }

        log(f"\n[fold={val_season}] train={tr_m.sum():,} val={va_m.sum():,}")
        for name, diag in diagnostics.items():
            vm = va_m
            dn = diag["delta_n"][vm & diag["valid_n"]]
            q = np.quantile(dn, [0.1, 0.5, 0.9]) if len(dn) else [np.nan] * 3
            log(f"  diag {name:8s}: known={diag['known'][vm].mean():.3%} "
                f"valid_n={diag['valid_n'][vm].mean():.3%} "
                f"invalid_n={diag['invalid_n'][vm].mean():.4%} "
                f"invalid_count={diag['invalid_count'][vm].mean():.4%} "
                f"delta_n_q10/50/90={q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}")
            log("    invalid_by_rate=" + ", ".join(
                f"{rate.replace('asof_', '')}:{bad[vm].mean():.3%}"
                for rate, bad in diag["invalid_by_rate"].items()))

        # 이 baseline은 pressure quick screen과 동일 spec/피처다. 검증 후 재사용한다.
        cache = f"{PRED_DIR}/pressure_ability_{val_season}.npz"
        p_base = None
        if os.path.exists(cache):
            z = np.load(cache)
            if np.array_equal(z["y"], yv):
                p_base = z["p_baseline"]
                log("  baseline reused from", cache)
        if p_base is None:
            spec0 = dict(SPEC); seed = spec0.pop("seed")
            model = make_lgbm_model(base_num, seed=seed, **spec0)
            model.fit(X.loc[tr_m, CAT_COLS + base_num], y_tr)
            p_base = model.predict_proba(X.loc[va_m, CAT_COLS + base_num])[:, 1]
            del model; gc.collect()

        br0, sc0, base = bss(yv, p_base)
        results[("baseline", val_season)] = (br0, sc0, 0.0, 0.0)
        log(f"  {'baseline':20s} brier={br0:.8f} BSS={sc0:8.2f}")
        saved = dict(y=yv, p_baseline=p_base)
        for name, new_cols in configs.items():
            spec0 = dict(SPEC); seed = spec0.pop("seed")
            t = time.time()
            model = make_lgbm_model(base_num + new_cols, seed=seed, **spec0)
            cols = CAT_COLS + base_num + new_cols
            model.fit(X.loc[tr_m, cols], y_tr)
            p = model.predict_proba(X.loc[va_m, cols])[:, 1]
            br, sc, _ = bss(yv, p)
            gain, se = paired(yv, p_base, p, base)
            clipped = np.mean((p <= 1e-6) | (p >= 1 - 1e-6))
            results[(name, val_season)] = (br, sc, gain, se)
            saved[f"p_{name}"] = p
            log(f"  {name:20s} brier={br:.8f} BSS={sc:8.2f} "
                f"gain={gain:+7.2f} SE={se:.2f} pred=[{p.min():.5f},{p.max():.5f}] "
                f"clip={clipped:.4%} time={time.time()-t:.0f}s new={len(new_cols)}")
            del model; gc.collect()
        saved["config_names"] = np.asarray(list(configs))
        for name, cols in configs.items():
            saved[f"cols_{name}"] = np.asarray(cols)
        np.savez_compressed(f"{PRED_DIR}/season_to_date_quick_{val_season}.npz", **saved)
        del X, X0, sh, blocks, diagnostics
        gc.collect()

    log("\n" + "=" * 110)
    log("SUMMARY — baseline 대비 동일 행 paired BSS gain (양수=개선)")
    log("=" * 110)
    for name in ["baseline", "pitcher_success", "pitcher_outcomes",
                 "plus_pitchmix", "plus_batter"]:
        rows = [results[(name, s)] for s in VAL_SEASONS]
        log(f"{name:20s} Brier=" + "/".join(f"{r[0]:.8f}" for r in rows)
            + " BSS=" + "/".join(f"{r[1]:.2f}" for r in rows)
            + " gain=" + "/".join(f"{r[2]:+.2f}" for r in rows)
            + f" mean_gain={np.mean([r[2] for r in rows]):+.2f}"
            + " SE=" + "/".join(f"{r[3]:.2f}" for r in rows))


if __name__ == "__main__":
    main()
