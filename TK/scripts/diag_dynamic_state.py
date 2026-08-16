"""선수별 동적 잠재 제구 상태를 v8 production 기준으로 검증한다.

멀티시즌 endpoint에서 공통 AR(1) 유지율과 선수별 변화분산을 경험적 베이즈로
추정한다. 현재 행의 season-to-date 성공률은 관측분산에 따라 Kalman gain 형태로
반영한다. 학습 행은 해당 season보다 엄격히 과거인 endpoint만 사용하며,
검증 행끼리는 서로 참조하지 않는다.

기준선은 diag_current_season_family.py가 저장한 production 동일 HGB8/current16
LGBM3의 25:75 family 혼합 + fold별 recenter_f=0.5다.
"""
import gc
import os
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived,
                      add_shrinkage, shrinkage_cols, fit_prior,
                      make_lgbm_model)
from diag_season_to_date import (DATA_DIR, PRED_DIR, VAL_SEASONS, K, GROUPS,
                                 _endpoints, _previous_lookup, _season_block,
                                 paired)
from diag_season_to_date_followup import exact_cols
from diag_current_season_family import (LGBM_SPECS, recenter_like_production,
                                        extrapolated_rate)


FAMILY_WEIGHT = 0.75
VAR_PRIOR_DF = 5.0
EPS = 1e-6


def log(*args):
    print(*args, flush=True)


def dynamic_cols(group):
    return [
        f"dyn_{group}_known", f"dyn_{group}_phi",
        f"dyn_{group}_prior_mean", f"dyn_{group}_volatility",
        f"dyn_{group}_gain", f"dyn_{group}_posterior_mean",
        f"dyn_{group}_posterior_sd", f"dyn_{group}_shock_z",
    ]


def fit_dynamic_lookup(history, spec, prior_rate):
    """과거 시즌 endpoint만으로 AR 유지율, 변화분산, 최신 상태를 고정한다."""
    entity, rate = spec["entity"], spec["rates"][0]
    if history.empty:
        return dict(phi=0.5, q0=0.0025, rows=pd.DataFrame())
    ep = _endpoints(history, spec).sort_values([entity, "season"]).copy()
    n = ep[f"end_n_{rate}"].to_numpy(dtype=float)
    count = ep[f"end_num_{rate}"].to_numpy(dtype=float)
    ep["state"] = (count + K * prior_rate) / (n + K)
    ep["dev"] = ep["state"] - prior_rate
    ep["prev_dev"] = ep.groupby(entity, sort=False)["dev"].shift(1)
    pairs = ep.dropna(subset=["prev_dev"])
    denom = float(np.square(pairs["prev_dev"]).sum())
    phi = (float((pairs["prev_dev"] * pairs["dev"]).sum()) / denom
           if denom > 1e-12 else 0.5)
    phi = float(np.clip(phi, 0.0, 1.0))
    ep["resid2"] = np.square(ep["dev"] - phi * ep["prev_dev"])
    resid = ep["resid2"].dropna().to_numpy(dtype=float)
    q0 = float(np.mean(resid)) if len(resid) else 0.0025
    q0 = float(np.clip(q0, 1e-5, 0.02))
    agg = ep.groupby(entity, sort=False)["resid2"].agg(["sum", "count"])
    agg["q"] = (agg["sum"].fillna(0.0) + VAR_PRIOR_DF * q0) / (
        agg["count"] + VAR_PRIOR_DF)
    latest = ep.drop_duplicates(entity, keep="last").set_index(entity)
    rows = latest[["state"]].join(agg[["q"]])
    rows["n"] = latest[f"end_n_{rate}"]
    return dict(phi=phi, q0=q0, rows=rows)


def apply_dynamic(block, ids, lookup, prior_rate, group):
    """train-fixed state와 현재 행 season-to-date 관측을 결합한다."""
    rows = lookup["rows"]
    if len(rows):
        last = ids.map(rows["state"])
        q = ids.map(rows["q"])
        last_n = ids.map(rows["n"])
    else:
        last = pd.Series(np.nan, index=ids.index)
        q = pd.Series(np.nan, index=ids.index)
        last_n = pd.Series(np.nan, index=ids.index)
    known = last.notna().to_numpy()
    last_v = last.fillna(prior_rate).to_numpy(dtype=float)
    q_v = q.fillna(lookup["q0"]).to_numpy(dtype=float)
    n_prev = last_n.fillna(0.0).to_numpy(dtype=float)
    phi = float(lookup["phi"])
    prior_mean = prior_rate + phi * (last_v - prior_rate)
    prev_uncertainty = np.clip(last_v * (1.0 - last_v) / (n_prev + K + 1.0),
                               1e-6, 0.02)
    prior_var = np.clip(q_v + prev_uncertainty, 1e-6, 0.04)

    n = block[f"std_{group}_n"].to_numpy(dtype=float)
    obs = block[f"std_{spec_rate(group)}"].to_numpy(dtype=float)
    valid = np.isfinite(n) & (n > 0) & np.isfinite(obs)
    obs_safe = np.clip(np.where(valid, obs, prior_mean), EPS, 1.0 - EPS)
    obs_var = np.clip(obs_safe * (1.0 - obs_safe) / (np.where(valid, n, 0.0) + 1.0),
                      1e-6, 0.05)
    gain = np.where(valid, prior_var / (prior_var + obs_var), 0.0)
    posterior = np.clip(prior_mean + gain * (obs_safe - prior_mean), EPS, 1.0 - EPS)
    posterior_var = np.clip((1.0 - gain) * prior_var, 1e-7, 0.04)
    shock = np.where(valid, (obs_safe - prior_mean) /
                     np.sqrt(prior_var + obs_var), 0.0)
    out = {
        f"dyn_{group}_known": known.astype(np.int8),
        f"dyn_{group}_phi": np.full(len(block), phi, dtype=np.float32),
        f"dyn_{group}_prior_mean": prior_mean,
        f"dyn_{group}_volatility": np.sqrt(q_v),
        f"dyn_{group}_gain": gain,
        f"dyn_{group}_posterior_mean": posterior,
        f"dyn_{group}_posterior_sd": np.sqrt(posterior_var),
        f"dyn_{group}_shock_z": np.clip(shock, -8.0, 8.0),
    }
    return pd.DataFrame(out, index=block.index).astype({
        c: "float32" for c in out if not c.endswith("_known")
    })


def spec_rate(group):
    return "asof_pitcher_success_rate" if group == "pitcher" else "asof_batter_success_rate"


def build_dynamic_features(raw, train_mask, blocks, prior):
    """각 행 season 이전 history만으로 동적 피처를 만든다."""
    outputs = {g: pd.DataFrame(index=raw.index, columns=dynamic_cols(g), dtype=float)
               for g in ("pitcher", "batter")}
    for season in sorted(raw["season"].unique()):
        rows = raw["season"].eq(season)
        history = raw.loc[train_mask & (raw["season"].to_numpy() < season)]
        for group in ("pitcher", "batter"):
            spec = GROUPS[group]
            lookup = fit_dynamic_lookup(history, spec, prior[spec_rate(group)])
            feat = apply_dynamic(blocks[group].loc[rows],
                                 raw.loc[rows, spec["entity"]], lookup,
                                 prior[spec_rate(group)], group)
            outputs[group].loc[rows, feat.columns] = feat.to_numpy()
    for group in outputs:
        outputs[group] = outputs[group].astype({
            c: ("int8" if c.endswith("_known") else "float32")
            for c in outputs[group].columns
        })
    return outputs


def mean_predictions(num_cols, X, train_mask, val_mask, last_mask, y):
    cols = CAT_COLS + num_cols
    pv, pl = [], []
    for i, spec0 in enumerate(LGBM_SPECS):
        spec = dict(spec0); seed = spec.pop("seed")
        t = time.time()
        model = make_lgbm_model(num_cols, seed=seed, **spec)
        model.fit(X.loc[train_mask, cols], y.loc[train_mask])
        pv.append(model.predict_proba(X.loc[val_mask, cols])[:, 1])
        pl.append(model.predict_proba(X.loc[last_mask, cols])[:, 1])
        log(f"    member {i+1}/{len(LGBM_SPECS)} seed={seed} {time.time()-t:.0f}s")
        del model; gc.collect()
    return np.mean(pv, axis=0), np.mean(pl, axis=0)


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET], raw["season"]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    base_num = raw_num + list(DERIVED_COLS) + shrinkage_cols()
    fold_results = {}
    os.makedirs(PRED_DIR, exist_ok=True)

    for val_season in VAL_SEASONS:
        train_mask = (seasons < val_season).to_numpy()
        val_mask = (seasons == val_season).to_numpy()
        last_train = int(seasons.loc[train_mask].max())
        last_mask = train_mask & seasons.eq(last_train).to_numpy()
        yv = y.loc[val_mask].to_numpy(dtype=float)
        prior = fit_prior(raw.loc[train_mask])
        X0 = pd.concat([base_df, add_shrinkage(base_df, prior, K)], axis=1)
        blocks = {}
        for group in ("pitcher", "batter"):
            previous = _previous_lookup(raw, train_mask, GROUPS[group])
            blocks[group], _ = _season_block(X0, previous, GROUPS[group], prior,
                                              zero_baseline=True)
        dynamics = build_dynamic_features(raw, train_mask, blocks, prior)
        X = pd.concat([X0, blocks["pitcher"], blocks["batter"],
                       dynamics["pitcher"], dynamics["batter"]], axis=1)
        current16 = (exact_cols(blocks["pitcher"], "pitcher", spec_rate("pitcher"))
                     + exact_cols(blocks["batter"], "batter", spec_rate("batter")))
        configs = {
            "pitcher_dynamic": current16 + dynamic_cols("pitcher"),
            "both_dynamic": current16 + dynamic_cols("pitcher") + dynamic_cols("batter"),
        }
        cache = np.load(f"{PRED_DIR}/current_season_family_{val_season}.npz")
        p_hgb, last_hgb = cache["p_hgb8"], cache["last_hgb8"]
        p_base_lgbm, last_base_lgbm = cache["p_current16"], cache["last_current16"]
        r_extrap = extrapolated_rate(y, seasons, train_mask, val_season)
        p0_raw = (1 - FAMILY_WEIGHT) * p_hgb + FAMILY_WEIGHT * p_base_lgbm
        p0_last = (1 - FAMILY_WEIGHT) * last_hgb + FAMILY_WEIGHT * last_base_lgbm
        p0, _, _, _ = recenter_like_production(p0_raw, p0_last, r_extrap, 0.5)
        base = yv.mean() * (1 - yv.mean())
        log(f"\n[fold={val_season}] train={train_mask.sum():,} val={val_mask.sum():,}")
        saved = dict(y=yv, p_baseline=p0, p_baseline_raw=p0_raw)
        fold_results[val_season] = dict(y=yv, baseline=p0, baseline_raw=p0_raw,
                                        r_extrap=r_extrap, p_hgb=p_hgb,
                                        last_hgb=last_hgb)
        for name, new_cols in configs.items():
            log(f"  LGBM3 {name} added={len(new_cols)-len(current16)}")
            pv, pl = mean_predictions(base_num + new_cols, X, train_mask,
                                      val_mask, last_mask, y)
            raw_pred = (1 - FAMILY_WEIGHT) * p_hgb + FAMILY_WEIGHT * pv
            last_pred = (1 - FAMILY_WEIGHT) * last_hgb + FAMILY_WEIGHT * pl
            pred, shift, _, _ = recenter_like_production(raw_pred, last_pred,
                                                          r_extrap, 0.5)
            gain, se = paired(yv, p0, pred, base)
            raw_gain, raw_se = paired(yv, p0_raw, raw_pred, base)
            log(f"    raw gain={raw_gain:+.2f} SE={raw_se:.2f} | "
                f"recenter gain={gain:+.2f} SE={se:.2f} shift={shift:+.5f}")
            saved[f"p_{name}"] = pred
            saved[f"p_{name}_raw"] = raw_pred
            fold_results[val_season][name] = (pred, raw_pred)
        np.savez_compressed(f"{PRED_DIR}/dynamic_state_{val_season}.npz", **saved)
        del X, X0, blocks, dynamics
        gc.collect()

    log("\nSUMMARY vs v8 HGB25/current16-LGBM75 + recenter_f=0.5")
    for name in ("pitcher_dynamic", "both_dynamic"):
        vals = []
        for season, fd in fold_results.items():
            base = fd["y"].mean() * (1 - fd["y"].mean())
            gain, se = paired(fd["y"], fd["baseline"], fd[name][0], base)
            vals.append(gain)
            log(f"  {name:16s} {season}: {gain:+.2f} (SE {se:.2f})")
        log(f"  {name:16s} mean: {np.mean(vals):+.2f}")


if __name__ == "__main__":
    main()
