"""Direct-Brier 회귀 진단 (v4 고정 EB 피처만 사용).

사용법 (저장소 루트에서 실행):
  python3 scripts/diag_direct_brier.py screen
  python3 scripts/diag_direct_brier.py screen_compare
  python3 scripts/diag_direct_brier.py confirm
  python3 scripts/diag_direct_brier.py compare

screen은 동일한 LightGBM 설정의 binary classifier와 L2 regressor 한 쌍으로
loss 차이를 넓게 확인한다. confirm은 HGB5+LightGBM3 회귀 이질 앙상블을 만들며,
compare는 기존 v4 고정-EB classifier 저장 예측(ens_hgb/lgbm)과 회귀 예측 및
고정 혼합비/OOF convex weight를 비교한다.

피처는 pipeline.py의 v4 계약(원본 + 파생 10 + 고정 EB 34)만 사용한다.
시대보정 EB, test 분포, 행간 통계는 사용하지 않는다.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, HGB_D, LGBM_D,
                      add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                      make_lgbm_model, bss)

DATA_DIR = "./open/data"
PRED_DIR = "./experiments/preds"
VAL_SEASONS = [2022, 2023, 2024]
K = 50
EPS = 1e-6

HGB_SPECS = [
    dict(seed=42, learning_rate=.03, max_leaf_nodes=63, min_samples_leaf=30, max_features=1.0),
    dict(seed=7, learning_rate=.05, max_leaf_nodes=31, min_samples_leaf=50, max_features=.7),
    dict(seed=2024, learning_rate=.02, max_leaf_nodes=95, min_samples_leaf=20, max_features=.8),
    dict(seed=1, learning_rate=.04, max_leaf_nodes=63, min_samples_leaf=100, max_features=.6),
    dict(seed=12345, learning_rate=.03, max_leaf_nodes=127, min_samples_leaf=40, max_features=.9),
]
LGBM_SPECS = [
    dict(seed=99, learning_rate=.03, num_leaves=63, min_child_samples=30,
         colsample_bytree=.8, subsample=.8),
    dict(seed=2718, learning_rate=.05, num_leaves=31, min_child_samples=50,
         colsample_bytree=.7, subsample=.7),
    dict(seed=31415, learning_rate=.02, num_leaves=127, min_child_samples=20,
         colsample_bytree=.9, subsample=.9),
]


def log(*args):
    print(*args, flush=True)


def preprocessor(num_cols):
    return ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS),
        ("num", "passthrough", num_cols),
    ])


def make_hgb_regressor(num_cols, seed, **overrides):
    params = {**HGB_D, **overrides}
    reg = HistGradientBoostingRegressor(loss="squared_error", random_state=seed, **params)
    return Pipeline([("pre", preprocessor(num_cols)), ("reg", reg)])


def make_lgbm_regressor(num_cols, seed, **overrides):
    import lightgbm as lgb
    params = {**LGBM_D, **overrides}
    reg = lgb.LGBMRegressor(objective="regression_l2", random_state=seed,
                            verbosity=-1, **params)
    return Pipeline([("pre", preprocessor(num_cols)), ("reg", reg)])


def clip_report(raw):
    raw = np.asarray(raw, dtype=float)
    return np.clip(raw, EPS, 1 - EPS), dict(
        raw_min=float(raw.min()), raw_max=float(raw.max()),
        clip_low=float(np.mean(raw < EPS)), clip_high=float(np.mean(raw > 1 - EPS)))


def paired(y, p_ref, p_new):
    """양수 diff는 p_new가 p_ref보다 좋은 BSS 환산 개선."""
    y = np.asarray(y, dtype=float)
    base = y.mean() * (1 - y.mean())
    d = ((p_ref - y) ** 2 - (p_new - y) ** 2) / base * 100000
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def calibration_resolution(y, p, n_bins=20):
    """등빈도 bin Murphy 진단. 분해 항등식은 bin 근사이므로 진단용이다."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, n_bins)
    r = y.mean()
    reliability = resolution = 0.0
    ece = 0.0
    for ix in groups:
        if len(ix) == 0:
            continue
        w, pk, yk = len(ix) / len(y), p[ix].mean(), y[ix].mean()
        reliability += w * (pk - yk) ** 2
        resolution += w * (yk - r) ** 2
        ece += w * abs(pk - yk)
    return reliability, resolution, ece


def describe(name, y, p):
    br, score, _ = bss(y, p)
    rel, res, ece = calibration_resolution(y, p)
    log(f"    {name:<22} brier={br:.8f} BSS={score:8.2f} "
        f"reliability={rel:.8f} resolution={res:.8f} ECE={ece:.6f}")
    return br, score


def load_data():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw_num = [c for c in raw_features if c not in CAT_COLS]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    num_cols = raw_num + list(DERIVED_COLS) + shrinkage_cols()
    return raw, base_df, num_cols


def fold_data(raw, base_df, num_cols, val_season):
    tr_m = (raw["season"] < val_season).to_numpy()
    va_m = (raw["season"] == val_season).to_numpy()
    prior = fit_prior(raw.loc[tr_m])
    # add_shrinkage is row-local after train-only prior is fixed.
    sh = add_shrinkage(base_df, prior, K)
    cols = CAT_COLS + num_cols
    xtr = pd.concat([base_df.loc[tr_m], sh.loc[tr_m]], axis=1)[cols]
    xva = pd.concat([base_df.loc[va_m], sh.loc[va_m]], axis=1)[cols]
    ytr = raw.loc[tr_m, TARGET]
    yva = raw.loc[va_m, TARGET].to_numpy(dtype=float)
    del sh
    gc.collect()
    return xtr, xva, ytr, yva


def run_screen():
    raw, base_df, num_cols = load_data()
    spec = dict(LGBM_SPECS[0])
    for season in VAL_SEASONS:
        log(f"\n[screen] fold={season}")
        xtr, xva, ytr, y = fold_data(raw, base_df, num_cols, season)
        kw = dict(spec); seed = kw.pop("seed")
        t = time.time()
        clf = make_lgbm_model(num_cols, seed=seed, **kw)
        clf.fit(xtr, ytr)
        p_cls = clf.predict_proba(xva)[:, 1]
        log(f"  classifier trained in {time.time()-t:.1f}s")
        del clf; gc.collect()
        t = time.time()
        reg = make_lgbm_regressor(num_cols, seed=seed, **kw)
        reg.fit(xtr, ytr)
        p_raw = reg.predict(xva)
        p_reg, cr = clip_report(p_raw)
        log(f"  regressor trained in {time.time()-t:.1f}s; clip={cr}")
        describe("paired classifier", y, p_cls)
        describe("paired regressor", y, p_reg)
        d, se = paired(y, p_cls, p_reg)
        log(f"    paired reg-cls diff={d:+.2f} (SE={se:.2f})")
        np.savez_compressed(f"{PRED_DIR}/direct_brier_screen_{season}.npz",
                            y=y, p_cls=p_cls, p_reg=p_reg, p_reg_raw=p_raw)
        del xtr, xva, ytr, reg, p_cls, p_reg, p_raw
        gc.collect()


def run_confirm():
    raw, base_df, num_cols = load_data()
    for season in VAL_SEASONS:
        log(f"\n[confirm] fold={season}")
        xtr, xva, ytr, y = fold_data(raw, base_df, num_cols, season)
        member_raw = []
        for family, specs, builder in [("hgb", HGB_SPECS, make_hgb_regressor),
                                       ("lgbm", LGBM_SPECS, make_lgbm_regressor)]:
            for i, spec0 in enumerate(specs, 1):
                spec = dict(spec0); seed = spec.pop("seed")
                t = time.time()
                model = builder(num_cols, seed=seed, **spec)
                model.fit(xtr, ytr)
                pred = model.predict(xva)
                member_raw.append(pred)
                clipped, cr = clip_report(pred)
                _, score, _ = bss(y, clipped)
                log(f"  {family}{i}/{len(specs)} BSS={score:.2f} {time.time()-t:.1f}s "
                    f"raw=[{cr['raw_min']:.5f},{cr['raw_max']:.5f}] "
                    f"clip=({cr['clip_low']:.6%},{cr['clip_high']:.6%})")
                del model, pred, clipped
                gc.collect()
        # 평균 뒤 clip: production 후보의 정확한 계약.
        raw_mean = np.vstack(member_raw).mean(axis=0)
        p_reg, cr = clip_report(raw_mean)
        log(f"  ensemble raw/clip: {cr}")
        describe("direct ensemble 5+3", y, p_reg)
        np.savez_compressed(f"{PRED_DIR}/direct_brier_confirm_{season}.npz",
                            y=y, p=p_reg, p_raw=raw_mean, n_hgb=len(HGB_SPECS),
                            n_lgbm=len(LGBM_SPECS))
        del xtr, xva, ytr, member_raw, raw_mean, p_reg
        gc.collect()


def classifier_v4(season):
    h = np.load(f"{PRED_DIR}/ens_hgb_{season}.npz")
    l = np.load(f"{PRED_DIR}/ens_lgbm_{season}.npz")
    assert np.array_equal(h["y"], l["y"])
    nh, nl = int(h["n"]), int(l["n"])
    return h["y"], (nh * h["p"] + nl * l["p"]) / (nh + nl)


def convex_classifier_weight(y, p_cls, p_reg):
    d = p_cls - p_reg
    den = float(d @ d)
    return float(np.clip((d @ (y - p_reg)) / den, 0.0, 1.0)) if den else 0.5


def run_compare():
    pooled = {"y": [], "c": [], "r": []}
    fold_rows = []
    for season in VAL_SEASONS:
        z = np.load(f"{PRED_DIR}/direct_brier_confirm_{season}.npz")
        y, p_cls = classifier_v4(season)
        assert np.array_equal(y, z["y"])
        p_reg = z["p"]
        log(f"\n[compare] fold={season}")
        _, s_cls = describe("classifier v4 5+3", y, p_cls)
        _, s_reg = describe("direct reg 5+3", y, p_reg)
        d, se = paired(y, p_cls, p_reg)
        log(f"    direct-v4 diff={d:+.2f} (SE={se:.2f})")
        row = dict(season=season, classifier=s_cls, regressor=s_reg,
                   direct_diff=d, direct_se=se)
        for wc in (.75, .50, .25):
            p = wc * p_cls + (1 - wc) * p_reg
            _, score = describe(f"mix cls/reg {wc:.2f}/{1-wc:.2f}", y, p)
            dm, sem = paired(y, p_cls, p)
            log(f"      mix-v4 diff={dm:+.2f} (SE={sem:.2f})")
            row[f"mix_{wc:.2f}"] = score
        wf = convex_classifier_weight(y, p_cls, p_reg)
        p_opt = wf * p_cls + (1 - wf) * p_reg
        _, so = describe(f"fold oracle w_cls={wf:.4f}", y, p_opt)
        row.update(w_fold=wf, score_fold_oracle=so)
        fold_rows.append(row)
        for k, a in [("y", y), ("c", p_cls), ("r", p_reg)]: pooled[k].append(a)

    yy = np.concatenate(pooled["y"]); cc = np.concatenate(pooled["c"]); rr = np.concatenate(pooled["r"])
    wp = convex_classifier_weight(yy, cc, rr)
    log(f"\n[OOF pooled convex] classifier weight={wp:.6f}, regressor weight={1-wp:.6f}")
    for row, season in zip(fold_rows, VAL_SEASONS):
        y, c = classifier_v4(season)
        r = np.load(f"{PRED_DIR}/direct_brier_confirm_{season}.npz")["p"]
        p = wp*c + (1-wp)*r
        d, se = paired(y, c, p)
        _, score = describe(f"pooled-weight fold {season}", y, p)
        log(f"      pooled-weight-v4 diff={d:+.2f} (SE={se:.2f})")
        row.update(pooled_weight_score=score, pooled_weight_diff=d,
                   pooled_weight_se=se)
    pd.DataFrame(fold_rows).to_csv(f"{PRED_DIR}/direct_brier_summary.csv", index=False)


def run_screen_compare():
    """단일 회귀 screen이 v4 classifier와 섞일 때의 다양성 이득까지 확인."""
    pooled = {"y": [], "c": [], "r": []}
    rows = []
    for season in VAL_SEASONS:
        z = np.load(f"{PRED_DIR}/direct_brier_screen_{season}.npz")
        y, p_cls = classifier_v4(season)
        assert np.array_equal(y, z["y"])
        p_reg = z["p_reg"]
        log(f"\n[screen_compare] fold={season}")
        _, s_cls = describe("classifier v4 5+3", y, p_cls)
        _, s_reg = describe("single direct LGBM", y, p_reg)
        d, se = paired(y, p_cls, p_reg)
        log(f"    direct-v4 diff={d:+.2f} (SE={se:.2f})")
        row = dict(season=season, classifier=s_cls, regressor=s_reg,
                   direct_diff=d, direct_se=se)
        for wc in (.75, .50, .25):
            p = wc*p_cls + (1-wc)*p_reg
            _, score = describe(f"mix cls/reg {wc:.2f}/{1-wc:.2f}", y, p)
            dm, sem = paired(y, p_cls, p)
            log(f"      mix-v4 diff={dm:+.2f} (SE={sem:.2f})")
            row[f"mix_{wc:.2f}_diff"] = dm
            row[f"mix_{wc:.2f}_se"] = sem
        wf = convex_classifier_weight(y, p_cls, p_reg)
        log(f"    fold oracle classifier weight={wf:.6f}")
        row["fold_oracle_w_cls"] = wf
        rows.append(row)
        for k, a in [("y", y), ("c", p_cls), ("r", p_reg)]: pooled[k].append(a)
    yy = np.concatenate(pooled["y"]); cc = np.concatenate(pooled["c"]); rr = np.concatenate(pooled["r"])
    wp = convex_classifier_weight(yy, cc, rr)
    log(f"\n[screen OOF pooled convex] classifier weight={wp:.6f}, regressor weight={1-wp:.6f}")
    for row, season in zip(rows, VAL_SEASONS):
        y, c = classifier_v4(season)
        r = np.load(f"{PRED_DIR}/direct_brier_screen_{season}.npz")["p_reg"]
        p = wp*c + (1-wp)*r
        d, se = paired(y, c, p)
        _, score = describe(f"pooled-weight fold {season}", y, p)
        log(f"      pooled-weight-v4 diff={d:+.2f} (SE={se:.2f})")
        row.update(pooled_w_cls=wp, pooled_score=score, pooled_diff=d, pooled_se=se)
    pd.DataFrame(rows).to_csv(f"{PRED_DIR}/direct_brier_screen_summary.csv", index=False)


if __name__ == "__main__":
    os.makedirs(PRED_DIR, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "screen"
    if mode == "screen": run_screen()
    elif mode == "screen_compare": run_screen_compare()
    elif mode == "confirm": run_confirm()
    elif mode == "compare": run_compare()
    else: raise SystemExit("mode must be screen|screen_compare|confirm|compare")
