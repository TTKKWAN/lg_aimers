"""관측된 EB 이력으로 설명되지 않는 선수/팀 고정효과를 진단한다.

기존 target encoding은 y 수준 자체를 ID에 붙여 asof 성공률과 중복되어 손해였다.
여기서는 y - EB_success라는 잔차만 ID별로 축소 추정한다. 학습 행에는 자기 정답을
뺀 leave-one-out 값을 사용하고, 검증 행에는 과거 학습 시즌에서 고정한 map만 쓴다.
"""
import gc
import os
import sys
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS, add_derived, add_shrinkage,
                       shrinkage_cols, fit_prior, make_lgbm_model, bss)

DATA_DIR = "./open/data"
PRED_DIR = "./experiments/preds"
VAL_SEASONS = [2022, 2023, 2024]
K = 50
SPEC = dict(seed=99, learning_rate=0.03, num_leaves=63, min_child_samples=30,
            colsample_bytree=0.8, subsample=0.8)


def log(*args):
    print(*args, flush=True)


def residual_effect_train_val(ids_tr, ids_va, residual, smooth):
    """train=leave-one-out, validation=학습 map 고정 적용."""
    work = pd.DataFrame({"id": ids_tr.to_numpy(), "r": np.asarray(residual, dtype=float)})
    stats = work.groupby("id", sort=False)["r"].agg(["sum", "count"])
    sums = work["id"].map(stats["sum"]).to_numpy(dtype=float)
    counts = work["id"].map(stats["count"]).to_numpy(dtype=float)
    global_r = float(work["r"].mean())
    train_effect = (sums - work["r"].to_numpy() + smooth * global_r) / (
        counts - 1.0 + smooth)
    mapping = (stats["sum"] + smooth * global_r) / (stats["count"] + smooth)
    val_effect = ids_va.map(mapping).fillna(global_r).to_numpy(dtype=float)
    return train_effect, val_effect


def build_effects(raw, sh, tr_m, va_m, y_tr, smooth, include_teams):
    specs = [
        ("pitcher_id", "sh_asof_pitcher_success_rate", "pitcher"),
        ("batter_id", "sh_asof_batter_success_rate", "batter"),
    ]
    if include_teams:
        specs += [
            ("pitcher_team_id", "sh_asof_pitcher_success_rate", "pitcher_team"),
            ("batter_team_id", "sh_asof_batter_success_rate", "batter_team"),
        ]
    tr_out, va_out = {}, {}
    for id_col, base_col, name in specs:
        base_tr = sh.loc[tr_m, base_col].to_numpy(dtype=float)
        residual = pd.Series(y_tr.to_numpy(dtype=float) - base_tr, index=y_tr.index)
        # 선수 고정효과에 시즌의 전체 수준 하락이 귀속되지 않도록 within-season
        # 중심화한다. 검증/평가 시즌의 target 평균은 전혀 사용하지 않는다.
        tr_season = raw.loc[tr_m, "season"]
        residual = residual - residual.groupby(tr_season).transform("mean")
        e_tr, e_va = residual_effect_train_val(
            raw.loc[tr_m, id_col], raw.loc[va_m, id_col], residual, smooth)
        tr_out[f"re_{name}"] = e_tr
        va_out[f"re_{name}"] = e_va
        tr_out[f"re_adjusted_{name}"] = base_tr + e_tr
        va_out[f"re_adjusted_{name}"] = sh.loc[va_m, base_col].to_numpy(dtype=float) + e_va
    return pd.DataFrame(tr_out, index=raw.index[tr_m]), pd.DataFrame(va_out, index=raw.index[va_m])


CONFIGS = {
    "baseline": None,
    "pitcher_only_s200": dict(smooth=200.0, include_teams=False, keep=["pitcher"]),
    "players_s200": dict(smooth=200.0, include_teams=False, keep=None),
    "players_s1000": dict(smooth=1000.0, include_teams=False, keep=None),
    "players_teams_s500": dict(smooth=500.0, include_teams=True, keep=None),
}

requested = sys.argv[1:]
unknown = [x for x in requested if x not in CONFIGS]
if unknown:
    raise SystemExit(f"unknown configs: {unknown}; choices={list(CONFIGS)}")
RUN_CONFIGS = {"baseline": CONFIGS["baseline"]}
if requested:
    RUN_CONFIGS.update({x: CONFIGS[x] for x in requested if x != "baseline"})
else:
    RUN_CONFIGS = CONFIGS


test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
raw_features = [c for c in test_cols if c != ID]
raw_num = [c for c in raw_features if c not in CAT_COLS]
raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                  usecols=raw_features + [TARGET])
y, seasons = raw[TARGET], raw["season"]
base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
base_num = list(raw_num) + list(DERIVED_COLS) + shrinkage_cols()
os.makedirs(PRED_DIR, exist_ok=True)
losses, scores = {}, {}

for val_season in VAL_SEASONS:
    tr_m = (seasons < val_season).to_numpy()
    va_m = (seasons == val_season).to_numpy()
    y_tr, y_va = y[tr_m], y[va_m]
    yv = y_va.to_numpy(dtype=float)
    sh = add_shrinkage(base_df, fit_prior(raw.loc[tr_m]), K)
    Xbase = pd.concat([base_df, sh], axis=1)
    log(f"\n[fold={val_season}] train={tr_m.sum():,} val={va_m.sum():,}")

    for name, cfg in RUN_CONFIGS.items():
        Xtr, Xva = Xbase.loc[tr_m].copy(), Xbase.loc[va_m].copy()
        extra_cols = []
        if cfg is not None:
            etr, eva = build_effects(raw, sh, tr_m, va_m, y_tr,
                                      cfg["smooth"], cfg["include_teams"])
            if cfg["keep"] == ["pitcher"]:
                keep = ["re_pitcher", "re_adjusted_pitcher"]
                etr, eva = etr[keep], eva[keep]
            extra_cols = list(etr.columns)
            Xtr = pd.concat([Xtr, etr], axis=1)
            Xva = pd.concat([Xva, eva], axis=1)
        num_cols = base_num + extra_cols
        cols = CAT_COLS + num_cols
        spec = dict(SPEC)
        seed = spec.pop("seed")
        t = time.time()
        model = make_lgbm_model(num_cols, seed=seed, **spec)
        model.fit(Xtr[cols], y_tr)
        p = model.predict_proba(Xva[cols])[:, 1]
        _, score, base = bss(yv, p)
        losses[(name, val_season)] = ((p - yv) ** 2, base)
        scores[(name, val_season)] = score
        np.savez_compressed(f"{PRED_DIR}/residual_{name}_{val_season}.npz", p=p, y=yv, n=1)
        log(f"  {name:24s} features={len(cols):3d} score={score:8.2f} time={time.time()-t:.0f}s")
        del model, Xtr, Xva
        gc.collect()
    del Xbase, sh
    gc.collect()

log("\n" + "=" * 92)
log("잔차 고정효과 QUICK 요약")
log("=" * 92)
for name in RUN_CONFIGS:
    fold_scores = [scores[(name, f)] for f in VAL_SEASONS]
    line = f"{name:24s} folds=" + "/".join(f"{s:7.2f}" for s in fold_scores)
    if name != "baseline":
        gains, ses = [], []
        for f in VAL_SEASONS:
            l0, base = losses[("baseline", f)]
            l1, _ = losses[(name, f)]
            d = (l0 - l1) / base * 100000
            gains.append(d.mean())
            ses.append(d.std(ddof=1) / np.sqrt(len(d)))
        line += " gains=" + "/".join(f"{g:+7.2f}" for g in gains)
        line += f" mean={np.mean(gains):+7.2f} SE=" + "/".join(f"{s:.1f}" for s in ses)
    log(line)
