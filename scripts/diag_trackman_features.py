"""trackman_history.csv 시즌 집계 피처 진단 — 여러 forward-chaining 폴드로 검증.

trackman_history는 개체(pitcher_id/batter_id) 단위 join이 불가능함이 이미 확인됨
(CLAUDE.md §7 v3). 안전하게 쓸 수 있는 유일한 방법은 리그/시즌 단위 집계뿐인데,
trackman_history 자체가 2025 데이터가 없어서 평가 시점(예: 이 폴드의 val_season)
값도 결국 추세 외삽으로 채워야 한다 — recenter_f와 같은 종류의 리스크.

그래서 recenter_f 진단에서 배운 교훈을 그대로 적용: **단일 폴드가 아니라
여러 forward-chaining 폴드(2022, 2023, 2024)에서 일관되게 이득이 나는지** 확인한
뒤에만 채택한다. 각 폴드에서 val_season의 trackman 집계값은 (실제 값이 있어도)
일부러 쓰지 않고, train에 쓸 수 있는 이전 시즌들만으로 선형 외삽해 만든다 —
2025 실전에서 할 수 있는 것과 동일한 절차를 그대로 재현하기 위함.
"""
import time

import numpy as np
import pandas as pd

from pipeline import (ID, TARGET, CAT_COLS, DERIVED_COLS,
                       add_derived, add_shrinkage, shrinkage_cols, fit_prior,
                       make_model, bss)

DATA_DIR = "./open/data"
VAL_SEASONS = [2022, 2023, 2024]
N_MEMBERS = 5
K = 50

TM_STAT_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension"]


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- 데이터 로드
test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig", nrows=0).columns
RAW_FEATURES = [c for c in test_cols if c != ID]
RAW_NUM = [c for c in RAW_FEATURES if c not in CAT_COLS]
raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                   usecols=RAW_FEATURES + [TARGET])
y, seasons = raw[TARGET], raw["season"]
base_df = pd.concat([raw[RAW_FEATURES], add_derived(raw)], axis=1)

log("trackman_history.csv 로드 및 시즌별 집계 중...")
tm = pd.read_csv(f"{DATA_DIR}/trackman_history.csv", encoding="utf-8-sig",
                  usecols=["season", "pitch_type_group"] + TM_STAT_COLS)
agg = tm.groupby("season")[TM_STAT_COLS].agg(["mean", "std"])
agg.columns = [f"tm_{c}_{s}" for c, s in agg.columns]
pt_share = (tm.groupby("season")["pitch_type_group"]
            .value_counts(normalize=True).unstack(fill_value=0.0))
pt_share.columns = [f"tm_share_{c}" for c in pt_share.columns]
season_stats = pd.concat([agg, pt_share], axis=1).sort_index()
TM_COLS = list(season_stats.columns)
log(f"시즌 집계 피처 {len(TM_COLS)}개: {TM_COLS}")
log(season_stats.round(3).to_string())


def extrapolate_row(avail_idx, target_season):
    """avail_idx(과거 시즌들)만으로 각 통계 컬럼을 선형 외삽해 target_season 값 산출."""
    xs = np.array(avail_idx, dtype=float)
    use = xs[-3:] if len(xs) >= 3 else xs
    out = {}
    for c in TM_COLS:
        ys = season_stats.loc[use.astype(int), c].to_numpy(dtype=float)
        if len(use) >= 2:
            b, a0 = np.polyfit(use, ys, 1)
            out[c] = float(a0 + b * target_season)
        else:
            out[c] = float(ys[-1])
    return out


def build_tm_lookup(val_season):
    """train 시즌은 실측값, val_season은 (train만으로) 외삽값 — 배포 시나리오 재현."""
    tr_idx = [s for s in season_stats.index if s < val_season]
    lut = season_stats.loc[tr_idx].copy()
    lut.loc[val_season] = extrapolate_row(tr_idx, val_season)
    return lut.sort_index()


HETERO = [
    dict(seed=42,    learning_rate=0.03,  max_leaf_nodes=63,  min_samples_leaf=30,  max_features=1.0),
    dict(seed=7,     learning_rate=0.05,  max_leaf_nodes=31,  min_samples_leaf=50,  max_features=0.7),
    dict(seed=2024,  learning_rate=0.02,  max_leaf_nodes=95,  min_samples_leaf=20,  max_features=0.8),
    dict(seed=1,     learning_rate=0.04,  max_leaf_nodes=63,  min_samples_leaf=100, max_features=0.6),
    dict(seed=12345, learning_rate=0.03,  max_leaf_nodes=127, min_samples_leaf=40,  max_features=0.9),
][:N_MEMBERS]

fold_results = []
for val_season in VAL_SEASONS:
    log(f"\n{'='*78}\nfold val_season={val_season}\n{'='*78}")
    tr_m = (seasons < val_season).to_numpy()
    va_m = (seasons == val_season).to_numpy()
    y_tr, y_va = y[tr_m], y[va_m]
    yv = y_va.to_numpy(dtype=float)

    prior = fit_prior(raw.loc[tr_m])
    sh = add_shrinkage(base_df, prior, K)
    num_cols_base = list(RAW_NUM) + list(DERIVED_COLS) + shrinkage_cols()

    lut = build_tm_lookup(val_season)
    tm_feats = seasons.map(lambda s: lut.loc[s] if s in lut.index else lut.iloc[-1])
    tm_df = pd.DataFrame(tm_feats.tolist(), index=base_df.index, columns=TM_COLS)

    Xtr_base = pd.concat([base_df.loc[tr_m], sh.loc[tr_m]], axis=1)
    Xva_base = pd.concat([base_df.loc[va_m], sh.loc[va_m]], axis=1)
    Xtr_tm = pd.concat([Xtr_base, tm_df.loc[tr_m]], axis=1)
    Xva_tm = pd.concat([Xva_base, tm_df.loc[va_m]], axis=1)

    variants = {
        "baseline":  (Xtr_base, Xva_base, num_cols_base),
        "+trackman": (Xtr_tm,   Xva_tm,   num_cols_base + TM_COLS),
    }

    preds = {}
    for name, (Xtr, Xva, num_cols) in variants.items():
        cols = CAT_COLS + num_cols
        ps = []
        for i, spec in enumerate(HETERO):
            spec = dict(spec)
            seed = spec.pop("seed")
            t = time.time()
            m = make_model(num_cols, seed=seed, **spec)
            m.fit(Xtr[cols], y_tr)
            p = m.predict_proba(Xva[cols])[:, 1]
            ps.append(p)
            log(f"  [{name}] member {i+1}/{N_MEMBERS} ({time.time()-t:.0f}s)")
        preds[name] = np.vstack(ps).mean(axis=0)

    _, s_base, base = bss(yv, preds["baseline"])
    _, s_tm, _ = bss(yv, preds["+trackman"])
    sq_base = (preds["baseline"] - yv) ** 2
    sq_tm = (preds["+trackman"] - yv) ** 2
    diff = (sq_base - sq_tm) / base * 100000
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    log(f"\n>> fold {val_season}: baseline={s_base:.2f}  +trackman={s_tm:.2f}  "
        f"diff={diff.mean():+.2f} (SE={se:.1f})")
    fold_results.append(dict(val_season=val_season, baseline=s_base, trackman=s_tm,
                              diff=diff.mean(), se=se))

log("\n" + "=" * 78)
log("요약 (baseline vs +trackman, raw 점수 — 재중심화 없음)")
log("=" * 78)
for r in fold_results:
    log(f"  fold {r['val_season']}: baseline={r['baseline']:7.2f}  "
        f"+trackman={r['trackman']:7.2f}  diff={r['diff']:+7.2f} (SE={r['se']:.1f})")
diffs = np.array([r["diff"] for r in fold_results])
log(f"\n폴드 평균 diff = {diffs.mean():+.2f} (폴드 std={diffs.std(ddof=1):.2f}, n={len(diffs)})")
