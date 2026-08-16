"""ens_hgb_*.npz / ens_lgbm_*.npz 로드해서 hgb-only vs hgb+lgbm 혼합 비교 (가벼움, 재학습 없음)."""
import numpy as np

from pipeline import bss

VAL_SEASONS = [2022, 2023, 2024]
PRED_DIR = "./experiments/preds"


def log(*a):
    print(*a, flush=True)


fold_results = []
for val_season in VAL_SEASONS:
    h = np.load(f"{PRED_DIR}/ens_hgb_{val_season}.npz")
    l = np.load(f"{PRED_DIR}/ens_lgbm_{val_season}.npz")
    assert np.array_equal(h["y"], l["y"])
    yv = h["y"]
    n_h, n_l = int(h["n"]), int(l["n"])
    p_h = h["p"]
    p_mix = (n_h * h["p"] + n_l * l["p"]) / (n_h + n_l)

    _, s_h, base = bss(yv, p_h)
    _, s_mix, _ = bss(yv, p_mix)
    sq_h = (p_h - yv) ** 2
    sq_mix = (p_mix - yv) ** 2
    diff = (sq_h - sq_mix) / base * 100000
    se = diff.std(ddof=1) / np.sqrt(len(diff))
    log(f"fold {val_season}: hgb{n_h}={s_h:7.2f}  hgb{n_h}+lgbm{n_l}={s_mix:7.2f}  "
        f"diff={diff.mean():+7.2f} (SE={se:.1f})")
    fold_results.append(dict(val_season=val_season, hgb=s_h, mixed=s_mix,
                              diff=diff.mean(), se=se))

diffs = np.array([r["diff"] for r in fold_results])
log(f"\n폴드 평균 diff = {diffs.mean():+.2f} (폴드 std={diffs.std(ddof=1):.2f}, n={len(diffs)})")
