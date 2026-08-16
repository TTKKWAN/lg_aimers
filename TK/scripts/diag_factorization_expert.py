"""저차원 pitcher×batter residual factorization expert를 검증한다.

fold-train에서 ``y - 0.5*(pitcher_EB+batter_EB)``를 투수×타자 sparse 행렬로
집계하고 pair count로 축소한 뒤 truncated SVD한다. validation의 observed 선수는
처음 보는 pair라도 두 잠재벡터의 내적으로 일반화하며 unseen 선수는 residual 0으로
fallback한다. 외부 패키지 없이 scipy/sklearn만 사용한다.
"""
import gc
import os

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from pipeline import ID, TARGET, add_derived, add_shrinkage, fit_prior
from diag_season_to_date import DATA_DIR, PRED_DIR, VAL_SEASONS, K, paired
from diag_current_season_family import recenter_like_production


RANKS = [8, 16, 32]
SHRINKS = [10.0, 30.0, 100.0]
SEEDS = [2026, 2718]
BLEND_WEIGHTS = [0.05, 0.10, 0.20, 0.30]


def log(*args):
    print(*args, flush=True)


def encode(train_values, all_values):
    vocab = pd.Index(pd.Series(train_values).astype(str).unique())
    return vocab.get_indexer(pd.Series(all_values).astype(str)), len(vocab)


def factor_predictions(pitcher_code, batter_code, n_pitcher, n_batter,
                       residual, train_mask, row_mask, rank, shrink, seed):
    pi = pitcher_code[train_mask]
    bi = batter_code[train_mask]
    rr = residual[train_mask]
    valid = (pi >= 0) & (bi >= 0) & np.isfinite(rr)
    sums = sparse.coo_matrix(
        (rr[valid], (pi[valid], bi[valid])), shape=(n_pitcher, n_batter)).tocsr()
    counts = sparse.coo_matrix(
        (np.ones(valid.sum(), dtype=np.float32), (pi[valid], bi[valid])),
        shape=(n_pitcher, n_batter)).tocsr()
    if not (np.array_equal(sums.indptr, counts.indptr)
            and np.array_equal(sums.indices, counts.indices)):
        raise RuntimeError("pair sum/count sparse layout mismatch")
    matrix = sums.copy().astype(np.float32)
    matrix.data /= (counts.data + shrink)
    k = min(rank, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=k, n_iter=7, random_state=seed)
    pitcher_emb = svd.fit_transform(matrix)
    batter_emb = svd.components_.T

    idx = np.flatnonzero(row_mask)
    p = pitcher_code[idx]
    b = batter_code[idx]
    out = np.zeros(len(idx), dtype=np.float32)
    known = (p >= 0) & (b >= 0)
    out[known] = np.sum(pitcher_emb[p[known]] * batter_emb[b[known]], axis=1)
    return out, float(svd.explained_variance_ratio_.sum())


def main():
    test_cols = pd.read_csv(f"{DATA_DIR}/test.csv", encoding="utf-8-sig",
                            nrows=0).columns
    raw_features = [c for c in test_cols if c != ID]
    raw = pd.read_csv(f"{DATA_DIR}/train.csv", encoding="utf-8-sig",
                      usecols=raw_features + [TARGET])
    y, seasons = raw[TARGET].to_numpy(float), raw["season"]
    base_df = pd.concat([raw[raw_features], add_derived(raw)], axis=1)
    results = []

    for val_season in VAL_SEASONS:
        tr = (seasons < val_season).to_numpy()
        va = seasons.eq(val_season).to_numpy()
        last_season = int(seasons.loc[tr].max())
        lm = tr & seasons.eq(last_season).to_numpy()
        prior = fit_prior(raw.loc[tr])
        sh = add_shrinkage(base_df, prior, K)
        eb = 0.5 * (sh["sh_asof_pitcher_success_rate"].to_numpy(float)
                    + sh["sh_asof_batter_success_rate"].to_numpy(float))
        residual = y - eb
        pcode, npitcher = encode(raw.loc[tr, "pitcher_id"], raw["pitcher_id"])
        bcode, nbatter = encode(raw.loc[tr, "batter_id"], raw["batter_id"])

        fam = np.load(f"{PRED_DIR}/current_season_family_{val_season}.npz")
        cat = np.load(f"{PRED_DIR}/catboost_confirm_{val_season}.npz")
        command = np.load(f"{PRED_DIR}/catboost_command_confirm_{val_season}.npz")
        p_v8 = 0.25 * fam["p_hgb8"] + 0.75 * fam["p_current16"]
        l_v8 = 0.25 * fam["last_hgb8"] + 0.75 * fam["last_current16"]
        p_cat = 0.5 * cat["p_expert2"] + 0.5 * command["p_all4_expert2"]
        l_cat = 0.5 * cat["last_expert2"] + 0.5 * command["last_all4_expert2"]
        ref_raw = 0.4 * p_v8 + 0.6 * p_cat
        ref_last = 0.4 * l_v8 + 0.6 * l_cat
        ref = command["p_all4_r50"]
        yv = command["y"].astype(float)
        base = yv.mean() * (1 - yv.mean())
        r_extrap = float(fam["r_extrap"])
        saved = dict(y=yv, p_reference=ref)
        log(f"\n[fold={val_season}] pitchers={npitcher} batters={nbatter} "
            f"known_val={(pcode[va]>=0).mean():.3%}/{(bcode[va]>=0).mean():.3%}")

        for rank in RANKS:
            for shrink in SHRINKS:
                pv, pl, evs = [], [], []
                for seed in SEEDS:
                    rv, explained = factor_predictions(
                        pcode, bcode, npitcher, nbatter, residual,
                        tr, va, rank, shrink, seed)
                    rl, _ = factor_predictions(
                        pcode, bcode, npitcher, nbatter, residual,
                        tr, lm, rank, shrink, seed)
                    pv.append(np.clip(eb[va] + rv, 1e-4, 1 - 1e-4))
                    pl.append(np.clip(eb[lm] + rl, 1e-4, 1 - 1e-4))
                    evs.append(explained)
                p_factor = np.mean(pv, axis=0)
                l_factor = np.mean(pl, axis=0)
                key = f"r{rank}_s{int(shrink)}"
                saved[f"p_{key}"] = p_factor
                saved[f"last_{key}"] = l_factor
                for weight in BLEND_WEIGHTS:
                    raw_pred = (1 - weight) * ref_raw + weight * p_factor
                    raw_last = (1 - weight) * ref_last + weight * l_factor
                    pred, shift, _, _ = recenter_like_production(
                        raw_pred, raw_last, r_extrap, 0.5)
                    gain, se = paired(yv, ref, pred, base)
                    results.append(dict(rank=rank, shrink=shrink, weight=weight,
                                        season=val_season, gain=gain, se=se,
                                        shift=shift, explained=np.mean(evs)))
                log(f"  rank={rank:2d} shrink={shrink:5.1f} "
                    f"explained={np.mean(evs):.3f}")
        np.savez_compressed(f"{PRED_DIR}/factorization_expert_{val_season}.npz", **saved)
        del sh, pcode, bcode
        gc.collect()

    result = pd.DataFrame(results)
    result.to_csv(f"{PRED_DIR}/factorization_expert_summary.csv", index=False)
    log("\nSUMMARY factorization blend vs stage2 all4-50")
    for rank in RANKS:
        for shrink in SHRINKS:
            for weight in BLEND_WEIGHTS:
                d = result[(result["rank"].eq(rank))
                           & (result["shrink"].eq(shrink))
                           & (result["weight"].eq(weight))]
                gains = d["gain"].to_numpy()
                if np.all(gains > 0) or weight == 0.10:
                    log(f"r={rank:2d} s={shrink:5.1f} w={weight:.2f} gain="
                        + "/".join(f"{x:+.2f}" for x in gains)
                        + f" mean={gains.mean():+.2f} SE="
                        + "/".join(f"{x:.2f}" for x in d["se"]))


if __name__ == "__main__":
    main()
