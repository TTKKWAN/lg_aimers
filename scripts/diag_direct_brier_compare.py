"""저장된 Direct-Brier screen 예측과 v4 classifier 앙상블 비교 (재학습 없음)."""
import csv
import numpy as np

PRED_DIR = "./experiments/preds"
SEASONS = [2022, 2023, 2024]


def bss(y, p):
    r = y.mean(); base = r * (1-r)
    br = np.mean((p-y)**2)
    return br, max(0.0, 100000*(1-br/base)), base


def paired(y, ref, new):
    base = y.mean() * (1-y.mean())
    d = ((ref-y)**2 - (new-y)**2) / base * 100000
    return d.mean(), d.std(ddof=1)/np.sqrt(len(d))


def decomp(y, p, bins=20):
    order = np.argsort(p, kind="stable")
    r = y.mean(); rel=res=ece=0.0
    for ix in np.array_split(order, bins):
        w=len(ix)/len(y); pk=p[ix].mean(); yk=y[ix].mean()
        rel += w*(pk-yk)**2; res += w*(yk-r)**2; ece += w*abs(pk-yk)
    return rel,res,ece


def describe(label,y,p):
    br,s,_=bss(y,p); rel,res,ece=decomp(y,p)
    print(f"    {label:<23} brier={br:.8f} BSS={s:8.2f} "
          f"reliability={rel:.8f} resolution={res:.8f} ECE={ece:.6f}",flush=True)
    return s


def load(season):
    h=np.load(f"{PRED_DIR}/ens_hgb_{season}.npz")
    l=np.load(f"{PRED_DIR}/ens_lgbm_{season}.npz")
    z=np.load(f"{PRED_DIR}/direct_brier_screen_{season}.npz")
    assert np.array_equal(h["y"],l["y"]) and np.array_equal(h["y"],z["y"])
    nh,nl=int(h["n"]),int(l["n"])
    return h["y"],(nh*h["p"]+nl*l["p"])/(nh+nl),z["p_reg"]


def convex_w(y,c,r):
    d=c-r
    # Dot products are deliberately expressed as sums; this avoids BLAS oversubscription.
    return float(np.clip(np.sum(d*(y-r))/np.sum(d*d),0,1))


rows=[]; pooled={"y":[],"c":[],"r":[]}
for season in SEASONS:
    y,c,r=load(season)
    print(f"\n[screen compare] fold={season}",flush=True)
    sc=describe("classifier v4 5+3",y,c); sr=describe("single direct LGBM",y,r)
    d,se=paired(y,c,r); print(f"    direct-v4 diff={d:+.2f} (SE={se:.2f})",flush=True)
    row={"season":season,"classifier_bss":sc,"regressor_bss":sr,
         "direct_diff":d,"direct_se":se}
    for wc in (.75,.5,.25):
        p=wc*c+(1-wc)*r; sm=describe(f"mix cls/reg {wc:.2f}/{1-wc:.2f}",y,p)
        dm,sem=paired(y,c,p)
        print(f"      mix-v4 diff={dm:+.2f} (SE={sem:.2f})",flush=True)
        row[f"mix_{wc:.2f}_bss"]=sm; row[f"mix_{wc:.2f}_diff"]=dm; row[f"mix_{wc:.2f}_se"]=sem
    wf=convex_w(y,c,r); row["fold_oracle_w_cls"]=wf
    print(f"    fold oracle classifier weight={wf:.6f}",flush=True)
    rows.append(row)
    for k,a in (("y",y),("c",c),("r",r)): pooled[k].append(a)

yy=np.concatenate(pooled["y"]); cc=np.concatenate(pooled["c"]); rr=np.concatenate(pooled["r"])
wp=convex_w(yy,cc,rr)
print(f"\n[OOF pooled convex] classifier={wp:.6f}, regressor={1-wp:.6f}",flush=True)
for row,season in zip(rows,SEASONS):
    y,c,r=load(season); p=wp*c+(1-wp)*r
    sp=describe(f"pooled-weight fold {season}",y,p); d,se=paired(y,c,p)
    print(f"      pooled-weight-v4 diff={d:+.2f} (SE={se:.2f})",flush=True)
    row.update(pooled_w_cls=wp,pooled_bss=sp,pooled_diff=d,pooled_se=se)

with open(f"{PRED_DIR}/direct_brier_screen_summary.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
