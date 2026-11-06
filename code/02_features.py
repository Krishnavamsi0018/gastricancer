"""Step 2: harmonisation, differential expression and leakage-free panel selection."""
import json, os
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(20240915)
CAND_CAP = 2000
W, OUT = "work", "work"
DISC = "GSE66229"; EXT = ["GSE54129", "GSE13911", "GSE79973"]

def load(gse):
    d = np.load(f"{W}/{gse}.npz", allow_pickle=True)
    return d["X"].astype(np.float64), list(d["genes"]), d["y"].astype(int), list(d["samples"])

def zscore_cohort(X):
    """Gene-wise z-score within a cohort: unsupervised, applied to each study
    independently, removing study-specific location/scale (batch) effects."""
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True); sd[sd < 1e-9] = 1e-9
    return (X - mu) / sd

# ---------------- load & align ----------------
Xd, genes_d, yd, sd_ = load(DISC)
common = set(genes_d)
store = {}
for g in EXT:
    Xe, ge, ye, se = load(g); store[g] = (Xe, ge, ye, se); common &= set(ge)
common = sorted(common)
print(f"Common genes across GPL570 detection cohorts: {len(common)}")
gi = {g: i for i, g in enumerate(genes_d)}
Xd_raw = Xd[[gi[g] for g in common], :]
Xd_z = zscore_cohort(Xd_raw).T                       # samples x genes
ext = {}
for g in EXT:
    Xe, ge, ye, se = store[g]
    idx = {k: i for i, k in enumerate(ge)}
    ext[g] = (zscore_cohort(Xe[[idx[k] for k in common], :]).T, ye, se)

np.savez_compressed(f"{OUT}/detect_matrices.npz",
                    Xtr=Xd_z.astype(np.float32), ytr=yd, genes=np.array(common),
                    **{f"X_{g}": ext[g][0].astype(np.float32) for g in EXT},
                    **{f"y_{g}": ext[g][1] for g in EXT})
# raw (un-z-scored) matrices kept for the batch-effect figure
np.savez_compressed(f"{OUT}/detect_raw.npz",
                    Xtr=Xd_raw.T.astype(np.float32),
                    **{f"X_{g}": store[g][0][[{k: i for i, k in enumerate(store[g][1])}[k]
                                              for k in common], :].T.astype(np.float32) for g in EXT})

# ---------------- differential expression (training cohort only) ----------------
A, B = Xd_z[yd == 1], Xd_z[yd == 0]
t, p = stats.ttest_ind(A, B, equal_var=False)
order = np.argsort(p); ranked = np.empty_like(order)
ranked[order] = np.arange(1, len(p) + 1)
q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
qv = np.empty_like(q); qv[order] = np.clip(q, 0, 1)
delta = A.mean(0) - B.mean(0)                      # standardised mean difference
def fast_auc(X, y):
    n1 = int((y == 1).sum()); n0 = int((y == 0).sum())
    r = np.apply_along_axis(stats.rankdata, 0, X)
    return (r[y == 1].sum(axis=0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
auc_uni = fast_auc(Xd_z, yd)
np.savez_compressed(f"{OUT}/de_stats.npz", genes=np.array(common), t=t, p=p, q=qv,
                    delta=delta, auc=auc_uni)
keep = (qv < 0.01) & (np.abs(delta) > 0.5)
print(f"DE genes at FDR<0.01 and |delta|>0.5 : {keep.sum()}")

# ---------------- nested-CV panel-size sweep ----------------
def select_panel(Xtr, ytr, k, n_boot=20, seed=0):
    """Filter (FDR + effect size) then L1 stability selection; return indices of top-k."""
    r = np.random.default_rng(seed)
    a, b = Xtr[ytr == 1], Xtr[ytr == 0]
    tt, pp = stats.ttest_ind(a, b, equal_var=False)
    o = np.argsort(pp)
    qq = np.minimum.accumulate((pp[o] * len(pp) / np.arange(1, len(pp) + 1))[::-1])[::-1]
    qv_ = np.empty_like(qq); qv_[o] = np.clip(qq, 0, 1)
    d_ = a.mean(0) - b.mean(0)
    cand = np.where((qv_ < 0.01) & (np.abs(d_) > 0.5))[0]
    if len(cand) < k:
        cand = np.argsort(pp)[:max(k, 500)]
    if len(cand) > CAND_CAP:                      # keep the CAND_CAP most significant
        cand = cand[np.argsort(pp[cand])[:CAND_CAP]]
    freq = np.zeros(len(cand))
    n = len(ytr)
    for b_ in range(n_boot):
        idx = r.choice(n, size=int(0.8 * n), replace=False)
        if len(np.unique(ytr[idx])) < 2:
            continue
        lr = LogisticRegression(penalty="l1", C=0.06, solver="liblinear", max_iter=3000)
        lr.fit(Xtr[np.ix_(idx, cand)], ytr[idx])
        freq += (np.abs(lr.coef_[0]) > 1e-8).astype(float)
    freq /= max(n_boot, 1)
    tie = -np.abs(d_[cand])
    ordering = np.lexsort((tie, -freq))
    return cand[ordering[:k]], freq, cand

KS = [5, 8, 10, 12, 15, 20, 25, 30, 40, 60, 100]
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=7)
fold_auc = {k: [] for k in KS}
for f, (tr, te) in enumerate(cv.split(Xd_z, yd)):
    sel_full, _, _ = select_panel(Xd_z[tr], yd[tr], max(KS), seed=f)
    for k in KS:
        s = sel_full[:k]
        clf = LogisticRegression(C=1.0, max_iter=5000)
        clf.fit(Xd_z[np.ix_(tr, s)], yd[tr])
        fold_auc[k].append(roc_auc_score(yd[te], clf.predict_proba(Xd_z[np.ix_(te, s)])[:, 1]))
    print(f"  outer fold {f+1}/30 done", flush=True)

means = {k: float(np.mean(v)) for k, v in fold_auc.items()}
sems = {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in fold_auc.items()}
best_k = max(means, key=means.get)
thresh = means[best_k] - sems[best_k]              # one-standard-error parsimony rule
k_star = min([k for k in KS if means[k] >= thresh])
print("panel sweep:", json.dumps({str(k): round(means[k], 4) for k in KS}, indent=0))
print(f"best k={best_k} (AUC {means[best_k]:.4f}); parsimonious k*={k_star} (AUC {means[k_star]:.4f})")

sel, freq, cand = select_panel(Xd_z, yd, k_star, n_boot=200, seed=99)
np.savez_compressed(f'{OUT}/stability.npz', freq=freq, cand=cand, genes=np.array(common))
panel = [common[i] for i in sel]
print("PANEL:", panel)
json.dump(dict(ks=KS, mean=means, sem=sems, best_k=best_k, k_star=k_star,
               panel=panel, panel_idx=[int(i) for i in sel],
               fold_auc={str(k): v for k, v in fold_auc.items()},
               n_de=int(keep.sum())),
          open(f"{OUT}/panel.json", "w"), indent=2)
