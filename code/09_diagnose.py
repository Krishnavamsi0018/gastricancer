"""Step 9 - decisive diagnostics.

(A) Within-cohort case-mix manipulation: hold the cohort fixed, vary only its tumour
    fraction, and see whether N1 specificity moves as the stress test predicts.
(B) Paired significance tests for N2 vs N1 (bootstrap dAUROC, McNemar).
(C) Runtime and complexity measurements for the implementation section.
"""
import json, time, warnings, numpy as np
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

W = "work"
EXT = ["GSE54129", "GSE13911", "GSE79973"]
SEED = 42

raw = np.load(f"{W}/detect_raw.npz", allow_pickle=True)
dz = np.load(f"{W}/detect_matrices.npz", allow_pickle=True)
genes = list(dz["genes"])
ytr = dz["ytr"].astype(int)
yext = {g: dz[f"y_{g}"].astype(int) for g in EXT}
panel = json.load(open(f"{W}/panel.json"))["panel"]
pidx = [genes.index(g) for g in panel]
rb = json.load(open(f"{W}/robust_results.json"))


def rank_norm(X):
    return np.apply_along_axis(stats.rankdata, 1, X) / X.shape[1]


def zcol(X):
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True)
    sd[sd < 1e-9] = 1e-9
    return (X - mu) / sd


R = {"train": rank_norm(raw["Xtr"].astype(np.float64))[:, pidx]}
for g in EXT:
    R[g] = rank_norm(raw[f"X_{g}"].astype(np.float64))[:, pidx]
mu_r, sd_r = R["train"].mean(0), R["train"].std(0)
sd_r[sd_r < 1e-9] = 1e-9
N2 = {k: (v - mu_r) / sd_r for k, v in R.items()}
N1 = {"train": dz["Xtr"].astype(np.float64)[:, pidx]}
for g in EXT:
    N1[g] = dz[f"X_{g}"].astype(np.float64)[:, pidx]


def mk(bp):
    return GradientBoostingClassifier(random_state=SEED, n_estimators=int(bp["n_estimators"]),
                                      learning_rate=float(bp["learning_rate"]),
                                      max_depth=int(bp["max_depth"]))


m1 = mk(rb["results"]["N1_cohort_zscore"]["Gradient Boosting"]["best_params"]).fit(N1["train"], ytr)
m2 = mk(rb["results"]["N2_sample_rank"]["Gradient Boosting"]["best_params"]).fit(N2["train"], ytr)

# ============ (A) within-cohort case-mix manipulation ============
print("=" * 78)
print("(A) WITHIN-COHORT CASE-MIX MANIPULATION")
print("    Cohort held fixed; only its tumour fraction is varied before N1 z-scoring.")
print("=" * 78)
out = {}
rng = np.random.default_rng(7)
for g in EXT:
    Xr = raw[f"X_{g}"].astype(np.float64)
    y = yext[g]
    Ti, Ni = np.where(y == 1)[0], np.where(y == 0)[0]
    out[g] = {}
    print(f"\n  {g}  ({len(Ti)} tumour / {len(Ni)} normal, native {100*len(Ti)/len(y):.0f}% tumour)")
    print(f"    {'target %T':>9s} {'actual %T':>9s} {'n':>4s} | {'N1 Sp':>6s} {'N1 AUROC':>8s} | {'N2 Sp':>6s}")
    for f in [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]:
        sp1, au1, sp2, act = [], [], [], []
        for rep in range(40):
            # keep every normal, subsample tumours to hit the target fraction
            nt = int(round(f * len(Ni) / (1 - f)))
            if nt < 3 or nt > len(Ti):
                continue
            it = rng.choice(Ti, nt, replace=False)
            idx = np.concatenate([it, Ni])
            yy = y[idx]
            act.append(yy.mean())
            p1 = m1.predict_proba(zcol(Xr[idx])[:, pidx])[:, 1]
            p2 = m2.predict_proba(N2[g][idx])[:, 1]
            tn, fp, fn, tp = confusion_matrix(yy, (p1 >= 0.5).astype(int), labels=[0, 1]).ravel()
            sp1.append(tn / max(tn + fp, 1))
            au1.append(roc_auc_score(yy, p1))
            tn2, fp2, _, _ = confusion_matrix(yy, (p2 >= 0.5).astype(int), labels=[0, 1]).ravel()
            sp2.append(tn2 / max(tn2 + fp2, 1))
        if not sp1:
            continue
        out[g][str(f)] = dict(sp1=float(np.mean(sp1)), sp1_sd=float(np.std(sp1)),
                              auc1=float(np.mean(au1)), sp2=float(np.mean(sp2)),
                              actual=float(np.mean(act)))
        print(f"    {f:9.2f} {np.mean(act):9.2f} {len(Ni)+nt:4d} | {np.mean(sp1):6.2f} "
              f"{np.mean(au1):8.3f} | {np.mean(sp2):6.2f}")

# ============ (B) paired significance tests ============
print("\n" + "=" * 78)
print("(B) PAIRED SIGNIFICANCE TESTS, N2 vs N1 (pooled external, n=221)")
print("=" * 78)
yy = np.concatenate([yext[g] for g in EXT])
p1 = np.concatenate([m1.predict_proba(N1[g])[:, 1] for g in EXT])
p2 = np.concatenate([m2.predict_proba(N2[g])[:, 1] for g in EXT])
a1, a2 = roc_auc_score(yy, p1), roc_auc_score(yy, p2)
rng = np.random.default_rng(11)
d = []
for _ in range(5000):
    i = rng.integers(0, len(yy), len(yy))
    if len(np.unique(yy[i])) < 2:
        continue
    d.append(roc_auc_score(yy[i], p2[i]) - roc_auc_score(yy[i], p1[i]))
d = np.array(d)
lo, hi = np.percentile(d, [2.5, 97.5])
pval = 2 * min((d <= 0).mean(), (d >= 0).mean())
print(f"  AUROC N1 = {a1:.4f}   AUROC N2 = {a2:.4f}")
print(f"  paired bootstrap dAUROC = {a2-a1:+.4f}  95% CI ({lo:+.4f}, {hi:+.4f})  p = {max(pval,1/len(d)):.4f}")
yh1, yh2 = (p1 >= 0.5).astype(int), (p2 >= 0.5).astype(int)
b = int(((yh1 == yy) & (yh2 != yy)).sum())
c = int(((yh1 != yy) & (yh2 == yy)).sum())
chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) else 0.0
p_mc = stats.chi2.sf(chi2, 1)
print(f"  McNemar: N1-only correct b={b}, N2-only correct c={c}, chi2={chi2:.2f}, p={p_mc:.2e}")
mc = {}
for g in EXT:
    q1 = m1.predict_proba(N1[g])[:, 1] >= 0.5
    q2 = m2.predict_proba(N2[g])[:, 1] >= 0.5
    yg = yext[g]
    bb = int(((q1 == yg) & (q2 != yg)).sum())
    cc = int(((q1 != yg) & (q2 == yg)).sum())
    x = (abs(bb - cc) - 1) ** 2 / (bb + cc) if (bb + cc) else 0.0
    mc[g] = dict(b=bb, c=cc, chi2=float(x), p=float(stats.chi2.sf(x, 1)))
    print(f"    {g}: b={bb} c={cc} chi2={x:.2f} p={stats.chi2.sf(x,1):.4f}")

# decomposition: ranking loss vs threshold loss, per cohort
print("\n  Decomposition of the N1 deficit per cohort:")
dec = {}
for g in EXT:
    q1 = m1.predict_proba(N1[g])[:, 1]
    q2 = m2.predict_proba(N2[g])[:, 1]
    yg = yext[g]
    au1, au2 = roc_auc_score(yg, q1), roc_auc_score(yg, q2)
    grid = np.unique(np.round(q1, 4))
    ba = []
    for t in grid:
        tn, fp, fn, tp = confusion_matrix(yg, (q1 >= t).astype(int), labels=[0, 1]).ravel()
        ba.append(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1)))
    tn, fp, fn, tp = confusion_matrix(yg, (q1 >= 0.5).astype(int), labels=[0, 1]).ravel()
    ba_fixed = 0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))
    dec[g] = dict(auc_n1=float(au1), auc_n2=float(au2), ba_fixed=float(ba_fixed),
                  ba_oracle=float(max(ba)), threshold_loss=float(max(ba) - ba_fixed),
                  ranking_loss=float(au2 - au1))
    print(f"    {g}: ranking loss (AUROC N2-N1) = {au2-au1:+.3f} | "
          f"threshold loss (oracle BA - fixed BA under N1) = {max(ba)-ba_fixed:+.3f}")

# ============ (C) runtime and complexity ============
print("\n" + "=" * 78)
print("(C) RUNTIME")
print("=" * 78)
t0 = time.perf_counter()
_ = rank_norm(raw["X_GSE13911"].astype(np.float64))
t_rank = (time.perf_counter() - t0) / raw["X_GSE13911"].shape[0]
t0 = time.perf_counter()
for _ in range(10):
    mk(rb["results"]["N2_sample_rank"]["Gradient Boosting"]["best_params"]).fit(N2["train"], ytr)
t_fit = (time.perf_counter() - t0) / 10
t0 = time.perf_counter()
for _ in range(200):
    m2.predict_proba(N2["GSE79973"][:1])
t_inf = (time.perf_counter() - t0) / 200
print(f"  rank transform per specimen (22,880 genes) : {t_rank*1000:.1f} ms")
print(f"  gradient-boosting fit (400 x 8)            : {t_fit*1000:.0f} ms")
print(f"  single-specimen inference                  : {t_inf*1000:.2f} ms")
runtime = dict(rank_ms=t_rank * 1000, fit_ms=t_fit * 1000, infer_ms=t_inf * 1000)

json.dump(dict(case_mix_within=out, delta_auc=dict(a1=a1, a2=a2, delta=a2 - a1,
                                                   lo=float(lo), hi=float(hi),
                                                   p=float(max(pval, 1 / len(d)))),
               mcnemar_pooled=dict(b=b, c=c, chi2=float(chi2), p=float(p_mc)),
               mcnemar_cohort=mc, decomposition=dec, runtime=runtime),
          open(f"{W}/diagnostics.json", "w"), indent=2)
print("\nsaved work/diagnostics.json")
