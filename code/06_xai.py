"""Step 6 - explainability: TreeSHAP attributions, cross-cohort stability, ablation."""
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
import shap
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

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


def rank_norm(X):
    return np.apply_along_axis(stats.rankdata, 1, X) / X.shape[1]


R = {"train": rank_norm(raw["Xtr"].astype(np.float64))[:, pidx]}
for g in EXT:
    R[g] = rank_norm(raw[f"X_{g}"].astype(np.float64))[:, pidx]
mu_r, sd_r = R["train"].mean(0), R["train"].std(0)
sd_r[sd_r < 1e-9] = 1e-9
N2 = {k: (v - mu_r) / sd_r for k, v in R.items()}
N1 = {"train": dz["Xtr"].astype(np.float64)[:, pidx]}
for g in EXT:
    N1[g] = dz[f"X_{g}"].astype(np.float64)[:, pidx]

rb = json.load(open(f"{W}/robust_results.json"))
BEST = "Gradient Boosting"
bp2 = rb["results"]["N2_sample_rank"][BEST]["best_params"]
bp1 = rb["results"]["N1_cohort_zscore"][BEST]["best_params"]


def mk(bp):
    return GradientBoostingClassifier(random_state=SEED,
                                      n_estimators=int(bp["n_estimators"]),
                                      learning_rate=float(bp["learning_rate"]),
                                      max_depth=int(bp["max_depth"]))


m2 = mk(bp2).fit(N2["train"], ytr)
m1 = mk(bp1).fit(N1["train"], ytr)
print("refit detection models; ext AUROC (N2):",
      {g: round(roc_auc_score(yext[g], m2.predict_proba(N2[g])[:, 1]), 3) for g in EXT}, flush=True)

expl = shap.TreeExplainer(m2)
sv = {k: expl.shap_values(v) for k, v in N2.items()}
sv = {k: (v[1] if isinstance(v, list) else v) for k, v in sv.items()}
base = float(np.atleast_1d(expl.expected_value)[-1])

glob = {k: np.abs(v).mean(0) for k, v in sv.items()}
signed = {k: (v * np.sign(N2[k] - N2["train"].mean(0))).mean(0) for k, v in sv.items()}

# cross-cohort agreement of the global attribution ranking
keys = ["train"] + EXT
agree = np.zeros((len(keys), len(keys)))
for i, a in enumerate(keys):
    for j, b in enumerate(keys):
        agree[i, j] = stats.spearmanr(glob[a], glob[b]).statistic

# multi-method importance agreement on the discovery cohort
pi = permutation_importance(m2, N2["train"], ytr, n_repeats=50, random_state=SEED,
                            scoring="roc_auc").importances_mean
lr = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]).fit(N2["train"], ytr)
lrc = np.abs(lr.named_steps["m"].coef_[0])
gain = m2.feature_importances_
de = np.load(f"{W}/de_stats.npz", allow_pickle=True)
gidx = [list(de["genes"]).index(g) for g in panel]
uni_auc = de["auc"][gidx]
methods = {"SHAP (mean |phi|)": glob["train"], "Permutation importance": pi,
           "GBM split gain": gain, "Logistic |coefficient|": lrc,
           "Univariate AUROC": np.abs(uni_auc - 0.5)}
mnames = list(methods)
magree = np.zeros((len(mnames), len(mnames)))
for i, a in enumerate(mnames):
    for j, b in enumerate(mnames):
        magree[i, j] = stats.spearmanr(methods[a], methods[b]).statistic

# ---------------- leave-one-gene-out ablation ----------------
full_ext = {g: roc_auc_score(yext[g], m2.predict_proba(N2[g])[:, 1]) for g in EXT}
pooled_y = np.concatenate([yext[g] for g in EXT])
full_pool = roc_auc_score(pooled_y, np.concatenate([m2.predict_proba(N2[g])[:, 1] for g in EXT]))
ablation = {}
for i, g_ in enumerate(panel):
    keep = [j for j in range(len(panel)) if j != i]
    mm = mk(bp2).fit(N2["train"][:, keep], ytr)
    pp = np.concatenate([mm.predict_proba(N2[c][:, keep])[:, 1] for c in EXT])
    ablation[g_] = dict(pooled_auroc=float(roc_auc_score(pooled_y, pp)),
                        delta=float(roc_auc_score(pooled_y, pp) - full_pool))
    print("  drop %-8s pooled ext AUROC %.4f (delta %+.4f)"
          % (g_, ablation[g_]["pooled_auroc"], ablation[g_]["delta"]), flush=True)

# ---------------- local explanations ----------------
p_ext = {g: m2.predict_proba(N2[g])[:, 1] for g in EXT}
cases = {}
gp = "GSE13911"
pv, yv = p_ext[gp], yext[gp]
cand = {"confident_tumour": int(np.argmax(np.where(yv == 1, pv, -1))),
        "confident_normal": int(np.argmin(np.where(yv == 0, pv, 2))),
        "false_negative": int(np.argmin(np.where(yv == 1, pv, 2))),
        "false_positive": int(np.argmax(np.where(yv == 0, pv, -1)))}
for k, i in cand.items():
    cases[k] = dict(cohort=gp, index=i, y=int(yv[i]), p=float(pv[i]), base=base,
                    shap={panel[j]: float(sv[gp][i, j]) for j in range(len(panel))},
                    value={panel[j]: float(N2[gp][i, j]) for j in range(len(panel))})

# ---------------- prognostic model SHAP ----------------
pa = np.load(f"{W}/prog_arrays.npz", allow_pickle=True)
pr = json.load(open(f"{W}/prog_results.json"))
ppanel = [str(x) for x in pa["panel"]]
Xu_sel = pa["Xu_sel"]
yprog = pa["y"]
pm = LGBMClassifier(random_state=202, n_jobs=-1, verbose=-1, n_estimators=200,
                    learning_rate=0.03, num_leaves=7, min_child_samples=15).fit(Xu_sel, yprog)
pe = shap.TreeExplainer(pm)
psv = pe.shap_values(Xu_sel)
psv = psv[1] if isinstance(psv, list) else psv
pglob = np.abs(psv).mean(0)

np.savez_compressed(f"{W}/xai.npz",
                    panel=np.array(panel), base=base,
                    **{f"sv_{k}": v for k, v in sv.items()},
                    **{f"X_{k}": N2[k] for k in sv},
                    ytr=ytr, **{f"y_{g}": yext[g] for g in EXT},
                    **{f"p_{g}": p_ext[g] for g in EXT},
                    p_train=m2.predict_proba(N2["train"])[:, 1],
                    agree=agree, keys=np.array(keys),
                    magree=magree, mnames=np.array(mnames),
                    **{f"imp_{i}": methods[m] for i, m in enumerate(mnames)},
                    ppanel=np.array(ppanel), psv=psv, pX=Xu_sel, pglob=pglob, yprog=yprog)
json.dump(dict(global_importance={k: {panel[i]: float(v[i]) for i in range(len(panel))}
                                  for k, v in glob.items()},
               signed={k: {panel[i]: float(v[i]) for i in range(len(panel))} for k, v in signed.items()},
               cohort_agreement=agree.tolist(), keys=keys,
               method_agreement=magree.tolist(), method_names=mnames,
               ablation=ablation, full_pooled=float(full_pool),
               full_ext={k: float(v) for k, v in full_ext.items()},
               cases=cases,
               prog_importance={ppanel[i]: float(pglob[i]) for i in range(len(ppanel))}),
          open(f"{W}/xai_results.json", "w"), indent=2)
print("done", flush=True)
