"""Step 5 - cross-cohort robustness: why external specificity collapses, and how to fix it.

N1  cohort-wise gene z-score              (composition-DEPENDENT: uses the cohort's own case mix)
N2  within-sample percentile rank         (composition-INDEPENDENT: each sample normalised alone)
Also: an explicit case-mix sensitivity experiment and a threshold/recalibration analysis.
"""
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss
from xgboost import XGBClassifier
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
print("panel:", panel, flush=True)


def rank_norm(X):
    """Within-sample percentile rank across all genes: invariant to any per-sample
    monotone rescaling and computed from a single sample, hence independent of the
    case mix of the batch it happens to arrive in."""
    return np.apply_along_axis(stats.rankdata, 1, X) / X.shape[1]


def zcol(X):
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd < 1e-9] = 1e-9
    return (X - mu) / sd


print("building N2 (within-sample rank) representations ...", flush=True)
R = {"train": rank_norm(raw["Xtr"].astype(np.float64))[:, pidx]}
for g in EXT:
    R[g] = rank_norm(raw[f"X_{g}"].astype(np.float64))[:, pidx]
mu_r, sd_r = R["train"].mean(0), R["train"].std(0)          # frozen at training time
sd_r[sd_r < 1e-9] = 1e-9
N2 = {k: (v - mu_r) / sd_r for k, v in R.items()}

N1 = {"train": dz["Xtr"].astype(np.float64)[:, pidx]}
for g in EXT:
    N1[g] = dz[f"X_{g}"].astype(np.float64)[:, pidx]


def models():
    return {
        "Logistic Regression (L2)": (Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]),
                                     {"m__C": [0.01, 0.1, 1, 10, 100]}),
        "Elastic-Net Logistic": (Pipeline([("s", StandardScaler()),
                                           ("m", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=8000))]),
                                 {"m__C": [0.1, 1, 10], "m__l1_ratio": [0.2, 0.5, 0.8]}),
        "Gaussian Naive Bayes": (Pipeline([("s", StandardScaler()), ("m", GaussianNB())]),
                                 {"m__var_smoothing": [1e-9, 1e-7, 1e-5]}),
        "k-Nearest Neighbours": (Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier())]),
                                 {"m__n_neighbors": [3, 5, 9, 15], "m__weights": ["uniform", "distance"]}),
        "SVM (RBF kernel)": (Pipeline([("s", StandardScaler()), ("m", SVC(probability=True, random_state=SEED))]),
                             {"m__C": [0.1, 1, 10, 100], "m__gamma": ["scale", 0.05, 0.2]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                          {"n_estimators": [400], "max_depth": [None, 6], "min_samples_leaf": [1, 3]}),
        "Extremely Randomised Trees": (ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
                                       {"n_estimators": [400], "max_depth": [None, 6], "min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=SEED),
                              {"n_estimators": [200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
        "XGBoost": (XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1, verbosity=0),
                    {"n_estimators": [300], "learning_rate": [0.05, 0.1], "max_depth": [2, 4],
                     "subsample": [0.8], "colsample_bytree": [0.8]}),
        "LightGBM": (LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
                     {"n_estimators": [300], "learning_rate": [0.05, 0.1], "num_leaves": [7, 15],
                      "min_child_samples": [10]}),
        "Multilayer Perceptron": (Pipeline([("s", StandardScaler()), ("m", MLPClassifier(max_iter=3000, random_state=SEED))]),
                                  {"m__hidden_layer_sizes": [(32,), (64, 32)], "m__alpha": [1e-4, 1e-2]}),
    }


def met(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(auroc=float(roc_auc_score(y, p)), auprc=float(average_precision_score(y, p)),
                accuracy=float((tp + tn) / len(y)),
                sensitivity=float(tp / max(tp + fn, 1)), specificity=float(tn / max(tn + fp, 1)),
                ppv=float(tp / max(tp + fp, 1)), npv=float(tn / max(tn + fn, 1)),
                balanced=float(0.5 * (tp / max(tp + fn, 1) + tn / max(tn + fp, 1))),
                brier=float(brier_score_loss(y, p)))


def boot_ci(y, p, fn, n=2000, seed=5):
    r = np.random.default_rng(seed)
    o = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        try:
            o.append(fn(y[i], p[i]))
        except Exception:
            pass
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if o else (np.nan, np.nan)


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)
splits = list(cv.split(N1["train"], ytr))
out = {"N1_cohort_zscore": {}, "N2_sample_rank": {}}
preds = {}
for tag, D in [("N1_cohort_zscore", N1), ("N2_sample_rank", N2)]:
    print("=== %s ===" % tag, flush=True)
    for name, (est, grid) in models().items():
        p_oof = np.zeros(len(ytr))
        cnt = np.zeros(len(ytr))
        for tr, te in splits:
            gs = GridSearchCV(est, grid, scoring="roc_auc",
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
            gs.fit(D["train"][tr], ytr[tr])
            p_oof[te] += gs.best_estimator_.predict_proba(D["train"][te])[:, 1]
            cnt[te] += 1
        p_oof /= np.maximum(cnt, 1)
        gs = GridSearchCV(est, grid, scoring="roc_auc",
                          cv=StratifiedKFold(10, shuffle=True, random_state=SEED), n_jobs=-1)
        gs.fit(D["train"], ytr)
        rec = {"internal_cv": met(ytr, p_oof),
               "best_params": {k: str(v) for k, v in gs.best_params_.items()}}
        rec["internal_cv"]["auroc_ci"] = boot_ci(ytr, p_oof, roc_auc_score)
        preds[f"{tag}|{name}|oof"] = p_oof
        for g in EXT:
            p = gs.best_estimator_.predict_proba(D[g])[:, 1]
            preds[f"{tag}|{name}|{g}"] = p
            rec[g] = met(yext[g], p)
            rec[g]["auroc_ci"] = boot_ci(yext[g], p, roc_auc_score)
        yp = np.concatenate([yext[g] for g in EXT])
        pp = np.concatenate([preds[f"{tag}|{name}|{g}"] for g in EXT])
        rec["POOLED"] = met(yp, pp)
        rec["POOLED"]["auroc_ci"] = boot_ci(yp, pp, roc_auc_score)
        out[tag][name] = rec
        print("  %-28s intCV=%.4f | pooled ext AUROC=%.3f Sn=%.2f Sp=%.2f BA=%.3f"
              % (name, rec["internal_cv"]["auroc"], rec["POOLED"]["auroc"],
                 rec["POOLED"]["sensitivity"], rec["POOLED"]["specificity"],
                 rec["POOLED"]["balanced"]), flush=True)

# ---------------- case-mix sensitivity experiment ----------------
print("case-mix sensitivity experiment ...", flush=True)
best_name = "Gradient Boosting"
est, grid = models()[best_name]
fits = {}
for tag, D in [("N1_cohort_zscore", N1), ("N2_sample_rank", N2)]:
    gs = GridSearchCV(est, grid, scoring="roc_auc",
                      cv=StratifiedKFold(10, shuffle=True, random_state=SEED), n_jobs=-1)
    gs.fit(D["train"], ytr)
    fits[tag] = gs.best_estimator_

pool_raw = np.vstack([raw[f"X_{g}"].astype(np.float64) for g in ["GSE13911", "GSE79973"]])
pool_y = np.concatenate([yext["GSE13911"], yext["GSE79973"]])
pool_rank = rank_norm(pool_raw)[:, pidx]
Ti = np.where(pool_y == 1)[0]
Ni = np.where(pool_y == 0)[0]
fracs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
mix = {"N1_cohort_zscore": {}, "N2_sample_rank": {}}
rng = np.random.default_rng(11)
NB = 60
for f in fracs:
    acc = {k: {"sp": [], "sn": [], "auc": []} for k in mix}
    for _ in range(NB):
        n_tot = 40
        nt = max(2, int(round(f * n_tot)))
        nn = max(2, n_tot - nt)
        it = rng.choice(Ti, min(nt, len(Ti)), replace=False)
        inn = rng.choice(Ni, min(nn, len(Ni)), replace=False)
        idx = np.concatenate([it, inn])
        yy = pool_y[idx]
        # N1: z-score computed WITHIN this synthetic batch -> depends on its case mix
        x1 = zcol(pool_raw[idx][:, pidx])
        p1 = fits["N1_cohort_zscore"].predict_proba(x1)[:, 1]
        # N2: each sample normalised on its own -> unaffected by the batch composition
        x2 = (pool_rank[idx] - mu_r) / sd_r
        p2 = fits["N2_sample_rank"].predict_proba(x2)[:, 1]
        for k, p in [("N1_cohort_zscore", p1), ("N2_sample_rank", p2)]:
            m = met(yy, p)
            acc[k]["sp"].append(m["specificity"])
            acc[k]["sn"].append(m["sensitivity"])
            acc[k]["auc"].append(m["auroc"])
    for k in mix:
        mix[k][str(f)] = {m: [float(np.mean(acc[k][m])), float(np.std(acc[k][m]))] for m in ["sp", "sn", "auc"]}
    print("  frac=%.1f  N1 Sp=%.2f Sn=%.2f | N2 Sp=%.2f Sn=%.2f"
          % (f, mix["N1_cohort_zscore"][str(f)]["sp"][0], mix["N1_cohort_zscore"][str(f)]["sn"][0],
             mix["N2_sample_rank"][str(f)]["sp"][0], mix["N2_sample_rank"][str(f)]["sn"][0]), flush=True)

# ---------------- threshold / recalibration analysis ----------------
thr_an = {}
for tag in out:
    thr_an[tag] = {}
    for g in EXT + ["POOLED"]:
        if g == "POOLED":
            yy = np.concatenate([yext[c] for c in EXT])
            pp = np.concatenate([preds[f"{tag}|{best_name}|{c}"] for c in EXT])
        else:
            yy, pp = yext[g], preds[f"{tag}|{best_name}|{g}"]
        gridt = np.unique(np.round(pp, 4))
        j = [0.5 * (met(yy, pp, t)["sensitivity"] + met(yy, pp, t)["specificity"]) for t in gridt]
        t_opt = float(gridt[int(np.argmax(j))])
        thr_an[tag][g] = {"fixed_0.5": met(yy, pp, 0.5), "oracle_youden": met(yy, pp, t_opt),
                          "t_opt": t_opt}

json.dump({"results": out, "case_mix": mix, "fracs": fracs, "threshold": thr_an,
           "panel": panel, "n_bootstrap_mix": NB},
          open(f"{W}/robust_results.json", "w"), indent=2)
np.savez_compressed(f"{W}/robust_preds.npz", ytr=ytr,
                    **{f"y_{g}": yext[g] for g in EXT},
                    **{k.replace("|", "__").replace(" ", "_"): v for k, v in preds.items()})
print("done", flush=True)
