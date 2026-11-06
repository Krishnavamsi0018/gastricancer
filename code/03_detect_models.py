"""Step 3 - Task 1 (detection): nested-CV model comparison + 3-cohort external validation."""
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
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             f1_score, matthews_corrcoef, brier_score_loss, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

W = "work"
EXT = ["GSE54129", "GSE13911", "GSE79973"]
SEED = 42

d = np.load(f"{W}/detect_matrices.npz", allow_pickle=True)
Xtr, ytr, genes = d["Xtr"].astype(np.float64), d["ytr"].astype(int), list(d["genes"])
ext = {g: (d[f"X_{g}"].astype(np.float64), d[f"y_{g}"].astype(int)) for g in EXT}
panel = json.load(open(f"{W}/panel.json"))["panel"]
K = len(panel)
print("Panel (k=%d): %s" % (K, panel), flush=True)


def models():
    return {
        "Logistic Regression (L2)": (
            Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]),
            {"m__C": [0.01, 0.1, 1, 10, 100]}),
        "Elastic-Net Logistic": (
            Pipeline([("s", StandardScaler()),
                      ("m", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=8000))]),
            {"m__C": [0.1, 1, 10], "m__l1_ratio": [0.2, 0.5, 0.8]}),
        "Gaussian Naive Bayes": (
            Pipeline([("s", StandardScaler()), ("m", GaussianNB())]),
            {"m__var_smoothing": [1e-9, 1e-7, 1e-5]}),
        "k-Nearest Neighbours": (
            Pipeline([("s", StandardScaler()), ("m", KNeighborsClassifier())]),
            {"m__n_neighbors": [3, 5, 9, 15], "m__weights": ["uniform", "distance"]}),
        "SVM (RBF kernel)": (
            Pipeline([("s", StandardScaler()), ("m", SVC(probability=True, random_state=SEED))]),
            {"m__C": [0.1, 1, 10, 100], "m__gamma": ["scale", 0.05, 0.2]}),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"n_estimators": [400], "max_depth": [None, 6], "min_samples_leaf": [1, 3]}),
        "Extremely Randomised Trees": (
            ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
            {"n_estimators": [400], "max_depth": [None, 6], "min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": [200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]}),
        "XGBoost": (
            XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1, verbosity=0),
            {"n_estimators": [300], "learning_rate": [0.05, 0.1], "max_depth": [2, 4],
             "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0]}),
        "LightGBM": (
            LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
            {"n_estimators": [300], "learning_rate": [0.05, 0.1], "num_leaves": [7, 15],
             "min_child_samples": [10]}),
        "Multilayer Perceptron": (
            Pipeline([("s", StandardScaler()), ("m", MLPClassifier(max_iter=3000, random_state=SEED))]),
            {"m__hidden_layer_sizes": [(32,), (64, 32)], "m__alpha": [1e-4, 1e-2]}),
    }


def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p),
                accuracy=accuracy_score(y, yh),
                sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1),
                ppv=tp / max(tp + fp, 1), npv=tn / max(tn + fn, 1),
                f1=f1_score(y, yh, zero_division=0), mcc=matthews_corrcoef(y, yh),
                brier=brier_score_loss(y, p), tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def boot_ci(y, p, fn, n=2000, seed=1):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = r.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        try:
            out.append(fn(y[i], p[i]))
        except Exception:
            pass
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def fold_panel(X, y, k, seed):
    """Panel selection repeated inside every outer training fold (leakage-free)."""
    a, b = X[y == 1], X[y == 0]
    tt, pp = stats.ttest_ind(a, b, equal_var=False)
    o = np.argsort(pp)
    qq = np.minimum.accumulate((pp[o] * len(pp) / np.arange(1, len(pp) + 1))[::-1])[::-1]
    qv = np.empty_like(qq)
    qv[o] = np.clip(qq, 0, 1)
    dl = a.mean(0) - b.mean(0)
    cand = np.where((qv < 0.01) & (np.abs(dl) > 0.5))[0]
    if len(cand) > 2000:
        cand = cand[np.argsort(pp[cand])[:2000]]
    r = np.random.default_rng(seed)
    freq = np.zeros(len(cand))
    n = len(y)
    for _ in range(20):
        i = r.choice(n, int(0.8 * n), replace=False)
        if len(np.unique(y[i])) < 2:
            continue
        lr = LogisticRegression(penalty="l1", C=0.06, solver="liblinear", max_iter=3000)
        lr.fit(X[np.ix_(i, cand)], y[i])
        freq += (np.abs(lr.coef_[0]) > 1e-8)
    return cand[np.lexsort((-np.abs(dl[cand]), -freq))[:k]]


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=SEED)
splits = list(cv.split(Xtr, ytr))
print("Pre-computing per-fold panels ...", flush=True)
fpanels = [fold_panel(Xtr[tr], ytr[tr], K, seed=i) for i, (tr, te) in enumerate(splits)]
json.dump([[genes[j] for j in fp] for fp in fpanels], open(f"{W}/fold_panels.json", "w"))
print("fold panels done", flush=True)

oof = {}
results = {}
for name, (est, grid) in models().items():
    oof_p = np.zeros(len(ytr))
    cnt = np.zeros(len(ytr))
    fold_auc = []
    for i, (tr, te) in enumerate(splits):
        s = fpanels[i]
        gs = GridSearchCV(est, grid, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
        gs.fit(Xtr[np.ix_(tr, s)], ytr[tr])
        p = gs.best_estimator_.predict_proba(Xtr[np.ix_(te, s)])[:, 1]
        oof_p[te] += p
        cnt[te] += 1
        fold_auc.append(roc_auc_score(ytr[te], p))
    oof_p /= np.maximum(cnt, 1)
    oof[name] = oof_p
    m = metrics(ytr, oof_p)
    m["auroc_ci"] = boot_ci(ytr, oof_p, roc_auc_score)
    m["auprc_ci"] = boot_ci(ytr, oof_p, average_precision_score)
    m["fold_auc_mean"] = float(np.mean(fold_auc))
    m["fold_auc_sd"] = float(np.std(fold_auc, ddof=1))
    results[name] = m
    print("  %-30s OOF AUROC=%.4f Sn=%.3f Sp=%.3f MCC=%.3f"
          % (name, m["auroc"], m["sensitivity"], m["specificity"], m["mcc"]), flush=True)

sidx = [genes.index(g) for g in panel]
ext_res = {}
ext_pred = {}
for name, (est, grid) in models().items():
    gs = GridSearchCV(est, grid, scoring="roc_auc",
                      cv=StratifiedKFold(10, shuffle=True, random_state=SEED), n_jobs=-1)
    gs.fit(Xtr[:, sidx], ytr)
    ext_res[name] = {"best_params": {k: str(v) for k, v in gs.best_params_.items()}}
    for g in EXT:
        Xe, ye = ext[g]
        p = gs.best_estimator_.predict_proba(Xe[:, sidx])[:, 1]
        ext_pred[f"{name}|{g}"] = p.tolist()
        m = metrics(ye, p)
        m["auroc_ci"] = boot_ci(ye, p, roc_auc_score)
        ext_res[name][g] = m
    yp = np.concatenate([ext[g][1] for g in EXT])
    pp_ = np.concatenate([np.array(ext_pred[f"{name}|{g}"]) for g in EXT])
    m = metrics(yp, pp_)
    m["auroc_ci"] = boot_ci(yp, pp_, roc_auc_score)
    ext_res[name]["POOLED"] = m
    print("  %-30s pooled-external AUROC=%.4f Sn=%.3f Sp=%.3f"
          % (name, m["auroc"], m["sensitivity"], m["specificity"]), flush=True)

json.dump(dict(internal=results, external=ext_res, panel=panel),
          open(f"{W}/detect_results.json", "w"), indent=2,
          default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
np.savez_compressed(f"{W}/detect_oof.npz", y=ytr,
                    **{k.replace(" ", "_"): v for k, v in oof.items()})
np.savez_compressed(f"{W}/detect_extpred.npz",
                    **{k.replace(" ", "_").replace("|", "__"): np.array(v) for k, v in ext_pred.items()},
                    **{f"ytrue__{g}": ext[g][1] for g in EXT})
print("done", flush=True)
