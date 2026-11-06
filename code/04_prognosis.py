"""Step 4 - Task 2 (risk assessment): 3-year mortality prediction and survival stratification.

Training / internal CV : GSE84437 (n=483, Illumina HT-12 v3, Korean cohort)
External validation    : GSE15459 (n=192, Affymetrix U133 Plus 2.0, Singapore cohort)
"""
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import survdiff

W = "work"
SEED = 202
HORIZON = 36.0          # months


def zc(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd[sd < 1e-9] = 1e-9
    return (X - mu) / sd


# ---------------- training cohort: GSE84437 ----------------
d84 = np.load(f"{W}/GSE84437.npz", allow_pickle=True)
g84 = list(d84["genes"])
X84 = d84["X"].astype(np.float64)
c84 = json.load(open(f"{W}/GSE84437_clin.json"))
FLD = ["duration overall survival", "death", "age", "sex", "ptstage", "pnstage"]
ok84 = [i for i in range(len(c84["death"]))
        if all(str(c84[f][i]).strip() != "" for f in FLD)]   # 433/483 complete cases
c84 = {k: [v[i] for i in ok84] for k, v in c84.items()}
X84 = X84[:, ok84]
os84 = np.array([float(v) for v in c84["duration overall survival"]])
ev84 = np.array([int(v) for v in c84["death"]])
age84 = np.array([float(v) for v in c84["age"]])
sex84 = np.array([1 if v.lower().startswith("m") else 0 for v in c84["sex"]])
T84 = np.array([int(v.strip().upper().replace("T", "")) for v in c84["ptstage"]])
N84 = np.array([int(v.strip().upper().replace("N", "")) for v in c84["pnstage"]])

# ---------------- external cohort: GSE15459 ----------------
d15 = np.load(f"{W}/GSE15459.npz", allow_pickle=True)
g15 = list(d15["genes"])
X15 = d15["X"].astype(np.float64)
c15 = json.load(open(f"{W}/GSE15459_clin.json"))
gsm15 = c15["gsm"]
out15 = c15["outcome"]
keep15 = [i for i, g in enumerate(gsm15) if g in out15]
rows = [out15[gsm15[i]] for i in keep15]
os15 = np.array([float(r["Overall.Survival (Months)**"]) for r in rows])
ev15 = np.array([int(float(r["Outcome (1=dead)"])) for r in rows])
age15 = np.array([float(r["Age_at_surgery"]) for r in rows])
sex15 = np.array([1 if str(r["Gender"]).lower().startswith("m") else 0 for r in rows])
stg15 = np.array([float(r["Stage"]) for r in rows])
lau15 = np.array([str(r["Laurenclassification"]) for r in rows])
X15 = X15[:, keep15]
print("GSE84437 n=%d (events %d) | GSE15459 n=%d (events %d)"
      % (len(os84), ev84.sum(), len(os15), ev15.sum()), flush=True)

common = sorted(set(g84) & set(g15))
i84 = {g: i for i, g in enumerate(g84)}
i15 = {g: i for i, g in enumerate(g15)}
Z84 = zc(X84[[i84[g] for g in common], :]).T
Z15 = zc(X15[[i15[g] for g in common], :]).T
print("common genes across platforms: %d" % len(common), flush=True)

# ---------------- 3-year mortality label ----------------
lab = np.full(len(os84), -1)
lab[(ev84 == 1) & (os84 <= HORIZON)] = 1
lab[os84 >= HORIZON] = 0
use = lab >= 0
y = lab[use].astype(int)
Xu = Z84[use]
print("labelled for %d-month mortality: n=%d (deaths %d, survivors %d); %d censored early and excluded"
      % (HORIZON, use.sum(), (y == 1).sum(), (y == 0).sum(), (~use).sum()), flush=True)


def cox_screen(X, t, e):
    """Score-test p-value for each gene from a univariable Cox model (fast approximation)."""
    order = np.argsort(t)
    Xs, es = X[order], e[order]
    n, p = Xs.shape
    csum = np.cumsum(Xs[::-1], axis=0)[::-1]
    csum2 = np.cumsum((Xs ** 2)[::-1], axis=0)[::-1]
    nrisk = np.arange(n, 0, -1).reshape(-1, 1)
    mean = csum / nrisk
    var = csum2 / nrisk - mean ** 2
    idx = es == 1
    U = (Xs[idx] - mean[idx]).sum(axis=0)
    V = np.clip(var[idx].sum(axis=0), 1e-12, None)
    z = U / np.sqrt(V)
    return z, 2 * stats.norm.sf(np.abs(z))


def select(Xtr, ytr, ttr, etr, k, seed):
    z, p = cox_screen(Xtr, ttr, etr)
    o = np.argsort(p)
    q = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    qv = np.empty_like(q)
    qv[o] = np.clip(q, 0, 1)
    cand = np.where(qv < 0.25)[0]
    if len(cand) < max(k, 50):
        cand = o[:1000]
    if len(cand) > 1500:
        cand = cand[np.argsort(p[cand])[:1500]]
    r = np.random.default_rng(seed)
    freq = np.zeros(len(cand))
    n = len(ytr)
    for _ in range(25):
        i = r.choice(n, int(0.8 * n), replace=False)
        if len(np.unique(ytr[i])) < 2:
            continue
        lr = LogisticRegression(penalty="l1", C=0.08, solver="liblinear", max_iter=4000)
        lr.fit(Xtr[np.ix_(i, cand)], ytr[i])
        freq += (np.abs(lr.coef_[0]) > 1e-8)
    return cand[np.lexsort((-np.abs(z[cand]), -freq))[:k]], cand, freq, z, qv


def models():
    return {
        "Logistic Regression (L2)": (
            Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))]),
            {"m__C": [0.01, 0.1, 1, 10]}),
        "Elastic-Net Logistic": (
            Pipeline([("s", StandardScaler()),
                      ("m", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=8000))]),
            {"m__C": [0.05, 0.5, 5], "m__l1_ratio": [0.2, 0.5, 0.8]}),
        "SVM (RBF kernel)": (
            Pipeline([("s", StandardScaler()), ("m", SVC(probability=True, random_state=SEED))]),
            {"m__C": [0.1, 1, 10], "m__gamma": ["scale", 0.05]}),
        "Random Forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1),
            {"n_estimators": [500], "max_depth": [4, 8, None], "min_samples_leaf": [3, 8]}),
        "Extremely Randomised Trees": (
            ExtraTreesClassifier(random_state=SEED, n_jobs=-1),
            {"n_estimators": [500], "max_depth": [4, 8, None], "min_samples_leaf": [3, 8]}),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=SEED),
            {"n_estimators": [150, 300], "learning_rate": [0.03, 0.08], "max_depth": [2, 3]}),
        "XGBoost": (
            XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1, verbosity=0),
            {"n_estimators": [200, 400], "learning_rate": [0.03, 0.08], "max_depth": [2, 3],
             "subsample": [0.8], "colsample_bytree": [0.8], "reg_lambda": [1.0, 5.0]}),
        "LightGBM": (
            LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
            {"n_estimators": [200, 400], "learning_rate": [0.03, 0.08], "num_leaves": [7, 15],
             "min_child_samples": [15]}),
        "Multilayer Perceptron": (
            Pipeline([("s", StandardScaler()), ("m", MLPClassifier(max_iter=4000, random_state=SEED))]),
            {"m__hidden_layer_sizes": [(32,), (64, 32)], "m__alpha": [1e-3, 1e-1]}),
    }


def boot_ci(y_, p_, fn, n=2000, seed=3):
    r = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = r.integers(0, len(y_), len(y_))
        if len(np.unique(y_[i])) < 2:
            continue
        try:
            out.append(fn(y_[i], p_[i]))
        except Exception:
            pass
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)


t_u, e_u = os84[use], ev84[use]
KS = [5, 10, 15, 20, 25, 30, 40, 60]
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=SEED)
splits = list(cv.split(Xu, y))

print("panel-size sweep (nested CV, Elastic-Net Logistic) ...", flush=True)
sweep = {k: [] for k in KS}
fsel = []
for i, (tr, te) in enumerate(splits):
    s_full, _, _, _, _ = select(Xu[tr], y[tr], t_u[tr], e_u[tr], max(KS), seed=i)
    fsel.append(s_full)
    for k in KS:
        s = s_full[:k]
        pipe = Pipeline([("s", StandardScaler()),
                         ("m", LogisticRegression(penalty="elasticnet", solver="saga",
                                                  l1_ratio=0.5, C=0.5, max_iter=8000))])
        pipe.fit(Xu[np.ix_(tr, s)], y[tr])
        sweep[k].append(roc_auc_score(y[te], pipe.predict_proba(Xu[np.ix_(te, s)])[:, 1]))
    print("  fold %d/%d" % (i + 1, len(splits)), flush=True)

mean = {k: float(np.mean(v)) for k, v in sweep.items()}
sem = {k: float(np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in sweep.items()}
bk = max(mean, key=mean.get)
kstar = min([k for k in KS if mean[k] >= mean[bk] - sem[bk]])
print("sweep:", {k: round(mean[k], 4) for k in KS}, "-> k* =", kstar, flush=True)

results = {}
oof = {}
for name, (est, grid) in models().items():
    p_oof = np.zeros(len(y))
    cnt = np.zeros(len(y))
    for i, (tr, te) in enumerate(splits):
        s = fsel[i][:kstar]
        gs = GridSearchCV(est, grid, scoring="roc_auc",
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
        gs.fit(Xu[np.ix_(tr, s)], y[tr])
        p_oof[te] += gs.best_estimator_.predict_proba(Xu[np.ix_(te, s)])[:, 1]
        cnt[te] += 1
    p_oof /= np.maximum(cnt, 1)
    oof[name] = p_oof
    yh = (p_oof >= 0.5).astype(int)
    tn, fp, fn_, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    results[name] = dict(auroc=roc_auc_score(y, p_oof), auroc_ci=boot_ci(y, p_oof, roc_auc_score),
                         auprc=average_precision_score(y, p_oof),
                         brier=brier_score_loss(y, p_oof),
                         sensitivity=tp / max(tp + fn_, 1), specificity=tn / max(tn + fp, 1),
                         accuracy=(tp + tn) / len(y))
    print("  %-28s OOF AUROC=%.3f [%.3f-%.3f]" % (name, results[name]["auroc"],
                                                  *results[name]["auroc_ci"]), flush=True)

# ---------------- final prognostic panel and risk score ----------------
sel, cand, freq, zsc, qv = select(Xu, y, t_u, e_u, kstar, seed=777)
panel = [common[i] for i in sel]
print("PROGNOSTIC PANEL:", panel, flush=True)

best = max(results, key=lambda k: results[k]["auroc"])
est, grid = models()[best]
gs = GridSearchCV(est, grid, scoring="roc_auc",
                  cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)
gs.fit(Xu[:, sel], y)
final = gs.best_estimator_
risk84 = final.predict_proba(Z84[:, sel])[:, 1]
risk15 = final.predict_proba(Z15[:, sel])[:, 1]

# reference model: clinical variables only (age, sex, pT, pN)
clin = np.column_stack([age84[use], sex84[use], T84[use], N84[use]])
p_clin = np.zeros(len(y))
cnt = np.zeros(len(y))
for tr, te in splits:
    pl = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))])
    pl.fit(clin[tr], y[tr])
    p_clin[te] += pl.predict_proba(clin[te])[:, 1]
    cnt[te] += 1
p_clin /= np.maximum(cnt, 1)
comb = np.column_stack([clin, oof[best]])
p_comb = np.zeros(len(y))
cnt = np.zeros(len(y))
for tr, te in splits:
    pl = Pipeline([("s", StandardScaler()), ("m", LogisticRegression(max_iter=5000))])
    pl.fit(comb[tr], y[tr])
    p_comb[te] += pl.predict_proba(comb[te])[:, 1]
    cnt[te] += 1
p_comb /= np.maximum(cnt, 1)
results["Clinical model (age, sex, pT, pN)"] = dict(
    auroc=roc_auc_score(y, p_clin), auroc_ci=boot_ci(y, p_clin, roc_auc_score),
    auprc=average_precision_score(y, p_clin), brier=brier_score_loss(y, p_clin))
results["Combined clinical + genomic"] = dict(
    auroc=roc_auc_score(y, p_comb), auroc_ci=boot_ci(y, p_comb, roc_auc_score),
    auprc=average_precision_score(y, p_comb), brier=brier_score_loss(y, p_comb))
print("  clinical-only AUROC=%.3f | combined AUROC=%.3f"
      % (results["Clinical model (age, sex, pT, pN)"]["auroc"],
         results["Combined clinical + genomic"]["auroc"]), flush=True)


def km(t, e, grid=None):
    tt = np.sort(np.unique(t[e == 1]))
    s, out = 1.0, []
    for x in tt:
        nr = (t >= x).sum()
        dd = ((t == x) & (e == 1)).sum()
        s *= (1 - dd / nr) if nr else 1
        out.append((float(x), float(s), int(nr)))
    return out


def cox(cols, names, t, e):
    Xc = np.column_stack(cols).astype(float)
    m = PHReg(t, Xc, status=e).fit()
    return {n: dict(hr=float(np.exp(m.params[i])),
                    lo=float(np.exp(m.params[i] - 1.96 * m.bse[i])),
                    hi=float(np.exp(m.params[i] + 1.96 * m.bse[i])),
                    p=float(m.pvalues[i])) for i, n in enumerate(names)}


surv = {}
for tag, r, t_, e_, extra in [("GSE84437", risk84, os84, ev84,
                               dict(age=age84, sex=sex84, T=T84, N=N84)),
                              ("GSE15459", risk15, os15, ev15,
                               dict(age=age15, sex=sex15, stage=stg15))]:
    q = np.quantile(r, [1 / 3, 2 / 3])
    grp = np.digitize(r, q)
    logrank = survdiff(t_, e_, grp)
    rz = (r - r.mean()) / r.std()
    uni = cox([rz], ["risk"], t_, e_)
    cols = [rz] + [extra[k] for k in extra]
    multi = cox(cols, ["risk"] + list(extra), t_, e_)
    hi_lo = cox([(grp == 2).astype(float)], ["high_vs_rest"], t_, e_)
    surv[tag] = dict(logrank_chi2=float(logrank[0]), logrank_p=float(logrank[1]),
                     univariable=uni, multivariable=multi, high_vs_rest=hi_lo,
                     km={str(g): km(t_[grp == g], e_[grp == g]) for g in [0, 1, 2]},
                     n=int(len(t_)), events=int(e_.sum()),
                     median_fu=float(np.median(t_)))
    print("%s: log-rank chi2=%.1f p=%.3g | HR per SD=%.2f (%.2f-%.2f) p=%.3g | adj HR=%.2f p=%.3g"
          % (tag, logrank[0], logrank[1], uni["risk"]["hr"], uni["risk"]["lo"], uni["risk"]["hi"],
             uni["risk"]["p"], multi["risk"]["hr"], multi["risk"]["p"]), flush=True)

# external discrimination at 1/3/5 years
ext_auc = {}
for h in [12, 36, 60]:
    lb = np.full(len(os15), -1)
    lb[(ev15 == 1) & (os15 <= h)] = 1
    lb[os15 >= h] = 0
    m_ = lb >= 0
    if len(np.unique(lb[m_])) == 2:
        ext_auc[h] = dict(auroc=float(roc_auc_score(lb[m_], risk15[m_])),
                          ci=boot_ci(lb[m_].astype(int), risk15[m_], roc_auc_score),
                          n=int(m_.sum()), events=int((lb[m_] == 1).sum()))
        print("  GSE15459 %d-month AUROC=%.3f [%.3f-%.3f] (n=%d)"
              % (h, ext_auc[h]["auroc"], *ext_auc[h]["ci"], ext_auc[h]["n"]), flush=True)

json.dump(dict(results=results, panel=panel, kstar=kstar, best_model=best,
               sweep_mean=mean, sweep_sem=sem, ks=KS,
               best_params={k: str(v) for k, v in gs.best_params_.items()},
               survival=surv, external_auc=ext_auc,
               n_train=int(use.sum()), n_events=int((y == 1).sum()),
               excluded_censored=int((~use).sum())),
          open(f"{W}/prog_results.json", "w"), indent=2,
          default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
np.savez_compressed(f"{W}/prog_arrays.npz", y=y, risk84=risk84, risk15=risk15,
                    os84=os84, ev84=ev84, os15=os15, ev15=ev15,
                    age84=age84, sex84=sex84, T84=T84, N84=N84,
                    age15=age15, sex15=sex15, stg15=stg15, lau15=lau15,
                    sel=sel, panel=np.array(panel), Xu_sel=Xu[:, sel], use=use,
                    p_clin=p_clin, p_comb=p_comb, freq=freq, cand=cand, zsc=zsc, qv=qv,
                    genes=np.array(common),
                    **{("oof_" + k.replace(" ", "_")): v for k, v in oof.items()})
print("done", flush=True)
