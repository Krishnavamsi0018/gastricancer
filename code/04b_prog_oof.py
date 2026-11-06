"""Step 4b - survival analysis on *out-of-fold* risk scores.

The first pass scored GSE84437 with a model fitted on the same patients, which
inflates the internal hazard ratio. Here the internal risk score for every
labelled patient is the cross-validated out-of-fold prediction; the small group
excluded from label construction (censored before the 3-year horizon) never
entered training at all and is scored with the final model.
"""
import json, warnings, numpy as np
warnings.filterwarnings("ignore")
from statsmodels.duration.hazard_regression import PHReg
from statsmodels.duration.survfunc import survdiff

W = "work"
a = np.load(f"{W}/prog_arrays.npz", allow_pickle=True)
res = json.load(open(f"{W}/prog_results.json"))
best = res["best_model"]

use = a["use"].astype(bool)
os84, ev84 = a["os84"], a["ev84"]
age84, sex84, T84, N84 = a["age84"], a["sex84"], a["T84"], a["N84"]
oof = a["oof_" + best.replace(" ", "_")]
risk_full = a["risk84"].copy()
risk_oof = risk_full.copy()
risk_oof[use] = oof                      # honest, cross-validated scores
print("internal risk score: %d/%d patients scored out-of-fold" % (use.sum(), len(use)), flush=True)

risk15, os15, ev15 = a["risk15"], a["os15"], a["ev15"]
age15, sex15, stg15 = a["age15"], a["sex15"], a["stg15"]


def km(t, e):
    out, s = [], 1.0
    for x in np.sort(np.unique(t[e == 1])):
        nr = (t >= x).sum()
        dd = ((t == x) & (e == 1)).sum()
        s *= (1 - dd / nr) if nr else 1
        out.append((float(x), float(s), int(nr)))
    return out


def atrisk(t, grid):
    return [int((t >= g).sum()) for g in grid]


def cox(cols, names, t, e):
    m = PHReg(t, np.column_stack(cols).astype(float), status=e).fit()
    return {n: dict(hr=float(np.exp(m.params[i])),
                    lo=float(np.exp(m.params[i] - 1.96 * m.bse[i])),
                    hi=float(np.exp(m.params[i] + 1.96 * m.bse[i])),
                    p=float(m.pvalues[i])) for i, n in enumerate(names)}


def cindex(t, e, score):
    """Harrell's c-index for a risk score (higher score = higher risk)."""
    num = den = 0.0
    for i in range(len(t)):
        if e[i] != 1:
            continue
        j = t > t[i]
        den += j.sum()
        num += (score[j] < score[i]).sum() + 0.5 * (score[j] == score[i]).sum()
    return float(num / den) if den else float("nan")


GRID = [0, 12, 24, 36, 48, 60, 72, 84, 96]
surv = {}
for tag, r, t_, e_, extra in [
        ("GSE84437", risk_oof, os84, ev84, dict(age=age84, sex=sex84, pT=T84, pN=N84)),
        ("GSE15459", risk15, os15, ev15, dict(age=age15, sex=sex15, stage=stg15))]:
    q = np.quantile(r, [1 / 3, 2 / 3])
    grp = np.digitize(r, q)
    lr = survdiff(t_, e_, grp)
    rz = (r - r.mean()) / r.std()
    uni = cox([rz], ["risk"], t_, e_)
    multi = cox([rz] + [extra[k] for k in extra], ["risk"] + list(extra), t_, e_)
    hilo = cox([(grp == 2).astype(float)], ["high_vs_rest"], t_, e_)
    ci_boot = []
    rng = np.random.default_rng(4)
    for _ in range(500):
        i = rng.integers(0, len(t_), len(t_))
        try:
            ci_boot.append(cindex(t_[i], e_[i], r[i]))
        except Exception:
            pass
    surv[tag] = dict(
        logrank_chi2=float(lr[0]), logrank_p=float(lr[1]),
        univariable=uni, multivariable=multi, high_vs_rest=hilo,
        cindex=cindex(t_, e_, r),
        cindex_ci=[float(np.percentile(ci_boot, 2.5)), float(np.percentile(ci_boot, 97.5))],
        km={str(g): km(t_[grp == g], e_[grp == g]) for g in [0, 1, 2]},
        atrisk={str(g): atrisk(t_[grp == g], GRID) for g in [0, 1, 2]},
        grid=GRID, group_n={str(g): int((grp == g).sum()) for g in [0, 1, 2]},
        n=int(len(t_)), events=int(e_.sum()), median_fu=float(np.median(t_)))
    print("%s  n=%d ev=%d | c-index=%.3f [%.3f-%.3f] | log-rank p=%.3g | HR/SD=%.2f (%.2f-%.2f) p=%.3g"
          " | adj HR=%.2f (%.2f-%.2f) p=%.3g"
          % (tag, len(t_), e_.sum(), surv[tag]["cindex"], *surv[tag]["cindex_ci"],
             lr[1], uni["risk"]["hr"], uni["risk"]["lo"], uni["risk"]["hi"], uni["risk"]["p"],
             multi["risk"]["hr"], multi["risk"]["lo"], multi["risk"]["hi"], multi["risk"]["p"]),
          flush=True)

# clinical-only reference c-index for context
for tag, cols, t_, e_ in [("GSE84437", [T84, N84], os84, ev84),
                          ("GSE15459", [stg15], os15, ev15)]:
    m = PHReg(t_, np.column_stack(cols).astype(float), status=e_).fit()
    lp = np.column_stack(cols).astype(float) @ m.params
    surv[tag]["clinical_cindex"] = cindex(t_, e_, lp)
    print("  %s clinical-stage-only c-index = %.3f" % (tag, surv[tag]["clinical_cindex"]), flush=True)

res["survival_oof"] = surv
json.dump(res, open(f"{W}/prog_results.json", "w"), indent=2,
          default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
np.savez_compressed(f"{W}/prog_risk_oof.npz", risk_oof=risk_oof, risk15=risk15)
print("done", flush=True)
