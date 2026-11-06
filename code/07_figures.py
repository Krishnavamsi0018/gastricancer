"""Step 7 - publication figures."""
import json, os, warnings, numpy as np
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.titleweight": "bold", "axes.titlelocation": "left",
})
BLUE, ORANGE, GREEN, RED = "#2C6E9B", "#E1812C", "#3A923A", "#C03D3E"
PURPLE, TEAL, GREY, GOLD = "#7B6FB0", "#17A2A2", "#8C8C8C", "#B8A32C"
PAL = [BLUE, ORANGE, GREEN, RED, PURPLE, TEAL, GREY, GOLD, "#D26FA0", "#5B8C5A", "#A0522D"]
W, F = "work", "figs"
os.makedirs(F, exist_ok=True)
EXT = ["GSE54129", "GSE13911", "GSE79973"]

panel = json.load(open(f"{W}/panel.json"))
PANEL = panel["panel"]
det = json.load(open(f"{W}/detect_results.json"))
rob = json.load(open(f"{W}/robust_results.json"))
xai = json.load(open(f"{W}/xai_results.json"))
prog = json.load(open(f"{W}/prog_results.json"))
diag = json.load(open(f"{W}/diagnostics.json"))
raw = np.load(f"{W}/detect_raw.npz", allow_pickle=True)
dzz = np.load(f"{W}/detect_matrices.npz", allow_pickle=True)
de = np.load(f"{W}/de_stats.npz", allow_pickle=True)
rp = np.load(f"{W}/robust_preds.npz", allow_pickle=True)
XA = np.load(f"{W}/xai.npz", allow_pickle=True)
PA = np.load(f"{W}/prog_arrays.npz", allow_pickle=True)
MODELS = list(rob["results"]["N1_cohort_zscore"].keys())


def lab(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(f"{F}/{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name, flush=True)


# ============================ Figure 1: study design ============================
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 78); ax.axis("off")

    def box(x, y, w, h, txt, fc, ec, fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.45,rounding_size=1.6",
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.45)

    def arrow(x1, y1, x2, y2, c="#555555"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, lw=0.9, color=c,
                                     connectionstyle="arc3,rad=0"))

    ax.text(50, 75.5, "Public gastric-tissue transcriptome cohorts (NCBI GEO)",
            ha="center", fontsize=8.6, fontweight="bold")
    box(1, 63, 30, 9.5, "DISCOVERY\nGSE66229 (ACRG)\n300 tumour / 100 normal\nAffymetrix U133 Plus 2.0",
        "#DCE9F2", BLUE)
    box(34, 63, 32, 9.5, "EXTERNAL DETECTION\nGSE54129 (111/21) · GSE13911 (38/31)\nGSE79973 (10/10)\nAffymetrix U133 Plus 2.0",
        "#FAE6D2", ORANGE)
    box(69, 63, 30, 9.5, "PROGNOSTIC\nGSE84437 (n=433, HT-12 v3)\nGSE15459 (n=192, U133 Plus 2.0)\noverall survival",
        "#DEEEDE", GREEN)

    box(4, 51, 44, 7.5, "Probe to gene-symbol collapse (max-mean probe)\n22,880 genes shared across U133 Plus 2.0 cohorts",
        "#F4F4F4", GREY, fs=7.0)
    box(53, 51, 44, 7.5, "Two normalisation strategies\nN1 cohort-wise gene z-score  ·  N2 within-sample rank",
        "#F4F4F4", GREY, fs=7.0)
    arrow(16, 63, 20, 58.6); arrow(50, 63, 40, 58.6); arrow(84, 63, 80, 58.6)

    box(4, 38.5, 44, 9, "TASK 1 — EARLY DETECTION\nWelch t + BH-FDR filter, L1 stability selection\n"
        "one-standard-error panel-size rule (nested 10x3 CV)\nparsimonious 8-gene panel",
        "#DCE9F2", BLUE, fs=7.0)
    box(53, 38.5, 44, 9, "TASK 2 — RISK ASSESSMENT\nunivariable Cox screen, L1 stability selection\n"
        "3-year mortality label (5x4 CV)\n25-gene prognostic panel",
        "#DEEEDE", GREEN, fs=7.0)
    arrow(26, 51, 26, 47.9); arrow(75, 51, 75, 47.9)

    box(4, 26.5, 44, 8.5, "11 classifiers, nested grid search\nlogistic · elastic-net · NB · k-NN · SVM · RF\n"
        "extra-trees · GB · XGBoost · LightGBM · MLP", "#FFFFFF", "#444444", fs=6.8)
    box(53, 26.5, 44, 8.5, "9 classifiers, nested grid search\nplus clinical baseline (age, sex, pT, pN)\n"
        "and combined clinical + genomic model", "#FFFFFF", "#444444", fs=6.8)
    arrow(26, 38.5, 26, 35.4); arrow(75, 38.5, 75, 35.4)

    box(1, 13.5, 31, 9, "External validation\n3 independent cohorts\nAUROC · Sn · Sp · balanced acc.\ncalibration · decision curves",
        "#FAE6D2", ORANGE, fs=6.8)
    box(34.5, 13.5, 31, 9, "Case-mix stress test\nsynthetic batches, tumour\nfraction 10-90%\nthreshold drift analysis",
        "#F6E0E0", RED, fs=6.8)
    box(68, 13.5, 31, 9, "Survival validation\nKaplan-Meier by tertile\nlog-rank · Cox HR\nc-index, external cohort",
        "#DEEEDE", GREEN, fs=6.8)
    arrow(20, 26.5, 16, 22.6); arrow(30, 26.5, 44, 22.6); arrow(75, 26.5, 83, 22.6)

    box(20, 2.5, 60, 8, "EXPLAINABILITY LAYER\nTreeSHAP global and local attributions · cross-cohort attribution stability\n"
        "multi-method importance concordance · leave-one-gene-out ablation",
        "#EDE9F5", PURPLE, fs=7.0)
    arrow(16, 13.5, 35, 10.7); arrow(50, 13.5, 50, 10.7); arrow(83, 13.5, 65, 10.7)
    save(fig, "fig01_design")


# ============================ Figure 2: cohort landscape ============================
def fig2():
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.35))
    Xs = {"GSE66229": raw["Xtr"].astype(float)}
    for g in EXT:
        Xs[g] = raw[f"X_{g}"].astype(float)
    ys = {"GSE66229": dzz["ytr"].astype(int)}
    for g in EXT:
        ys[g] = dzz[f"y_{g}"].astype(int)
    names = ["GSE66229", "GSE54129", "GSE13911", "GSE79973"]

    ax = axes[0]; lab(ax, "a")
    n_t = [int((ys[n] == 1).sum()) for n in names]
    n_n = [int((ys[n] == 0).sum()) for n in names]
    yy = np.arange(4)
    ax.barh(yy, n_t, color=RED, height=0.6)
    ax.barh(yy, n_n, left=n_t, color=BLUE, height=0.6)
    ax.text(150, 0, "tumour", ha="center", va="center", color="white",
            fontsize=6.4, fontweight="bold")
    ax.text(350, 0, "normal", ha="center", va="center", color="white",
            fontsize=6.4, fontweight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels([n + "\n(%.0f%% tumour)" % (100 * n_t[i] / (n_t[i] + n_n[i]))
                        for i, n in enumerate(names)], fontsize=6.0)
    ax.invert_yaxis()
    ax.set_xlabel("Samples"); ax.set_xlim(0, 430)
    ax.set_title("Cohort composition", fontsize=7.8)

    def pcplot(ax, mats, title, letter):
        lab(ax, letter)
        allX = np.vstack([mats[n] for n in names])
        top = np.argsort(allX.var(0))[-2000:]
        P = PCA(2, random_state=0).fit(allX[:, top])
        for i, n in enumerate(names):
            Z = P.transform(mats[n][:, top])
            ax.scatter(Z[:, 0], Z[:, 1], s=4.5, c=PAL[i], alpha=0.6, lw=0, label=n)
        ax.set_xlabel(f"PC1 ({P.explained_variance_ratio_[0]*100:.0f}%)")
        ax.set_ylabel(f"PC2 ({P.explained_variance_ratio_[1]*100:.0f}%)")
        ax.set_title(title, fontsize=7.8)

    pcplot(axes[1], Xs, "Raw log-intensity", "b")
    Z1 = {"GSE66229": dzz["Xtr"].astype(float)}
    for g in EXT:
        Z1[g] = dzz[f"X_{g}"].astype(float)
    pcplot(axes[2], Z1, "N1 cohort z-score", "c")
    RN = {n: np.apply_along_axis(stats.rankdata, 1, Xs[n]) / Xs[n].shape[1] for n in names}
    pcplot(axes[3], RN, "N2 within-sample rank", "d")
    h = [Line2D([], [], marker="o", ls="", ms=4, color=PAL[i], label=n)
         for i, n in enumerate(names)]
    fig.legend(handles=h, frameon=False, fontsize=6.6, ncol=4,
               loc="lower center", bbox_to_anchor=(0.58, -0.10))
    fig.tight_layout(w_pad=1.4)
    save(fig, "fig02_cohorts")


# ============================ Figure 3: feature selection ============================
def fig3():
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.35))
    genes = list(de["genes"]); q = de["q"]; d_ = de["delta"]
    ax = axes[0]; lab(ax, "a")
    lq = -np.log10(np.clip(q, 1e-300, None))
    sig = (q < 0.01) & (np.abs(d_) > 0.5)
    ax.scatter(d_[~sig], lq[~sig], s=1.5, c="#D5D5D5", lw=0, rasterized=True)
    ax.scatter(d_[sig & (d_ > 0)], lq[sig & (d_ > 0)], s=1.8, c=RED, lw=0, rasterized=True)
    ax.scatter(d_[sig & (d_ < 0)], lq[sig & (d_ < 0)], s=1.8, c=BLUE, lw=0, rasterized=True)
    offs = {"LYVE1": (-34, 30), "MYOC": (-36, 8), "PLCXD3": (-40, -14),
            "ADIPOQ": (-40, -40), "FIGF": (-26, -62), "CLDN1": (10, 12),
            "INHBA": (10, -4), "DIO2": (10, -20)}
    for g_ in PANEL:
        i = genes.index(g_)
        ax.scatter([d_[i]], [lq[i]], s=11, facecolor="none", edgecolor="k", lw=0.6, zorder=5)
        ax.annotate(g_, (d_[i], lq[i]), fontsize=5.6, fontweight="bold",
                    xytext=offs.get(g_, (6, 6)), textcoords="offset points",
                    arrowprops=dict(arrowstyle="-", lw=0.4, color="#666666"))
    ax.axhline(2, ls=":", lw=0.6, c="k")
    ax.axvline(0.5, ls=":", lw=0.6, c="k"); ax.axvline(-0.5, ls=":", lw=0.6, c="k")
    ax.set_xlim(-3.4, 3.0)
    ax.set_xlabel("Standardised mean difference"); ax.set_ylabel(r"$-\log_{10}$ FDR")
    ax.set_title(f"Differential expression\n({int(sig.sum()):,} genes at FDR<0.01)", fontsize=7.5)

    ax = axes[1]; lab(ax, "b")
    ks = panel["ks"]; m = [panel["mean"][str(k)] for k in ks]; s = [panel["sem"][str(k)] for k in ks]
    ax.errorbar(ks, m, yerr=s, marker="o", ms=3.2, lw=1.1, c=BLUE, capsize=2, zorder=3)
    bk = panel["best_k"]
    ax.axhline(panel["mean"][str(bk)] - panel["sem"][str(bk)], ls="--", lw=0.8, c=RED)
    ax.axvline(panel["k_star"], ls=":", lw=1.0, c=GREEN)
    ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 40, 100])
    ax.set_xticklabels([5, 10, 20, 40, 100])
    ax.set_xlabel("Panel size k"); ax.set_ylabel("Nested-CV AUROC")
    ax.set_title("Panel-size selection", fontsize=7.8)
    ax.legend(handles=[Line2D([], [], color=RED, ls="--", lw=1, label="one-SE bound"),
                       Line2D([], [], color=GREEN, ls=":", lw=1.2, label=f"k* = {panel['k_star']}")],
              frameon=False, loc="lower center", fontsize=6, bbox_to_anchor=(0.5, -0.02))

    ax = axes[2]; lab(ax, "c")
    st = np.load(f"{W}/stability.npz", allow_pickle=True)
    fr, cd, gn = st["freq"], st["cand"], list(st["genes"])
    o = np.argsort(-fr)[:18]
    nm = [gn[cd[i]] for i in o][::-1]
    vv = fr[o][::-1]
    cols = [GREEN if n in PANEL else "#C2C2C2" for n in nm]
    ax.barh(np.arange(len(nm)), vv, color=cols, height=0.68)
    ax.set_yticks(np.arange(len(nm))); ax.set_yticklabels(nm, fontsize=5.2)
    ax.set_xlabel("Selection frequency"); ax.set_xlim(0, 1.15)
    ax.set_title("L1 stability selection", fontsize=7.8)
    ax.legend(handles=[Line2D([], [], color=GREEN, lw=4, label="final panel"),
                       Line2D([], [], color="#C2C2C2", lw=4, label="not selected")],
              frameon=False, loc="lower right", fontsize=5.6, handlelength=1.1)

    ax = axes[3]; lab(ax, "d")
    X = dzz["Xtr"].astype(float); y = dzz["ytr"].astype(int)
    gi = [list(dzz["genes"]).index(g_) for g_ in PANEL]
    for i, j in enumerate(gi):
        for cl, c, off in [(0, BLUE, -0.18), (1, RED, 0.18)]:
            bp = ax.boxplot([X[y == cl, j]], positions=[i + off], widths=0.3, patch_artist=True,
                            showfliers=False, medianprops=dict(color="k", lw=0.7),
                            whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                            boxprops=dict(lw=0.5))
            bp["boxes"][0].set_facecolor(c); bp["boxes"][0].set_alpha(0.8)
    ax.set_xticks(np.arange(len(PANEL)))
    ax.set_xticklabels(PANEL, rotation=55, ha="right", fontsize=5.6)
    ax.set_ylim(-3.2, 4.6)
    ax.set_ylabel("Expression (z)"); ax.set_title("Panel genes, discovery", fontsize=7.8)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=4, label="Normal"),
                       Line2D([], [], color=RED, lw=4, label="Tumour")],
              frameon=False, fontsize=5.8, loc="upper right", ncol=2, handlelength=1.1)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig03_features")


# ============================ Figure 4: detection performance ============================
def key(tag, model, split):
    return f"{tag}__{model}__{split}".replace(" ", "_")


def fig4():
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.35))
    ytr = rp["ytr"]
    ax = axes[0]; lab(ax, "a")
    for i, m in enumerate(MODELS):
        fpr, tpr, _ = roc_curve(ytr, rp[key("N2_sample_rank", m, "oof")])
        ax.plot(fpr, tpr, lw=0.9, c=PAL[i % len(PAL)], alpha=0.85)
    ax.plot([0, 1], [0, 1], ls=":", lw=0.7, c="k")
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Internal 10x3 CV\n(discovery, n=400)", fontsize=7.5)
    ax.text(0.62, 0.22, "11 models\nAUROC 0.997-1.000", fontsize=6.2, ha="center")

    ax = axes[1]; lab(ax, "b")
    yy = np.concatenate([rp[f"y_{g}"] for g in EXT])
    hs = []
    for i, (tag, ls, cc, nm) in enumerate([("N1_cohort_zscore", "--", ORANGE, "N1 z-score"),
                                           ("N2_sample_rank", "-", BLUE, "N2 rank")]):
        pp = np.concatenate([rp[key(tag, "Gradient Boosting", g)] for g in EXT])
        fpr, tpr, _ = roc_curve(yy, pp)
        a = rob["results"][tag]["Gradient Boosting"]["POOLED"]["auroc"]
        ax.plot(fpr, tpr, ls=ls, lw=1.5, c=cc)
        hs.append(Line2D([], [], color=cc, ls=ls, lw=1.4, label=f"{nm} ({a:.3f})"))
    ax.plot([0, 1], [0, 1], ls=":", lw=0.7, c="k")
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Pooled external\n(n=221)", fontsize=7.5)
    ax.legend(handles=hs, frameon=False, loc="lower right", fontsize=6, handlelength=1.3)

    ax = axes[2]; lab(ax, "c")
    for i, g in enumerate(EXT):
        for tag, ls in [("N1_cohort_zscore", "--"), ("N2_sample_rank", "-")]:
            fpr, tpr, _ = roc_curve(rp[f"y_{g}"], rp[key(tag, "Gradient Boosting", g)])
            ax.plot(fpr, tpr, ls=ls, lw=1.1, c=PAL[i], alpha=0.9)
    ax.plot([0, 1], [0, 1], ls=":", lw=0.7, c="k")
    h = [Line2D([], [], color=PAL[i], lw=1.4, label=g.replace("GSE", "GSE ")) for i, g in enumerate(EXT)]
    h += [Line2D([], [], color="k", lw=1.0, ls="--", label="N1"),
          Line2D([], [], color="k", lw=1.0, ls="-", label="N2")]
    ax.legend(handles=h, frameon=False, loc="lower right", fontsize=5.5, handlelength=1.2)
    ax.set_xlabel("1 - specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Per external cohort", fontsize=7.5)

    ax = axes[3]; lab(ax, "d")
    short = {"Logistic Regression (L2)": "Logistic (L2)", "Elastic-Net Logistic": "Elastic-net",
             "Gaussian Naive Bayes": "Naive Bayes", "k-Nearest Neighbours": "k-NN",
             "SVM (RBF kernel)": "SVM-RBF", "Random Forest": "Random forest",
             "Extremely Randomised Trees": "Extra trees", "Gradient Boosting": "Grad. boosting",
             "XGBoost": "XGBoost", "LightGBM": "LightGBM", "Multilayer Perceptron": "MLP"}
    yy = np.arange(len(MODELS))
    a1 = [rob["results"]["N1_cohort_zscore"][m]["POOLED"]["auroc"] for m in MODELS]
    a2 = [rob["results"]["N2_sample_rank"][m]["POOLED"]["auroc"] for m in MODELS]
    ic = [rob["results"]["N2_sample_rank"][m]["internal_cv"]["auroc"] for m in MODELS]
    ax.barh(yy - 0.22, a1, height=0.4, color=ORANGE)
    ax.barh(yy + 0.22, a2, height=0.4, color=BLUE)
    ax.scatter(ic, yy, s=11, marker="D", c="k", zorder=5)
    ax.set_yticks(yy); ax.set_yticklabels([short[m] for m in MODELS], fontsize=5.8)
    ax.invert_yaxis(); ax.set_xlim(0.5, 1.03); ax.set_xlabel("AUROC")
    ax.set_title("Model comparison", fontsize=7.8)
    ax.legend(handles=[Line2D([], [], color=ORANGE, lw=4, label="External N1"),
                       Line2D([], [], color=BLUE, lw=4, label="External N2"),
                       Line2D([], [], marker="D", ls="", ms=3.4, color="k", label="Internal CV")],
              frameon=False, fontsize=5.6, loc="lower center", ncol=3,
              bbox_to_anchor=(0.45, -0.58), handlelength=1.0, columnspacing=0.8)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig04_detection")


# ============================ Figure 5: generalisation gap ============================
def fig5():
    fig = plt.figure(figsize=(7.4, 4.75))
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.46)
    short = {"Logistic Regression (L2)": "Logistic (L2)", "Elastic-Net Logistic": "Elastic-net",
             "Gaussian Naive Bayes": "Naive Bayes", "k-Nearest Neighbours": "k-NN",
             "SVM (RBF kernel)": "SVM-RBF", "Random Forest": "Random forest",
             "Extremely Randomised Trees": "Extra trees", "Gradient Boosting": "Grad. boosting",
             "XGBoost": "XGBoost", "LightGBM": "LightGBM", "Multilayer Perceptron": "MLP"}

    ax = fig.add_subplot(gs[0, 0]); lab(ax, "a")
    yy = np.arange(len(MODELS))
    s1 = [rob["results"]["N1_cohort_zscore"][m]["POOLED"]["specificity"] for m in MODELS]
    s2 = [rob["results"]["N2_sample_rank"][m]["POOLED"]["specificity"] for m in MODELS]
    ax.barh(yy - 0.2, s1, height=0.38, color=ORANGE)
    ax.barh(yy + 0.2, s2, height=0.38, color=BLUE)
    ax.set_yticks(yy); ax.set_yticklabels([short[m] for m in MODELS], fontsize=5.2)
    ax.invert_yaxis(); ax.set_xlabel("External specificity"); ax.set_xlim(0, 1.0)
    ax.set_title("Specificity recovered", fontsize=7.5)
    ax.legend(handles=[Line2D([], [], color=ORANGE, lw=4, label="N1 z-score"),
                       Line2D([], [], color=BLUE, lw=4, label="N2 rank")],
              frameon=False, fontsize=5.2, loc="lower center", ncol=2,
              bbox_to_anchor=(0.5, -0.46), handlelength=1.0, columnspacing=0.9)

    ax = fig.add_subplot(gs[0, 1]); lab(ax, "b")
    fr = rob["fracs"]
    for tag, c, nm in [("N1_cohort_zscore", ORANGE, "N1 z-score"), ("N2_sample_rank", BLUE, "N2 rank")]:
        sp = np.array([rob["case_mix"][tag][str(f)]["sp"][0] for f in fr])
        sd = np.array([rob["case_mix"][tag][str(f)]["sp"][1] for f in fr])
        ax.plot(fr, sp, "-o", ms=2.8, lw=1.2, c=c, label=nm)
        ax.fill_between(fr, sp - sd, sp + sd, color=c, alpha=0.16, lw=0)
    ax.axvline(0.75, ls=":", lw=0.9, c="k")
    ax.annotate("training\ncase mix", (0.75, 1.10), fontsize=5.4, ha="center", va="top")
    ax.set_xlabel("Tumour fraction of batch"); ax.set_ylabel("Specificity")
    ax.set_ylim(0, 1.14); ax.set_title("Synthetic batches", fontsize=7.5)
    ax.legend(frameon=False, loc="lower right", fontsize=5.6, handlelength=1.2)

    ax = fig.add_subplot(gs[0, 2]); lab(ax, "c")
    cm = diag["case_mix_within"]
    for i, g in enumerate(EXT):
        d = cm[g]
        if not d:
            continue
        xs = [d[k]["actual"] for k in sorted(d, key=float)]
        ys = [d[k]["sp1"] for k in sorted(d, key=float)]
        ax.plot(xs, ys, "-o", ms=3, lw=1.2, c=PAL[i], label=g.replace("GSE", ""))
        ax.plot(xs, [d[k]["sp2"] for k in sorted(d, key=float)], ls="--", lw=0.9,
                c=PAL[i], alpha=0.55)
    ax.set_xlabel("Tumour fraction of batch"); ax.set_ylabel("Specificity")
    ax.set_ylim(-0.03, 1.05); ax.set_title("Within each cohort", fontsize=7.5)
    ax.legend(frameon=False, fontsize=5.4, loc="upper left", handlelength=1.2, title=None)
    ax.text(0.97, 0.04, "solid N1 · dashed N2", transform=ax.transAxes, ha="right",
            fontsize=5.0, style="italic")

    ax = fig.add_subplot(gs[1, 0]); lab(ax, "d")
    dc = diag["decomposition"]
    xs = np.arange(len(EXT))
    rank = [max(dc[g]["ranking_loss"], 0) for g in EXT]
    thr = [dc[g]["threshold_loss"] for g in EXT]
    ax.bar(xs - 0.19, thr, width=0.36, color=RED, label="Threshold drift")
    ax.bar(xs + 0.19, rank, width=0.36, color=PURPLE, label="Ranking loss")
    ax.set_xticks(xs); ax.set_xticklabels([g.replace("GSE", "") for g in EXT], fontsize=6)
    ax.set_ylabel("Recoverable performance")
    ax.set_title("Failure-mode split", fontsize=7.5)
    ax.legend(frameon=False, fontsize=5.4, handlelength=1.0)

    ax = fig.add_subplot(gs[1, 1]); lab(ax, "e")
    ticks = []
    for i, (tag, c) in enumerate([("N1_cohort_zscore", ORANGE), ("N2_sample_rank", BLUE)]):
        for j, g in enumerate(EXT):
            p = rp[key(tag, "Gradient Boosting", g)]; yv = rp[f"y_{g}"]
            x = i * 3.5 + j
            ax.scatter(np.random.default_rng(j).normal(x - 0.17, 0.05, (yv == 0).sum()),
                       p[yv == 0], s=3.2, c=BLUE, alpha=0.7, lw=0)
            ax.scatter(np.random.default_rng(j + 7).normal(x + 0.17, 0.05, (yv == 1).sum()),
                       p[yv == 1], s=3.2, c=RED, alpha=0.5, lw=0)
            ticks.append((x, g.replace("GSE", "")))
    ax.axhline(0.5, ls="--", lw=0.8, c="k")
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks], fontsize=5.0, rotation=55, ha="right")
    ax.text(1, 1.19, "N1", ha="center", fontsize=7, fontweight="bold", color=ORANGE)
    ax.text(4.5, 1.19, "N2", ha="center", fontsize=7, fontweight="bold", color=BLUE)
    ax.set_ylabel("Predicted probability"); ax.set_ylim(-0.06, 1.30)
    ax.set_xlim(-0.7, 6.2); ax.set_title("Score distributions", fontsize=7.5)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=3, label="Normal"),
                       Line2D([], [], marker="o", ls="", color=RED, ms=3, label="Tumour")],
              frameon=False, fontsize=5.2, loc="center", bbox_to_anchor=(0.5, 0.5),
              handletextpad=0.2)

    ax = fig.add_subplot(gs[1, 2]); lab(ax, "f")
    M = np.array([[rob["results"][t][m]["POOLED"]["balanced"] for t in
                   ["N1_cohort_zscore", "N2_sample_rank"]] for m in MODELS])
    im = ax.imshow(M, cmap="RdYlBu", vmin=0.55, vmax=0.95, aspect="auto")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["N1", "N2"], fontsize=6.5)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([short[m] for m in MODELS], fontsize=5.2)
    for i in range(len(MODELS)):
        for j in range(2):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5.0)
    ax.set_title("Balanced accuracy", fontsize=7.5)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.05)
    save(fig, "fig05_gap")


# ============================ Figure 6: calibration / DCA / CM ============================
def fig6():
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.15))
    ax = axes[0]; lab(ax, "a")
    yy = np.concatenate([rp[f"y_{g}"] for g in EXT])
    for tag, c, nm in [("N1_cohort_zscore", ORANGE, "N1"), ("N2_sample_rank", BLUE, "N2")]:
        pp = np.concatenate([rp[key(tag, "Gradient Boosting", g)] for g in EXT])
        bins = np.linspace(0, 1, 9)
        idx = np.digitize(pp, bins[1:-1])
        xs, ys_ = [], []
        for b in range(8):
            m = idx == b
            if m.sum() >= 5:
                xs.append(pp[m].mean()); ys_.append(yy[m].mean())
        ax.plot(xs, ys_, "-o", ms=3, lw=1.1, c=c, label=nm)
    ax.plot([0, 1], [0, 1], ls=":", lw=0.8, c="k")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("External calibration"); ax.legend(frameon=False, loc="upper left")

    ax = axes[1]; lab(ax, "b")
    prev = yy.mean()
    th = np.linspace(0.01, 0.95, 120)
    for tag, c, nm in [("N1_cohort_zscore", ORANGE, "N1"), ("N2_sample_rank", BLUE, "N2")]:
        pp = np.concatenate([rp[key(tag, "Gradient Boosting", g)] for g in EXT])
        nb = []
        for t in th:
            tp = ((pp >= t) & (yy == 1)).sum(); fp = ((pp >= t) & (yy == 0)).sum()
            nb.append(tp / len(yy) - fp / len(yy) * (t / (1 - t)))
        ax.plot(th, nb, lw=1.2, c=c, label=nm)
    ax.plot(th, [prev - (1 - prev) * (t / (1 - t)) for t in th], ls="--", lw=0.8, c=GREY, label="Treat all")
    ax.axhline(0, ls=":", lw=0.8, c="k")
    ax.set_ylim(-0.25, prev + 0.05)
    ax.set_xlabel("Threshold probability"); ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve"); ax.legend(frameon=False, fontsize=6)

    for k, (tag, nm) in enumerate([("N1_cohort_zscore", "N1 cohort z-score"),
                                   ("N2_sample_rank", "N2 within-sample rank")]):
        ax = axes[2 + k]; lab(ax, "cd"[k])
        pp = np.concatenate([rp[key(tag, "Gradient Boosting", g)] for g in EXT])
        cm = confusion_matrix(yy, (pp >= 0.5).astype(int))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > cm.max() * 0.55 else "k", fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. normal", "Pred. tumour"], fontsize=6)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Normal", "Tumour"], fontsize=6)
        sp = cm[0, 0] / cm[0].sum(); sn = cm[1, 1] / cm[1].sum()
        ax.set_title(f"{nm}\nSn={sn:.2f}  Sp={sp:.2f}", fontsize=7)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig06_clinical")


# ============================ Figure 7: SHAP global ============================
def fig7():
    fig = plt.figure(figsize=(7.4, 4.7))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.5)
    sv = XA["sv_train"]; Xv = XA["X_train"]
    order = np.argsort(np.abs(sv).mean(0))
    ax = fig.add_subplot(gs[0, :2]); lab(ax, "a")
    rng = np.random.default_rng(0)
    for i, j in enumerate(order):
        v = Xv[:, j]
        vn = (v - v.min()) / max(np.ptp(v), 1e-9)
        ax.scatter(sv[:, j], i + rng.uniform(-0.24, 0.24, len(v)), c=vn, cmap="coolwarm",
                   s=3.2, lw=0, alpha=0.75, vmin=0, vmax=1)
    ax.axvline(0, lw=0.7, c="k")
    ax.set_yticks(range(len(order))); ax.set_yticklabels([PANEL[j] for j in order], fontsize=7)
    ax.set_xlabel("SHAP value (log-odds contribution)")
    ax.set_title("TreeSHAP attributions, discovery cohort", fontsize=7.8)
    cax = ax.inset_axes([0.845, 0.06, 0.028, 0.34])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=5.5, length=1.5)
    cb.ax.set_title("Expr.", fontsize=5.5, pad=2)

    ax = fig.add_subplot(gs[0, 2]); lab(ax, "b")
    keys = ["train"] + EXT
    wdt = 0.2
    for i, k in enumerate(keys):
        g = np.abs(XA[f"sv_{k}"]).mean(0)
        ax.barh(np.arange(len(PANEL)) + (i - 1.5) * wdt, g[order], height=wdt,
                color=PAL[i], label="Discovery" if k == "train" else k.replace("GSE", ""))
    ax.set_yticks(range(len(order))); ax.set_yticklabels([PANEL[j] for j in order], fontsize=6.2)
    ax.set_xlabel(r"mean $|\phi|$"); ax.set_title("Per cohort", fontsize=7.8)
    ax.legend(frameon=False, fontsize=5.4, loc="lower right", handlelength=1.0)

    ax = fig.add_subplot(gs[1, 0]); lab(ax, "c")
    A = np.array(xai["cohort_agreement"])
    ax.imshow(A, cmap="YlGnBu", vmin=0.3, vmax=1)
    n = ["Disc.", "54129", "13911", "79973"]
    ax.set_xticks(range(4)); ax.set_xticklabels(n, fontsize=5.8, rotation=40, ha="right")
    ax.set_yticks(range(4)); ax.set_yticklabels(n, fontsize=5.8)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{A[i,j]:.2f}", ha="center", va="center", fontsize=5.6,
                    color="white" if A[i, j] > 0.8 else "k")
    ax.set_title("Attribution stability\n(Spearman)", fontsize=7.5)

    ax = fig.add_subplot(gs[1, 1]); lab(ax, "d")
    M = np.array(xai["method_agreement"])
    ax.imshow(M, cmap="YlGnBu", vmin=-0.5, vmax=1)
    sh = ["SHAP", "Permut.", "Gain", "|LR coef|", "Uni-AUROC"]
    ax.set_xticks(range(len(sh))); ax.set_xticklabels(sh, fontsize=5.4, rotation=40, ha="right")
    ax.set_yticks(range(len(sh))); ax.set_yticklabels(sh, fontsize=5.4)
    for i in range(len(sh)):
        for j in range(len(sh)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=5.2,
                    color="white" if M[i, j] > 0.75 else "k")
    ax.set_title("Method concordance", fontsize=7.5)

    ax = fig.add_subplot(gs[1, 2]); lab(ax, "e")
    ab = xai["ablation"]
    gsx = sorted(ab, key=lambda g: ab[g]["delta"])
    vals = [ab[g]["delta"] for g in gsx]
    ax.barh(range(len(gsx)), vals, color=[RED if v < -0.01 else GREY for v in vals], height=0.66)
    ax.axvline(0, lw=0.7, c="k")
    ax.set_yticks(range(len(gsx))); ax.set_yticklabels(gsx, fontsize=6.2)
    ax.set_xlabel(r"$\Delta$ pooled external AUROC")
    ax.set_title("Leave-one-gene-out", fontsize=7.5)
    save(fig, "fig07_shap_global")


# ============================ Figure 8: SHAP local ============================
def fig8():
    cs = xai["cases"]
    titles = {"confident_tumour": "Confident true positive",
              "confident_normal": "Confident true negative",
              "false_negative": "Hardest tumour (lowest score)",
              "false_positive": "False positive"}
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4))
    for a, (k, ax) in enumerate(zip(["confident_tumour", "confident_normal",
                                     "false_negative", "false_positive"], axes)):
        lab(ax, "abcd"[a])
        c = cs[k]
        sh = c["shap"]
        o = sorted(sh, key=lambda g: abs(sh[g]))
        v = [sh[g] for g in o]
        ax.barh(range(len(o)), v, color=[RED if x > 0 else BLUE for x in v], height=0.66)
        ax.axvline(0, lw=0.7, c="k")
        ax.set_yticks(range(len(o)))
        ax.set_yticklabels([f"{g} = {c['value'][g]:+.1f}" for g in o], fontsize=5.6)
        ax.set_xlabel(r"$\phi$ (log-odds)")
        ax.set_title(f"{titles[k]}\np={c['p']:.3f}, true={'tumour' if c['y']==1 else 'normal'}",
                     fontsize=7)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig08_shap_local")


# ============================ Figure 9: panel heatmap ============================
def fig9():
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5),
                             gridspec_kw={"width_ratios": [4, 1.4, 1.2, 0.9]})
    names = ["GSE66229", "GSE54129", "GSE13911", "GSE79973"]
    mats = {"GSE66229": XA["X_train"]}
    for g in EXT:
        mats[g] = XA[f"X_{g}"]
    ys = {"GSE66229": XA["ytr"]}
    for g in EXT:
        ys[g] = XA[f"y_{g}"]
    for i, (n, ax) in enumerate(zip(names, axes)):
        X = mats[n]; y = ys[n]
        o = np.argsort(y * 1000 + XA[f"p_{n}"] if n != "GSE66229" else y * 1000 + XA["p_train"])
        M = X[o].T
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
                       interpolation="nearest")
        ax.set_xticks([])
        if i == 0:
            ax.set_yticks(range(len(PANEL))); ax.set_yticklabels(PANEL, fontsize=6)
        else:
            ax.set_yticks([])
        b = int((y[o] == 0).sum())
        ax.axvline(b - 0.5, c="k", lw=1.0)
        ax.set_title(f"{n}\nn={len(y)}", fontsize=7)
        ax.set_xlabel("normal | tumour", fontsize=6)
    fig.colorbar(im, ax=axes, fraction=0.012, pad=0.012, label="rank-normalised z")
    save(fig, "fig09_heatmap")


# ============================ Figure 10: prognosis ============================
def fig10():
    fig = plt.figure(figsize=(7.4, 4.7))
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.46)
    sv = prog["survival_oof"]

    ax = fig.add_subplot(gs[0, 0]); lab(ax, "a")
    ks = prog["ks"]; m = [prog["sweep_mean"][str(k)] for k in ks]
    s = [prog["sweep_sem"][str(k)] for k in ks]
    ax.errorbar(ks, m, yerr=s, marker="o", ms=3, lw=1.1, c=GREEN, capsize=2)
    ax.axvline(prog["kstar"], ls=":", lw=1.0, c=RED)
    ax.annotate(f"k* = {prog['kstar']}", (prog["kstar"], 0.535), fontsize=6, color=RED,
                xytext=(3, 0), textcoords="offset points")
    ax.set_xlabel("Panel size k"); ax.set_ylabel("Nested-CV AUROC")
    ax.set_title("Prognostic panel size", fontsize=7.6)

    ax = fig.add_subplot(gs[0, 1:]); lab(ax, "b")
    mods = [k for k in prog["results"] if not k.startswith("Clinical") and not k.startswith("Combined")]
    mods += ["Clinical model (age, sex, pT, pN)", "Combined clinical + genomic"]
    vals = [prog["results"][k]["auroc"] for k in mods]
    cis = [prog["results"][k].get("auroc_ci", [np.nan, np.nan]) for k in mods]
    o = np.argsort(vals)
    short = {"Logistic Regression (L2)": "Logistic (L2)", "Elastic-Net Logistic": "Elastic-net",
             "SVM (RBF kernel)": "SVM-RBF", "Random Forest": "Random forest",
             "Extremely Randomised Trees": "Extra trees", "Gradient Boosting": "Grad. boosting",
             "XGBoost": "XGBoost", "LightGBM": "LightGBM", "Multilayer Perceptron": "MLP",
             "Clinical model (age, sex, pT, pN)": "Clinical only",
             "Combined clinical + genomic": "Clinical + genomic"}
    for r, i in enumerate(o):
        c = GREEN if ("Clinical" in mods[i] or "Combined" in mods[i]) else BLUE
        lo, hi = cis[i]
        if not np.isnan(lo):
            ax.plot([lo, hi], [r, r], c=c, lw=1.1, solid_capstyle="butt")
        ax.plot([vals[i]], [r], "o", ms=4, c=c)
    ax.set_yticks(range(len(o))); ax.set_yticklabels([short[mods[i]] for i in o], fontsize=6)
    ax.axvline(0.5, ls=":", lw=0.8, c="k")
    ax.set_xlim(0.45, 0.82); ax.set_ylim(-0.7, len(o) - 0.3)
    ax.set_xlabel("AUROC for 3-year mortality (GSE84437, cross-validated)")
    ax.set_title("Risk-model comparison", fontsize=7.6)

    for i, (tag, ttl) in enumerate([("GSE84437", "GSE84437 (internal, out-of-fold)"),
                                    ("GSE15459", "GSE15459 (external)")]):
        ax = fig.add_subplot(gs[1, i]); lab(ax, "cd"[i])
        S = sv[tag]
        for g, c, nm in [("0", BLUE, "Low"), ("1", GOLD, "Intermediate"), ("2", RED, "High")]:
            cur = np.array(S["km"][g])
            t = np.concatenate([[0], cur[:, 0]]); s_ = np.concatenate([[1], cur[:, 1]])
            ax.step(t, s_, where="post", c=c, lw=1.3, label=f"{nm} (n={S['group_n'][g]})")
        ax.set_xlim(0, 100); ax.set_ylim(0, 1.02)
        ax.set_xlabel("Months since surgery"); ax.set_ylabel("Overall survival")
        ax.text(0.97, 0.97, f"log-rank p = {S['logrank_p']:.3g}\n"
                            f"HR per SD = {S['univariable']['risk']['hr']:.2f} "
                            f"({S['univariable']['risk']['lo']:.2f}-{S['univariable']['risk']['hi']:.2f})\n"
                            f"c-index = {S['cindex']:.3f}",
                ha="right", va="top", transform=ax.transAxes, fontsize=5.6)
        ax.set_title(ttl, fontsize=7.2)
        ax.legend(frameon=False, fontsize=5.8, loc="lower left", handlelength=1.2)

    ax = fig.add_subplot(gs[1, 2]); lab(ax, "e")
    labs_, lo_, hi_, hr_, cols = [], [], [], [], []
    pretty = {"risk": "Risk score/SD", "age": "Age", "sex": "Male sex",
              "pT": "pT", "pN": "pN", "stage": "Stage"}
    for tag in ["GSE84437", "GSE15459"]:
        mv = sv[tag]["multivariable"]
        for k in mv:
            hr_.append(mv[k]["hr"]); lo_.append(mv[k]["lo"]); hi_.append(mv[k]["hi"])
            labs_.append(f"{tag[3:]}· {pretty.get(k, k)}")
            cols.append(RED if k == "risk" else GREY)
    yy = np.arange(len(hr_))
    for i in range(len(hr_)):
        ax.plot([lo_[i], hi_[i]], [yy[i], yy[i]], c=cols[i], lw=1.0)
        ax.plot([hr_[i]], [yy[i]], "o", ms=3.4, c=cols[i])
    ax.axvline(1, ls=":", lw=0.8, c="k")
    ax.set_yticks(yy); ax.set_yticklabels(labs_, fontsize=5.6)
    ax.invert_yaxis(); ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 4]); ax.set_xticklabels(["0.5", "1", "2", "4"])
    ax.set_xlim(0.45, 5.5)
    ax.minorticks_off()
    ax.set_xlabel("Adjusted hazard ratio (95% CI)")
    ax.set_title("Multivariable Cox", fontsize=7.6)
    save(fig, "fig10_prognosis")


# ============================ Figure 11: prognostic SHAP ============================
def fig11():
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7),
                             gridspec_kw={"width_ratios": [1.55, 1, 1]})
    psv = XA["psv"]; pX = XA["pX"]; pp = [str(x) for x in XA["ppanel"]]
    o = np.argsort(np.abs(psv).mean(0))[-14:]
    ax = axes[0]; lab(ax, "a")
    rng = np.random.default_rng(1)
    for i, j in enumerate(o):
        v = pX[:, j]; vn = (v - v.min()) / max(np.ptp(v), 1e-9)
        ax.scatter(psv[:, j], i + rng.uniform(-0.24, 0.24, len(v)), c=vn, cmap="coolwarm",
                   s=3, lw=0, alpha=0.7, vmin=0, vmax=1)
    ax.axvline(0, lw=0.7, c="k")
    ax.set_yticks(range(len(o))); ax.set_yticklabels([pp[j] for j in o], fontsize=6)
    ax.set_xlabel("SHAP value (log-odds, 3-year mortality)")
    ax.set_title("Prognostic model attributions", fontsize=7.6)
    cax = ax.inset_axes([0.03, 0.05, 0.025, 0.28])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["low", "high"])
    cb.ax.tick_params(labelsize=5.2, length=1.5)
    cb.ax.set_title("Expr.", fontsize=5.2, pad=2)

    ax = axes[1]; lab(ax, "b")
    ea = prog["external_auc"]
    hs = sorted(int(h) for h in ea)
    v = np.array([ea[str(h)]["auroc"] for h in hs])
    lo = np.array([ea[str(h)]["ci"][0] for h in hs])
    hi = np.array([ea[str(h)]["ci"][1] for h in hs])
    ax.errorbar(hs, v, yerr=[v - lo, hi - v], fmt="-o", ms=4, lw=1.2, c=GREEN, capsize=3)
    ax.axhline(0.5, ls=":", lw=0.8, c="k")
    ax.set_xticks(hs); ax.set_ylim(0.4, 0.85)
    ax.set_xlabel("Horizon (months)"); ax.set_ylabel("AUROC")
    ax.set_title("External discrimination\n(GSE15459)", fontsize=7.6)

    ax = axes[2]; lab(ax, "c")
    sv = prog["survival_oof"]
    tags = ["GSE84437", "GSE15459"]
    gen = [sv[t]["cindex"] for t in tags]
    clin = [sv[t]["clinical_cindex"] for t in tags]
    x = np.arange(2)
    ax.bar(x - 0.18, gen, width=0.34, color=BLUE)
    ax.bar(x + 0.18, clin, width=0.34, color=ORANGE)
    for i, t in enumerate(tags):
        lo_, hi_ = sv[t]["cindex_ci"]
        ax.plot([i - 0.18, i - 0.18], [lo_, hi_], c="k", lw=0.8)
    ax.axhline(0.5, ls=":", lw=0.8, c="k")
    ax.set_xticks(x); ax.set_xticklabels(tags, fontsize=6.5)
    ax.set_ylim(0.4, 0.85); ax.set_ylabel("Harrell c-index")
    ax.set_title("Discrimination vs stage", fontsize=7.6)
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=4, label="Genomic score"),
                       Line2D([], [], color=ORANGE, lw=4, label="Stage only")],
              frameon=False, fontsize=5.8, loc="upper left", handlelength=1.0)
    fig.tight_layout(w_pad=1.7)
    save(fig, "fig11_prog_xai")


for f in [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11]:
    try:
        f()
    except Exception as e:
        import traceback
        print("FAILED", f.__name__, type(e).__name__, e)
        traceback.print_exc()
print("figures done")
