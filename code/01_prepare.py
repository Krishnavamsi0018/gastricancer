"""Step 1: parse GEO series matrices, map probes to gene symbols, harmonise cohorts."""
import gzip, os, re, json
import numpy as np

DATA = "data"; OUT = "work"
os.makedirs(OUT, exist_ok=True)

# ---------- platform annotation ----------
def load_platform(path, id_col, sym_col):
    m = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = None
        for line in fh:
            if header is None:
                if line.startswith("ID\t"):
                    header = line.rstrip("\n").split("\t")
                    i_id = header.index(id_col); i_sy = header.index(sym_col)
                continue
            if line.startswith("!"):
                break
            p = line.rstrip("\n").split("\t")
            if len(p) <= max(i_id, i_sy):
                continue
            sym = p[i_sy].strip()
            if sym and sym not in ("", "---"):
                m[p[i_id].strip()] = sym.split("///")[0].strip()
    return m

# ---------- series matrix ----------
def parse_series(path):
    meta = {}
    probes, rows, samples = [], [], None
    intable = False
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("!series_matrix_table_end"):
                break
            if intable:
                p = line.rstrip("\n").split("\t")
                if samples is None:
                    samples = [x.strip('"') for x in p[1:]]
                    continue
                probes.append(p[0].strip('"'))
                rows.append([np.nan if (v == "" or v == "null") else float(v) for v in p[1:]])
                continue
            if line.startswith("!series_matrix_table_begin"):
                intable = True; continue
            if line.startswith("!Sample_"):
                k = line.split("\t")[0][1:]
                vals = [x.strip('"') for x in line.rstrip("\n").split("\t")[1:]]
                meta.setdefault(k, []).append(vals)
    X = np.asarray(rows, dtype=np.float32)          # probes x samples
    return meta, probes, samples, X

def collapse_to_genes(X, probes, pmap):
    """Collapse probes to genes keeping the probe with the highest mean expression."""
    mean = np.nanmean(X, axis=1)
    best = {}
    for i, pr in enumerate(probes):
        g = pmap.get(pr)
        if g is None:
            continue
        if g not in best or mean[i] > mean[best[g]]:
            best[g] = i
    genes = sorted(best)
    idx = np.array([best[g] for g in genes])
    return X[idx, :], genes

def maybe_log2(X, name):
    mx = np.nanmax(X)
    if mx > 50:
        print(f"   [{name}] max={mx:.1f} -> applying log2(x+1)")
        X = np.log2(np.clip(X, 0, None) + 1.0)
    else:
        print(f"   [{name}] max={mx:.1f} -> already log-scale")
    return X

def label_from_meta(meta, n, gse):
    """Return 1 = gastric adenocarcinoma, 0 = non-malignant gastric mucosa.

    Explicit per-cohort rules are used rather than keyword heuristics, because in
    several series the *source* field names the disease for every sample
    (e.g. GSE54129: "surgically resected from patient of gastric cancer") while the
    tissue characteristic carries the actual tumour/normal designation.
    """
    chars = meta.get("Sample_characteristics_ch1", [])
    titles = meta.get("Sample_title", [[""] * n])[0]

    def char_rows(prefix):
        for row in chars:
            if all(v.lower().startswith(prefix) for v in row if v.strip()):
                return [v.lower() for v in row]
        return None

    y = np.full(n, -1)
    tis = char_rows("tissue:")
    if gse == "GSE66229":                      # tissue: Gastric tumor / normal gastric tissue
        y = np.array([1 if "tumor" in v else 0 for v in tis])
    elif gse == "GSE54129":                    # tissue: normal / gastric cancer tissue
        y = np.array([0 if "normal" in v else 1 for v in tis])
    elif gse == "GSE13911":                    # a bare TUMOR / NORMAL characteristic row
        row = None
        for r in chars:
            up = {v.strip().upper() for v in r}
            if up <= {"TUMOR", "NORMAL"} and up:
                row = r; break
        y = np.array([1 if v.strip().upper() == "TUMOR" else 0 for v in row])
    elif gse == "GSE79973":                    # tissue: gastric adenocarcinoma / gastric mucosa
        y = np.array([1 if "adenocarcinoma" in v else 0 for v in tis])
    else:                                      # prognostic cohorts: all tumour
        y = np.ones(n, dtype=int)
    return y.astype(int), [str(t) for t in titles]


if __name__ == "__main__":
    print("Loading platform annotations ...")
    gpl570 = load_platform(f"{DATA}/GPL570_platform.txt", "ID", "Gene Symbol")
    gpl6947 = load_platform(f"{DATA}/GPL6947_platform.txt", "ID", "Symbol")
    print(f"   GPL570 probes mapped: {len(gpl570)},  GPL6947: {len(gpl6947)}")

    cohorts = {
        "GSE66229": ("GPL570", "discovery"),
        "GSE54129": ("GPL570", "external"),
        "GSE13911": ("GPL570", "external"),
        "GSE79973": ("GPL570", "external"),
        "GSE15459": ("GPL570", "prognostic"),
        "GSE84437": ("GPL6947", "prognostic"),
    }
    summary = {}
    for gse, (plat, role) in cohorts.items():
        print(f"Parsing {gse} ({plat}, {role}) ...")
        meta, probes, samples, X = parse_series(f"{DATA}/{gse}.txt.gz")
        print(f"   raw matrix: {X.shape[0]} probes x {X.shape[1]} samples")
        X = maybe_log2(X, gse)
        pmap = gpl570 if plat == "GPL570" else gpl6947
        Xg, genes = collapse_to_genes(X, probes, pmap)
        y, blob = label_from_meta(meta, len(samples), gse)
        print(f"   genes: {len(genes)};  labels tumour={int((y==1).sum())} normal={int((y==0).sum())} unk={int((y==-1).sum())}")
        np.savez_compressed(f"{OUT}/{gse}.npz",
                            X=Xg.astype(np.float32), genes=np.array(genes),
                            samples=np.array(samples), y=y, platform=plat, role=role)
        json.dump({k: v for k, v in meta.items() if k.startswith("Sample_char")
                   or k in ("Sample_geo_accession", "Sample_title")},
                  open(f"{OUT}/{gse}_meta.json", "w"))
        summary[gse] = dict(platform=plat, role=role, n=len(samples), genes=len(genes),
                            tumour=int((y == 1).sum()), normal=int((y == 0).sum()),
                            unknown=int((y == -1).sum()), example=str(blob[0])[:160])
    json.dump(summary, open(f"{OUT}/cohort_summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
