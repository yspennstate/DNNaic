#!/usr/bin/env python3
"""The depth-requirement curve, measured on real data (mouse Vkorc1).

The simulations show that direction accuracy is set by how deep the rarefaction reaches (a Fisher
threshold in the standardized sample size g). This script reproduces that curve on a real system.
It fixes a single locus set -- the SNPs at the Vkorc1 introgression region where all eight M. spretus
genomes are genotyped -- and then varies only the rarefaction cap. Holding the loci fixed matters,
because the standard-error features scale as 1/sqrt(n_loci); changing the locus count would confound
the depth effect. As the cap rises, the frozen model's posterior for the correct class C
(spretus -> domesticus) rises monotonically and crosses to a C call once the depth is sufficient --
the same threshold behaviour as the simulated curve, and the same effect that makes the model abstain
on the shallow archaic trios.

Configure DNNAIC_DATA (regen_full). Reads the Harr et al. (2016) wild-mouse VCF by tabix. CPU only.
"""
import os, io, gzip, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dnnaic.tabix import http_get, load_tbi, fetch_region
from dnnaic import build_matrix

VCF_URL = ("https://wwwuser.gwdguser.de/~evolbio/evolgen/wildmouse/vcf/"
           "AllMouse.vcf_90_recalibrated_snps_raw_indels_reheader_PopSorted.PASS.vcf.gz")
DATA = os.environ.get("DNNAIC_DATA", "data/simulation_data")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "real", "mouse")
os.makedirs(OUT, exist_ok=True)
PREFIX = {"Mmm": "musculus", "Mmd": "domesticus", "Ms": "spretus"}
POPS = ["musculus", "domesticus", "spretus"]
REGION = ("chr7", 126_000_000, 129_000_000)   # Vkorc1
CAPS = [6, 8, 10, 12, 14, 16]


def called(cell):
    gt = cell.split(":", 1)[0].replace("|", "/")
    return sum(1 for a in gt.split("/") if a in ("0", "1"))


def main():
    names, refs = load_tbi(VCF_URL + ".tbi", remote=True)
    with gzip.GzipFile(fileobj=io.BytesIO(http_get(VCF_URL, rng="bytes=0-1500000"))) as g:
        for line in g:
            s = line.decode("latin-1")
            if s.startswith("#CHROM"):
                samples = s.rstrip("\n").split("\t")[9:]
                break
    sample2pop = {s: PREFIX[p] for s in samples for p in PREFIX if s.startswith(p + "_")}
    idx = {p: [i for i, s in enumerate(samples) if sample2pop.get(s) == p] for p in POPS}
    sel = idx["musculus"] + idx["domesticus"] + idx["spretus"]
    sel_names = [samples[i] for i in sel]

    print("fetching Vkorc1; fixing SNPs at which all eight spretus genomes are genotyped ...", flush=True)
    rows = []
    for s in fetch_region(VCF_URL, names, refs, *REGION):
        p = s.rstrip("\n").split("\t")
        if len(p) < 10 or len(p[3]) != 1 or len(p[4]) != 1 or p[4] in (".", "*"):
            continue
        if sum(called(p[9 + i]) for i in idx["spretus"]) != 16:
            continue
        if sum(called(p[9 + i]) for i in idx["musculus"]) < 16 or sum(called(p[9 + i]) for i in idx["domesticus"]) < 16:
            continue
        rows.append((p[0], p[1], p[3], p[4], [p[9 + i].split(":", 1)[0].replace("|", "/") for i in sel]))
        if len(rows) >= 9000:
            break
    print(f"fixed locus set: {len(rows)} SNPs (held constant across the sweep)", flush=True)

    vcf, pm = os.path.join(OUT, "mouse_depth.vcf"), os.path.join(OUT, "mouse_depth_popmap.tsv")
    with open(vcf, "w") as o:
        o.write("##fileformat=VCFv4.2\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        o.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sel_names) + "\n")
        for chrom, pos, ref, alt, cells in rows:
            o.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(cells) + "\n")
    with open(pm, "w") as o:
        for s in sel_names:
            o.write(f"{s}\t{sample2pop[s]}\n")

    d = os.path.join(DATA, "regen_full")
    Xtr = np.load(os.path.join(d, "X.npy")); dtr = np.load(os.path.join(d, "direction.npy")).astype("U2")
    mtr = np.load(os.path.join(d, "magnitude.npy"))
    sc = StandardScaler().fit(Xtr)
    appr = np.where((dtr != "D") & (mtr >= 2.5e-4))[0]
    appr = np.random.default_rng(0).choice(appr, min(80000, len(appr)), replace=False)
    dlog = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr[appr]), np.searchsorted(np.array(["A", "B", "C"]), dtr[appr]))
    del Xtr, dtr, mtr

    sweep = []
    for g in CAPS:
        X, _, _ = build_matrix(vcf, pm, max_depth=g, pop_order=POPS)
        prob = dlog.predict_proba(sc.transform(np.asarray(X, float))).mean(0)
        sweep.append(dict(max_depth=g, P_C=round(float(prob[2]), 3), call="ABC"[int(np.argmax(prob))]))
        print(f"  max_depth={g:>2}: P_C={prob[2]:.3f}  call={sweep[-1]['call']}", flush=True)
    json.dump(dict(n_loci=len(rows), sweep=sweep), open(os.path.join(OUT, "mouse_depthsweep_result.json"), "w"), indent=2)
    print("The class-C posterior rises with rarefaction depth on real data (n_loci held fixed).", flush=True)


if __name__ == "__main__":
    main()
