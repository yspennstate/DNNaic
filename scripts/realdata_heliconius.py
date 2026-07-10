#!/usr/bin/env python3
"""Exploratory positive-panel application to the Heliconius introgression complex.

H. melpomene is the documented donor of red wing-pattern alleles to the co-mimics H. timareta and
H. cydno (Pardo-Diaz et al. 2012; Wallbank et al. 2016; Martin et al. 2013). The trio maps to the
caterpillar tree ((P1,P2),P3) as P1=cydno, P2=timareta, P3=melpomene, so that adaptive-locus
direction corresponds to class C. This single positive panel is not a validation of the frozen
direction head: ``realdata_heliconius_robustness.py`` shows that the intended allopatric control is
also called C at high uncalibrated score.

Input: Simon Martin's whole-genome .geno(.gz) from the ABBA_BABA_whole_genome tutorial
(https://github.com/simonhmartin/tutorials) with a two-column sample<TAB>race popmap. Set the paths
below or via HEL_GENO / HEL_POP. Computes Patterson's D (detects, cannot orient), builds the DNNaic
28-D feature matrix with PADZE, applies the frozen sim-trained model (from DNNAIC_DATA/regen_full),
and reports the model-free pair-private asymmetry. Treat every learned score as uncalibrated and
off-distribution. CPU only."""
import os, gzip, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from dnnaic import build_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "real", "heliconius"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("DNNAIC_DATA", "data/simulation_data")
GENO = os.environ.get("HEL_GENO", os.path.join(OUT, "hel.geno.gz"))
POP = os.environ.get("HEL_POP", os.path.join(OUT, "hel.pop.txt"))
NSITES = 25000
# race-level demes chosen as single populations, matching the sim design (Amazonian co-mimics)
SEL = {"cydno": ["cyd_chi"], "timareta": ["tim_txn"], "melpomene": ["mel_ama"], "numata": ["num"]}
POPS = ["cydno", "timareta", "melpomene", "numata"]


def geno_to_vcf():
    samp2race = {}
    for line in open(POP):
        p = line.split()
        if len(p) >= 2: samp2race[p[0]] = p[1]
    sample2pop = {s: pop for pop, races in SEL.items() for s, r in samp2race.items() if r in races}
    f = gzip.open(GENO, "rt")
    header = f.readline().rstrip("\n").split("\t")
    colidx = {n: i for i, n in enumerate(header)}
    need = [s for s in sample2pop if s in colidx]
    popcols = {p: [colidx[s] for s in need if sample2pop[s] == p] for p in POPS}
    BASES = set("ACGT"); rows = []
    for line in f:
        parts = line.rstrip("\n").split("\t")
        cnt = {}; gts = {}
        for s in need:
            g = parts[colidx[s]]; a, b = g[0], g[2] if len(g) >= 3 else "N"
            gts[s] = (a, b)
            for x in (a, b):
                if x in BASES: cnt[x] = cnt.get(x, 0) + 1
        alleles = sorted(cnt, key=lambda k: -cnt[k])
        if len(alleles) != 2: continue
        ref, alt = alleles
        popcnt = {}; ok = True
        for p in POPS:
            r = a_ = 0
            for s in [s for s in need if sample2pop[s] == p]:
                for x in gts[s]:
                    if x == ref: r += 1
                    elif x == alt: a_ += 1
            popcnt[p] = (r, a_)
            if p != "numata" and (r + a_) < 16: ok = False
        if not ok: continue
        ta = sum(popcnt[p][1] for p in POPS[:3]); tr = sum(popcnt[p][0] for p in POPS[:3])
        if ta == 0 or tr == 0: continue
        rows.append((parts[0], parts[1], ref, alt, [gts[s] for s in need]))
    if len(rows) > NSITES:
        idx = np.linspace(0, len(rows) - 1, NSITES).astype(int); rows = [rows[i] for i in idx]
    vcf = os.path.join(OUT, "hel_quad.vcf")
    with open(vcf, "w") as o:
        o.write("##fileformat=VCFv4.2\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        o.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(need) + "\n")
        for chrom, pos, ref, alt, gtlist in rows:
            cells = ["/".join("0" if x == ref else ("1" if x == alt else ".") for x in ab) for ab in gtlist]
            o.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(cells) + "\n")
    for fn, keep in [("popmap3.tsv", POPS[:3]), ("popmap4.tsv", POPS)]:
        with open(os.path.join(OUT, fn), "w") as o:
            for s in need:
                if sample2pop[s] in keep: o.write(f"{s}\t{sample2pop[s]}\n")
    return vcf, os.path.join(OUT, "popmap3.tsv"), len(rows)


def patterson_D(alt, chroms, P1, P2, P3, O="numata"):
    flip = alt[O] > 0.5
    d = {p: np.where(flip, 1 - alt[p], alt[p]) for p in (P1, P2, P3, O)}
    abba = (1 - d[P1]) * d[P2] * d[P3]; baba = d[P1] * (1 - d[P2]) * d[P3]
    num, den = abba.sum() - baba.sum(), abba.sum() + baba.sum()
    D = num / den if den else np.nan
    ub = np.unique(chroms); ths = []
    for b in ub:
        m = chroms != b; n = abba[m].sum() - baba[m].sum(); dd = abba[m].sum() + baba[m].sum()
        ths.append(n / dd if dd else np.nan)
    ths = np.array(ths); g = len(ub); se = np.sqrt((g - 1) / g * np.nansum((ths - np.nanmean(ths)) ** 2))
    return float(D), float(D / se if se else np.nan), float(abba.sum()), float(baba.sum())


def main():
    from padze import read_vcf
    vcf, pm3, nsites = geno_to_vcf()
    print(f"[heliconius] {nsites} biallelic SNPs", flush=True)
    # Patterson's D from per-locus ALT frequencies
    loci = read_vcf(vcf, os.path.join(OUT, "popmap4.tsv"))
    ip = {p: i for i, p in enumerate(loci.populations)}
    alt = {p: [] for p in loci.populations}; chroms = []
    for li, cm in enumerate(loci.count_matrices):
        if cm.shape[1] != 2: continue
        N = cm.sum(1)
        if np.any(N == 0): continue
        for p in loci.populations: alt[p].append(cm[ip[p], 1] / N[ip[p]])
        chroms.append(loci.locus_ids[li].rpartition(":")[0])
    alt = {p: np.array(v) for p, v in alt.items()}; chroms = np.array(chroms)
    D, Z, abba, baba = patterson_D(alt, chroms, "cydno", "timareta", "melpomene")
    print(f"[heliconius] D(cydno,timareta,melpomene;numata) = {D:+.3f} Z={Z:+.1f} ABBA={abba:.0f} BABA={baba:.0f}", flush=True)
    # DNNaic frozen model (P1=cydno, P2=timareta, P3=melpomene)
    X, cols, _ = build_matrix(vcf, pm3, max_depth=100, pop_order=["cydno", "timareta", "melpomene"])
    X = np.asarray(X, float)
    d = os.path.join(DATA, "regen_full")
    Xtr = np.load(os.path.join(d, "X.npy")); dtr = np.load(os.path.join(d, "direction.npy")); mtr = np.load(os.path.join(d, "magnitude.npy"))
    sc = StandardScaler().fit(Xtr); appr = (dtr != "D") & (mtr >= 2.5e-4); cls = np.array(["A", "B", "C"])
    gate = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ((dtr != "D") & (mtr >= 2.5e-4)).astype(int))
    dlog = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr[appr]), np.searchsorted(cls, dtr[appr]))
    dhgb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=0).fit(Xtr[appr], np.searchsorted(cls, dtr[appr]))
    gp = gate.predict_proba(sc.transform(X))[:, 1]
    plog = dlog.predict_proba(sc.transform(X)).mean(0); phgb = dhgb.predict_proba(X).mean(0)
    res = dict(n_snps=nsites, D=round(D, 3), Z=round(Z, 1), ABBA=round(abba), BABA=round(baba),
               gate_mean=round(float(gp.mean()), 3),
               direction_logit=dict(zip([str(c) for c in cls], [round(float(x), 3) for x in plog])),
               direction_hgb=dict(zip([str(c) for c in cls], [round(float(x), 3) for x in phgb])))
    print(f"[heliconius] exploratory uncalibrated gate={res['gate_mean']} dir(logit)={res['direction_logit']}", flush=True)
    json.dump(res, open(os.path.join(OUT, "heliconius_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
