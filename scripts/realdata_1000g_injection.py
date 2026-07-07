#!/usr/bin/env python3
"""Injected-signal validation on real 1000 Genomes genotype backgrounds.

Natural human trios carry no labelled within-population introgression, so we validate direction
recovery by injecting a KNOWN signal into real genotypes: a fraction f of a recipient population's
individuals are turned into migrants carrying a random donor individual's real genotypes (real
allele frequencies and linkage), then the released PADZE and the frozen simulation-trained model are
run on the result. Trio CEU=P1, CHB=P2 (sisters), YRI=P3 (outgroup): class B = CHB->YRI, C = YRI->CHB
(the hard reversed pair), D = control. The method recovers the injected direction once the signal is
appreciable (f>=0.1) and abstains on the control, the same appreciable-rate dependence as the
simulations.

Streams the five benchmark windows over HTTP range requests (no full download). CPU only.
Configure DNNAIC_DATA (regen_full) and DNNAIC_1000G_TBI (or let it download the .tbi files)."""
import os, random, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dnnaic import build_matrix
from dnnaic.tabix import http_get, load_tbi, fetch_region

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "real", "1000g_injection"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("DNNAIC_DATA", "data/simulation_data")
TBIDIR = os.environ.get("DNNAIC_1000G_TBI", OUT)
KG_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
KG_TBI = "ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz.tbi"
PANEL_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"
WINDOWS = [("1", 50_000_000, 50_250_000), ("2", 100_000_000, 100_250_000), ("8", 30_000_000, 30_250_000),
           ("15", 60_000_000, 60_250_000), ("22", 20_000_000, 20_250_000)]
NPER = 50


def load_panel():
    dst = os.path.join(OUT, "panel.txt")
    if not os.path.exists(dst):
        open(dst, "wb").write(http_get(PANEL_URL))
    return {w[0]: w[1] for w in (ln.split() for ln in open(dst)) if len(w) >= 2 and w[0] != "sample"}


def kg_samples():
    dst = os.path.join(OUT, "kg_samples.txt")
    if os.path.exists(dst):
        return open(dst).read().split()
    import gzip, io
    data = http_get(KG_URL.format(c="22"), rng="bytes=0-300000")
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as g:
        for line in g:
            s = line.decode("latin-1")
            if s.startswith("#CHROM"):
                samp = s.rstrip("\n").split("\t")[9:]; open(dst, "w").write("\n".join(samp)); return samp
    raise SystemExit("no header")


def build_base_trio(pop_of, samples, colidx):
    """CEU(0..49), CHB(50..99), YRI(100..149) genotype matrix + record templates."""
    want = []
    for p in ("CEU", "CHB", "YRI"):
        want += [s for s in samples if pop_of.get(s) == p][:NPER]
    idx = [colidx[s] for s in want]
    recs = []
    for c, beg, end in WINDOWS:
        dst = os.path.join(TBIDIR, KG_TBI.format(c=c))
        if not os.path.exists(dst):
            open(dst, "wb").write(http_get(KG_URL.format(c=c) + ".tbi"))
        names, refs = load_tbi(dst)
        for s in fetch_region(KG_URL.format(c=c), names, refs, c, beg, end):
            p = s.rstrip("\n").split("\t")
            if len(p) < 9 + len(samples) or len(p[3]) != 1 or len(p[4]) != 1:
                continue
            sel = [p[9 + i].split(":")[0] for i in idx]
            al = "".join(sel)
            if "0" not in al or "1" not in al:
                continue
            recs.append((p[:8], sel))
    print(f"base trio: {len(recs)} biallelic SNPs, {len(want)} samples", flush=True)
    return want, recs


def injected_matrix(want, recs, direction, f, seed):
    CEU, CHB, YRI = range(0, 50), range(50, 100), range(100, 150)
    donor_recip = {"B": (list(CHB), list(YRI)), "C": (list(YRI), list(CHB)), "D": (None, None)}[direction]
    donor, recip = donor_recip
    rng = random.Random(seed)
    g = [list(sel) for _, sel in recs]
    if donor is not None:
        migs = rng.sample(recip, max(1, int(round(f * len(recip)))))
        for mi in migs:
            di = rng.choice(donor)
            for j in range(len(g)):
                g[j][mi] = g[j][di]
    vcf = os.path.join(OUT, f"inj_{direction}_f{f}_s{seed}.vcf")
    pm = os.path.join(OUT, "popmap.tsv")
    with open(pm, "w") as o:
        for i, s in enumerate(want):
            o.write(f"{s}\t{'CEU' if i < 50 else ('CHB' if i < 100 else 'YRI')}\n")
    with open(vcf, "w") as o:
        o.write("##fileformat=VCFv4.1\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        o.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(want) + "\n")
        for (meta, _), row in zip(recs, g):
            o.write("\t".join(meta) + "\tGT\t" + "\t".join(row) + "\n")
    X, cols, _ = build_matrix(vcf, pm, max_depth=100, pop_order=["CEU", "CHB", "YRI"])
    os.remove(vcf)
    return np.asarray(X, float)


def main():
    pop_of = load_panel(); samples = kg_samples(); colidx = {s: i for i, s in enumerate(samples)}
    want, recs = build_base_trio(pop_of, samples, colidx)
    d = os.path.join(DATA, "regen_full")
    Xtr = np.load(os.path.join(d, "X.npy")); dtr = np.load(os.path.join(d, "direction.npy")); mtr = np.load(os.path.join(d, "magnitude.npy"))
    sc = StandardScaler().fit(Xtr); appr = (dtr != "D") & (mtr >= 2.5e-4); cls = np.array(["A", "B", "C"])
    gate = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr), ((dtr != "D") & (mtr >= 2.5e-4)).astype(int))
    dclf = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr[appr]), np.searchsorted(cls, dtr[appr]))
    res = []
    for direction, f in [("D", 0.0), ("B", 0.05), ("C", 0.05), ("B", 0.1), ("C", 0.1), ("B", 0.2), ("C", 0.2)]:
        correct = 0; gates = []
        for rep in range(4):
            X = injected_matrix(want, recs, direction, f, seed=1000 * rep + int(f * 100))
            gp = float(gate.predict_proba(sc.transform(X))[:, 1].mean())
            call = cls[dclf.predict_proba(sc.transform(X)).mean(0).argmax()]
            gates.append(gp)
            if direction != "D" and call == direction:
                correct += 1
        r = dict(inject=direction, f=f, mean_gate=round(float(np.mean(gates)), 3),
                 direction_accuracy=(None if direction == "D" else f"{correct}/4"))
        print(r, flush=True); res.append(r)
    json.dump(res, open(os.path.join(OUT, "injection_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
