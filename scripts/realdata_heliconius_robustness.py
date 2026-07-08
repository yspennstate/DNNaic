#!/usr/bin/env python3
"""Robustness of the Heliconius direction call: is C = melpomene -> timareta stable across race
trios and across the genome? Two checks. (1) Multiple (cydno, timareta, melpomene) race trios: the
call should be C wherever there is appreciable gene flow (large |D|) and should not assert C where
there is none (an allopatric control). (2) A leave-one-chromosome-out jackknife on the canonical
trio: dropping any single chromosome (locus count held near constant, so the n_loci-dependent
standard-error features stay on scale) should leave the call at C.

Reads Simon Martin's whole-genome .geno for the ABBA-BABA tutorial (downloaded if absent), streamed
with reservoir sampling so the ~10^6-SNP file is never held in memory. Set DNNAIC_DATA to the
regen_full arrays. CPU only.
"""
import os, gzip, io, random, json, urllib.request, ssl, numpy as np
from padze import read_vcf
from dnnaic import build_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "real", "heliconius"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("DNNAIC_DATA", "data/simulation_data")
GENO = os.environ.get("HEL_GENO", os.path.join(OUT, "hel.geno.gz"))
POP = os.environ.get("HEL_POP", os.path.join(OUT, "hel.pop.txt"))
BASE = "https://github.com/simonhmartin/tutorials/raw/master/ABBA_BABA_whole_genome/data"
GENO_URL, POP_URL = f"{BASE}/hel92.DP8MP4BIMAC2HET75dist250.geno.gz", f"{BASE}/hel92.pop.txt"
TARGET, BASES = 15000, set("ACGT")
# introgressing sympatric trios (documented mel->tim = C) + an allopatric control (mel_ros)
TRIOS = [("cyd_chi", "tim_txn", "mel_ama"), ("cyd_zel", "tim_flo", "mel_mel"),
         ("cyd_chi", "tim_flo", "mel_mal"), ("cyd_zel", "tim_txn", "mel_ama"),
         ("cyd_zel", "tim_flo", "mel_ros")]
_CTX = ssl.create_default_context(); _CTX.check_hostname = False; _CTX.verify_mode = ssl.CERT_NONE


def _download(url, dst):
    if not os.path.exists(dst):
        print(f"downloading {os.path.basename(dst)} ...", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=_CTX) as r, open(dst, "wb") as o:
            o.write(r.read())


def load_popmap():
    _download(POP_URL, POP)
    m = {}
    for ln in open(POP):
        p = ln.split()
        if len(p) >= 2:
            m[p[0]] = p[1]
    return m


def build_vcf(samp2race, sel, seed, tag):
    sample2pop = {s: pop for pop, races in sel.items() for s, r in samp2race.items() if r in races}
    f = gzip.open(GENO, "rt"); header = f.readline().rstrip("\n").split("\t")
    colidx = {n: i for i, n in enumerate(header)}
    need = [s for s in sample2pop if s in colidx]; pops = list(sel.keys())
    rng = random.Random(seed); reservoir = []; seen = 0
    for line in f:
        parts = line.rstrip("\n").split("\t"); cnt = {}; gts = {}
        for s in need:
            g = parts[colidx[s]]; a, b = g[0], (g[2] if len(g) >= 3 else "N"); gts[s] = (a, b)
            for x in (a, b):
                if x in BASES:
                    cnt[x] = cnt.get(x, 0) + 1
        al = sorted(cnt, key=lambda k: -cnt[k])
        if len(al) != 2:
            continue
        ref, alt = al; ok = True
        for p in pops:
            if p == "numata":
                continue
            tot = sum(1 for s in need if sample2pop[s] == p for x in gts[s] if x in (ref, alt))
            if tot < 16:
                ok = False; break
        if not ok:
            continue
        seen += 1; rec = (parts[0], parts[1], ref, alt, [gts[s] for s in need])
        if len(reservoir) < TARGET:
            reservoir.append(rec)
        else:
            j = rng.randint(0, seen - 1)
            if j < TARGET:
                reservoir[j] = rec
    f.close(); reservoir.sort(key=lambda r: (r[0], int(r[1])))
    vcf = os.path.join(OUT, f"robust_{tag}.vcf")
    with open(vcf, "w") as o:
        o.write("##fileformat=VCFv4.2\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        o.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(need) + "\n")
        for chrom, pos, ref, alt, gtl in reservoir:
            cells = ["/".join("0" if x == ref else ("1" if x == alt else ".") for x in ab) for ab in gtl]
            o.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(cells) + "\n")
    pm3, pm4 = os.path.join(OUT, f"robust_pm3_{tag}.tsv"), os.path.join(OUT, f"robust_pm4_{tag}.tsv")
    with open(pm3, "w") as o:
        for s in need:
            if sample2pop[s] in pops[:3]:
                o.write(f"{s}\t{sample2pop[s]}\n")
    with open(pm4, "w") as o:
        for s in need:
            o.write(f"{s}\t{sample2pop[s]}\n")
    return vcf, pm3, pm4, len(reservoir)


def patterson_D(vcf, pm4, P1, P2, P3, O="numata"):
    loci = read_vcf(vcf, pm4); ip = {p: i for i, p in enumerate(loci.populations)}
    alt = {p: [] for p in loci.populations}; chroms = []
    for li, cm in enumerate(loci.count_matrices):
        if cm.shape[1] != 2:
            continue
        N = cm.sum(1)
        if np.any(N == 0):
            continue
        for p in loci.populations:
            alt[p].append(cm[ip[p], 1] / N[ip[p]])
        chroms.append(loci.locus_ids[li].rpartition(":")[0])
    alt = {p: np.array(v) for p, v in alt.items()}; chroms = np.array(chroms)
    flip = alt[O] > 0.5; dd = {p: np.where(flip, 1 - alt[p], alt[p]) for p in (P1, P2, P3, O)}
    abba = (1 - dd[P1]) * dd[P2] * dd[P3]; baba = dd[P1] * (1 - dd[P2]) * dd[P3]
    D = (abba.sum() - baba.sum()) / (abba.sum() + baba.sum())
    ub = np.unique(chroms); ths = []
    for b in ub:
        m = chroms != b; n = abba[m].sum() - baba[m].sum(); de = abba[m].sum() + baba[m].sum()
        ths.append(n / de if de else np.nan)
    ths = np.array(ths); se = np.sqrt((len(ub) - 1) / len(ub) * np.nansum((ths - np.nanmean(ths)) ** 2))
    return float(D), float(D / se if se else np.nan)


def direction(vcf, pm3, sc, dlog):
    X, _, _ = build_matrix(vcf, pm3, max_depth=100, pop_order=["cydno", "timareta", "melpomene"])
    prob = dlog.predict_proba(sc.transform(np.asarray(X, float))).mean(0)
    return "ABC"[int(np.argmax(prob))], round(float(prob[2]), 3)


def main():
    _download(GENO_URL, GENO); samp2race = load_popmap()
    d = os.path.join(DATA, "regen_full")
    Xtr = np.load(os.path.join(d, "X.npy")); dtr = np.load(os.path.join(d, "direction.npy")).astype("U2")
    mtr = np.load(os.path.join(d, "magnitude.npy"))
    sc = StandardScaler().fit(Xtr)
    appr = np.where((dtr != "D") & (mtr >= 2.5e-4))[0]
    appr = np.random.default_rng(0).choice(appr, min(80000, len(appr)), replace=False)
    dlog = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr[appr]), np.searchsorted(np.array(["A", "B", "C"]), dtr[appr]))
    del Xtr, dtr, mtr
    print("model ready", flush=True)

    print("\n== multi-trio ==", flush=True); trio_res = []
    canonical = None
    for i, (cr, tr, mr) in enumerate(TRIOS):
        sel = {"cydno": [cr], "timareta": [tr], "melpomene": [mr], "numata": ["num"]}
        vcf, pm3, pm4, n = build_vcf(samp2race, sel, 100 + i, f"t{i}")
        D, Z = patterson_D(vcf, pm4, "cydno", "timareta", "melpomene")
        call, pc = direction(vcf, pm3, sc, dlog)
        trio_res.append(dict(trio=f"{cr}/{tr}/{mr}", D=round(D, 3), Z=round(Z, 1), call=call, P_C=pc))
        print(f"  {cr}/{tr}/{mr}: D={D:+.3f} Z={Z:+.1f} -> {call} (P_C={pc})", flush=True)
        if i == 0:
            canonical = (vcf, pm3)
    intro = [r for r in trio_res if abs(r["Z"]) > 3]
    print(f"  {sum(r['call']=='C' for r in intro)}/{len(intro)} introgressing trios call C", flush=True)

    print("\n== leave-one-chromosome-out jackknife (canonical trio) ==", flush=True)
    vcf, pm3 = canonical
    with open(vcf) as f:
        lines = f.readlines()
    hdr = [l for l in lines if l.startswith("#")]; data = [l for l in lines if not l.startswith("#")]
    chroms = sorted(set(l.split("\t", 1)[0] for l in data))
    jk = os.path.join(OUT, "robust_jk.vcf"); jk_res = []
    for c in chroms:
        sub = [l for l in data if not l.startswith(c + "\t")]
        if len(sub) < 1000:
            continue
        with open(jk, "w") as o:
            o.writelines(hdr); o.writelines(sub)
        call, pc = direction(jk, pm3, sc, dlog); jk_res.append((c, call, pc))
        print(f"  drop {c} ({len(sub)} SNP): {call} (P_C={pc})", flush=True)
    print(f"  C stable in {sum(1 for _, cl, _ in jk_res if cl == 'C')}/{len(jk_res)} folds", flush=True)
    json.dump(dict(trios=trio_res, jackknife=[dict(drop=c, call=cl, P_C=pc) for c, cl, pc in jk_res]),
              open(os.path.join(OUT, "robustness_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
