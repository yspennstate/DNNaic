#!/usr/bin/env python3
"""A second labelled real-data system: Mus spretus -> M. m. domesticus (wild-mouse genomes).

The warfarin/anticoagulant-resistance haplotype at Vkorc1 (chromosome 7) introgressed from the
outgroup species M. spretus into M. m. domesticus and swept under anthropogenic selection (Song et
al. 2011). In DNNaic's tree ((P1,P2),P3) we set P1 = M. m. musculus, P2 = M. m. domesticus (the
recipient), P3 = M. spretus (the divergent donor), so the documented spretus -> domesticus flow is
class C -- the same donor->recipient geometry as the Heliconius case, in a mammal.

The genotypes are the Harr et al. (2016) wild-mouse joint call (mm10), read from the remote bgzipped
VCF over HTTP range requests with the pure-stdlib tabix client. Unlike Heliconius (broad gene flow),
the spretus introgression is largely localized to the Vkorc1 sweep, so the orientation is strongest
in that region and dilutes genome-wide -- a property this script reports directly by running both a
genome-wide window set and the Vkorc1 region. Rarefaction depth is set by M. spretus (eight genomes,
16 gene copies); requiring all of them genotyped takes the depth to 16, near the Heliconius range.

Configure via environment:
  DNNAIC_DATA   directory containing regen_full/ (the frozen-model training arrays)
  MOUSE_REGION  "vkorc1" for the chr7 introgression locus, else a genome-wide window set (default)

CPU only.
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

PREFIX = {"Mmm": "musculus", "Mmd": "domesticus", "Ms": "spretus"}   # P1, P2 (recipient), P3 (donor)
POPS = ["musculus", "domesticus", "spretus"]
MIN_COPIES = {"musculus": 16, "domesticus": 16, "spretus": 16}       # all 8 spretus -> depth 16
GENOME_WINDOWS = [("chr1", 50_000_000, 53_000_000), ("chr2", 100_000_000, 103_000_000),
                  ("chr4", 80_000_000, 83_000_000), ("chr8", 60_000_000, 63_000_000),
                  ("chr11", 50_000_000, 53_000_000), ("chr13", 80_000_000, 83_000_000),
                  ("chr17", 40_000_000, 43_000_000), ("chr19", 30_000_000, 33_000_000)]
VKORC1_WINDOW = [("chr7", 126_000_000, 129_000_000)]                  # the adaptive-introgression locus
TARGET, PER_WINDOW = 15000, 2000


def header_samples():
    with gzip.GzipFile(fileobj=io.BytesIO(http_get(VCF_URL, rng="bytes=0-1500000"))) as g:
        for line in g:
            s = line.decode("latin-1")
            if s.startswith("#CHROM"):
                return s.rstrip("\n").split("\t")[9:]
    raise SystemExit("no #CHROM header")


def called(cell):
    gt = cell.split(":", 1)[0].replace("|", "/")
    return sum(1 for a in gt.split("/") if a in ("0", "1"))


def collect(names, refs, samples, idx, windows, per_window):
    sel = idx["musculus"] + idx["domesticus"] + idx["spretus"]
    rows = []
    for chrom, beg, end in windows:
        kept = 0
        for line in fetch_region(VCF_URL, names, refs, chrom, beg, end):
            p = line.rstrip("\n").split("\t")
            if len(p) < 10 or len(p[3]) != 1 or len(p[4]) != 1 or p[4] in (".", "*"):
                continue
            if any(sum(called(p[9 + i]) for i in idx[pop]) < MIN_COPIES[pop] for pop in POPS):
                continue
            rows.append((chrom, p[1], p[3], p[4], [p[9 + i].split(":", 1)[0].replace("|", "/") for i in sel]))
            kept += 1
            if kept >= per_window or len(rows) >= TARGET:
                break
        if len(rows) >= TARGET:
            break
    return rows, [samples[i] for i in sel]


def orient(rows, sel_names, sample2pop, sc, dlog):
    vcf, pm = os.path.join(OUT, "mouse.vcf"), os.path.join(OUT, "mouse_popmap.tsv")
    with open(vcf, "w") as o:
        o.write("##fileformat=VCFv4.2\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        o.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sel_names) + "\n")
        for chrom, pos, ref, alt, cells in rows:
            o.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(cells) + "\n")
    with open(pm, "w") as o:
        for s in sel_names:
            o.write(f"{s}\t{sample2pop[s]}\n")
    X, cols, _ = build_matrix(vcf, pm, max_depth=100, pop_order=POPS)
    X = np.asarray(X, float); ix = {c: i for i, c in enumerate(cols)}
    prob = dlog.predict_proba(sc.transform(X)).mean(0)
    p13, p23 = float(np.nanmean(X[:, ix["pihat_13_mean"]])), float(np.nanmean(X[:, ix["pihat_23_mean"]]))
    return dict(n_snp=len(rows), depths=int(X.shape[0]),
                direction=dict(zip("ABC", [round(float(v), 3) for v in prob])),
                call="ABC"[int(np.argmax(prob))],
                model_free_ratio=round(p23 / p13, 3) if p13 else None)


def main():
    names, refs = load_tbi(VCF_URL + ".tbi", remote=True)
    samples = header_samples()
    sample2pop = {s: PREFIX[pre] for s in samples for pre in PREFIX if s.startswith(pre + "_")}
    idx = {p: [i for i, s in enumerate(samples) if sample2pop.get(s) == p] for p in POPS}
    print("samples per population:", {p: len(idx[p]) for p in POPS}, flush=True)

    d = os.path.join(DATA, "regen_full")
    Xtr = np.load(os.path.join(d, "X.npy")); dtr = np.load(os.path.join(d, "direction.npy")).astype("U2")
    mtr = np.load(os.path.join(d, "magnitude.npy"))
    sc = StandardScaler().fit(Xtr)
    appr = np.where((dtr != "D") & (mtr >= 2.5e-4))[0]
    appr = np.random.default_rng(0).choice(appr, min(80000, len(appr)), replace=False)
    dlog = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr[appr]), np.searchsorted(np.array(["A", "B", "C"]), dtr[appr]))
    del Xtr, dtr, mtr

    region = os.environ.get("MOUSE_REGION", "genomewide")
    windows, per_window = (VKORC1_WINDOW, TARGET) if region == "vkorc1" else (GENOME_WINDOWS, PER_WINDOW)
    rows, sel_names = collect(names, refs, samples, idx, windows, per_window)
    res = orient(rows, sel_names, sample2pop, sc, dlog); res["region"] = region
    print(f"[mouse:{region}] n_snp={res['n_snp']} depths={res['depths']} "
          f"direction={res['direction']} -> {res['call']} (documented: C = spretus->domesticus)", flush=True)
    print(f"[mouse:{region}] model-free pihat(dom,spretus)/pihat(mus,spretus) = {res['model_free_ratio']} "
          f"(>1 supports spretus->domesticus)", flush=True)
    json.dump(res, open(os.path.join(OUT, f"mouse_{region}_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
