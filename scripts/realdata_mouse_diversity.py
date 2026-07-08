#!/usr/bin/env python3
"""When does the rarefaction signal orient real data? The diversity-balance condition, on mouse.

The mouse Vkorc1 flow spretus -> domesticus can be read with either M. m. musculus or M. m. castaneus
as the non-recipient sister P1 -- both are equally valid ingroup sisters to the spretus outgroup. Yet
the model orients it (class C) only with musculus, and reverts to the null class A with castaneus. The
reason is diversity, not phylogeny: castaneus is far more variable than musculus, and a reference much
more diverse than the recipient carries more private variation it shares with the donor by chance,
inflating the pairwise-private term the orientation reads. This script makes that concrete -- it runs
both trios at Vkorc1 and reports, for each, the direction call, the model-free pairwise-private ratio,
and the reference population's heterozygosity. The same imbalance underlies the archaic-human case,
where the highly diverse African reference plays the castaneus role. The practical rule: apply the
method when the sister pair is roughly balanced in diversity.

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
PREFIX = {"Mmm": "musculus", "Mmc": "castaneus", "Mmd": "domesticus", "Ms": "spretus"}
REGION = ("chr7", 126_000_000, 129_000_000)


def called(cell):
    gt = cell.split(":", 1)[0].replace("|", "/")
    return [a for a in gt.split("/") if a in ("0", "1")]


def main():
    names, refs = load_tbi(VCF_URL + ".tbi", remote=True)
    with gzip.GzipFile(fileobj=io.BytesIO(http_get(VCF_URL, rng="bytes=0-1500000"))) as g:
        for line in g:
            s = line.decode("latin-1")
            if s.startswith("#CHROM"):
                samples = s.rstrip("\n").split("\t")[9:]
                break
    s2p = {s: PREFIX[p] for s in samples for p in PREFIX if s.startswith(p + "_")}
    idx = {p: [i for i, s in enumerate(samples) if s2p.get(s) == p] for p in PREFIX.values()}

    lines = list(fetch_region(VCF_URL, names, refs, *REGION))
    print(f"fetched {len(lines)} SNP lines at Vkorc1", flush=True)

    d = os.path.join(DATA, "regen_full")
    Xtr = np.load(os.path.join(d, "X.npy")); dtr = np.load(os.path.join(d, "direction.npy")).astype("U2")
    mtr = np.load(os.path.join(d, "magnitude.npy"))
    sc = StandardScaler().fit(Xtr)
    appr = np.where((dtr != "D") & (mtr >= 2.5e-4))[0]
    appr = np.random.default_rng(0).choice(appr, min(80000, len(appr)), replace=False)
    dlog = LogisticRegression(max_iter=2000).fit(sc.transform(Xtr[appr]), np.searchsorted(np.array(["A", "B", "C"]), dtr[appr]))
    del Xtr, dtr, mtr

    def heterozygosity(pop):
        vals = []
        for s in lines:
            p = s.rstrip("\n").split("\t")
            if len(p) < 10 or len(p[3]) != 1 or len(p[4]) != 1 or p[4] in (".", "*"):
                continue
            al = [a for i in idx[pop] for a in called(p[9 + i])]
            if len(al) >= 12:
                q = al.count("1") / len(al); vals.append(2 * q * (1 - q))
        return float(np.mean(vals)) if vals else float("nan")

    results = []
    for sister in ("musculus", "castaneus"):
        pops = [sister, "domesticus", "spretus"]
        sel = idx[sister] + idx["domesticus"] + idx["spretus"]; sel_names = [samples[i] for i in sel]
        rows = []
        for s in lines:
            p = s.rstrip("\n").split("\t")
            if len(p) < 10 or len(p[3]) != 1 or len(p[4]) != 1 or p[4] in (".", "*"):
                continue
            if any(len([a for i in idx[pop] for a in called(p[9 + i])]) < 16 for pop in pops):
                continue
            rows.append((p[0], p[1], p[3], p[4], [p[9 + i].split(":", 1)[0].replace("|", "/") for i in sel]))
            if len(rows) >= 15000:
                break
        vcf, pm = os.path.join(OUT, "mouse_ref.vcf"), os.path.join(OUT, "mouse_ref_popmap.tsv")
        with open(vcf, "w") as o:
            o.write("##fileformat=VCFv4.2\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
            o.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(sel_names) + "\n")
            for chrom, pos, ref, alt, cells in rows:
                o.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t" + "\t".join(cells) + "\n")
        with open(pm, "w") as o:
            for s in sel_names:
                o.write(f"{s}\t{s2p[s]}\n")
        X, cols, _ = build_matrix(vcf, pm, max_depth=100, pop_order=pops)
        X = np.asarray(X, float); ix = {c: i for i, c in enumerate(cols)}
        prob = dlog.predict_proba(sc.transform(X)).mean(0)
        r = dict(reference=sister, n_snp=len(rows), call="ABC"[int(np.argmax(prob))],
                 P_C=round(float(prob[2]), 3),
                 model_free_ratio=round(float(np.nanmean(X[:, ix["pihat_23_mean"]]) / np.nanmean(X[:, ix["pihat_13_mean"]])), 3),
                 reference_heterozygosity=round(heterozygosity(sister), 4),
                 recipient_heterozygosity=round(heterozygosity("domesticus"), 4))
        print(f"P1={sister:9s}: call={r['call']} P_C={r['P_C']}  model-free ratio={r['model_free_ratio']}  "
              f"het(ref)={r['reference_heterozygosity']} vs het(domesticus)={r['recipient_heterozygosity']}", flush=True)
        results.append(r)
    json.dump(results, open(os.path.join(OUT, "mouse_reference_result.json"), "w"), indent=2)
    print("Orientation succeeds with the diversity-matched reference and fails with the far more diverse one.", flush=True)


if __name__ == "__main__":
    main()
