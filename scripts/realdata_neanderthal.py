#!/usr/bin/env python3
"""Neanderthal introgression into non-Africans, on real human + archaic genotypes.

Tree (((African, Eurasian), Neanderthal), ancestral). Documented (Green et al. 2010; Prufer et al.
2014): Neanderthals contributed ~2% ancestry to non-Africans and essentially none to sub-Saharan
Africans. In DNNaic's tree ((P1,P2),P3) we set P1 = African (YRI), P2 = Eurasian (recipient),
P3 = Neanderthal (donor, outgroup to modern humans), so Neanderthal -> Eurasian is class C.

Patterson's D(African, Eurasian, Neanderthal; ancestral) DETECTS the gene flow (D>0, ABBA>BABA) but
is symmetric under donor/recipient exchange; the labelled ABBA/BABA asymmetry localises the excess
sharing to the Eurasian. The Neanderthal population here is the two high-coverage genomes Altai and
Vindija 33.19 (four gene copies): archaic data offers only a handful of high-coverage genomes, so
the common rarefaction depth is very shallow -- far below the depth the sim-trained model needs
(see scripts on the depth requirement), which is why its gate correctly abstains. The site-based D
does not depend on rarefaction depth and still detects.

Modern genotypes: 1000 Genomes phase 3. All data are GRCh37 and are sliced over HTTP range requests
using each file's .tbi index (no full download). CPU only.

Configure via environment:
  DNNAIC_DATA        directory containing regen_full/ (the frozen-model training arrays)
  DNNAIC_1000G_TBI   directory with the 1000G phase3 per-chromosome .tbi files
                     (downloaded automatically if unset)
"""
import os, sys, gzip, io, ssl, struct, time, urllib.request, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from dnnaic import build_matrix

CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
UA = {"User-Agent": "Mozilla/5.0"}
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "data", "real", "neanderthal"); os.makedirs(OUT, exist_ok=True)
DATA = os.environ.get("DNNAIC_DATA", "data/simulation_data")
TBIDIR = os.environ.get("DNNAIC_1000G_TBI", OUT)

KG_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
KG_TBI = "ALL.chr{c}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz.tbi"
PANEL_URL = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel"
ARCHAIC = {"Altai": "https://ftp.eva.mpg.de/neandertal/Vindija/VCF/Altai/chr{c}_mq25_mapab100.vcf.gz",
           "Vindija": "https://ftp.eva.mpg.de/neandertal/Vindija/VCF/Vindija33.19/chr{c}_mq25_mapab100.vcf.gz"}
NEA_SAMPLES = ["NEA_Altai", "NEA_Vindija"]
WINDOWS = [("1", 50_000_000, 50_250_000), ("2", 100_000_000, 100_250_000), ("8", 30_000_000, 30_250_000),
           ("15", 60_000_000, 60_250_000), ("22", 20_000_000, 20_250_000)]


def _get(url, rng=None, tries=4, timeout=180):
    last = None
    for k in range(tries):
        try:
            h = dict(UA)
            if rng: h["Range"] = rng
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout, context=CTX).read()
        except Exception as e:
            last = e; time.sleep(3 * (k + 1))
    raise last


def _read_tbi(raw):
    with gzip.open(io.BytesIO(raw), "rb") as f: raw = f.read()
    off = 0; assert raw[:4] == b"TBI\x01"; off += 4
    (n_ref, fmt, cs, cb, ce, meta, skip, l_nm) = struct.unpack_from("<8i", raw, off); off += 32
    names = [n.decode() for n in raw[off:off + l_nm].split(b"\x00") if n]; off += l_nm
    refs = []
    for _ in range(n_ref):
        (n_bin,) = struct.unpack_from("<i", raw, off); off += 4
        for _ in range(n_bin):
            (bid, n_ch) = struct.unpack_from("<Ii", raw, off); off += 8; off += 16 * n_ch
        (n_intv,) = struct.unpack_from("<i", raw, off); off += 4
        intv = list(struct.unpack_from("<%dQ" % n_intv, raw, off)); off += 8 * n_intv
        refs.append(intv)
    return names, refs


_TBI = {}
def get_tbi(key, remote=False):
    if key in _TBI: return _TBI[key]
    raw = _get(key) if remote else open(key, "rb").read()
    v = _read_tbi(raw); _TBI[key] = v; return v


def fetch_region(url, names, refs, chrom, beg, end, extra=22_000_000):
    intv = refs[names.index(chrom)]; w = beg >> 14
    if w >= len(intv): return []
    data = _get(url, rng=f"bytes={intv[w] >> 16}-{(intv[w] >> 16) + extra}")
    out = []
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as g:
            for line in g:
                s = line.decode("latin-1")
                if s.startswith("#"): continue
                p = s.split("\t", 8)
                if p[0] != chrom: continue
                pos = int(p[1])
                if pos < beg: continue
                if pos > end: break
                out.append(s)
    except (EOFError, OSError): pass
    return out


def ensure_1000g_tbi():
    for c, _, _ in WINDOWS:
        dst = os.path.join(TBIDIR, KG_TBI.format(c=c))
        if not os.path.exists(dst):
            open(dst, "wb").write(_get(KG_URL.format(c=c) + ".tbi"))


def load_panel():
    dst = os.path.join(OUT, "panel.txt")
    if not os.path.exists(dst):
        open(dst, "wb").write(_get(PANEL_URL))
    pop = {}
    for ln in open(dst):
        w = ln.split()
        if len(w) >= 2 and w[0] != "sample": pop[w[0]] = w[1]
    return pop


def kg_samples():
    dst = os.path.join(OUT, "kg_samples.txt")
    if os.path.exists(dst): return open(dst).read().split()
    data = _get(KG_URL.format(c="22"), rng="bytes=0-300000")
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as g:
        for line in g:
            s = line.decode("latin-1")
            if s.startswith("#CHROM"):
                samp = s.rstrip("\n").split("\t")[9:]; open(dst, "w").write("\n".join(samp)); return samp
    raise SystemExit("no #CHROM header")


def parse_archaic(s):
    p = s.rstrip("\n").split("\t")
    if len(p) < 10 or len(p[3]) != 1: return None
    gt = p[9].split(":")[0].replace("|", "/")
    if "/" not in gt: return None
    a, b = gt.split("/")[:2]
    if a == "." or b == ".": return None
    alts = p[4].split(",")
    def nuc(code):
        code = int(code)
        if code == 0: return p[3]
        if code <= len(alts): return alts[code - 1] if len(alts[code - 1]) == 1 else None
        return None
    n0, n1 = nuc(a), nuc(b)
    if n0 is None or n1 is None: return None
    return int(p[1]), p[3], (n0, n1)


def parse_AA(info):
    for kv in info.split(";"):
        if kv.startswith("AA="):
            aa = kv[3:].split("|")[0].strip().upper()
            return aa if aa in ("A", "C", "G", "T") else None
    return None


def prefetch():
    W = {}
    for c, beg, end in WINDOWS:
        kn, kr = get_tbi(os.path.join(TBIDIR, KG_TBI.format(c=c)))
        kg = fetch_region(KG_URL.format(c=c), kn, kr, c, beg, end)
        arch = {}
        for nm, tmpl in ARCHAIC.items():
            url = tmpl.format(c=c); n, r = get_tbi(url + ".tbi", remote=True)
            d = {}
            for s in fetch_region(url, n, r, c, beg, end):
                rr = parse_archaic(s)
                if rr: d[rr[0]] = (rr[1], rr[2])
            arch[nm] = d
        W[c] = (kg, arch)
        print(f"  chr{c}: {len(kg)} KG lines; Nea Altai/Vind={len(arch['Altai'])}/{len(arch['Vindija'])}", flush=True)
    return W


def recode(entry, ref, alt):
    aref, (n0, n1) = entry
    if aref != ref: return None
    codes = []
    for al in (n0, n1):
        if al == ref: codes.append("0")
        elif al == alt: codes.append("1")
        else: return None
    return codes


def build_trio(W, pop_of, samples, colidx, P1, P2, tag, n_per_pop=50):
    afr = [s for s in samples if pop_of.get(s) == P1][:n_per_pop]
    eur = [s for s in samples if pop_of.get(s) == P2][:n_per_pop]
    ai = [colidx[s] for s in afr]; ei = [colidx[s] for s in eur]
    vcf = os.path.join(OUT, f"nea_{tag}.vcf"); pm = os.path.join(OUT, f"neapm_{tag}.tsv")
    with open(pm, "w") as g:
        for s in afr: g.write(f"{s}\t{P1}\n")
        for s in eur: g.write(f"{s}\t{P2}\n")
        for s in NEA_SAMPLES: g.write(f"{s}\tNeanderthal\n")
    dP1, dP2, dP3, blocks = [], [], [], []; nsnp = 0
    with open(vcf, "w") as out:
        out.write("##fileformat=VCFv4.1\n##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")
        out.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(afr + eur + NEA_SAMPLES) + "\n")
        for c, beg, end in WINDOWS:
            kg, arch = W[c]
            for s in kg:
                p = s.rstrip("\n").split("\t")
                if len(p) < 9 + len(samples) or len(p[3]) != 1 or len(p[4]) != 1: continue
                ref, alt, pos, info = p[3], p[4], int(p[1]), p[7]
                aa = parse_AA(info)
                if aa is None or aa not in (ref, alt): continue
                ea, ev = arch["Altai"].get(pos), arch["Vindija"].get(pos)
                if ea is None or ev is None: continue
                ca, cv = recode(ea, ref, alt), recode(ev, ref, alt)
                if ca is None or cv is None: continue
                nea_cells = [f"{ca[0]}/{ca[1]}", f"{cv[0]}/{cv[1]}"]
                der = alt if aa == ref else ref; dc = "1" if der == alt else "0"
                agt = [p[9 + i].split(":")[0] for i in ai]; egt = [p[9 + i].split(":")[0] for i in ei]
                def dfreq(gts):
                    n = d = 0
                    for gt in gts:
                        for a in gt.replace("|", "/").split("/"):
                            if a in ("0", "1"): n += 1; d += (a == dc)
                    return (d / n if n else np.nan), n
                f1, n1 = dfreq(agt); f2, n2 = dfreq(egt)
                nea_der = sum(x == dc for x in ca + cv) / 4.0
                if n1 == 0 or n2 == 0: continue
                out.write("\t".join(p[:8]) + "\tGT\t" + "\t".join(agt + egt + nea_cells) + "\n")
                dP1.append(f1); dP2.append(f2); dP3.append(nea_der); blocks.append(c); nsnp += 1
    print(f"[{tag}] {nsnp} intersected biallelic SNPs, {len(afr)}+{len(eur)} modern + 2 Neanderthals", flush=True)
    return vcf, pm, (np.array(dP1), np.array(dP2), np.array(dP3), np.array(blocks)), nsnp


def patterson_D(dP1, dP2, dP3, blocks):
    abba = (1 - dP1) * dP2 * dP3; baba = dP1 * (1 - dP2) * dP3
    num, den = np.nansum(abba) - np.nansum(baba), np.nansum(abba) + np.nansum(baba)
    D = num / den if den else np.nan
    ub = np.unique(blocks); ths = []
    for b in ub:
        mmask = blocks != b; n = np.nansum(abba[mmask]) - np.nansum(baba[mmask]); dd = np.nansum(abba[mmask]) + np.nansum(baba[mmask])
        ths.append(n / dd if dd else np.nan)
    ths = np.array(ths); gk = len(ub); var = (gk - 1) / gk * np.nansum((ths - np.nanmean(ths)) ** 2)
    se = np.sqrt(var); return float(D), float(D / se if se else np.nan), float(np.nansum(abba)), float(np.nansum(baba)), int(gk)


def load_model():
    d = os.path.join(DATA, "regen_full")
    X = np.load(os.path.join(d, "X.npy")); dr = np.load(os.path.join(d, "direction.npy")); mg = np.load(os.path.join(d, "magnitude.npy"))
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    keep = (dr == "D") | (mg >= 2.5e-4)
    gate = LogisticRegression(max_iter=2000).fit(Xs[keep], ((dr != "D") & (mg >= 2.5e-4))[keep].astype(int))
    appr = (dr != "D") & (mg >= 2.5e-4); cls = np.array(["A", "B", "C"])
    dlog = LogisticRegression(max_iter=2000).fit(Xs[appr], np.searchsorted(cls, dr[appr]))
    return sc, gate, dlog, cls


def main():
    pop_of = load_panel(); ensure_1000g_tbi(); samples = kg_samples(); colidx = {s: i for i, s in enumerate(samples)}
    print("prefetching windows...", flush=True); W = prefetch()
    sc, gate, dlog, cls = load_model(); print("model ready", flush=True)
    trios = [("YRI", "CEU", "YRI_CEU_NEA"), ("YRI", "CHB", "YRI_CHB_NEA"),
             ("YRI", "LWK", "YRI_LWK_NEA"), ("CEU", "CHB", "CEU_CHB_NEA")]
    res = []
    for P1, P2, tag in trios:
        vcf, pm, ds, nsnp = build_trio(W, pop_of, samples, colidx, P1, P2, tag)
        D, Z, abba, baba, nblk = patterson_D(*ds)
        X, cols, loci = build_matrix(vcf, pm, max_depth=100, pop_order=[P1, P2, "Neanderthal"])
        X = np.asarray(X, float); gp = gate.predict_proba(sc.transform(X))[:, 1]
        dp = dlog.predict_proba(sc.transform(X)).mean(0)
        r = dict(trio=f"{P1}/{P2}/Neanderthal", n_snps=nsnp, D=round(D, 4), Z=round(Z, 2),
                 ABBA=round(abba, 1), BABA=round(baba, 1), gate_mean=round(float(gp.mean()), 3),
                 direction=dict(zip([str(x) for x in cls], [round(float(x), 3) for x in dp])))
        print(f"[{tag}] D={D:+.4f} Z={Z:+.2f} ABBA={abba:.0f}>BABA={baba:.0f}  gate={r['gate_mean']}  dir={r['direction']}", flush=True)
        res.append(r)
    json.dump(res, open(os.path.join(OUT, "neanderthal_result.json"), "w"), indent=2)
    print("done ->", os.path.join(OUT, "neanderthal_result.json"))


if __name__ == "__main__":
    main()
