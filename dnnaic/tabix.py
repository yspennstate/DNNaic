"""Minimal pure-stdlib tabix client: slice a genomic region from a remote bgzipped VCF using a
local or remote .tbi index and HTTP range requests. No bioinformatics CLI, no full download.

Used by the real-data scripts to fetch only the windows they need from the 1000 Genomes and archaic
archives. The archives are served over TLS with certificates this client does not verify (public
read-only data)."""
import gzip, io, ssl, struct, time, urllib.request

_CTX = ssl.create_default_context(); _CTX.check_hostname = False; _CTX.verify_mode = ssl.CERT_NONE
_UA = {"User-Agent": "Mozilla/5.0"}


def http_get(url, rng=None, tries=4, timeout=120):
    """GET url (optionally a byte Range), retrying transient network errors."""
    last = None
    for k in range(tries):
        try:
            h = dict(_UA)
            if rng:
                h["Range"] = rng
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout, context=_CTX).read()
        except Exception as e:  # noqa: BLE001 - retry any network failure
            last = e; time.sleep(3 * (k + 1))
    raise last


def read_tbi(raw):
    """Parse a .tbi (raw bytes) -> (contig_names, linear_index_per_contig)."""
    with gzip.open(io.BytesIO(raw), "rb") as f:
        raw = f.read()
    off = 0
    assert raw[off:off + 4] == b"TBI\x01", "not a tabix index"
    off += 4
    (n_ref, fmt, cs, cb, ce, meta, skip, l_nm) = struct.unpack_from("<8i", raw, off); off += 32
    names = [n.decode() for n in raw[off:off + l_nm].split(b"\x00") if n]; off += l_nm
    refs = []
    for _ in range(n_ref):
        (n_bin,) = struct.unpack_from("<i", raw, off); off += 4
        for _ in range(n_bin):
            (_bid, n_ch) = struct.unpack_from("<Ii", raw, off); off += 8; off += 16 * n_ch
        (n_intv,) = struct.unpack_from("<i", raw, off); off += 4
        intv = list(struct.unpack_from("<%dQ" % n_intv, raw, off)); off += 8 * n_intv
        refs.append(intv)
    return names, refs


def load_tbi(source, remote=False, _cache={}):
    if source in _cache:
        return _cache[source]
    raw = http_get(source) if remote else open(source, "rb").read()
    v = read_tbi(raw); _cache[source] = v; return v


def fetch_region(vcf_url, names, refs, chrom, beg, end, extra=22_000_000):
    """Return the VCF data lines overlapping chrom:[beg,end], sliced by HTTP range."""
    intv = refs[names.index(chrom)]
    w = beg >> 14
    if w >= len(intv):
        return []
    coff = intv[w] >> 16
    data = http_get(vcf_url, rng=f"bytes={coff}-{coff + extra}")
    out = []
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(data)) as g:
            for line in g:
                s = line.decode("latin-1")
                if s.startswith("#"):
                    continue
                p = s.split("\t", 8)
                if p[0] != chrom:
                    continue
                pos = int(p[1])
                if pos < beg:
                    continue
                if pos > end:
                    break
                out.append(s)
    except (EOFError, OSError):
        pass
    return out
