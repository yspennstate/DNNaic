# External benchmark manifests

These fixed manifests support `scripts/external_benchmarks.py`. They add one small targeted
dataset and one larger genome-wide dataset as **exploratory out-of-distribution diagnostics**,
not as independent validation of a classifier trained only on the repository simulations.

## Andean ducks (small)

The Dryad release for Graham et al. contains 10 low-elevation yellow-billed pintails (P1),
10 high-elevation pintails (P2), and 10 high-elevation speckled teal (P3). The beta-globin
panel is a published positive (D=0.78, fD=0.68), with ST-high -> YBP-high inferred in a
separate IMa2 analysis; alpha-globin is a published negative control (D=-0.04, P=0.477).
The actual quartet is `((YBP-high,YBP-low),(ST-high,ST-low))`, so this is explicitly also a
topology/domain-shift test rather than a clean match to the pectinate training tree.

## Scarlet runner bean (large)

The OSF VCF contains 237 samples. Three author-metadata-derived manifests reproduce the
published Table S4 population definitions while following the canonical relatedness-filtered
sample set: Cult-SUR-CH / Cult-TMVB / Wild-TMVB-CDMX is the primary class-C benchmark
(D=0.109339, f4-ratio=0.421888, z=8.37171), and replacing the donor with Wild-TMVB-Tepoz is a
published replicate (D=0.107835, f4-ratio=0.156145, z=6.95236). A third manifest uses the
published null Cult-SMOCC / Cult-TMVB-Spain / Cult-TMVB contrast (D=0.0008093, P=0.96497).
The direction head has no no-event class, so the null is evaluated as a discordance/abstention
diagnostic, not by demanding a null label.

All three bean panels are selected from one six-population callable-site intersection before
the deterministic 15,000-locus cap. Their locus identities and order are therefore identical;
population or donor changes cannot silently change the tested sites.

The released provenance has a one-sample inconsistency that matters for exact comparison:
the canonical 183-sample metadata includes `FrijCol_11` and excludes `FrijCol_10`, whereas
several Dsuite scenario maps use `FrijCol_10` while still labelling the group n=19. This
manifest follows the canonical included set (`FrijCol_11`). It must not be described as a
verbatim replay of those Table S4 scenario files.

Exact URLs, byte sizes, hashes, citations, and the author-metadata commit are in `sources.json`.
Large VCFs and derived filtered VCFs remain ignored under `data/real/`.

Example:

```bash
python scripts/external_benchmarks.py \
  --data-root /path/to/simulation_data \
  --duck-root /path/to/extracted/dryad \
  --runner-vcf /path/to/coccineus.recode.vcf
```
