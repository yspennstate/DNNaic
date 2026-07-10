# External dataset diagnostics (2026-07-10)

These are exploratory simulation-to-data transfer diagnostics, **not classifier validation**.
The analysis verifies source hashes, fixes author-derived sample manifests before scoring,
requires biallelic polymorphic SNPs with at least 16 called copies per population, and fits the
simulation reference on the external panels' identical `g=2..16` depth grid.

## Results

Scores are uncalibrated OOD model scores, not probabilities or posteriors. Ratios use depths
`g>=8`; the normalized ratio uses the prespecified alpha denominator clip `1e-12`.

| Panel | Loci | Published expectation | Raw ratio | Normalized ratio | Call | Leading score | RMS z | Max abs z |
|---|---:|---|---:|---:|---|---:|---:|---:|
| Andean duck beta-globin | 2,298 | C: ST-high -> YBP-high | 0.856 | 2.135 | B | B=1.000 | 26.10 | 108.28 |
| Andean duck alpha-globin | 499 | negative control | 1.831 | 1.667 | A | A=1.000 | 47.00 | 108.33 |
| Runner bean, CDMX donor | 12,284 | C: wild -> Cult-TMVB | 4.138 | 2.304 | C | C=1.000 | 13.77 | 37.08 |
| Runner bean, Tepoz donor | 12,284 | C: wild -> Cult-TMVB | 3.964 | 2.209 | C | C=0.9618 | 14.50 | 43.61 |
| Runner bean, published null | 12,284 | abstain / no direction | 0.264 | 0.459 | B | B=1.000 | 18.26 | 71.86 |

The three bean panels use the identical ordered locus set, SHA-256
`a9b48df5d9f4d9810d90239187146898e34e3a4a75989fb42795a739758a0b94`. It contains all
12,284 loci in the six-population callable-site intersection that are polymorphic in each of the
CDMX-positive, Tepoz-positive, and published-null trios. No locus cap is reached.

## Interpretation

- Duck transfer fails: the published positive calls B rather than expected C. The alpha negative
  calls A, but the head has no no-event class, and both small panels are extremely far from the
  simulation feature distribution.
- Runner bean is a useful positive/control stress test. Both independently published donor
  contrasts call the expected class C, while the published null does not call C, on exactly the
  same loci. The raw and normalized sharing ratios also separate both positives from the null.
- This is still not external validation. Every panel has severe feature shift, the head has no
  no-event class, and the bean class-C direction depends on the paper's wild-to-crop demographic
  interpretation because D/f4 excess sharing alone is not directional. Target-demography training
  and independent labelled controls remain necessary for a biological direction claim.

The canonical runner-bean metadata includes `FrijCol_11` and excludes `FrijCol_10`. Several
released Dsuite scenario maps make the opposite one-sample substitution. These runs use the
canonical included set and are comparisons to, not verbatim replays of, Table S4.

Machine-readable results: `results.json`, SHA-256
`a475872d3d2e6126cc2ded05236775e0b932b7625876a1fc67db874de83f8ff5`.
