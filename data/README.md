# Data

This directory holds the data *methods* and *pointers*, not the bulk data. See
[DATA_SOURCES.md](DATA_SOURCES.md) for every dataset's public location and DOI.

## What the model consumes

DNNaic reads a 28-dimensional feature vector per rarefaction depth `g`, built by PADZE from a VCF
plus a population map (`sample<TAB>population`, three populations):

```
[ g,
  alpha_1(mean,var,se), alpha_2, alpha_3,     # allelic richness, per population
  pi_1, pi_2, pi_3,                            # private allelic richness
  pihat_12, pihat_13, pihat_23 ]               # pairwise-private richness
```

Populations map to the caterpillar tree `((P1,P2),P3)`: P1 and P2 are the sister populations and
P3 is the more divergent one. The mapping is explicit (never inferred from alphabetical order).
`dnnaic.build_matrix(vcf, popmap, pop_order=[P1, P2, P3])` returns this matrix.

## Simulation arrays

The coalescent study ships as per-row arrays (`X.npy`, `direction.npy`, `magnitude.npy`,
`groups.npy`) under `simulation_data/`, fetched from Zenodo (see DATA_SOURCES.md) or regenerated
with `scripts/simulate_demography.py` followed by PADZE feature extraction. Point the loaders at
the directory with the `DNNAIC_DATA` environment variable:

```
export DNNAIC_DATA=/path/to/simulation_data
python scripts/direction_detection.py --dataset regen_full
```

`groups.npy` is the true simulation-replicate identifier. Every split groups by it, so no replicate
is shared between train and test — the leakage-free protocol the paper proves is the correct one.

## Real data

The 1000 Genomes and archaic (Neanderthal) analyses stream only the required genomic windows over
HTTP range requests using each file's `.tbi` index, so no full-genome download is needed. The
Heliconius analysis reads the whole-genome `.geno` file from the tutorial repository. Exact
locations are in DATA_SOURCES.md.
