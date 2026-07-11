# Harpagifer external transfer diagnostic (2026-07-11)

This bundle records a guarded transfer stress test, not an accuracy benchmark.
The source paper's dominant same-SNP BayesAss estimate maps to candidate class
A (north-Patagonia P1 -> south-Patagonia P2), but the history is bidirectional
and multi-edge. The four rows below are analytic sensitivities of one biological
comparison. None is independent validation or ground truth.

## Frozen result

- Code commit: `a23796e60d7aac5b12c55e767d5844f2d4541c5c`
- Result bytes: `163270`
- Result SHA-256: `6ec8dcd473da847605c6d1500d4c9d4bdb0b3b7f6d50422929c04710e3c329e3`
- Source VCF: 1,517,411 bytes, SHA-256
  `7dbc3686e4a24ef36c2a358b55d3d95ff5f2f8d2340087099119335c9b0474a8`
- Thread variables: OMP, OpenBLAS, MKL, and NumExpr all fixed to `1`
- Git state at run: clean

An independent detached-worktree run at the same commit produced raw SHA-256
`c1b2da8f286096d5ea67f441439de41c11c99e2d5fe59fbbdcb2076beb152e64`.
After replacing only repository, derived-cache, and result path prefixes, its
JSON was exactly equal to this artifact. Raw bytes differ because audited paths
are intentionally retained. The Dryad ZIP wrapper is not pinned: repeated
official downloads produced different wrapper timestamps/hashes but the same
byte-identical pinned VCF.

## Source and grouping guardrails

The VCF has 118 samples and 2,993 unique biallelic PASS SNPs. Its 353,174
genotype cells include 34,619 fully missing calls (9.8023%) and no partial or
invalid calls. Every `CHROM` is the artificial value `0`, and the reference-free
Tassel header says the biological reference allele is unknown. REF is therefore
not ancestral, and physical linkage cannot be reconstructed.

No author individual crosswalk was released. The benchmark reconstructs
positive-site blocks from VCF column order plus published contiguous site order:

- P1: TEM--IC3, 50 samples (north Patagonia)
- P2: PY--PW, 43 samples (south Patagonia)
- P3: HP, 25 samples (the paper's Falklands/Malvinas M1 cluster)

Three source totals disagree: Supplementary Table S1 gives 117/TEM=2; the VCF
has 118/TEM-like=3; main-text Table 3 totals 128. The nine-row block record is
also not a complete collection-site roster. Accordingly, the grouping status is
`reconstructed_from_VCF_column_order_using_published_contiguous_site_order`,
never “author crosswalk.”

## Results

| Sensitivity | Loci | Raw head | Raw gate | Max RMS-z | P2 projection | corrected f3-like |
|---|---:|---:|---:|---:|---:|---:|
| all samples, standard | 2,993 | A | 1.0 | 23.784 | 0.227138 | 0.000288 |
| all samples, within-pop polymorphism | 2,977 | A | 1.0 | 23.883 | 0.229907 | 0.000350 |
| <=25% sample missingness, standard | 2,993 | A | 1.0 | 23.854 | 0.227443 | 0.000422 |
| <=25% sample missingness, within-pop polymorphism | 2,977 | A | 1.0 | 23.953 | 0.230722 | 0.000465 |

All four raw heads agree with candidate A and all raw gates saturate at 1.0.
However, all four exceed the prespecified heuristic severe-OOD rule by a wide
margin and therefore have status `abstain_severe_OOD`. These raw outputs are not
posterior probabilities, event validation, or correct predictions. Accuracy is
`null`; independent validations are zero.

The <=25% missingness sensitivity removes ten of 50 P1 samples, one of 43 P2,
and none of 25 P3, so it is strongly group-reweighting rather than an improved
panel. Both sample scopes nevertheless retain the same exact ordered locus sets:
2,993 standard loci and 2,977 within-population-polymorphic loci.

At the prespecified `|p3-p1| >= 0.95` frequency threshold, every run has zero
diagnostic loci (maximum 0.781 for full samples and 0.751 after the missingness
filter), consistent with the source paper's no-diagnostic-allele statement. The
threshold was not lowered after seeing the data. Every finite-copy-corrected
f3-like estimate is positive and every naive IID-locus interval crosses zero,
so there is no negative-f3 certificate. IID resampling is descriptive only: it
conditions on fixed samples, reconstructed grouping, released ascertainment,
and the same-data label, while ignoring linkage and grouping/label uncertainty.

## Biological interpretation

The source paper reports P1->P2 18.8%, reciprocal P2->P1 2.03%, P1->M1 1.15%,
and P2->M1 2.54%. BayesAss used the same SNPs with 10,000 iterations and 10%
burn-in; the percentages are neither pulse ancestry fractions nor per-generation
msprime migration rates. The paper also concludes that the nominal taxa form
one evolutionary unit. Candidate A is therefore a literature-dominant
sensitivity forced through a one-edge head, not biological truth.

Bernal-Duran et al. (2024; DOI `10.1111/mec.17360`) provide a separate
H. antarcticus system with 143 fish, 20,778 neutral SNPs, and independent ROMS
particle-connectivity matrices. It does not validate this label, but it is a
stronger independent follow-up dataset.

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python scripts/harpagifer_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --source-vcf /path/to/Hbi_Hpal_118_2993_6May19_GEO.vcf \
  --cache-dir /path/to/empty-derived-cache \
  --result-dir results/harpagifer_external_benchmark_2026_07_11
```
