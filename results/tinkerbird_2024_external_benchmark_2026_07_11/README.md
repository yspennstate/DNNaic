# Tinkerbird 2024 sample-disjoint transfer benchmark

This bundle evaluates the released depth-matched DNNaic direction and
appreciable-flow heads on the 452-bird Stacks ddRAD release from Sebastianelli
et al. (2024; paper DOI
[`10.1038/s41467-024-47305-5`](https://doi.org/10.1038/s41467-024-47305-5),
data DOI
[`10.6084/m9.figshare.25308376.v1`](https://doi.org/10.6084/m9.figshare.25308376.v1),
CC BY 4.0). It is a guarded transfer/specificity stress test, not external
validation and not an accuracy estimate.

## Source and sample audit

The pinned 148,766,627-byte VCF has SHA-256
`6438a889ad91b865237e6a4e5169bfbe61e9eea884696fa7fdcfef83d68c7c30`.
It contains 452 unique birds and 84,112 biallelic PASS SNP rows: 82,309 on
`SUPER_1` through `SUPER_44`, 1,157 on Z, 5 on W, 31 on S76, and 610 on
unplaced scaffolds. This reconciles the paper's 82,950 ddRAD SNP count as the
82,309 numbered-autosomal plus 641 S76/unplaced non-sex rows. All 38,018,624
genotype cells are either full diploid biallelic calls or fully missing.

The primary direction holdout uses exact author `ref_extoni` (P1, n=23), nine
Mpofu chrysoconus males selected by sex/locality/taxon (P2), and exact author
`ref_pusillus` (P3, n=8). All 40 holdout birds are individual-disjoint from the
95 females used for the published mating-direction analysis. It is not
study-independent: source MAF/missingness filters used all 452 birds, and the
candidate system label comes from the same study system. A second 14-daughter
panel deliberately reuses the ancestry-derived label and is circular. The two
gate controls are qualitative biological near-nulls, not proven historical
zero-flow populations.

Candidate class C means predominant/asymmetric pusillus ancestry into an
extoni-background contact group, never an exclusive migration edge. The paper
reports z=6.949 in the main text but z=6.714 in Supplementary Table S14. More
recent phylogenomics supports bidirectional introgression tails and core-range
effects ([`10.1093/sysbio/syaf033`](https://doi.org/10.1093/sysbio/syaf033)).

## Result

One source-ordered SNP per Stacks RAD locus is primary. The table shows its
standard and within-population-polymorphism ascertainment variants. The latter
is deliberately strong ascertainment, not a quality filter.

| panel | filter | loci | direction | candidate | gate | direction RMS z | gate RMS z | projection P1->P3 | corrected f3-like |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| disjoint holdout | standard | 8,745 | A | C | 1.000 | 22.85 | 25.94 | 0.12028 | -0.01768 |
| circular daughters | standard | 8,745 | A | C | 1.000 | 24.63 | 27.83 | 0.25006 | -0.03609 |
| geographic near-null | standard | 13,412 | A | none | 1.000 | 21.67 | 27.13 | 0.02904 | 0.00010 |
| recent-cross near-null | standard | 15,000 | B | none | 1.000 | 19.95 | 24.93 | 0.06779 | -0.00727 |
| disjoint holdout | within-pop | 1,632 | B | C | 1.000 | 24.64 | 23.76 | 0.14328 | -0.01622 |
| circular daughters | within-pop | 1,632 | B | C | 1.000 | 23.61 | 23.90 | 0.25111 | -0.02811 |
| geographic near-null | within-pop | 2,315 | A | none | 1.000 | 23.69 | 22.60 | 0.07467 | -0.00092 |
| recent-cross near-null | within-pop | 2,921 | A | none | 1.000 | 21.00 | 21.87 | 0.13190 | -0.00846 |

Across both primary and linked sensitivity scopes, all 16 panels are severe
OOD. None of the eight weak/circular candidate-C cases predicts C: all four
standard cases predict A and all four within-population cases predict B. All
eight qualitative near-null cases receive an appreciable score from
`0.999999999999162` to `1.0` and cross the nominal 0.5 threshold. The aggregate
result therefore explicitly abstains. This is simultaneous direction-transfer,
filter-stability, and gate-specificity failure; it is never “0% accuracy,” a
validated natural-data call, or a calibrated probability statement.

The model-free geometry is descriptive. For the primary standard holdout, the
projection is 0.12028 (44-chromosome block-bootstrap interval
0.10768-0.13297) and the finite-called-copy-corrected f3-like statistic is
-0.01768 (-0.01999 to -0.01520). The corresponding circular-panel values are
0.25006 (0.23553-0.26439) and -0.03609 (-0.03881 to -0.03323). These statistics
do not establish temporal direction. The correction is exact only under
independent binomial chromosome sampling; the bootstrap is fixed-panel and
does not include individual-sampling uncertainty.

## Ascertainment guardrails

- `g=16` forces all eight P3 reference birds to be completely called at every
  retained locus, while larger P1/P2 groups may retain missing calls. This can
  shift private/richness channels.
- ddRAD restriction-site sampling, global MAF >=5%, <=20% missingness over all
  452 birds, and mapping to a *P. pusillus* reference affect the same channels.
- The linked multi-SNP-per-RAD sensitivity uses capped reservoirs. It is not
  paired to the primary scope; the standard direction scopes share only 4,218
  sites.
- Direction, geographic, and recent-cross families use separate locus draws.
  Their primary all-three intersections are 7,404 standard and 893
  within-population loci, so cross-family gate contrasts remain qualitative.
- The diagnostic projection selects loci using the observed P1/P3 endpoint
  frequencies from these same samples and remains descriptive.

## Reproduction

Run from clean code commit
`95e437d2bb3f1515e004d964c8458f3bd50083e3`, with all BLAS/OpenMP thread
variables fixed to one:

```bash
python scripts/tinkerbird_2024_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --source-vcf /path/to/southern_africa_biallelic_snps_minDP4_MaxMiss20_MAF5.vcf.gz \
  --female-metadata /path/to/MS_SouthernAfrica_ddRADS_95SympF_14Mar23.xlsx \
  --master-metadata /path/to/MS_Tinkerbird138_Master_06Oct23.xlsx \
  --supplement /path/to/41467_2024_47305_MOESM1_ESM.pdf \
  --cache-dir /path/to/derived \
  --result-dir results/tinkerbird_2024_external_benchmark_2026_07_11
```

The clean run records `dirty_at_run=false`. `results.json` is 584,071 bytes
with SHA-256
`1c9573456f8a7c05e61d93a574d1c578e93f8f2cfc1f2e7e731a79563dded7dd`.
The deterministic numbered-autosome VCF hashes to
`20eac2cfca28e96221bf606fab61a6bdea30a76546572cacb781efa6688af9f4`.
The 23,395-row source-first RAD-thinned VCF hashes to
`489174990961616b3c79ee2eebab29d44bd0e13c5b7c720c321bd82ddc16923c`;
its ordered semantic-key digest is
`a78ee8f7f57e81745d7f745d3c600435ed800c1611e5bf9a1fe57468253200ef`.
An independent detached-worktree run was identical after normalizing only
repository, cache, and result paths. Both canonical JSON objects are 380,765
bytes and hash to
`16322ab95cfa016402bbd5bbe854c103f6f6d3c0a95af06e8bd1cb4d31f958d7`.
