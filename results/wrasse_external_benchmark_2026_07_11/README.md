# Corkwing-wrasse candidate-direction transfer benchmark

This bundle evaluates the released depth-matched DNNaic heads on the balanced
six-population corkwing-wrasse 2bRAD release from Faust et al. (2018; paper DOI
[`10.1098/rsos.171752`](https://doi.org/10.1098/rsos.171752), data DOI
[`10.5061/dryad.tv553`](https://doi.org/10.5061/dryad.tv553), CC0). It is a
candidate-direction and gate-specificity stress test, not external validation
or an accuracy estimate.

## Source and label audit

The pinned 18,733,292-byte VCF has SHA-256
`c05741f03ecdb2403f173cf249eb910281632f65bccad27aaaaaf848ffb2e21a`.
It contains 240 fish, exactly 40 in each of Flatanger (`FKH`), Austevoll
(`SMAU`), Stavanger (`SMID`), Kristiansand (`SMTF`), Stromstad (`SMST`), and
Kungsbacka (`SMKB`), at 4,372 biallelic 2bRAD loci. The pinned metadata
mechanically reconstructs every VCF ID, including all 48 documented `a`
technical-replicate suffixes.

The Genepop release retains 4,357 loci after 15 author HWE exclusions. The
NewHybrids file names 200 further, disjoint loci used to generate same-data
labels. The runner verifies all 240 body rows and all 48,000 NewHybrids
genotype cells exactly against the VCF in metadata order. The primary source
removes both sets before filtering, leaving 4,157 loci; its deterministic VCF
hashes to
`885baa9a027015b5e6869fa7fc3e02ab8dddd01909b0b9a9e99df160e5337351`.
An all-released-locus scope is retained only as circular sensitivity.

## Biological mapping and limits

The candidate panels map Austevoll to P1, all 40 Flatanger fish to P2, and one
of the three southern references to P3. Documented anthropogenic transport of
southern fish into Flatanger therefore gives candidate class C, P3 to P2. The
paper reports FKH48a/FKH50a as clear southern-genotype escapees, FKH67 as an
F1, and twelve potential western backcrosses. All 40 Flatanger fish are kept;
there is no hybrid-only individual selection.

This is one mixed-history recipient cohort, repeated across three donor
references, two locus scopes, and two filters—not twelve independent positive
events. Descriptive role-changed contrasts instead use Stavanger as P1 and
Austevoll as P2 on the same loci. They change both western roles, share fish,
and are neither matched controls nor no-flow populations. The 2018 article
also reports potential backcrosses in Stavanger and Kristiansand.

Current evidence makes a one-edge interpretation especially unsafe:

- Mattingsdal et al. (2020; [`10.1111/mec.15310`](https://doi.org/10.1111/mec.15310))
  infer ongoing bidirectional contact with older/background asymmetry mainly
  west to south.
- A 1,766-fish survey (2021;
  [`10.1111/eva.13220`](https://doi.org/10.1111/eva.13220)) reports six
  high-probability southern-origin Flatanger fish and 70 potential hybrids
  there, concentrated at the northern edge and reaching about 20% locally.
- A 2026 mesocosm study
  ([`10.1111/eva.70214`](https://doi.org/10.1111/eva.70214)) finds strong
  selective winter mortality in hybrid offspring and suggests potential
  assortative mating.

## Result

All six panels share one exact ordered locus set within each scope/filter. The
table shows the primary label-locus-excluded scope.

| panel | filter | loci | raw direction | candidate | raw gate | direction RMS z | gate RMS z | projection P1->P3 | corrected f3-like |
|---|---|---:|---|---|---:|---:|---:|---:|---:|
| candidate SMTF | standard | 4,097 | A | C | 1.000 | 15.71 | 16.96 | 0.12389 | 0.00508 |
| candidate SMST | standard | 4,097 | A | C | 1.000 | 15.75 | 17.08 | 0.12137 | 0.00498 |
| candidate SMKB | standard | 4,097 | A | C | 1.000 | 15.75 | 17.08 | 0.11080 | 0.00525 |
| comparator SMTF | standard | 4,097 | A | none | 1.000 | 14.37 | 15.14 | 0.00742 | 0.00339 |
| comparator SMST | standard | 4,097 | A | none | 1.000 | 14.37 | 15.22 | 0.00323 | 0.00350 |
| comparator SMKB | standard | 4,097 | A | none | 1.000 | 14.38 | 15.21 | 0.00227 | 0.00352 |
| candidate SMTF | within-pop | 2,962 | A | C | 1.000 | 19.94 | 21.24 | 0.15814 | 0.00406 |
| candidate SMST | within-pop | 2,962 | A | C | 1.000 | 19.88 | 21.24 | 0.15578 | 0.00389 |
| candidate SMKB | within-pop | 2,962 | A | C | 1.000 | 19.86 | 21.24 | 0.14196 | 0.00429 |
| comparator SMTF | within-pop | 2,962 | A | none | 1.000 | 18.89 | 19.87 | 0.00454 | 0.00409 |
| comparator SMST | within-pop | 2,962 | A | none | 1.000 | 18.82 | 19.86 | 0.00188 | 0.00418 |
| comparator SMKB | within-pop | 2,962 | A | none | 1.000 | 18.79 | 19.83 | 0.00029 | 0.00423 |

All 24 panels are severe OOD and explicitly abstain. The raw direction head
emits A for every panel, so none of the 12 repeated candidate-C sensitivities
returns C. Raw A here means Austevoll P1 to Flatanger P2 and can reflect the
dominant western/range-expansion background in a cohort with superimposed
recent southern introgression; it is not evidence that the documented event
ran backward. A mutually exclusive single-edge head cannot isolate both.

Every raw gate is numerically 1.0. Consequently all 12 same-locus role-changed
contrasts are probability-ceiling ties with a raw score delta of zero. This is
gate saturation, not equality, causal matching, or a successful comparison.
The conclusion is a direction-transfer and gate-specificity failure under
severe OOD, never “0% accuracy” or a validated biological call.

The model-free projections consistently separate the candidate geometry
(0.111-0.158) from the role-changed contrasts (0.0003-0.0074) in the primary
scope. All corrected f3-like values remain positive, so there is no formal
negative-f3 certificate. The source has no physical chromosomes or linkage
map: each `CHROM` is one 2bRAD tag and `POS` is only within-tag position.
Reported bootstrap intervals are naive IID-locus sensitivities, not calibrated
chromosome-block confidence intervals.

## Reproduction

Run from clean code commit
`07fbb8858a00d89521b79aef6334e52cb1cc5b3c`, with all BLAS/OpenMP thread
variables fixed to one:

```bash
python scripts/wrasse_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --source-vcf /path/to/west.filt.maf0.01.recode.vcf \
  --metadata /path/to/Sampleinfo_metadata.txt \
  --hwe-genepop /path/to/west_genepop4357ID.txt \
  --newhybrid /path/to/newhybrid200SNPs.dat \
  --archive /path/to/dryad_tv553_v1.zip \
  --cache-dir /path/to/derived \
  --result-dir results/wrasse_external_benchmark_2026_07_11
```

The clean run records `dirty_at_run=false`. `results.json` is 911,019 bytes
with SHA-256
`89c7d14d20098297be9a2f1b719d1148ed9c6b2868ccfc034dda24d72a1be16b`.
The exact shared-locus counts/hashes are:

- primary standard: 4,097,
  `5459b30a59326669dc35fe811fda161f6596e1d97aeda5dce2b9ba412cafa1aa`;
- primary within-population: 2,962,
  `01046aa4a57decc44904ae4a253e2c8df63f838c6a625979efb28ebfbe1cbf4d`;
- all-locus standard: 4,311,
  `da256f6855efce965bec0ef285490d4468f2566a8e941e2b568663a3485bb6d7`;
- all-locus within-population: 3,103,
  `efaae46e4380dc9a49f91bfaf24fed7d954b91fac168933aaaa76cff5c7caeb5`.

An independent detached-worktree run was identical after normalizing only
repository, cache, and result paths. Both canonical JSON objects are 588,017
bytes and hash to
`72419544d82c92304200178b5639f77d4d9277800c11fee28741485832cf4453`.
