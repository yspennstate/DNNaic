# Further external benchmarks (2026-07-11)

This bundle adds one exact sampled-trio null and one paired positive/negative
dataset. It is an out-of-distribution transfer audit, not classifier
validation. None of these panels supplies gold-standard donor/recipient truth,
so the three-way A/B/C outputs below are descriptive only.

## Results

| Panel/filter | Classical expectation | Loci | Gate score | Direction score call | Direction RMS z | Gate RMS z |
|---|---|---:|---:|---|---:|---:|
| Scrub-jay null, standard | author D=0.00619, Z=0.205, null | 8,635 | 0.8803 | A | 12.65 | 11.97 |
| Scrub-jay null, within-pop polymorphic | same source null; much narrower filter | 627 | 1.0000 | B | 25.12 | 27.48 |
| Malawi pelagic positive, standard shared | D=0.08553, Z=4.195 | 15,000 of 197,539 | 3.84e-8 | B | 21.47 | 20.69 |
| Malawi deep-benthic negative, standard shared | D=-0.01207, Z=-0.586, null | 15,000 of 197,539 | 1.98e-12 | A | 20.60 | 19.73 |
| Malawi pelagic, within-pop polymorphic | D=0.01846, Z=1.657 | 7,674 | 1.0000 | B | 22.38 | 23.94 |
| Malawi deep-benthic, within-pop polymorphic | D=-0.01644, Z=-1.401, null | 7,674 | 1.0000 | B | 22.30 | 23.81 |

The standard scrub-jay panel is a false-positive gate call. The standard
Malawi filter correctly abstains on its matched negative but also misses the
significant pelagic positive. Under the deliberately stricter filter, both
Malawi panels saturate the gate and receive the same direction call, while the
classical positive itself is no longer significant. Thus the learned gate is
neither a stable D surrogate nor a reliable natural-data abstention rule.

All six panels are far outside the standardized simulation support. The
smallest direction-head RMS z is 12.65; the largest gate-head RMS z is 27.48.
The softmax and gate values are therefore uncalibrated scores, not probabilities
or posteriors.

## Filter and comparison contract

The primary contract requires a biallelic PASS/dot SNP, at least 16 called gene
copies in each population, and both alleles somewhere in every declared trio.
The robustness filter additionally requires both alleles within every
population. That stricter rule discards fixed between-population differences
that remain valid rarefaction input and should not replace the primary result.

Both Malawi trios are subsets of one four-group callable-site intersection and
use the same ordered locus reservoir. The raw intersection contains 197,539
eligible loci under the standard contract and 7,674 under the strict contract.
This makes positive/negative differences independent of locus selection. The
frequency-based D values use `Nbrichardi` as the outgroup and a delete-one 1-Mb
block jackknife; the full formula, counts, and block totals are in
`results.json`.

## Reproduction

The result was generated from clean code commit
`e9579116d2e4ce1bb3f26ec4f0057dae4e28a984`:

```text
python scripts/further_external_benchmarks.py \
  --data-root /path/to/simulation_data \
  --scrub-vcf /path/to/unzipped.filtered.vcf.gz \
  --malawi-vcf /path/to/Malinsky_et_al_2018_LakeMalawiCichlids_scaffold_0.vcf.gz
```

The source files are fixed by size and SHA-256 in
`data/external_benchmarks/sources.json`; the exact population manifests are
committed beside it. Large source and derived VCFs remain outside Git.
`results.json` is 212,012 bytes with SHA-256
`a5f6c62e0692caddabdf565970258f1e1f9a4a62768dfd203fd0afa7a610ddb5`.
An independent detached-worktree run at the same commit used a separate cache
and output directory. After normalizing only the absolute repository and cache
prefixes, the complete JSON objects were identical (canonical normalized
SHA-256 `02d92551482999f0ebd60b42716c7751b2eee56b5c86a9e8f711d37cb370902d`).
