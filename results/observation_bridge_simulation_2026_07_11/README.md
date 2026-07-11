# Observation-process bridge (exploratory negative result)

This result is an exploratory simulation bridge, not a replacement paper result and not
evidence of real-world direction accuracy. It tests whether the current A/B/C direction
features survive realistic changes in sampling and ascertainment.

## Frozen run

- Source commit: `f356402e4a6a23b021886fad8fcec708e56ce17f`
- Configuration SHA-256: `f6e92005a300af41df59a0979abdafdbb726394489dd92b10d0b7e413f1f32cc`
- Simulation checkpoint SHA-256: `a8215b46fb408eb295c797ba03eeeab448bd3d476e9128060bb3c24e0b33d8d8`
- `results.json` SHA-256: `805ec7141bd83c078c347c2296e0ac3ebc0186aeb588d4680dab6420fedbb27b`
- Canonical `analysis` SHA-256: `5edd7d4cb5054722ca82400068c737541afb27486b8de168c38f9de0b03a69a2`,
  computed from sorted compact UTF-8 JSON; an independent second analysis from the
  immutable checkpoint reproduced this hash exactly.
- Bank: 24 exact rate families, 72 A/B/C parent genealogies, 20 independent 50 kb
  contigs per parent, and 10 paired observation views per parent.
- Eligibility: 720/720 feature curves usable; no parent or rate family was excluded.

Every parent retains per-copy genotypes. One fixed population sample is reused across loci
and all same-size views, the 32-copy sample is nested in the 64-copy sample, and an A/B/C
rate family is modeled only when all three parents and all ten views are valid.

## Main results

| Representation | Genealogy CV, appreciable accuracy | Exact-rate-family macro balanced accuracy | Natural bundles within cross-fit source p99 | C-grid ceiling |
|---|---:|---:|---:|---|
| Raw mean/variance/SE | 0.778 | 0.797 | 0/14 | No |
| Raw mean/variance (SE removed) | **0.819** | **0.836** | 0/14 | Yes: 1/8 genealogy and 4/8 rate folds selected C=1000 |
| Orbit composition mean/variance | 0.747 | 0.750 | 1/14 | Yes: 1/8 rate folds selected C=1000 |

The frozen exploratory thresholds fail. The hardest paired views are the 64-locus cap and
the within-population-polymorphism filter. When the within-population filter is entirely
held out from training, appreciable balanced accuracy is 0.389 for all three
representations. The orbit-composition representation improves full-fit natural RMS-z in
only 1/14 result-file bundles and has a median structured/raw bundle ratio of 1.685.

Only the exact-rate-family CV is fully blocked against the matched A/B/C construction.
The parent-grouped genealogy, leave-one-view-out, and leave-one-factor-out diagnostics
share exact rates and nuisance profiles across A/B/C: 11--14 of 24 rate families occur on
both sides of each saved fold. Those diagnostics are therefore conditional and optimistic,
not independent transfer estimates. The appreciable-rate subset contains only six exact
rate families (18 parents); the two fold seeds are repeated analyses of the same biological
bank, not additional biological replicates.

## Interpretation guardrails

- Removing SE is the only promising simulation-only signal, but its model-selection grid
  reaches the upper boundary and it establishes no natural support. It requires a wider or
  unpenalized model comparison before interpretation.
- The composition representation does not solve the observation-domain shift and should
  not be promoted.
- Nuisance-transfer claims require a new bank that blocks every A/B/C rate family together,
  saves row-level holdout predictions, and reports family-clustered uncertainty.
- Natural rows are unlabeled, correlated result/filter sensitivities. Candidate agreement
  is descriptive only; there is no external accuracy denominator.
- SNPs within each contig remain linked while PADZE's raw SE treats retained loci
  nominally. This bank improves the design but does not make that SE an effective-locus
  estimator.
- The next model should explicitly learn observation/ascertainment families and must be
  evaluated on held-out families, matched nulls, and all six ordered population edges.

The complete machine-readable fold, prediction, support, provenance, and validity ledgers
are in `results.json`. The 1.9 MB checkpoint is kept outside Git and is identified by its
hash above.
