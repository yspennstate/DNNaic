# Independent stdpopsim catalog benchmark

This frozen external simulation bank evaluates two demographic systems that were
not used to construct DNNaic's canonical training bank. It contains 30 focal-edge
positive realizations and 30 focal-absent controls for each system (120 records,
60 declared positive/control families). The full bank ran on the Azure worker at
nice 15 with one numerical thread and no GPU.

The primary full-feature direction result is **0.80 balanced accuracy** across
the two declared positive classes: 30/30 class-C recalls for the Ashkenazi
`CEU -> WAJ` pulse panel and 18/30 class-B recalls for the CanFam `ISW -> GLJ`
panel. The frozen appreciable-migration gate separates focal-positive from
focal-absent rows with equal-panel macro ROC AUC 0.9778 (human 0.9556, dog 1.0).
The gate result is score discrimination only, not threshold accuracy or effect
calibration.

This is the corrected v2 bundle. In every committed semantic layer--the
configuration job manifest, checkpoint metadata, simulation ledger, and
prediction ledger--focal-absent controls now have `direction_truth: null`.
Their counterfactual B/C panel candidate remains available only through the
separate `panel_candidate_direction` field. The 60 controls are excluded from
direction accuracy, while all 60 positives retain truth equal to their declared
panel candidate.

## Design and accounting

- Ashkenazi panel: stdpopsim `HomSap` catalog background, candidate class C,
  with the focal `CEU -> WAJ` pulse present or ablated.
- CanFam panel: stdpopsim `CanFam` catalog background, candidate class B, with
  focal `ISW -> GLJ` flow present or ablated while the model's deeper jackal
  background is retained in both conditions.
- Each record is one independently seeded 1 Mb ancestry/mutation realization.
  The 60 positive rows are eligible for declared-direction recall; the 60
  focal-absent controls have no direction truth and are excluded from direction
  accuracy.
- This adds two demographic systems, not 60 independent systems. Repeats,
  positive/control conditions, representations, and gate scores are correlated
  analytic views and must not be counted as additional external models.

## Sensitivity and transfer limits

The headline is representation-dependent. `raw_mean_variance` reaches 0.50
balanced accuracy (the constant-class panel baseline), while
`orbit_composition_mean_variance` reaches 0.05. The primary `raw_all` rows are
also strongly shifted from canonical training: median RMS-z is 7.72 for the
human panel and 16.14 for CanFam, with CanFam 95th-percentile max-absolute-z
70.94. The human focal parameter is a pulse proportion, whereas the frozen gate
was trained on sustained migration rates. The CanFam focal rate is far outside
the training range. Consequently, these are known-model transfer diagnostics,
not evidence that DNNaic causally isolated the focal edge in arbitrary catalog
models or calibrated a biological effect size.

## Frozen v2 provenance

- Source commit: `c0584877da9def78ed78669e187d1c7737f824de`
- Runner SHA-256: `91644fc0b26514a72b2b94a00f94788be2490ff550f14881d9ba8f8ae9e2f90e`
- Configuration SHA-256: `1c1a48dfa4d73e4360dafb74b2b504ff40b4689374a64450562a335b424452dc`
- `results.json`: 485,914 bytes; SHA-256
  `4ed901917a15684f384da481ea9ef54f498a6efadab35285eda780b00788b74e`
- Checkpoint: 2,473,685 bytes; SHA-256
  `b97439ee52f21bd2fe1d5836973c4c88aa142fd8eeb6641bbb9490d3c10c8f17`;
  120 curves with shape `120 x 198 x 28`. The checkpoint and retained audit
  trees are not committed, so this public bundle is auditable but not a
  self-contained bit-for-bit regeneration package.
- Ordered record-curve ledger SHA-256:
  `d247e178938100407235736e3a58c14063c62aa66fe0e1a08657590a1b289a4f`.
- Both source snapshots report zero tracked diff bytes and the empty tracked-
  diff SHA-256; the run directory itself was untracked output.
- Exact runtime: Python 3.12.3, NumPy 2.5.1, scikit-learn 1.9.0, PADZE 0.1.0,
  stdpopsim 0.3.0, msprime 1.4.2, and tskit 1.0.3.

The full runner is Azure-only. With the canonical training arrays and an
approved healthy compute state available, run:

```bash
python scripts/stdpopsim_independent_catalog_benchmark.py \
  --canonical-root /path/to/canonical_regen_full \
  --cache-dir /path/to/independent_catalog_cache \
  --result-dir results/stdpopsim_independent_catalog_benchmark_2026_07_11 \
  --compute-state /var/local/compute_health.json \
  --compute-target azure
```

The recorded run used the explicit owner-authorized stopped-trading and closing-
session overrides; those flags should be supplied only under the corresponding
owner authorization. The complete argv, seed ledger, model definitions, package
versions, per-record curve hashes, predictions, and guardrails are in
`results.json`.
