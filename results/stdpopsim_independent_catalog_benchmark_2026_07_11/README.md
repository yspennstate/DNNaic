# Independent stdpopsim catalog benchmark

> **Provisional v1 artifact:** the prediction ledger correctly excludes controls
> from direction accuracy, but the lower-level v1 job/simulation ledgers retain
> the panel's counterfactual candidate class under the ambiguous name
> `direction_truth`. Do not cite or merge this v1 bundle. A v2 rerun separates
> `panel_candidate_direction` from condition-specific truth and sets control
> truth to null throughout.

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

## Frozen provenance

- Source commit: `2dd19f1fd159e0cbedcc638cd18b651d3607e6c4`
- Runner SHA-256: `a07525f02f49d0345d8aee0ef5f11aae9035b7351af2a8b898eb7455e9e5b6d2`
- Configuration SHA-256: `ad17b6416a9b44b0b16e010e88d204648814b971501a2b6de92d8add6ffd3a39`
- `results.json`: 470,531 bytes; SHA-256
  `99a2a0cf3859f204379d1d156dd915b568bc82fd9bdccf54ff9e4b7f9a0b3d80`
- Checkpoint: 2,473,066 bytes; SHA-256
  `d22aa641d1aaae5b7b2b2f2d5089afca8c1cbba4c4938e686496f3c018b0257a`;
  120 curves with shape `120 x 198 x 28`. The checkpoint and retained audit
  trees are not committed, so this public bundle is auditable but not a
  self-contained bit-for-bit regeneration package.
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
