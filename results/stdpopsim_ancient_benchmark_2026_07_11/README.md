# Official stdpopsim ancient focal-event transfer benchmark

This directory records a known-truth synthetic transfer stress test using five
focal events from two official `stdpopsim` human demographic models. It is not
natural-data accuracy, an ancient-DNA sample-size reproduction, or five
independent demographic systems. The frozen DNNaic heads were not refit or
tuned on these simulations.

## Frozen run

- Catalog/runtime: `stdpopsim==0.3.0`, `msprime==1.4.2`, `tskit==1.0.3`,
  `padze==0.1.0`, NumPy 2.5.1, and scikit-learn 1.9.0.
- Bank: five focal positive/control panels, 30 independently seeded pairs per
  panel, 300 jobs total. Each job uses one 1 Mb contig, 100 diploid individuals
  per population at the catalog sampling times, and PADZE `g=2..199`.
- Source commit: `5278103b40ac0124556a22035a4ecd3e6d55d50f`.
- Configuration SHA-256:
  `18cce351a81f73918cd0d87e96fc81ce95bf32e11b1969cd69f0b4f4566f3c7d`.
- `results.json`: 837,370 bytes; SHA-256
  `6b8d1654385b590fe5718811918c49bc02ac775330225f0706ea3e3660347799`.
- External feature checkpoint: 6,147,622 bytes; SHA-256
  `1d5732407165d1a27dba384ac768184ebc335c3e22a68b8c430c734ddc26260b`
  (`300 x 198 x 28` float32 curves).
- Ordered 300-curve ledger SHA-256:
  `74097d3a2319a8b8e46299c2ee39e1b14cc0aa693ed13313fdb26fb4d8fda2d8`.

The run was tracked-clean at both source snapshots. Its broader Git status was
dirty only because an unrelated untracked scratch directory existed; the
machine-readable provenance must not be summarized as a completely clean
worktree.

## Direction result

The frozen primary `raw_all` head obtains B/C balanced accuracy 0.666667:
B recall is 60/60 and C recall is 30/90. Equal-panel macro accuracy is 0.60,
which only ties the 0.60 always-C panel baseline because three of the five
panels have C truth. A constant B or C rule has balanced accuracy 0.50.

Primary correct calls by focal panel are:

- WHG to NEO, truth B: 30/30;
- CHG to YAM, truth C: 30/30;
- NEO to Bronze, truth B: 30/30;
- Neanderthal to Han, truth C: 0/30 (all called B);
- WHG to Sardinian, truth C: 0/30 (all called A).

The target-blind mean/variance-only sensitivity obtains balanced accuracy
0.833333 and equal-panel macro accuracy 0.80 by recovering the Neanderthal-to-
Han panel; it still misses all WHG-to-Sardinian rows. The orbit-composition
sensitivity has the same 0.666667 balanced accuracy and 0.60 macro accuracy as
the primary head. These are secondary representations and do not replace the
frozen primary result.

Domain shift remains severe. For `raw_all`, scaler RMS-z has median 5.716,
p95 17.345, and p95 maximum absolute z 53.597; the Neanderthal-to-Han panel
alone has median RMS-z 16.554. Wilson intervals in `results.json` summarize
Monte Carlo variation inside fixed panels, not across-demography uncertainty.

## Focal-present versus focal-absent score diagnostic

The frozen appreciable-migration score has pooled ROC AUC 0.684511 and
equal-panel macro AUC 0.838111. Per-panel AUCs in the order above are 1.000,
1.000, 0.998889, 0.691667, and 0.500. The pooled statistic is confounded by
panel baselines: 80% of its positive/negative comparisons are cross-panel, and
the cross-panel-only AUC is 0.646111.

At threshold 0.5, all 150 positives are called positive and none of the 150
focal-absent controls is called negative. These controls remove only the focal
event while retaining the rest of each catalog history; they are not global
no-admixture nulls. The score is therefore a within-panel focal-ranking
diagnostic here, not evidence of calibrated sensitivity or specificity. Its
canonical target is sustained migration of at least 2.5e-4, whereas these
panels contain 25--50% formation ancestries or approximately 3% ancient
pulses.

## Independent audit and interpretation

A separate read-only audit verified all 300 manifest rows, unique seeds,
positive/control families, curve hashes, classifier probabilities, gate
metrics, package versions, and source snapshots. It independently fetched the
official tagged HomSap model source and matched SHA-256
`ca1adb03f251b7fc293323ef9fe4e77ec9e705a9ad38c21382946bed7c791e1c`.
For all ten retained tree sequences, it rechecked file hashes, samples,
catalog times, focal-only demographic differences, allele-count ledgers, and
regenerated PADZE float64/float32 curve hashes exactly.

The five panels are nested in only two catalog models and share large parts of
their demographic histories. Thirty simulations per panel increase Monte
Carlo precision but do not create independent biological systems. There is no
class-A truth panel, positive/control genealogies use independent seeds, and
ancient sampling plus episodic large ancestry fractions are far outside the
continuous-migration training target. The defensible conclusion is therefore
mixed transfer under a synthetic known-focal-event bank, with two complete
class-C failures—not a natural or literature-benchmark accuracy estimate.

`results.json` contains the full source, configuration, model, prediction,
simulation, and audit ledgers. The feature checkpoint and ten retained raw
tree sequences have been hash-verified but are not committed here; this compact
bundle remains inspection-only until those artifacts are deposited with an
immutable inventory and DOI.
