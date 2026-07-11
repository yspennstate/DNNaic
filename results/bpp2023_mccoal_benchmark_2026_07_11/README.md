# BPP 4.8.7 current-release sensitivity benchmark

This directory records a known-truth synthetic transfer experiment using the
current official BPP release, not a natural-data accuracy estimate and not a
numerical reproduction of Ji et al.'s published detection-power experiment.
The frozen DNNaic head was not refit on any BPP output.

## Result

- `results.json`: 644,994 bytes; SHA-256
  `213cf7c26cac74b1cc77d331ed566acd716d032dde4c515108c48a643547fa71`
- Feature checkpoint: 1,873,051 bytes; SHA-256
  `0fd638131bdac604742adb124654203e2d4cd74e6dcf1335cf44db5fbb9d0c61`
- Configuration SHA-256:
  `482d5807cf1ef7dc413ae7508b83e1c631f3107f3948b33781b9bf6e6a8900dc`
- Ordered record/curve/count-file ledger SHA-256:
  `1be8eaf89698883e8de1b8dbe227289af094efee739c7560a4817228cbcef5cb`
- Runner source commit: `d3f417b823a728f436176ef077d92de1aa7d2d1f`

The primary `raw_all` representation has B/C balanced accuracy 0.000: all 30
B and all 30 C positives were called A. The mean/variance representation is
also 0.000. The orbit-composition representation is 0.500 because it calls all
positive rows B. The frozen gate has sensitivity 27/60 = 0.45, reconstructed-D
specificity 20/30 = 0.666667, and AUC 0.627222.

Gate sensitivity / D specificity / AUC by demographic scale are:

- 0.5x: 0 / 1 / 0.800
- 1x: 0.35 / 1 / 0.860
- 2x: 1 / 0 / 0.805

These are summaries across a fixed heterogeneous grid, not population-sampling
confidence intervals.

This is severe out-of-distribution transfer, not in-domain accuracy. Relative
to the frozen canonical scaler, `raw_all` has RMS-z median 13.5672, RMS-z p95
15.1394, and p95 maximum-|z| 44.8046. The corresponding diagnostics are
13.9437 / 14.6155 / 44.8046 for `raw_mean_variance`, and
8.6955 / 9.6997 / 27.8385 for the orbit representation. The negative direction
result therefore demonstrates OOD failure under this derivative BPP bank; it
does not estimate the classifier's in-domain error.

## Release-invariance audit

An independent audit compared this BPP 4.8.7 run with the pinned BPP 4.6.1
run. All 90 semantic count ledgers, count summaries, float32 and float64 curve
hashes, checkpoint curve bytes, predictions, fitted frozen models, phi/scale
summaries, and gate metrics are exactly identical. The three retained raw
alignments have zero normalized mismatches after applying only BPP's required
sample-name syntax change from `key^POP` in 4.6.1 to `POP^key` in 4.8.7.
Thus the negative direction result is release-robust and is not a parser-format
artifact.

## Pinned upstream sources

- [BPP latest release](https://github.com/bpp/bpp/releases/latest), verified as
  v4.8.7 on 2026-07-11.
- [Exact BPP 4.8.7 Linux archive](https://github.com/bpp/bpp/releases/download/v4.8.7/bpp-4.8.7-linux-x86_64.tar.gz):
  3,447,450 bytes; SHA-256
  `577306b8dafa80114d09e61f460633dd567eff9c67d5f878bbc7ae9d74cf69f2`.
- Embedded `bin/bpp`: 5,325,464 bytes; SHA-256
  `6c8828704e1037788e02d6943cc6cbb61d05d6aadbdd976095b71fc965e8e90e`.
- [Ji et al. simulation-control archive](https://doi.org/10.5281/zenodo.6993702)
  ([exact file](https://zenodo.org/api/records/6993702/files/simulation-control-files.tgz/content)):
  2,068 bytes; MD5 `c233514a93ad48fc67da3be14fa93264`; SHA-256
  `492c0d8a316d7349a77b5482c46693b4e8a5a05acce4b617651b3ad1f4ef3b02`.
- [Ji et al. paper](https://doi.org/10.1093/sysbio/syac077).

## Interpretation and reproducibility limits

The official paper used four haploid copies per population and reported BPP
detection power. This derivative uses 200 haploid copies for R/Q/D, 500 linked
sites in each of 500 independent loci, and maps R/Q/D to P1/P2/P3. Its B/C
direction accuracy must not be compared numerically with published BPP
detection power. Episodic `phi` is also not DNNaic's continuous per-generation
migration-rate estimand.

The D rows reconstruct the paper's inflow-asymmetric `phi=0` boundary as a
donor-phi-zero/resident-phi-one control because the Zenodo archive has no
separate null MCcoal file. D is therefore a reconstructed boundary control,
not a published Ji et al. null bank.

`results.json` contains the row-level ledger needed to inspect or recompute
every reported prediction, probability, hash, and metric. Durable curve/parser
rederivation additionally requires the feature checkpoint, all 90 per-job
count checkpoints, and the retained raw B/C/D alignment triad. Those audit
artifacts have been hash-verified but are not committed here; this compact
bundle must be treated as inspection-only until they are deposited in an
immutable versioned archive with an inventory and DOI.
