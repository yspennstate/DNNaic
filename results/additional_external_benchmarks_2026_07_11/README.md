# Additional external benchmarks (2026-07-11)

This bundle adds one directional positive benchmark and two no-introgression specificity
stress tests. It is a transfer audit, **not classifier validation**: every panel is severely
out of the simulation distribution, and the scores below are explicitly uncalibrated.

## Result

The frozen simulation heads fail all four tests in the same qualitative way.

| panel | loci | published expectation | direction call | class-C score | gate score | direction RMS z |
|---|---:|---|---|---:|---:|---:|
| giraffe, standard contract | 15,000 | C, Nubian P3 -> reticulated P2 | A | 1.59e-13 | 1.0 | 16.31 |
| giraffe, within-population-polymorphic robustness | 10,588 | C, Nubian P3 -> reticulated P2 | A | 7.27e-6 | 1.0 | 22.06 |
| brook trout, LFA hatchery null | 15,000 | gate should abstain | A | 8.43e-17 | 1.0 | 17.60 |
| brook trout, LFR hatchery null | 15,000 | gate should abstain | A | 1.87e-11 | 1.0 | 18.00 |

For giraffe, the author-supplied Dsuite row is D=0.211001, Z=25.4689,
p=2.3e-16, f4-ratio=0.199169. OrientAGraph and the paper's asymmetric ancestry
analysis—not D alone—support Nubian-to-Laikipia-reticulated direction. The raw
pair-private sharing ratio also points toward excess P2/P3 sharing (6.55 under the standard
contract and 4.28 under the stricter filter), but the simulation head confidently calls the
wrong direction. The conclusion is invariant to the disputed filtering choice: the standard
contract has 72,759 eligible loci before its fixed-seed 15,000 cap, while the deliberately
stricter within-population-polymorphic analysis retains all 10,588 loci.

For brook trout, the release contains 444 samples and 16,226 VCF variant rows (the Figshare
description says 16,336 SNPs). The study found no significant captive-to-wild introgression
overall, aside from one UTA individual with hatchery ancestry. Yet both hatchery-versus-BAK
panels saturate the matched gate at 1.0 and produce class A rather than the candidate P3->P2
class C. The panels share the identical 15,000 ordered loci (SHA-256
`5000f2995707e447b42b23cbc3748f06b32253b104305f39edf655371f75e396`).

These results extend the manuscript's Heliconius control failure: a frozen gate trained on
the simulation family is not a general natural-data specificity detector. The extreme feature
shifts (direction RMS z 16.3–22.1; gate RMS z 16.3–23.2) make the appropriate response
abstention and target-matched retraining, not reinterpretation of the saturated scores.

## Provenance and reproduction

`results.json` was generated from clean code commit
`98e231da8bb1dc0db6c49818fe6874917e0fcc39`. It records source/derived hashes,
sample manifests, every feature matrix, fitted-head training-array hashes, locus filters and
counts, sharing ratios, all class scores, and complete standardized feature vectors.
The committed LF-normalized JSON SHA-256 is
`a550d37365e6d03d6148df0a57fc62e896cc696d4b1bc83fa8e6f01b5d7166dd`.
An independent detached worktree at the same commit reran all four panels; after normalizing
only the temporary worktree prefix in recorded manifest paths, the JSON was byte-identical.

```bash
python scripts/additional_external_benchmarks.py \
  --data-root /path/to/simulation_data \
  --giraffe-vcf /path/to/dsuite_introgression/snps.sampled.vcf \
  --brook-vcf /path/to/BT_ALL.vcf
```

Source licenses, immutable hashes, DOI links, archive-member paths and exact sample maps are
in `data/external_benchmarks/sources.json` and the committed TSV manifests. Large raw and
derived VCFs are intentionally not redistributed in Git.
