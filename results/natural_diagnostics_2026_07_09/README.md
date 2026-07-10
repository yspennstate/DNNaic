# Natural-panel diagnostics (2026-07-09)

These are exploratory simulation-to-natural stress tests, not external
validation of introgression direction.  Both analyses use the paper's primary
direction head: one 54-D vector per panel (mean and population SD across the
available rarefaction depths for each of 27 non-depth coordinates), scored by a
StandardScaler and multinomial logistic regression fit to all 2,700 positive
canonical simulation replicates.  Natural softmax outputs are uncalibrated
out-of-distribution scores.

## Mouse: one locus panel for both P1 choices

`mouse.json` records the complete provenance and sensitivity results.

- Source: Harr wild-mouse VCF, GRCm38/mm10, chr7:126,000,000--129,000,000.
- The cached BGZF fetch crossed the requested end (last retained source record
  128,999,980), contains 146,950 source records, and has SHA-256
  `87733c3faf3640a542cc94e231f52d055c4af9d3b4ce390d6af53c5a09c2ba7a`.
- A single four-population intersection required at least 16 called gene copies
  in musculus, castaneus, domesticus, and spretus and polymorphism after either
  P1 was omitted.  Of 78,788 eligible loci, 15,000 deterministic evenly spaced
  loci spanning 126,000,024--128,999,980 were used in the same order and with
  the same REF/ALT coding for both panels.

At the manuscript convention `g >= 8`:

| P1 | raw P2--P3/P1--P3 | diversity-normalized | 54-D call |
|---|---:|---:|---|
| musculus | 3.8315 | 1.2538 | C |
| castaneus | 1.4418 | 1.1071 | C |

The old raw sign reversal disappears when loci and the model protocol are held
fixed; normalization is therefore not shown to repair that reversal.  The
normalized ratio is sensitive to the minimum depth: at `g >= 2` it is 0.8218
and 0.8327, respectively, and crosses one as the cutoff rises.  Denominator
clips from no clipping through `1e-4` give identical values because the observed
denominators are not near zero.

## Heliconius: discordance, not validation

`heliconius.json` records source hashes, every sample ID, filters, reservoir
seeds, per-panel chromosome counts, ratio sensitivity, model scores, and the
canonical-panel chromosome omissions.

- Source `.geno.gz` SHA-256:
  `79ac9932a0480b946085c64b42d711602852b373452c7f5cdd46853f85a0ff66`.
- Source race map SHA-256:
  `edc3cc2a82592e544560b80d1dc8031a76e1983dd226f6b1c0508dcce029d71a`.
- Each panel has 10 diploid individuals (20 possible gene copies) per focal
  population, requires at least 16 called copies in each population, and uses a
  seeded 15,000-SNP reservoir across all 21 chromosomes.
- Exactly four pre-existing geographic trios and one genuine allopatric
  `mel_ros` control are reported.  No second control is invented.

At `g >= 8`:

| Panel (cydno/timareta/melpomene race) | raw ratio | normalized ratio | 54-D call |
|---|---:|---:|---|
| chi/txn/ama | 2.5629 | 3.2807 | C |
| zel/flo/mel | 1.8825 | 2.4065 | C |
| chi/flo/mal | 2.0448 | 2.6485 | C |
| zel/txn/ama | 2.4855 | 3.2793 | C |
| zel/flo/ros (allopatric control) | 0.6260 | 0.8216 | C |

The sharing ratios descriptively separate these four chosen trios from the one
control, but panels reuse populations, individuals, and linked loci and do not
have exact race-level donor/recipient ground truth.  More importantly, the
primary direction head calls C even in the allopatric control.  It also calls C
in all 21 leave-one-chromosome-out versions of the first trio (uncalibrated C
scores 0.999999982--0.999999991).  Stability here is evidence of systematic
simulation-to-natural extrapolation, not biological validation.

## Reproduction

After installing the package and setting the simulation-array path:

```text
python scripts/realdata_mouse_diversity.py --data-root /path/to/simulation_data
python scripts/realdata_heliconius_robustness.py \
  --data-root /path/to/simulation_data
```

Both scripts cache source inputs under `data/real`, emit full JSON here, and
duplicate the latest result beside the generated VCF caches.  Cached local
source files may be supplied with `--geno` and `--popmap` for Heliconius.
