# Yellowstone cutthroat--rainbow trout transfer stress test

This result was generated from hash-pinned Dryad version 2 files at clean code
commit `0c78eb5363ce2d39be69d0c1e1b2e53e5cae3b66`. The frozen primary panel is
SFOwlCreek Yellowstone cutthroat trout P1 (61), Trout Creek P2 (58), and Story
Hatchery rainbow trout P3 (20). Candidate class C is an ecological
introduction-history prior—not an exclusive edge or accuracy label.

| correlated sensitivity | loci | raw head | direction RMS-z | gate | gate RMS-z | P2 projection toward P3 (95% chromosome-block interval) |
|---|---:|---:|---:|---:|---:|---:|
| main, source GT, standard | 11,758 | C | 18.46 | 1.0 | 23.57 | 0.764 (0.755--0.774) |
| main, source GT, within-population polymorphic | 2,591 | C | 20.85 | 1.0 | 30.58 | 0.781 (0.771--0.792) |
| main, unique PL argmin, standard | 12,170 | C | 22.13 | 1.0 | 27.68 | 0.668 (0.658--0.680) |
| main, unique PL argmin, within-population polymorphic | 3,653 | C | 22.09 | 1.0 | 31.40 | 0.694 (0.681--0.707) |
| Tensleep reference swap, source GT | 9,642 | C | 15.75 | 1.0 | 14.99 | 0.661 (0.649--0.672) |
| directly stocked Big Creek, source GT | 9,721 | C | 17.11 | 1.0 | 17.14 | 0.778 (0.767--0.788) |
| same-species candidate-null diagnostic, source GT | 7,563 | B | 14.83 | 0.9999 | 18.16 | -0.040 (-0.050---0.029) |

All seven rows exceed the heuristic severe-OOD RMS-z > 10 diagnostic and
abstain. The six raw C outputs are correlated candidate concordances, not 6/6
accuracy; the accuracy denominator is null. The candidate-null gate also
saturates, showing that the learned gate is not an OOD detector.

The source-GT main projection (0.764) and PL sensitivity (0.668) straddle the
published site-level RBT ancestry comparator 0.7382, but that comparator was
computed from the same SNPs. Projection and plug-in f3 are descriptive sample
frequency geometry, not bounded ancestry, migration, or temporal direction.
The bootstrap resamples 29 chromosomes, not fish.

The source audit found complete population--library confounding, strongly
different GT call rates (P1 0.790, P2 0.548, P3 0.717), 1,789,601 informative
PL cells that the released converter zeroes when GT is masked, duplicated Dryad
v2 paths, absent per-individual q/Q labels, and non-held-out global/panel locus
ascertainment. The corrigendum DOI is recorded, but its publisher-blocked text
remains substantively unaudited.

A detached CRLF checkout reproduced the main LF checkout bit-for-bit after
removing absolute paths, runtime argv, and the recorded working-tree line-ending
representation: canonical semantic SHA-256
`aa8126845390e393374c512eebf2451a1e300000c5a3a97fd702aca9fbf2003d`.
