# Four-population, 12-direction extension

The primary exchangeable benchmark contains 240 independent coalescent replicates across all 12 forward-time ordered edges.

## Leakage-free baseline

| model | exact 12-way | unordered pair | orientation given pair | donor | recipient |
|---|---:|---:|---:|---:|---:|
| logistic | 0.500 | 0.617 | 0.811 | 0.567 | 0.646 |
| equivariant_logistic | 0.629 | 0.762 | 0.825 | 0.654 | 0.721 |
| mp_bulk_lda | 0.487 | 0.646 | 0.755 | 0.567 | 0.592 |
| rbf_kernel | 0.492 | 0.625 | 0.787 | 0.571 | 0.629 |
| mlp | 0.454 | 0.571 | 0.796 | 0.529 | 0.600 |
| deep_feature_rbf | 0.442 | 0.558 | 0.791 | 0.512 | 0.588 |
| deep_mp_rbf | 0.450 | 0.567 | 0.794 | 0.517 | 0.596 |
| label_invariant_logistic | 0.071 | 0.142 | 0.500 | 0.267 | 0.208 |

Chance levels are 1/12 for the exact edge, 1/6 for the unordered pair, 1/2 for orientation conditional on the right pair, and 1/4 for donor or recipient.
A separate binary reversal audit, given the true unordered pair, reaches 0.787 (189/240).

## Frozen nuisance transfer

| family | logistic | equivariant logistic | MP-bulk LDA | RBF | MLP | deep-feature RBF | deep+MP+RBF |
|---|---:|---:|---:|---:|---:|---:|---:|
| weak_signal | 0.135 | 0.167 | 0.135 | 0.104 | 0.125 | 0.125 | 0.094 |
| unequal_ne | 0.302 | 0.458 | 0.365 | 0.271 | 0.250 | 0.229 | 0.240 |
| balanced_tree | 0.271 | 0.354 | 0.323 | 0.177 | 0.208 | 0.198 | 0.156 |
| half_sequence | 0.250 | 0.490 | 0.271 | 0.083 | 0.240 | 0.219 | 0.167 |

Frozen transfer uses models fit once on the baseline family. Failures are part of the result; within-family refits are reported in `results.json` only to distinguish loss of transfer from loss of information.

All intervals, classwise counts, confusion matrices, epoch audits, seed ledgers, negative controls, software versions, and artifact hashes are in `results.json`.
