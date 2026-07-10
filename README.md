# DNNaic

DNNaic infers the *direction* of introgression between populations from allelic-rarefaction
summary statistics, without an outgroup. It is a two-stage fully connected network: the first stage
detects appreciable gene flow, the second orients it, both from the same compact feature vector.
The features are nine rarefaction statistics — allelic richness, private allelic richness, and
pairwise-private richness across three populations, each summarized across loci by its mean,
variance, and standard error — computed with [PADZE](https://github.com/Andres42611/PADZE).

This repository holds the code, data pointers, and manuscript for the paper *Inferring the
Direction of Introgression from Allelic Rarefaction Statistics with Deep Learning* (del Castillo and
Shmalo).

## What it does

Classical statistics such as Patterson's *D* detect gene flow but, being symmetric under exchange of
donor and recipient, cannot say which population donated to which. The rarefaction statistics keep
the per-population labels that *D* discards, so a direction is recoverable from them. On coalescent
simulations, three-way direction accuracy rises with the migration rate to 96.6% at the largest
fixed rate and 99.7% across the appreciable band (75.6% averaged over all rates), and a
detect-then-orient gate flags appreciable migration at ROC-AUC 0.99 with a well-calibrated score.
A logistic model on the same features matches the network, as the theory predicts, so the signal
lives in the features rather than the architecture.

A separate four-population benchmark asks the combinatorially complete question: which of all
12 ordered donor--recipient edges generated the data? On 240 independent baseline simulations,
ordinary logistic regression reaches 50.0% exact accuracy and training-fold-only augmentation by
all 24 population relabellings raises it to 62.9% (chance 8.3%). The symmetry-aware model recovers
the unordered pair at 76.3% and orients 82.5% of correctly recovered pairs. Raw RBF, MLP,
learned-feature RBF, and Marchenko--Pastur-whitened neural-kernel variants do not beat the linear
rule. Frozen nuisance shifts are reported, including the failures; this is a controlled scaling
result, not a claim of universal demographic transfer.

Natural data expose an important boundary. In four selected *Heliconius* geographic trios,
Patterson's *D* and a raw pair-private contrast support excess sharing, but an intended allopatric
control reverses the raw contrast while the frozen direction head still calls the same class at an
uncalibrated score of 0.992. High confidence and 21/21 leave-one-chromosome stability are therefore
systematic simulation-to-natural extrapolation, not external validation. On the Neanderthal system
the gate instead abstains because the available archaic sample is shallow and off distribution.
Natural direction calls require target-demography training and independent labelled controls.

## Install

```
pip install -e .            # exposes the `dnnaic` package and its dependencies
pip install -e '.[simulate]'   # also install msprime/tskit to regenerate simulations
```

The feature engine, PADZE, installs from PyPI as a dependency (`pip install padze`).

## Reproduce

The simulation feature arrays are on Zenodo (see [data/DATA_SOURCES.md](data/DATA_SOURCES.md));
point the loaders at them and run:

```
export DNNAIC_DATA=/path/to/simulation_data

python scripts/direction_detection.py      # per-rate direction accuracy; detection gate scored against the zero-migration control
python scripts/direction_curve.py          # per-rate direction with exact fixed rates and the wider rate bands reported separately
python scripts/appreciable_gate.py         # detection gate scored against both the weak positive rates and the control, with calibration
python scripts/matched_exposure.py         # direction with the migration exposure m*T held equal across the three classes
python scripts/nuisance_transfer.py        # direction under held-out demographies, the model frozen before any new replicate is scored
python scripts/moment_ablation.py          # variance-vs-mean orientation ablation
python scripts/depth_requirement.py        # how orientation degrades at shallow rarefaction depth
python scripts/make_figures.py             # schematic and exploratory figures
python scripts/twelve_direction_extension.py --workers 2  # all 12 edges; CPU-only, checkpointed
```

The 12-edge command writes its complete seed ledger, realized migration epochs, fold-overlap
audit, uncertainty, negative controls, nuisance-transfer results, and hashes to
`results/twelve_direction_2026_07_10/`. The committed result uses 20 independent baseline
replicates and 8 nuisance-shift replicates per edge (624 simulations total).

To regenerate the simulations from scratch instead of downloading them:

```
python scripts/simulate_demography.py --out-dir data/raw/trees --seed 12345
# then PADZE feature extraction (see data/README.md)
```

Real-data analyses stream only the windows they need over HTTP range requests:

```
python scripts/realdata_heliconius.py       # exploratory positive-panel analysis (butterflies)
python scripts/realdata_heliconius_robustness.py  # fixed-protocol panel/control specificity audit
python scripts/realdata_mouse.py            # second taxon: M. spretus -> M. m. domesticus (Vkorc1)
python scripts/realdata_mouse_depth.py      # the depth-requirement curve, measured on real mouse data
python scripts/realdata_mouse_diversity.py  # when it works: the diversity-balance condition (mouse)
python scripts/realdata_neanderthal.py      # archaic introgression into non-Africans
python scripts/realdata_1000g_injection.py  # injected-signal recovery on real human backgrounds
python scripts/external_benchmarks.py --help # depth-matched duck/runner-bean positive-control OOD audit
python scripts/additional_external_benchmarks.py --help # giraffe/brook-trout transfer audit
python scripts/further_external_benchmarks.py --help # scrub-jay and matched Malawi positive/null audit
```

The external benchmark runner records its fixed sample manifests, source/training-array hashes,
shared-polymorphic locus contract, uncalibrated scores, OOD diagnostics, and published controls
under `results/external_benchmarks_2026_07_10/`; these stress tests are not classifier validation.
The additional bundles apply the same guardrails to giraffe, brook trout, scrub jay, and
matched Lake Malawi positive/negative panels.

## Layout

```
paper/       manuscript source, bibliography, and figures
dnnaic/      importable library: feature contract, data loaders, leakage-free evaluation
scripts/     runnable analysis and figure code
data/        data methods and dataset pointers (bulk data is fetched, not committed)
tests/       feature-contract test
```

## Citing this work

If you use this code or the rarefaction-feature approach, please cite the paper:

> A. del Castillo and Y. Shmalo. Inferring the Direction of Introgression from Allelic Rarefaction
> Statistics with Deep Learning.

The simulated dataset has its own archive (DOI [10.5281/zenodo.21233067](https://doi.org/10.5281/zenodo.21233067)),
and the feature engine is [PADZE](https://github.com/Andres42611/PADZE).

## License

MIT — see [LICENSE](LICENSE).
