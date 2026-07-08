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

On real data the frozen model recovers the documented *melpomene*→*timareta* direction in the
*Heliconius* butterfly complex, where a symmetric *D* detects the gene flow but cannot orient it.
On the Neanderthal system it reproduces the classic excess sharing between non-Africans and the
archaic genome, but there the gate abstains rather than orienting: with only a couple of
high-coverage archaic genomes the rarefaction depth is far shallower than the method needs, and the
paper is explicit about that boundary.

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

python scripts/direction_detection.py      # per-rate direction accuracy + detection gate
python scripts/moment_ablation.py          # variance-vs-mean orientation ablation
python scripts/depth_requirement.py        # how orientation degrades at shallow rarefaction depth
python scripts/make_figures.py             # schematic and exploratory figures
```

To regenerate the simulations from scratch instead of downloading them:

```
python scripts/simulate_demography.py --out-dir data/raw/trees --seed 12345
# then PADZE feature extraction (see data/README.md)
```

Real-data analyses stream only the windows they need over HTTP range requests:

```
python scripts/realdata_heliconius.py       # labelled adaptive introgression (butterflies)
python scripts/realdata_mouse.py            # second taxon: M. spretus -> M. m. domesticus (Vkorc1)
python scripts/realdata_mouse_depth.py      # the depth-requirement curve, measured on real mouse data
python scripts/realdata_neanderthal.py      # archaic introgression into non-Africans
python scripts/realdata_1000g_injection.py  # injected-signal recovery on real human backgrounds
```

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
