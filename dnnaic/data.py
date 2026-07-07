"""Loaders for the simulation feature arrays and the leakage-free evaluation protocol.

The arrays (from Zenodo 10.5281/zenodo.21233067) live under a data root set by the
``DNNAIC_DATA`` environment variable, or passed explicitly. Each dataset directory holds
``X.npy`` (per-row 28-D features), ``direction.npy`` (A/B/C/D), ``magnitude.npy`` (per-row
migration rate), and ``groups.npy`` (the true simulation-replicate id).

Because rarefaction yields 198 correlated rows per replicate, all evaluation groups by
``groups.npy`` so that no replicate is shared between train and test (Theorem: row-level
splits are optimistically biased).
"""
from __future__ import annotations
import os
import numpy as np

CLASSES = np.array(["A", "B", "C"])

# 28-D layout: col 0 = depth g; statistic i in 0..8 -> mean=1+3i, var=2+3i, se=3+3i.
MEAN_COLS = [1 + 3 * i for i in range(9)]
VAR_COLS = [2 + 3 * i for i in range(9)]
SE_COLS = [3 + 3 * i for i in range(9)]
# feature-channel column sets used in the ablations
CHANNELS = dict(
    full=list(range(28)),
    richness=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],          # symmetric alpha
    private=[0] + list(range(10, 19)),                 # pi
    pairwise=[0] + list(range(19, 28)),                # pihat
    asymmetric=[0] + list(range(10, 28)),              # pi + pihat
)
MOMENT_SETS = dict(
    mean_only=[0] + MEAN_COLS,
    mean_var=[0] + sorted(MEAN_COLS + VAR_COLS),
    mean_se=[0] + sorted(MEAN_COLS + SE_COLS),
    var_only=[0] + VAR_COLS,
    full=list(range(28)),
)


def data_root(root=None):
    return root or os.environ.get("DNNAIC_DATA", "data/simulation_data")


def load_dataset(name, root=None):
    """Return (X, direction, groups, magnitude) for one dataset directory."""
    d = os.path.join(data_root(root), name)
    X = np.load(os.path.join(d, "X.npy"))
    direction = np.load(os.path.join(d, "direction.npy"), allow_pickle=True).astype("U8")
    groups = np.load(os.path.join(d, "groups.npy"), allow_pickle=True).astype("U40")
    magnitude = np.load(os.path.join(d, "magnitude.npy"))
    return X, direction, groups, magnitude


def group_folds(group_index, seed, n_splits=5):
    """Yield (train_rows, test_rows) with whole replicates held out (GroupKFold, shuffled)."""
    n_groups = int(group_index.max()) + 1
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)
    fold_of = np.empty(n_groups, int)
    for f in range(n_splits):
        fold_of[perm[f::n_splits]] = f
    row_fold = fold_of[group_index]
    for f in range(n_splits):
        yield np.where(row_fold != f)[0], np.where(row_fold == f)[0]


def aggregate_to_replicates(probs, group_index, test_rows, n_groups):
    """Average per-row class probabilities up to one vector per replicate."""
    k = probs.shape[1]
    num = np.zeros((n_groups, k))
    den = np.zeros(n_groups)
    np.add.at(num, group_index[test_rows], probs)
    np.add.at(den, group_index[test_rows], 1.0)
    seen = den > 0
    out = np.zeros((n_groups, k))
    out[seen] = num[seen] / den[seen, None]
    return out, seen


def expected_calibration_error(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= edges[i + 1]))
        if m.sum():
            e += abs(y[m].mean() - p[m].mean()) * m.sum()
    return e / len(y)
