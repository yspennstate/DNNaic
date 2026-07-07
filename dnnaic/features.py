"""Build the DNNaic 28-D input from a VCF, using the PADZE rarefaction engine.

The feature vector at a standardized sample size (rarefaction depth) g is

    [ g,
      alpha_1(mean,var,se), alpha_2(...), alpha_3(...),     # allelic richness
      pi_1(...), pi_2(...), pi_3(...),                      # private allelic richness
      pihat_12(...), pihat_13(...), pihat_23(...) ]         # pairwise-private richness

i.e. nine rarefaction statistics, each summarized across loci by its mean, variance, and
standard error, plus the depth g. Populations are mapped to the caterpillar tree ((P1,P2),P3):
P1,P2 are the sister populations and P3 is the more divergent one. The mapping must be given
explicitly (alphabetical order is not relied on) via `pop_order`.
"""
from __future__ import annotations
import numpy as np
from padze import compute_features, read_vcf

CONTRACT_BLOCKS = ["alpha_1", "alpha_2", "alpha_3",
                   "pi_1", "pi_2", "pi_3",
                   "pihat_12", "pihat_13", "pihat_23"]
MOMENTS = ("mean", "variance", "se")


def _reorder_populations(loci, pop_order):
    cur = list(loci.populations)
    if sorted(pop_order) != sorted(cur):
        raise ValueError(f"pop_order {pop_order} does not match populations {cur}")
    perm = [cur.index(p) for p in pop_order]
    loci.populations = list(pop_order)
    loci.count_matrices = [cm[perm, :] for cm in loci.count_matrices]
    loci.sample_sizes = loci.sample_sizes[:, perm]
    return loci


def build_matrix(vcf, popmap, max_depth=None, pop_order=None):
    """Return (X, columns, loci): X is (n_depths, 28) in the DNNaic contract order."""
    loci = read_vcf(vcf, popmap)
    if len(loci.populations) != 3:
        raise ValueError(f"DNNaic contract is 3-population; got {loci.populations}")
    if pop_order:
        loci = _reorder_populations(loci, list(pop_order))
    depths = None
    if max_depth:
        hi = min(max_depth, loci.max_depth())
        depths = np.arange(2, hi + 1, dtype=np.int64)
    table = compute_features(loci, depths=depths, pihat_sizes=(2,),
                             moments=MOMENTS, bias_corrected=True)
    mat, cols = table.to_frame()
    ix = {c: i for i, c in enumerate(cols)}
    order = [ix["g"]]
    for b in CONTRACT_BLOCKS:
        for m in MOMENTS:
            order.append(ix[f"{b}_{m}"])
    X = mat[:, order].astype(float)
    contract_cols = ["g"] + [f"{b}_{m}" for b in CONTRACT_BLOCKS for m in MOMENTS]
    return X, contract_cols, loci
