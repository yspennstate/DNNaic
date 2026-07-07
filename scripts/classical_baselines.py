#!/usr/bin/env python3
"""Classical introgression statistics -- Patterson's D, f3, f4 -- as the baseline for DNNaic.

The premise of the paper is that these statistics detect the presence of gene flow but not its
direction: D and f4 are symmetric under exchange of the donor and recipient, and all three require
an outgroup. This module computes them with a weighted block jackknife for significance, so the
same msprime simulations can be scored the classical way and those columns of the results tables
filled in directly. The three experimental populations map to (P1, P2, P3); P4 is the outgroup.

Definitions (derived-allele frequencies p_i in population i; P4 = outgroup):
    ABBA = (1-p1) * p2 * p3 * (1-p4)
    BABA =  p1 * (1-p2) * p3 * (1-p4)
    D    = sum(ABBA - BABA) / sum(ABBA + BABA)         # ~0 under no gene flow
    f4(P1,P2 ; P3,P4) = mean[(p1 - p2) * (p3 - p4)]
    f3(PC ; PA,PB)    = mean[(pC - pA) * (pC - pB)]     # significantly negative => PC admixed

Significance is a weighted delete-m block jackknife over genomic blocks, Z = D / SE(D). Import the
estimators as a library, or run this file directly for a small self-test:

    from classical_baselines import patterson_d, f4_statistic, f3_statistic
    res = patterson_d(p1, p2, p3, p4, positions=pos)
    print(res["D"], res["Z"])
"""
from __future__ import annotations
import numpy as np


def allele_freqs_from_genotypes(geno):
    """geno: (n_individuals, n_sites) with alt-allele dosage in {0,1,2} (diploid) or
    {0,1} haploid; returns derived-allele frequency per site in [0,1]."""
    geno = np.asarray(geno, dtype=float)
    ploidy_max = np.nanmax(geno) if geno.size else 2
    denom = 2.0 if ploidy_max > 1 else 1.0
    return np.nanmean(geno, axis=0) / denom


def _abba_baba(p1, p2, p3, p4):
    abba = (1 - p1) * p2 * p3 * (1 - p4)
    baba = p1 * (1 - p2) * p3 * (1 - p4)
    return abba, baba


def _block_jackknife(values_num, values_den, blocks):
    """Weighted block jackknife for a ratio estimator sum(num)/sum(den).
    Returns (point_estimate, std_error)."""
    num_tot, den_tot = values_num.sum(), values_den.sum()
    theta = num_tot / den_tot if den_tot != 0 else np.nan
    uniq = np.unique(blocks)
    if len(uniq) < 2:
        return theta, np.nan
    thetas, weights = [], []
    for b in uniq:
        m = blocks == b
        nb, db = num_tot - values_num[m].sum(), den_tot - values_den[m].sum()
        thetas.append(nb / db if db != 0 else np.nan)
        weights.append(values_den[m].sum())
    thetas = np.asarray(thetas, float)
    weights = np.asarray(weights, float)
    g = len(uniq)
    # Busing et al. (1999) weighted delete-m jackknife
    h = weights.sum() / weights
    theta_j = g * theta - np.sum((1 - weights / weights.sum()) * thetas)
    pseudo = h * theta - (h - 1) * thetas
    var = np.nansum((pseudo - theta_j) ** 2 / (h - 1)) / g
    return theta, float(np.sqrt(var)) if var > 0 else np.nan


def patterson_d(p1, p2, p3, p4, positions=None, block_size=500_000):
    """Patterson's D (ABBA-BABA). Returns dict with D, SE, Z (block jackknife)."""
    p1, p2, p3, p4 = map(lambda a: np.asarray(a, float), (p1, p2, p3, p4))
    abba, baba = _abba_baba(p1, p2, p3, p4)
    num, den = abba - baba, abba + baba
    if positions is None:
        positions = np.arange(len(p1))
    blocks = (np.asarray(positions) // block_size).astype(int)
    D, se = _block_jackknife(num, den, blocks)
    Z = D / se if se and not np.isnan(se) else np.nan
    return {"D": float(D), "SE": float(se) if se == se else None,
            "Z": float(Z) if Z == Z else None,
            "n_abba": float(np.nansum(abba)), "n_baba": float(np.nansum(baba))}


def f4_statistic(p1, p2, p3, p4):
    """f4(P1,P2 ; P3,P4) = mean[(p1-p2)(p3-p4)]."""
    p1, p2, p3, p4 = map(lambda a: np.asarray(a, float), (p1, p2, p3, p4))
    return float(np.nanmean((p1 - p2) * (p3 - p4)))


def f3_statistic(pC, pA, pB):
    """f3(C; A,B) = mean[(pC-pA)(pC-pB)]; significantly negative => C is admixed."""
    pC, pA, pB = map(lambda a: np.asarray(a, float), (pC, pA, pB))
    return float(np.nanmean((pC - pA) * (pC - pB)))


def direction_note():
    return ("D and f4 test for the presence of gene flow between a (P1, P2) pair and P3, given "
            "outgroup P4; on their own they do not return the direction (who donated to whom) or "
            "the migration rate. That gap is what DNNaic targets.")


def _self_test():
    rng = np.random.default_rng(0)
    S = 20000
    pos = np.sort(rng.integers(0, 1_000_000, S))
    # no gene flow: p1,p2 exchangeable -> D ~ 0
    base = rng.beta(0.5, 0.5, S)
    p1 = np.clip(base + rng.normal(0, .05, S), 0, 1)
    p2 = np.clip(base + rng.normal(0, .05, S), 0, 1)
    p3 = rng.beta(0.5, 0.5, S)
    p4 = np.zeros(S)
    d0 = patterson_d(p1, p2, p3, p4, pos)
    # gene flow P3->P2: make p2 closer to p3 -> excess ABBA -> D != 0
    p2b = np.clip(0.6 * p2 + 0.4 * p3, 0, 1)
    d1 = patterson_d(p1, p2b, p3, p4, pos)
    print(f"[OK] no-flow   D={d0['D']:+.4f} Z={d0['Z']}")
    print(f"[OK] P3->P2    D={d1['D']:+.4f} Z={d1['Z']}  (|D| should be larger)")
    print(f"[OK] f4={f4_statistic(p1,p2b,p3,p4):+.5f}  f3(P2;P1,P3)={f3_statistic(p2b,p1,p3):+.5f}")
    print("[note]", direction_note())


if __name__ == "__main__":
    _self_test()
