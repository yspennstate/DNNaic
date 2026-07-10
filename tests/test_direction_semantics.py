"""Check the archived direction semantics against the simulation arrays.

The forward donor -> recipient meaning of each class is not a convention we are
free to choose: it is fixed by how the released arrays were simulated. This test
recovers it from the features. Under continuous migration the donor population
leaks private alleles into the recipient, so its private allelic richness (pi)
falls relative to the no-migration control while the recipient's allelic richness
(alpha) rises. Reading those two signals off the archive must reproduce
``dnnaic.semantics.CASES``; if it does not, the mapping is wrong.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from dnnaic.semantics import CASES, class_for_forward_edge, forward_edge, msprime_pair

# Contract column order (see dnnaic/features.py): g, then alpha_1/2/3, pi_1/2/3,
# pihat_12/13/23, each as (mean, variance, se).
ALPHA_MEAN = {"P1": 1, "P2": 4, "P3": 7}
PI_MEAN = {"P1": 10, "P2": 13, "P3": 16}
POPS = ("P1", "P2", "P3")

STRONG_RATE = 2.5e-4
DATA_ROOT = Path(
    os.environ.get(
        "DNNAIC_SIM_DATA",
        r"C:\Users\owner\ADZEProjects\ADZE-IntrogressionDNNs\data\simulation_data\regen_full",
    )
)


def test_msprime_and_forward_are_consistent():
    # msprime source->dest is backwards; forward gene flow is dest->source.
    for label, case in CASES.items():
        if case.forward_donor is None:
            assert case.msprime_source is None and case.msprime_dest is None
            continue
        assert case.forward_donor == case.msprime_dest
        assert case.forward_recipient == case.msprime_source
    # round-trip
    assert class_for_forward_edge("P1", "P2") == "A"
    assert class_for_forward_edge("P3", "P2") == "C"
    assert forward_edge("A") == ("P1", "P2")
    assert msprime_pair("C") == ("P2", "P3")


def test_not_the_reversed_convention():
    # Guard against silently adopting the reversed (donor==msprime source) mapping.
    assert CASES["A"].forward_donor == "P1"
    assert CASES["C"].forward_recipient == "P2"


@pytest.mark.skipif(
    not (DATA_ROOT / "X.npy").exists(),
    reason="simulation arrays not present (fetched from Zenodo); set DNNAIC_SIM_DATA",
)
def test_donor_recipient_recovered_from_archive():
    X = np.load(DATA_ROOT / "X.npy", mmap_mode="r")
    direction = np.load(DATA_ROOT / "direction.npy")
    magnitude = np.load(DATA_ROOT / "magnitude.npy")

    control = direction == "D"
    assert control.sum() > 0, "no control rows found"

    def col_means(mask, col):
        return float(np.asarray(X[mask, col]).mean())

    base_pi = {p: col_means(control, PI_MEAN[p]) for p in POPS}
    base_alpha = {p: col_means(control, ALPHA_MEAN[p]) for p in POPS}

    for label in ("A", "B", "C"):
        sel = (direction == label) & np.isclose(magnitude, STRONG_RATE, rtol=0.05)
        assert sel.sum() > 0, f"no strong-rate rows for class {label}"
        d_pi = {p: col_means(sel, PI_MEAN[p]) - base_pi[p] for p in POPS}
        d_alpha = {p: col_means(sel, ALPHA_MEAN[p]) - base_alpha[p] for p in POPS}
        donor = min(POPS, key=lambda p: d_pi[p])       # largest private-richness drop
        recipient = max(POPS, key=lambda p: d_alpha[p])  # largest allelic-richness rise
        case = CASES[label]
        assert donor == case.forward_donor, (
            f"class {label}: features say donor={donor}, mapping says {case.forward_donor}"
        )
        assert recipient == case.forward_recipient, (
            f"class {label}: features say recipient={recipient}, mapping says {case.forward_recipient}"
        )
