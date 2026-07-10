"""Structural tests for the all-12-direction experiment."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "twelve_direction_extension.py"
SPEC = importlib.util.spec_from_file_location("twelve_direction_extension", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_all_ordered_edges_are_present_once():
    assert len(MODULE.DIRECTIONS) == 12
    assert len(set(MODULE.DIRECTIONS)) == 12
    assert set(MODULE.DIRECTIONS) == {
        f"P{i}->P{j}" for i in range(1, 5) for j in range(1, 5) if i != j
    }


def test_forward_backward_epoch_and_exposure_are_explicit():
    report = MODULE.validate_demography("baseline", "P1->P3")
    assert report["backwards_time_msprime_mapping"] == "P3->P1"
    assert np.isclose(report["integrated_single_lineage_hazard_m_times_T"], 0.25)
    nonzero = [
        entry
        for epoch in report["actual_epochs"]
        for entry in epoch["nonzero_migration"]
    ]
    assert nonzero == [{"source": "P3", "dest": "P1", "rate": 0.001}]


def test_feature_relabeling_is_a_bijection_and_identity_is_exact():
    d = len(MODULE.FEATURE_COLUMNS)
    X = np.arange(2 * d, dtype=float).reshape(2, d)
    identity = MODULE.PERMUTATIONS.index((0, 1, 2, 3))
    assert np.array_equal(
        MODULE.permute_features(X, MODULE.PERM_FEATURE_TARGETS[identity]), X
    )
    for target in MODULE.PERM_FEATURE_TARGETS:
        assert np.array_equal(np.sort(target), np.arange(d))


def test_label_invariant_ablation_is_unchanged_by_any_population_relabeling():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(3, len(MODULE.FEATURE_COLUMNS)))
    baseline = MODULE.label_invariant_features(X)
    for target in MODULE.PERM_FEATURE_TARGETS:
        transformed = MODULE.permute_features(X, target)
        assert np.allclose(MODULE.label_invariant_features(transformed), baseline)


def test_mp_bulk_lda_separates_a_small_full_rank_control():
    rng = np.random.default_rng(11)
    X = np.vstack([
        rng.normal(loc=-2.0, scale=0.5, size=(30, 6)),
        rng.normal(loc=2.0, scale=0.5, size=(30, 6)),
    ])
    y = np.repeat([0, 1], 30)
    model = MODULE.MPBulkLDA().fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
    assert model.aspect_ratio_ < 1
