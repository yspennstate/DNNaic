from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "additional_external_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("additional_external_benchmarks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_depth_matched_gate_features_contract():
    table = np.zeros((2, 15, 28), dtype=float)
    table[:, :, 0] = np.arange(2, 17)
    table[0, :, 1:] = np.arange(15)[:, None]
    table[1, :, 1:] = 100 + np.arange(15)[:, None]

    features, indices = MODULE.depth_matched_gate_features(table)

    assert indices == [1, 2, 3, 5, 7, 10, 14]
    assert features.shape == (2, 216)
    assert np.all(features[0, -27:] == 7)
    assert np.all(features[1, -27:] == 107)


def test_depth_matched_gate_rejects_mismatched_grids():
    table = np.zeros((2, 15, 28), dtype=float)
    table[:, :, 0] = np.arange(2, 17)
    table[1, -1, 0] = 99
    with pytest.raises(ValueError, match="depth grid"):
        MODULE.depth_matched_gate_features(table)


@pytest.mark.parametrize(
    "name, populations, n_samples",
    [
        ("brook_trout_shared.tsv", {"AFP", "BAK", "LFA", "LFR"}, 102),
        ("brook_trout_lfa_null.tsv", {"AFP", "BAK", "LFA"}, 73),
        ("brook_trout_lfr_null.tsv", {"AFP", "BAK", "LFR"}, 79),
    ],
)
def test_brook_trout_manifest_contract(name, populations, n_samples):
    path = Path(__file__).resolve().parents[1] / "data" / "external_benchmarks" / name
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()[1:]]
    assert len(rows) == n_samples
    assert len({row[0] for row in rows}) == n_samples
    assert {row[1] for row in rows} == populations


def test_giraffe_manifest_contract():
    path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "external_benchmarks"
        / "giraffe_nubian_reticulated.tsv"
    )
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()[1:]]
    counts = {population: sum(row[1] == population for row in rows) for population in {row[1] for row in rows}}
    assert counts == {
        "Reticulated_14-18": 22,
        "Reticulated_8-13": 9,
        "Nubian_3": 11,
    }
    assert len({row[0] for row in rows}) == 42
