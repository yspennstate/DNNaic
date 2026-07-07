"""The 28-column feature order DNNaic depends on is a contract; this pins it.

`build_matrix` must always emit the depth `g` first, then the nine rarefaction statistics in a
fixed block order, each expanded into (mean, variance, se) — regardless of the column order PADZE
happens to return. The pairwise-private block in particular must be 12, 13, 23, because the model
was trained on that ordering.
"""
from __future__ import annotations

import numpy as np

import dnnaic
from dnnaic import features


EXPECTED_BLOCKS = [
    "alpha_1", "alpha_2", "alpha_3",
    "pi_1", "pi_2", "pi_3",
    "pihat_12", "pihat_13", "pihat_23",
]


def _fake_padze(monkeypatch, source_cols):
    """Make build_matrix read a synthetic feature table whose values encode column index."""
    source = np.arange(len(source_cols), dtype=float).reshape(1, -1)

    class FakeTable:
        def to_frame(self):
            return source, source_cols

    class FakeLoci:
        populations = ["P1", "P2", "P3"]

    monkeypatch.setattr(features, "read_vcf", lambda vcf, popmap: FakeLoci())
    monkeypatch.setattr(features, "compute_features", lambda *a, **k: FakeTable())


def test_contract_blocks_are_stable():
    assert dnnaic.CONTRACT_BLOCKS == EXPECTED_BLOCKS
    assert dnnaic.MOMENTS == ("mean", "variance", "se")


def test_build_matrix_reorders_to_contract(monkeypatch):
    # Hand build_matrix a deliberately scrambled column order and confirm it re-sorts.
    scrambled_blocks = ["pihat_23", "pi_2", "alpha_1", "pihat_12", "alpha_3",
                        "pi_1", "pihat_13", "alpha_2", "pi_3"]
    scrambled_moments = ["se", "variance", "mean"]
    source_cols = [f"{b}_{m}" for b in scrambled_blocks for m in scrambled_moments] + ["g"]
    _fake_padze(monkeypatch, source_cols)

    X, cols, _ = dnnaic.build_matrix("input.vcf", "popmap.tsv")

    expected = ["g"] + [f"{b}_{m}" for b in EXPECTED_BLOCKS for m in dnnaic.MOMENTS]
    assert cols == expected
    assert len(cols) == 28

    # Each output value is the source-column index it was pulled from, so a correct
    # reordering means output value == index of that name in the scrambled source.
    src_ix = {c: i for i, c in enumerate(source_cols)}
    assert X[0].tolist() == [float(src_ix[c]) for c in expected]


def test_pairwise_private_block_is_12_13_23(monkeypatch):
    source_cols = ["g"] + [f"{b}_{m}" for b in EXPECTED_BLOCKS for m in features.MOMENTS]
    _fake_padze(monkeypatch, source_cols)

    _, cols, _ = dnnaic.build_matrix("input.vcf", "popmap.tsv")

    assert cols[19:28] == [
        "pihat_12_mean", "pihat_12_variance", "pihat_12_se",
        "pihat_13_mean", "pihat_13_variance", "pihat_13_se",
        "pihat_23_mean", "pihat_23_variance", "pihat_23_se",
    ]
