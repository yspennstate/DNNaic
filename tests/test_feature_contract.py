"""Regression coverage for DNNaic-facing 28-column feature order."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[1]
EXPECTED_BLOCKS = [
    "alpha_1", "alpha_2", "alpha_3",
    "pi_1", "pi_2", "pi_3",
    "pihat_12", "pihat_13", "pihat_23",
]


def _load_module(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, REPO / relpath)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dnnaic_builders_share_pihat_block_order():
    build_dataset = _load_module("dnnaic_build_dataset", "scripts/repro/build_dataset.py")
    real_to_dnnaic = _load_module(
        "real_to_dnnaic_matrix", "scripts/pipeline/real_to_dnnaic_matrix.py"
    )

    assert build_dataset.CANON_STATS == EXPECTED_BLOCKS
    assert real_to_dnnaic.CONTRACT_BLOCKS == EXPECTED_BLOCKS


def test_real_to_dnnaic_matrix_uses_12_13_23_pihat_columns(monkeypatch):
    real_to_dnnaic = _load_module(
        "real_to_dnnaic_matrix_for_columns", "scripts/pipeline/real_to_dnnaic_matrix.py"
    )

    source_cols = ["g"] + [
        f"{block}_{moment}"
        for block in EXPECTED_BLOCKS
        for moment in real_to_dnnaic.MOMENTS
    ]
    source_matrix = np.arange(len(source_cols), dtype=float).reshape(1, -1)

    class FakeTable:
        def to_frame(self):
            return source_matrix, source_cols

    class FakeLoci:
        populations = ["P1", "P2", "P3"]

    monkeypatch.setattr(real_to_dnnaic, "read_vcf", lambda vcf, popmap: FakeLoci())
    monkeypatch.setattr(real_to_dnnaic, "compute_features", lambda *args, **kwargs: FakeTable())

    X, cols, _ = real_to_dnnaic.build_matrix("input.vcf", "popmap.tsv")

    assert cols[19:28] == [
        "pihat_12_mean", "pihat_12_variance", "pihat_12_se",
        "pihat_13_mean", "pihat_13_variance", "pihat_13_se",
        "pihat_23_mean", "pihat_23_variance", "pihat_23_se",
    ]
    assert X.tolist() == source_matrix.tolist()
