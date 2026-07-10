from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "further_external_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("further_external_benchmarks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "name, expected",
    [
        ("scrub_jay_null.tsv", {"iw": 18, "mw": 15, "s": 10}),
        (
            "lake_malawi_mbuna_pelagic.tsv",
            {"A_calliptera": 20, "mbuna": 8, "pelagic": 9},
        ),
        (
            "lake_malawi_deep_benthic.tsv",
            {"A_calliptera": 20, "mbuna": 8, "deep_benthic": 10},
        ),
        (
            "lake_malawi_shared.tsv",
            {"A_calliptera": 20, "mbuna": 8, "pelagic": 9, "deep_benthic": 10},
        ),
    ],
)
def test_manifest_contracts(name, expected):
    path = Path(__file__).resolve().parents[1] / "data" / "external_benchmarks" / name
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()[1:]]
    counts = {
        population: sum(row[1] == population for row in rows)
        for population in {row[1] for row in rows}
    }
    assert counts == expected
    assert len({row[0] for row in rows}) == sum(expected.values())


def test_d_site_components_has_expected_abba_baba_orientation():
    assert MODULE.d_site_components(0.0, 1.0, 1.0, 0.0) == (1.0, 0.0)
    assert MODULE.d_site_components(1.0, 0.0, 1.0, 0.0) == (0.0, 1.0)


def test_source_artifact_contracts_are_immutable():
    assert MODULE.SCRUB_JAY["bytes"] == 11_132_546
    assert MODULE.SCRUB_JAY["sha256"] == (
        "04e297ecfe3b5509c9419f0e14f1f7cba16ee493caebf5f7f46a5dcf8faa431a"
    )
    assert MODULE.LAKE_MALAWI["bytes"] == 49_824_023
    assert MODULE.LAKE_MALAWI["sha256"] == (
        "8132246ce809f4f4efa77d174c595a469aa7534c2cb8ee9fbf5472f67202c2b7"
    )


def test_scrub_jay_exact_null_is_not_directional_truth():
    row = MODULE.SCRUB_JAY["author_dsuite_row"]
    assert row["D"] == pytest.approx(0.00619049)
    assert row["Z"] == pytest.approx(0.205384)
    assert row["p"] == pytest.approx(0.418636)
