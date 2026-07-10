from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "directional_external_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("directional_external_benchmarks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "ciona_jersey_southampton_robusta.tsv",
            {"Jer12": 16, "Sth12": 16, "CioAB": 16},
        ),
        (
            "ciona_jersey_poole_robusta.tsv",
            {"Jer12": 16, "Poo12": 16, "CioAB": 16},
        ),
        (
            "ciona_shared.tsv",
            {"Jer12": 16, "Sth12": 16, "Poo12": 16, "CioAB": 16},
        ),
    ],
)
def test_ciona_manifest_contracts(name, expected):
    path = Path(__file__).resolve().parents[1] / "data" / "external_benchmarks" / name
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()[1:]]
    counts = {
        population: sum(row[1] == population for row in rows)
        for population in {row[1] for row in rows}
    }
    assert counts == expected
    assert len({row[0] for row in rows}) == sum(expected.values())


def test_subset_region_is_inclusive_and_preserves_header(tmp_path):
    source = tmp_path / "source.vcf"
    source.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
        "chromosome5\t9\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
        "chromosome5\t10\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
        "chromosome5\t20\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
        "chromosome5\t21\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
        "chromosome4\t15\t.\tA\tG\t.\tPASS\t.\tGT\t0/1\n",
        encoding="utf-8",
    )
    output = tmp_path / "region.vcf"
    audit = MODULE.subset_region(source, output, "chromosome5", 10, 20)
    rows = [line for line in output.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    assert [int(row.split("\t")[1]) for row in rows] == [10, 20]
    assert audit["source_rows"] == 5
    assert audit["retained_rows"] == 2


def test_thin_one_per_id_prefix_keeps_first_source_ordered_snp(tmp_path):
    source = tmp_path / "source.vcf"
    source.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\n"
        "chromosome5\t10\ttagA:1:+\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
        "chromosome5\t11\ttagA:2:+\tA\tC\t.\tPASS\t.\tGT\t0/1\n"
        "chromosome5\t20\ttagB:1:-\tA\tG\t.\tPASS\t.\tGT\t0/1\n",
        encoding="utf-8",
    )
    output = tmp_path / "thinned.vcf"
    audit = MODULE.thin_one_per_id_prefix(source, output)
    rows = [line for line in output.read_text(encoding="utf-8").splitlines() if not line.startswith("#")]
    assert [row.split("\t")[2] for row in rows] == ["tagA:1:+", "tagB:1:-"]
    assert audit["source_rows"] == 3
    assert audit["retained_rows"] == 2
    assert audit["unique_prefixes"] == 2


def test_frequency_sharing_comparator_orientation(tmp_path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "sample\tpopulation\nJ\tJer12\nS\tSth12\nP\tPoo12\nC\tCioAB\n",
        encoding="utf-8",
    )
    vcf = tmp_path / "shared.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tJ\tS\tP\tC\n"
        "chromosome5\t10\ttagA:1:+\tA\tG\t.\tPASS\t.\tGT\t0/0\t1/1\t0/0\t1/1\n",
        encoding="utf-8",
    )
    result = MODULE.frequency_sharing_comparator(vcf, manifest)
    assert result["n_loci"] == 1
    assert result["unpolarized_reference_invariant_f4"] == -1.0
    assert result["mean_squared_frequency_distance_to_CioAB"] == {
        "Jer12": 1.0,
        "Sth12": 0.0,
        "Poo12": 1.0,
    }


def test_ciona_source_and_direction_contract():
    assert MODULE.CIONA["bytes"] == 109_974_779
    assert MODULE.CIONA["sha256"] == (
        "e0a3586c11a65f5d0419b08d14827fd6cf61d2e2705dd74340137dee310a761c"
    )
    assert MODULE.CIONA["figure3_window"] == {
        "chromosome": "chromosome5",
        "start": 500_000,
        "end": 2_000_000,
        "basis": "chromosome-5 interval used for the source Figure 3 ancestry/tree analysis",
    }
    assert MODULE.CIONA["southampton_exact_interval"] == {
        "chromosome": "chromosome5",
        "start": 661_065,
        "end": 1_174_846,
        "basis": "pooled Southampton HMM-positive interval reported in source Table S4",
    }
