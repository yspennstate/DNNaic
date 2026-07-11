from pathlib import Path

import numpy as np

from scripts import seabass_external_benchmark as seabass
from scripts.external_benchmarks import read_manifest


def test_seabass_source_contract_is_pinned():
    assert seabass.FILES["ped"]["bytes"] == 3_412_429
    assert seabass.FILES["ped"]["sha256"] == (
        "9e199dc329e6bd822888cb14ff9563a18d0a0ef2ca241f9c827cfb623ce670e1"
    )
    assert seabass.FILES["ancestry"]["bytes"] == 29_166
    assert seabass.FILES["summary"]["sha256"] == (
        "38026d593814732cdafff0f5132fbf4d2a13dbcd460636946ed53f86c4a48e0f"
    )


def test_seabass_manifests_are_exact_and_matched():
    union = read_manifest(seabass.UNION_MANIFEST, require_three=False)
    positive = read_manifest(seabass.POSITIVE_MANIFEST)
    control = read_manifest(seabass.CONTROL_MANIFEST)
    counts = {label: list(union.values()).count(label) for label in set(union.values())}
    assert counts == {"PENI": 29, "SINE": 27, "VIGO": 30, "MED": 10}
    assert len(union) == 96
    assert set(positive) == {
        sample for sample, label in union.items() if label in {"PENI", "SINE", "MED"}
    }
    assert set(control) == {
        sample for sample, label in union.items() if label in {"PENI", "VIGO", "MED"}
    }
    assert {sample for sample, label in positive.items() if label == "PENI"} == {
        sample for sample, label in control.items() if label == "PENI"
    }
    assert {sample for sample, label in positive.items() if label == "MED"} == {
        sample for sample, label in control.items() if label == "MED"
    }


def test_ped_map_conversion_uses_nominal_alleles_and_iids(tmp_path: Path):
    map_path = tmp_path / "toy.map"
    ped_path = tmp_path / "toy.ped"
    vcf_path = tmp_path / "toy.vcf"
    map_path.write_text("1 rs1 0 10\n25 rs2 1 20\n", encoding="ascii")
    ped_path.write_text(
        "S1 S1 0 0 1 -9 A G 0 0\n"
        "S2 S2 0 0 2 -9 A A C C\n"
        "S3 S3 0 0 1 -9 G G C T\n",
        encoding="ascii",
    )
    audit = seabass.convert_ped_map_to_vcf(
        ped_path,
        map_path,
        vcf_path,
        expected_samples=3,
        expected_markers=2,
        expected_sha256=None,
    )
    lines = vcf_path.read_text(encoding="ascii").splitlines()
    assert audit["missing_genotypes"] == 1
    assert lines[4].endswith("\tS1\tS2\tS3")
    assert lines[5].split("\t")[3:5] == ["A", "G"]
    assert lines[5].split("\t")[-3:] == ["0/1", "0/0", "1/1"]
    assert lines[6].split("\t")[0:5] == ["25", "20", "rs2", "C", "T"]
    assert lines[6].split("\t")[-3:] == ["./.", "0/0", "0/1"]


def test_frequency_projection_recovers_halfway_panel(tmp_path: Path):
    manifest = tmp_path / "panel.tsv"
    manifest.write_text("s1\tP1\ns2\tP2\ns3\tP3\n", encoding="utf-8")
    vcf = tmp_path / "panel.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
        "1\t1\trs1\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\n"
        "1\t2\trs2\tC\tT\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\n",
        encoding="utf-8",
    )
    result = seabass.frequency_projection(vcf, manifest, ("P1", "P2", "P3"))
    assert result["n_loci"] == 2
    assert np.isclose(result["P2_projection_from_P1_toward_P3"], 0.5)
    assert result["mean_squared_frequency_distance"] == {
        "P1_P2": 0.25,
        "P2_P3": 0.25,
        "P1_P3": 1.0,
    }
