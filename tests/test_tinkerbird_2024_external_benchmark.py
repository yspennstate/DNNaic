import json
from pathlib import Path

import numpy as np
import pytest

from scripts import tinkerbird_2024_external_benchmark as tinkerbird


def test_2024_source_contract_and_urls_are_pinned():
    assert tinkerbird.FILES["vcf"]["bytes"] == 148_766_627
    assert tinkerbird.FILES["vcf"]["sha256"] == (
        "6438a889ad91b865237e6a4e5169bfbe61e9eea884696fa7fdcfef83d68c7c30"
    )
    assert tinkerbird.FILES["female_metadata"]["md5"] == (
        "c982c71fa9ebfa9c2dedf11e0d0877db"
    )
    assert tinkerbird.FILES["master_metadata"]["bytes"] == 53_230
    assert tinkerbird.SOURCE_CONTRACT == {
        "samples": 452,
        "variant_rows": 84_112,
        "anchored_autosomes": 82_309,
        "Z": 1_157,
        "W": 5,
        "S76_non_numbered": 31,
        "unplaced_scaffolds": 610,
        "autosomal_RAD_loci": 23_395,
        "one_per_RAD_ordered_key_sha256": (
            "a78ee8f7f57e81745d7f745d3c600435ed800c1611e5bf9a1fe57468253200ef"
        ),
        "paper_reported_SNPs": 82_950,
    }


def test_2024_manifest_contracts_are_exact_and_disjoint():
    audit = tinkerbird.audit_manifests()
    assert audit["counts"]["direction_holdout"] == {
        "ExtoniRefExact23": 23,
        "MpofuChrysoconusMale9": 9,
        "PusillusRefExact8": 8,
    }
    assert audit["counts"]["direction_direct"] == {
        "ExtoniRefExact23": 23,
        "ExtoniMotherDaughter14": 14,
        "PusillusRefExact8": 8,
    }
    assert audit["counts"]["direction_females_95"] == 95
    assert audit["sample_disjoint_holdout"] is True
    assert audit["direction_union_subgroup_contract"] is True
    assert audit["direct_panel_subset_of_direction_females"] is True
    assert audit["recent_cross_panel_subset_of_direction_females"] is True


def test_autosome_and_stacks_RAD_locus_contracts():
    assert tinkerbird.is_autosome("SUPER_1")
    assert tinkerbird.is_autosome("SUPER_44")
    assert not tinkerbird.is_autosome("SUPER_Z")
    assert not tinkerbird.is_autosome("SUPER_W")
    assert not tinkerbird.is_autosome("S76")
    assert tinkerbird.rad_locus_id("880:483:-", "SUPER_9", "150942") == "880"
    for malformed in (".", "880", "880:483", "abc:483:-", "880:483:?"):
        with pytest.raises(ValueError, match="expected Stacks"):
            tinkerbird.rad_locus_id(malformed, "SUPER_9", "150942")


def test_one_per_RAD_locus_keeps_first_source_ordered_snp(tmp_path: Path):
    source = tmp_path / "source.vcf"
    output = tmp_path / "thinned.vcf"
    source.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
        "SUPER_1\t10\t880:1:-\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
        "SUPER_1\t11\t880:2:-\tC\tT\t.\tPASS\t.\tGT\t0/1\n"
        "SUPER_2\t20\t901:1:-\tG\tA\t.\tPASS\t.\tGT\t0/1\n",
        encoding="utf-8",
    )
    audit = tinkerbird.thin_one_per_rad_locus(
        source,
        output,
        expected_source_rows=None,
        expected_RAD_loci=None,
        expected_ordered_key_sha256=None,
    )
    variants = [
        line for line in output.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    assert len(variants) == 2
    assert [line.split("\t")[1] for line in variants] == ["10", "20"]
    assert audit["RAD_loci"] == 2
    assert audit["ordered_key_sha256"] == (
        "bf12a4ab1f0064af184fea3f932d099cf1df82393f56469fba18d5a3bb5c1796"
    )


def _write_toy_panel(
    tmp_path: Path,
    rows: list[tuple[str, str, str]],
    sample_order: tuple[str, str, str] = ("s1", "s2", "s3"),
) -> tuple[Path, Path]:
    manifest = tmp_path / "panel.tsv"
    manifest.write_text(
        "sample\tpopulation\n"
        "s1\tP1\n"
        "s2\tP2\n"
        "s3\tP3\n",
        encoding="utf-8",
    )
    by_sample = {
        sample: [row[index] for row in rows]
        for index, sample in enumerate(("s1", "s2", "s3"))
    }
    vcf = tmp_path / "panel.vcf"
    lines = [
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(sample_order),
    ]
    for index in range(len(rows)):
        genotypes = [by_sample[sample][index] for sample in sample_order]
        lines.append(
            f"SUPER_{index + 1}\t{index + 1}\t{100 + index}:1:-\tA\tG\t.\tPASS\t.\tGT\t"
            + "\t".join(genotypes)
        )
    vcf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return vcf, manifest


def test_frequency_geometry_is_reference_and_header_invariant(tmp_path: Path):
    rows = [("0/0", "0/1", "1/1"), ("0/0", "0/1", "1/1")]
    vcf, manifest = _write_toy_panel(tmp_path, rows)
    result = tinkerbird.frequency_geometry(
        vcf, manifest, ("P1", "P2", "P3"), bootstrap_replicates=20
    )
    assert np.isclose(result["P2_projection_from_P1_toward_P3_all_loci"], 0.5)
    assert np.isclose(result["f3_like_plugin_P2_P1_P3"], -0.25)
    assert np.isclose(result["mean_P2_finite_called_copy_correction"], 0.25)
    assert np.isclose(result["f3_P2_P1_P3_finite_called_copy_corrected"], -0.5)
    assert result["chromosome_block_bootstrap"]["chromosome_blocks"] == 2
    assert "label reuse" not in result["chromosome_block_bootstrap"]["guardrail"]

    flipped_dir = tmp_path / "flipped"
    flipped_dir.mkdir()
    flipped, flipped_manifest = _write_toy_panel(
        flipped_dir,
        [("1/1", "1/0", "0/0"), ("1/1", "1/0", "0/0")],
        sample_order=("s3", "s1", "s2"),
    )
    other = tinkerbird.frequency_geometry(
        flipped,
        flipped_manifest,
        ("P1", "P2", "P3"),
        bootstrap_replicates=20,
    )
    assert np.isclose(
        other["P2_projection_from_P1_toward_P3_all_loci"],
        result["P2_projection_from_P1_toward_P3_all_loci"],
    )
    assert np.isclose(
        other["f3_P2_P1_P3_finite_called_copy_corrected"],
        result["f3_P2_P1_P3_finite_called_copy_corrected"],
    )


def test_frequency_geometry_serializes_empty_diagnostic_as_null(tmp_path: Path):
    vcf, manifest = _write_toy_panel(
        tmp_path, [("0/0", "0/1", "0/1"), ("0/0", "0/1", "0/1")]
    )
    result = tinkerbird.frequency_geometry(
        vcf, manifest, ("P1", "P2", "P3"), bootstrap_replicates=20
    )
    assert result["diagnostic_loci"] == 0
    assert result["P2_projection_from_P1_toward_P3_diagnostic_loci"] is None
    assert result["chromosome_block_bootstrap"]["projection_diagnostic_loci"] is None
    json.dumps(result, allow_nan=False)


def test_scope_roles_and_cross_family_overlap_are_explicit(tmp_path: Path):
    assert tinkerbird.SCOPE_ROLES["autosome_one_per_RAD_locus"].startswith("primary:")
    assert "pseudoreplication" in tinkerbird.SCOPE_ROLES["autosome_all_snps"]
    paths = {}
    for family in ("direction", "geographic", "recent"):
        family_dir = tmp_path / family
        family_dir.mkdir()
        paths[family], _ = _write_toy_panel(
            family_dir,
            [("0/0", "0/1", "1/1"), ("0/0", "0/1", "1/1")],
        )
    overlap = tinkerbird.cross_family_locus_overlap(paths)
    assert overlap["family_loci"] == {
        "direction": 2,
        "geographic": 2,
        "recent": 2,
    }
    assert overlap["all_three_intersection"] == 2
    assert overlap["comparison_eligible"] is False
    paired = tinkerbird.pair_locus_overlap(
        paths["direction"], paths["geographic"], "primary", "sensitivity"
    )
    assert paired["intersection"] == 2
    assert paired["primary_loci"] == 2
    assert paired["sensitivity_loci"] == 2
    assert paired["comparison_eligible"] is False


def test_outcome_summary_abstains_instead_of_reporting_accuracy():
    panels = [
        {
            "external_expectation": {
                "benchmark_role": "sample_disjoint_direction_holdout",
                "candidate_class": "C",
            },
            "simulation_head": {"predicted_class": "A"},
            "simulation_gate": {"appreciable_score": 1.0, "called_at_0.5": True},
            "adjudication": {"matches_candidate_majority": False, "severe_OOD": True},
        },
        {
            "external_expectation": {
                "benchmark_role": "gate_near_null",
                "candidate_class": None,
            },
            "simulation_head": {"predicted_class": "B"},
            "simulation_gate": {"appreciable_score": 0.99, "called_at_0.5": True},
            "adjudication": {"matches_candidate_majority": None, "severe_OOD": True},
        },
    ]
    summary = tinkerbird.summarize_panel_outcomes(panels)
    assert summary["severe_OOD_panels"] == 2
    assert summary["abstain_due_to_severe_OOD"] is True
    assert summary["accuracy_estimate"] is None
    assert summary["direction_candidate_panels"]["matches_candidate"] == 0
    assert summary["gate_near_null_panels"]["called_appreciable_at_0.5"] == 1
