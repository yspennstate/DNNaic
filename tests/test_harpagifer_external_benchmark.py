import json
from pathlib import Path

import numpy as np

from scripts import harpagifer_external_benchmark as harp


def test_harpagifer_source_and_reconstruction_contracts_are_guarded():
    assert harp.FILE == {
        "id": 1_231_632,
        "key": "Hbi_Hpal_118_2993_6May19_GEO.vcf",
        "archive_member": "Hbi_Hpal_118_2993_6May19_GEO.vcf",
        "download": "https://datadryad.org/api/v2/files/1231632/download",
        "bytes": 1_517_411,
        "sha256": "7dbc3686e4a24ef36c2a358b55d3d95ff5f2f8d2340087099119335c9b0474a8",
    }
    assert "sha256" not in harp.ARCHIVE
    assert harp.ARCHIVE["digest_policy"].startswith("not_pinned")
    record = harp.validate_sources_record()
    assert record["version_id"] == 156_912
    assert record["archive"]["digest_policy"] == harp.ARCHIVE["digest_policy"]
    assert record["file"]["sha256"] == harp.FILE["sha256"]
    assert record["supplement_file"]["md5"] == "52c023123a8e9eeda9f7743f3b3a7466"
    assert record["mapping_provenance"]["status"] == (
        "reconstructed_from_VCF_column_order_using_published_contiguous_site_order"
    )
    assert len(record["mapping_provenance"]["known_discrepancies"]) == 3

    blocks = harp.read_blocks()
    assert [row["site"] for row in blocks] == [
        "TEM",
        "FP",
        "CSB",
        "PB",
        "IC3",
        "PY",
        "FPI",
        "PW",
        "HP",
    ]
    assert sum(
        int(row["end_exclusive"]) - int(row["start_zero_based"])
        for row in blocks
    ) == 118
    assert sum(int(row["supplement_SNP_n"]) for row in blocks) == 117
    assert sum(harp.MAIN_TABLE3_SAMPLE_COUNTS) == 128
    assert blocks[0]["site"] == "TEM"
    assert int(blocks[0]["end_exclusive"]) == 3
    assert int(blocks[0]["supplement_SNP_n"]) == 2
    assert harp.SOURCE_CONTRACT["locus_semantic_sha256"] == (
        "d2e7393b4361c3a895453e8ef17a5b9b58c5787c171b03b22d935d10bb765263"
    )
    assert harp.EXPECTED_FILTERS["standard_contract"]["loci"] == 2_993
    assert harp.EXPECTED_FILTERS["within_population_polymorphism"]["loci"] == 2_977


def _write_toy_panel(tmp_path: Path):
    manifest = tmp_path / "panel.tsv"
    manifest.write_text(
        "sample\tpopulation\n"
        "s1\tNorthPatagonia\n"
        "s2\tSouthPatagonia\n"
        "s3\tFalklandsMalvinas\n",
        encoding="utf-8",
    )
    vcf = tmp_path / "panel.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
        "0\t1\tT0_1\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t0/1\n"
        "0\t2\tT0_2\tC\tT\t.\tPASS\t.\tGT\t0/1\t0/0\t1/1\n",
        encoding="utf-8",
    )
    return vcf, manifest


def test_harpagifer_geometry_is_flip_invariant_and_zero_diagnostic_is_null(
    tmp_path: Path,
):
    vcf, manifest = _write_toy_panel(tmp_path)
    result = harp.frequency_geometry(vcf, manifest, bootstrap_replicates=20)
    assert result["n_loci"] == 2
    assert result["maximum_abs_P3_minus_P1_frequency"] == 0.5
    assert result["diagnostic_loci"] == 0
    assert result["P2_projection_from_P1_toward_P3_diagnostic_loci"] is None
    assert result["iid_locus_bootstrap"]["projection_diagnostic_loci"] is None
    assert "not chromosome-block" in result["iid_locus_bootstrap"]["guardrail"]
    json.dumps(result, allow_nan=False)

    flipped = tmp_path / "flipped.vcf"
    flipped.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts3\ts1\ts2\n"
        "0\t1\tT0_1\tG\tA\t.\tPASS\t.\tGT\t1/0\t1/1\t1/0\n"
        "0\t2\tT0_2\tT\tC\t.\tPASS\t.\tGT\t0/0\t1/0\t1/1\n",
        encoding="utf-8",
    )
    other = harp.frequency_geometry(flipped, manifest, bootstrap_replicates=20)
    assert np.isclose(
        other["P2_projection_from_P1_toward_P3_all_loci"],
        result["P2_projection_from_P1_toward_P3_all_loci"],
    )
    assert np.isclose(
        other["f3_P2_P1_P3_finite_called_copy_corrected"],
        result["f3_P2_P1_P3_finite_called_copy_corrected"],
    )


def test_harpagifer_summary_never_counts_sensitivities_as_accuracy():
    panels = []
    for prediction in ("A", "A", "B", "C"):
        panels.append(
            {
                "adjudication": {
                    "natural_data_call_status": "abstain_severe_OOD",
                    "severe_OOD": True,
                    "matches_candidate_reference": prediction == "A",
                },
                "simulation_head": {"predicted_class": prediction},
                "simulation_gate": {"called_at_0.5": True},
            }
        )
    summary = harp.summarize_outcomes(panels)
    assert summary["analytic_sensitivity_runs"] == 4
    assert summary["unique_biological_systems"] == 1
    assert summary["candidate_comparisons"] == 1
    assert summary["independent_validation_panels"] == 0
    assert summary["accuracy_estimate"] is None
    assert summary["accuracy_available"] is False
    assert summary["abstained_panels"] == 4
    assert summary["raw_OOD_head_matches_candidate_A"] == 2
