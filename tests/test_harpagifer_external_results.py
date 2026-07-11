import hashlib
import json
import math
from pathlib import Path


RESULT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "harpagifer_external_benchmark_2026_07_11"
    / "results.json"
)


def _all_floats_finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_floats_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_floats_finite(item) for item in value)
    return True


def test_harpagifer_result_is_clean_guarded_and_finite():
    raw = RESULT.read_bytes()
    result = json.loads(raw)
    assert len(raw) == 163_270
    assert hashlib.sha256(raw).hexdigest() == (
        "6ec8dcd473da847605c6d1500d4c9d4bdb0b3b7f6d50422929c04710e3c329e3"
    )
    assert result["schema_version"] == "dnnaic-harpagifer-external-benchmark-v1"
    assert result["git"] == {
        "commit": "a23796e60d7aac5b12c55e767d5844f2d4541c5c",
        "dirty_at_run": False,
    }
    assert set(result["runtime"]["thread_environment"].values()) == {"1"}

    source = result["source"]
    assert source["verified"]["vcf"]["sha256"] == (
        "7dbc3686e4a24ef36c2a358b55d3d95ff5f2f8d2340087099119335c9b0474a8"
    )
    archive = source["verified"]["archive_wrapper_observation"]
    assert archive["verification_status"] == "observed_only_not_pinned"
    assert source["sources_record"]["content"]["archive"]["digest_policy"].startswith(
        "not_pinned"
    )
    assert "sha256" not in source["sources_record"]["content"]["archive"]
    contract = source["source_vcf_contract"]
    assert contract["samples"] == 118
    assert contract["variant_rows"] == 2_993
    assert contract["genotype_cells"] == 353_174
    assert contract["fully_missing_genotype_cells"] == 34_619
    assert contract["partial_or_invalid_genotype_cells"] == 0
    assert contract["CHROM_values"] == ["0"]
    assert contract["ordered_CHROM_POS_ID_REF_ALT_sha256"] == (
        "d2e7393b4361c3a895453e8ef17a5b9b58c5787c171b03b22d935d10bb765263"
    )
    assert "not be interpreted as ancestral" in contract["allele_orientation_guardrail"]

    mapping = source["mapping_audit"]
    assert mapping["status"] == (
        "reconstructed_from_VCF_column_order_using_published_contiguous_site_order"
    )
    assert mapping["population_counts"] == {
        "FalklandsMalvinas": 25,
        "NorthPatagonia": 50,
        "SouthPatagonia": 43,
    }
    assert len(mapping["known_discrepancies"]) == 3
    assert [block["VCF_n"] for block in mapping["site_blocks"]] == [
        3,
        14,
        15,
        7,
        11,
        15,
        15,
        13,
        25,
    ]
    assert [block["supplement_SNP_n"] for block in mapping["site_blocks"]] == [
        2,
        14,
        15,
        7,
        11,
        15,
        15,
        13,
        25,
    ]

    manifests = source["manifest_audit"]
    assert manifests["all_released_samples"]["population_counts"] == {
        "FalklandsMalvinas": 25,
        "NorthPatagonia": 50,
        "SouthPatagonia": 43,
    }
    assert manifests["sample_missingness_le_0_25"]["population_counts"] == {
        "FalklandsMalvinas": 25,
        "NorthPatagonia": 40,
        "SouthPatagonia": 42,
    }
    assert manifests["sample_missingness_sensitivity"][
        "excluded_samples_in_VCF_order"
    ] == [
        "HBi_034",
        "HBi_038",
        "HBi_040",
        "HBi_026",
        "HBi_029",
        "HBi_031",
        "HBi_014",
        "HBi_107",
        "HBi_115",
        "HBi_126",
        "HBi_313",
    ]

    outcome = result["outcome_summary"]
    assert outcome["analytic_sensitivity_runs"] == 4
    assert outcome["unique_biological_systems"] == 1
    assert outcome["candidate_comparisons"] == 1
    assert outcome["independent_validation_panels"] == 0
    assert outcome["accuracy_available"] is False
    assert outcome["accuracy_estimate"] is None
    assert outcome["candidate_label_status"] == (
        "literature_dominant_same_SNP_sensitivity"
    )
    assert outcome["severe_OOD_panels"] == outcome["abstained_panels"] == 4
    assert outcome["raw_OOD_head_prediction_counts"] == {"A": 4, "B": 0, "C": 0}
    assert outcome["raw_OOD_gate_threshold_crossings_at_0.5"] == 4

    expected_filters = {
        "standard_contract": (
            2_993,
            "a72e91972c32a39363d2c6133f8822cde43ead12f97931b89d05abbee5ad76a6",
        ),
        "within_population_polymorphism": (
            2_977,
            "c4a2a74166e672091fd5bc3f17404f93369f6d495ea060016e47c7a316ba0ac5",
        ),
    }
    observed_hashes = {name: set() for name in expected_filters}
    projections = []
    corrected_f3 = []
    for panel in result["panels"]:
        expectation = panel["external_expectation"]
        assert expectation["candidate_class"] == "A"
        assert expectation["expected_gate"] is None
        assert expectation["accuracy_eligible"] is False
        assert panel["simulation_head"]["predicted_class"] == "A"
        assert panel["simulation_gate"]["appreciable_score"] == 1.0
        adjudication = panel["adjudication"]
        assert adjudication["accuracy_eligible"] is False
        assert adjudication["severe_OOD"] is True
        assert adjudication["natural_data_call_status"] == "abstain_severe_OOD"
        assert "heuristic" in adjudication["severe_OOD_rule"]

        filter_name = (
            "within_population_polymorphism"
            if panel["panel_id"].endswith("within_population_polymorphism")
            else "standard_contract"
        )
        loci, digest = expected_filters[filter_name]
        assert panel["padze"]["n_loci_kept"] == loci
        assert panel["input_audit"]["ordered_locus_sha256"] == digest
        observed_hashes[filter_name].add(digest)

        geometry = panel["model_free_comparator"]
        assert geometry["diagnostic_loci"] == 0
        assert geometry["P2_projection_from_P1_toward_P3_diagnostic_loci"] is None
        assert geometry["iid_locus_bootstrap"]["projection_diagnostic_loci"] is None
        assert "not chromosome-block" in geometry["iid_locus_bootstrap"]["guardrail"]
        interval = geometry["iid_locus_bootstrap"][
            "f3_finite_called_copy_corrected"
        ]["percentile_95_interval"]
        assert interval[0] < 0 < interval[1]
        projections.append(geometry["P2_projection_from_P1_toward_P3_all_loci"])
        corrected_f3.append(geometry["f3_P2_P1_P3_finite_called_copy_corrected"])

    assert all(len(hashes) == 1 for hashes in observed_hashes.values())
    assert projections == [
        0.22713767698150844,
        0.22990693418477878,
        0.22744298452106082,
        0.23072153021785027,
    ]
    assert min(corrected_f3) > 0
    assert _all_floats_finite(result)
