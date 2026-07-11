import json

import pytest

from dnnaic.semantics import class_for_forward_edge
from scripts import dingo_weeks_2025_external_benchmark as dingo


def test_dingo_source_panel_and_locus_contracts_are_pinned():
    assert dingo.FILES["vcf"] == {
        "id": 49_199_059,
        "key": "Weeks_etal_434.vcf.gz",
        "url": "https://ndownloader.figshare.com/files/49199059",
        "bytes": 5_266_043,
        "md5": "cc3a6f753726c289cbcb84842ba1ed80",
        "sha256": "42620f03b3768dc71617f198372a09ff4d42e4654d30c79fc960b972ac7b8125",
    }
    assert dingo.FILES["metadata"] == {
        "id": 49_199_065,
        "key": "Weeks_meta_434.txt",
        "url": "https://ndownloader.figshare.com/files/49199065",
        "bytes": 9_848,
        "md5": "3e77e058c1f3dd7aee1ad832ef382e2f",
        "sha256": "d7cc4fe11dc36d56a52221d648ecb4da1cab878db42e0abfc324ffc139315acb",
    }
    assert dingo.FILES["geolocation"] == {
        "id": 49_199_062,
        "key": "Weeks_meta_geolocation_434.txt",
        "url": "https://ndownloader.figshare.com/files/49199062",
        "bytes": 16_212,
        "md5": "d6ac2d30ed054378f4098d6b8f19c516",
        "sha256": "4ee68f300e14674f771a85264cbdd281033b9ec0292862738c6b993a736ce821",
    }
    assert dingo.METADATA_COUNTS == {
        "alpine": 248,
        "back": 8,
        "desert": 58,
        "dog": 39,
        "hybrid": 7,
        "mallee": 74,
    }
    assert dingo.PANEL_COUNTS == {"P1": 248, "P2": 8, "P3": 39}
    assert dingo.PAPER_COUNTS == {
        "alpine": 248,
        "back": 8,
        "desert": 74,
        "dog": 39,
        "hybrid": 7,
        "mallee": 58,
    }
    assert dingo.GEOLOCATION_COUNTS == {"Alpine": 248, "Desert": 77, "Mallee": 55, "NIL": 3}
    assert dingo.SOURCE_FILTER_COUNTS == {
        "PASS": 2_233,
        "VQSRTrancheSNP99.00to99.90": 125,
        "VQSRTrancheSNP99.90to100.00": 108,
    }
    assert dingo.EXPECTED_FILTER_LOCI == {
        ("paper_release_all_rows", "standard_contract"): 2_193,
        ("paper_release_all_rows", "within_population_polymorphism"): 1_594,
        ("vcf_PASS_only", "standard_contract"): 1_992,
        ("vcf_PASS_only", "within_population_polymorphism"): 1_551,
    }
    assert dingo.SOURCE_ELIGIBILITY["all_release_rows"] == {
        "minimum_16_called_copies": 2_201,
        "pooled_polymorphic": 2_193,
        "within_each_population_polymorphic": 1_594,
    }
    assert dingo.SOURCE_ELIGIBILITY["vcf_PASS_only"] == {
        "minimum_16_called_copies": 1_999,
        "pooled_polymorphic": 1_992,
        "within_each_population_polymorphic": 1_551,
    }
    assert dingo.NORMALISED_ALLPASS == {
        "bytes": 39_485_888,
        "sha256": "c15274389e2601ae230e3092684f65aa01d158f2b33d00ea7961f2dd345ac9c6",
    }
    assert dingo.PANEL_MANIFEST_BYTES == 3_245
    assert dingo.PANEL_MANIFEST_SHA256 == (
        "12eed285869fa3426b82442ebbb44e866af77cf6013c7853776d421d20f23d45"
    )
    assert dingo.PUBLISHED_FST == {"P1_P2": 0.07, "P2_P3": 0.17, "P1_P3": 0.27}
    assert class_for_forward_edge("P3", "P2") == "C"


def test_dingo_sources_record_preserves_discrepancies_and_guardrails():
    validated = dingo.validate_sources_record()
    record = validated["record"]
    assert {key: validated[key] for key in ("bytes", "sha256")} == dingo.SOURCE_RECORD_CONTRACT
    assert record["data_doi"] == "10.6084/m9.figshare.27022555.v1"
    assert record["license"] == "CC-BY-4.0"
    assert record["paper"]["doi"] == "10.1093/evlett/qrae057"
    assert record["paper"]["online_publication_date"] == "2024-10-19"
    assert record["paper"]["issue_date"] == "2025-02"
    assert record["analysis_design"]["candidate_class"] == "C"
    assert record["analysis_design"]["pedigree_dog_introgression_component_available"] is True
    assert record["analysis_design"]["exclusive_single_edge_truth_available"] is False
    assert record["analysis_design"]["formal_direction_accuracy_eligible"] is False
    assert len(record["release_discrepancies"]) == 3
    assert "desert=58 and mallee=74" in record["release_discrepancies"][0]
    assert "Desert=77, Mallee=55" in record["release_discrepancies"][1]
    assert "233 inherited non-PASS" in record["release_discrepancies"][2]
    assert all(
        group not in record["analysis_design"][role]
        for role in ("P1", "P2", "P3")
        for group in ("desert", "mallee")
    )


def _panel(prediction: str, direction_rms: float, gate_rms: float) -> dict:
    return {
        "simulation_head": {"predicted_class": prediction},
        "simulation_feature_shift": {"rms_z": direction_rms},
        "simulation_gate_feature_shift": {"rms_z": gate_rms},
    }


@pytest.mark.parametrize("prediction", ["A", "B", "C"])
@pytest.mark.parametrize("direction_rms,severe", [(10.0, False), (10.000001, True)])
def test_dingo_adjudication_never_turns_one_backcross_cohort_into_accuracy(
    prediction, direction_rms, severe
):
    result = dingo.adjudicate_panel(
        _panel(prediction, direction_rms, 1.0)
    )
    assert result["natural_data_call_status"] == (
        "abstain_severe_OOD" if severe else "descriptive_candidate_concordance_only"
    )
    assert result["pedigree_dog_introgression_component_available"] is True
    assert result["exclusive_single_edge_truth_available"] is False
    assert result["formal_direction_accuracy_eligible"] is False
    assert result["raw_head_matches_candidate_C"] is (prediction == "C")
    assert result["direction_call_accepted"] is False
    assert "accepted_direction_call" not in result
    assert "accepted_call_matches_candidate_C" not in result
    assert "heuristic diagnostic, not calibrated support" in result["severe_OOD_rule"]
    assert result["gate_truth_available"] is False
    assert result["gate_accuracy_eligible"] is False


def test_dingo_summary_counts_filters_as_sensitivities_not_trials():
    panels = []
    for prediction, direction_rms in (("A", 18.0), ("A", 19.0), ("C", 2.0), ("B", 3.0)):
        panel = _panel(prediction, direction_rms, 1.0)
        panel["adjudication"] = dingo.adjudicate_panel(panel)
        panels.append(panel)
    outcome = dingo.summarize_outcomes(panels)
    assert outcome["analytic_filter_sensitivity_runs"] == 4
    assert outcome["unique_biological_systems"] == 1
    assert outcome["independent_pedigree_dog_component_units"] == 1
    assert outcome["exclusive_single_edge_truth_units"] == 0
    assert outcome["independent_sample_level_units"] == 0
    assert outcome["correlated_filter_sensitivities_not_trials"] is True
    assert outcome["raw_head_prediction_counts"] == {"A": 2, "B": 1, "C": 1}
    assert outcome["severe_OOD_panels"] == 2
    assert outcome["descriptive_nonsevere_panels"] == 2
    assert outcome["accepted_direction_calls"] == 0
    assert outcome["direction_accuracy_estimate"] is None
    assert outcome["gate_accuracy_estimate"] is None
    json.dumps(outcome, allow_nan=False)
