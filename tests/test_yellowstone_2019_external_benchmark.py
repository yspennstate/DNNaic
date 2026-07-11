import json

import pytest

from dnnaic.semantics import class_for_forward_edge
from scripts import yellowstone_2019_external_benchmark as yellowstone


def test_yellowstone_source_and_release_contracts_are_pinned():
    assert yellowstone.FILES["vcf"] == {
        "key": "variants_0.6_maf0.05.recode.vcf.gz",
        "copied_file_id": 109_334,
        "created_file_id": 109_343,
        "url": "https://datadryad.org/api/v2/files/109343/download",
        "bytes": 47_687_900,
        "md5": "ddfa5ab0307957daaea5f9ad5c4afe19",
        "sha256": "c0341a3a9dc11206e460907bea3d618f535245c7993d9cfce39c12c2b9b8bc86",
    }
    assert yellowstone.FILES["metadata"]["sha256"] == (
        "05d71fb77dac9cf61f80a54f4ac8c8fdf79f52423a9fd1e5d9b21f299cd60f46"
    )
    assert yellowstone.FILES["predictors"]["sha256"] == (
        "a9663d2a8922e3fdace4b90598becf8991ca57a11be626bc353e87bc89d8abfb"
    )
    assert yellowstone.FILES["response"]["sha256"] == (
        "01e4e84e18ae21aa88ed9c85ccd3ee2148b4e522b2170d3c110ad72433e3b9b1"
    )
    assert yellowstone.FILES["converter"]["sha256"] == (
        "f68cba59fc020cf59f97f021a7385f0cb03c0ade1c922aa3fbc95126e80fa0bc"
    )
    assert yellowstone.SOURCE_VCF_CONTRACT["samples"] == 1_286
    assert yellowstone.SOURCE_VCF_CONTRACT["variants"] == 12_666
    assert yellowstone.SOURCE_VCF_CONTRACT["chromosomes"] == 29
    assert yellowstone.SOURCE_VCF_CONTRACT["ordered_locus_sha256"] == (
        "12b9a0f39fe7a3a12db057ce78e913a6848e5a5e7b5819843a0e0169c6643f59"
    )
    assert yellowstone.CONVERTER_CELL_CONTRACT == {
        "cells": 16_288_476,
        "converter_zeroed_informative_PL_cells": 1_789_601,
        "source_GT_called": 11_399_315,
        "source_GT_missing": 4_889_161,
        "unique_PL_argmin_called": 13_170_198,
        "unique_PL_argmin_with_DP_positive": 13_170_198,
    }


def test_yellowstone_sources_record_preserves_truth_and_archive_guardrails():
    validated = yellowstone.validate_sources_record()
    assert validated["canonical_lf"] == yellowstone.SOURCE_RECORD_CANONICAL_LF_CONTRACT
    record = validated["record"]
    assert record["data_doi"] == "10.5061/dryad.6s7d02q"
    assert record["version_id"] == 29_052
    assert record["license"] == "CC0-1.0"
    assert record["paper"]["doi"] == "10.1111/mec.15175"
    assert record["paper"]["corrigendum_doi"] == "10.1111/mec.15381"
    design = record["analysis_design"]
    assert design["candidate_class"] == "C"
    assert design["direction_truth_available"] is False
    assert design["exclusive_single_edge_truth_available"] is False
    assert design["formal_direction_accuracy_eligible"] is False
    assert design["gate_truth_available"] is False
    assert len(record["release_discrepancies"]) == 4
    assert "every one of nine paths twice" in record["release_discrepancies"][0]
    assert "Per-individual Entropy q/Q" in record["release_discrepancies"][1]
    assert "0 0 0" in record["release_discrepancies"][2]
    assert "Corrigendum DOI" in record["release_discrepancies"][3]
    assert class_for_forward_edge("P3", "P2") == "C"


def test_yellowstone_sample_normalization_and_genotype_decoders_are_explicit():
    assert yellowstone.normalize_vcf_sample(
        "/project/WagnerLab/emandevi/YSCxRBT_march2017/bwa_assem/aln_EGM16_0909.sorted.bam"
    ) == "EGM16_0909"
    assert yellowstone.normalize_vcf_sample(r"C:\data\aln_EGM1964_001.sorted.bam") == "EGM1964_001"
    with pytest.raises(ValueError, match="unexpected Yellowstone"):
        yellowstone.normalize_vcf_sample("sample.bam")

    assert yellowstone.decode_source_gt("0/0") == "0/0"
    assert yellowstone.decode_source_gt("1|0") == "0/1"
    assert yellowstone.decode_source_gt("./.") == "./."
    with pytest.raises(ValueError, match="unexpected biallelic"):
        yellowstone.decode_source_gt("2/2")


@pytest.mark.parametrize(
    "value,call,parsed",
    [
        ("0,10,20", "0/0", (0, 10, 20)),
        ("10,0,20", "0/1", (10, 0, 20)),
        ("20,10,0", "1/1", (20, 10, 0)),
        ("0,0,20", "./.", (0, 0, 20)),
        (".,.,.", "./.", None),
        ("0,10", "./.", None),
        ("bad,10,20", "./.", None),
    ],
)
def test_unique_pl_argmin_rule_never_breaks_ties(value, call, parsed):
    assert yellowstone.unique_pl_argmin(value) == (call, parsed)


def test_yellowstone_panel_matrix_is_correlated_and_hash_pinned():
    assert len(yellowstone.RUN_SPECS) == 7
    assert yellowstone.PANEL_SPECS["main"]["counts"] == {"P1": 61, "P2": 58, "P3": 20}
    assert yellowstone.PANEL_SPECS["main"]["manifest_sha256"] == (
        "a6e1fad0047e4c0beedeae2dc165eba24acdf9853e4f57626677a7af90ff80bd"
    )
    assert yellowstone.PANEL_SPECS["candidate_null"]["candidate_class"] is None
    assert yellowstone.PANEL_SPECS["big_direct_stock"]["counts"] == {
        "P1": 19,
        "P2": 61,
        "P3": 20,
    }
    assert yellowstone.EXPECTED_PANEL_LOCI[("main", "source_GT", "standard_contract")] == (
        11_758,
        "9ef1e1f65321ab60540f2c90e1aa12d4f78ac2449c8686a0a295d78c7e75a274",
    )
    assert yellowstone.EXPECTED_PANEL_LOCI[
        ("main", "unique_PL_argmin", "within_population_polymorphism")
    ] == (
        3_653,
        "822b251fe3411f576331534511101e3ecd8c02194a8d8a785b57027b4c701bef",
    )
    assert set(yellowstone.EXPECTED_PREPARED_VCF) == {
        f"yellowstone_{panel}_{representation}_{locus_filter}"
        for panel, representation, locus_filter, _strict in yellowstone.RUN_SPECS
    }


def _fake_panel(prediction="C", direction_rms=2.0, gate_rms=1.0):
    return {
        "simulation_head": {"predicted_class": prediction},
        "simulation_feature_shift": {"rms_z": direction_rms},
        "simulation_gate_feature_shift": {"rms_z": gate_rms},
    }


@pytest.mark.parametrize("candidate", [None, "C"])
@pytest.mark.parametrize("prediction", ["A", "B", "C"])
@pytest.mark.parametrize("direction_rms,severe", [(10.0, False), (10.000001, True)])
def test_yellowstone_adjudication_never_manufactures_truth(candidate, prediction, direction_rms, severe):
    result = yellowstone.adjudicate_panel(
        _fake_panel(prediction, direction_rms=direction_rms), candidate
    )
    assert result["natural_data_call_status"] == (
        "abstain_severe_OOD" if severe else "descriptive_only_no_gold_label"
    )
    assert result["direction_truth_available"] is False
    assert result["exclusive_single_edge_truth_available"] is False
    assert result["gate_truth_available"] is False
    assert result["direction_call_accepted"] is False
    assert result["formal_direction_accuracy_eligible"] is False
    assert result["gate_accuracy_eligible"] is False
    assert result["raw_head_matches_candidate"] == (
        None if candidate is None else prediction == candidate
    )
    assert "heuristic diagnostic, not calibrated support" in result["severe_OOD_rule"]
    assert "accepted_direction_call" not in result


def test_yellowstone_summary_never_counts_sensitivities_as_trials():
    panels = []
    for index, (_name, _representation, _filter_name, _strict) in enumerate(yellowstone.RUN_SPECS):
        panel = _fake_panel("C" if index < 6 else "B", direction_rms=11.0)
        panel["adjudication"] = yellowstone.adjudicate_panel(
            panel, "C" if index < 6 else None
        )
        panels.append(panel)
    outcome = yellowstone.summarize_outcomes(panels)
    assert outcome["analytic_correlated_sensitivity_rows"] == 7
    assert outcome["unique_biological_systems"] == 1
    assert outcome["independent_direction_truth_units"] == 0
    assert outcome["independent_gate_truth_units"] == 0
    assert outcome["raw_head_prediction_counts"] == {"B": 1, "C": 6}
    assert outcome["raw_candidate_C_concordant_sensitivity_rows"] == 6
    assert outcome["candidate_C_sensitivity_rows"] == 6
    assert outcome["accuracy_denominator"] is None
    assert outcome["severe_OOD_panels"] == 7
    assert outcome["accepted_direction_calls"] == 0
    assert outcome["direction_accuracy_estimate"] is None
    assert outcome["gate_accuracy_estimate"] is None
    json.dumps(outcome, allow_nan=False)
    with pytest.raises(AssertionError, match="panel count"):
        yellowstone.summarize_outcomes(panels[:-1])
