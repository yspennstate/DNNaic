from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "stdpopsim_independent_catalog_benchmark_2026_07_11"
    / "results.json"
)
EXPECTED_RESULT_SHA256 = (
    "99a2a0cf3859f204379d1d156dd915b568bc82fd9bdccf54ff9e4b7f9a0b3d80"
)


def load_result() -> dict:
    raw = RESULT.read_bytes()
    assert len(raw) == 470_531
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_RESULT_SHA256
    return json.loads(raw)


def test_frozen_result_provenance_and_checkpoint_contract():
    result = load_result()
    assert result["schema_version"] == (
        "dnnaic-stdpopsim-independent-catalog-benchmark-v1"
    )
    assert result["status"] == (
        "known_truth_two_species_focal_ablation_synthetic_transfer"
    )
    for revision in (result["git"], result["final_source_recheck"]):
        assert revision["commit"] == "2dd19f1fd159e0cbedcc638cd18b651d3607e6c4"
        assert revision["script_sha256"] == (
            "a07525f02f49d0345d8aee0ef5f11aae9035b7351af2a8b898eb7455e9e5b6d2"
        )
        assert revision["tracked_dirty_at_snapshot"] is False
        assert revision["tracked_diff_bytes"] == 0
    assert result["configuration_sha256"] == (
        "ad17b6416a9b44b0b16e010e88d204648814b971501a2b6de92d8add6ffd3a39"
    )
    checkpoint = result["checkpoint"]
    assert checkpoint["sha256"] == (
        "d22aa641d1aaae5b7b2b2f2d5089afca8c1cbba4c4938e686496f3c018b0257a"
    )
    assert checkpoint["bytes"] == 2_473_066
    assert checkpoint["records"] == 120
    assert checkpoint["complete_positive_control_families"] == 60
    assert checkpoint["stored_curve_shape"] == [120, 198, 28]


def test_result_accounting_keeps_controls_out_of_direction_accuracy():
    analysis = load_result()["analysis"]
    assert {
        key: analysis[key]
        for key in (
            "records",
            "families",
            "species",
            "focal_panels",
            "direction_accuracy_rows",
            "focal_absent_controls_excluded_from_direction_accuracy",
        )
    } == {
        "records": 120,
        "families": 60,
        "species": 2,
        "focal_panels": 2,
        "direction_accuracy_rows": 60,
        "focal_absent_controls_excluded_from_direction_accuracy": 60,
    }
    assert analysis["truth_counts"] == {"B": 30, "C": 30}

    ledger = analysis["prediction_ledger"]
    assert len(ledger) == len({row["job_id"] for row in ledger}) == 120
    assert Counter(row["condition"] for row in ledger) == {
        "positive": 60,
        "control": 60,
    }
    assert Counter(row["panel_id"] for row in ledger) == {
        "ashk_ceu_to_waj": 60,
        "canfam_isw_to_glj": 60,
    }
    controls = [row for row in ledger if row["condition"] == "control"]
    assert all(row["included_in_direction_accuracy"] is False for row in controls)
    assert all(row["direction_truth"] is None for row in controls)
    assert all(row["raw_all_correct"] is None for row in controls)

    simulations = analysis["simulation_record_ledger"]
    assert len({row["engine_seed"] for row in simulations}) == 120
    assert len({row["engine_derived_ancestry_seed"] for row in simulations}) == 120
    assert len({row["engine_derived_mutation_seed"] for row in simulations}) == 120
    family_conditions = {
        family: Counter(
            row["condition"] for row in simulations if row["family_id"] == family
        )
        for family in {row["family_id"] for row in simulations}
    }
    assert len(family_conditions) == 60
    assert all(counts == {"positive": 1, "control": 1} for counts in family_conditions.values())


def test_frozen_direction_and_gate_metrics_include_transfer_guardrails():
    analysis = load_result()["analysis"]
    representations = analysis["representations"]
    raw = representations["raw_all"]
    assert raw["B_C_balanced_accuracy"] == pytest.approx(0.8)
    assert raw["B_recall"]["successes"] == 18
    assert raw["B_recall"]["n"] == 30
    assert raw["C_recall"]["successes"] == 30
    assert raw["C_recall"]["n"] == 30
    assert raw["per_panel"]["ashk_ceu_to_waj"]["predicted_class_counts"] == {
        "C": 30
    }
    assert raw["per_panel"]["canfam_isw_to_glj"]["predicted_class_counts"] == {
        "B": 18,
        "C": 12,
    }
    assert representations["raw_mean_variance"]["B_C_balanced_accuracy"] == pytest.approx(0.5)
    assert representations["orbit_composition_mean_variance"][
        "B_C_balanced_accuracy"
    ] == pytest.approx(0.05)
    assert raw["scaler_rms_z_median"] > 8
    assert raw["scaler_max_abs_z_p95"] > 60

    gate = analysis["frozen_gate_score_discrimination"]
    assert gate["equal_panel_macro_positive_control_roc_auc"] == pytest.approx(
        0.9777777777777779
    )
    assert gate["per_panel"]["ashk_ceu_to_waj"][
        "positive_control_roc_auc"
    ] == pytest.approx(0.9555555555555556)
    assert gate["per_panel"]["canfam_isw_to_glj"][
        "positive_control_roc_auc"
    ] == pytest.approx(1.0)
    assert "score-only" in gate["status"]
    assert "not 60 independent systems" in analysis["guardrail"]


def test_runtime_was_single_threaded_low_priority_azure_cpu():
    runtime = load_result()["runtime"]
    assert runtime["compute_target"] == "azure"
    assert runtime["cuda_visible_devices"] == ""
    assert runtime["process_priority"]["priority_class"] == 15
    assert runtime["process_priority"]["verified"] is True
    assert set(runtime["thread_environment"].values()) == {"1"}
    assert runtime["packages"] == {
        "msprime": "1.4.2",
        "numpy": "2.5.1",
        "padze": "0.1.0",
        "scikit-learn": "1.9.0",
        "stdpopsim": "0.3.0",
        "tskit": "1.0.3",
    }
