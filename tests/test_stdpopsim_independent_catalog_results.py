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
    "4ed901917a15684f384da481ea9ef54f498a6efadab35285eda780b00788b74e"
)


def load_result() -> dict:
    raw = RESULT.read_bytes()
    assert len(raw) == 485_914
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_RESULT_SHA256
    return json.loads(raw)


def test_frozen_result_provenance_and_checkpoint_contract():
    result = load_result()
    assert result["schema_version"] == (
        "dnnaic-stdpopsim-independent-catalog-benchmark-v2"
    )
    assert result["status"] == (
        "known_truth_two_species_focal_ablation_synthetic_transfer"
    )
    for revision in (result["git"], result["final_source_recheck"]):
        assert revision["commit"] == "c0584877da9def78ed78669e187d1c7737f824de"
        assert revision["script_sha256"] == (
            "91644fc0b26514a72b2b94a00f94788be2490ff550f14881d9ba8f8ae9e2f90e"
        )
        assert revision["tracked_dirty_at_snapshot"] is False
        assert revision["tracked_diff_bytes"] == 0
    assert result["configuration_sha256"] == (
        "1c1a48dfa4d73e4360dafb74b2b504ff40b4689374a64450562a335b424452dc"
    )
    checkpoint = result["checkpoint"]
    assert checkpoint["sha256"] == (
        "b97439ee52f21bd2fe1d5836973c4c88aa142fd8eeb6641bbb9490d3c10c8f17"
    )
    assert checkpoint["bytes"] == 2_473_685
    assert checkpoint["schema_version"] == (
        "dnnaic-stdpopsim-independent-catalog-checkpoint-v2"
    )
    assert checkpoint["configuration_sha256"] == result["configuration_sha256"]
    assert checkpoint["record_curve_hash_ledger_sha256"] == (
        "d247e178938100407235736e3a58c14063c62aa66fe0e1a08657590a1b289a4f"
    )
    assert checkpoint["records"] == 120
    assert checkpoint["complete_positive_control_families"] == 60
    assert checkpoint["stored_curve_shape"] == [120, 198, 28]


def test_control_truth_is_null_in_every_committed_semantic_ledger():
    result = load_result()
    configuration = result["configuration"]
    assert configuration["evaluation"]["control_direction_truth"] is None
    assert (
        configuration["evaluation"]["controls_excluded_from_direction_accuracy"]
        is True
    )
    assert (
        configuration["evaluation"]["panel_candidate_direction_retained_for_controls"]
        is True
    )

    ledgers = (
        configuration["job_manifest"],
        result["analysis"]["simulation_record_ledger"],
        result["analysis"]["prediction_ledger"],
    )
    for ledger in ledgers:
        assert len(ledger) == 120
        controls = [row for row in ledger if row["condition"] == "control"]
        positives = [row for row in ledger if row["condition"] == "positive"]
        assert len(controls) == len(positives) == 60
        assert all(row["direction_truth"] is None for row in controls)
        assert Counter(row["panel_candidate_direction"] for row in controls) == {
            "B": 30,
            "C": 30,
        }
        assert all(
            row["direction_truth"] == row["panel_candidate_direction"]
            for row in positives
        )

    predictions = result["analysis"]["prediction_ledger"]
    controls = [row for row in predictions if row["condition"] == "control"]
    positives = [row for row in predictions if row["condition"] == "positive"]
    assert all(row["included_in_direction_accuracy"] is False for row in controls)
    assert all(row["raw_all_correct"] is None for row in controls)
    assert all(row["included_in_direction_accuracy"] is True for row in positives)
    assert all(isinstance(row["raw_all_correct"], bool) for row in positives)


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
