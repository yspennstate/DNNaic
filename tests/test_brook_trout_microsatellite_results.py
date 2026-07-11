from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "results"
    / "brook_trout_microsatellite_benchmark_2026_07_11"
    / "results.json"
)
EXPECTED_RESULT_SHA256 = (
    "d69ee22f2447526f2f0d5c7577b14e219e756dd60966dd75838deab9ffed6615"
)


def load_result() -> dict:
    raw = RESULT.read_bytes()
    assert len(raw) == 985_008
    assert hashlib.sha256(raw).hexdigest() == EXPECTED_RESULT_SHA256
    return json.loads(raw)


def test_frozen_result_provenance_sources_and_runtime():
    result = load_result()
    assert result["schema_version"] == (
        "dnnaic-brook-trout-microsatellite-benchmark-v1"
    )
    assert result["status"] == (
        "management_history_and_same_marker_transfer_diagnostic_not_accuracy"
    )
    for revision in (result["git"], result["final_git_recheck"]):
        assert revision["commit"] == "0a6cfd35ee062e94caa1e97feca62b9e3f3a4f8e"
        assert revision["script_sha256"] == (
            "a37a1d87a1fb18fb67d002fc3840c8fda276e7fff6aa5e24deab3aa89e345045"
        )
        assert revision["tracked_dirty_at_snapshot"] is False
        assert revision["tracked_diff_bytes"] == 0
    assert result["configuration_sha256"] == (
        "39ef95df10f024432c3df81d0bed58592d87fcb701bd6bde81d652827c482d4c"
    )
    expected_sources = {
        "pa_workbook": "54d112251103793e341ac97df066b73b1173c77ddf355ee862d2e7cf4eb1d1e4",
        "ns_genepop": "055ec1d7006368df3974483fc0f8042ee2176d3a21f615e7f5e1d777af3a5dda",
        "ns_population_names": "3ab94da572756545e833917069b33970e438f8a06d53cae9ed1e437ce19aa61b",
        "ns_introgression": "48c9d06475b33fbcd990c6d99f59b3fcda466da0e4d7bac38b841c9c0d4d88c4",
    }
    assert {
        name: audit["sha256"] for name, audit in result["source_audits"].items()
    } == expected_sources
    assert {
        name: audit["sha256"]
        for name, audit in result["source_final_recheck"].items()
    } == expected_sources

    runtime = result["runtime"]
    assert runtime["compute_target"] == "azure"
    assert runtime["cuda_visible_devices"] == ""
    assert runtime["process_priority"]["priority_class"] == 15
    assert runtime["process_priority"]["verified"] is True
    assert set(runtime["thread_environment"].values()) == {"1"}
    assert runtime["packages"] == {
        "numpy": "2.5.1",
        "padze": "0.1.0",
        "scikit-learn": "1.9.0",
    }


def test_parser_and_panel_accounting_are_exact_and_non_independent():
    result = load_result()
    parsers = result["parser_audits"]
    assert parsers["pennsylvania"]["individuals"] == 2_048
    assert parsers["pennsylvania"]["wild_rows_raw"] == 1_748
    assert parsers["pennsylvania"]["hatchery_rows"] == 300
    assert parsers["pennsylvania"]["called_genotype_pairs"] == 24_441
    assert parsers["pennsylvania"]["missing_genotype_pairs"] == 135
    assert "SfoC-79" in parsers["pennsylvania"]["paper_source_mismatch"]
    assert "six FLAG/POLE exclusions" in parsers["pennsylvania"][
        "paper_cohort_guardrail"
    ]
    assert parsers["nova_scotia_genepop"]["individuals"] == 1_729
    assert parsers["nova_scotia_genepop"]["populations"] == 39
    assert parsers["nova_scotia_genepop"]["loci"] == 100
    assert parsers["nova_scotia_genepop"]["called_genotype_pairs"] == 168_141
    assert parsers["nova_scotia_genepop"]["missing_genotype_pairs"] == 4_759

    loci = result["selection_audits"]["loci"]
    assert loci["pennsylvania"]["standard_loci"] == 12
    assert loci["nova_scotia"]["standard_loci"] == 90
    assert loci["nova_scotia"]["strict_loci"] == 62
    assert loci["record_views"] == 28
    assert loci["descriptive_primary_views"] == 11
    accounting = result["analysis"]["descriptive_panel_accounting"]
    assert accounting == {
        "distinct_named_primary_target_or_river_views": 11,
        "guardrail": (
            "11 is not 11 independent validation units and is not an accuracy denominator; "
            "subsamples, no-record panels, and strict-locus reruns are correlated views"
        ),
        "nova_scotia_no_recorded_stocking_contrasts": 3,
        "nova_scotia_no_recorded_stocking_named_systems": 2,
        "nova_scotia_no_recorded_stocking_record_views": 6,
        "nova_scotia_recorded_stocking_river_systems": 8,
        "pennsylvania_targets": 3,
    }


def test_every_record_is_unaccepted_severe_ood_and_curve_is_valid():
    records = load_result()["analysis"]["records"]
    assert len(records) == 28
    assert Counter(record["contract_role"] for record in records) == {
        "primary": 11,
        "locus_filter_sensitivity": 8,
        "reference_sensitivity": 3,
        "no_recorded_stocking_sensitivity": 3,
        "no_recorded_stocking_strict_locus_sensitivity": 3,
    }
    assert Counter(record["dataset"] for record in records) == {
        "nova_scotia_lehnert_2020": 22,
        "pennsylvania_white_2018": 6,
    }
    for record in records:
        assert record["direction_call_accepted"] is False
        assert record["formal_direction_accuracy_eligible"] is False
        assert record["gate_accuracy_eligible"] is False
        assert record["adjudication"]["direction_call_accepted"] is False
        assert record["adjudication"]["formal_direction_accuracy_eligible"] is False
        assert record["adjudication"]["gate_accuracy_eligible"] is False
        assert record["adjudication"]["severe_OOD_heuristic"] is True
        curve = np.asarray(record["curve"], dtype=float)
        assert curve.shape == (15, 28)
        assert np.isfinite(curve).all()


def test_candidate_concordance_is_descriptive_and_representation_dependent():
    result = load_result()
    analysis = result["analysis"]
    primary = [
        record for record in analysis["records"] if record["contract_role"] == "primary"
    ]
    assert Counter(record["direction"]["raw_all"]["call"] for record in primary) == {
        "C": 9,
        "A": 2,
    }
    assert Counter(
        record["direction"]["raw_all"]["call"]
        for record in primary
        if record["dataset"] == "pennsylvania_white_2018"
    ) == {"C": 3}
    assert Counter(
        record["direction"]["raw_all"]["call"]
        for record in primary
        if record["dataset"] == "nova_scotia_lehnert_2020"
    ) == {"C": 6, "A": 2}

    expected = {
        "raw_all": (9, 11),
        "raw_mean_variance": (6, 11),
        "orbit_composition_mean_variance": (1, 11),
    }
    for name, (calls, views) in expected.items():
        concordance = analysis["representations"][name][
            "descriptive_primary_candidate_concordance"
        ]
        assert (concordance["C_calls"], concordance["views"]) == (calls, views)
        assert "not formal accuracy" in concordance["interpretation"]
    raw = analysis["representations"]["raw_all"]
    assert raw["external_rms_z"]["median"] > 500
    assert raw["external_rms_z"]["p95_max_abs"] > 6_000
    assert raw["nova_scotia_C_score_vs_same_locus_published_Q_delta_spearman"] < 0
    assert analysis["depth_matched_gate"]["contract"]["feature_dimension"] == 216


def test_pennsylvania_stocking_context_preserves_source_conflict():
    config = load_result()["configuration"]
    context = config["pennsylvania"]["stocking_context"]
    assert context["DOUB"]["source_internal_conflict"] is True
    assert context["DOUB"]["table_1_stocking_at_sample_location"] is True
    assert context["DOUB"]["discussion_states_not_directly_stocked"] is True
    assert context["LICK"]["table_1_stocking_at_sample_location"] is False
    assert context["LICK"]["table_1_stocking_within_2_km"] is True
    assert context["CONK"]["no_stocking_record_more_than_50_years"] is True
    assert config["evaluation"]["formal_accuracy_eligible"] is False
    assert config["evaluation"]["direction_calls_accepted"] is False
    assert config["evaluation"]["gate_accuracy_eligible"] is False
