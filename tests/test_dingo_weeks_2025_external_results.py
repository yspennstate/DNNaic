import hashlib
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "results" / "dingo_weeks_2025_external_benchmark_2026_07_11" / "results.json"
RESULT_BYTES = 156_314
RESULT_SHA256 = "88d0d808abd7e592e87cfee7623ec2311005c10741a5d7f885b06b1d1e09efca"

EXPECTED = {
    "dingo_backcross_paper_release_all_rows_standard_contract": {
        "loci": 2_193,
        "ordered": "d5b469c2e5f3f62ca0703dc931aa84bb58bab3ddf0485b5b01e3b134b12ca00c",
        "vcf": "05130efd130b6375c0949b59d2ced8bfc68d5c0a833b56f0ebb9c2aa0dcd6519",
        "scores": [0.9999999999887088, 1.1291169165667953e-11, 1.4153278493809165e-19],
        "direction_rms": 18.640794019688478,
        "gate_rms": 21.62988166782979,
        "projection": 0.24934671800775357,
        "projection_ci": [0.2169730738703886, 0.280843093331984],
        "f3": 0.0028926153667235147,
        "f3_ci": [0.0005854935932664341, 0.004969458431874807],
    },
    "dingo_backcross_paper_release_all_rows_within_population_polymorphism": {
        "loci": 1_594,
        "ordered": "b931307dff9a253c3cec54c101c68bd5e8398640c6264a1a454e559363715e84",
        "vcf": "e674ab8994aa25235d3e3e434d2cd0dac9341bcb89001cc6a65292b407bf4aed",
        "scores": [0.999805468407981, 2.24324767887024e-29, 0.0001945315920190896],
        "direction_rms": 22.090515951305612,
        "gate_rms": 24.254030714283306,
        "projection": 0.2698133977247148,
        "projection_ci": [0.2363761906876593, 0.30596946709125944],
        "f3": -0.0005183146466660117,
        "f3_ci": [-0.0032238384407369306, 0.002115994383338977],
    },
    "dingo_backcross_vcf_PASS_only_standard_contract": {
        "loci": 1_992,
        "ordered": "79696ea682909b94d4a64f1eb84d989367aa58948fa66d11bbe453019ce0be6f",
        "vcf": "2f400d4c3e30ef28ba3572fdd042c20383658971cb79cf1ddd90eb0967d8303b",
        "scores": [0.9997235815539942, 0.0002764184460058413, 7.172813039398417e-20],
        "direction_rms": 19.143894406360058,
        "gate_rms": 21.979696868888194,
        "projection": 0.24513881514382543,
        "projection_ci": [0.21288393540007133, 0.2781004927407181],
        "f3": 0.002928512104239658,
        "f3_ci": [0.0006436049048608769, 0.005126130213917385],
    },
    "dingo_backcross_vcf_PASS_only_within_population_polymorphism": {
        "loci": 1_551,
        "ordered": "6dd0696039348e18ff2e5db4f8c08f4760cefe59f1c3e15c5eaffcdbe16b75dd",
        "vcf": "98f44ca8522aeb6815f27f95e4ddeadaf557cfa0fe6f5f2bf5be5fd27f3938fb",
        "scores": [0.9999264312477194, 1.47416533734108e-27, 7.356875228065496e-05],
        "direction_rms": 22.310441216357354,
        "gate_rms": 24.404381880584918,
        "projection": 0.26838940732068867,
        "projection_ci": [0.23508225953153414, 0.30607449728034064],
        "f3": -0.0005347900409355745,
        "f3_ci": [-0.003230618666354379, 0.0020076815943439255],
    },
}


def _reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _load():
    raw = RESULT.read_bytes()
    assert len(raw) == RESULT_BYTES
    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    return json.loads(raw, parse_constant=_reject_nonfinite)


def test_dingo_result_provenance_runtime_and_outcome_are_guarded():
    result = _load()
    assert result["schema_version"] == "dnnaic-dingo-weeks-2025-external-benchmark-v1"
    assert result["git"] == {
        "commit": "ae90b6d955a81a3cdb5785a6c23c3be7a8549117",
        "dirty_at_run": False,
    }
    assert result["source"]["sources_record"]["canonical_lf"] == {
        "bytes": 3_480,
        "sha256": "30e8c5088aaaea37099d2be64b4b5868ec081b59f3372ba3ae7d7ee0ec9a16ad",
    }
    assert result["source"]["sources_record"]["working_tree"] == {
        "bytes": 3_480,
        "line_endings_normalized_for_contract": False,
        "sha256": "30e8c5088aaaea37099d2be64b4b5868ec081b59f3372ba3ae7d7ee0ec9a16ad",
    }
    assert result["source"]["release_filter_normalization"]["derived_vcf"]["sha256"] == (
        "c15274389e2601ae230e3092684f65aa01d158f2b33d00ea7961f2dd345ac9c6"
    )
    assert result["analysis_design"]["manifest"]["sha256"] == (
        "12eed285869fa3426b82442ebbb44e866af77cf6013c7853776d421d20f23d45"
    )
    assert result["analysis_design"]["exclusive_single_edge_truth_available"] is False
    assert result["analysis_design"]["formal_direction_accuracy_eligible"] is False
    assert result["runtime"]["packages"] == {
        "numpy": "2.4.3",
        "padze": "0.1.0",
        "scikit-learn": "1.8.0",
    }
    assert result["runtime"]["thread_environment"] == {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    assert result["outcome"] == {
        "abstained_panels": 4,
        "accepted_direction_calls": 0,
        "accuracy_guardrail": (
            "The pedigree anchors the dog-introgressing component, not an exclusive one-edge "
            "population history or an accuracy rate; four filters reuse the same eight P2 animals "
            "and reference samples."
        ),
        "analytic_filter_sensitivity_runs": 4,
        "correlated_filter_sensitivities_not_trials": True,
        "descriptive_nonsevere_panels": 0,
        "direction_accuracy_estimate": None,
        "exclusive_single_edge_truth_units": 0,
        "gate_accuracy_estimate": None,
        "independent_pedigree_dog_component_units": 1,
        "independent_sample_level_units": 0,
        "raw_head_matches_candidate_C": 0,
        "raw_head_prediction_counts": {"A": 4},
        "severe_OOD_panels": 4,
        "unique_biological_systems": 1,
    }


def test_dingo_four_correlated_panel_results_are_exact_and_all_abstain():
    result = _load()
    assert len(result["panels"]) == 4
    for panel in result["panels"]:
        expected = EXPECTED[panel["panel_id"]]
        assert panel["padze"]["n_loci_kept"] == expected["loci"]
        assert panel["input_audit"]["ordered_locus_sha256"] == expected["ordered"]
        assert panel["input_audit"]["derived_vcf"]["sha256"] == expected["vcf"]
        assert panel["input_audit"]["derived_popmap"] == {
            "bytes": 3_245,
            "path": panel["input_audit"]["derived_popmap"]["path"],
            "sha256": "12eed285869fa3426b82442ebbb44e866af77cf6013c7853776d421d20f23d45",
        }
        assert panel["input_audit"]["population_called_copy_counts"]["P2"] == {
            "individuals": 8,
            "maximum": 16,
            "mean": 16.0,
            "minimum": 16,
        }
        assert panel["input_audit"]["P2_complete_case_ascertainment"][
            "all_retained_loci_complete_case_in_P2"
        ] is True
        assert panel["simulation_head"]["predicted_class"] == "A"
        assert [panel["simulation_head"]["scores"][key] for key in ("A", "B", "C")] == pytest.approx(
            expected["scores"], rel=0, abs=0
        )
        assert panel["simulation_feature_shift"]["rms_z"] == expected["direction_rms"]
        assert panel["simulation_gate"]["appreciable_score"] == 1.0
        assert panel["simulation_gate_feature_shift"]["rms_z"] == expected["gate_rms"]

        geometry = panel["model_free_comparator"]
        assert geometry["diagnostic_loci"] == 0
        assert geometry["P2_projection_from_P1_toward_P3_diagnostic_loci"] is None
        assert geometry["P2_projection_from_P1_toward_P3_all_loci"] == expected["projection"]
        assert geometry["f3_P2_P1_P3"] == expected["f3"]
        bootstrap = geometry["scaffold_block_bootstrap"]
        assert bootstrap["projection_all_loci"]["percentile_95_interval"] == expected["projection_ci"]
        assert bootstrap["f3_P2_P1_P3"]["percentile_95_interval"] == expected["f3_ci"]
        assert (bootstrap["seed"], bootstrap["requested_replicates"], bootstrap["scaffold_blocks"]) == (
            20260711,
            500,
            38,
        )

        adjudication = panel["adjudication"]
        assert adjudication["severe_OOD"] is True
        assert adjudication["natural_data_call_status"] == "abstain_severe_OOD"
        assert adjudication["direction_call_accepted"] is False
        assert adjudication["formal_direction_accuracy_eligible"] is False
        assert adjudication["raw_head_matches_candidate_C"] is False
        assert "accepted_direction_call" not in adjudication
