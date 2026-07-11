import hashlib
import json
import math
from pathlib import Path


RESULT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "oyster_2017_external_benchmark_2026_07_11"
    / "results.json"
)
CODE_COMMIT = "cd9097b62c7d1d3b38b850250d7ed351d06af6d6"


def _walk_finite(value):
    if isinstance(value, dict):
        for child in value.values():
            _walk_finite(child)
    elif isinstance(value, list):
        for child in value:
            _walk_finite(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_oyster_2017_result_bundle_is_frozen_reproduced_and_guarded():
    raw = RESULT.read_bytes()
    assert len(raw) == 180_323
    assert hashlib.sha256(raw).hexdigest() == (
        "ae5e68fb9541f2239e96edd37af288760235334b8b7b0f68171663453e0927ff"
    )
    text = raw.decode("utf-8")
    assert "candidate-null crossing-sensitivity stress test" in text
    for stale in (
        "near-null specificity stress test",
        "low or episodic gene flow",
        "repeats the source near-null",
        "same-SNP DAPC",
    ):
        assert stale not in text
    result = json.loads(raw)
    _walk_finite(result)

    assert result["schema_version"] == "dnnaic-oyster-2017-external-benchmark-v1"
    assert result["git"] == {"commit": CODE_COMMIT, "dirty_at_run": False}
    assert result["runtime"]["thread_environment"] == {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }

    source = result["source"]
    assert source["verified"]["workbook"] == {
        "path": source["verified"]["workbook"]["path"],
        "bytes": 729_706,
        "md5": "572a079597af8530b15aaffd07325b55",
        "sha256": "e0f6983f1a15c9d7a1aeb4a76e220f24b1d4c766600502413b2cb5c4fdde8029",
    }
    assert set(source["verified"]) == {"workbook", "retrieval_contract"}
    assert "acquisition route" in source["verified"]["retrieval_contract"]
    assert source["sources_record"]["sha256"] == (
        "3fab919785961a4c6bf6088197191de2c5358a6c04bd0c1020551b75e8f55c90"
    )
    assert source["population_record"]["sha256"] == (
        "de9190e7abb65647d92e8f9064d0c63d549c641561d85282bf95b072fcefd1b2"
    )
    assert len(source["population_record"]["rows"]) == 8

    workbook = source["workbook_audit"]
    assert (workbook["dimension"], workbook["samples"], workbook["loci"]) == (
        "A1:CNJ93",
        90,
        1_200,
    )
    assert (workbook["genotype_pairs"], workbook["missing_genotype_pairs"]) == (
        108_000,
        3_647,
    )
    assert workbook["genotype_pair_counts"] == {
        "0/0": 3_647,
        "1/1": 68_298,
        "1/2": 23_494,
        "2/2": 12_561,
    }
    assert workbook["partial_missing_pairs"] == 0
    assert workbook["reversed_heterozygote_pairs_2_slash_1"] == 0
    assert workbook["missing_original_sample_ids"] == ["28", "29", "31"]
    assert workbook["sample_missingness"]["samples_below_0_95_call_rate"] == [
        "20", "24", "36", "38", "50", "57", "66", "70", "84"
    ]
    assert workbook["locus_missingness"] == {
        "loci_below_0_95_call_rate": 387,
        "minimum_call_rate": 0.9,
        "minimum_called_individuals": 81,
    }
    assert workbook["globally_polymorphic_loci"] == 1_200
    assert workbook["genalex_semantic_sha256"] == (
        "9f58cf09e53c8353ea5d1ec272b0af7ac7ae05af257b12b6acb014c392be825e"
    )

    derived = source["derived_source_vcf"]
    assert {
        key: derived[key]
        for key in ("bytes", "sha256", "ordered_locus_id_sha256", "samples", "loci")
    } == {
        "bytes": 470_039,
        "sha256": "7d978cb745008e880a023f4c6347c54d50abd9c19cfb5daeba1f964fc829d756",
        "ordered_locus_id_sha256": "48f1c20c0bb01bad52330eae6bf5775ffd3cc2bf74e9d7eae374165546614632",
        "samples": 90,
        "loci": 1_200,
    }
    manifests = source["manifest_audit"]
    assert manifests["W"]["population_counts"] == {"WB2": 12, "WOC": 12, "WWC": 12}
    assert manifests["Q"]["population_counts"] == {"QB2": 9, "QOC": 12, "QWC": 12}
    assert manifests["union"]["samples"] == 69
    assert manifests["excluded_nonbenchmark_reference_cohorts"]["samples"] == 21

    expected_filters = {
        "standard_contract": (
            1_101,
            82,
            17,
            "edafecad96e334e40dfff485bb99ba6e1354a0d7841af7257851673c286a75ed",
            "f55a7c9d365de973394eaae962a165b09632b0bfe8eba85d5d09774e36756629",
        ),
        "within_population_polymorphism": (
            589,
            82,
            529,
            "df6659235f9030e4ecec84c36be610239e0912d8986daf2f053144892d71ec26",
            "d79c788ba837504fa709ba1653157f3b03155b2c1cc165ddc5ee038028e6c810",
        ),
    }
    for filter_name, (loci, insufficient, not_poly, locus_hash, id_hash) in expected_filters.items():
        audit = source["shared_filter_audits"][filter_name]
        assert audit["counts"] == {
            "source_variant_rows": 1_200,
            "insufficient_called_copies": insufficient,
            "not_polymorphic_in_every_panel": not_poly,
            "eligible_before_cap": loci,
            "retained_after_cap": loci,
        }
        assert audit["ordered_locus_sha256"] == locus_hash
        assert audit["ordered_locus_id_sha256"] == id_hash
        assert all(values["minimum"] >= 16 for values in audit["population_called_copy_counts"].values())
        assert "QBB2 (n=9)" in audit["joint_called_copy_guardrail"]
        assert "conditioned on Q missingness" in audit["joint_called_copy_guardrail"]

    summary = result["outcome_summary"]
    assert summary["analytic_sensitivity_runs"] == 4
    assert summary["correlated_site_comparisons"] == 2
    assert summary["unique_biological_systems"] == 1
    assert summary["independent_validation_panels"] == 0
    assert summary["accuracy_available"] is False
    assert summary["accuracy_estimate"] is None
    assert summary["specificity_available"] is False
    assert summary["specificity_estimate"] is None
    assert summary["raw_gate_below_0_5"] == 0
    assert summary["raw_gate_crossings_at_0_5"] == 4
    assert summary["raw_head_prediction_counts"] == {"A": 3, "B": 1, "C": 0}
    assert summary["raw_counterfactual_C_calls"] == 0
    assert summary["severe_OOD_panels"] == 4
    assert summary["abstained_panels"] == 4

    expected_panels = {
        "oyster_W_standard_contract": {
            "order": ("WWC", "WOC", "WB2"),
            "loci": 1_101,
            "locus_hash": "edafecad96e334e40dfff485bb99ba6e1354a0d7841af7257851673c286a75ed",
            "id_hash": "f55a7c9d365de973394eaae962a165b09632b0bfe8eba85d5d09774e36756629",
            "vcf_hash": "ddf559197778b9992f79df4ffbdb6265f74376955c292fc21ee16aac347206d5",
            "popmap_hash": "2a3d2118e8db81cf3102b1c0b26074ec965ba0684bd7ded6ee3257c9eb54b98d",
            "prediction": "A",
            "direction_rms": 21.503260061435082,
            "gate_rms": 24.77741669649106,
            "projection": 0.18729882749884988,
            "maximum": 0.75,
            "f3": 0.0014585621033936587,
            "f3_ci": [-4.2498214434198044e-05, 0.0029471978181004822],
            "excluded": [],
        },
        "oyster_Q_standard_contract": {
            "order": ("QWC", "QOC", "QB2"),
            "loci": 1_101,
            "locus_hash": "edafecad96e334e40dfff485bb99ba6e1354a0d7841af7257851673c286a75ed",
            "id_hash": "f55a7c9d365de973394eaae962a165b09632b0bfe8eba85d5d09774e36756629",
            "vcf_hash": "5b165711d050941db95953e16d860bdb3ef5ceb8cf10a416ec3edc4559439b42",
            "popmap_hash": "4ee13d74ae2cdd60bacc42bad409fc22e6f487920864bb2b449401b5d505addd",
            "prediction": "A",
            "direction_rms": 21.102756366240772,
            "gate_rms": 24.41150935661631,
            "projection": 0.19714260686102314,
            "maximum": 0.6666666666666667,
            "f3": 0.0027353075903415805,
            "f3_ci": [0.0010222858232934332, 0.004626361836460063],
            "excluded": ["28", "29", "31"],
        },
        "oyster_W_within_population_polymorphism": {
            "order": ("WWC", "WOC", "WB2"),
            "loci": 589,
            "locus_hash": "df6659235f9030e4ecec84c36be610239e0912d8986daf2f053144892d71ec26",
            "id_hash": "d79c788ba837504fa709ba1653157f3b03155b2c1cc165ddc5ee038028e6c810",
            "vcf_hash": "851dea0d97074937eeab97eb9b2046de2aedd1f786587ea82cfd18e7f9f793b0",
            "popmap_hash": "2a3d2118e8db81cf3102b1c0b26074ec965ba0684bd7ded6ee3257c9eb54b98d",
            "prediction": "A",
            "direction_rms": 24.237104447606015,
            "gate_rms": 25.09431403768693,
            "projection": 0.2163830760129003,
            "maximum": 0.6666666666666666,
            "f3": 0.0012577468388301892,
            "f3_ci": [-0.0014158858357375165, 0.0036767584777889677],
            "excluded": [],
        },
        "oyster_Q_within_population_polymorphism": {
            "order": ("QWC", "QOC", "QB2"),
            "loci": 589,
            "locus_hash": "df6659235f9030e4ecec84c36be610239e0912d8986daf2f053144892d71ec26",
            "id_hash": "d79c788ba837504fa709ba1653157f3b03155b2c1cc165ddc5ee038028e6c810",
            "vcf_hash": "a8cd7794bb8d5ad84a54e7a6e3e517ed55f82a0c0c047e132d06977e624d210f",
            "popmap_hash": "4ee13d74ae2cdd60bacc42bad409fc22e6f487920864bb2b449401b5d505addd",
            "prediction": "B",
            "direction_rms": 25.664701279809375,
            "gate_rms": 25.386792176025004,
            "projection": 0.26880826490466125,
            "maximum": 0.611111111111111,
            "f3": 0.0031567853884143945,
            "f3_ci": [0.00043481775749743006, 0.006399513051392203],
            "excluded": ["28", "29", "31"],
        },
    }
    assert [panel["panel_id"] for panel in result["panels"]] == list(expected_panels)
    for panel in result["panels"]:
        expected = expected_panels[panel["panel_id"]]
        order = panel["population_order"]
        assert (order["P1"], order["P2"], order["P3"]) == expected["order"]
        expectation = panel["external_expectation"]
        assert expectation["expected_gate"] is None
        assert expectation["candidate_class_if_event_present"] == "C"
        assert "candidate_class" not in expectation
        assert expectation["direction_truth_available"] is False
        assert expectation["gate_truth_available"] is False
        assert expectation["accuracy_eligible"] is False
        assert expectation["specificity_eligible"] is False
        assert expectation["same_data_excluded_ids"] == expected["excluded"]
        assert "conditioned on Q missingness" in expectation["joint_called_copy_guardrail"]
        assert panel["padze"]["n_loci_kept"] == expected["loci"]
        audit = panel["input_audit"]
        assert audit["ordered_locus_sha256"] == expected["locus_hash"]
        assert audit["ordered_locus_id_sha256"] == expected["id_hash"]
        assert audit["derived_vcf"]["sha256"] == expected["vcf_hash"]
        assert audit["derived_popmap"]["sha256"] == expected["popmap_hash"]
        assert panel["simulation_head"]["predicted_class"] == expected["prediction"]
        assert "OOD-detector" in panel["simulation_head"]["interpretation"]
        assert panel["simulation_gate"]["appreciable_score"] == 1.0
        assert "OOD-detector" in panel["simulation_gate"]["interpretation"]
        assert panel["simulation_feature_shift"]["rms_z"] == expected["direction_rms"]
        assert panel["simulation_gate_feature_shift"]["rms_z"] == expected["gate_rms"]
        adjudication = panel["adjudication"]
        assert adjudication["severe_OOD"] is True
        assert adjudication["natural_data_call_status"] == "abstain_severe_OOD"
        assert adjudication["literature_gate_relation"] == (
            "qualitatively_in_tension_with_detection_limited_near_null"
        )
        assert adjudication["counterfactual_direction_relation"] == (
            "raw_class_differs_from_counterfactual_exposure_orientation_if_event_present"
        )
        assert adjudication["accuracy_eligible"] is False
        assert adjudication["specificity_eligible"] is False
        assert "matches_candidate_reference" not in adjudication

        geometry = panel["model_free_comparator"]
        assert geometry["P2_projection_from_P1_toward_P3_all_loci"] == expected["projection"]
        assert geometry["maximum_abs_P3_minus_P1_frequency"] == expected["maximum"]
        assert geometry["diagnostic_loci"] == 0
        assert geometry["P2_projection_from_P1_toward_P3_diagnostic_loci"] is None
        assert geometry["f3_P2_P1_P3_finite_called_copy_corrected"] == expected["f3"]
        bootstrap = geometry["iid_locus_bootstrap"]
        assert bootstrap["seed"] == 20260711
        assert bootstrap["requested_replicates"] == 500
        assert bootstrap["projection_diagnostic_loci"] is None
        assert bootstrap["f3_finite_called_copy_corrected"]["percentile_95_interval"] == expected["f3_ci"]
        assert "independent binomial called-copy sampling" in geometry["interpretation"]
        assert "not generally unbiased" in geometry["interpretation"]
        assert "does not prove a near-null" in geometry["interpretation"]
        assert "not chromosome-block" in bootstrap["guardrail"]
