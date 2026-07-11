import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "results" / "guppy_2020_external_benchmark_2026_07_11" / "results.json"
RESULT_BYTES = 158_895
RESULT_SHA256 = "ec56470bfbc27c61f582e7b78f0f96f93d58a3de7644c3ed018f103cc871be0c"

EXPECTED = {
    "guppy_caigual_standard_contract": {
        "loci": 6_877,
        "ordered": "e2133443750113e175a463df5490d32d2262c30c46f95f317cc90b124da7d338",
        "vcf": (1_492_537, "4683a85f0c81810728c85e13b541ba0e9b5a35e7e229c3a489d7987464ea9920"),
        "popmap": (460, "51164cdf7692f98a035371f8ce5c482bfda2de2be7b85139143ef05c41bb71ec"),
        "prediction": "C",
        "scores": {
            "A": 7.4791748549222515e-154,
            "B": 2.094624301339592e-37,
            "C": 1.0,
        },
        "direction_rms": 30.952734244818274,
        "gate_rms": 36.838694537873806,
        "projection": 0.8746341723507026,
        "projection_ci": [0.8540026148113955, 0.8920690134906344],
        "diagnostic": 26,
        "diagnostic_projection": 0.8936709704124564,
        "f3": 0.002360675457077613,
        "f3_ci": [0.00013506787031444936, 0.004299854508945391],
        "blocks": 23,
    },
    "guppy_caigual_within_population_polymorphism": {
        "loci": 30,
        "ordered": "caaa48c0fd276886937e210ae2d8f62a5fc7e070dc7cf9d8105b4d7a6b744c39",
        "vcf": (6_962, "865418741f8b0247d38fa94ea4d5ed8363c912598008f9e21bae1ab1409f6373"),
        "popmap": (460, "51164cdf7692f98a035371f8ce5c482bfda2de2be7b85139143ef05c41bb71ec"),
        "prediction": "B",
        "scores": {
            "A": 5.3928200858249755e-75,
            "B": 1.0,
            "C": 2.258292223999404e-244,
        },
        "direction_rms": 151.40226069954912,
        "gate_rms": 156.02788079215068,
        "projection": 0.9365762641326927,
        "projection_ci": [0.8446436573372081, 0.9905381669478136],
        "diagnostic": 0,
        "diagnostic_projection": None,
        "f3": -0.0033280363711605167,
        "f3_ci": [-0.009379741641679644, 0.0037187907714660395],
        "blocks": 16,
    },
    "guppy_taylor_standard_contract": {
        "loci": 6_696,
        "ordered": "e9dbd8c73fec4e2d5261f9a58479b46e3b3314a060b31570033d916e4bf1ae88",
        "vcf": (1_533_642, "30ec1717e7e251db0812018600d19da8eb2c08309f513892a06ad2e3afb324d1"),
        "popmap": (490, "1197e3fc32abb5593c1edca40d1d22cc95c0eb43325b88eb9bdc8ee9440936d7"),
        "prediction": "C",
        "scores": {
            "A": 7.221598560005558e-148,
            "B": 9.96497258305149e-29,
            "C": 1.0,
        },
        "direction_rms": 30.86182288727328,
        "gate_rms": 35.933346863477034,
        "projection": 0.7645435759174224,
        "projection_ci": [0.7425394610394277, 0.7848512434190703],
        "diagnostic": 27,
        "diagnostic_projection": 0.8159488820247709,
        "f3": -0.007482690492605634,
        "f3_ci": [-0.009553179447156845, -0.005560767621522535],
        "blocks": 23,
    },
    "guppy_taylor_within_population_polymorphism": {
        "loci": 22,
        "ordered": "b333cecf5dccc51b0b1984d1dfce01c7192cb0865ee6379a0223e24c7488cae9",
        "vcf": (5_513, "b06ef1b7ccd944b1bc343d4b6d7e7537b09f4786c24019eb4f4892dad6605d92"),
        "popmap": (490, "1197e3fc32abb5593c1edca40d1d22cc95c0eb43325b88eb9bdc8ee9440936d7"),
        "prediction": "B",
        "scores": {
            "A": 5.967984858909072e-156,
            "B": 1.0,
            "C": 1.6765210623120013e-258,
        },
        "direction_rms": 159.70568302344665,
        "gate_rms": 164.2563740436904,
        "projection": 0.7061959307011579,
        "projection_ci": [0.5133760873989067, 1.1836671243615637],
        "diagnostic": 0,
        "diagnostic_projection": None,
        "f3": 0.013827481735058367,
        "f3_ci": [-0.013349130400642457, 0.0587276607508524],
        "blocks": 15,
    },
}


def _reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _load():
    raw = RESULT.read_bytes()
    assert len(raw) == RESULT_BYTES
    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    return json.loads(raw, parse_constant=_reject_nonfinite)


def test_guppy_result_provenance_sources_and_outcome_are_guarded():
    result = _load()
    assert result["schema_version"] == "dnnaic-guppy-2020-external-benchmark-v1"
    assert result["git"] == {
        "commit": "48254193bb0ffbde21a5b1f72e9fc388542d7c23",
        "dirty_at_run": False,
    }
    source = result["source"]
    assert source["repository_commit"] == "ac8ec0cdf29dec539494b49d8bdf32ff6f0197f2"
    assert source["repository_tree_metadata_pin"] == "eac1fe39081906b691f857e4493864db66361b02"
    assert source["repository_license"] is None
    assert "does not reconstruct the tree" in source["repository_tree_verification"]
    assert "runtime-verified VCFs" in source["paper_release_discrepancy"]
    assert "prospective held-out" in source["locus_ascertainment_guardrail"]
    assert source["sources_record"]["canonical_lf"] == {
        "bytes": 4_765,
        "sha256": "2691009ccc844526c254c06795e5f6855bf49ffd781325fa4e1c4ea0ad805b60",
    }
    assert source["sources_record"]["working_tree"] == {
        "bytes": 4_765,
        "line_endings_normalized_for_contract": False,
        "sha256": "2691009ccc844526c254c06795e5f6855bf49ffd781325fa4e1c4ea0ad805b60",
    }
    blobs = {
        key: value["git_blob_sha1"]
        for key, value in source["verified_files"].items()
    }
    assert blobs == {
        "NCA": "471ab9e4952cfd72d9dd53e298393ef73b51632b",
        "NTY": "d5282c035bd85fe48d475df7dfa895f50696f61f",
        "PCA": "9e767e652387657fea2aad8bd23eb951eb6684fc",
        "PTY": "464d42c9e88c6ffe4bc0d6676b4914e10c25f2b8",
        "SGS": "0a63f9b8bab4607f45dca6ddf7ac66ae203d476b",
        "format_script": "dd2463fe0beb5de95d42a1cb46df31684c8b3617",
    }
    combined = source["combined_VCF_audit"]
    assert combined["variants"] == 11_417
    assert combined["samples"] == 86
    assert combined["chromosomes"] == 23
    assert combined["ordered_locus_sha256"] == (
        "3daefba0c6ac6dd3f8b285633bf09be2329ad0436b7528f5be7e6b9bc12e7c73"
    )
    assert (combined["combined_vcf"]["bytes"], combined["combined_vcf"]["sha256"]) == (
        4_305_263,
        "83a00efa9836673e0a894b794ea9b3d2820defd3ff453bafa7d55a830b53b49f",
    )
    assert combined["missing_genotypes"] == {
        "NCA": 42_125,
        "NTY": 21_959,
        "PCA": 31_345,
        "PTY": 40_625,
        "SGS": 13_353,
    }

    design = result["analysis_design"]
    assert design["experimental_flow_direction_available"] is True
    assert design["exclusive_single_edge_truth_available"] is False
    assert design["formal_direction_accuracy_eligible"] is False
    assert design["gate_truth_available"] is False
    assert design["release_locus_ascertainment_outcome_blind"] is False
    assert design["benchmark_locus_filters_outcome_blind"] is False
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
    training_hashes = result["direction_head"]["training_array_audit"]
    assert training_hashes["X.npy"]["sha256"] == (
        "8a0a54b8d827301d47235ee196026687522180a9bcce07f2c52936e9d9bb56f5"
    )
    assert training_hashes["direction.npy"]["sha256"] == (
        "a956a5bb90e147e3c0a4bf8527e0f8a3c8bd6d522fbc57f5e7a34742fdad7632"
    )
    assert result["outcome"] == {
        "abstained_panels": 4,
        "accepted_direction_calls": 0,
        "accuracy_denominator": None,
        "analytic_correlated_sensitivity_rows": 4,
        "correlated_filters_not_trials": True,
        "descriptive_nonsevere_panels": 0,
        "direction_accuracy_estimate": None,
        "ecological_recipient_units": 2,
        "gate_accuracy_estimate": None,
        "guardrail": (
            "Two drainage manipulations share SGS and each contributes standard/strict views; "
            "four rows are not four independent validations or an accuracy denominator."
        ),
        "independent_formal_accuracy_units": 0,
        "raw_candidate_C_concordant_sensitivity_rows": 2,
        "raw_head_prediction_counts": {"B": 2, "C": 2},
        "severe_OOD_panels": 4,
        "shared_source_proxy": True,
    }


def test_guppy_four_correlated_results_are_exact_and_all_abstain():
    result = _load()
    assert len(result["panels"]) == 4
    for panel in result["panels"]:
        expected = EXPECTED[panel["panel_id"]]
        audit = panel["input_audit"]
        assert panel["padze"]["n_loci_kept"] == expected["loci"]
        assert audit["ordered_locus_sha256"] == expected["ordered"]
        assert (audit["derived_vcf"]["bytes"], audit["derived_vcf"]["sha256"]) == expected["vcf"]
        assert (audit["derived_popmap"]["bytes"], audit["derived_popmap"]["sha256"]) == expected[
            "popmap"
        ]
        assert panel["simulation_head"]["predicted_class"] == expected["prediction"]
        assert panel["simulation_head"]["scores"] == expected["scores"]
        assert panel["simulation_feature_shift"]["rms_z"] == expected["direction_rms"]
        assert panel["simulation_gate"]["appreciable_score"] == 1.0
        assert panel["simulation_gate_feature_shift"]["rms_z"] == expected["gate_rms"]

        geometry = panel["model_free_comparator"]
        assert geometry["P2_projection_from_P1_toward_P3_all_loci"] == expected["projection"]
        assert geometry["P2_projection_from_P1_toward_P3_diagnostic_loci"] == expected[
            "diagnostic_projection"
        ]
        assert geometry["diagnostic_loci"] == expected["diagnostic"]
        assert geometry["f3_P2_P1_P3"] == expected["f3"]
        bootstrap = geometry["scaffold_block_bootstrap"]
        assert bootstrap["projection_all_loci"]["percentile_95_interval"] == expected[
            "projection_ci"
        ]
        assert bootstrap["f3_P2_P1_P3"]["percentile_95_interval"] == expected["f3_ci"]
        assert (
            bootstrap["seed"],
            bootstrap["requested_replicates"],
            bootstrap["scaffold_blocks"],
        ) == (20260711, 500, expected["blocks"])

        expectation = panel["external_expectation"]
        assert expectation["candidate_class"] == "C"
        assert expectation["locus_ascertainment_outcome_blind"] is False
        assert "prospective held-out" in expectation["locus_ascertainment_guardrail"]
        adjudication = panel["adjudication"]
        assert adjudication["raw_head_matches_candidate_C"] is (
            expected["prediction"] == "C"
        )
        assert adjudication["severe_OOD"] is True
        assert adjudication["natural_data_call_status"] == "abstain_severe_OOD"
        assert adjudication["direction_call_accepted"] is False
        assert adjudication["formal_direction_accuracy_eligible"] is False
        assert adjudication["gate_truth_available"] is False
        assert "accepted_direction_call" not in adjudication
