import hashlib
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
RESULT = REPO / "results" / "yellowstone_2019_external_benchmark_2026_07_11" / "results.json"
RESULT_BYTES = 274_675
RESULT_SHA256 = "a06707aa389281df1f75ae78ad9297408fd1a0dde3da2be690ad9575dbe9c671"

EXPECTED = {
    "yellowstone_main_source_GT_standard_contract": {
        "loci": 11_758,
        "ordered": "9ef1e1f65321ab60540f2c90e1aa12d4f78ac2449c8686a0a295d78c7e75a274",
        "vcf": (6_985_231, "15b75a25ea71df90e2c8717ba67f0c9439cb08124cc396d9ceba199a745046f6"),
        "pop": (2_007, "a6e1fad0047e4c0beedeae2dc165eba24acdf9853e4f57626677a7af90ff80bd"),
        "prediction": "C",
        "scores": {"A": 3.6406944166606586e-72, "B": 2.104264483567692e-50, "C": 1.0},
        "direction_rms": 18.46228424542268,
        "gate": 1.0,
        "gate_rms": 23.572798492968758,
        "projection": 0.7635691170364513,
        "projection_ci": [0.7547666653279745, 0.7736158272751034],
        "diagnostic": 677,
        "diagnostic_projection": 0.7910392617505657,
        "f3": -0.021297262268159346,
        "f3_ci": [-0.022557807297044524, -0.02000176699216101],
    },
    "yellowstone_main_source_GT_within_population_polymorphism": {
        "loci": 2_591,
        "ordered": "bad416e8a0ed253a1b87c9089453bd40cf7ed393b3aea5fd3b8eae96b52b7b42",
        "vcf": (1_540_619, "a8e40b0a3aa493aee11fbfcf8a8bfe516422b77fa60794eddaac496a57884be9"),
        "pop": (2_007, "a6e1fad0047e4c0beedeae2dc165eba24acdf9853e4f57626677a7af90ff80bd"),
        "prediction": "C",
        "scores": {"A": 3.822709028324694e-83, "B": 3.674237639725522e-30, "C": 1.0},
        "direction_rms": 20.849353210166697,
        "gate": 1.0,
        "gate_rms": 30.5827956059364,
        "projection": 0.7807760556432648,
        "projection_ci": [0.7711740357705111, 0.7924251903062469],
        "diagnostic": 35,
        "diagnostic_projection": 0.7861860527504719,
        "f3": -0.02676958505331159,
        "f3_ci": [-0.030366324516997657, -0.02357333832627145],
    },
    "yellowstone_main_unique_PL_argmin_standard_contract": {
        "loci": 12_170,
        "ordered": "7ac1f0ef6d59f80bb1bc78a061aa691663c996a1d4f9630e7d40222824678050",
        "vcf": (7_229_934, "bfdcb9a8ba6614919606747517cd6e196efb9bdcb724531f640f20817d2df10b"),
        "pop": (2_007, "a6e1fad0047e4c0beedeae2dc165eba24acdf9853e4f57626677a7af90ff80bd"),
        "prediction": "C",
        "scores": {"A": 5.510290359797524e-87, "B": 6.2306851026958546e-68, "C": 1.0},
        "direction_rms": 22.12606606701905,
        "gate": 1.0,
        "gate_rms": 27.67756616157849,
        "projection": 0.6682046548218018,
        "projection_ci": [0.6580200212793368, 0.6799162022108765],
        "diagnostic": 898,
        "diagnostic_projection": 0.720226683629716,
        "f3": -0.029738490096258056,
        "f3_ci": [-0.03143474147823404, -0.028200167028765016],
    },
    "yellowstone_main_unique_PL_argmin_within_population_polymorphism": {
        "loci": 3_653,
        "ordered": "822b251fe3411f576331534511101e3ecd8c02194a8d8a785b57027b4c701bef",
        "vcf": (2_171_362, "5fc1bcba5b0f47abe35eaa4c1df734294144fc52fe152aa799f27ad7fa3870cc"),
        "pop": (2_007, "a6e1fad0047e4c0beedeae2dc165eba24acdf9853e4f57626677a7af90ff80bd"),
        "prediction": "C",
        "scores": {"A": 4.083946783718065e-101, "B": 1.798298857835287e-42, "C": 1.0},
        "direction_rms": 22.094713439095663,
        "gate": 1.0,
        "gate_rms": 31.402535790121107,
        "projection": 0.694418002007789,
        "projection_ci": [0.6808186956256895, 0.7074256069578662],
        "diagnostic": 140,
        "diagnostic_projection": 0.723823854280132,
        "f3": -0.03664860662328105,
        "f3_ci": [-0.04052222073194423, -0.03339983273415285],
    },
    "yellowstone_tensleep_reference_source_GT_standard_contract": {
        "loci": 9_642,
        "ordered": "eb29d4d242b91ff443cf161a77b14575f4c024bbd955b7e5c91e8c9cdf7c5c42",
        "vcf": (4_108_114, "8b4cd4a189a59bde19493e6ea846e9778a6da36fcc1dcc3791f522bc3378a420"),
        "pop": (1_358, "811d6236cb8a8c38150289e3bbd0ae50b25030e105dcf99bea9a7fbce93b2157"),
        "prediction": "C",
        "scores": {"A": 1.2016641330720033e-45, "B": 1.865102021864812e-46, "C": 1.0},
        "direction_rms": 15.746170450180504,
        "gate": 1.0,
        "gate_rms": 14.99456153609515,
        "projection": 0.6609975786644456,
        "projection_ci": [0.648655144604619, 0.6722600596294804],
        "diagnostic": 41,
        "diagnostic_projection": 0.7504249340570467,
        "f3": -0.007835942029886994,
        "f3_ci": [-0.008683766980996132, -0.006975420765629532],
    },
    "yellowstone_big_direct_stock_source_GT_standard_contract": {
        "loci": 9_721,
        "ordered": "4ad22ffff991cf8515ef1d0230e61f921b8c06b2331f72c98d4786526fc612d8",
        "vcf": (4_258_457, "9d36721a33393a9f19deab039c148ad2102fc9954de2c2c21ebf7cc14df1e6b4"),
        "pop": (1_400, "fbd36f7cf328de7b099c14aeb2638940f1e164b42fada6b6d8b4743f524cbb7a"),
        "prediction": "C",
        "scores": {"A": 9.418208615986073e-45, "B": 2.780098420715265e-48, "C": 1.0},
        "direction_rms": 17.10801668825018,
        "gate": 1.0,
        "gate_rms": 17.135175741334116,
        "projection": 0.7778281635959968,
        "projection_ci": [0.767360936515718, 0.7877394237737498],
        "diagnostic": 41,
        "diagnostic_projection": 0.858410133034308,
        "f3": -0.003991122820203574,
        "f3_ci": [-0.004673625656391551, -0.0032437894764513653],
    },
    "yellowstone_candidate_null_source_GT_standard_contract": {
        "loci": 7_563,
        "ordered": "6965a13e3ee0f0f5706286e22f983657c99aedcf4a6cd51071700b2e3df576e2",
        "vcf": (3_313_401, "1b985601a2f1cfd096263854dcafb336849a1830fa78cceae0aae51313764b5f"),
        "pop": (1_461, "2f7d79812915b8307c5310301c3a5adcb0700afb92f2beb79b2a398357ba8715"),
        "prediction": "B",
        "scores": {"A": 8.106224100179532e-12, "B": 0.9999999999918938, "C": 2.5214196747929773e-24},
        "direction_rms": 14.827706748186607,
        "gate": 0.9999188723558187,
        "gate_rms": 18.162333222874658,
        "projection": -0.03995033546484742,
        "projection_ci": [-0.049879628462150744, -0.029407298220497535],
        "diagnostic": 41,
        "diagnostic_projection": 0.023290357786555218,
        "f3": 0.01228830548624287,
        "f3_ci": [0.011230060519791132, 0.013437840556200545],
    },
}


def _reject_nonfinite(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _load():
    raw = RESULT.read_bytes()
    assert len(raw) == RESULT_BYTES
    assert hashlib.sha256(raw).hexdigest() == RESULT_SHA256
    return json.loads(raw, parse_constant=_reject_nonfinite)


def test_yellowstone_result_provenance_sources_and_outcome_are_guarded():
    result = _load()
    assert result["schema_version"] == "dnnaic-yellowstone-2019-external-benchmark-v1"
    assert result["git"] == {
        "commit": "0c78eb5363ce2d39be69d0c1e1b2e53e5cae3b66",
        "dirty_at_run": False,
    }
    assert result["source"]["sources_record"]["canonical_lf"] == {
        "bytes": 4_133,
        "sha256": "3633e5aa098b2223a79361c239b87ed0d46b9239262280808ec9daac9fe2a62e",
    }
    assert result["source"]["verified_files"]["vcf"]["sha256"] == (
        "c0341a3a9dc11206e460907bea3d618f535245c7993d9cfce39c12c2b9b8bc86"
    )
    assert result["source"]["corrigendum_audit"].endswith("remains unaudited")
    assert "not prospectively held out" in result["source"]["locus_ascertainment_guardrail"]
    assert result["analysis_design"]["direction_truth_available"] is False
    assert result["analysis_design"]["exclusive_single_edge_truth_available"] is False
    assert result["analysis_design"]["formal_direction_accuracy_eligible"] is False
    assert result["analysis_design"]["gate_truth_available"] is False
    assert result["analysis_design"]["locus_ascertainment_is_prospective_held_out"] is False
    assert result["analysis_design"]["published_same_data_RBT_ancestry_comparator"] == 0.7382
    assert "same SNPs" in result["analysis_design"]["published_comparator_guardrail"]
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
        "abstained_panels": 7,
        "accepted_direction_calls": 0,
        "accuracy_denominator": None,
        "accuracy_guardrail": (
            "Seven representation/filter/reference/target rows reuse one two-species system and "
            "supply no accuracy denominator. Published ancestry summaries reuse the same SNPs."
        ),
        "analytic_correlated_sensitivity_rows": 7,
        "candidate_C_sensitivity_rows": 6,
        "correlated_sensitivities_not_trials": True,
        "descriptive_nonsevere_panels": 0,
        "direction_accuracy_estimate": None,
        "gate_accuracy_estimate": None,
        "independent_direction_truth_units": 0,
        "independent_gate_truth_units": 0,
        "raw_candidate_C_concordant_sensitivity_rows": 6,
        "raw_head_prediction_counts": {"B": 1, "C": 6},
        "severe_OOD_panels": 7,
        "unique_biological_systems": 1,
    }


def test_yellowstone_representation_converter_and_batch_diagnostics_are_exact():
    result = _load()
    audit = result["source"]["VCF_audit"]
    assert audit["normalized_representations"]["source_GT"]["sha256"] == (
        "9e06f9fb38347e4867cdd4e8b11678517e96df75f8e3f461271393fdcde0bbb3"
    )
    assert audit["normalized_representations"]["unique_PL_argmin"]["sha256"] == (
        "b5acf651bca6e842fa16e32038a6e7780813cad8f7070a3050400edc838385d9"
    )
    assert audit["converter_cell_audit"]["converter_zeroed_informative_PL_cells"] == 1_789_601
    assert audit["converter_zeroed_informative_PL_GQ_range"] == [1, 9]
    assert audit["converter_zeroed_informative_PL_all_DP_positive"] is True
    diagnostics = audit["representation_diagnostics"]
    assert diagnostics["SFOwlCreek"]["source_GT_call_rate"] == 0.7900886069068346
    assert diagnostics["Trout"]["source_GT_call_rate"] == 0.5483128876111447
    assert diagnostics["StoryHatchery"]["source_GT_call_rate"] == 0.7173377546186641
    assert diagnostics["SFOwlCreek"]["GT_PL_concordance_given_both"] == 0.9961292226475723
    assert diagnostics["Trout"]["GT_PL_concordance_given_both"] == 0.9872750530930587
    assert diagnostics["StoryHatchery"]["GT_PL_concordance_given_both"] == 0.9900520245533871
    assert result["source"]["metadata_audit"]["main_population_library_sets_disjoint"] is True
    overlap = result["analysis_design"]["representation_overlap"]
    assert overlap["standard_contract"]["intersection"] == 11_758
    assert overlap["standard_contract"]["fraction_source_GT_in_PL"] == 1.0
    assert overlap["within_population_polymorphism"]["intersection"] == 2_584
    assert overlap["within_population_polymorphism"]["ordered_intersection_sha256"] == (
        "17ee36a5d5213951e3851d7ece5be6f2a189b8cc97455c3d1248c8ae33c3c662"
    )


def test_yellowstone_seven_correlated_panel_results_are_exact_and_all_abstain():
    result = _load()
    assert len(result["panels"]) == 7
    for panel in result["panels"]:
        expected = EXPECTED[panel["panel_id"]]
        audit = panel["input_audit"]
        assert panel["padze"]["n_loci_kept"] == expected["loci"]
        assert audit["ordered_locus_sha256"] == expected["ordered"]
        assert (audit["derived_vcf"]["bytes"], audit["derived_vcf"]["sha256"]) == expected["vcf"]
        assert (audit["derived_popmap"]["bytes"], audit["derived_popmap"]["sha256"]) == expected["pop"]
        assert panel["simulation_head"]["predicted_class"] == expected["prediction"]
        assert panel["simulation_head"]["scores"] == expected["scores"]
        assert panel["simulation_feature_shift"]["rms_z"] == expected["direction_rms"]
        assert panel["simulation_gate"]["appreciable_score"] == expected["gate"]
        assert panel["simulation_gate_feature_shift"]["rms_z"] == expected["gate_rms"]

        geometry = panel["model_free_comparator"]
        assert geometry["P2_projection_from_P1_toward_P3_all_loci"] == expected["projection"]
        assert geometry["P2_projection_from_P1_toward_P3_diagnostic_loci"] == expected[
            "diagnostic_projection"
        ]
        assert geometry["diagnostic_loci"] == expected["diagnostic"]
        assert geometry["f3_P2_P1_P3"] == expected["f3"]
        bootstrap = geometry["scaffold_block_bootstrap"]
        assert bootstrap["projection_all_loci"]["percentile_95_interval"] == expected["projection_ci"]
        assert bootstrap["f3_P2_P1_P3"]["percentile_95_interval"] == expected["f3_ci"]
        assert (bootstrap["seed"], bootstrap["requested_replicates"], bootstrap["scaffold_blocks"]) == (
            20260711,
            500,
            29,
        )
        guardrail = geometry["finite_sample_and_uncertainty_guardrail"]
        assert "plug-in product" in guardrail
        assert "resamples chromosomes, not fish" in guardrail

        adjudication = panel["adjudication"]
        assert adjudication["severe_OOD"] is True
        assert adjudication["natural_data_call_status"] == "abstain_severe_OOD"
        assert adjudication["direction_truth_available"] is False
        assert adjudication["gate_truth_available"] is False
        assert adjudication["direction_call_accepted"] is False
        assert adjudication["formal_direction_accuracy_eligible"] is False
        assert "accepted_direction_call" not in adjudication
