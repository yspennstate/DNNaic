import hashlib
import json
import math
from pathlib import Path


RESULT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "hantarcticus_2024_external_benchmark_2026_07_11"
    / "results.json"
)
CODE_COMMIT = "c20b987778afcc68da24a9136164dfccc2e066a9"


def _walk_finite(value):
    if isinstance(value, dict):
        for child in value.values():
            _walk_finite(child)
    elif isinstance(value, list):
        for child in value:
            _walk_finite(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def test_hantarcticus_2024_result_bundle_is_frozen_and_guarded():
    raw = RESULT.read_bytes()
    assert len(raw) == 198_806
    assert hashlib.sha256(raw).hexdigest() == (
        "0c086fa425450623358182c4eff190ddd0f219c1defe511355d90bfebc054151"
    )
    result = json.loads(raw)
    _walk_finite(result)

    assert result["schema_version"] == "dnnaic-hantarcticus-2024-external-benchmark-v1"
    assert result["git"] == {"commit": CODE_COMMIT, "dirty_at_run": False}
    assert result["runtime"]["thread_environment"] == {
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }

    source = result["source"]
    assert source["verified"]["vcf"]["sha256"] == (
        "48d832ade62ef3ad21ced7869e6f2a9e5c418593978e6260725be0ba02f998a5"
    )
    assert source["verified"]["matrices"]["sha256"] == (
        "3ac56229b68ff9c77de9517015e52dfa766bc3e5590cd4b5e502e8a6aefb3456"
    )
    audit = source["source_vcf_contract"]
    assert (audit["samples"], audit["variant_rows"]) == (143, 20_778)
    assert (audit["genotype_cells"], audit["fully_missing_genotype_cells"]) == (
        2_971_254,
        344_218,
    )
    assert audit["partial_or_invalid_genotype_cells"] == 0
    assert audit["samples_above_0_25_missingness"] == []
    assert audit["ordered_CHROM_POS_ID_REF_ALT_sha256"] == (
        "979472625148268308c23f592bb792b05955143252149af992a63f008782718e"
    )
    mapping = source["mapping_audit"]
    assert mapping["VCF_total"] == 143
    assert mapping["paper_Table1_total"] == 133
    assert mapping["VCF_site_counts"]["FIB"] == 19
    assert mapping["paper_Table1_site_counts"]["FIB"] == 9

    archive = source["biophysical_archive_audit"]
    assert archive["daily_connectivity_matrices"] == 4_000
    assert archive["nonmetadata_member_name_size_crc_sha256"] == (
        "77520ae3e34eec41ddd5273fc49481ac5dc9dac06113c66d3c6f1bff0714f189"
    )
    assert archive["day100_inventory_sha256"] == (
        "17215bc00c9e2544aa6f238c1420fe3b549a8e8cb5d6c7f88d56b70f7d699f36"
    )
    assert archive["sum_40_day100_matrices_sha256"] == (
        "4a02f1a77ff8b7630772e6969e959cf479cb44b6ecb5089fb7d913cd0f7133d0"
    )
    expected_edges = {
        "doi_to_fha_hos": {
            "counts": (92, 10, 32, 7, 0.023, 0.0025),
            "comparable": 4,
            "undefined": [],
        },
        "ais_to_hos_doi": {
            "counts": (50, 13, 19, 7, 0.0125, 0.0032500000000000003),
            "comparable": 3,
            "undefined": [
                {
                    "forward_defined": True,
                    "reciprocal_defined": False,
                    "season": "Eday_1780",
                }
            ],
        },
    }
    for panel_id, expected in expected_edges.items():
        edge = archive["candidate_edges"][panel_id]
        observed = (
            edge["raw_day100_settlers_across_40_runs"],
            edge["raw_reciprocal_day100_settlers_across_40_runs"],
            edge["runs_with_forward_settlers"],
            edge["runs_with_reciprocal_settlers"],
            edge["four_season_mean_fraction_of_100_released"],
            edge["four_season_mean_reciprocal_fraction_of_100_released"],
        )
        assert observed == expected["counts"]
        assert edge["all_four_seasons_forward_exceeds_reciprocal"] is True
        assert edge["destination_conditional_comparable_seasons"] == expected[
            "comparable"
        ]
        assert edge["destination_conditional_undefined_seasons"] == expected[
            "undefined"
        ]
        assert (
            edge["all_comparable_seasons_forward_destination_share_exceeds_reciprocal"]
            is True
        )
    assert archive["candidate_edges"]["doi_to_fha_hos"][
        "all_four_seasons_destination_conditional_comparable"
    ] is True
    assert archive["candidate_edges"]["ais_to_hos_doi"][
        "all_four_seasons_destination_conditional_comparable"
    ] is False

    summary = result["outcome_summary"]
    assert summary["analytic_sensitivity_runs"] == 4
    assert summary["unique_biological_systems"] == 1
    assert summary["correlated_candidate_comparisons"] == 2
    assert summary["independent_validation_panels"] == 0
    assert summary["accuracy_estimate"] is None
    assert summary["accuracy_available"] is False
    assert summary["severe_OOD_panels"] == 4
    assert summary["abstained_panels"] == 4
    assert summary["raw_OOD_head_prediction_counts"] == {"A": 4, "B": 0, "C": 0}
    assert summary["raw_OOD_gate_threshold_crossings_at_0.5"] == 4

    expected_panels = {
        "hantarcticus_2024_doi_to_fha_hos_standard_contract": (
            16_299,
            "bd8019539f8bda8108a8a204fd7f610ba492ceabc3b4438fd7137bb1183cce86",
            18.56381271739207,
            17.621349701853745,
            0.48015889455876315,
            0.7152777777777778,
            0.002193219299302999,
        ),
        "hantarcticus_2024_doi_to_fha_hos_within_population_polymorphism": (
            12_074,
            "32bdc7739277a66b4f83faa33f8be768f9210a7eed2528ffc3d26872ac74c20f",
            21.240329512596468,
            21.91038133193092,
            0.47894713749570283,
            0.7152777777777778,
            0.0022582892808202075,
        ),
        "hantarcticus_2024_ais_to_hos_doi_standard_contract": (
            16_301,
            "8a161bcfdf0d4ab393841f0167b111e74e1793543c793cd34a6ebe5f2504a31f",
            18.34069954607346,
            17.389996948986038,
            0.34829777162057235,
            0.6076388888888888,
            0.002073957283648253,
        ),
        "hantarcticus_2024_ais_to_hos_doi_within_population_polymorphism": (
            11_931,
            "23d34f029bebba494acd96795d78bf633c63a92aada49c4a8b5ef87387e65933",
            21.154126934748884,
            21.824034012393877,
            0.3642751785619913,
            0.6076388888888888,
            0.0007499315389234234,
        ),
    }
    assert [panel["panel_id"] for panel in result["panels"]] == list(expected_panels)
    for panel in result["panels"]:
        loci, locus_hash, direction_rms, gate_rms, projection, maximum, f3 = expected_panels[
            panel["panel_id"]
        ]
        assert panel["padze"]["n_loci_kept"] == loci
        assert panel["input_audit"]["ordered_locus_sha256"] == locus_hash
        assert panel["external_expectation"]["expected_gate"] is None
        assert panel["external_expectation"]["accuracy_eligible"] is False
        assert panel["simulation_head"]["predicted_class"] == "A"
        assert panel["simulation_gate"]["appreciable_score"] == 1.0
        assert panel["adjudication"]["accuracy_eligible"] is False
        assert panel["adjudication"]["severe_OOD"] is True
        assert panel["adjudication"]["natural_data_call_status"] == "abstain_severe_OOD"
        assert panel["simulation_feature_shift"]["rms_z"] == direction_rms
        assert panel["simulation_gate_feature_shift"]["rms_z"] == gate_rms
        geometry = panel["model_free_comparator"]
        assert geometry["P2_projection_from_P1_toward_P3_all_loci"] == projection
        assert geometry["maximum_abs_P3_minus_P1_frequency"] == maximum
        assert geometry["diagnostic_loci"] == 0
        assert geometry["P2_projection_from_P1_toward_P3_diagnostic_loci"] is None
        assert geometry["f3_P2_P1_P3_finite_called_copy_corrected"] == f3
        assert f3 > 0
        assert geometry["iid_locus_bootstrap"][
            "f3_finite_called_copy_corrected"
        ]["percentile_95_interval"][0] > 0
