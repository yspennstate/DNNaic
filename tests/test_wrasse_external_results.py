import hashlib
import json
import math
from pathlib import Path


RESULT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "wrasse_external_benchmark_2026_07_11"
    / "results.json"
)


def _all_floats_finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_floats_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_floats_finite(item) for item in value)
    return True


def test_wrasse_result_is_clean_guarded_and_finite():
    raw = RESULT.read_bytes()
    result = json.loads(raw)
    assert len(raw) == 911_019
    assert hashlib.sha256(raw).hexdigest() == (
        "89c7d14d20098297be9a2f1b719d1148ed9c6b2868ccfc034dda24d72a1be16b"
    )
    assert result["schema_version"] == "dnnaic-wrasse-external-benchmark-v1"
    assert result["git"] == {
        "commit": "07fbb8858a00d89521b79aef6334e52cb1cc5b3c",
        "dirty_at_run": False,
    }
    assert set(result["runtime"]["thread_environment"].values()) == {"1"}
    assert result["source"]["verified"]["files"]["vcf"]["sha256"] == (
        "c05741f03ecdb2403f173cf249eb910281632f65bccad27aaaaaf848ffb2e21a"
    )
    source = result["source"]["source_vcf_contract"]
    assert source["samples"] == 240
    assert source["variant_rows"] == source["unique_2bRAD_tag_CHROM_values"] == 4_372
    assert source["partial_or_invalid_genotype_cells"] == 0
    metadata = result["source"]["metadata_and_manifest_audit"]
    assert metadata["exact_VCF_metadata_ID_set_match"] is True
    assert metadata["population_counts"] == {
        "FKH": 40,
        "SMAU": 40,
        "SMID": 40,
        "SMKB": 40,
        "SMST": 40,
        "SMTF": 40,
    }
    labels = result["source"]["label_locus_exclusion_audit"]
    assert labels["HWE_excluded_loci"] == 15
    assert labels["NewHybrids_label_loci"] == 200
    assert labels["NewHybrids_body_rows"] == 240
    assert labels["NewHybrids_genotype_cells_exactly_match_VCF"] == 48_000
    assert labels["exclusion_sets_disjoint"] is True
    assert result["source"]["primary_source_extraction"]["derived_vcf"][
        "sha256"
    ] == "885baa9a027015b5e6869fa7fc3e02ab8dddd01909b0b9a9e99df160e5337351"

    expected_shared = {
        "primary_HWE_and_label_loci_excluded__standard_contract": (
            4_097,
            "5459b30a59326669dc35fe811fda161f6596e1d97aeda5dce2b9ba412cafa1aa",
        ),
        "primary_HWE_and_label_loci_excluded__within_population_polymorphism": (
            2_962,
            "01046aa4a57decc44904ae4a253e2c8df63f838c6a625979efb28ebfbe1cbf4d",
        ),
        "all_released_loci_sensitivity__standard_contract": (
            4_311,
            "da256f6855efce965bec0ef285490d4468f2566a8e941e2b568663a3485bb6d7",
        ),
        "all_released_loci_sensitivity__within_population_polymorphism": (
            3_103,
            "efaae46e4380dc9a49f91bfaf24fed7d954b91fac168933aaaa76cff5c7caeb5",
        ),
    }
    for key, (loci, digest) in expected_shared.items():
        audit = result["shared_locus_audits"][key]
        assert audit["counts"]["retained_after_cap"] == loci
        assert audit["ordered_locus_sha256"] == digest

    outcome = result["outcome_summary"]
    assert outcome["panels"] == outcome["severe_OOD_panels"] == 24
    assert outcome["abstained_panels"] == 24
    assert outcome["abstain_due_to_severe_OOD"] is True
    assert outcome["accuracy_estimate"] is None
    assert outcome["independent_validation_panels"] == 0
    assert outcome["unique_biological_recipient_cohorts"] == 1
    assert outcome["candidate_direction_sensitivities"] == {
        "abstained_panels": 12,
        "n": 12,
        "raw_OOD_head_matches_candidate_C": 0,
        "raw_OOD_head_prediction_counts": {"A": 12, "B": 0, "C": 0},
    }
    assert outcome["role_swapped_comparators"] == {
        "abstained_panels": 12,
        "n": 12,
        "raw_OOD_gate_threshold_crossings_at_0.5": 12,
    }
    assert outcome["same_locus_role_changed_contrasts"] == {
        "gate_probability_ceiling_ties": 12,
        "n": 12,
        "raw_gate_delta_range": [0.0, 0.0],
    }

    candidate = []
    comparators = []
    grouped_hashes = {}
    for panel in result["panels"]:
        expectation = panel["external_expectation"]
        assert expectation["accuracy_eligible"] is False
        assert expectation["expected_gate"] is None
        assert panel["adjudication"]["accuracy_eligible"] is False
        assert panel["adjudication"]["severe_OOD"] is True
        assert panel["adjudication"]["natural_data_call_status"] == "abstain_severe_OOD"
        assert panel["simulation_head"]["predicted_class"] == "A"
        assert panel["simulation_gate"]["appreciable_score"] == 1.0
        assert panel["model_free_comparator"][
            "f3_P2_P1_P3_finite_called_copy_corrected"
        ] > 0
        bootstrap = panel["model_free_comparator"]["iid_locus_bootstrap"]
        assert "not chromosome-block" in bootstrap["guardrail"]
        scope_filter = (
            expectation["scope"],
            panel["panel_id"].rsplit("_", 2)[-2]
            if panel["panel_id"].endswith("standard_contract")
            else "within_population_polymorphism",
        )
        grouped_hashes.setdefault(scope_filter, set()).add(
            panel["input_audit"]["ordered_locus_sha256"]
        )
        if expectation["benchmark_role"] == "candidate_direction_sensitivity":
            candidate.append(panel)
            assert expectation["candidate_class"] == "C"
            assert panel["adjudication"]["matches_candidate_reference"] is False
        else:
            comparators.append(panel)
            assert expectation["benchmark_role"] == "role_swapped_comparator"
            assert expectation["candidate_class"] is None
    assert len(candidate) == len(comparators) == 12
    assert all(len(hashes) == 1 for hashes in grouped_hashes.values())

    primary_standard_candidate = [
        panel
        for panel in candidate
        if panel["external_expectation"]["scope"]
        == "primary_HWE_and_label_loci_excluded"
        and panel["panel_id"].endswith("standard_contract")
    ]
    primary_standard_comparators = [
        panel
        for panel in comparators
        if panel["external_expectation"]["scope"]
        == "primary_HWE_and_label_loci_excluded"
        and panel["panel_id"].endswith("standard_contract")
    ]
    candidate_projection = [
        panel["model_free_comparator"]["P2_projection_from_P1_toward_P3_all_loci"]
        for panel in primary_standard_candidate
    ]
    comparator_projection = [
        panel["model_free_comparator"]["P2_projection_from_P1_toward_P3_all_loci"]
        for panel in primary_standard_comparators
    ]
    assert min(candidate_projection) > max(comparator_projection)
    assert _all_floats_finite(result)
