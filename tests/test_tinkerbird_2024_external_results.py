import hashlib
import json
import math
from pathlib import Path


RESULT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "tinkerbird_2024_external_benchmark_2026_07_11"
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


def test_tinkerbird_2024_result_is_clean_guarded_and_finite():
    raw = RESULT.read_bytes()
    result = json.loads(raw)
    assert len(raw) == 584_071
    assert hashlib.sha256(raw).hexdigest() == (
        "1c9573456f8a7c05e61d93a574d1c578e93f8f2cfc1f2e7e731a79563dded7dd"
    )
    assert result["schema_version"] == "dnnaic-tinkerbird-2024-external-benchmark-v1"
    assert result["git"] == {
        "commit": "95e437d2bb3f1515e004d964c8458f3bd50083e3",
        "dirty_at_run": False,
    }
    assert set(result["runtime"]["thread_environment"].values()) == {"1"}
    assert result["source"]["verified_files"]["vcf"]["sha256"] == (
        "6438a889ad91b865237e6a4e5169bfbe61e9eea884696fa7fdcfef83d68c7c30"
    )
    assert result["source"]["source_vcf_contract"][
        "chromosome_assignment_counts"
    ] == {
        "S76_non_numbered": 31,
        "W": 5,
        "Z": 1_157,
        "anchored_autosomes": 82_309,
        "unplaced_scaffolds": 610,
    }
    thinning = result["source"]["one_per_RAD_locus_extraction"]
    assert thinning["retained_variant_rows"] == 23_395
    assert thinning["derived_vcf"]["sha256"] == (
        "489174990961616b3c79ee2eebab29d44bd0e13c5b7c720c321bd82ddc16923c"
    )
    assert thinning["ordered_key_sha256"] == (
        "a78ee8f7f57e81745d7f745d3c600435ed800c1611e5bf9a1fe57468253200ef"
    )

    outcome = result["outcome_summary"]
    assert outcome["panels"] == outcome["severe_OOD_panels"] == 16
    assert outcome["abstain_due_to_severe_OOD"] is True
    assert outcome["accuracy_estimate"] is None
    assert outcome["direction_candidate_panels"] == {
        "matches_candidate": 0,
        "n": 8,
        "prediction_counts": {"A": 4, "B": 4, "C": 0},
    }
    assert outcome["gate_near_null_panels"]["n"] == 8
    assert outcome["gate_near_null_panels"]["called_appreciable_at_0.5"] == 8
    assert outcome["gate_near_null_panels"]["minimum_score"] > 0.999999999

    direction = []
    near_null = []
    for panel in result["panels"]:
        expectation = panel["external_expectation"]
        assert expectation["accuracy_eligible"] is False
        assert panel["adjudication"]["accuracy_eligible"] is False
        assert panel["adjudication"]["severe_OOD"] is True
        assert panel["adjudication"]["natural_data_call_status"] == "abstain_severe_OOD"
        assert panel["simulation_gate"]["interpretation"].endswith(
            "not a probability or posterior"
        )
        comparator = panel["model_free_comparator"]
        assert "f3_P2_P1_P3_unbiased_for_P2_sampling" not in comparator
        assert "f3_P2_P1_P3_finite_called_copy_corrected" in comparator
        assert "label reuse" not in comparator["chromosome_block_bootstrap"]["guardrail"]
        p3_calls = panel["input_audit"]["population_called_copy_counts"][
            "PusillusRefExact8"
        ]
        assert p3_calls["individuals"] == 8
        assert p3_calls["minimum"] == p3_calls["maximum"] == 16
        if expectation["scope"] == "autosome_one_per_RAD_locus":
            assert comparator["n_loci"] == comparator["RAD_loci"]
            assert expectation["scope_role"].startswith("primary:")
        else:
            assert "sensitivity:" in expectation["scope_role"]
        if expectation.get("candidate_class") is not None:
            direction.append(panel)
        if expectation["benchmark_role"] == "gate_near_null":
            near_null.append(panel)

    assert len(direction) == len(near_null) == 8
    assert {panel["external_expectation"]["candidate_class"] for panel in direction} == {
        "C"
    }
    assert not any(panel["adjudication"]["matches_candidate_majority"] for panel in direction)
    assert all(panel["simulation_gate"]["called_at_0.5"] for panel in near_null)

    overlap = result["locus_overlap_audit"]
    assert overlap["cross_scope_locus_overlap"]["direction"]["standard_contract"][
        "intersection"
    ] == 4_218
    assert overlap["cross_family_locus_overlap"]["autosome_one_per_RAD_locus"][
        "standard_contract"
    ]["all_three_intersection"] == 7_404
    assert overlap["cross_family_locus_overlap"]["autosome_one_per_RAD_locus"][
        "within_population_polymorphism"
    ]["all_three_intersection"] == 893
    assert _all_floats_finite(result)
