import json
import math
from pathlib import Path


RESULT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "tinkerbird_external_benchmark_2026_07_11"
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


def test_tinkerbird_result_is_clean_guarded_and_finite():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["schema_version"] == "dnnaic-tinkerbird-external-benchmark-v2"
    assert result["git"] == {
        "commit": "23518cfc88489da6d222c519d8e7f13159770346",
        "dirty_at_run": False,
    }
    assert set(result["runtime"]["thread_environment"].values()) == {"1"}
    assert result["published_sample_audit"]["accuracy_eligible"] is False
    assert result["population_design"]["panmictic_tree_leaf_contract_satisfied"] is False
    assert result["source"]["verified_file"]["sha256"] == (
        "51144aabaddac820269af2f8ff5648393b69a20be3c0398a72ca4d9c83756a51"
    )
    assert result["source"]["source_vcf_contract"]["scaffold_assignment_counts"] == {
        "Z": 1_533,
        "anchored_autosome": 57_913,
        "unplaced": 45_487,
    }
    thinning = result["source"]["one_per_scaffold_extraction"]
    assert thinning["source_variant_rows"] == 57_913
    assert thinning["source_scaffolds"] == 8_815
    assert thinning["retained_scaffolds"] == 8_744
    assert thinning["scaffolds_without_structurally_eligible_SNP"] == 71

    panels = {panel["panel_id"]: panel for panel in result["panels"]}
    assert len(panels) == 4
    assert {
        panel["padze"]["n_loci_kept"] for panel in panels.values()
    } == {15_000, 8_638, 3_985}
    for panel in panels.values():
        assert panel["external_expectation"]["candidate_class"] == "C"
        assert panel["external_expectation"]["accuracy_eligible"] is False
        assert panel["simulation_head"]["predicted_class"] == "A"
        assert panel["simulation_gate"]["appreciable_score"] == 1.0
        assert panel["adjudication"]["matches_candidate_majority"] is False
        assert panel["adjudication"]["severe_OOD"] is True
        assert panel["adjudication"]["accuracy_eligible"] is False
    assert _all_floats_finite(result)
