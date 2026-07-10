from __future__ import annotations

import json
from pathlib import Path


RESULT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "additional_external_benchmarks_2026_07_11"
    / "results.json"
)


def load_panels():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["git"] == {
        "commit": "98e231da8bb1dc0db6c49818fe6874917e0fcc39",
        "dirty_at_run": False,
    }
    assert result["direction_head"]["dimension"] == 54
    assert result["gate_head"]["dimension"] == 216
    assert result["sources"]["giraffe"]["verified_file"]["sha256"] == (
        "7e7f4345df0129329f99db5f05e7cd120d86e0027d3afa1bc7df80c457fb95b1"
    )
    assert result["sources"]["brook_trout"]["verified_file"]["sha256"] == (
        "ea3754560e62c9ae22c6d1ad988c75ce31a0d0382f7aa233217e41c1fac7b69c"
    )
    return {panel["panel_id"]: panel for panel in result["panels"]}


def test_giraffe_positive_fails_under_both_locus_contracts():
    panels = load_panels()
    standard = panels["giraffe_nubian_to_laikipia_reticulated_standard_contract"]
    strict = panels[
        "giraffe_nubian_to_laikipia_reticulated_within_population_polymorphism"
    ]
    assert standard["external_expectation"]["expected_class"] == "C"
    assert strict["external_expectation"]["expected_class"] == "C"
    assert standard["simulation_head"]["predicted_class"] == "A"
    assert strict["simulation_head"]["predicted_class"] == "A"
    assert standard["input_audit"]["counts"]["eligible_before_cap"] == 72_759
    assert standard["padze"]["n_loci_kept"] == 15_000
    assert strict["input_audit"]["counts"]["eligible_before_cap"] == 10_588
    assert strict["padze"]["n_loci_kept"] == 10_588
    assert standard["simulation_head"]["scores"]["C"] < 1e-10
    assert strict["simulation_head"]["scores"]["C"] < 1e-4
    assert standard["sharing_ratio_diagnostics"]["by_minimum_depth"]["8"][
        "raw_ratio_of_depth_means"
    ] > 6
    assert strict["sharing_ratio_diagnostics"]["by_minimum_depth"]["8"][
        "raw_ratio_of_depth_means"
    ] > 4


def test_brook_trout_nulls_trigger_saturated_gate_and_same_loci():
    panels = load_panels()
    nulls = [
        panels["brook_trout_lfa_to_baker_brook_null"],
        panels["brook_trout_lfr_to_baker_brook_null"],
    ]
    assert {panel["simulation_head"]["predicted_class"] for panel in nulls} == {"A"}
    assert all(panel["simulation_gate"]["appreciable_score"] > 0.999 for panel in nulls)
    assert all(panel["simulation_feature_shift"]["rms_z"] > 17 for panel in nulls)
    assert {panel["padze"]["n_loci_kept"] for panel in nulls} == {15_000}
    assert {
        panel["input_audit"]["ordered_locus_sha256"] for panel in nulls
    } == {"5000f2995707e447b42b23cbc3748f06b32253b104305f39edf655371f75e396"}
