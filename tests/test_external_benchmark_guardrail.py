"""Keep the external benchmark controls and depth contract visible."""
from __future__ import annotations

import json
from pathlib import Path


RESULT = (
    Path(__file__).parents[1]
    / "results"
    / "external_benchmarks_2026_07_10"
    / "results.json"
)


def panels():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert "not classifier validation" in result["status"]
    head = result["simulation_head_audit"]
    assert head["depth_grid"] == list(range(2, 17))
    assert head["dimension"] == 54
    assert head["training_n"] == 2700
    assert head["training_array_audit"]["X.npy"]["sha256"] == (
        "8a0a54b8d827301d47235ee196026687522180a9bcce07f2c52936e9d9bb56f5"
    )
    return {panel["panel_id"]: panel for panel in result["panels"]}


def test_duck_positive_disagrees_and_both_panels_are_severely_ood():
    result = panels()
    positive = result["andean_duck_beta_positive"]
    negative = result["andean_duck_alpha_negative"]
    assert positive["external_expectation"]["expected_class_if_transferable"].startswith("C")
    assert positive["simulation_head"]["predicted_class"] == "B"
    assert negative["simulation_head"]["predicted_class"] == "A"
    assert positive["simulation_feature_shift"]["max_abs_z"] > 100
    assert negative["simulation_feature_shift"]["max_abs_z"] > 100
    assert positive["padze"]["n_loci_kept"] == 2298
    assert negative["padze"]["n_loci_kept"] == 499


def test_bean_positives_and_null_separate_on_identical_polymorphic_loci():
    result = panels()
    ids = [
        "runner_bean_cdmx_wild_to_cultivar_positive",
        "runner_bean_tepoz_wild_to_cultivar_replicate",
        "runner_bean_published_null",
    ]
    bean = [result[panel_id] for panel_id in ids]
    assert len({panel["input_audit"]["ordered_locus_sha256"] for panel in bean}) == 1
    assert {panel["padze"]["n_loci_kept"] for panel in bean} == {12_284}
    assert [panel["simulation_head"]["predicted_class"] for panel in bean] == ["C", "C", "B"]
    assert bean[0]["simulation_head"]["scores"]["C"] > 0.999
    assert bean[1]["simulation_head"]["scores"]["C"] > 0.95
    assert bean[2]["simulation_head"]["scores"]["C"] < 1e-40

    raw = [
        panel["sharing_ratio_diagnostics"]["by_minimum_depth"]["8"]["raw_ratio_of_depth_means"]
        for panel in bean
    ]
    assert raw[0] > 3 and raw[1] > 3
    assert raw[2] < 0.5
    assert all(
        panel["input_audit"]["counts"]["not_polymorphic_in_every_panel"] > 0
        for panel in bean
    )
