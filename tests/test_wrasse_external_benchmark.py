import json
from pathlib import Path

import numpy as np

from scripts import wrasse_external_benchmark as wrasse


def test_wrasse_source_and_scope_contracts_are_pinned():
    assert wrasse.ARCHIVE == {
        "key": "dryad_tv553_v1.zip",
        "bytes": 24_227_679,
        "sha256": "15474df5ba0808db77f403f56076a17b7c3821e31a62a9c996e0a81468bd1620",
        "md5": "e042b4c3cbe108d2f2f743dfa640c303",
    }
    assert wrasse.FILES["vcf"]["sha256"] == (
        "c05741f03ecdb2403f173cf249eb910281632f65bccad27aaaaaf848ffb2e21a"
    )
    assert wrasse.SOURCE_CONTRACT["variant_rows"] == 4_372
    assert wrasse.SOURCE_CONTRACT["primary_source_loci"] == 4_157
    assert wrasse.SCOPE_ROLES[
        "primary_HWE_and_label_loci_excluded"
    ].startswith("primary:")
    assert "circular" in wrasse.SCOPE_ROLES["all_released_loci_sensitivity"]
    sources = json.loads(wrasse.SOURCES_RECORD.read_text(encoding="utf-8"))
    assert sources["archive"]["sha256"] == wrasse.ARCHIVE["sha256"]
    assert sources["files"]["vcf"]["sha256"] == wrasse.FILES["vcf"]["sha256"]
    assert sources["files"]["newhybrid"]["bytes"] == wrasse.FILES["newhybrid"][
        "bytes"
    ]
    assert sources["papers"]["selection_update_2026"] == "10.1111/eva.70214"


def test_wrasse_panel_config_is_balanced_and_guarded():
    specs = wrasse.read_panel_specs()
    assert len(specs) == 6
    positives = [
        spec
        for spec in specs.values()
        if spec["benchmark_role"] == "candidate_direction_sensitivity"
    ]
    controls = [
        spec
        for spec in specs.values()
        if spec["benchmark_role"] == "role_swapped_comparator"
    ]
    assert {spec["candidate_class"] for spec in positives} == {"C"}
    assert {spec["candidate_class"] for spec in controls} == {None}
    assert {spec["P3"] for spec in positives} == {"SMTF", "SMST", "SMKB"}
    assert {spec["P3"] for spec in controls} == {"SMTF", "SMST", "SMKB"}
    assert {(spec["P1"], spec["P2"]) for spec in positives} == {("SMAU", "FKH")}
    assert {(spec["P1"], spec["P2"]) for spec in controls} == {("SMID", "SMAU")}


def _write_toy_panel(tmp_path: Path):
    manifest = tmp_path / "panel.tsv"
    manifest.write_text(
        "sample\tpopulation\n"
        "s1\tP1\n"
        "s2\tP2\n"
        "s3\tP3\n",
        encoding="utf-8",
    )
    vcf = tmp_path / "panel.vcf"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\n"
        "tag1\t1\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\n"
        "tag2\t1\t.\tC\tT\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\n",
        encoding="utf-8",
    )
    return vcf, manifest


def test_wrasse_frequency_geometry_is_reference_invariant_and_finite(tmp_path: Path):
    vcf, manifest = _write_toy_panel(tmp_path)
    result = wrasse.frequency_geometry(
        vcf, manifest, ("P1", "P2", "P3"), bootstrap_replicates=20
    )
    assert result["n_loci"] == 2
    assert np.isclose(result["P2_projection_from_P1_toward_P3_all_loci"], 0.5)
    assert np.isclose(result["f3_like_plugin_P2_P1_P3"], -0.25)
    assert np.isclose(result["mean_P2_finite_called_copy_correction"], 0.25)
    assert np.isclose(result["f3_P2_P1_P3_finite_called_copy_corrected"], -0.5)
    bootstrap = result["iid_locus_bootstrap"]
    assert bootstrap["loci"] == 2
    assert "not chromosome-block" in bootstrap["guardrail"]
    json.dumps(result, allow_nan=False)

    flipped = tmp_path / "flipped.vcf"
    flipped.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts3\ts1\ts2\n"
        "tag1\t1\t.\tG\tA\t.\tPASS\t.\tGT\t0/0\t1/1\t1/0\n"
        "tag2\t1\t.\tT\tC\t.\tPASS\t.\tGT\t0/0\t1/1\t1/0\n",
        encoding="utf-8",
    )
    other = wrasse.frequency_geometry(
        flipped, manifest, ("P1", "P2", "P3"), bootstrap_replicates=20
    )
    assert np.isclose(
        other["P2_projection_from_P1_toward_P3_all_loci"],
        result["P2_projection_from_P1_toward_P3_all_loci"],
    )
    assert np.isclose(
        other["f3_P2_P1_P3_finite_called_copy_corrected"],
        result["f3_P2_P1_P3_finite_called_copy_corrected"],
    )


def test_wrasse_summary_abstains_and_calls_gate_ties_saturation():
    panels = [
        {
            "external_expectation": {
                "benchmark_role": "candidate_direction_sensitivity"
            },
            "adjudication": {
                "severe_OOD": True,
                "natural_data_call_status": "abstain_severe_OOD",
                "matches_candidate_reference": False,
            },
            "simulation_head": {"predicted_class": "A"},
            "simulation_gate": {"called_at_0.5": True},
        },
        {
            "external_expectation": {"benchmark_role": "role_swapped_comparator"},
            "adjudication": {
                "severe_OOD": True,
                "natural_data_call_status": "abstain_severe_OOD",
                "matches_candidate_reference": None,
            },
            "simulation_head": {"predicted_class": "A"},
            "simulation_gate": {"called_at_0.5": True},
        },
    ]
    paired = [
        {
            "candidate_minus_comparator_raw_OOD_gate_score": 0.0,
            "gate_probability_ceiling_tie": True,
        }
    ]
    summary = wrasse.summarize_outcomes(panels, paired)
    assert summary["abstained_panels"] == 2
    assert summary["abstain_due_to_severe_OOD"] is True
    assert summary["accuracy_estimate"] is None
    assert summary["candidate_direction_sensitivities"][
        "raw_OOD_head_matches_candidate_C"
    ] == 0
    assert summary["same_locus_role_changed_contrasts"][
        "gate_probability_ceiling_ties"
    ] == 1
    assert summary["same_locus_role_changed_contrasts"]["raw_gate_delta_range"] == [
        0.0,
        0.0,
    ]
