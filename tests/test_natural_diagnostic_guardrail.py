"""Keep the natural-panel negative control visible in future edits."""
from __future__ import annotations

import json
from pathlib import Path


RESULT = (
    Path(__file__).parents[1]
    / "results"
    / "natural_diagnostics_2026_07_09"
    / "heliconius.json"
)


def test_heliconius_control_falsifies_natural_head_specificity():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["status"] == "exploratory_discordance_not_validation"
    geographic = [panel for panel in result["panels"] if panel["kind"] == "geographic_trio"]
    controls = [panel for panel in result["panels"] if panel["kind"] == "allopatric_control"]
    assert len(geographic) == 4
    assert len(controls) == 1
    assert all(panel["direction_call"] == "C" for panel in geographic)
    assert all(panel["primary_raw_ratio_g_ge_8"] > 1 for panel in geographic)
    control = controls[0]
    assert control["primary_raw_ratio_g_ge_8"] < 1
    assert control["direction_call"] == "C"
    assert control["uncalibrated_softmax_scores"]["C"] > 0.99


def test_chromosome_stability_is_not_misreported_as_validation():
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    audit = result["canonical_race_trio_1_leave_one_chromosome_out"]
    assert len(audit) == 21
    assert {entry["direction_call"] for entry in audit} == {"C"}
    assert min(entry["uncalibrated_softmax_scores"]["C"] for entry in audit) > 0.99999998
    assert "not ground truth" in result["interpretation"]["external_context"].lower()
