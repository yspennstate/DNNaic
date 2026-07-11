import json
from pathlib import Path
import zipfile

import pytest

from dnnaic.semantics import class_for_forward_edge
from scripts import oyster_2017_external_benchmark as oyster


def test_oyster_source_panel_and_filter_contracts_are_guarded():
    assert oyster.FILE == {
        "id": 21_290,
        "key": "SNP_data_M12109.xlsx",
        "archive_member": "SNP_data_M12109.xlsx",
        "download": "https://datadryad.org/api/v2/files/21290/download",
        "bytes": 729_706,
        "md5": "572a079597af8530b15aaffd07325b55",
        "sha256": "e0f6983f1a15c9d7a1aeb4a76e220f24b1d4c766600502413b2cb5c4fdde8029",
    }
    assert "sha256" not in oyster.ARCHIVE
    record = oyster.validate_sources_record()
    assert record["version_id"] == 3_411
    assert record["paper"]["doi"] == "10.3354/meps12109"
    assert record["analysis_design"]["expected_gate"] is None
    assert record["analysis_design"]["direction_truth_available"] is False
    assert record["analysis_design"]["gate_truth_available"] is False
    assert record["analysis_design"]["independent_validation_panels"] == 0
    assert class_for_forward_edge("P3", "P2") == "C"

    rows = oyster.read_panel_record()
    assert len(rows) == 8
    assert sum(int(row["expected_n"]) for row in rows) == 90
    assert oyster.PANEL_SPECS["W"]["population_order"] == ("WWC", "WOC", "WB2")
    assert oyster.PANEL_SPECS["Q"]["population_order"] == ("QWC", "QOC", "QB2")
    assert oyster.PANEL_SPECS["W"]["same_data_excluded_ids"] == []
    assert oyster.PANEL_SPECS["Q"]["same_data_excluded_ids"] == ["28", "29", "31"]
    assert oyster.EXPECTED_FILTERS["standard_contract"]["loci"] == 1_101
    assert oyster.EXPECTED_FILTERS["within_population_polymorphism"]["loci"] == 589


def _toy_xlsx(path: Path, *, formula: bool = False, unsafe_target: bool = False):
    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="target" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    target = "../../escape.xml" if unsafe_target else "worksheets/sheet1.xml"
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="worksheet" Target="{target}"/>'
        '</Relationships>'
    )
    shared = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<si><r><t>shared</t></r><r><t>-text</t></r></si></sst>'
    )
    formula_xml = "<f>1+1</f>" if formula else ""
    sheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<dimension ref="A1:B2"/><sheetData><row r="1">'
        '<c r="A1" t="s"><v>0</v></c>'
        '<c r="B1" t="inlineStr"><is><t>inline</t></is></c>'
        '</row><row r="2">'
        f'<c r="A2">{formula_xml}<v>1</v></c><c r="B2"><v>2</v></c>'
        '</row></sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("xl/workbook.xml", workbook)
        bundle.writestr("xl/_rels/workbook.xml.rels", relationships)
        bundle.writestr("xl/sharedStrings.xml", shared)
        bundle.writestr("xl/worksheets/sheet1.xml", sheet)


def test_ooxml_reader_resolves_named_sheet_and_text_types(tmp_path: Path):
    path = tmp_path / "toy.xlsx"
    _toy_xlsx(path)
    result = oyster.read_xlsx_sheet(path, "target")
    assert result["dimension"] == "A1:B2"
    assert result["cells"] == {
        (0, 0): "shared-text",
        (0, 1): "inline",
        (1, 0): "1",
        (1, 1): "2",
    }
    assert oyster._column_index("CNJ") == 2_401


def test_ooxml_reader_rejects_formulas_and_unsafe_targets(tmp_path: Path):
    formula = tmp_path / "formula.xlsx"
    _toy_xlsx(formula, formula=True)
    with pytest.raises(ValueError, match="formulas"):
        oyster.read_xlsx_sheet(formula, "target")

    unsafe = tmp_path / "unsafe.xlsx"
    _toy_xlsx(unsafe, unsafe_target=True)
    with pytest.raises(ValueError, match="unsafe"):
        oyster.read_xlsx_sheet(unsafe, "target")


def test_genalex_pair_conversion_rejects_partial_missingness():
    assert oyster.decode_genalex_pair(0, 0) == "./."
    assert oyster.decode_genalex_pair(1, 1) == "0/0"
    assert oyster.decode_genalex_pair(1, 2) == "0/1"
    assert oyster.decode_genalex_pair(2, 1) == "0/1"
    assert oyster.decode_genalex_pair(2, 2) == "1/1"
    with pytest.raises(ValueError, match="partial"):
        oyster.decode_genalex_pair(0, 1)
    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        oyster.decode_genalex_pair(1, 3)


@pytest.mark.parametrize("prediction", ["A", "B", "C"])
@pytest.mark.parametrize("gate", [0.2, 0.8])
@pytest.mark.parametrize("severe", [False, True])
def test_oyster_adjudication_never_creates_truth_or_accuracy(prediction, gate, severe):
    result = oyster.adjudicate_panel(
        prediction,
        gate,
        11.0 if severe else 2.0,
        1.0,
    )
    assert result["natural_data_call_status"] == (
        "abstain_severe_OOD" if severe else "descriptive_only_no_gold_label"
    )
    assert result["direction_truth_available"] is False
    assert result["gate_truth_available"] is False
    assert result["accuracy_eligible"] is False
    assert result["specificity_eligible"] is False
    assert "matches_candidate_reference" not in result
    assert "correct" not in result


def test_oyster_summary_counts_correlated_sensitivities_not_trials():
    panels = []
    for prediction, gate, severe in (
        ("A", 0.2, True),
        ("B", 0.8, True),
        ("C", 0.3, False),
        ("C", 0.9, False),
    ):
        panels.append(
            {
                "simulation_head": {"predicted_class": prediction},
                "simulation_gate": {"appreciable_score": gate},
                "adjudication": {
                    "severe_OOD": severe,
                    "natural_data_call_status": "abstain_severe_OOD" if severe else "descriptive_only_no_gold_label",
                },
            }
        )
    summary = oyster.summarize_outcomes(panels)
    assert summary["analytic_sensitivity_runs"] == 4
    assert summary["correlated_site_comparisons"] == 2
    assert summary["unique_biological_systems"] == 1
    assert summary["independent_validation_panels"] == 0
    assert summary["accuracy_estimate"] is None
    assert summary["specificity_estimate"] is None
    assert summary["raw_gate_below_0_5"] == 2
    assert summary["raw_gate_crossings_at_0_5"] == 2
    assert summary["raw_head_prediction_counts"] == {"A": 1, "B": 1, "C": 2}
    json.dumps(summary, allow_nan=False)
