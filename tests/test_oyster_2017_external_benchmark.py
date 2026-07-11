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
    assert oyster.PANEL_RECORD_SHA256 == (
        "4eaa7255f777c04be58a73a48c83c0a048314144d7df6b113d17e7194dd0669d"
    )
    assert oyster.DERIVED_SOURCE_CONTRACT == {
        "bytes": 470_039,
        "sha256": "7d978cb745008e880a023f4c6347c54d50abd9c19cfb5daeba1f964fc829d756",
        "ordered_locus_id_sha256": oyster.SOURCE_CONTRACT["ordered_locus_sha256"],
        "samples": 90,
        "loci": 1_200,
    }
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


def test_source_download_verifies_direct_bytes_and_has_stable_archive_provenance(
    tmp_path: Path, monkeypatch
):
    workbook = tmp_path / oyster.FILE["key"]
    archive = tmp_path / oyster.ARCHIVE["key"]
    calls = []

    def fake_download(url, output):
        calls.append(url)
        output.write_bytes(b"verified")

    def fake_verify(path):
        assert path.read_bytes() == b"verified"
        return {"path": str(path), "sha256": "canonical"}

    monkeypatch.setattr(oyster, "_download", fake_download)
    monkeypatch.setattr(oyster, "_verify_workbook", fake_verify)
    result = oyster.ensure_source(workbook, archive, download_missing=True)
    assert calls == [oyster.DRYAD_FILE_DOWNLOAD]
    assert set(result) == {"workbook", "retrieval_contract"}
    assert "acquisition route" in result["retrieval_contract"]


def test_source_download_falls_back_after_direct_hash_mismatch_without_wrapper_digest(
    tmp_path: Path, monkeypatch
):
    workbook = tmp_path / oyster.FILE["key"]
    archive = tmp_path / oyster.ARCHIVE["key"]
    calls = []

    def fake_download(url, output):
        calls.append(url)
        output.write_bytes(b"bad" if url == oyster.DRYAD_FILE_DOWNLOAD else b"wrapper")

    def fake_verify(path):
        if path.read_bytes() != b"verified":
            raise AssertionError("hash mismatch")
        return {"path": str(path), "sha256": "canonical"}

    def fake_extract(observed_archive, output):
        assert observed_archive.read_bytes() == b"wrapper"
        output.write_bytes(b"verified")

    monkeypatch.setattr(oyster, "_download", fake_download)
    monkeypatch.setattr(oyster, "_verify_workbook", fake_verify)
    monkeypatch.setattr(oyster, "_extract_workbook", fake_extract)
    result = oyster.ensure_source(workbook, archive, download_missing=True)
    assert calls == [oyster.DRYAD_FILE_DOWNLOAD, oyster.DRYAD_ARCHIVE]
    assert set(result) == {"workbook", "retrieval_contract"}
    assert "ZIP-wrapper" in result["retrieval_contract"]


def test_oyster_frequency_geometry_preserves_f3_assumption_and_linkage_caveats(tmp_path: Path):
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
        "0\t1\tLocus1\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/1\t1/1\n"
        "0\t2\tLocus2\tA\tC\t.\tPASS\t.\tGT\t0/1\t0/0\t1/1\n",
        encoding="utf-8",
    )
    result = oyster.oyster_frequency_geometry(vcf, manifest, ("P1", "P2", "P3"))
    assert "independent binomial called-copy sampling" in result["interpretation"]
    assert "not generally unbiased" in result["interpretation"]
    assert "does not prove a near-null" in result["interpretation"]
    assert "not chromosome-block" in result["iid_locus_bootstrap"]["guardrail"]


def test_materialized_vcf_preserves_sample_locus_gt_order_and_lf(tmp_path: Path):
    workbook = {
        "samples": ["s1", "s2"],
        "loci": ["Locus1", "Locus2"],
        "genotypes": [
            [(1, 1), (0, 0)],
            [(1, 2), (2, 2)],
        ],
    }
    output = tmp_path / "source.vcf"
    audit = oyster.materialize_source_vcf(workbook, output)
    raw = output.read_bytes()
    assert b"\r\n" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert lines[-3] == "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2"
    assert lines[-2] == "0\t1\tLocus1\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/1"
    assert lines[-1] == "0\t2\tLocus2\tA\tC\t.\tPASS\t.\tGT\t./.\t1/1"
    assert audit["samples"] == 2
    assert audit["loci"] == 2


def test_manifest_materialization_accounts_for_all_90_released_samples(tmp_path: Path):
    samples = [str(value) for value in range(1, 94) if value not in (28, 29, 31)]
    populations = [
        population
        for population, count in oyster.WORKBOOK_POPULATION_COUNTS.items()
        for _ in range(count)
    ]
    paths, audit = oyster.materialize_manifests(
        {"samples": samples, "populations": populations}, tmp_path
    )
    assert audit["W"]["population_counts"] == {"WB2": 12, "WOC": 12, "WWC": 12}
    assert audit["Q"]["population_counts"] == {"QB2": 9, "QOC": 12, "QWC": 12}
    assert audit["union"]["samples"] == 69
    assert audit["excluded_nonbenchmark_reference_cohorts"]["samples"] == 21
    assert sum(item["samples"] for key, item in audit.items() if key in ("union", "excluded_nonbenchmark_reference_cohorts")) == 90
    assert len(oyster.read_manifest(paths["W"])) == 36
    assert len(oyster.read_manifest(paths["Q"])) == 33


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
