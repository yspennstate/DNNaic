import hashlib
import json

import pytest

from dnnaic.semantics import class_for_forward_edge
from scripts import guppy_2020_external_benchmark as guppy


def test_guppy_source_revision_files_and_no_license_are_pinned():
    assert guppy.COMMIT == "ac8ec0cdf29dec539494b49d8bdf32ff6f0197f2"
    assert guppy.TREE == "eac1fe39081906b691f857e4493864db66361b02"
    assert guppy.FILES["NCA"] == {
        "key": "nchr_NCA_mapped.vcf",
        "url": (
            "https://raw.githubusercontent.com/gbradburd/guppy_seln/"
            "ac8ec0cdf29dec539494b49d8bdf32ff6f0197f2/1_data/nchr_NCA_mapped.vcf"
        ),
        "bytes": 4_967_968,
        "sha256": "4cf713eac4243808c85800b5b192ff19b8132d66b240c6cc9f48f5658ab2c940",
        "git_blob": "471ab9e4952cfd72d9dd53e298393ef73b51632b",
    }
    assert {population: spec["bytes"] for population, spec in guppy.FILES.items()} == {
        "NCA": 4_967_968,
        "NTY": 4_829_947,
        "PCA": 5_276_051,
        "PTY": 6_252_249,
        "SGS": 2_870_728,
    }
    assert guppy.FORMAT_SCRIPT["git_blob"] == "dd2463fe0beb5de95d42a1cb46df31684c8b3617"
    validated = guppy.validate_sources_record()
    assert validated["canonical_lf"] == guppy.SOURCE_RECORD_CANONICAL_LF_CONTRACT
    record = validated["record"]
    assert record["repository_license"] is None
    assert record["data_use_policy"].startswith("Fetch exact bytes at runtime only")
    assert record["paper"]["doi"] == "10.1016/j.cub.2019.11.062"
    assert record["original_experiment"]["doi"] == "10.1111/eva.12356"
    assert len(record["release_discrepancies"]) == 3
    assert "12,407 SNPs" in record["release_discrepancies"][0]
    assert "runtime-verified VCFs" in record["release_discrepancies"][0]
    assert "BIM" not in record["release_discrepancies"][0]
    assert "NCA is post-flow" in record["release_discrepancies"][1]
    assert "license=null" in record["release_discrepancies"][2]
    design = record["analysis_design"]
    assert design["release_locus_ascertainment_outcome_blind"] is False
    assert design["benchmark_locus_filters_outcome_blind"] is False
    assert "post-flow P2" in design["locus_ascertainment_guardrail"]
    assert "not prospective held-out" in design["locus_ascertainment_guardrail"]


def _git_blob(payload):
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def test_guppy_git_blob_sha1_is_computed_from_actual_bytes(tmp_path):
    payload = b"author-deposited source bytes\n"
    path = tmp_path / "source.vcf"
    path.write_bytes(payload)
    assert guppy.git_blob_sha1(path) == _git_blob(payload)


def test_guppy_corrupt_cached_source_is_replaced_and_verified(tmp_path, monkeypatch):
    payload = b"correct pinned payload\n"
    path = tmp_path / "source.vcf"
    path.write_bytes(b"corrupt")
    spec = {
        "url": "https://example.invalid/source.vcf",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "git_blob": _git_blob(payload),
    }
    calls = []

    def fake_download(url, output):
        calls.append(url)
        output.write_bytes(payload)

    monkeypatch.setattr(guppy, "_download", fake_download)
    verified = guppy.ensure_source(path, spec, download_missing=True)
    assert calls == [spec["url"]]
    assert path.read_bytes() == payload
    assert verified["git_blob_sha1"] == spec["git_blob"]


def test_guppy_bad_download_is_removed(tmp_path, monkeypatch):
    payload = b"correct pinned payload\n"
    path = tmp_path / "source.vcf"
    spec = {
        "url": "https://example.invalid/source.vcf",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "git_blob": _git_blob(payload),
    }

    def fake_download(_url, output):
        output.write_bytes(b"wrong")

    monkeypatch.setattr(guppy, "_download", fake_download)
    with pytest.raises(ValueError, match="expected"):
        guppy.ensure_source(path, spec, download_missing=True)
    assert not path.exists()


def test_guppy_sample_and_panel_contracts_are_exact():
    assert len(guppy.SAMPLE_IDS["NCA"]) == 18
    assert "NCA-11" not in guppy.SAMPLE_IDS["NCA"]
    assert "NCA-15" not in guppy.SAMPLE_IDS["NCA"]
    assert len(guppy.SAMPLE_IDS["NTY"]) == 17
    assert guppy.SAMPLE_IDS["PCA"] == [
        "PCA-01", "PCA-02", "PCA-03", "PCA-06", "PCA-07", "PCA-09", "PCA-10",
        "PCA-11", "PCA-12", "PCA-13", "PCA-14", "PCA-15", "PCA-17", "PCA-18",
        "PCA-19", "PCA-20", "PCA-21", "PCA-22", "PCA-23",
    ]
    assert guppy.SAMPLE_IDS["PTY"] == [f"PTY-{value:02d}" for value in range(1, 24)]
    assert guppy.SAMPLE_IDS["SGS"] == [f"SGS-{value:02d}" for value in range(2, 11)]
    assert sum(len(samples) for samples in guppy.SAMPLE_IDS.values()) == 86
    assert guppy.PANEL_SPECS["caigual"]["groups"] == {"P1": "NCA", "P2": "PCA", "P3": "SGS"}
    assert guppy.PANEL_SPECS["taylor"]["groups"] == {"P1": "NTY", "P2": "PTY", "P3": "SGS"}
    assert guppy.PANEL_SPECS["caigual"]["manifest_sha256"] == (
        "51164cdf7692f98a035371f8ce5c482bfda2de2be7b85139143ef05c41bb71ec"
    )
    assert guppy.PANEL_SPECS["taylor"]["manifest_sha256"] == (
        "1197e3fc32abb5593c1edca40d1d22cc95c0eb43325b88eb9bdc8ee9440936d7"
    )
    assert class_for_forward_edge("P3", "P2") == "C"


def test_guppy_locus_join_filter_and_derived_contracts_are_exact():
    assert guppy.SOURCE_VARIANT_CONTRACT == {
        "variants": 11_417,
        "chromosomes": 23,
        "ordered_locus_sha256": "3daefba0c6ac6dd3f8b285633bf09be2329ad0436b7528f5be7e6b9bc12e7c73",
    }
    assert guppy.COMBINED_VCF_CONTRACT == {
        "bytes": 4_305_263,
        "sha256": "83a00efa9836673e0a894b794ea9b3d2820defd3ff453bafa7d55a830b53b49f",
    }
    assert guppy.EXPECTED_PANEL_LOCI == {
        ("caigual", "standard_contract"): (
            6_877,
            "e2133443750113e175a463df5490d32d2262c30c46f95f317cc90b124da7d338",
        ),
        ("caigual", "within_population_polymorphism"): (
            30,
            "caaa48c0fd276886937e210ae2d8f62a5fc7e070dc7cf9d8105b4d7a6b744c39",
        ),
        ("taylor", "standard_contract"): (
            6_696,
            "e9dbd8c73fec4e2d5261f9a58479b46e3b3314a060b31570033d916e4bf1ae88",
        ),
        ("taylor", "within_population_polymorphism"): (
            22,
            "b333cecf5dccc51b0b1984d1dfce01c7192cb0865ee6379a0223e24c7488cae9",
        ),
    }
    assert len(guppy.EXPECTED_PREPARED_VCF) == 4


def test_guppy_gt_decoder_normalizes_heterozygote_order():
    assert guppy.decode_gt("0/0") == "0/0"
    assert guppy.decode_gt("1/0") == "0/1"
    assert guppy.decode_gt("0|1") == "0/1"
    assert guppy.decode_gt("./.") == "./."
    with pytest.raises(ValueError, match="unexpected biallelic"):
        guppy.decode_gt("2/2")


def _source_vcf_text(population, variants, *, samples=None, source_filter="PASS", source_format="GT:DP:AD:GL"):
    samples = guppy.SAMPLE_IDS[population] if samples is None else samples
    lines = ["##fileformat=VCFv4.2"] + [f"##guppy_test_{index}=x" for index in range(9)]
    lines.append(
        "\t".join(
            ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
            + samples
        )
    )
    for chrom, pos, ref, alt in variants:
        cell = "0/0:1:1,0:0,1,2" if source_format == "GT:DP:AD:GL" else "0/0:1"
        lines.append(
            "\t".join(
                [chrom, str(pos), ".", ref, alt, ".", source_filter, ".", source_format]
                + [cell] * len(samples)
            )
        )
    return "\n".join(lines) + "\n"


def _write_source_vcfs(tmp_path, overrides=None):
    overrides = overrides or {}
    paths = {}
    default = [("1", 100, "A", "G")]
    for population in ("NCA", "NTY", "PCA", "PTY", "SGS"):
        options = overrides.get(population, {})
        path = tmp_path / f"{population}.vcf"
        path.write_text(
            _source_vcf_text(population, options.get("variants", default), **{
                key: value for key, value in options.items() if key != "variants"
            }),
            encoding="utf-8",
            newline="\n",
        )
        paths[population] = path
    return paths


def test_guppy_five_vcf_join_is_exact_and_resets_source_info(tmp_path, monkeypatch):
    paths = _write_source_vcfs(tmp_path)
    combined_samples = [
        sample
        for population in ("NCA", "NTY", "PCA", "PTY", "SGS")
        for sample in guppy.SAMPLE_IDS[population]
    ]
    metadata = ["##fileformat=VCFv4.2"] + [f"##guppy_test_{index}=x" for index in range(9)]
    header = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT"]
    expected_text = "\n".join(
        metadata
        + ["##DNNaic_join=exact shared locus key; source INFO reset because AF/NS are not population-specific"]
        + ["\t".join(header + combined_samples)]
        + ["\t".join(["1", "100", ".", "A", "G", ".", "PASS", ".", "GT"] + ["0/0"] * 86)]
    ) + "\n"
    expected_bytes = expected_text.encode("utf-8")
    monkeypatch.setattr(
        guppy,
        "SOURCE_VARIANT_CONTRACT",
        {
            "variants": 1,
            "chromosomes": 1,
            "ordered_locus_sha256": hashlib.sha256(b"1\t100\tA\tG\n").hexdigest(),
        },
    )
    monkeypatch.setattr(
        guppy,
        "COMBINED_VCF_CONTRACT",
        {"bytes": len(expected_bytes), "sha256": hashlib.sha256(expected_bytes).hexdigest()},
    )
    output = tmp_path / "combined.vcf"
    audit = guppy.materialize_combined_vcf(paths, output)
    assert output.read_bytes() == expected_bytes
    assert audit["source_files_share_exact_ordered_keys"] is True
    assert audit["samples"] == 86
    assert audit["missing_genotypes"] == {
        "NCA": 0,
        "NTY": 0,
        "PCA": 0,
        "PTY": 0,
        "SGS": 0,
    }
    assert audit["INFO_policy"].startswith("reset to dot")


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"NTY": {"variants": [("1", 100, "C", "G")]}}, "locus keys/order differ"),
        ({"NTY": {"variants": []}}, "unequal variant-row counts"),
        (
            {"NTY": {"samples": ["NTY-X"] + guppy.SAMPLE_IDS["NTY"][1:]}},
            "source sample IDs changed",
        ),
        ({"NTY": {"source_filter": "LowQual"}}, "FILTER/FORMAT changed"),
        ({"NTY": {"source_format": "GT:DP"}}, "FILTER/FORMAT changed"),
        (
            {
                "NTY": {"variants": [("1", 200, "C", "T"), ("1", 100, "A", "G")]},
                "NCA": {"variants": [("1", 100, "A", "G"), ("1", 200, "C", "T")]},
                "PCA": {"variants": [("1", 100, "A", "G"), ("1", 200, "C", "T")]},
                "PTY": {"variants": [("1", 100, "A", "G"), ("1", 200, "C", "T")]},
                "SGS": {"variants": [("1", 100, "A", "G"), ("1", 200, "C", "T")]},
            },
            "locus keys/order differ",
        ),
    ],
)
def test_guppy_join_rejects_source_drift(tmp_path, overrides, match):
    paths = _write_source_vcfs(tmp_path, overrides)
    with pytest.raises(AssertionError, match=match):
        guppy.materialize_combined_vcf(paths, tmp_path / "combined.vcf")


def test_format_script_audit_exposes_nca_comment_error(tmp_path):
    path = tmp_path / "format_guppy_data.R"
    path.write_text(
        "#\tNCA - post gene flow headwater Caigual\n"
        "#\tPCA - post gene flow headwater Caigual\n",
        encoding="utf-8",
    )
    result = guppy.audit_format_script(path)
    assert "NCA as post-flow" in result["discrepancy"]
    assert result["benchmark_mapping"] == "NCA pre-flow P1; PCA post-flow P2"


def _panel(prediction="C", direction_rms=2.0, gate_rms=1.0):
    return {
        "simulation_head": {"predicted_class": prediction},
        "simulation_feature_shift": {"rms_z": direction_rms},
        "simulation_gate_feature_shift": {"rms_z": gate_rms},
    }


@pytest.mark.parametrize("prediction", ["A", "B", "C"])
@pytest.mark.parametrize("direction_rms,severe", [(10.0, False), (10.000001, True)])
def test_guppy_adjudication_never_turns_manipulations_into_accuracy(prediction, direction_rms, severe):
    result = guppy.adjudicate_panel(_panel(prediction, direction_rms))
    assert result["natural_data_call_status"] == (
        "abstain_severe_OOD" if severe else "descriptive_candidate_concordance_only"
    )
    assert result["experimental_flow_direction_available"] is True
    assert result["exclusive_single_edge_truth_available"] is False
    assert result["formal_direction_accuracy_eligible"] is False
    assert result["gate_truth_available"] is False
    assert result["direction_call_accepted"] is False
    assert result["raw_head_matches_candidate_C"] is (prediction == "C")
    assert "heuristic diagnostic, not calibrated support" in result["severe_OOD_rule"]
    assert "accepted_direction_call" not in result


def test_guppy_summary_counts_two_units_not_four_trials():
    panels = []
    for prediction in ("C", "B", "C", "B"):
        panel = _panel(prediction, direction_rms=11.0)
        panel["adjudication"] = guppy.adjudicate_panel(panel)
        panels.append(panel)
    outcome = guppy.summarize_outcomes(panels)
    assert outcome["analytic_correlated_sensitivity_rows"] == 4
    assert outcome["ecological_recipient_units"] == 2
    assert outcome["shared_source_proxy"] is True
    assert outcome["independent_formal_accuracy_units"] == 0
    assert outcome["raw_head_prediction_counts"] == {"B": 2, "C": 2}
    assert outcome["raw_candidate_C_concordant_sensitivity_rows"] == 2
    assert outcome["accuracy_denominator"] is None
    assert outcome["severe_OOD_panels"] == 4
    assert outcome["accepted_direction_calls"] == 0
    assert outcome["direction_accuracy_estimate"] is None
    assert outcome["gate_accuracy_estimate"] is None
    json.dumps(outcome, allow_nan=False)
    with pytest.raises(AssertionError, match="four correlated"):
        guppy.summarize_outcomes(panels[:-1])
