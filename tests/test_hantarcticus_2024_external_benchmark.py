import json
from pathlib import Path

import numpy as np

from scripts import hantarcticus_2024_external_benchmark as h2024


def test_source_and_panel_contracts_are_guarded():
    assert h2024.FILES["vcf"]["bytes"] == 12_581_530
    assert h2024.FILES["matrices"]["sha256"] == (
        "3ac56229b68ff9c77de9517015e52dfa766bc3e5590cd4b5e502e8a6aefb3456"
    )
    assert h2024.SOURCE_CONTRACT["samples"] == 143
    assert h2024.SOURCE_CONTRACT["variant_rows"] == 20_778
    assert h2024.MODEL_AXIS_ORDER == (
        "NKG",
        "FIB",
        "CHB",
        "DIS",
        "BST",
        "FHA",
        "DOI",
        "AIS",
        "MOT",
        "MIN",
        "GRE",
        "HOS",
    )
    assert h2024.ARCHIVE_TO_PAPER_ORIGIN_INDICES == (
        0,
        1,
        2,
        3,
        4,
        5,
        10,
        6,
        7,
        11,
        8,
        9,
    )
    assert h2024.RAW_ARCHIVE_TO_PAPER_ROW_INDICES == (
        2,
        3,
        0,
        4,
        5,
        1,
        6,
        7,
        8,
        9,
        10,
        11,
    )
    assert tuple(h2024.PANELS) == ("doi_to_fha_hos", "ais_to_hos_doi")
    assert h2024.PANELS["doi_to_fha_hos"]["population_order"] == (
        "DOI",
        "FHA",
        "HOS",
    )
    assert h2024.PANELS["ais_to_hos_doi"]["population_order"] == (
        "AIS",
        "HOS",
        "DOI",
    )
    assert h2024.EXPECTED_FILTERS["doi_to_fha_hos"]["standard_contract"][
        "loci"
    ] == 16_299
    assert h2024.EXPECTED_FILTERS["ais_to_hos_doi"][
        "within_population_polymorphism"
    ]["loci"] == 11_931


def test_prefix_mapping_preserves_known_source_discrepancies():
    assert h2024.sample_site("HA_003") == "FIB"
    assert h2024.sample_site("HAC_08") == "FIB"
    assert h2024.sample_site("HGR_01") == "GRE"
    assert h2024.sample_site("HAR_26") == "AIS"
    assert h2024.sample_site("HLT_08") == "HOS"

    rows = h2024.read_prefix_record()
    by_prefix = {row["vcf_prefix"]: row for row in rows}
    assert by_prefix["HGR"]["mapping_status"] == "VCF_prefix_plus_locality_reconstruction"
    assert by_prefix["HAR"]["mapping_status"] == "cross_source_GBIF_reconstruction"
    assert sum(int(row["VCF_n"]) for row in rows) == 143
    assert sum(h2024.PAPER_TABLE1_COUNTS.values()) == 133


def test_edge_summary_keeps_source_release_and_destination_shares_separate():
    season_D = {}
    day100_runs = {}
    origin = h2024.MODEL_AXIS_ORDER.index("DOI")
    destination = h2024.MODEL_AXIS_ORDER.index("FHA")
    for season in h2024.SEASONS:
        matrix = np.zeros((12, 12), dtype=float)
        matrix[destination, origin] = 2.0
        matrix[destination, destination] = 2.0
        matrix[origin, destination] = 0.5
        matrix[origin, origin] = 1.5
        season_D[season] = matrix
        for number in range(1, 11):
            day100_runs[(season, f"r{number:02d}")] = matrix.copy()

    result = h2024._edge_summary(season_D, day100_runs, "DOI", "FHA")
    assert result["all_four_seasons_forward_positive"] is True
    assert result["all_four_seasons_forward_exceeds_reciprocal"] is True
    assert result[
        "all_four_seasons_forward_destination_share_exceeds_reciprocal"
    ] is True
    assert result["four_season_mean_fraction_of_100_released"] == 0.02
    assert result["four_season_mean_reciprocal_fraction_of_100_released"] == 0.005
    assert result["mean_of_season_destination_conditional_shares"] == 0.5
    assert result["mean_of_season_reciprocal_destination_conditional_shares"] == 0.25


def test_sources_record_marks_published_implementation_and_prose_conflict():
    record = json.loads(h2024.SOURCES_RECORD.read_text(encoding="utf-8"))
    assert record["normalization_guardrail"]["status"] == (
        "published_implementation_resolved_but_prose_conflicts"
    )
    assert "current_paper_connectivity_figure" in record["records"]
    assert record["mapping_guardrails"]["sample_order_sha256"] == (
        h2024.SOURCE_CONTRACT["ordered_sample_sha256"]
    )


def test_summary_never_reports_accuracy_from_correlated_stress_runs():
    panels = []
    for prediction, severe, called in (("A", True, True), ("C", True, False)):
        panels.append(
            {
                "adjudication": {
                    "natural_data_call_status": "abstain_severe_OOD",
                    "severe_OOD": severe,
                    "matches_candidate_reference": prediction == "A",
                },
                "simulation_head": {"predicted_class": prediction},
                "simulation_gate": {"called_at_0.5": called},
            }
        )
    result = h2024.summarize_outcomes(panels)
    assert result["accuracy_estimate"] is None
    assert result["independent_validation_panels"] == 0
    assert result["correlated_candidate_comparisons"] == 2
    assert result["abstained_panels"] == 2


def test_tracked_source_records_are_inside_repository():
    root = Path(__file__).resolve().parents[1]
    assert h2024.PREFIX_RECORD.is_relative_to(root)
    assert h2024.SOURCES_RECORD.is_relative_to(root)
