from __future__ import annotations

import math
import hashlib

import numpy as np
import pytest

from scripts import brook_trout_microsatellite_benchmark as benchmark


def _groups() -> tuple[list[benchmark.Individual], ...]:
    groups = []
    pairs = (
        lambda index: (
            (1, 1) if index % 2 == 0 else (1, 2),
            (4, 4),
        ),
        lambda index: (
            (1, 2) if index % 2 == 0 else (2, 2),
            (4, 5) if index % 2 == 0 else (5, 5),
        ),
        lambda index: (
            (2, 3) if index % 2 == 0 else (3, 3),
            (5, 6) if index % 2 == 0 else (6, 6),
        ),
    )
    for population, builder in enumerate(pairs, start=1):
        groups.append([
            benchmark.Individual(
                sample_id=f"P{population}-{index}",
                population=f"P{population}",
                alleles=builder(index),
                source_ordinal=index + 1,
            )
            for index in range(8)
        ])
    return tuple(groups)


def test_source_contracts_and_direction_semantics_are_pinned():
    assert benchmark.EXPECTED_DIRECTION == "C"
    assert len(benchmark.SOURCE_CONTRACTS) == 4
    for contract in benchmark.SOURCE_CONTRACTS.values():
        assert contract["bytes"] > 0
        assert len(contract["sha256"]) == 64
        int(contract["sha256"], 16)
        assert contract["license"] == "CC0-1.0"
    assert len(benchmark.NS_PANELS) == 8
    assert len({panel.biological_system_id for panel in benchmark.NS_PANELS}) == 8
    assert {panel.p3 for panel in benchmark.NS_PANELS} == {"FM", "MR"}
    assert [(panel.p1, panel.p2, panel.p3) for panel in benchmark.NS_PANELS] == [
        ("FO", "WA", "FM"),
        ("MI", "Ang", "MR"),
        ("RA", "Roc", "FM"),
        ("Tho", "GL", "FM"),
        ("BU", "CO", "FM"),
        ("PO", "LakH", "MR"),
        ("GE", "MC", "FM"),
        ("AL", "RD", "MR"),
    ]
    assert [(panel.p1, panel.p2, panel.p3) for panel in benchmark.NS_SENSITIVITY_PANELS] == [
        ("Cla", "Kel", "FM"),
        ("GR", "MO", "FM"),
        ("Bou2", "Bou1", "FM"),
    ]
    assert len({
        panel.biological_system_id
        for panel in (*benchmark.NS_PANELS, *benchmark.NS_SENSITIVITY_PANELS)
    }) == 10
    assert benchmark.PA_TARGETS == {
        "DOUB": {"n": 154, "mean_p_wild": 0.95, "wild": 129, "introgressed": 25, "hatchery": 0},
        "LICK": {"n": 50, "mean_p_wild": 0.96, "wild": 42, "introgressed": 8, "hatchery": 0},
        "CONK": {"n": 50, "mean_p_wild": 0.94, "wild": 36, "introgressed": 14, "hatchery": 0},
    }
    assert benchmark.PA_STOCKING_CONTEXT["DOUB"] == {
        "table_1_stocking_at_sample_location": True,
        "table_1_stocking_within_2_km": True,
        "discussion_states_not_directly_stocked": True,
        "no_stocking_record_more_than_50_years": False,
        "direct_stocking_status": "conflicting_table_1_direct_vs_discussion_not_direct",
        "source_internal_conflict": True,
    }
    assert benchmark.PA_STOCKING_CONTEXT["LICK"][
        "table_1_stocking_at_sample_location"
    ] is False
    assert benchmark.PA_STOCKING_CONTEXT["LICK"]["table_1_stocking_within_2_km"] is True
    assert benchmark.PA_STOCKING_CONTEXT["CONK"][
        "no_stocking_record_more_than_50_years"
    ] is True
    assert "source-internal conflict" in benchmark.PA_DIRECTION_EVIDENCE["DOUB"]
    assert "no stocking record" in benchmark.PA_DIRECTION_EVIDENCE["CONK"]
    transcription = {
        "targets": benchmark.PA_TARGETS,
        "reference_sites": list(benchmark.PA_REFERENCE_SITES),
        "stocking_context": benchmark.PA_STOCKING_CONTEXT,
    }
    observed = hashlib.sha256(benchmark._canonical_json(transcription)).hexdigest()
    assert observed == benchmark.PA_PUBLISHED_COMPARATOR_PROVENANCE[
        "manual_transcription_contract_sha256"
    ]
    assert "Table 1" in benchmark.PA_PUBLISHED_COMPARATOR_PROVENANCE["locator"]
    assert "Table 1" in benchmark.PA_PUBLISHED_COMPARATOR_PROVENANCE["direct_stocking_locator"]
    assert "DOUB" in benchmark.PA_PUBLISHED_COMPARATOR_PROVENANCE["source_internal_conflict"]


@pytest.mark.parametrize(
    "token,expected",
    [("001001", (1, 1)), ("123456", (123, 456)), ("000000", (0, 0))],
)
def test_fixed_width_genepop_decoder(token, expected):
    assert benchmark.decode_genepop_token(token, "toy") == expected


@pytest.mark.parametrize("token", ["00100", "0010001", "00A001", "000001", "001000"])
def test_genepop_decoder_rejects_malformed_or_partial_missing(token):
    with pytest.raises(ValueError):
        benchmark.decode_genepop_token(token, "toy")


def test_shared_locus_filter_is_global_and_strict_is_a_subset():
    panel = {"panel_id": "toy", "groups": _groups()}
    standard, strict, audit = benchmark.eligible_shared_loci(
        ["multi", "mono_in_P1"], [panel]
    )
    assert standard == ["multi", "mono_in_P1"]
    assert strict == ["multi"]
    assert audit["standard_loci"] == 2
    assert audit["strict_loci"] == 1


def test_multiallelic_count_matrix_preserves_every_allele_and_gene_copy():
    counts, sizes, audit = benchmark.panel_to_counts(
        ["multi", "mono_in_P1"],
        _groups(),
        ["multi", "mono_in_P1"],
        require_within_population_polymorphism=False,
    )
    assert len(counts) == 2
    assert counts[0].shape == (3, 3)
    assert counts[0].tolist() == [
        [12, 4, 0],
        [4, 12, 0],
        [0, 4, 12],
    ]
    assert sizes.tolist() == [[16, 16, 16], [16, 16, 16]]
    assert audit["alleles_per_locus"]["multiallelic_loci"] == 2
    assert audit["missing_copy_fraction"] == 0.0


def test_frequency_geometry_is_invariant_to_allele_column_permutation():
    counts, _, _ = benchmark.panel_to_counts(
        ["multi", "mono_in_P1"],
        _groups(),
        ["multi", "mono_in_P1"],
        require_within_population_polymorphism=False,
    )
    first = benchmark.multiallelic_frequency_geometry(counts[:1], ["multi"])
    second = benchmark.multiallelic_frequency_geometry(
        [counts[0][:, [2, 0, 1]]], ["multi"]
    )
    assert first["mean_projection"] == pytest.approx(second["mean_projection"])
    assert first["denominator_weighted_projection"] == pytest.approx(
        second["denominator_weighted_projection"]
    )
    assert first["mean_f3"] == pytest.approx(second["mean_f3"])
    assert first["locus_geometry_sha256"] == second["locus_geometry_sha256"]


def test_panel_count_builder_rejects_cross_population_sample_overlap():
    groups = list(_groups())
    groups[1] = [groups[0][0], *groups[1][1:]]
    with pytest.raises(ValueError, match="pairwise disjoint"):
        benchmark.panel_to_counts(
            ["multi", "mono_in_P1"],
            tuple(groups),
            ["multi", "mono_in_P1"],
            require_within_population_polymorphism=False,
        )


def test_depth_matched_gate_contract_is_exactly_g2_through_g16():
    table = np.zeros((2, 15, 28), dtype=float)
    table[:, :, 0] = benchmark.PRIMARY_DEPTHS
    table[0, :, 1:] = np.arange(27)[None, :]
    table[1, :, 1:] = np.arange(27)[None, :] + 1
    features, audit = benchmark.depth_matched_gate_features(table)
    assert features.shape == (2, 216)
    assert audit["feature_dimension"] == 216
    assert audit["selected_depth_row_indices_zero_based"] == [1, 2, 3, 5, 7, 10, 14]
    assert audit["selected_depths"] == [3, 4, 5, 7, 9, 12, 16]
    assert "seven unique" in audit["description"]


def test_full_runner_is_fail_closed_to_the_exact_azure_host():
    expected = benchmark.require_azure_execution_target(
        "azure", os_name="posix", hostname="trading-linux-az"
    )
    assert expected["verified_azure_host"] is True
    with pytest.raises(RuntimeError, match="requires --compute-target azure"):
        benchmark.require_azure_execution_target(
            "local", os_name="posix", hostname="trading-linux-az"
        )
    with pytest.raises(RuntimeError, match="bound to POSIX host"):
        benchmark.require_azure_execution_target(
            "azure", os_name="nt", hostname="MATH-ROSS20"
        )
    with pytest.raises(RuntimeError, match="bound to POSIX host"):
        benchmark.require_azure_execution_target(
            "azure", os_name="posix", hostname="some-other-box"
        )


def test_spearman_ties_and_constant_scores_are_handled_without_nan():
    assert benchmark.spearman_correlation([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert benchmark.spearman_correlation([1, 1, 1], [10, 20, 30]) is None


def test_candidate_concordance_never_creates_an_accuracy_claim():
    records = [
        {"direction": {"raw_all": {"call": "C"}}},
        {"direction": {"raw_all": {"call": "A"}}},
    ]
    summary = benchmark._candidate_concordance(records, "raw_all")
    assert summary["C_calls"] == 1
    assert summary["views"] == 2
    assert summary["fraction"] == 0.5
    assert "accuracy" not in " ".join(summary).lower()
    assert "not formal accuracy" in summary["interpretation"]


def test_descriptive_accounting_distinguishes_contrasts_from_record_views():
    records = []
    for target in ("doub", "lick", "conk"):
        records.append({
            "panel_id": f"pa_{target}_clean8_pool",
            "dataset": "pennsylvania_white_2018",
            "biological_system_id": f"pa_{target}",
            "contract_role": "primary",
            "published_comparator": {},
        })
    for system in (
        "annapolis", "baddeck", "cornwallis", "east_river_pictou",
        "lahave", "margaree", "musquodoboit", "river_denys",
    ):
        records.extend({
            "panel_id": f"ns_{system}",
            "dataset": "nova_scotia_lehnert_2020",
            "biological_system_id": f"ns_{system}",
            "contract_role": role,
            "published_comparator": {"recorded_stocking_exposure": True},
        } for role in ("primary", "locus_filter_sensitivity"))
    for panel, biological_system in (
        ("ns_saint_marys", "ns_saint_marys"),
        ("ns_saint_marys_east", "ns_saint_marys"),
        ("ns_st_marys_bay", "ns_st_marys_bay"),
    ):
        records.extend({
            "panel_id": panel,
            "dataset": "nova_scotia_lehnert_2020",
            "biological_system_id": biological_system,
            "contract_role": role,
            "published_comparator": {"recorded_stocking_exposure": False},
        } for role in (
            "no_recorded_stocking_sensitivity",
            "no_recorded_stocking_strict_locus_sensitivity",
        ))
    accounting = benchmark._descriptive_panel_accounting(records)
    assert accounting["nova_scotia_no_recorded_stocking_contrasts"] == 3
    assert accounting["nova_scotia_no_recorded_stocking_record_views"] == 6
    assert accounting["nova_scotia_no_recorded_stocking_named_systems"] == 2


def test_pinned_shared_locus_hashes_are_full_sha256_values():
    for value in (
        benchmark.EXPECTED_NS_STANDARD_LOCUS_SHA256,
        benchmark.EXPECTED_NS_STRICT_LOCUS_SHA256,
        benchmark.EXPECTED_PA_ROW_LEDGER_SHA256,
        benchmark.EXPECTED_NS_ROW_LEDGER_SHA256,
        benchmark.EXPECTED_NS_INTROGRESSION_LEDGER_SHA256,
    ):
        assert len(value) == 64
        assert math.isfinite(float(int(value, 16)))
