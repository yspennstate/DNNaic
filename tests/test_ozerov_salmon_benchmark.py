from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import ozerov_salmon_benchmark as benchmark


def test_sources_paper_estimators_and_candidate_direction_are_pinned():
    assert benchmark.EXPECTED_DIRECTION == "C"
    expected = {
        "microsatellite": (
            345_280,
            "32c53c1f4f700a5c3d2d5b50a461b381",
            "6483ccf6771fa4d0d9452cff5efb75a9cb2852cf6162268572fea6c3c2b5418d",
            68_891,
        ),
        "pooled_snp": (
            629_100,
            "c7dfb0b8f0191ace362d22fc5a5155e4",
            "6d1325739bc56175b4597ee075edc2795fe89e169e737d5afe0a9b175495b3f6",
            68_892,
        ),
    }
    for name, contract in benchmark.SOURCE_CONTRACTS.items():
        assert (
            contract["bytes"],
            contract["md5"],
            contract["sha256"],
            contract["dryad_file_id"],
        ) == expected[name]
        assert contract["license"] == "CC0-1.0"
        assert contract["data_doi"] == "10.5061/dryad.p00gd"
        assert contract["paper_doi"] == "10.1111/mec.13570"
        assert contract["version_id"] == 20_386

    published = benchmark.PUBLISHED_COMPARATOR
    assert published["paper_doi"] == "10.1111/mec.13570"
    assert published["microsatellite_Q"] == {
        "estimate": 0.606,
        "ci95": [0.542, 0.671],
        "markers": "17 individual microsatellites",
        "P1": "Loobu 1996-99, n=81",
        "P2": "Loobu 2007-08, n=77",
        "P3": "pooled Narva 1998-2009, n=720",
    }
    assert published["microsatellite_I"]["estimate"] == 0.616
    assert published["microsatellite_I"]["ci95"] == [0.552, 0.680]
    assert published["microsatellite_I"]["markers"] == "17 individual microsatellites"
    assert "pooled Narva 1998-2009" in published["microsatellite_I"]["P3"]
    assert published["pooled_snp_S_hat"]["estimate"] == 0.567
    assert published["pooled_snp_S_hat"]["ci95"] == [0.533, 0.601]
    assert published["pooled_snp_S_hat"]["P3"].startswith("Nar06")
    assert "before large-scale releases" in published["stocking_timeline_guardrail"]
    assert "not as never previously stocked" in published["stocking_timeline_guardrail"]
    assert "pre-release" not in benchmark.__doc__


def test_population_locus_missingness_and_normalized_ledgers_are_pinned():
    assert benchmark.GROUP_COUNTS == {
        "Kei9697a": 54, "Kei0708a": 67, "Kei0910a": 98, "Kei1112a": 95,
        "Vas9699a": 45, "Vas0708a": 30, "Vas0910a": 97, "Vas1112a": 67,
        "Kun9697a": 71, "Kun0708a": 57, "Kun0910a": 70, "Kun1112a": 120,
        "Loobu9699a": 81, "Loobu0708a": 77, "Loobu0910a": 102,
        "Loobu1112a": 104,
        "Narva98a": 45, "Narva01a": 73, "Narva04a": 129, "Narva05a": 77,
        "Narva06a": 112, "Narva07a": 80, "Narva08a": 95, "Narva09a": 109,
        "Neva9798a": 97,
    }
    assert sum(benchmark.GROUP_COUNTS.values()) == 2_052
    assert benchmark.NARVA_MICROSATELLITE_GROUPS == (
        "Narva98a", "Narva01a", "Narva04a", "Narva05a",
        "Narva06a", "Narva07a", "Narva08a", "Narva09a",
    )
    assert sum(
        benchmark.GROUP_COUNTS[name]
        for name in benchmark.NARVA_MICROSATELLITE_GROUPS
    ) == 720
    assert benchmark.GROUP_COUNTS["Narva06a"] == 112
    assert benchmark.MICROSATELLITE_LOCI == (
        "SSsp2210", "SSsp2216", "SsaD157", "Ssa407", "SSspG7",
        "SSsp3016", "SSsp2201", "Ssa14", "SSsp1605", "SSOSL85",
        "SSOSL438", "Ssa197", "Ssa289", "Ssa85", "Ssa171",
        "SSOSL417", "Ssa202",
    )
    assert benchmark.STRICT_MICROSATELLITE_LOCI == tuple(
        locus for locus in benchmark.MICROSATELLITE_LOCI if locus != "Ssa14"
    )
    assert benchmark.EXPECTED_MISSING_BY_LOCUS == {
        "SSsp2210": 0, "SSsp2216": 6, "SsaD157": 61, "Ssa407": 11,
        "SSspG7": 2, "SSsp3016": 0, "SSsp2201": 123, "Ssa14": 69,
        "SSsp1605": 126, "SSOSL85": 21, "SSOSL438": 3, "Ssa197": 3,
        "Ssa289": 39, "Ssa85": 0, "Ssa171": 24, "SSOSL417": 27,
        "Ssa202": 50,
    }
    assert benchmark.EXPECTED_MISSING_BY_POPULATION == {
        "Kei9697a": 0, "Kei0708a": 39, "Kei0910a": 7, "Kei1112a": 67,
        "Vas9699a": 0, "Vas0708a": 2, "Vas0910a": 1, "Vas1112a": 41,
        "Kun9697a": 6, "Kun0708a": 2, "Kun0910a": 0, "Kun1112a": 167,
        "Loobu9699a": 2, "Loobu0708a": 5, "Loobu0910a": 0,
        "Loobu1112a": 0,
        "Narva98a": 10, "Narva01a": 8, "Narva04a": 138, "Narva05a": 14,
        "Narva06a": 6, "Narva07a": 28, "Narva08a": 10, "Narva09a": 0,
        "Neva9798a": 12,
    }
    assert sum(benchmark.EXPECTED_MISSING_BY_LOCUS.values()) == 565
    assert sum(benchmark.EXPECTED_MISSING_BY_POPULATION.values()) == 565
    assert benchmark.EXPECTED_MISSING_BY_POPULATION["Loobu9699a"] == 2
    assert benchmark.EXPECTED_MISSING_BY_POPULATION["Loobu0708a"] == 5
    assert sum(
        benchmark.EXPECTED_MISSING_BY_POPULATION[name]
        for name in benchmark.NARVA_MICROSATELLITE_GROUPS
    ) == 214
    assert benchmark.EXPECTED_MISSING_BY_LOCUS["SSsp1605"] == 126
    assert benchmark.EXPECTED_MISSING_BY_LOCUS["SSsp2201"] == 123
    assert benchmark.EXPECTED_MISSING_PER_INDIVIDUAL == {
        0: 1_842, 1: 79, 2: 31, 3: 36, 4: 32,
        5: 15, 6: 9, 7: 5, 8: 3,
    }
    assert {
        "locus": benchmark.EXPECTED_LOCUS_SHA256,
        "locus_newline": benchmark.EXPECTED_LOCUS_NEWLINE_SHA256,
        "strict_locus": benchmark.EXPECTED_STRICT_LOCUS_SHA256,
        "sample": benchmark.EXPECTED_SAMPLE_LEDGER_SHA256,
        "groups": benchmark.EXPECTED_GROUP_COUNT_SHA256,
        "snp_headers": benchmark.EXPECTED_SNP_HEADER_SHA256,
        "snp_records": benchmark.EXPECTED_SNP_RECORD_LEDGER_SHA256,
        "selection": benchmark.EXPECTED_LOCUS_SELECTION_LEDGER_SHA256,
    } == {
        "locus": "3bebd24695b4936df8e0c4ec526de80a7deb4e0b67910c1bec64510751c5c4a1",
        "locus_newline": "904e10f4c75adc35e8e1c9265876aee147c8aee0508726e542168b1ebdb2863f",
        "strict_locus": "9d81b14188c78ef4ef170dcacebbed0faef284ca31b5f087789c10a3bd61932e",
        "sample": "379a3a47136bd2564671947ff91db08cedcf9b2ddfbccf10b11d917a025a2767",
        "groups": "0668c7328fef59c6a0ab33d3073fe557d8fe14e9a588b9613455ab4067370fb9",
        "snp_headers": "691222805a1b0b6df7552df5c492adebafb5a268b485ee0bab1632cb2c9c8f52",
        "snp_records": "e6f380402be791d29aafd5f1f0a78b554eeedc00f785b531a409a8f682dbdaae",
        "selection": "b4ee09259a00ff534929de85a46a040ed767e7fd6c8de1fd294a78d9b3f0ab4c",
    }
    assert benchmark.SNP_HEADERS == (
        "Vas96-99", "Vas07-08", "Vas09-10", "Vas11-12",
        "Kei96-97", "Kei07-08", "Kei09-10", "Kei11-12",
        "Loo96-99", "Loo07-08", "Loo09-10", "Loo11-12",
        "Kun96-97", "Kun07-08", "Kun09-10", "Kun11-12",
        "Nev96-97", "Nar06",
    )


def test_full_runner_is_fail_closed_to_the_exact_azure_host():
    target = benchmark.require_azure_execution_target(
        "azure", os_name="posix", hostname="trading-linux-az"
    )
    assert target["verified_azure_host"] is True
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
            "azure", os_name="posix", hostname="other-host"
        )


def test_configuration_recursively_disallows_accuracy_acceptance_and_truth_units():
    config = benchmark.configuration({})
    json.dumps(config, allow_nan=False)

    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if "accuracy" in key.lower() or "accepted" in key.lower():
                    assert child is False or child is None
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(config)
    evaluation = config["evaluation"]
    assert evaluation["independent_direction_truth_units"] == 0
    assert evaluation["independent_gate_truth_units"] == 0
    assert evaluation["direction_calls_accepted"] is False
    assert evaluation["gate_calls_accepted"] is False
    assert config["panel"]["published_estimates_are_truth_labels"] is False
    assert config["panel"]["management_arrow_is_gate_truth"] is False
    assert config["pooled_snp_contract"]["used_for_padze_scoring"] is False
    assert config["pooled_snp_contract"]["used_for_truth_construction"] is False
    assert config["microsatellite_contract"]["missing_0_0_pairs"] == 565


@pytest.mark.skipif(
    not os.environ.get("DNNAIC_OZEROV_SOURCE_DIR"),
    reason="set DNNAIC_OZEROV_SOURCE_DIR to run the exact external-source contract",
)
def test_exact_sources_parse_build_and_remain_descriptive_and_unaccepted():
    source = Path(os.environ["DNNAIC_OZEROV_SOURCE_DIR"])
    microsatellite_path = source / benchmark.SOURCE_CONTRACTS["microsatellite"]["file"]
    snp_path = source / benchmark.SOURCE_CONTRACTS["pooled_snp"]["file"]
    benchmark.verify_source(microsatellite_path, "microsatellite")
    benchmark.verify_source(snp_path, "pooled_snp")

    loci, individuals, microsatellite_audit = benchmark.parse_microsatellite_workbook(
        microsatellite_path
    )
    headers, snps, snp_audit = benchmark.parse_pooled_snp_workbook(snp_path)
    assert loci == benchmark.MICROSATELLITE_LOCI
    assert headers == benchmark.SNP_HEADERS
    assert len(individuals) == len({individual.sample_id for individual in individuals}) == 2_052
    assert microsatellite_audit["diploid_genotype_pairs"] == 34_884
    assert microsatellite_audit["called_pairs"] == 34_319
    assert microsatellite_audit["missing_pairs"] == 565
    assert microsatellite_audit["individuals_with_missing_calls"] == 210
    assert microsatellite_audit["missing_pairs_by_population"] == (
        benchmark.EXPECTED_MISSING_BY_POPULATION
    )
    assert microsatellite_audit["partial_missing_pairs"] == 0
    assert microsatellite_audit["all_nonmissing_pairs_nondecreasing"] is True
    assert microsatellite_audit["normalized_sample_ledger_sha256"] == (
        benchmark.EXPECTED_SAMPLE_LEDGER_SHA256
    )
    assert len(snps) == 1_986
    assert snp_audit["frequency_values"] == 35_748
    assert snp_audit["all_frequencies_finite_in_unit_interval"] is True
    assert (
        snp_audit["null_chromosome_rows"],
        snp_audit["null_female_map_positions"],
        snp_audit["null_male_map_positions"],
        snp_audit["null_chromosome_female_male_triplets"],
    ) == (64, 64, 64, 64)
    assert snp_audit["normalized_snp_record_ledger_sha256"] == (
        benchmark.EXPECTED_SNP_RECORD_LEDGER_SHA256
    )

    panels = benchmark.make_panels(individuals)
    assert [[len(group) for group in panel["groups"]] for panel in panels] == [
        [81, 77, 720],
        [81, 77, 112],
    ]
    records, selection = benchmark.build_records(loci, panels, compute_state=None)
    assert [record["contract_role"] for record in records] == [
        "primary",
        "published_snp_donor_alignment_sensitivity",
        "within_population_polymorphic_locus_sensitivity",
    ]
    assert [record["count_audit"]["loci"] for record in records] == [17, 17, 16]
    assert [
        record["count_audit"]["ordered_allele_count_ledger_sha256"]
        for record in records
    ] == [
        "3499bd2f374013542029819c3d032ff0b257b967aec9a6095787bac6062788d0",
        "c7c34759403ba3039e5ca9c9ac08bf49dcc8802e6d402fc67b452b9a8113f637",
        "b0798ef86f9b9b7402017db3c2aa5a7ff001bd0d92cf5942cda5592f99995855",
    ]
    assert [record["count_audit"]["missing_copy_fraction"] for record in records] == pytest.approx([
        0.014806378132118492,
        0.002832244008714624,
        0.012599658314350837,
    ])
    assert selection["standard_loci"] == 17
    assert selection["strict_loci"] == 16
    assert selection["strict_excluded_loci"] == ["Ssa14"]
    assert selection["selection_ledger_sha256"] == (
        "b4ee09259a00ff534929de85a46a040ed767e7fd6c8de1fd294a78d9b3f0ab4c"
    )
    assert selection["independent_direction_truth_units"] == 0
    assert selection["independent_gate_truth_units"] == 0

    for record in records:
        curve = np.asarray(record["curve"], dtype=float)
        assert curve.shape == (15, 28)
        assert np.isfinite(curve).all()
        assert np.array_equal(curve[:, 0], benchmark.brook.PRIMARY_DEPTHS)
        assert record["direction_call_accepted"] is False
        assert record["gate_call_accepted"] is False
        assert record["formal_direction_accuracy_eligible"] is False
        assert record["formal_gate_accuracy_eligible"] is False
        assert record["gate_accuracy_eligible"] is False
        assert record["direction_accuracy_estimate"] is None
        assert record["gate_accuracy_estimate"] is None
        assert record["independent_direction_truth_units"] == 0
        assert record["independent_gate_truth_units"] == 0
        assert record["published_estimates_are_truth_labels"] is False
        assert record["management_arrow_is_gate_truth"] is False

    primary_geometry = records[0]["frequency_geometry"]
    assert primary_geometry["mean_projection"] == pytest.approx(0.7250153663980101)
    assert primary_geometry["median_projection"] == pytest.approx(0.6910142970827352)
    assert primary_geometry["denominator_weighted_projection"] == pytest.approx(
        0.6846443546002623
    )
    assert primary_geometry["mean_f3"] == pytest.approx(-0.03618282775536649)
    assert primary_geometry["negative_f3_loci"] == 14
    narva06_geometry = records[1]["frequency_geometry"]
    assert narva06_geometry["mean_projection"] == pytest.approx(0.731495843112867)
    assert narva06_geometry["median_projection"] == pytest.approx(0.6693669150732611)
    assert narva06_geometry["denominator_weighted_projection"] == pytest.approx(
        0.671038171249914
    )
    assert narva06_geometry["mean_f3"] == pytest.approx(-0.03571883542867122)
    assert narva06_geometry["negative_f3_loci"] == 14

    snp_geometry = benchmark.pooled_snp_frequency_geometry(snps)
    assert snp_geometry["mean_projection"] == pytest.approx(-0.11691207363282481)
    assert snp_geometry["median_projection"] == pytest.approx(0.5544214578227046)
    assert snp_geometry["denominator_weighted_projection"] == pytest.approx(
        0.6044532175815932
    )
    assert snp_geometry["mean_f3"] == pytest.approx(-0.0035933309527123166)
    assert snp_geometry["negative_f3_snps"] == 1_104
    assert "not the published S-hat estimator" in snp_geometry["description"]
