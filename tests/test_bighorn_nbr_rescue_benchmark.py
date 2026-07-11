from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts import bighorn_nbr_rescue_benchmark as benchmark


def test_source_contracts_and_management_direction_are_pinned():
    assert benchmark.EXPECTED_DIRECTION == "C"
    assert benchmark.GROUP_COUNTS == {
        "ParentalPop1": 13,
        "Hybrid": 188,
        "ParentalPop2": 18,
    }
    expected = {
        "genotypes": (
            343_141,
            "7cb626b9efd947b7ad9c064b9c765c7f",
            "30fce85b28fa38bff8c3c72cf862f2eb7cdfdfd996c54c7faccfa03c9b3c719c",
        ),
        "modeling": (
            58_542,
            "f25bea9804a95baad59f7463cbebc8de",
            "075d3816a6127494d679df6a3eaeebb334280b14a371a026c623cf24339e082a",
        ),
    }
    for name, contract in benchmark.SOURCE_CONTRACTS.items():
        assert (contract["bytes"], contract["md5"], contract["sha256"]) == expected[name]
        assert contract["license"] == "CC0-1.0"
        assert contract["merritt_ark"] == "ark:/13030/m5fr4w25"


@pytest.mark.parametrize(
    "token,expected",
    [
        ("158/188", (158, 188)),
        ("173/172", (172, 173)),
        ("1/1", (1, 1)),
        ("NA/NA", (0, 0)),
    ],
)
def test_multiallelic_genotype_decoder(token, expected):
    assert benchmark.decode_genotype(token, "toy") == expected


@pytest.mark.parametrize("token", ["NA/188", "158/NA", "158", "1/2/3", "0/2", "-1/2"])
def test_genotype_decoder_rejects_malformed_or_partial_missing(token):
    with pytest.raises(ValueError):
        benchmark.decode_genotype(token, "toy")


def test_ordinal_identity_never_collapses_repeated_raw_ids():
    assert benchmark.ordinal_sample_id(4, "264") != benchmark.ordinal_sample_id(5, "264")
    assert benchmark.ordinal_sample_id(4, "264") == "nbr-col-004:264"
    with pytest.raises(ValueError):
        benchmark.ordinal_sample_id(3, "264")


def test_called_genotype_rate_contract_enforces_population_specific_75_percent():
    def individual(sample_id, pairs, ordinal):
        return benchmark.brook.Individual(
            sample_id=sample_id,
            population=sample_id[0],
            alleles=tuple(pairs),
            source_ordinal=ordinal,
        )

    groups = tuple(
        [
            individual(f"{name}{index}", [(1, 1), (1, 1) if index < called else (0, 0)], 4 + index)
            for index in range(4)
        ]
        for name, called in (("a", 3), ("b", 2), ("c", 4))
    )
    passing, audit = benchmark.called_genotype_rate_contract(["L1", "L2"], groups)
    assert passing == ["L1"]
    assert audit["required_called_individuals"] == {"P1": 3, "P2": 3, "P3": 3}
    assert audit["minimum_observed_called_individuals"] == {"P1": 3, "P2": 2, "P3": 4}
    assert audit["minimum_called_genotype_fraction_per_population"] == 0.75


def test_panel_marks_management_arrow_but_not_independent_pedigree_truth():
    panel = benchmark.make_panel(([], [], []))
    comparator = panel["published_comparator"]
    assert comparator["management_intervention_direction_independent_of_loci"] is True
    assert comparator["pedigree_membership_independent_of_scored_loci"] is False
    assert comparator["independent_truth"] is False
    assert panel["expected_candidate_direction"] == "C"
    assert panel["management_translocation_direction_known"] is True
    assert panel["individual_labels_independent_of_markers"] is False
    assert "operational" in panel["operational_tree_guardrail"]


def test_full_runner_is_fail_closed_to_the_exact_azure_host():
    target = benchmark.require_azure_execution_target(
        "azure", os_name="posix", hostname="trading-linux-az"
    )
    assert target["verified_azure_host"] is True
    with pytest.raises(RuntimeError):
        benchmark.require_azure_execution_target(
            "local", os_name="posix", hostname="trading-linux-az"
        )
    with pytest.raises(RuntimeError):
        benchmark.require_azure_execution_target(
            "azure", os_name="nt", hostname="MATH-ROSS20"
        )


def test_configuration_recursively_disallows_accuracy_and_acceptance():
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
    assert config["evaluation"]["independent_direction_truth_units"] == 0
    assert config["evaluation"]["independent_gate_truth_units"] == 0
    assert config["evaluation"]["modeling_table_used_for_scoring"] is False
    assert config["evaluation"]["modeling_table_used_for_truth"] is False
    assert config["locus_contract"]["primary_autosomal_standard_loci"] == 187
    assert config["locus_contract"]["published_representation_sensitivity_loci"] == 195
    assert config["locus_contract"]["autosomal_strict_sensitivity_loci"] == 148


def test_pinned_locus_and_normalized_ledgers_are_full_sha256():
    assert {
        "all_standard": benchmark.EXPECTED_ALL_STANDARD_LOCUS_SHA256,
        "all_strict": benchmark.EXPECTED_ALL_STRICT_LOCUS_SHA256,
        "autosomal_standard": benchmark.EXPECTED_AUTOSOMAL_STANDARD_LOCUS_SHA256,
        "autosomal_strict": benchmark.EXPECTED_AUTOSOMAL_STRICT_LOCUS_SHA256,
        "all_newline": benchmark.EXPECTED_ALL_ORDERED_NEWLINE_SHA256,
        "autosomal_newline": benchmark.EXPECTED_AUTOSOMAL_ORDERED_NEWLINE_SHA256,
        "locus_ledger": benchmark.EXPECTED_LOCUS_LEDGER_SHA256,
        "sample_ledger": benchmark.EXPECTED_SAMPLE_LEDGER_SHA256,
        "model_ledger": benchmark.EXPECTED_MODEL_LEDGER_SHA256,
    } == {
        "all_standard": "8fd7f056917a1bda8703f466c75d472b95987c11c567e3cd3a42756833858490",
        "all_strict": "26c8ab9564bde0f21d66d57c19200b4b3bd0a9980ab78a7791bca1f580f9ba13",
        "autosomal_standard": "6eb8aa3c41e44ccc54a91c9c515dd8a777554dc1836b5f7a981962c13083b504",
        "autosomal_strict": "e8fbfa9344bb332e2d6fffb38875cdddb0a33939034d5d3ede9457b9e16f27ce",
        "all_newline": "8e0b8ae56515519491726a2f7fe307e0e8544074b0d1a198797b24b5ae60ff91",
        "autosomal_newline": "0eab9ad7c57dce0fe0b2e12d7b62c4e28f7b8e08ebe0fd27582b7042c7d1d82b",
        "locus_ledger": "98e5678f0455f6d1148b3c856575156c5ebf3bf427726e43a5fad71dfd13bbfc",
        "sample_ledger": "2c6a51e1f6463ac2edc6f5b196185af39081516ea65eec4e8795f7b139079c16",
        "model_ledger": "ca41f02085f4cfa39e0883b7e7ada8450a083809c4168ff67fe69bc784a5d475",
    }


def test_x_linked_locus_contract_is_explicit_and_exact():
    assert benchmark.EXPECTED_X_LOCI == (
        "MAF48", "MCM158", "MNS46A", "FCB19", "CSRD81", "AE25", "CP131", "MCM25"
    )


@pytest.mark.skipif(
    not os.environ.get("DNNAIC_NBR_SOURCE_DIR"),
    reason="set DNNAIC_NBR_SOURCE_DIR to run the exact external-source contract",
)
def test_exact_external_sources_parse_build_and_remain_unaccepted():
    source = Path(os.environ["DNNAIC_NBR_SOURCE_DIR"])
    genotype_path = source / benchmark.SOURCE_CONTRACTS["genotypes"]["file"]
    modeling_path = source / benchmark.SOURCE_CONTRACTS["modeling"]["file"]
    benchmark.verify_source(genotype_path, "genotypes")
    benchmark.verify_source(modeling_path, "modeling")

    loci, autosomal, groups, genotype_audit = benchmark.parse_genotypes(genotype_path)
    modeling_audit = benchmark.parse_modeling_table(modeling_path, loci)
    assert (len(loci), len(autosomal), [len(group) for group in groups]) == (
        195,
        187,
        [13, 188, 18],
    )
    assert genotype_audit["descending_call_contexts"] == [
        {"locus": "BM719", "source_tsv_column": 191, "source_token": "173/172"},
        {"locus": "BM719", "source_tsv_column": 195, "source_token": "173/172"},
        {"locus": "BM719", "source_tsv_column": 197, "source_token": "173/172"},
    ]
    assert modeling_audit["excluded_from_scoring"] is True
    assert modeling_audit["excluded_from_truth_construction"] is True

    records, selection = benchmark.build_records(
        loci, autosomal, benchmark.make_panel(groups), compute_state=None
    )
    rate = selection["called_genotype_rate_contract"]
    assert rate["passing_loci"] == 195
    assert rate["required_called_individuals"] == {"P1": 10, "P2": 141, "P3": 14}
    assert rate["minimum_observed_called_individuals"] == {
        "P1": 10,
        "P2": 157,
        "P3": 14,
    }
    assert [record["count_audit"]["loci"] for record in records] == [187, 195, 148]
    assert [record["contract_role"] for record in records] == [
        "primary",
        "published_representation_sensitivity",
        "strong_ascertainment_locus_filter_sensitivity",
    ]
    for record in records:
        assert np.asarray(record["curve"]).shape == (15, 28)
        assert record["direction_call_accepted"] is False
        assert record["gate_call_accepted"] is False
        assert record["formal_direction_accuracy_eligible"] is False
        assert record["formal_gate_accuracy_eligible"] is False
        assert record["direction_accuracy_estimate"] is None
        assert record["gate_accuracy_estimate"] is None
