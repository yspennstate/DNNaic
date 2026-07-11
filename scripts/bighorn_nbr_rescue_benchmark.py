#!/usr/bin/env python3
"""Score the National Bison Range bighorn-sheep genetic-rescue panel.

The source is a 195-locus multiallelic microsatellite matrix.  Every allele is
retained and passed directly to PADZE.  The management candidate is migrant
ParentPop2 P3 -> hybrid descendants P2 (DNNaic class C), relative to resident
ParentPop1 P1.

The translocation arrow is independent management history, but the published
pedigree was updated with a genome-wide microsatellite panel overlapping these
loci.  The primary autosomal view and two correlated locus sensitivities are
therefore unaccepted ecological transfer diagnostics, not independent
direction-accuracy or gate-accuracy rows.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dnnaic.semantics import class_for_forward_edge
from scripts import brook_trout_microsatellite_benchmark as brook
from scripts import stdpopsim_neanderthal_benchmark as stdbench
from scripts import structured_transfer_pilot as structured


SCHEMA_VERSION = "dnnaic-bighorn-nbr-rescue-benchmark-v1"
DEFAULT_RESULT_DIR = REPO / "results" / "bighorn_nbr_rescue_benchmark_2026_07_11"
EXPECTED_DIRECTION = class_for_forward_edge("P3", "P2")
GROUP_ORDER = ("ParentalPop1", "Hybrid", "ParentalPop2")
GROUP_COUNTS = {"ParentalPop1": 13, "Hybrid": 188, "ParentalPop2": 18}
EXPECTED_ALL_STANDARD_LOCUS_SHA256 = "8fd7f056917a1bda8703f466c75d472b95987c11c567e3cd3a42756833858490"
EXPECTED_ALL_STRICT_LOCUS_SHA256 = "26c8ab9564bde0f21d66d57c19200b4b3bd0a9980ab78a7791bca1f580f9ba13"
EXPECTED_AUTOSOMAL_STANDARD_LOCUS_SHA256 = "6eb8aa3c41e44ccc54a91c9c515dd8a777554dc1836b5f7a981962c13083b504"
EXPECTED_AUTOSOMAL_STRICT_LOCUS_SHA256 = "e8fbfa9344bb332e2d6fffb38875cdddb0a33939034d5d3ede9457b9e16f27ce"
EXPECTED_ALL_ORDERED_NEWLINE_SHA256 = "8e0b8ae56515519491726a2f7fe307e0e8544074b0d1a198797b24b5ae60ff91"
EXPECTED_AUTOSOMAL_ORDERED_NEWLINE_SHA256 = "0eab9ad7c57dce0fe0b2e12d7b62c4e28f7b8e08ebe0fd27582b7042c7d1d82b"
EXPECTED_LOCUS_LEDGER_SHA256 = "98e5678f0455f6d1148b3c856575156c5ebf3bf427726e43a5fad71dfd13bbfc"
EXPECTED_SAMPLE_LEDGER_SHA256 = "2c6a51e1f6463ac2edc6f5b196185af39081516ea65eec4e8795f7b139079c16"
EXPECTED_MODEL_LEDGER_SHA256 = "ca41f02085f4cfa39e0883b7e7ada8450a083809c4168ff67fe69bc784a5d475"
MIN_CALLED_GENOTYPE_FRACTION = 0.75
EXPECTED_X_LOCI = (
    "MAF48", "MCM158", "MNS46A", "FCB19", "CSRD81", "AE25", "CP131", "MCM25"
)

SOURCE_CONTRACTS = {
    "genotypes": {
        "file": "NBR genotyoes.txt",
        "bytes": 343_141,
        "md5": "7cb626b9efd947b7ad9c064b9c765c7f",
        "sha256": "30fce85b28fa38bff8c3c72cf862f2eb7cdfdfd996c54c7faccfa03c9b3c719c",
        "dryad_file_id": 20_410,
        "merritt_ark": "ark:/13030/m5fr4w25",
        "merritt_version": 1,
        "stable_object_key": "ark:/13030/m5fr4w25|1|producer/NBR genotyoes.txt",
        "download": "https://datadryad.org/downloads/file_stream/20410",
        "data_doi": "10.5061/dryad.gv13nm40",
        "paper_doi": "10.1111/j.1365-294X.2011.05427.x",
        "license": "CC0-1.0",
    },
    "modeling": {
        "file": "NBR modeling_table.txt",
        "bytes": 58_542,
        "md5": "f25bea9804a95baad59f7463cbebc8de",
        "sha256": "075d3816a6127494d679df6a3eaeebb334280b14a371a026c623cf24339e082a",
        "dryad_file_id": 20_411,
        "merritt_ark": "ark:/13030/m5fr4w25",
        "merritt_version": 1,
        "stable_object_key": "ark:/13030/m5fr4w25|1|producer/NBR modeling_table.txt",
        "download": "https://datadryad.org/downloads/file_stream/20411",
        "data_doi": "10.5061/dryad.gv13nm40",
        "paper_doi": "10.1111/j.1365-294X.2011.05427.x",
        "license": "CC0-1.0",
    },
}

GENOTYPE_TOKEN = re.compile(r"^([0-9]+)/([0-9]+)$")


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _ordered_newline_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def verify_source(path: Path, contract_name: str) -> dict:
    contract = SOURCE_CONTRACTS[contract_name]
    size = path.stat().st_size
    sha256 = structured.sha256_file(path)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if (size, md5, sha256) != (
        contract["bytes"], contract["md5"], contract["sha256"]
    ):
        raise RuntimeError(f"{contract_name} source contract changed")
    return {
        "path": str(path.resolve()),
        "bytes": size,
        "md5": md5,
        "sha256": sha256,
        "contract": contract,
    }


def decode_genotype(token: str, context: str) -> tuple[int, int]:
    if token == "NA/NA":
        return 0, 0
    match = GENOTYPE_TOKEN.fullmatch(token)
    if match is None:
        raise ValueError(f"{context}: malformed microsatellite genotype {token!r}")
    first, second = map(int, match.groups())
    if first <= 0 or second <= 0:
        raise ValueError(f"{context}: zero/negative called allele")
    return tuple(sorted((first, second)))


def ordinal_sample_id(source_tsv_column: int, raw_id: str) -> str:
    if source_tsv_column < 4 or not raw_id:
        raise ValueError("sample identity requires an actual TSV sample column and raw ID")
    return f"nbr-col-{source_tsv_column:03d}:{raw_id}"


def _read_tsv(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [list(row) for row in csv.reader(handle, delimiter="\t")]


def parse_genotypes(
    path: Path,
) -> tuple[list[str], list[str], tuple[list[brook.Individual], ...], dict]:
    rows = _read_tsv(path)
    if len(rows) != 197 or any(len(row) != 222 for row in rows):
        raise AssertionError("NBR genotype matrix shape changed")
    if rows[0][:3] != ["Locus", "lg", "Markerpos."] or rows[1][:3] != ["", "", ""]:
        raise AssertionError("NBR genotype headers changed")

    labels = rows[0][3:]
    raw_ids = rows[1][3:]
    if Counter(labels) != Counter(GROUP_COUNTS) or any(not value for value in raw_ids):
        raise AssertionError("NBR sample label/ID contract changed")
    raw_id_counts = Counter(raw_ids)
    repeated_groups = sum(count > 1 for count in raw_id_counts.values())
    repeated_extra_columns = sum(count - 1 for count in raw_id_counts.values())
    if (len(set(raw_ids)), repeated_groups, repeated_extra_columns) != (187, 31, 32):
        raise AssertionError("NBR repeated raw-ID contract changed")

    locus_rows = rows[2:]
    locus_names = [row[0] for row in locus_rows]
    if len(locus_names) != 195 or len(set(locus_names)) != 195:
        raise AssertionError("NBR locus identities changed")
    linkage_groups = [row[1] for row in locus_rows]
    if set(linkage_groups) != {str(value) for value in range(1, 28)}:
        raise AssertionError("NBR linkage-group contract changed")
    x_loci = [row[0] for row in locus_rows if row[1] == "27"]
    autosomal_loci = [row[0] for row in locus_rows if row[1] != "27"]
    if tuple(x_loci) != EXPECTED_X_LOCI or len(autosomal_loci) != 187:
        raise AssertionError("NBR X/autosomal locus partition changed")
    if (
        _ordered_newline_sha256(locus_names) != EXPECTED_ALL_ORDERED_NEWLINE_SHA256
        or _ordered_newline_sha256(autosomal_loci)
        != EXPECTED_AUTOSOMAL_ORDERED_NEWLINE_SHA256
    ):
        raise AssertionError("NBR ordered source-locus identities changed")
    locus_ledger = hashlib.sha256()
    for row in locus_rows:
        if not row[1] or not row[2] or not math.isfinite(float(row[2])):
            raise ValueError(f"{row[0]}: invalid linkage metadata")
        locus_ledger.update(_canonical_json({
            "locus": row[0],
            "linkage_group": row[1],
            "marker_position": row[2],
        }))
        locus_ledger.update(b"\n")
    if locus_ledger.hexdigest() != EXPECTED_LOCUS_LEDGER_SHA256:
        raise AssertionError("NBR normalized locus ledger changed")

    column_signatures = [
        tuple(row[3 + index] for row in locus_rows)
        for index in range(len(labels))
    ]
    if len(set(column_signatures)) != len(column_signatures):
        raise AssertionError("NBR distinct genotype columns unexpectedly collapsed")

    individuals = []
    called = Counter()
    missing = Counter()
    descending_calls = []
    sample_ledger = hashlib.sha256()
    for sample_ordinal, (label, raw_id) in enumerate(zip(labels, raw_ids), start=1):
        source_tsv_column = sample_ordinal + 3
        pairs = []
        for locus_row in locus_rows:
            token = locus_row[source_tsv_column - 1]
            match = GENOTYPE_TOKEN.fullmatch(token)
            if match is not None and int(match.group(1)) > int(match.group(2)):
                descending_calls.append((locus_row[0], source_tsv_column, token))
            pair = decode_genotype(
                token, f"TSV column {source_tsv_column} locus {locus_row[0]}"
            )
            missing[label] += int(pair == (0, 0))
            called[label] += int(pair != (0, 0))
            pairs.append(pair)
        individual = brook.Individual(
            sample_id=ordinal_sample_id(source_tsv_column, raw_id),
            population=label,
            alleles=tuple(pairs),
            source_ordinal=source_tsv_column,
        )
        individuals.append(individual)
        sample_ledger.update(_canonical_json(asdict(individual)))
        sample_ledger.update(b"\n")

    if dict(called) != {
        "ParentalPop1": 2_403, "ParentalPop2": 3_415, "Hybrid": 35_968
    } or dict(missing) != {
        "ParentalPop1": 132, "ParentalPop2": 95, "Hybrid": 692
    }:
        raise AssertionError("NBR called/missing genotype contract changed")
    if sample_ledger.hexdigest() != EXPECTED_SAMPLE_LEDGER_SHA256:
        raise AssertionError("NBR normalized sample ledger changed")
    if descending_calls != [
        ("BM719", 191, "173/172"),
        ("BM719", 195, "173/172"),
        ("BM719", 197, "173/172"),
    ]:
        raise AssertionError("NBR allele-order normalization contract changed")

    groups = tuple([
        individual for individual in individuals if individual.population == label
    ] for label in GROUP_ORDER)
    if [len(group) for group in groups] != [13, 188, 18]:
        raise AssertionError("NBR P1/P2/P3 group sizes changed")
    return locus_names, autosomal_loci, groups, {
        "rows": len(rows),
        "columns": len(rows[0]),
        "loci": len(locus_names),
        "sample_columns": len(individuals),
        "group_counts": dict(Counter(labels)),
        "called_genotype_pairs": dict(called),
        "missing_genotype_pairs": dict(missing),
        "missing_pair_fraction": sum(missing.values()) / (195 * 219),
        "raw_numeric_ids_distinct": len(set(raw_ids)),
        "repeated_raw_id_groups": repeated_groups,
        "extra_columns_from_repeated_raw_ids": repeated_extra_columns,
        "distinct_full_genotype_columns": len(set(column_signatures)),
        "descending_called_pairs_normalized": len(descending_calls),
        "descending_call_contexts": [
            {"locus": locus, "source_tsv_column": column, "source_token": token}
            for locus, column, token in descending_calls
        ],
        "unphased_allele_contract": "each called microsatellite pair is sorted numerically",
        "autosomal_loci": len(autosomal_loci),
        "x_linked_loci": len(x_loci),
        "x_linked_locus_names": x_loci,
        "all_ordered_locus_newline_sha256": _ordered_newline_sha256(locus_names),
        "autosomal_ordered_locus_newline_sha256": _ordered_newline_sha256(autosomal_loci),
        "x_linked_guardrail": (
            "the paper forced male calls homozygous at linkage-group 27 loci; the primary view "
            "excludes these eight X-linked loci and the all-locus view is a sensitivity"
        ),
        "identity_contract": (
            "immutable one-based TSV column (4..222) plus raw ID; never deduplicate by raw ID"
        ),
        "ordered_locus_metadata_ledger_sha256": locus_ledger.hexdigest(),
        "ordered_sample_genotype_ledger_sha256": sample_ledger.hexdigest(),
        "group_sample_ledgers": {
            name: brook._sample_ledger(group)
            for name, group in zip(("P1", "P2", "P3"), groups)
        },
    }


def parse_modeling_table(path: Path, locus_names: Sequence[str]) -> dict:
    rows = _read_tsv(path)
    if len(rows) != 137 or any(len(row) != 202 for row in rows):
        raise AssertionError("NBR modeling-table shape changed")
    expected_meta = [
        "ID", "Gender", "h.index", "Year_Birth", "Longevity", "Progeny", "Introduced"
    ]
    if rows[0][:7] != expected_meta or rows[0][7:] != list(locus_names):
        raise AssertionError("NBR modeling-table header/locus order changed")

    ledger = hashlib.sha256()
    raw_ids = []
    sexes = Counter()
    introduced = Counter()
    h_index = []
    missing = 0
    for source_row, row in enumerate(rows[1:], start=2):
        raw_ids.append(row[0])
        sexes[row[1]] += 1
        introduced[row[6]] += 1
        value = float(row[2])
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"modeling row {source_row}: invalid hybrid index")
        h_index.append(value)
        for token in row[7:]:
            if token not in {"0", "1", "2", "NA"}:
                raise ValueError(f"modeling row {source_row}: invalid ancestry token")
            missing += int(token == "NA")
        ledger.update(_canonical_json({"source_row": source_row, "values": row}))
        ledger.update(b"\n")

    counts = Counter(raw_ids)
    if (
        sexes != Counter({"M": 70, "F": 66})
        or introduced != Counter({"0": 124, "1": 12})
        or len(set(raw_ids)) != 120
        or sum(count > 1 for count in counts.values()) != 16
        or missing != 460
        or len(set(map(tuple, rows[1:]))) != 136
    ):
        raise AssertionError("NBR modeling-table identity/value contract changed")
    if ledger.hexdigest() != EXPECTED_MODEL_LEDGER_SHA256:
        raise AssertionError("NBR normalized modeling-table ledger changed")
    return {
        "rows": 136,
        "columns": 202,
        "loci": 195,
        "distinct_raw_ids": 120,
        "repeated_raw_id_groups": 16,
        "all_rows_distinct": True,
        "sex_counts": dict(sexes),
        "introduced_indicator_counts": dict(introduced),
        "hybrid_index": {
            "minimum": min(h_index),
            "mean": float(np.mean(h_index)),
            "maximum": max(h_index),
            "source": "INTROGRESS analysis from the same microsatellite panel",
            "independent_truth": False,
        },
        "missing_ancestry_genotype_cells": missing,
        "ordered_row_ledger_sha256": ledger.hexdigest(),
        "join_guardrail": (
            "raw IDs repeat in both files and row multiplicities differ; the modeling table is "
            "audited as a separate same-marker comparator and is never joined to genotype columns"
        ),
        "excluded_from_scoring": True,
        "excluded_from_truth_construction": True,
    }


def called_genotype_rate_contract(
    locus_names: Sequence[str],
    groups: Sequence[Sequence[brook.Individual]],
    *,
    minimum_fraction: float = MIN_CALLED_GENOTYPE_FRACTION,
) -> tuple[list[str], dict]:
    if len(groups) != 3 or any(not group for group in groups):
        raise ValueError("call-rate validation requires three nonempty populations")
    if not 0 < minimum_fraction <= 1:
        raise ValueError("minimum called-genotype fraction must lie in (0, 1]")
    if any(
        len(individual.alleles) != len(locus_names)
        for group in groups
        for individual in group
    ):
        raise ValueError("call-rate validation genotype width changed")

    passing = []
    ledger = []
    called_by_population = [[] for _ in groups]
    for locus_index, locus in enumerate(locus_names):
        called = [
            sum(individual.alleles[locus_index] != (0, 0) for individual in group)
            for group in groups
        ]
        fractions = [count / len(group) for count, group in zip(called, groups)]
        eligible = all(value >= minimum_fraction for value in fractions)
        if eligible:
            passing.append(str(locus))
        for population, count in enumerate(called):
            called_by_population[population].append(count)
        ledger.append({
            "locus": str(locus),
            "called_individuals": called,
            "called_fractions": fractions,
            "passes": bool(eligible),
        })

    return passing, {
        "minimum_called_genotype_fraction_per_population": minimum_fraction,
        "required_called_individuals": {
            name: math.ceil(minimum_fraction * len(group))
            for name, group in zip(("P1", "P2", "P3"), groups)
        },
        "minimum_observed_called_individuals": {
            name: min(values)
            for name, values in zip(("P1", "P2", "P3"), called_by_population)
        },
        "minimum_observed_called_fraction": {
            name: min(values) / len(group)
            for name, values, group in zip(
                ("P1", "P2", "P3"), called_by_population, groups
            )
        },
        "passing_loci": len(passing),
        "input_loci": len(locus_names),
        "selection_ledger_sha256": hashlib.sha256(_canonical_json(ledger)).hexdigest(),
    }


def make_panel(groups: Sequence[Sequence[brook.Individual]]) -> dict:
    return {
        "panel_id": "nbr_resident_hybrid_migrant",
        "dataset": "miller_2012_nbr_bighorn_rescue",
        "biological_system_id": "nbr_genetic_rescue_1985_1994",
        "contract_role": "primary",
        "groups": tuple(groups),
        "expected_candidate_direction": EXPECTED_DIRECTION,
        "population_semantics": {
            "P1": "resident NBR founders and pedigree-defined pure progeny (ParentalPop1)",
            "P2": "hybrid descendants (Hybrid)",
            "P3": "translocated migrants and pedigree-defined pure progeny (ParentalPop2)",
            "forward_candidate": "P3->P2",
        },
        "candidate_direction_evidence": (
            "documented 1985 and 1990-1994 translocations into the NBR population independently "
            "fix the management arrow from migrants toward hybrid descendants"
        ),
        "management_translocation_direction_known": True,
        "individual_labels_independent_of_markers": False,
        "operational_tree_guardrail": (
            "((P1,P2),P3) is an operational DNNaic mapping only; P2 is a related, admixed, "
            "multigeneration descendant group rather than a clean population lineage"
        ),
        "published_comparator": {
            "parental_group_counts": {"ParentalPop1": 13, "ParentalPop2": 18},
            "hybrid_group_count": 188,
            "translocation_periods": ["1985", "1990-1994"],
            "management_intervention_direction_independent_of_loci": True,
            "pedigree_membership_independent_of_scored_loci": False,
            "reason_not_independent": (
                "the NBR pedigree was updated with a >200-locus genome-wide microsatellite data "
                "set overlapping this 195-locus panel"
            ),
            "independent_truth": False,
        },
        "selection_guardrail": (
            "one related longitudinal rescue system and three correlated locus views; 188 hybrid "
            "columns are not independent direction trials"
        ),
    }


def _panel_record(
    panel: Mapping[str, object],
    locus_names: Sequence[str],
    selected_loci: Sequence[str],
    *,
    filter_name: str,
    strict: bool,
    compute_state: Path | None,
) -> dict:
    groups = panel["groups"]
    counts, sizes, count_audit = brook.panel_to_counts(
        locus_names,
        groups,
        selected_loci,
        require_within_population_polymorphism=strict,
    )
    curve, padze_audit = brook.loci_to_curve(
        counts,
        sizes,
        selected_loci,
        groups,
        n_loci_read=len(locus_names),
        source=f"NBR bighorn rescue filter {filter_name}",
        filters=[
            "at least 75% called diploid genotypes in each of P1/P2/P3",
            "at least 16 called gene copies in P1/P2/P3 as a technical PADZE floor",
            "globally polymorphic in the fixed triplet; every allele retained",
            *(["polymorphic within every population"] if strict else []),
            *(
                ["published-representation sensitivity includes eight X-linked loci with forced male homozygosity"]
                if any(locus in EXPECTED_X_LOCI for locus in selected_loci)
                else ["autosomal view excludes linkage group 27"]
            ),
        ],
        compute_state=compute_state,
    )
    output = {key: value for key, value in panel.items() if key != "groups"}
    output.update({
        "panel_view_id": f"{panel['panel_id']}__{filter_name}",
        "locus_filter": filter_name,
        "count_audit": count_audit,
        "frequency_geometry": brook.multiallelic_frequency_geometry(counts, selected_loci),
        "padze": padze_audit,
        "curve": curve,
        "formal_direction_accuracy_eligible": False,
        "formal_gate_accuracy_eligible": False,
        "gate_accuracy_eligible": False,
        "direction_call_accepted": False,
        "gate_call_accepted": False,
        "independent_direction_truth_units": 0,
        "independent_gate_truth_units": 0,
        "direction_accuracy_estimate": None,
        "gate_accuracy_estimate": None,
        "ground_truth_guardrail": (
            "management history fixes a candidate arrow, but overlapping-locus pedigree labels, "
            "related descendants, and continuous multi-generation rescue violate clean accuracy semantics"
        ),
    })
    return output


def build_records(
    locus_names: Sequence[str],
    autosomal_loci: Sequence[str],
    panel: Mapping[str, object],
    *,
    compute_state: Path | None,
) -> tuple[list[dict], dict]:
    call_rate_loci, call_rate_audit = called_genotype_rate_contract(
        locus_names, panel["groups"]
    )
    if call_rate_loci != list(locus_names):
        raise AssertionError("NBR per-population 75% call-rate contract changed")
    standard, strict, audit = brook.eligible_shared_loci(locus_names, [panel])
    if (len(standard), len(strict)) != (195, 156):
        raise AssertionError("NBR shared-locus counts changed")
    if (
        audit["standard_ordered_locus_sha256"] != EXPECTED_ALL_STANDARD_LOCUS_SHA256
        or audit["strict_ordered_locus_sha256"] != EXPECTED_ALL_STRICT_LOCUS_SHA256
    ):
        raise AssertionError("NBR ordered shared-locus identities changed")
    autosomal_set = set(autosomal_loci)
    if len(autosomal_set) != 187 or [
        locus for locus in locus_names if locus in autosomal_set
    ] != list(autosomal_loci):
        raise AssertionError("NBR autosomal locus order changed")
    autosomal_standard = [locus for locus in standard if locus in autosomal_set]
    autosomal_strict = [locus for locus in strict if locus in autosomal_set]
    if (len(autosomal_standard), len(autosomal_strict)) != (187, 148):
        raise AssertionError("NBR autosomal shared-locus counts changed")
    if (
        hashlib.sha256(_canonical_json(autosomal_standard)).hexdigest()
        != EXPECTED_AUTOSOMAL_STANDARD_LOCUS_SHA256
        or hashlib.sha256(_canonical_json(autosomal_strict)).hexdigest()
        != EXPECTED_AUTOSOMAL_STRICT_LOCUS_SHA256
    ):
        raise AssertionError("NBR ordered autosomal locus identities changed")
    audit = dict(audit)
    audit.update({
        "primary_panel": "187 autosomal standard-contract loci",
        "primary_autosomal_loci": len(autosomal_standard),
        "primary_autosomal_ordered_locus_sha256": EXPECTED_AUTOSOMAL_STANDARD_LOCUS_SHA256,
        "published_representation_sensitivity_loci": len(standard),
        "autosomal_strict_sensitivity_loci": len(autosomal_strict),
        "autosomal_strict_ordered_locus_sha256": EXPECTED_AUTOSOMAL_STRICT_LOCUS_SHA256,
        "x_linked_loci_excluded_from_primary": list(EXPECTED_X_LOCI),
        "called_genotype_rate_contract": call_rate_audit,
    })
    primary = dict(panel)
    published_panel = dict(panel)
    published_panel["contract_role"] = "published_representation_sensitivity"
    strict_panel = dict(panel)
    strict_panel["contract_role"] = "strong_ascertainment_locus_filter_sensitivity"
    strict_panel["selection_guardrail"] = (
        "strongly ascertained within-every-population-polymorphic subset; correlated sensitivity only"
    )
    records = [
        _panel_record(
            primary, locus_names, autosomal_standard,
            filter_name="autosomal_187_standard_contract", strict=False,
            compute_state=compute_state,
        ),
        _panel_record(
            published_panel, locus_names, standard,
            filter_name="all_195_published_representation_sensitivity", strict=False,
            compute_state=compute_state,
        ),
        _panel_record(
            strict_panel, locus_names, autosomal_strict,
            filter_name="autosomal_148_within_population_polymorphic_strong_ascertainment",
            strict=True,
            compute_state=compute_state,
        ),
    ]
    return records, audit


def analyze_records(
    records: Sequence[dict],
    canonical_root: Path,
    *,
    compute_state: Path | None,
) -> dict:
    if compute_state is not None:
        structured.compute_gate(compute_state)
    canonical = structured.load_canonical(canonical_root, max_depth=16)
    if canonical["audit"]["array_contracts"] != brook.CANONICAL_ARRAY_CONTRACTS:
        raise RuntimeError("canonical training array contract changed")
    labels = np.asarray(canonical["labels"])
    rates = np.asarray(canonical["rates"], dtype=float)
    table = np.asarray(canonical["table"], dtype=float)
    positive = np.isin(labels, ["A", "B", "C"])
    counts = dict(zip(*np.unique(labels, return_counts=True)))
    if counts != {"A": 900, "B": 900, "C": 900, "D": 500}:
        raise RuntimeError("canonical direction class counts changed")
    external = np.stack([record["curve"] for record in records]).astype(float)

    representations = {}
    direction_models = {}
    for name in brook.REPRESENTATIONS:
        train = structured.representation_features(table[positive], name)
        target = structured.representation_features(external, name)
        scaler, model = structured._fit_model(train, labels[positive], C=1.0)
        z = scaler.transform(target)
        probabilities = model.predict_proba(z)
        calls = model.classes_[np.argmax(probabilities, axis=1)].astype(str)
        class_index = {str(label): index for index, label in enumerate(model.classes_)}
        for index, record in enumerate(records):
            record.setdefault("direction", {})[name] = {
                "call": str(calls[index]),
                "candidate_C_concordant": bool(calls[index] == EXPECTED_DIRECTION),
                "scores": {
                    label: float(probabilities[index, position])
                    for label, position in class_index.items()
                },
                "score_interpretation": "uncalibrated OOD score; not a posterior probability",
                "feature_shift": brook._z_audit(z[index]),
            }
        primary = [record for record in records if record["contract_role"] == "primary"]
        sensitivity = [record for record in records if record["contract_role"] != "primary"]
        representations[name] = {
            "status": "target-blind fixed canonical C=1; raw_all is primary",
            "feature_dimension": int(train.shape[1]),
            "management_candidate_concordance": brook._candidate_concordance(primary, name),
            "correlated_locus_sensitivity": brook._candidate_concordance(sensitivity, name),
            "external_rms_z": [record["direction"][name]["feature_shift"]["rms_z"] for record in records],
        }
        direction_models[name] = stdbench._model_payload(
            scaler, model, feature_columns=structured.representation_columns(name)
        )

    gate_train, gate_contract = brook.depth_matched_gate_features(table)
    gate_external, external_contract = brook.depth_matched_gate_features(external)
    if gate_contract != external_contract:
        raise AssertionError("canonical/NBR depth-matched gate contracts differ")
    gate_target = (positive & (rates >= structured.APPRECIABLE)).astype(int)
    gate_scaler, gate_model = structured._fit_model(gate_train, gate_target, C=1.0)
    gate_z = gate_scaler.transform(gate_external)
    positive_index = int(np.flatnonzero(gate_model.classes_ == 1)[0])
    gate_scores = gate_model.predict_proba(gate_z)[:, positive_index]
    for index, record in enumerate(records):
        record["depth_matched_gate"] = {
            "appreciable_score": float(gate_scores[index]),
            "called_at_0_5": bool(gate_scores[index] >= 0.5),
            "score_interpretation": (
                "uncalibrated g=2..16 OOD score; not the frozen full-depth gate or a posterior"
            ),
            "feature_shift": brook._z_audit(gate_z[index]),
        }
        raw_rms = record["direction"]["raw_all"]["feature_shift"]["rms_z"]
        gate_rms = record["depth_matched_gate"]["feature_shift"]["rms_z"]
        record["adjudication"] = {
            "direction_call_accepted": False,
            "gate_call_accepted": False,
            "formal_direction_accuracy_eligible": False,
            "formal_gate_accuracy_eligible": False,
            "gate_accuracy_eligible": False,
            "direction_accuracy_estimate": None,
            "gate_accuracy_estimate": None,
            "severe_OOD_heuristic": bool(max(raw_rms, gate_rms) > 10),
            "severe_OOD_threshold": "max(raw direction RMS-z, depth-matched gate RMS-z) > 10",
            "decision_basis": (
                "overlapping-locus pedigree labels, related hybrid descendants, and target-model "
                "mismatch prevent acceptance regardless of management-arrow concordance"
            ),
        }

    ledger = [{
        "panel_view_id": record["panel_view_id"],
        "contract_role": record["contract_role"],
        "raw_all_call": record["direction"]["raw_all"]["call"],
        "raw_all_C_score": record["direction"]["raw_all"]["scores"]["C"],
        "raw_all_rms_z": record["direction"]["raw_all"]["feature_shift"]["rms_z"],
        "depth_matched_gate_score": record["depth_matched_gate"]["appreciable_score"],
        "depth_matched_gate_rms_z": record["depth_matched_gate"]["feature_shift"]["rms_z"],
        "direction_call_accepted": False,
        "gate_call_accepted": False,
        "formal_direction_accuracy_eligible": False,
        "formal_gate_accuracy_eligible": False,
    } for record in records]
    for record in records:
        record["curve"] = np.asarray(record["curve"], dtype=float).tolist()
    return {
        "records": list(records),
        "prediction_ledger": ledger,
        "representations": representations,
        "direction_models": direction_models,
        "depth_matched_gate": {
            "contract": gate_contract,
            "training_target": "canonical A/B/C rate >=2.5e-4 versus weak A/B/C plus D",
            "model": stdbench._model_payload(gate_scaler, gate_model),
            "guardrail": "no NBR row has independent gate truth",
        },
        "canonical_source_audit": canonical["audit"],
        "descriptive_panel_accounting": {
            "management_interventions": 1,
            "primary_views": 1,
            "correlated_locus_sensitivities": 2,
            "independent_direction_truth_units": 0,
            "independent_gate_truth_units": 0,
            "guardrail": "one rescue system, not 188 hybrid trials or three independent panels",
        },
        "guardrail": (
            "candidate-direction concordance is not accuracy because pedigree membership was "
            "updated with overlapping microsatellites and the sampled descendants are related"
        ),
    }


def require_azure_execution_target(
    compute_target: str,
    *,
    os_name: str | None = None,
    hostname: str | None = None,
) -> dict:
    return brook.require_azure_execution_target(
        compute_target, os_name=os_name, hostname=hostname
    )


def configuration(revision: dict) -> dict:
    helper_sources = {}
    for name, module in {
        "brook_multiallelic": brook,
        "canonical_pipeline": structured,
        "stdpopsim_model_payload": stdbench,
    }.items():
        path = Path(module.__file__).resolve()
        helper_sources[name] = {"path": str(path), "sha256": structured.sha256_file(path)}
    return {
        "schema_version": SCHEMA_VERSION,
        "source_revision": {
            key: revision.get(key)
            for key in (
                "commit", "script_sha256", "head_script_sha256", "head_blob_oid",
                "worktree_blob_oid", "tracked_diff_sha256", "tracked_dirty_at_snapshot",
            )
        },
        "helper_sources": helper_sources,
        "sources": SOURCE_CONTRACTS,
        "panel": {
            "P1": "ParentalPop1",
            "P2": "Hybrid",
            "P3": "ParentalPop2",
            "expected_candidate_direction": EXPECTED_DIRECTION,
            "group_counts": GROUP_COUNTS,
            "management_intervention_direction_independent_of_loci": True,
            "pedigree_membership_independent_of_scored_loci": False,
            "operational_tree_only": True,
        },
        "locus_contract": {
            "primary_autosomal_standard_loci": 187,
            "published_representation_sensitivity_loci": 195,
            "autosomal_strict_sensitivity_loci": 148,
            "all_loci_strict_contract_count": 156,
            "minimum_called_copies_per_population": brook.MIN_CALLED_COPIES,
            "minimum_called_genotype_fraction_per_population": MIN_CALLED_GENOTYPE_FRACTION,
            "required_called_individuals": {"P1": 10, "P2": 141, "P3": 14},
            "primary_autosomal_ordered_locus_sha256": EXPECTED_AUTOSOMAL_STANDARD_LOCUS_SHA256,
            "all_standard_ordered_locus_sha256": EXPECTED_ALL_STANDARD_LOCUS_SHA256,
            "autosomal_strict_ordered_locus_sha256": EXPECTED_AUTOSOMAL_STRICT_LOCUS_SHA256,
            "all_strict_ordered_locus_sha256": EXPECTED_ALL_STRICT_LOCUS_SHA256,
            "x_linked_loci": list(EXPECTED_X_LOCI),
            "x_linked_primary_exclusion": (
                "male calls were forced homozygous; retain only in the published-representation sensitivity"
            ),
            "all_microsatellite_alleles_retained": True,
        },
        "padze": {
            "depths": brook.PRIMARY_DEPTHS.tolist(),
            "moments": list(stdbench.MOMENTS),
            "pihat_sizes": [2],
            "bias_corrected": True,
        },
        "evaluation": {
            "representations": list(brook.REPRESENTATIONS),
            "primary_representation": "raw_all",
            "C": 1.0,
            "formal_direction_accuracy_eligible": False,
            "formal_gate_accuracy_eligible": False,
            "gate_accuracy_eligible": False,
            "direction_calls_accepted": False,
            "gate_calls_accepted": False,
            "direction_accuracy_estimate": None,
            "gate_accuracy_estimate": None,
            "independent_direction_truth_units": 0,
            "independent_gate_truth_units": 0,
            "modeling_table_used_for_scoring": False,
            "modeling_table_used_for_truth": False,
        },
        "canonical_training_contract": {
            "replicates": 3_200,
            "label_counts": {"A": 900, "B": 900, "C": 900, "D": 500},
            "array_contracts": brook.CANONICAL_ARRAY_CONTRACTS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--genotypes", type=Path, required=True)
    parser.add_argument("--modeling-table", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--compute-state", type=Path, default=structured.DEFAULT_COMPUTE_STATE)
    parser.add_argument("--compute-target", choices=("local", "azure"), default="local")
    parser.add_argument("--allow-stopped-trading-compute", action="store_true")
    parser.add_argument("--allow-closing-owner-session", action="store_true")
    args = parser.parse_args()
    try:
        execution_target = require_azure_execution_target(args.compute_target)
    except RuntimeError as exc:
        parser.error(str(exc))

    os.environ[structured.COMPUTE_TARGET_ENV] = args.compute_target
    if args.allow_stopped_trading_compute:
        os.environ[structured.STOPPED_TRADING_AUTH_ENV] = "1"
    if args.allow_closing_owner_session:
        os.environ[structured.AZURE_CLOSING_OWNER_AUTH_ENV] = "1"
    initial_gate = structured.compute_gate(args.compute_state)
    priority = structured.set_below_normal_priority()
    revision = structured.git_revision(script=Path(__file__))
    structured.require_clean_tracked_revision(revision)

    source_audits = {
        "genotypes": verify_source(args.genotypes, "genotypes"),
        "modeling": verify_source(args.modeling_table, "modeling"),
    }
    locus_names, autosomal_loci, groups, genotype_audit = parse_genotypes(args.genotypes)
    modeling_audit = parse_modeling_table(args.modeling_table, locus_names)
    panel = make_panel(groups)
    config = configuration(revision)
    config_sha256 = hashlib.sha256(_canonical_json(config)).hexdigest()
    records, locus_audit = build_records(
        locus_names, autosomal_loci, panel, compute_state=args.compute_state
    )
    pre_analysis_gate = structured.compute_gate(args.compute_state)
    analysis = analyze_records(records, args.canonical_root, compute_state=args.compute_state)

    final_revision = structured.git_revision(script=Path(__file__))
    structured.require_revision_unchanged(revision, final_revision)
    final_sources = {
        "genotypes": verify_source(args.genotypes, "genotypes"),
        "modeling": verify_source(args.modeling_table, "modeling"),
    }
    runtime = structured.runtime_audit(priority)
    runtime["packages"]["padze"] = importlib_metadata.version("padze")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "management_intervention_direction_transfer_diagnostic_not_accuracy",
        "git": revision,
        "final_git_recheck": final_revision,
        "initial_compute_gate": initial_gate,
        "pre_analysis_compute_gate": pre_analysis_gate,
        "runtime": runtime,
        "execution_target": execution_target,
        "configuration": config,
        "configuration_sha256": config_sha256,
        "source_audits": source_audits,
        "source_final_recheck": final_sources,
        "parser_audits": {
            "genotypes": genotype_audit,
            "modeling_table": modeling_audit,
        },
        "selection_audits": {"loci": locus_audit},
        "analysis": analysis,
    }
    with structured.SingleWriterLease(
        args.result_dir, ".bighorn_nbr_rescue_result.lock"
    ):
        output = args.result_dir / "results.json"
        output_audit = structured.write_json_atomic(output, result, indent=2)
    primary = analysis["representations"]["raw_all"]["management_candidate_concordance"]
    print(json.dumps({
        "output": output_audit,
        "configuration_sha256": config_sha256,
        "raw_all_management_candidate_C_calls": primary["C_calls"],
        "raw_all_primary_views": primary["views"],
        "formal_direction_accuracy_eligible": False,
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
