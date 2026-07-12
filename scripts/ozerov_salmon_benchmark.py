#!/usr/bin/env python3
"""Score the Ozerov et al. Baltic-salmon stocking panel as a transfer diagnostic.

The individual workbook supplies 17 multiallelic microsatellites.  The primary
operational mapping is Loobu before large-scale releases (P1), 2007-08 Loobu (P2), and the
pooled 1998-2009 Narva hatchery samples used by the paper's microsatellite
estimators (P3).  A Narva-2006 donor view aligns the individual data with the
paper's pooled-SNP estimator, and a 16-locus within-population-polymorphic view
is a correlated ascertainment sensitivity.

The documented Narva-to-Loobu stocking history motivates DNNaic candidate C,
but neither the management arrow nor the paper's same-data ancestry estimates
provide an independent DNNaic direction or gate truth label.  Every output is
therefore descriptive and unaccepted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dnnaic.semantics import class_for_forward_edge
from scripts import brook_trout_microsatellite_benchmark as brook
from scripts import oyster_2017_external_benchmark as oyster
from scripts import stdpopsim_neanderthal_benchmark as stdbench
from scripts import structured_transfer_pilot as structured


SCHEMA_VERSION = "dnnaic-ozerov-salmon-benchmark-v1"
DEFAULT_RESULT_DIR = REPO / "results" / "ozerov_salmon_benchmark_2026_07_12"
EXPECTED_DIRECTION = class_for_forward_edge("P3", "P2")
REPRESENTATIONS = brook.REPRESENTATIONS

SOURCE_CONTRACTS = {
    "microsatellite": {
        "file": "Ozerov_et_al_2016_Individual_microsatellite_genotypes.xlsx",
        "bytes": 345_280,
        "md5": "32c53c1f4f700a5c3d2d5b50a461b381",
        "sha256": "6483ccf6771fa4d0d9452cff5efb75a9cb2852cf6162268572fea6c3c2b5418d",
        "dryad_file_id": 68_891,
        "license": "CC0-1.0",
        "record": "https://doi.org/10.5061/dryad.p00gd",
        "version": "https://datadryad.org/stash/dataset/doi:10.5061/dryad.p00gd",
        "version_id": 20_386,
        "download": "https://datadryad.org/api/v2/files/68891/download",
        "data_doi": "10.5061/dryad.p00gd",
        "paper_doi": "10.1111/mec.13570",
    },
    "pooled_snp": {
        "file": "Ozerov_et_al_2016_estimated_SNP_allele_frequencies.xlsx",
        "bytes": 629_100,
        "md5": "c7dfb0b8f0191ace362d22fc5a5155e4",
        "sha256": "6d1325739bc56175b4597ee075edc2795fe89e169e737d5afe0a9b175495b3f6",
        "dryad_file_id": 68_892,
        "license": "CC0-1.0",
        "record": "https://doi.org/10.5061/dryad.p00gd",
        "version": "https://datadryad.org/stash/dataset/doi:10.5061/dryad.p00gd",
        "version_id": 20_386,
        "download": "https://datadryad.org/api/v2/files/68892/download",
        "data_doi": "10.5061/dryad.p00gd",
        "paper_doi": "10.1111/mec.13570",
    },
}

MICROSATELLITE_LOCI = (
    "SSsp2210",
    "SSsp2216",
    "SsaD157",
    "Ssa407",
    "SSspG7",
    "SSsp3016",
    "SSsp2201",
    "Ssa14",
    "SSsp1605",
    "SSOSL85",
    "SSOSL438",
    "Ssa197",
    "Ssa289",
    "Ssa85",
    "Ssa171",
    "SSOSL417",
    "Ssa202",
)
STRICT_MICROSATELLITE_LOCI = tuple(
    locus for locus in MICROSATELLITE_LOCI if locus != "Ssa14"
)
SNP_HEADERS = (
    "Vas96-99",
    "Vas07-08",
    "Vas09-10",
    "Vas11-12",
    "Kei96-97",
    "Kei07-08",
    "Kei09-10",
    "Kei11-12",
    "Loo96-99",
    "Loo07-08",
    "Loo09-10",
    "Loo11-12",
    "Kun96-97",
    "Kun07-08",
    "Kun09-10",
    "Kun11-12",
    "Nev96-97",
    "Nar06",
)
GROUP_COUNTS = {
    "Kei9697a": 54,
    "Kei0708a": 67,
    "Kei0910a": 98,
    "Kei1112a": 95,
    "Vas9699a": 45,
    "Vas0708a": 30,
    "Vas0910a": 97,
    "Vas1112a": 67,
    "Kun9697a": 71,
    "Kun0708a": 57,
    "Kun0910a": 70,
    "Kun1112a": 120,
    "Loobu9699a": 81,
    "Loobu0708a": 77,
    "Loobu0910a": 102,
    "Loobu1112a": 104,
    "Narva98a": 45,
    "Narva01a": 73,
    "Narva04a": 129,
    "Narva05a": 77,
    "Narva06a": 112,
    "Narva07a": 80,
    "Narva08a": 95,
    "Narva09a": 109,
    "Neva9798a": 97,
}
NARVA_MICROSATELLITE_GROUPS = (
    "Narva98a",
    "Narva01a",
    "Narva04a",
    "Narva05a",
    "Narva06a",
    "Narva07a",
    "Narva08a",
    "Narva09a",
)
EXPECTED_MISSING_BY_LOCUS = {
    "SSsp2210": 0,
    "SSsp2216": 6,
    "SsaD157": 61,
    "Ssa407": 11,
    "SSspG7": 2,
    "SSsp3016": 0,
    "SSsp2201": 123,
    "Ssa14": 69,
    "SSsp1605": 126,
    "SSOSL85": 21,
    "SSOSL438": 3,
    "Ssa197": 3,
    "Ssa289": 39,
    "Ssa85": 0,
    "Ssa171": 24,
    "SSOSL417": 27,
    "Ssa202": 50,
}
EXPECTED_MISSING_BY_POPULATION = {
    "Kei9697a": 0,
    "Kei0708a": 39,
    "Kei0910a": 7,
    "Kei1112a": 67,
    "Vas9699a": 0,
    "Vas0708a": 2,
    "Vas0910a": 1,
    "Vas1112a": 41,
    "Kun9697a": 6,
    "Kun0708a": 2,
    "Kun0910a": 0,
    "Kun1112a": 167,
    "Loobu9699a": 2,
    "Loobu0708a": 5,
    "Loobu0910a": 0,
    "Loobu1112a": 0,
    "Narva98a": 10,
    "Narva01a": 8,
    "Narva04a": 138,
    "Narva05a": 14,
    "Narva06a": 6,
    "Narva07a": 28,
    "Narva08a": 10,
    "Narva09a": 0,
    "Neva9798a": 12,
}
EXPECTED_MISSING_PER_INDIVIDUAL = {
    0: 1_842,
    1: 79,
    2: 31,
    3: 36,
    4: 32,
    5: 15,
    6: 9,
    7: 5,
    8: 3,
}

EXPECTED_LOCUS_SHA256 = "3bebd24695b4936df8e0c4ec526de80a7deb4e0b67910c1bec64510751c5c4a1"
EXPECTED_LOCUS_NEWLINE_SHA256 = "904e10f4c75adc35e8e1c9265876aee147c8aee0508726e542168b1ebdb2863f"
EXPECTED_STRICT_LOCUS_SHA256 = "9d81b14188c78ef4ef170dcacebbed0faef284ca31b5f087789c10a3bd61932e"
EXPECTED_SAMPLE_LEDGER_SHA256 = "379a3a47136bd2564671947ff91db08cedcf9b2ddfbccf10b11d917a025a2767"
EXPECTED_GROUP_COUNT_SHA256 = "0668c7328fef59c6a0ab33d3073fe557d8fe14e9a588b9613455ab4067370fb9"
EXPECTED_SNP_HEADER_SHA256 = "691222805a1b0b6df7552df5c492adebafb5a268b485ee0bab1632cb2c9c8f52"
EXPECTED_SNP_RECORD_LEDGER_SHA256 = "e6f380402be791d29aafd5f1f0a78b554eeedc00f785b531a409a8f682dbdaae"
EXPECTED_LOCUS_SELECTION_LEDGER_SHA256 = "b4ee09259a00ff534929de85a46a040ed767e7fd6c8de1fd294a78d9b3f0ab4c"

PUBLISHED_COMPARATOR = {
    "paper_doi": "10.1111/mec.13570",
    "citation": "Ozerov et al. (2016), Molecular Ecology 25:1275-1293",
    "locator": "Table 2, journal page 1285, From Narva to Loobu, recipient 2007-08",
    "checked_pdf": "https://media.rmk.ee/files/Ozerov_et_al-2016-Molecular_Ecology.pdf",
    "independently_checked_on": "2026-07-12",
    "candidate_arrow": "Narva hatchery source -> Loobu recipient",
    "microsatellite_Q": {
        "estimate": 0.606,
        "ci95": [0.542, 0.671],
        "markers": "17 individual microsatellites",
        "P1": "Loobu 1996-99, n=81",
        "P2": "Loobu 2007-08, n=77",
        "P3": "pooled Narva 1998-2009, n=720",
    },
    "microsatellite_I": {
        "estimate": 0.616,
        "ci95": [0.552, 0.680],
        "markers": "17 individual microsatellites",
        "P1": "Loobu 1996-99, n=81",
        "P2": "Loobu 2007-08, n=77",
        "P3": "pooled Narva 1998-2009, n=720",
    },
    "pooled_snp_S_hat": {
        "estimate": 0.567,
        "ci95": [0.533, 0.601],
        "markers": "1,986 pooled-DNA SNP frequencies",
        "P1": "Loo96-99",
        "P2": "Loo07-08",
        "P3": "Nar06, source pool n=42 in Table 1",
    },
    "estimator_guardrail": (
        "Q and I are distinct 17-microsatellite estimators; S-hat is the pooled-SNP estimator. "
        "They are published ancestry estimates, not independent DNNaic direction or gate labels."
    ),
    "stocking_timeline_guardrail": (
        "The paper describes regular large-scale Loobu releases beginning in 2002 and also "
        "discusses direct stocking in 1996-97; describe P1 as before large-scale releases, "
        "not as never previously stocked."
    ),
}


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _ordered_newline_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source(path: Path, contract_name: str) -> dict:
    contract = SOURCE_CONTRACTS[contract_name]
    if not path.is_file():
        raise FileNotFoundError(path)
    audit = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "md5": _md5_file(path),
        "sha256": structured.sha256_file(path),
    }
    if any(audit[key] != contract[key] for key in ("bytes", "md5", "sha256")):
        raise RuntimeError(f"{contract_name} source bytes do not match the pinned contract")
    return {**contract, **audit}


def _exact_int(value: str, context: str) -> int:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{context}: expected an integer, got {value!r}") from exc
    integer = int(number)
    if not math.isfinite(number) or number != integer:
        raise ValueError(f"{context}: expected an exact integer, got {value!r}")
    return integer


def _map_position(value: str, context: str) -> float | None:
    if value.strip().lower() == "n/a":
        return None
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{context}: invalid map position {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context}: nonfinite map position")
    return number


def parse_microsatellite_workbook(
    path: Path,
) -> tuple[tuple[str, ...], list[brook.Individual], dict]:
    sheet = oyster.read_xlsx_sheet(path, "Pooled")
    if sheet["dimension"] != "A1:AI2056" or (
        sheet["n_rows"], sheet["n_columns"]
    ) != (2_056, 35):
        raise AssertionError("Ozerov microsatellite worksheet shape changed")
    if sheet["merged_cell_ranges"] != 0:
        raise AssertionError("Ozerov microsatellite worksheet gained merged cells")
    cells = sheet["cells"]
    loci = tuple(cells.get((3, column), "").strip() for column in range(1, 35, 2))
    if loci != MICROSATELLITE_LOCI:
        raise AssertionError(f"Ozerov microsatellite locus order changed: {loci}")
    expected_header = ["Sample"]
    for locus in MICROSATELLITE_LOCI:
        expected_header.extend((locus, ""))
    observed_header = [cells.get((3, column), "").strip() for column in range(35)]
    if observed_header != expected_header:
        raise AssertionError("Ozerov microsatellite two-column header contract changed")

    individuals: list[brook.Individual] = []
    seen_ids: set[str] = set()
    group_counts: Counter[str] = Counter()
    missing_by_locus: Counter[str] = Counter()
    missing_by_population: Counter[str] = Counter()
    missing_per_individual: Counter[int] = Counter()
    sample_ledger = hashlib.sha256()
    called_pairs = 0
    missing_pairs = 0
    for row in range(4, sheet["n_rows"]):
        sample_id = cells.get((row, 0), "").strip()
        match = re.fullmatch(r"(.+a)_(\d+)", sample_id)
        if match is None:
            raise ValueError(f"microsatellite row {row + 1}: malformed sample ID {sample_id!r}")
        population = match.group(1)
        if population not in GROUP_COUNTS:
            raise ValueError(f"microsatellite row {row + 1}: unknown population {population!r}")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate microsatellite sample ID: {sample_id}")
        seen_ids.add(sample_id)
        pairs = []
        individual_missing = 0
        for locus_index, locus in enumerate(MICROSATELLITE_LOCI):
            first = _exact_int(
                cells.get((row, 1 + 2 * locus_index), ""),
                f"microsatellite row {row + 1} {locus} first allele",
            )
            second = _exact_int(
                cells.get((row, 2 + 2 * locus_index), ""),
                f"microsatellite row {row + 1} {locus} second allele",
            )
            if first < 0 or second < 0:
                raise ValueError(f"microsatellite row {row + 1} {locus}: negative allele")
            if (first == 0) != (second == 0):
                raise ValueError(f"microsatellite row {row + 1} {locus}: partial missing call")
            if first > second:
                raise ValueError(f"microsatellite row {row + 1} {locus}: descending allele pair")
            missing = first == 0
            missing_pairs += int(missing)
            called_pairs += int(not missing)
            individual_missing += int(missing)
            missing_by_locus[locus] += int(missing)
            pairs.append((first, second))
        individual = brook.Individual(
            sample_id=sample_id,
            population=population,
            alleles=tuple(pairs),
            source_ordinal=row + 1,
        )
        individuals.append(individual)
        group_counts[population] += 1
        missing_by_population[population] += individual_missing
        missing_per_individual[individual_missing] += 1
        sample_ledger.update(_canonical_json({
            "sample_id": sample_id,
            "population": population,
            "alleles": pairs,
            "source_ordinal": row + 1,
        }))
        sample_ledger.update(b"\n")

    observed_groups = dict(group_counts)
    observed_missing = {locus: missing_by_locus[locus] for locus in MICROSATELLITE_LOCI}
    observed_population_missing = {
        population: missing_by_population[population] for population in GROUP_COUNTS
    }
    if observed_groups != GROUP_COUNTS or sum(observed_groups.values()) != 2_052:
        raise AssertionError("Ozerov microsatellite population counts changed")
    if observed_missing != EXPECTED_MISSING_BY_LOCUS:
        raise AssertionError("Ozerov microsatellite missing-call distribution changed")
    if observed_population_missing != EXPECTED_MISSING_BY_POPULATION:
        raise AssertionError("Ozerov population missing-call distribution changed")
    if dict(missing_per_individual) != EXPECTED_MISSING_PER_INDIVIDUAL:
        raise AssertionError("Ozerov per-individual missing-call distribution changed")
    if (called_pairs, missing_pairs) != (34_319, 565):
        raise AssertionError("Ozerov microsatellite called/missing pair counts changed")
    locus_sha = hashlib.sha256(_canonical_json(list(loci))).hexdigest()
    group_sha = hashlib.sha256(_canonical_json(observed_groups)).hexdigest()
    ledger_sha = sample_ledger.hexdigest()
    if locus_sha != EXPECTED_LOCUS_SHA256:
        raise AssertionError("Ozerov normalized locus identity changed")
    if _ordered_newline_sha256(loci) != EXPECTED_LOCUS_NEWLINE_SHA256:
        raise AssertionError("Ozerov newline locus identity changed")
    if group_sha != EXPECTED_GROUP_COUNT_SHA256 or ledger_sha != EXPECTED_SAMPLE_LEDGER_SHA256:
        raise AssertionError("Ozerov normalized sample ledger changed")
    return loci, individuals, {
        "sheet_name": "Pooled",
        "dimension": sheet["dimension"],
        "individuals": len(individuals),
        "unique_sample_ids": len(seen_ids),
        "loci": len(loci),
        "diploid_genotype_pairs": called_pairs + missing_pairs,
        "called_pairs": called_pairs,
        "missing_pairs": missing_pairs,
        "missing_pair_fraction": float(missing_pairs / (called_pairs + missing_pairs)),
        "individuals_with_missing_calls": int(len(individuals) - missing_per_individual[0]),
        "missing_pairs_by_locus": observed_missing,
        "missing_pairs_by_population": observed_population_missing,
        "missing_pairs_per_individual_distribution": {
            str(key): value for key, value in sorted(missing_per_individual.items())
        },
        "all_nonmissing_pairs_nondecreasing": True,
        "partial_missing_pairs": 0,
        "population_counts": observed_groups,
        "ordered_locus_sha256": locus_sha,
        "ordered_locus_newline_sha256": EXPECTED_LOCUS_NEWLINE_SHA256,
        "population_count_sha256": group_sha,
        "normalized_sample_ledger_sha256": ledger_sha,
    }


def parse_pooled_snp_workbook(path: Path) -> tuple[tuple[str, ...], list[dict], dict]:
    sheet = oyster.read_xlsx_sheet(path, "Sheet1")
    if sheet["dimension"] != "A1:V1990" or (
        sheet["n_rows"], sheet["n_columns"]
    ) != (1_990, 22):
        raise AssertionError("Ozerov pooled-SNP worksheet shape changed")
    if sheet["merged_cell_ranges"] != 0:
        raise AssertionError("Ozerov pooled-SNP worksheet gained merged cells")
    cells = sheet["cells"]
    headers = tuple(cells.get((3, column), "").strip() for column in range(4, 22))
    if headers != SNP_HEADERS:
        raise AssertionError(f"Ozerov pooled-SNP header order changed: {headers}")
    header_sha = hashlib.sha256(_canonical_json(list(headers))).hexdigest()
    if header_sha != EXPECTED_SNP_HEADER_SHA256:
        raise AssertionError("Ozerov normalized pooled-SNP headers changed")

    records = []
    ledger = hashlib.sha256()
    seen_names: set[str] = set()
    null_female = 0
    null_male = 0
    null_chromosome = 0
    null_metadata_triplets = 0
    frequency_count = 0
    for row in range(4, sheet["n_rows"]):
        snp = cells.get((row, 0), "").strip()
        chromosome = cells.get((row, 1), "").strip()
        if not snp or snp in seen_names:
            raise ValueError(f"pooled-SNP row {row + 1}: absent or duplicate SNP name")
        if not chromosome:
            raise ValueError(f"pooled-SNP row {row + 1}: absent chromosome field")
        seen_names.add(snp)
        female = _map_position(
            cells.get((row, 2), "").strip(), f"pooled-SNP row {row + 1} female map"
        )
        male = _map_position(
            cells.get((row, 3), "").strip(), f"pooled-SNP row {row + 1} male map"
        )
        frequencies = {}
        for column, header in enumerate(headers, start=4):
            raw = cells.get((row, column), "").strip()
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(
                    f"pooled-SNP row {row + 1} {header}: invalid frequency {raw!r}"
                ) from exc
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"pooled-SNP row {row + 1} {header}: frequency outside [0,1]")
            frequencies[header] = value
            frequency_count += 1
        record = {
            "snp": snp,
            "chromosome": chromosome,
            "female_cM": female,
            "male_cM": male,
            "frequencies": frequencies,
        }
        records.append(record)
        ledger.update(_canonical_json(record))
        ledger.update(b"\n")
        null_flags = (
            chromosome.lower() == "n/a",
            female is None,
            male is None,
        )
        if len(set(null_flags)) != 1:
            raise ValueError(
                f"pooled-SNP row {row + 1}: chromosome/female/male null fields do not coincide"
            )
        null_chromosome += int(null_flags[0])
        null_female += int(null_flags[1])
        null_male += int(null_flags[2])
        null_metadata_triplets += int(all(null_flags))

    if len(records) != 1_986 or frequency_count != 35_748:
        raise AssertionError("Ozerov pooled-SNP row/frequency counts changed")
    if (null_chromosome, null_female, null_male, null_metadata_triplets) != (
        64, 64, 64, 64
    ):
        raise AssertionError("Ozerov pooled-SNP map-missingness changed")
    ledger_sha = ledger.hexdigest()
    if ledger_sha != EXPECTED_SNP_RECORD_LEDGER_SHA256:
        raise AssertionError("Ozerov normalized pooled-SNP ledger changed")
    return headers, records, {
        "sheet_name": "Sheet1",
        "dimension": sheet["dimension"],
        "snps": len(records),
        "unique_snp_names": len(seen_names),
        "population_frequency_columns": len(headers),
        "frequency_values": frequency_count,
        "all_frequencies_finite_in_unit_interval": True,
        "null_chromosome_rows": null_chromosome,
        "null_female_map_positions": null_female,
        "null_male_map_positions": null_male,
        "null_chromosome_female_male_triplets": null_metadata_triplets,
        "ordered_population_header_sha256": header_sha,
        "normalized_snp_record_ledger_sha256": ledger_sha,
        "raw_xml_precision_guardrail": (
            "the OOXML raw numeric values are used; display format 0.000 is not rounded into data"
        ),
    }


def make_panels(individuals: Sequence[brook.Individual]) -> list[dict]:
    by_population = {
        name: [individual for individual in individuals if individual.population == name]
        for name in GROUP_COUNTS
    }
    observed = {name: len(group) for name, group in by_population.items()}
    if observed != GROUP_COUNTS:
        raise AssertionError("Ozerov panel population counts changed")
    narva_pooled = [
        individual
        for individual in individuals
        if individual.population in NARVA_MICROSATELLITE_GROUPS
    ]
    if len(narva_pooled) != 720:
        raise AssertionError("pooled Narva microsatellite reference count changed")
    common = {
        "dataset": "ozerov_2016_baltic_salmon",
        "biological_system_id": "loobu_narva_stocking_1996_2009",
        "expected_candidate_direction": EXPECTED_DIRECTION,
        "candidate_direction_evidence": (
            "Table 2 explicitly evaluates Narva-to-Loobu stocking ancestry; management history "
            "therefore fixes candidate P3->P2 independently of the classifier call"
        ),
        "operational_tree_guardrail": (
            "P1/P2/P3 is an operational DNNaic mapping of a temporal stocking contrast, not proof "
            "of an exclusive pulse on the canonical three-population tree"
        ),
        "published_estimates_are_truth_labels": False,
        "management_arrow_is_gate_truth": False,
        "selection_guardrail": (
            "one Loobu-Narva system with correlated donor and locus views, not three independent trials"
        ),
    }
    primary = {
        **common,
        "panel_id": "loobu_0708_narva_9809_pooled",
        "contract_role": "primary",
        "groups": (
            by_population["Loobu9699a"],
            by_population["Loobu0708a"],
            narva_pooled,
        ),
        "population_semantics": {
            "P1": "Loobu 1996-99 before large-scale releases, n=81",
            "P2": "Loobu 2007-08 enhanced cohort, n=77",
            "P3": "pooled Narva 1998-2009 hatchery samples, n=720",
            "forward_candidate": "P3->P2",
        },
        "published_comparator_alignment": (
            "exact individual-microsatellite population alignment for Table 2 Q and I"
        ),
    }
    narva06 = {
        **common,
        "panel_id": "loobu_0708_narva_06_donor_sensitivity",
        "contract_role": "published_snp_donor_alignment_sensitivity",
        "groups": (
            by_population["Loobu9699a"],
            by_population["Loobu0708a"],
            by_population["Narva06a"],
        ),
        "population_semantics": {
            "P1": "Loobu 1996-99, n=81",
            "P2": "Loobu 2007-08, n=77",
            "P3": "Narva 2006 individual microsatellites, n=112",
            "forward_candidate": "P3->P2",
        },
        "published_comparator_alignment": (
            "donor-year sensitivity aligned to pooled-SNP S-hat; not the exact Q/I reference"
        ),
    }
    return [primary, narva06]


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
        source=f"Ozerov Baltic salmon panel {panel['panel_id']} filter {filter_name}",
        filters=[
            "at least 16 called gene copies in P1/P2/P3",
            "globally polymorphic in the fixed triplet; every microsatellite allele retained",
            "exact 0/0 source pairs retained as missing diploid calls",
            *(["polymorphic within every population"] if strict else []),
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
        "direction_accuracy_estimate": None,
        "gate_accuracy_estimate": None,
        "independent_direction_truth_units": 0,
        "independent_gate_truth_units": 0,
        "ground_truth_guardrail": (
            "stocking history fixes a candidate arrow, while published estimates reuse these "
            "populations and markers and do not establish exclusive topology or gate truth"
        ),
    })
    return output


def build_records(
    locus_names: Sequence[str],
    panels: Sequence[Mapping[str, object]],
    *,
    compute_state: Path | None,
) -> tuple[list[dict], dict]:
    if len(panels) != 2 or [panel["contract_role"] for panel in panels] != [
        "primary",
        "published_snp_donor_alignment_sensitivity",
    ]:
        raise ValueError("Ozerov panel contract requires primary and Narva06 sensitivity")
    standard, strict, audit = brook.eligible_shared_loci(locus_names, panels)
    if standard != list(MICROSATELLITE_LOCI) or strict != list(STRICT_MICROSATELLITE_LOCI):
        raise AssertionError("Ozerov standard/strict locus membership changed")
    if (
        audit["standard_ordered_locus_sha256"] != EXPECTED_LOCUS_SHA256
        or audit["strict_ordered_locus_sha256"] != EXPECTED_STRICT_LOCUS_SHA256
        or audit["selection_ledger_sha256"] != EXPECTED_LOCUS_SELECTION_LEDGER_SHA256
    ):
        raise AssertionError("Ozerov locus selection ledger changed")
    strict_panel = dict(panels[0])
    strict_panel["contract_role"] = "within_population_polymorphic_locus_sensitivity"
    strict_panel["selection_guardrail"] = (
        "same primary individuals with Ssa14 removed; correlated locus sensitivity only"
    )
    records = [
        _panel_record(
            panels[0],
            locus_names,
            standard,
            filter_name="all_17_standard_contract",
            strict=False,
            compute_state=compute_state,
        ),
        _panel_record(
            panels[1],
            locus_names,
            standard,
            filter_name="narva06_all_17_donor_alignment_sensitivity",
            strict=False,
            compute_state=compute_state,
        ),
        _panel_record(
            strict_panel,
            locus_names,
            strict,
            filter_name="pooled_narva_16_within_population_polymorphic",
            strict=True,
            compute_state=compute_state,
        ),
    ]
    audit = dict(audit)
    audit.update({
        "primary_standard_loci": len(standard),
        "correlated_strict_sensitivity_loci": len(strict),
        "strict_excluded_loci": [
            locus for locus in standard if locus not in set(strict)
        ],
        "record_views": len(records),
        "biological_systems": 1,
        "independent_direction_truth_units": 0,
        "independent_gate_truth_units": 0,
    })
    return records, audit


def pooled_snp_frequency_geometry(records: Sequence[Mapping[str, object]]) -> dict:
    projections = []
    numerators = []
    denominators = []
    f3_values = []
    zero_distance = 0
    ledger = []
    for record in records:
        frequencies = record["frequencies"]
        p1 = float(frequencies["Loo96-99"])
        p2 = float(frequencies["Loo07-08"])
        p3 = float(frequencies["Nar06"])
        donor_axis = p3 - p1
        denominator = donor_axis * donor_axis
        if denominator > 0:
            numerator = (p2 - p1) * donor_axis
            projection = numerator / denominator
            projections.append(projection)
            numerators.append(numerator)
            denominators.append(denominator)
        else:
            numerator = None
            projection = None
            zero_distance += 1
        # Sum over both biallelic frequency coordinates, A and 1-A.
        f3 = 2.0 * (p2 - p1) * (p2 - p3)
        f3_values.append(f3)
        ledger.append([record["snp"], numerator, denominator, projection, f3])
    if len(f3_values) != 1_986:
        raise AssertionError("Ozerov pooled-SNP geometry row count changed")
    return {
        "description": (
            "raw pooled-allele-frequency projection of Loo07-08 minus Loo96-99 onto Nar06 "
            "minus Loo96-99 and f3(P2;P1,P3); the unweighted mean projection is unstable "
            "under tiny P1-P3 denominators; descriptive, not the published S-hat estimator"
        ),
        "projection_snps": len(projections),
        "zero_P1_P3_distance_snps": zero_distance,
        "mean_projection": float(np.mean(projections)),
        "median_projection": float(np.median(projections)),
        "denominator_weighted_projection": float(sum(numerators) / sum(denominators)),
        "mean_f3": float(np.mean(f3_values)),
        "negative_f3_snps": int(sum(value < 0 for value in f3_values)),
        "snp_geometry_sha256": hashlib.sha256(_canonical_json(ledger)).hexdigest(),
        "published_comparison_guardrail": (
            "this direct geometry is not Q, I, or S-hat; compare magnitudes descriptively only"
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


def analyze_records(
    records: Sequence[dict],
    canonical_root: Path,
    *,
    pooled_snp_geometry: Mapping[str, object],
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
    observed_counts = {
        str(label): int(count)
        for label, count in zip(*np.unique(labels, return_counts=True))
    }
    if observed_counts != {"A": 900, "B": 900, "C": 900, "D": 500}:
        raise RuntimeError("canonical direction class counts changed")
    if len(records) != 3:
        raise ValueError("Ozerov analysis requires three correlated record views")
    external = np.stack([record["curve"] for record in records]).astype(float)

    representations = {}
    direction_models = {}
    for name in REPRESENTATIONS:
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
        sensitivities = [
            record for record in records if record["contract_role"] != "primary"
        ]
        if len(primary) != 1 or len(sensitivities) != 2:
            raise AssertionError("Ozerov primary/sensitivity accounting changed")
        representations[name] = {
            "status": "target-blind fixed canonical C=1; raw_all is primary",
            "feature_dimension": int(train.shape[1]),
            "management_candidate_concordance": brook._candidate_concordance(primary, name),
            "correlated_sensitivity_concordance": brook._candidate_concordance(
                sensitivities, name
            ),
            "primary_and_sensitivity_calls": [
                record["direction"][name]["call"] for record in records
            ],
            "external_rms_z": [
                record["direction"][name]["feature_shift"]["rms_z"]
                for record in records
            ],
        }
        direction_models[name] = stdbench._model_payload(
            scaler,
            model,
            feature_columns=structured.representation_columns(name),
        )

    gate_train, gate_contract = brook.depth_matched_gate_features(table)
    gate_external, external_contract = brook.depth_matched_gate_features(external)
    if gate_contract != external_contract:
        raise AssertionError("canonical/Ozerov depth-matched gate contracts differ")
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
            "independent_direction_truth_units": 0,
            "independent_gate_truth_units": 0,
            "severe_OOD_heuristic": bool(max(raw_rms, gate_rms) > 10),
            "severe_OOD_threshold": (
                "max(raw_all direction RMS-z, depth-matched gate RMS-z) > 10"
            ),
            "decision_basis": (
                "same-system temporal sampling, same-data ancestry estimates, continuous stocking, "
                "and unresolved exclusive topology prevent acceptance regardless of candidate concordance"
            ),
        }

    prediction_ledger = [
        {
            "panel_view_id": record["panel_view_id"],
            "contract_role": record["contract_role"],
            "expected_candidate_direction": record["expected_candidate_direction"],
            "raw_all_call": record["direction"]["raw_all"]["call"],
            "raw_all_C_score": record["direction"]["raw_all"]["scores"]["C"],
            "raw_all_rms_z": record["direction"]["raw_all"]["feature_shift"]["rms_z"],
            "depth_matched_gate_score": record["depth_matched_gate"]["appreciable_score"],
            "depth_matched_gate_rms_z": record["depth_matched_gate"]["feature_shift"]["rms_z"],
            "severe_OOD_heuristic": record["adjudication"]["severe_OOD_heuristic"],
            "direction_call_accepted": False,
            "gate_call_accepted": False,
            "formal_direction_accuracy_eligible": False,
            "formal_gate_accuracy_eligible": False,
        }
        for record in records
    ]
    for record in records:
        record["curve"] = np.asarray(record["curve"], dtype=float).tolist()
    return {
        "records": list(records),
        "prediction_ledger": prediction_ledger,
        "representations": representations,
        "direction_models": direction_models,
        "depth_matched_gate": {
            "contract": gate_contract,
            "training_target": "canonical A/B/C rate >=2.5e-4 versus weak A/B/C plus D",
            "training_target_counts": {
                "appreciable": int(gate_target.sum()),
                "other": int((gate_target == 0).sum()),
            },
            "model": stdbench._model_payload(gate_scaler, gate_model),
            "guardrail": "no Ozerov row has an independent gate truth",
        },
        "published_comparator": PUBLISHED_COMPARATOR,
        "pooled_snp_direct_frequency_geometry": dict(pooled_snp_geometry),
        "canonical_source_audit": canonical["audit"],
        "descriptive_panel_accounting": {
            "biological_systems": 1,
            "primary_views": 1,
            "correlated_donor_sensitivities": 1,
            "correlated_locus_sensitivities": 1,
            "independent_direction_truth_units": 0,
            "independent_gate_truth_units": 0,
            "guardrail": (
                "one temporal stocking system, not three independent validation trials or 1,986 SNP trials"
            ),
        },
        "guardrail": (
            "candidate-direction concordance is not accuracy: management history and published "
            "ancestry estimates do not supply independent exclusive-topology labels"
        ),
    }


def configuration(revision: Mapping[str, object]) -> dict:
    helper_sources = {}
    for name, module in {
        "brook_multiallelic": brook,
        "ooxml_reader": oyster,
        "canonical_pipeline": structured,
        "stdpopsim_model_payload": stdbench,
    }.items():
        path = Path(module.__file__).resolve()
        helper_sources[name] = {
            "path": str(path),
            "sha256": structured.sha256_file(path),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_revision": {
            key: revision.get(key)
            for key in (
                "commit",
                "script_sha256",
                "head_script_sha256",
                "head_blob_oid",
                "worktree_blob_oid",
                "tracked_diff_sha256",
                "tracked_dirty_at_snapshot",
            )
        },
        "helper_sources": helper_sources,
        "sources": SOURCE_CONTRACTS,
        "published_comparator": PUBLISHED_COMPARATOR,
        "panel": {
            "P1": "Loobu9699a",
            "P2": "Loobu0708a",
            "P3_primary": list(NARVA_MICROSATELLITE_GROUPS),
            "P3_primary_n": 720,
            "P3_donor_year_sensitivity": "Narva06a",
            "P3_donor_year_sensitivity_n": 112,
            "expected_candidate_direction": EXPECTED_DIRECTION,
            "expected_candidate_forward_edge": "P3->P2",
            "population_counts": GROUP_COUNTS,
            "population_count_sha256": EXPECTED_GROUP_COUNT_SHA256,
            "one_biological_system": True,
            "published_estimates_are_truth_labels": False,
            "management_arrow_is_gate_truth": False,
            "operational_tree_only": True,
        },
        "microsatellite_contract": {
            "individuals": 2_052,
            "loci": list(MICROSATELLITE_LOCI),
            "standard_loci": 17,
            "within_population_polymorphic_loci": 16,
            "strict_excluded_loci": ["Ssa14"],
            "ordered_locus_sha256": EXPECTED_LOCUS_SHA256,
            "ordered_locus_newline_sha256": EXPECTED_LOCUS_NEWLINE_SHA256,
            "strict_ordered_locus_sha256": EXPECTED_STRICT_LOCUS_SHA256,
            "normalized_sample_ledger_sha256": EXPECTED_SAMPLE_LEDGER_SHA256,
            "called_pairs": 34_319,
            "missing_0_0_pairs": 565,
            "partial_missing_pairs": 0,
            "all_nonmissing_pairs_nondecreasing": True,
            "all_microsatellite_alleles_retained": True,
        },
        "pooled_snp_contract": {
            "snps": 1_986,
            "population_headers": list(SNP_HEADERS),
            "frequency_values": 35_748,
            "ordered_population_header_sha256": EXPECTED_SNP_HEADER_SHA256,
            "normalized_snp_record_ledger_sha256": EXPECTED_SNP_RECORD_LEDGER_SHA256,
            "null_chromosome_and_both_map_fields": 64,
            "used_for_padze_scoring": False,
            "used_for_truth_construction": False,
            "use": "descriptive direct frequency geometry and published-comparator alignment",
        },
        "padze": {
            "depths": brook.PRIMARY_DEPTHS.tolist(),
            "moments": list(stdbench.MOMENTS),
            "pihat_sizes": [2],
            "bias_corrected": True,
            "allele_contract": (
                "all nonzero microsatellite alleles retained; exact 0/0 pairs treated as missing"
            ),
        },
        "evaluation": {
            "representations": list(REPRESENTATIONS),
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
        },
        "canonical_training_contract": {
            "replicates": 3_200,
            "label_counts": {"A": 900, "B": 900, "C": 900, "D": 500},
            "array_contracts": brook.CANONICAL_ARRAY_CONTRACTS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microsatellite", type=Path, required=True)
    parser.add_argument("--pooled-snp", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument(
        "--compute-state", type=Path, default=structured.DEFAULT_COMPUTE_STATE
    )
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
        "microsatellite": verify_source(args.microsatellite, "microsatellite"),
        "pooled_snp": verify_source(args.pooled_snp, "pooled_snp"),
    }
    loci, individuals, microsatellite_audit = parse_microsatellite_workbook(
        args.microsatellite
    )
    snp_headers, snp_records, snp_audit = parse_pooled_snp_workbook(args.pooled_snp)
    if snp_headers != SNP_HEADERS:
        raise AssertionError("Ozerov pooled-SNP header contract changed after parsing")
    panels = make_panels(individuals)
    config = configuration(revision)
    config_sha256 = hashlib.sha256(_canonical_json(config)).hexdigest()
    records, locus_selection_audit = build_records(
        loci, panels, compute_state=args.compute_state
    )
    snp_geometry = pooled_snp_frequency_geometry(snp_records)
    pre_analysis_gate = structured.compute_gate(args.compute_state)
    analysis = analyze_records(
        records,
        args.canonical_root,
        pooled_snp_geometry=snp_geometry,
        compute_state=args.compute_state,
    )

    final_revision = structured.git_revision(script=Path(__file__))
    structured.require_revision_unchanged(revision, final_revision)
    final_sources = {
        "microsatellite": verify_source(args.microsatellite, "microsatellite"),
        "pooled_snp": verify_source(args.pooled_snp, "pooled_snp"),
    }
    runtime = structured.runtime_audit(priority)
    runtime["packages"]["padze"] = importlib_metadata.version("padze")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "same_system_stocking_transfer_diagnostic_not_accuracy",
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
            "microsatellite": microsatellite_audit,
            "pooled_snp": snp_audit,
        },
        "selection_audits": {
            "loci": locus_selection_audit,
            "panels": {
                "record_views": 3,
                "biological_systems": 1,
                "primary_views": 1,
                "correlated_sensitivities": 2,
            },
        },
        "analysis": analysis,
    }
    with structured.SingleWriterLease(args.result_dir, ".ozerov_salmon_result.lock"):
        output = args.result_dir / "results.json"
        output_audit = structured.write_json_atomic(output, result, indent=2)
    primary = analysis["representations"]["raw_all"]["management_candidate_concordance"]
    print(json.dumps({
        "output": output_audit,
        "configuration_sha256": config_sha256,
        "raw_all_candidate_C_calls": primary["C_calls"],
        "raw_all_primary_views": primary["views"],
        "formal_direction_accuracy_eligible": False,
        "formal_gate_accuracy_eligible": False,
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
