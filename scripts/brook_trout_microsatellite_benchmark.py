#!/usr/bin/env python3
"""Score two stocking-history brook-trout microsatellite datasets.

This runner preserves every observed microsatellite allele and constructs PADZE
``LociData`` directly.  It deliberately does not use the repository's biallelic
VCF bridge, which would discard these multiallelic loci.

The biological candidate is hatchery P3 -> wild P2 (DNNaic class C).  Management
histories motivate that direction with evidence that varies by panel, but the
Pennsylvania and Nova Scotia target labels were inferred from the same markers
scored here.  Consequently every
call is an unaccepted, circular transfer diagnostic: no row is eligible for a
formal direction-accuracy or gate-accuracy denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
from typing import Iterable, Mapping, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dnnaic.semantics import class_for_forward_edge
from scripts import oyster_2017_external_benchmark as oyster
from scripts import stdpopsim_neanderthal_benchmark as stdbench
from scripts import structured_transfer_pilot as structured


SCHEMA_VERSION = "dnnaic-brook-trout-microsatellite-benchmark-v1"
PRIMARY_DEPTHS = np.arange(2, 17, dtype=np.int64)
MIN_CALLED_COPIES = 16
REPRESENTATIONS = (
    "raw_all",
    "raw_mean_variance",
    "orbit_composition_mean_variance",
)
EXPECTED_DIRECTION = class_for_forward_edge("P3", "P2")
DEFAULT_RESULT_DIR = REPO / "results" / "brook_trout_microsatellite_benchmark_2026_07_11"
EXPECTED_NS_STANDARD_LOCUS_SHA256 = "8a885c3cfb7166d92c8c35776411df29124dcb8b34d334a1dd1e1f3906cb2bd9"
EXPECTED_NS_STRICT_LOCUS_SHA256 = "22c023165573f435a6c9490acd8df3f8f7a2f123ec6e250b369ac6521d01567d"
EXPECTED_PA_ROW_LEDGER_SHA256 = "d0f1a3277cb8cd9e141277057cf764001e4322d38e5f80caccfcfcdce808f3c5"
EXPECTED_NS_ROW_LEDGER_SHA256 = "809ceeddc2c21fd1383872672ead0a1b9a4bd6a336ef7d9ae57847cc9c88aebf"
EXPECTED_NS_INTROGRESSION_LEDGER_SHA256 = "2b25d657892adb4ddfcceba1c9719bde8143203db5407e6cc0c1620e3308619d"
EXPECTED_NS_SENSITIVITY_LOCUS_COUNTS = {
    "ns_saint_marys": (90, 58),
    "ns_saint_marys_east": (90, 54),
    "ns_st_marys_bay": (89, 59),
}

CANONICAL_ARRAY_CONTRACTS = {
    "X.npy": {
        "bytes": 141_926_528,
        "sha256": "8a0a54b8d827301d47235ee196026687522180a9bcce07f2c52936e9d9bb56f5",
    },
    "design.npy": {
        "bytes": 25_344_128,
        "sha256": "beb06a522b59e10f311e5a130190159679b9c10595e30260e63c2f20a9c4500e",
    },
    "direction.npy": {
        "bytes": 2_534_528,
        "sha256": "a956a5bb90e147e3c0a4bf8527e0f8a3c8bd6d522fbc57f5e7a34742fdad7632",
    },
    "groups.npy": {
        "bytes": 76_032_128,
        "sha256": "e1a7c621e915615a178d44b4ce59c77da2d9c1f7549019acab587fe17da71a86",
    },
    "magnitude.npy": {
        "bytes": 5_068_928,
        "sha256": "417933cdad099ae4468253588ec9eb83ed323a34635c2e4dd0144cf13b59ee3c",
    },
}

SOURCE_CONTRACTS = {
    "pa_workbook": {
        "file": "White_et_al_Brook_Trout_Introgression.xlsx",
        "bytes": 253_709,
        "sha256": "54d112251103793e341ac97df066b73b1173c77ddf355ee862d2e7cf4eb1d1e4",
        "license": "CC0-1.0",
        "record": "https://zenodo.org/records/4979837",
        "download": (
            "https://zenodo.org/api/records/4979837/files/"
            "White_et_al_Brook_Trout_Introgression.xlsx/content"
        ),
        "data_doi": "10.5061/dryad.mb37t1q",
        "paper_doi": "10.1111/eva.12646",
    },
    "ns_genepop": {
        "file": "Lehnert_Brooktrout_100Microsatellites.txt",
        "bytes": 1_231_812,
        "sha256": "055ec1d7006368df3974483fc0f8042ee2176d3a21f615e7f5e1d777af3a5dda",
        "license": "CC0-1.0",
        "record": "https://zenodo.org/records/5013491",
        "download": (
            "https://zenodo.org/api/records/5013491/files/"
            "Lehnert_Brooktrout_100Microsatellites.txt/content"
        ),
        "data_doi": "10.5061/dryad.rv15dv44w",
        "paper_doi": "10.1111/eva.12923",
    },
    "ns_population_names": {
        "file": "Pop_Names_Lehnert_Brooktrout_100micros.csv",
        "bytes": 1_454,
        "sha256": "3ab94da572756545e833917069b33970e438f8a06d53cae9ed1e437ce19aa61b",
        "license": "CC0-1.0",
        "record": "https://zenodo.org/records/5013491",
        "download": (
            "https://zenodo.org/api/records/5013491/files/"
            "Pop_Names_Lehnert_Brooktrout_100micros.csv/content"
        ),
        "data_doi": "10.5061/dryad.rv15dv44w",
        "paper_doi": "10.1111/eva.12923",
    },
    "ns_introgression": {
        "file": "Introgression_AnthroEnviro_data.csv",
        "bytes": 9_656,
        "sha256": "48c9d06475b33fbcd990c6d99f59b3fcda466da0e4d7bac38b841c9c0d4d88c4",
        "license": "CC0-1.0",
        "record": "https://zenodo.org/records/5013491",
        "download": (
            "https://zenodo.org/api/records/5013491/files/"
            "Introgression_AnthroEnviro_data.csv/content"
        ),
        "data_doi": "10.5061/dryad.rv15dv44w",
        "paper_doi": "10.1111/eva.12923",
    },
}

PA_LOCI = (
    "B52", "C113", "C115", "C129", "C24", "C28",
    "C38", "C86", "C88", "D100", "D75", "D91",
)
PA_REFERENCE_SITES = ("MILA", "LEVL", "EAST", "UNT", "BEAR", "HUCK", "YELL", "SSR")
PA_HATCHERY_STRAINS = ("BELL", "OSW", "TYL", "BNSPB", "BNSPLP")
PA_TARGETS = {
    "DOUB": {"n": 154, "mean_p_wild": 0.95, "wild": 129, "introgressed": 25, "hatchery": 0},
    "LICK": {"n": 50, "mean_p_wild": 0.96, "wild": 42, "introgressed": 8, "hatchery": 0},
    "CONK": {"n": 50, "mean_p_wild": 0.94, "wild": 36, "introgressed": 14, "hatchery": 0},
}
PA_STOCKING_CONTEXT = {
    "DOUB": {
        "table_1_stocking_at_sample_location": True,
        "table_1_stocking_within_2_km": True,
        "discussion_states_not_directly_stocked": True,
        "no_stocking_record_more_than_50_years": False,
        "direct_stocking_status": "conflicting_table_1_direct_vs_discussion_not_direct",
        "source_internal_conflict": True,
    },
    "LICK": {
        "table_1_stocking_at_sample_location": False,
        "table_1_stocking_within_2_km": True,
        "discussion_states_not_directly_stocked": True,
        "no_stocking_record_more_than_50_years": False,
        "direct_stocking_status": "not_direct; stocking_within_2_km",
        "source_internal_conflict": False,
    },
    "CONK": {
        "table_1_stocking_at_sample_location": False,
        "table_1_stocking_within_2_km": False,
        "discussion_states_not_directly_stocked": True,
        "no_stocking_record_more_than_50_years": True,
        "direct_stocking_status": "not_direct; no_record_more_than_50_years",
        "source_internal_conflict": False,
    },
}
PA_DIRECTION_EVIDENCE = {
    "DOUB": (
        "source-internal conflict: Table 1 marks stocking at the sample location and within 2 km, "
        "but the Discussion says every site with more than 10% assigned introgression was not "
        "directly stocked; P3->P2 direction evidence is ambiguous"
    ),
    "LICK": (
        "Table 1 marks stocking within 2 km but not at the sample location, consistent with the "
        "Discussion's not-directly-stocked statement; P3->P2 is a landscape-scale candidate only"
    ),
    "CONK": (
        "not directly stocked and no stocking record for more than 50 years; the paper discusses "
        "legacy or unrecorded private stocking, so P3->P2 is a same-marker candidate only"
    ),
}
PA_PUBLISHED_COMPARATOR_PROVENANCE = {
    "paper_doi": "10.1111/eva.12646",
    "pmcid": "PMC6183464",
    "locator": "White et al. (2018), Evolutionary Applications 11:1567-1581, Table 1",
    "table_fields": "site N, mean p(wild), wild/introgressed/hatchery assignment counts",
    "stocking_history_locator": "Supporting Information Table S1",
    "direct_stocking_locator": (
        "Table 1 footnotes 1/2 plus Discussion statement that all sites with more than 10% "
        "assigned introgression were not directly stocked"
    ),
    "source_internal_conflict": (
        "DOUB has Table 1 footnotes 1 and 2 (stocking at sample location and within 2 km), "
        "but the Discussion categorically says all >10% assigned-introgression sites were not "
        "directly stocked"
    ),
    "checked_source": (
        "https://www.streamnet.org/app/hsrg/docs/"
        "White%20et%20al%202018%20Evolutionary%20Applications.pdf"
    ),
    "independently_checked_on": "2026-07-11",
    "manual_transcription_contract_sha256": (
        "7fe7731c99d356ecd6347a670958caebeed5ab61e80a73f79e13fba7f8e1ad5e"
    ),
}


@dataclass(frozen=True)
class Individual:
    sample_id: str
    population: str
    alleles: tuple[tuple[int, int], ...]
    source_ordinal: int


@dataclass(frozen=True)
class NSPanelSpec:
    panel_id: str
    biological_system_id: str
    river_system: str
    p1: str
    p2: str
    p3: str
    source_component: str
    geographic_guardrail: str | None = None


NS_PANELS = (
    NSPanelSpec("ns_annapolis", "ns_annapolis", "Annapolis", "FO", "WA", "FM", "FM"),
    NSPanelSpec("ns_baddeck", "ns_baddeck", "Baddeck", "MI", "Ang", "MR", "MR"),
    NSPanelSpec("ns_cornwallis", "ns_cornwallis", "Cornwallis", "RA", "Roc", "FM", "FM"),
    NSPanelSpec("ns_east_river_pictou", "ns_east_river_pictou", "EastRiverPictou", "Tho", "GL", "FM", "FM"),
    NSPanelSpec("ns_lahave", "ns_lahave", "Lahave", "BU", "CO", "FM", "FM"),
    NSPanelSpec("ns_margaree", "ns_margaree", "Margaree", "PO", "LakH", "MR", "MR"),
    NSPanelSpec("ns_musquodoboit", "ns_musquodoboit", "Musquodoboit", "GE", "MC", "FM", "FM"),
    NSPanelSpec("ns_river_denys", "ns_river_denys", "RiverDenys", "AL", "RD", "MR", "MR"),
)

NS_SENSITIVITY_PANELS = (
    NSPanelSpec(
        "ns_saint_marys",
        "ns_saint_marys",
        "SaintMarysRiver",
        "Cla",
        "Kel",
        "FM",
        "FM",
        "zero recorded stocking; within-main-branch unreported-source sensitivity",
    ),
    NSPanelSpec(
        "ns_saint_marys_east",
        "ns_saint_marys",
        "SaintMarysRiver",
        "GR",
        "MO",
        "FM",
        "FM",
        "east-branch correlated view of the same broader Saint Mary's biological system",
    ),
    NSPanelSpec(
        "ns_st_marys_bay",
        "ns_st_marys_bay",
        "StMarysBay",
        "Bou2",
        "Bou1",
        "FM",
        "FM",
        "zero recorded stocking; unreported-source sensitivity",
    ),
)


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _sample_ledger(individuals: Sequence[Individual]) -> dict:
    ids = [individual.sample_id for individual in individuals]
    ordinals = [int(individual.source_ordinal) for individual in individuals]
    if len(ids) != len(set(ids)):
        raise ValueError("sample identities are not unique")
    return {
        "individuals": len(ids),
        "ordered_sample_id_sha256": hashlib.sha256(_canonical_json(ids)).hexdigest(),
        "ordered_source_ordinals": ordinals,
        "ordered_source_ordinal_sha256": hashlib.sha256(_canonical_json(ordinals)).hexdigest(),
        "first_sample_id": ids[0] if ids else None,
        "last_sample_id": ids[-1] if ids else None,
    }


def verify_source(path: Path, contract_name: str) -> dict:
    contract = SOURCE_CONTRACTS[contract_name]
    if not path.is_file():
        raise FileNotFoundError(path)
    audit = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": structured.sha256_file(path),
    }
    if audit["bytes"] != contract["bytes"] or audit["sha256"] != contract["sha256"]:
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


def parse_pa_workbook(path: Path) -> tuple[tuple[str, ...], list[Individual], dict]:
    sheet = oyster.read_xlsx_sheet(path, "Microsatellite Genotypes")
    if sheet["dimension"] != "A1:Z2049" or (sheet["n_rows"], sheet["n_columns"]) != (2049, 26):
        raise AssertionError("Pennsylvania genotype worksheet shape changed")
    cells = sheet["cells"]
    header = [cells.get((0, column), "").strip() for column in range(26)]
    expected = ["ID", "Site"]
    for locus in PA_LOCI:
        expected.extend((f"{locus}_a", f"{locus}_b"))
    if header != expected:
        raise AssertionError(f"Pennsylvania locus/header contract changed: {header}")

    individuals = []
    called_pairs = 0
    missing_pairs = 0
    row_ledger = hashlib.sha256()
    for row in range(1, sheet["n_rows"]):
        raw_id = cells.get((row, 0), "").strip()
        site = cells.get((row, 1), "").strip()
        if not raw_id or not site:
            raise ValueError(f"Pennsylvania row {row + 1} lacks ID/site")
        pairs = []
        for locus_index, locus in enumerate(PA_LOCI):
            first = _exact_int(cells.get((row, 2 + 2 * locus_index), ""), f"row {row + 1} {locus}_a")
            second = _exact_int(cells.get((row, 3 + 2 * locus_index), ""), f"row {row + 1} {locus}_b")
            if (first == 0) != (second == 0):
                raise ValueError(f"row {row + 1} {locus}: partial missing genotype")
            if first < 0 or second < 0:
                raise ValueError(f"row {row + 1} {locus}: negative allele")
            missing_pairs += int(first == 0)
            called_pairs += int(first != 0)
            pairs.append((first, second))
        individual = Individual(
            sample_id=f"pa-row-{row + 1:04d}:{site}:{raw_id}",
            population=site,
            alleles=tuple(pairs),
            source_ordinal=row,
        )
        individuals.append(individual)
        row_ledger.update(_canonical_json(asdict(individual)))
        row_ledger.update(b"\n")

    site_counts = Counter(individual.population for individual in individuals)
    hatchery = set(PA_HATCHERY_STRAINS)
    raw_hatchery = sum(site_counts[site] for site in hatchery)
    raw_wild = len(individuals) - raw_hatchery
    repeated_site_id_keys = Counter(
        individual.sample_id.split(":", 1)[1] for individual in individuals
    )
    repeated = sum(count > 1 for count in repeated_site_id_keys.values())
    if (len(individuals), raw_wild, raw_hatchery) != (2048, 1748, 300):
        raise AssertionError("Pennsylvania raw cohort counts changed")
    if (called_pairs, missing_pairs, repeated) != (24_441, 135, 24):
        raise AssertionError("Pennsylvania genotype/missing/identity semantics changed")
    if row_ledger.hexdigest() != EXPECTED_PA_ROW_LEDGER_SHA256:
        raise AssertionError("Pennsylvania normalized row ledger changed")
    for site, expected_count in {
        **{name: 50 for name in ("MILA", "LEVL", "EAST", "BEAR", "YELL")},
        "UNT": 48,
        "HUCK": 22,
        "SSR": 50,
        "DOUB": 154,
        "LICK": 50,
        "CONK": 50,
        "BELL": 100,
        "OSW": 50,
        "TYL": 50,
        "BNSPB": 50,
        "BNSPLP": 50,
    }.items():
        if site_counts[site] != expected_count:
            raise AssertionError(f"Pennsylvania site count changed for {site}")
    return PA_LOCI, individuals, {
        "worksheet_member": sheet["member"],
        "worksheet_dimension": sheet["dimension"],
        "individuals": len(individuals),
        "wild_rows_raw": raw_wild,
        "hatchery_rows": raw_hatchery,
        "called_genotype_pairs": called_pairs,
        "missing_genotype_pairs": missing_pairs,
        "partial_missing_pairs": 0,
        "call_fraction": called_pairs / (called_pairs + missing_pairs),
        "site_counts": dict(sorted(site_counts.items())),
        "repeated_site_plus_raw_id_keys": repeated,
        "identity_contract": "immutable worksheet row ordinal plus site plus raw ID",
        "ordered_row_ledger_sha256": row_ledger.hexdigest(),
        "paper_source_mismatch": (
            "paper methods name SfoC-79; deposited workbook contains C115 and no C79"
        ),
        "paper_cohort_guardrail": (
            "raw workbook has 1748 wild rows; paper final 1742 cannot be reconstructed because "
            "the six FLAG/POLE exclusions are not identified"
        ),
    }


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_ns_population_names(path: Path) -> tuple[list[dict[str, str]], dict]:
    rows = _read_csv_rows(path)
    if len(rows) != 39 or list(rows[0]) != ["Code", "River System", "Site"]:
        raise AssertionError("Nova Scotia population-name table changed")
    codes = [row["Code"].strip() for row in rows]
    if any(not code for code in codes) or len(codes) != len(set(codes)):
        raise ValueError("Nova Scotia population codes are blank or duplicated")
    return rows, {
        "populations": len(rows),
        "ordered_population_code_sha256": hashlib.sha256(_canonical_json(codes)).hexdigest(),
        "ordered_population_codes": codes,
    }


def parse_ns_introgression(path: Path) -> tuple[dict[str, dict], dict]:
    rows = _read_csv_rows(path)
    required = {
        "Code", "RiverSystem", "STR_locPrior", "Hatchery_included",
        "Introgression_FM", "Introgression_MR", "Introgression_total",
        "Total_Fish_Stocked", "FallStocking_n", "SpringStocking_n",
    }
    if len(rows) != 33 or not required.issubset(rows[0]):
        raise AssertionError("Nova Scotia introgression table changed")
    parsed = {}
    for row in rows:
        code = row["Code"].strip()
        values = {name: float(row[name]) for name in (
            "Introgression_FM", "Introgression_MR", "Introgression_total"
        )}
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in values.values()):
            raise ValueError(f"Nova Scotia introgression value is outside [0,1] for {code}")
        stocking = {
            name: _exact_int(row[name], f"{code} {name}")
            for name in ("Total_Fish_Stocked", "FallStocking_n", "SpringStocking_n")
        }
        if any(value < 0 for value in stocking.values()):
            raise ValueError(f"Nova Scotia stocking count is negative for {code}")
        if stocking["FallStocking_n"] + stocking["SpringStocking_n"] != stocking["Total_Fish_Stocked"]:
            raise ValueError(f"Nova Scotia stocking components do not sum for {code}")
        if not math.isclose(
            values["Introgression_FM"] + values["Introgression_MR"],
            values["Introgression_total"], rel_tol=0, abs_tol=1e-12,
        ):
            raise ValueError(f"Nova Scotia introgression components do not sum for {code}")
        parsed[code] = {
            "river_system": row["RiverSystem"].strip(),
            "structure_location_prior": row["STR_locPrior"].strip(),
            "hatchery_included": row["Hatchery_included"].strip(),
            **values,
            **stocking,
        }
    if len(parsed) != 33:
        raise ValueError("Nova Scotia introgression codes are duplicated")
    semantic_sha256 = hashlib.sha256(_canonical_json(parsed)).hexdigest()
    if semantic_sha256 != EXPECTED_NS_INTROGRESSION_LEDGER_SHA256:
        raise AssertionError("Nova Scotia normalized introgression ledger changed")
    return parsed, {
        "wild_rows": len(parsed),
        "hatchery_included_counts": dict(sorted(Counter(
            value["hatchery_included"] for value in parsed.values()
        ).items())),
        "location_prior_counts": dict(sorted(Counter(
            value["structure_location_prior"] for value in parsed.values()
        ).items())),
        "semantic_row_sha256": semantic_sha256,
    }


GENEPOP_TOKEN = re.compile(r"^[0-9]{6}$")


def decode_genepop_token(token: str, context: str) -> tuple[int, int]:
    if GENEPOP_TOKEN.fullmatch(token) is None:
        raise ValueError(f"{context}: malformed token {token!r}")
    first, second = int(token[:3]), int(token[3:])
    if (first == 0) != (second == 0):
        raise ValueError(f"{context}: partial missing genotype")
    return first, second


def parse_genepop(
    path: Path,
    population_codes: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, list[Individual]], dict]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if not lines:
        raise ValueError("empty Genepop source")
    first_pop = next((index for index, line in enumerate(lines[1:], start=1)
                      if line.strip().lower() == "pop"), None)
    if first_pop is None:
        raise ValueError("Genepop source has no Pop delimiter")
    loci = tuple(line.strip() for line in lines[1:first_pop] if line.strip())
    if len(loci) != 100 or len(loci) != len(set(loci)):
        raise AssertionError("Nova Scotia Genepop locus contract changed")
    blocks: list[list[tuple[str, tuple[tuple[int, int], ...], int]]] = []
    current: list[tuple[str, tuple[tuple[int, int], ...], int]] | None = None
    called_pairs = 0
    missing_pairs = 0
    all_raw_ids = []
    row_ordinal = 0
    for line_number, line in enumerate(lines[first_pop:], start=first_pop + 1):
        if line.strip().lower() == "pop":
            current = []
            blocks.append(current)
            continue
        if not line.strip():
            continue
        if current is None or line.count(",") != 1:
            raise ValueError(f"Genepop line {line_number}: malformed sample row")
        raw_id, payload = line.split(",", 1)
        raw_id = raw_id.strip()
        tokens = payload.split()
        if not raw_id or len(tokens) != len(loci):
            raise ValueError(f"Genepop line {line_number}: sample/locus count changed")
        pairs = []
        for locus, token in zip(loci, tokens):
            first, second = decode_genepop_token(
                token, f"Genepop line {line_number} {locus}"
            )
            missing_pairs += int(first == 0)
            called_pairs += int(first != 0)
            pairs.append((first, second))
        row_ordinal += 1
        current.append((raw_id, tuple(pairs), row_ordinal))
        all_raw_ids.append(raw_id)
    if len(blocks) != len(population_codes) or any(not block for block in blocks):
        raise AssertionError("Nova Scotia Genepop population-section count changed")
    populations = {}
    row_ledger = hashlib.sha256()
    for code, block in zip(population_codes, blocks):
        individuals = []
        for raw_id, pairs, ordinal in block:
            if not raw_id.startswith(f"{code}_"):
                raise ValueError(f"sample {raw_id!r} does not match population block {code}")
            individual = Individual(
                sample_id=f"ns-row-{ordinal:04d}:{raw_id}",
                population=code,
                alleles=pairs,
                source_ordinal=ordinal,
            )
            individuals.append(individual)
            row_ledger.update(_canonical_json(asdict(individual)))
            row_ledger.update(b"\n")
        populations[code] = individuals
    if len(all_raw_ids) != 1729 or len(set(all_raw_ids)) != 1729:
        raise AssertionError("Nova Scotia sample identity/count contract changed")
    if (called_pairs, missing_pairs) != (168_141, 4_759):
        raise AssertionError("Nova Scotia called/missing genotype contract changed")
    if row_ledger.hexdigest() != EXPECTED_NS_ROW_LEDGER_SHA256:
        raise AssertionError("Nova Scotia normalized genotype row ledger changed")
    return loci, populations, {
        "title": lines[0],
        "loci": len(loci),
        "populations": len(populations),
        "individuals": len(all_raw_ids),
        "called_genotype_pairs": called_pairs,
        "missing_genotype_pairs": missing_pairs,
        "partial_missing_pairs": 0,
        "call_fraction": called_pairs / (called_pairs + missing_pairs),
        "population_sample_counts": {
            code: len(populations[code]) for code in population_codes
        },
        "ordered_locus_sha256": hashlib.sha256(_canonical_json(list(loci))).hexdigest(),
        "ordered_row_ledger_sha256": row_ledger.hexdigest(),
    }


def _population_rows(
    individuals: Sequence[Individual], population: str
) -> list[Individual]:
    return sorted(
        (individual for individual in individuals if individual.population == population),
        key=lambda individual: individual.source_ordinal,
    )


def _first_rows(
    individuals: Sequence[Individual], population: str, count: int
) -> list[Individual]:
    rows = _population_rows(individuals, population)
    if len(rows) < count:
        raise ValueError(f"{population}: requested {count} rows from only {len(rows)}")
    return rows[:count]


def make_pa_panels(individuals: Sequence[Individual]) -> tuple[list[dict], dict]:
    reference_pool = []
    reference_sites = {}
    for site in PA_REFERENCE_SITES:
        selected = _first_rows(individuals, site, 22)
        reference_pool.extend(selected)
        reference_sites[site] = _sample_ledger(selected)
    donor_pool = []
    donor_strains = {}
    for strain in PA_HATCHERY_STRAINS:
        selected = _first_rows(individuals, strain, 50)
        donor_pool.extend(selected)
        donor_strains[strain] = _sample_ledger(selected)
    levl = _population_rows(individuals, "LEVL")
    if (len(reference_pool), len(donor_pool), len(levl)) != (176, 250, 50):
        raise AssertionError("Pennsylvania deterministic selection contract changed")

    panels = []
    for target, published in PA_TARGETS.items():
        recipient = _population_rows(individuals, target)
        if len(recipient) != published["n"]:
            raise AssertionError(f"Pennsylvania target count changed for {target}")
        common = {
            "dataset": "pennsylvania_white_2018",
            "biological_system_id": f"pa_{target.lower()}",
            "expected_candidate_direction": EXPECTED_DIRECTION,
            "candidate_direction_evidence": PA_DIRECTION_EVIDENCE[target],
            "population_semantics": {
                "P1": "wild reference",
                "P2": f"wild target {target}",
                "P3": "balanced hatchery-strain proxy",
                "forward_candidate": "P3->P2",
            },
            "published_comparator": {
                **published,
                "introgressed_fraction": published["introgressed"] / published["n"],
                "label_source": "STRUCTURE assignments from the same 12 loci",
                "stocking_context": PA_STOCKING_CONTEXT[target],
                "outcome_selected_high_same_marker_introgression": True,
                "provenance": PA_PUBLISHED_COMPARATOR_PROVENANCE,
                "independent_truth": False,
            },
            "selection_guardrail": (
                "targets were preselected for high same-marker assigned introgression; workbook-"
                "order deterministic sensitivity is not the authors' unreleased random wild/"
                "hatchery centroid subset"
            ),
        }
        panels.append({
            **common,
            "panel_id": f"pa_{target.lower()}_clean8_pool",
            "contract_role": "primary",
            "groups": (reference_pool, recipient, donor_pool),
            "reference_contract": (
                "first 22 workbook rows from each of eight sites with zero published non-wild "
                "assignments in the same-locus STRUCTURE analysis"
            ),
        })
        panels.append({
            **common,
            "panel_id": f"pa_{target.lower()}_levl_reference",
            "contract_role": "reference_sensitivity",
            "groups": (levl, recipient, donor_pool),
            "reference_contract": "all LEVL rows; topology/sister relationship is not established",
        })
    return panels, {
        "reference_pool": {
            "sites": reference_sites,
            "combined": _sample_ledger(reference_pool),
            "selection": "first 22 source-order rows per site",
        },
        "levl_reference": _sample_ledger(levl),
        "hatchery_pool": {
            "strains": donor_strains,
            "combined": _sample_ledger(donor_pool),
            "selection": "first 50 source-order rows per strain; BELL is the only subsampled strain",
        },
        "topology_guardrail": (
            "the eight-site P1 pool is structured and not one population; LEVL avoids pooling "
            "but is not established as the targets' sister/reference population"
        ),
    }


def make_ns_panels(
    populations: Mapping[str, Sequence[Individual]],
    introgression: Mapping[str, Mapping[str, object]],
) -> tuple[list[dict], dict]:
    panels = []
    selection_ledger = []
    panel_contracts = [(spec, "base") for spec in NS_PANELS]
    panel_contracts.extend(
        (spec, "no_recorded_stocking_sensitivity")
        for spec in NS_SENSITIVITY_PANELS
    )
    for spec, contract_role in panel_contracts:
        p1 = list(populations[spec.p1])
        p2 = list(populations[spec.p2])
        p3 = list(populations[spec.p3])
        low = introgression[spec.p1]
        high = introgression[spec.p2]
        if low["river_system"] != spec.river_system or high["river_system"] != spec.river_system:
            raise AssertionError(f"{spec.panel_id}: published river-system contract changed")
        if not float(high["Introgression_total"]) > float(low["Introgression_total"]):
            raise AssertionError(f"{spec.panel_id}: P2 is no longer the higher published-Q site")
        total_stocked = int(high["Total_Fish_Stocked"])
        if contract_role == "base" and total_stocked <= 0:
            raise AssertionError(f"{spec.panel_id}: primary panel lacks recorded stocking")
        if contract_role != "base" and total_stocked != 0:
            raise AssertionError(f"{spec.panel_id}: no-record sensitivity now has recorded stocking")
        included = str(high["hatchery_included"])
        if spec.source_component not in included.split("_"):
            raise AssertionError(f"{spec.panel_id}: chosen source was not included by the paper")
        source_field = f"Introgression_{spec.source_component}"
        component_delta = float(high[source_field]) - float(low[source_field])
        if included == "FM_MR":
            other = "MR" if spec.source_component == "FM" else "FM"
            other_delta = float(high[f"Introgression_{other}"]) - float(low[f"Introgression_{other}"])
            if component_delta < other_delta:
                raise AssertionError(f"{spec.panel_id}: chosen source is not the larger published component")
        published = {
            "P1_code": spec.p1,
            "P2_code": spec.p2,
            "P3_code": spec.p3,
            "P1_introgression_total": float(low["Introgression_total"]),
            "P2_introgression_total": float(high["Introgression_total"]),
            "delta_introgression_total": float(high["Introgression_total"]) - float(low["Introgression_total"]),
            "P3_raw_genepop_population_code": spec.p3,
            "published_centered_source_component": spec.source_component,
            "source_component_delta": component_delta,
            "hatchery_sources_in_published_analysis": included,
            "total_fish_stocked": total_stocked,
            "fall_stocking_n": int(high["FallStocking_n"]),
            "spring_stocking_n": int(high["SpringStocking_n"]),
            "recorded_stocking_exposure": bool(total_stocked > 0),
            "label_source": "mean STRUCTURE Q from the same 100 loci",
            "outcome_selected_min_max_within_scope": True,
            "selection_scope": (
                "recorded-stocking river system"
                if contract_role == "base"
                else "precommitted no-record river branch/system sensitivity"
            ),
            "independent_truth": False,
        }
        panels.append({
            "panel_id": spec.panel_id,
            "dataset": "nova_scotia_lehnert_2020",
            "biological_system_id": spec.biological_system_id,
            "contract_role": contract_role,
            "groups": (p1, p2, p3),
            "expected_candidate_direction": EXPECTED_DIRECTION,
            "candidate_direction_evidence": (
                "recorded stocking fixes hatchery-to-wild management direction"
                if total_stocked > 0
                else (
                    "no recorded stocking; FM was included by the paper to detect possible "
                    "unreported stocking, so class C is a same-marker exploratory candidate only"
                )
            ),
            "population_semantics": {
                "P1": f"lower published-Q wild site {spec.p1}",
                "P2": f"higher published-Q wild site {spec.p2}",
                "P3": (
                    f"raw Genepop population {spec.p3} proxy; this code is not the "
                    f"paper's centered {spec.source_component} source"
                ),
                "forward_candidate": "P3->P2",
                "hatchery_included_is_not_exposure_evidence": True,
            },
            "published_comparator": published,
            "geographic_guardrail": spec.geographic_guardrail,
            "selection_guardrail": (
                "P1/P2 are precommitted published-Q min/max rows and therefore favorably, "
                "circularly outcome-selected; for FM_MR systems MR is additionally chosen because "
                "its same-marker Q delta is larger; raw P3 is not the paper's simulated centered source"
            ),
        })
        selection_ledger.append({
            **asdict(spec),
            "contract_role": contract_role,
            "P1_samples": _sample_ledger(p1),
            "P2_samples": _sample_ledger(p2),
            "P3_samples": _sample_ledger(p3),
            **published,
        })
    if len(panels) != 11 or len({panel["biological_system_id"] for panel in panels}) != 10:
        raise AssertionError("Nova Scotia descriptive panel contract changed")
    return panels, {
        "panels": selection_ledger,
        "excluded_singleton_rivers": ["SalmonRiverTruro", "UpperMedway"],
        "source_proxy_guardrail": (
            "raw Genepop FM is only the n=49 domestic strain even though Introgression_FM names "
            "the paper's centered five-group Fraser's Mills source; MR is also raw. These name-"
            "colliding proxies are not exact reproductions of either centered source"
        ),
    }


def _alleles_for_group(
    individuals: Sequence[Individual], locus_index: int
) -> list[int]:
    return [
        allele
        for individual in individuals
        for allele in individual.alleles[locus_index]
        if allele != 0
    ]


def eligible_shared_loci(
    locus_names: Sequence[str],
    panels: Sequence[Mapping[str, object]],
    *,
    min_called_copies: int = MIN_CALLED_COPIES,
) -> tuple[list[str], list[str], dict]:
    if not panels or min_called_copies < 2:
        raise ValueError("shared-locus selection requires panels and at least two called copies")
    standard = []
    strict = []
    ledger = []
    for locus_index, locus in enumerate(locus_names):
        standard_ok = True
        strict_ok = True
        panel_rows = []
        for panel in panels:
            groups = panel["groups"]
            observed = [_alleles_for_group(group, locus_index) for group in groups]
            called = [len(values) for values in observed]
            cardinality = [len(set(values)) for values in observed]
            globally_polymorphic = len(set().union(*(set(values) for values in observed))) >= 2
            current_standard = min(called) >= min_called_copies and globally_polymorphic
            current_strict = current_standard and min(cardinality) >= 2
            standard_ok &= current_standard
            strict_ok &= current_strict
            panel_rows.append({
                "panel_id": panel["panel_id"],
                "called_copies": called,
                "allele_cardinality": cardinality,
                "globally_polymorphic": globally_polymorphic,
            })
        if standard_ok:
            standard.append(str(locus))
        if strict_ok:
            strict.append(str(locus))
        ledger.append({
            "locus": str(locus),
            "standard": bool(standard_ok),
            "within_population_polymorphic": bool(strict_ok),
            "panels": panel_rows,
        })
    if not set(strict).issubset(standard):
        raise AssertionError("strict loci are not a subset of standard loci")
    return standard, strict, {
        "input_loci": len(locus_names),
        "minimum_called_copies_per_population": min_called_copies,
        "standard_loci": len(standard),
        "strict_loci": len(strict),
        "standard_ordered_locus_sha256": hashlib.sha256(_canonical_json(standard)).hexdigest(),
        "strict_ordered_locus_sha256": hashlib.sha256(_canonical_json(strict)).hexdigest(),
        "selection_ledger_sha256": hashlib.sha256(_canonical_json(ledger)).hexdigest(),
    }


def panel_to_counts(
    locus_names: Sequence[str],
    groups: Sequence[Sequence[Individual]],
    selected_loci: Sequence[str],
    *,
    require_within_population_polymorphism: bool,
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    if len(groups) != 3 or any(not group for group in groups):
        raise ValueError("a PADZE panel requires three nonempty populations")
    sample_sets = [set(individual.sample_id for individual in group) for group in groups]
    if any(sample_sets[first] & sample_sets[second] for first, second in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("P1/P2/P3 sample identities must be pairwise disjoint")
    if any(
        len(individual.alleles) != len(locus_names)
        for group in groups
        for individual in group
    ):
        raise ValueError("individual genotype width differs from the locus contract")
    index = {str(locus): position for position, locus in enumerate(locus_names)}
    if len(index) != len(locus_names) or len(selected_loci) != len(set(selected_loci)):
        raise ValueError("locus names or selection are duplicated")
    count_matrices = []
    sample_sizes = []
    count_ledger = hashlib.sha256()
    cardinalities = []
    for locus in selected_loci:
        try:
            locus_index = index[str(locus)]
        except KeyError as exc:
            raise ValueError(f"selected locus {locus!r} is absent") from exc
        observed = [_alleles_for_group(group, locus_index) for group in groups]
        labels = sorted(set().union(*(set(values) for values in observed)))
        if len(labels) < 2 or min(map(len, observed)) < MIN_CALLED_COPIES:
            raise ValueError(f"selected locus {locus} violates the standard contract")
        if require_within_population_polymorphism and min(len(set(values)) for values in observed) < 2:
            raise ValueError(f"selected locus {locus} violates the strict contract")
        label_index = {value: position for position, value in enumerate(labels)}
        counts = np.zeros((3, len(labels)), dtype=np.int64)
        for population_index, values in enumerate(observed):
            for allele in values:
                counts[population_index, label_index[allele]] += 1
        sizes = counts.sum(axis=1)
        if np.any(counts < 0) or not np.array_equal(sizes, np.asarray(list(map(len, observed)))):
            raise AssertionError("microsatellite count construction failed conservation")
        count_matrices.append(counts)
        sample_sizes.append(sizes)
        cardinalities.append(len(labels))
        count_ledger.update(_canonical_json({
            "locus": str(locus), "allele_labels": labels, "counts": counts.tolist()
        }))
        count_ledger.update(b"\n")
    if len(count_matrices) < 2:
        raise ValueError("fewer than two loci survive the panel contract")
    sizes = np.vstack(sample_sizes).astype(np.int64, copy=False)
    denominator = len(selected_loci) * sum(2 * len(group) for group in groups)
    return count_matrices, sizes, {
        "loci": len(count_matrices),
        "ordered_locus_sha256": hashlib.sha256(_canonical_json(list(selected_loci))).hexdigest(),
        "ordered_allele_count_ledger_sha256": count_ledger.hexdigest(),
        "alleles_per_locus": {
            "minimum": min(cardinalities),
            "mean": float(np.mean(cardinalities)),
            "maximum": max(cardinalities),
            "multiallelic_loci": int(sum(value > 2 for value in cardinalities)),
        },
        "called_copies": {
            name: {
                "minimum": int(sizes[:, population].min()),
                "mean": float(sizes[:, population].mean()),
                "maximum": int(sizes[:, population].max()),
            }
            for population, name in enumerate(("P1", "P2", "P3"))
        },
        "missing_copy_fraction": float(1.0 - sizes.sum() / denominator),
        "sample_ledgers": {
            name: _sample_ledger(group)
            for name, group in zip(("P1", "P2", "P3"), groups)
        },
        "within_population_polymorphism_required": require_within_population_polymorphism,
    }


def multiallelic_frequency_geometry(
    count_matrices: Sequence[np.ndarray], locus_names: Sequence[str]
) -> dict:
    if len(count_matrices) != len(locus_names):
        raise ValueError("frequency-geometry count/locus lengths differ")
    projections = []
    projection_numerators = []
    projection_denominators = []
    f3_values = []
    zero_reference_donor_distance = 0
    ledger = []
    for locus, counts in zip(locus_names, count_matrices):
        counts = np.asarray(counts, dtype=float)
        frequencies = counts / counts.sum(axis=1, keepdims=True)
        p1, p2, p3 = frequencies
        donor_axis = p3 - p1
        denominator = float(np.dot(donor_axis, donor_axis))
        if denominator > 0:
            numerator = float(np.dot(p2 - p1, donor_axis))
            projection = float(numerator / denominator)
            projections.append(projection)
            projection_numerators.append(numerator)
            projection_denominators.append(denominator)
        else:
            numerator = None
            projection = None
            zero_reference_donor_distance += 1
        f3 = float(np.dot(p2 - p1, p2 - p3))
        f3_values.append(f3)
        ledger.append([str(locus), numerator, denominator, projection, f3])
    return {
        "description": (
            "equal-locus aligned-frequency projection of P2-P1 onto P3-P1 and f3(P2;P1,P3); "
            "descriptive same-marker geometry, not an independent direction method"
        ),
        "projection_loci": len(projections),
        "zero_P1_P3_distance_loci": zero_reference_donor_distance,
        "mean_projection": None if not projections else float(np.mean(projections)),
        "median_projection": None if not projections else float(np.median(projections)),
        "denominator_weighted_projection": (
            None
            if not projection_denominators
            else float(sum(projection_numerators) / sum(projection_denominators))
        ),
        "P1_P3_squared_distance": {
            "minimum_nonzero": None if not projection_denominators else float(min(projection_denominators)),
            "median_nonzero": None if not projection_denominators else float(np.median(projection_denominators)),
            "sum": float(sum(projection_denominators)),
        },
        "mean_f3": float(np.mean(f3_values)),
        "negative_f3_loci": int(sum(value < 0 for value in f3_values)),
        "locus_geometry_sha256": hashlib.sha256(_canonical_json(ledger)).hexdigest(),
    }


def loci_to_curve(
    count_matrices: Sequence[np.ndarray],
    sample_sizes: np.ndarray,
    locus_names: Sequence[str],
    groups: Sequence[Sequence[Individual]],
    *,
    n_loci_read: int,
    source: str,
    filters: Sequence[str],
    compute_state: Path | None = None,
) -> tuple[np.ndarray, dict]:
    from padze import LociData, Metadata, compute_features

    if compute_state is not None:
        structured.compute_gate(compute_state)
    denominator = len(locus_names) * sum(2 * len(group) for group in groups)
    loci = LociData(
        populations=["P1", "P2", "P3"],
        count_matrices=[np.asarray(matrix, dtype=np.int64) for matrix in count_matrices],
        sample_sizes=np.asarray(sample_sizes, dtype=np.int64),
        locus_ids=list(locus_names),
        metadata=Metadata(
            source=source,
            populations=["P1", "P2", "P3"],
            sample_ids={
                name: [individual.sample_id for individual in group]
                for name, group in zip(("P1", "P2", "P3"), groups)
            },
            ploidy={
                individual.sample_id: 2
                for group in groups
                for individual in group
            },
            n_loci_read=int(n_loci_read),
            n_loci_kept=len(locus_names),
            filters_applied=list(filters),
            missing_fraction=float(1.0 - np.asarray(sample_sizes).sum() / denominator),
        ),
    )
    table = compute_features(
        loci,
        depths=PRIMARY_DEPTHS,
        pihat_sizes=(2,),
        moments=stdbench.MOMENTS,
        bias_corrected=True,
    )
    matrix, columns = table.to_frame()
    column_index = {column: index for index, column in enumerate(columns)}
    try:
        curve = matrix[:, [column_index[column] for column in stdbench.CURVE_COLUMNS]].astype(np.float64)
    except KeyError as exc:
        raise RuntimeError(f"brook-trout PADZE feature contract changed: {exc}") from exc
    if curve.shape != (15, 28) or not np.isfinite(curve).all():
        raise AssertionError("brook-trout PADZE curve is invalid")
    if not np.array_equal(curve[:, 0], PRIMARY_DEPTHS):
        raise AssertionError("brook-trout PADZE depth grid changed")
    return curve, {
        "n_loci_read": int(n_loci_read),
        "n_loci_kept": len(locus_names),
        "depths": PRIMARY_DEPTHS.tolist(),
        "columns": list(stdbench.CURVE_COLUMNS),
        "bias_corrected": True,
        "pihat_sizes": [2],
        "curve_sha256_float64": _sha256_array(curve.astype("<f8", copy=False)),
        "curve_sha256_float32": _sha256_array(curve.astype("<f4", copy=False)),
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
    count_matrices, sample_sizes, count_audit = panel_to_counts(
        locus_names,
        groups,
        selected_loci,
        require_within_population_polymorphism=strict,
    )
    curve, padze_audit = loci_to_curve(
        count_matrices,
        sample_sizes,
        selected_loci,
        groups,
        n_loci_read=len(locus_names),
        source=f"{panel['dataset']} panel {panel['panel_id']} filter {filter_name}",
        filters=[
            (
                "recorded-stocking primary panels share one frozen list; no-record "
                "sensitivities use documented intersections fixed before scoring"
            )
            if panel["dataset"] == "nova_scotia_lehnert_2020"
            else "all 12 deposited Pennsylvania loci",
            "at least 16 called gene copies in P1/P2/P3",
            "globally polymorphic in the frozen triplet; all alleles retained",
            *(["polymorphic within every population"] if strict else []),
        ],
        compute_state=compute_state,
    )
    output = {key: value for key, value in panel.items() if key != "groups"}
    output.update({
        "panel_view_id": f"{panel['panel_id']}__{filter_name}",
        "locus_filter": filter_name,
        "count_audit": count_audit,
        "frequency_geometry": multiallelic_frequency_geometry(count_matrices, selected_loci),
        "padze": padze_audit,
        "curve": curve,
        "formal_direction_accuracy_eligible": False,
        "gate_accuracy_eligible": False,
        "direction_call_accepted": False,
        "ground_truth_guardrail": (
            "management history motivates P3->P2 with panel-specific strength only conditional "
            "on detectable ancestry; the published comparator reuses these loci and exclusive "
            "topology is unknown"
        ),
    })
    return output


def build_records(
    pa_loci: Sequence[str],
    pa_panels: Sequence[Mapping[str, object]],
    ns_loci: Sequence[str],
    ns_panels: Sequence[Mapping[str, object]],
    *,
    compute_state: Path | None,
) -> tuple[list[dict], dict]:
    pa_standard, pa_strict, pa_filter_audit = eligible_shared_loci(pa_loci, pa_panels)
    if pa_standard != list(pa_loci) or pa_strict != list(pa_loci):
        raise AssertionError("not all Pennsylvania loci satisfy both frozen contracts")
    ns_primary_panels = [panel for panel in ns_panels if panel["contract_role"] == "base"]
    ns_standard, ns_strict, ns_filter_audit = eligible_shared_loci(
        ns_loci, ns_primary_panels
    )
    if (len(ns_standard), len(ns_strict)) != (90, 62):
        raise AssertionError(
            f"Nova Scotia shared-locus contract changed: {len(ns_standard)}/{len(ns_strict)}"
        )
    if (
        ns_filter_audit["standard_ordered_locus_sha256"] != EXPECTED_NS_STANDARD_LOCUS_SHA256
        or ns_filter_audit["strict_ordered_locus_sha256"] != EXPECTED_NS_STRICT_LOCUS_SHA256
    ):
        raise AssertionError("Nova Scotia shared ordered-locus identities changed")

    records = []
    for panel in pa_panels:
        records.append(_panel_record(
            panel,
            pa_loci,
            pa_standard,
            filter_name="all_deposited_loci",
            strict=False,
            compute_state=compute_state,
        ))
    sensitivity_locus_views = {}
    for panel in ns_panels:
        if panel["contract_role"] != "base":
            eligible_standard, eligible_strict, _ = eligible_shared_loci(
                ns_loci, [panel]
            )
            standard_intersection = [
                locus for locus in ns_standard if locus in set(eligible_standard)
            ]
            strict_intersection = [
                locus for locus in ns_strict if locus in set(eligible_strict)
            ]
            sensitivity_locus_views[panel["panel_id"]] = {
                "standard_loci": len(standard_intersection),
                "strict_loci": len(strict_intersection),
                "standard_ordered_locus_sha256": hashlib.sha256(
                    _canonical_json(standard_intersection)
                ).hexdigest(),
                "strict_ordered_locus_sha256": hashlib.sha256(
                    _canonical_json(strict_intersection)
                ).hexdigest(),
                "guardrail": "intersection with the recorded-stocking primary list; did not select it",
            }
            if (len(standard_intersection), len(strict_intersection)) != (
                EXPECTED_NS_SENSITIVITY_LOCUS_COUNTS[panel["panel_id"]]
            ):
                raise AssertionError(
                    f"{panel['panel_id']}: no-record sensitivity locus contract changed"
                )
            records.append(_panel_record(
                panel,
                ns_loci,
                standard_intersection,
                filter_name="no_record_primary_standard_intersection",
                strict=False,
                compute_state=compute_state,
            ))
            strict_panel = dict(panel)
            strict_panel["contract_role"] = "no_recorded_stocking_strict_locus_sensitivity"
            records.append(_panel_record(
                strict_panel,
                ns_loci,
                strict_intersection,
                filter_name="no_record_primary_strict_intersection",
                strict=True,
                compute_state=compute_state,
            ))
            continue
        standard_panel = dict(panel)
        standard_panel["contract_role"] = "primary"
        records.append(_panel_record(
            standard_panel,
            ns_loci,
            ns_standard,
            filter_name="shared_standard",
            strict=False,
            compute_state=compute_state,
        ))
        strict_panel = dict(panel)
        strict_panel["contract_role"] = "locus_filter_sensitivity"
        records.append(_panel_record(
            strict_panel,
            ns_loci,
            ns_strict,
            filter_name="shared_within_population_polymorphic",
            strict=True,
            compute_state=compute_state,
        ))
    if len(records) != 28:
        raise AssertionError("brook-trout record bank changed")
    primary = [record for record in records if record["contract_role"] == "primary"]
    if len(primary) != 11 or len({record["biological_system_id"] for record in primary}) != 11:
        raise AssertionError("descriptive primary-view accounting changed")
    ns_filter_audit["no_recorded_stocking_sensitivity_intersections"] = sensitivity_locus_views
    return records, {
        "pennsylvania": pa_filter_audit,
        "nova_scotia": ns_filter_audit,
        "record_views": len(records),
        "descriptive_primary_views": len(primary),
        "distinct_named_primary_target_or_river_views": 11,
        "guardrail": (
            "11 is descriptive panel accounting, not independent validation systems or an "
            "accuracy denominator; reference, no-record, and locus-filter sensitivities are correlated"
        ),
    }


def depth_matched_gate_features(table: np.ndarray) -> tuple[np.ndarray, dict]:
    table = structured.validate_curve_table(table)
    if table.shape[1] != len(PRIMARY_DEPTHS) or not np.array_equal(table[0, :, 0], PRIMARY_DEPTHS):
        raise ValueError("depth-matched gate requires g=2..16")
    indices = np.unique(np.round(np.geomspace(1, table.shape[1] - 1, 8)).astype(int))
    curves = table[:, :, 1:]
    features = np.concatenate(
        [curves[:, indices, :].reshape(len(table), -1), curves.mean(axis=1)], axis=1
    )
    if features.shape[1] != 216 or not np.isfinite(features).all():
        raise AssertionError("depth-matched gate feature contract changed")
    return features, {
        "feature_dimension": 216,
        "selected_depth_row_indices_zero_based": indices.tolist(),
        "selected_depths": table[0, indices, 0].astype(int).tolist(),
        "description": (
            "27 coordinates at seven unique g=2..16 log-spaced rows plus the across-depth mean"
        ),
    }


def require_azure_execution_target(
    compute_target: str,
    *,
    os_name: str | None = None,
    hostname: str | None = None,
) -> dict:
    actual_os = os.name if os_name is None else str(os_name)
    actual_host = socket.gethostname() if hostname is None else str(hostname)
    if compute_target != "azure":
        raise RuntimeError("full empirical scoring requires --compute-target azure")
    if actual_os != "posix" or actual_host != "trading-linux-az":
        raise RuntimeError(
            "full empirical scoring is bound to POSIX host trading-linux-az; "
            f"observed os={actual_os!r}, host={actual_host!r}"
        )
    return {
        "compute_target": compute_target,
        "os_name": actual_os,
        "hostname": actual_host,
        "verified_azure_host": True,
    }


def _z_audit(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    return {
        "rms_z": float(np.sqrt(np.mean(values ** 2))),
        "max_abs_z": float(np.max(np.abs(values))),
        "coordinates_abs_z_gt_3": int(np.sum(np.abs(values) > 3)),
        "coordinates_abs_z_gt_5": int(np.sum(np.abs(values) > 5)),
        "z_sha256_float64": _sha256_array(values.astype("<f8", copy=False)),
    }


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman_correlation(
    first: Sequence[float], second: Sequence[float]
) -> float | None:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman correlation requires equal nontrivial vectors")
    first_rank = _average_ranks(first)
    second_rank = _average_ranks(second)
    if np.std(first_rank) == 0 or np.std(second_rank) == 0:
        return None
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _candidate_concordance(records: Sequence[dict], representation: str) -> dict:
    calls = [record["direction"][representation]["call"] for record in records]
    return {
        "candidate_direction": EXPECTED_DIRECTION,
        "C_calls": int(sum(call == EXPECTED_DIRECTION for call in calls)),
        "views": len(calls),
        "fraction": float(np.mean(np.asarray(calls) == EXPECTED_DIRECTION)),
        "interpretation": (
            "descriptive agreement with a management/history-motivated candidate, not formal accuracy"
        ),
    }


def _descriptive_panel_accounting(records: Sequence[Mapping[str, object]]) -> dict:
    primary = [record for record in records if record["contract_role"] == "primary"]
    pa_primary = [
        record for record in primary
        if record["dataset"] == "pennsylvania_white_2018"
    ]
    ns_primary = [
        record for record in primary
        if record["dataset"] == "nova_scotia_lehnert_2020"
    ]
    ns_no_record = [
        record for record in records
        if record["dataset"] == "nova_scotia_lehnert_2020"
        and not bool(record["published_comparator"]["recorded_stocking_exposure"])
    ]
    accounting = {
        "distinct_named_primary_target_or_river_views": len({
            record["biological_system_id"] for record in primary
        }),
        "pennsylvania_targets": len({record["panel_id"] for record in pa_primary}),
        "nova_scotia_recorded_stocking_river_systems": len({
            record["biological_system_id"] for record in ns_primary
        }),
        "nova_scotia_no_recorded_stocking_contrasts": len({
            record["panel_id"] for record in ns_no_record
        }),
        "nova_scotia_no_recorded_stocking_record_views": len(ns_no_record),
        "nova_scotia_no_recorded_stocking_named_systems": len({
            record["biological_system_id"] for record in ns_no_record
        }),
        "guardrail": (
            "11 is not 11 independent validation units and is not an accuracy denominator; "
            "subsamples, no-record panels, and strict-locus reruns are correlated views"
        ),
    }
    observed = tuple(accounting[key] for key in (
        "distinct_named_primary_target_or_river_views",
        "pennsylvania_targets",
        "nova_scotia_recorded_stocking_river_systems",
        "nova_scotia_no_recorded_stocking_contrasts",
        "nova_scotia_no_recorded_stocking_record_views",
        "nova_scotia_no_recorded_stocking_named_systems",
    ))
    if observed != (11, 3, 8, 3, 6, 2):
        raise AssertionError(f"descriptive panel accounting changed: {observed}")
    return accounting


def analyze_records(
    records: Sequence[dict],
    canonical_root: Path,
    *,
    compute_state: Path | None,
) -> dict:
    if compute_state is not None:
        structured.compute_gate(compute_state)
    canonical = structured.load_canonical(canonical_root, max_depth=16)
    if canonical["audit"]["array_contracts"] != CANONICAL_ARRAY_CONTRACTS:
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
    external = np.stack([record["curve"] for record in records]).astype(float)

    representation_summaries = {}
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
                "feature_shift": _z_audit(z[index]),
            }
        primary = [record for record in records if record["contract_role"] == "primary"]
        sensitivities = [record for record in records if record["contract_role"] != "primary"]
        ns_primary = [record for record in primary if record["dataset"] == "nova_scotia_lehnert_2020"]
        ns_scores = [record["direction"][name]["scores"]["C"] for record in ns_primary]
        ns_deltas = [record["published_comparator"]["delta_introgression_total"] for record in ns_primary]
        rho = spearman_correlation(ns_scores, ns_deltas)
        representation_summaries[name] = {
            "status": "target-blind fixed canonical C=1; raw_all is primary",
            "feature_dimension": int(train.shape[1]),
            "descriptive_primary_candidate_concordance": _candidate_concordance(primary, name),
            "descriptive_primary_by_dataset": {
                dataset: _candidate_concordance(
                    [record for record in primary if record["dataset"] == dataset], name
                )
                for dataset in ("pennsylvania_white_2018", "nova_scotia_lehnert_2020")
            },
            "correlated_sensitivity_concordance": _candidate_concordance(sensitivities, name),
            "nova_scotia_C_score_vs_same_locus_published_Q_delta_spearman": rho,
            "correlation_guardrail": (
                "outcome-selected P1/P2 and same-marker Q values make this descriptive and circular"
            ),
            "external_rms_z": {
                "median": float(np.median(np.sqrt(np.mean(z ** 2, axis=1)))),
                "p95": float(np.quantile(np.sqrt(np.mean(z ** 2, axis=1)), 0.95)),
                "p95_max_abs": float(np.quantile(np.max(np.abs(z), axis=1), 0.95)),
            },
        }
        direction_models[name] = stdbench._model_payload(
            scaler, model, feature_columns=structured.representation_columns(name)
        )

    gate_train, gate_contract = depth_matched_gate_features(table)
    gate_external, external_gate_contract = depth_matched_gate_features(external)
    if gate_contract != external_gate_contract:
        raise AssertionError("canonical/external depth-matched gate contracts differ")
    gate_target = (positive & (rates >= structured.APPRECIABLE)).astype(int)
    gate_scaler, gate_model = structured._fit_model(gate_train, gate_target, C=1.0)
    gate_z = gate_scaler.transform(gate_external)
    positive_index = int(np.flatnonzero(gate_model.classes_ == 1)[0])
    gate_score = gate_model.predict_proba(gate_z)[:, positive_index]
    for index, record in enumerate(records):
        record["depth_matched_gate"] = {
            "appreciable_score": float(gate_score[index]),
            "called_at_0_5": bool(gate_score[index] >= 0.5),
            "score_interpretation": (
                "uncalibrated g=2..16 OOD score; not the frozen full-g=2..199 gate and not a posterior"
            ),
            "feature_shift": _z_audit(gate_z[index]),
        }
        raw_rms = record["direction"]["raw_all"]["feature_shift"]["rms_z"]
        gate_rms = record["depth_matched_gate"]["feature_shift"]["rms_z"]
        record["adjudication"] = {
            "direction_call_accepted": False,
            "formal_direction_accuracy_eligible": False,
            "gate_accuracy_eligible": False,
            "severe_OOD_heuristic": bool(max(raw_rms, gate_rms) > 10),
            "severe_OOD_threshold": "max(raw_all direction RMS-z, depth-matched gate RMS-z) > 10",
            "decision_basis": (
                "same-locus literature comparator, unresolved three-population topology, and "
                "continuous-migration target mismatch prevent acceptance regardless of OOD score"
            ),
        }

    prediction_ledger = []
    for record in records:
        prediction_ledger.append({
            "panel_view_id": record["panel_view_id"],
            "dataset": record["dataset"],
            "biological_system_id": record["biological_system_id"],
            "contract_role": record["contract_role"],
            "expected_candidate_direction": record["expected_candidate_direction"],
            "raw_all_call": record["direction"]["raw_all"]["call"],
            "raw_all_C_score": record["direction"]["raw_all"]["scores"]["C"],
            "raw_all_rms_z": record["direction"]["raw_all"]["feature_shift"]["rms_z"],
            "depth_matched_gate_score": record["depth_matched_gate"]["appreciable_score"],
            "depth_matched_gate_rms_z": record["depth_matched_gate"]["feature_shift"]["rms_z"],
            "severe_OOD_heuristic": record["adjudication"]["severe_OOD_heuristic"],
            "direction_call_accepted": False,
        })
    for record in records:
        record["curve"] = np.asarray(record["curve"], dtype=float).tolist()
    return {
        "records": list(records),
        "prediction_ledger": prediction_ledger,
        "representations": representation_summaries,
        "direction_models": direction_models,
        "depth_matched_gate": {
            "contract": gate_contract,
            "training_target": "canonical A/B/C rate >=2.5e-4 versus weak A/B/C plus D",
            "training_target_counts": {
                "appreciable": int(gate_target.sum()),
                "other": int((gate_target == 0).sum()),
            },
            "model": stdbench._model_payload(gate_scaler, gate_model),
            "guardrail": (
                "this g=2..16, 216-D refit is used by external panels but is not the frozen "
                "full-depth 243-D gate; no brook-trout row has an independent gate truth"
            ),
        },
        "canonical_source_audit": canonical["audit"],
        "descriptive_panel_accounting": _descriptive_panel_accounting(records),
        "guardrail": (
            "candidate-direction concordance is not accuracy: published labels reuse the scored "
            "microsatellites, stocking records do not prove realized introgression, and topology "
            "and continuous-migration assumptions are unresolved"
        ),
    }


def configuration(revision: dict) -> dict:
    oyster_source = Path(oyster.__file__).resolve()
    transcription_payload = {
        "targets": PA_TARGETS,
        "reference_sites": list(PA_REFERENCE_SITES),
        "stocking_context": PA_STOCKING_CONTEXT,
    }
    transcription_sha256 = hashlib.sha256(
        _canonical_json(transcription_payload)
    ).hexdigest()
    if transcription_sha256 != PA_PUBLISHED_COMPARATOR_PROVENANCE[
        "manual_transcription_contract_sha256"
    ]:
        raise AssertionError("Pennsylvania manual paper transcription changed")
    return {
        "schema_version": SCHEMA_VERSION,
        "source_revision": {
            key: revision.get(key)
            for key in (
                "commit", "script_sha256", "head_script_sha256", "head_blob_oid",
                "worktree_blob_oid", "tracked_diff_sha256", "tracked_dirty_at_snapshot",
            )
        },
        "helper_source": {
            "path": str(oyster_source),
            "sha256": structured.sha256_file(oyster_source),
            "purpose": "formula-free bounded OOXML worksheet reader",
        },
        "sources": SOURCE_CONTRACTS,
        "pennsylvania": {
            "loci": list(PA_LOCI),
            "reference_sites": list(PA_REFERENCE_SITES),
            "reference_rows_per_site": 22,
            "hatchery_strains": list(PA_HATCHERY_STRAINS),
            "hatchery_rows_per_strain": 50,
            "targets": PA_TARGETS,
            "stocking_context": PA_STOCKING_CONTEXT,
            "direction_evidence": PA_DIRECTION_EVIDENCE,
            "published_comparator_provenance": PA_PUBLISHED_COMPARATOR_PROVENANCE,
            "published_transcription_payload_sha256": transcription_sha256,
            "source_paper_locus_mismatch": "methods SfoC-79 versus deposited C115",
        },
        "nova_scotia": {
            "primary_panels": [asdict(panel) for panel in NS_PANELS],
            "correlated_sensitivity_panels": [
                asdict(panel) for panel in NS_SENSITIVITY_PANELS
            ],
            "shared_standard_loci": 90,
            "shared_within_population_polymorphic_loci": 62,
            "minimum_called_copies_per_population": MIN_CALLED_COPIES,
            "raw_source_name_collision_guardrail": (
                "raw P3 code FM is the n=49 domestic-strain section; Introgression_FM is the "
                "paper's centered five-group Fraser's Mills source"
            ),
        },
        "padze": {
            "depths": PRIMARY_DEPTHS.tolist(),
            "moments": list(stdbench.MOMENTS),
            "pihat_sizes": [2],
            "bias_corrected": True,
            "allele_contract": "all numeric microsatellite alleles retained; no biallelic recoding",
        },
        "evaluation": {
            "candidate_direction": EXPECTED_DIRECTION,
            "candidate_forward_edge": "P3->P2",
            "representations": list(REPRESENTATIONS),
            "primary_representation": "raw_all",
            "C": 1.0,
            "formal_accuracy_eligible": False,
            "direction_calls_accepted": False,
            "gate_accuracy_eligible": False,
        },
        "canonical_training_contract": {
            "replicates": 3200,
            "label_counts": {"A": 900, "B": 900, "C": 900, "D": 500},
            "array_contracts": CANONICAL_ARRAY_CONTRACTS,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pa-workbook", type=Path, required=True)
    parser.add_argument("--ns-genepop", type=Path, required=True)
    parser.add_argument("--ns-population-names", type=Path, required=True)
    parser.add_argument("--ns-introgression", type=Path, required=True)
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
        "pa_workbook": verify_source(args.pa_workbook, "pa_workbook"),
        "ns_genepop": verify_source(args.ns_genepop, "ns_genepop"),
        "ns_population_names": verify_source(args.ns_population_names, "ns_population_names"),
        "ns_introgression": verify_source(args.ns_introgression, "ns_introgression"),
    }
    pa_loci, pa_individuals, pa_parser_audit = parse_pa_workbook(args.pa_workbook)
    population_rows, population_audit = parse_ns_population_names(args.ns_population_names)
    population_codes = [row["Code"].strip() for row in population_rows]
    ns_loci, ns_populations, genepop_audit = parse_genepop(args.ns_genepop, population_codes)
    introgression, introgression_audit = parse_ns_introgression(args.ns_introgression)
    if set(introgression) != set(population_codes) - {"Derby", "FL", "FM", "MR", "Seatrout", "TL"}:
        raise AssertionError("Nova Scotia wild/hatchery population partition changed")

    pa_panels, pa_selection_audit = make_pa_panels(pa_individuals)
    ns_panels, ns_selection_audit = make_ns_panels(ns_populations, introgression)
    config = configuration(revision)
    config_sha256 = hashlib.sha256(_canonical_json(config)).hexdigest()
    records, locus_selection_audit = build_records(
        pa_loci,
        pa_panels,
        ns_loci,
        ns_panels,
        compute_state=args.compute_state,
    )
    pre_analysis_gate = structured.compute_gate(args.compute_state)
    analysis = analyze_records(records, args.canonical_root, compute_state=args.compute_state)

    final_revision = structured.git_revision(script=Path(__file__))
    structured.require_revision_unchanged(revision, final_revision)
    final_source_recheck = {
        "pa_workbook": verify_source(args.pa_workbook, "pa_workbook"),
        "ns_genepop": verify_source(args.ns_genepop, "ns_genepop"),
        "ns_population_names": verify_source(args.ns_population_names, "ns_population_names"),
        "ns_introgression": verify_source(args.ns_introgression, "ns_introgression"),
    }
    runtime = structured.runtime_audit(priority)
    runtime["packages"]["padze"] = importlib_metadata.version("padze")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "management_history_and_same_marker_transfer_diagnostic_not_accuracy",
        "git": revision,
        "final_git_recheck": final_revision,
        "initial_compute_gate": initial_gate,
        "pre_analysis_compute_gate": pre_analysis_gate,
        "runtime": runtime,
        "execution_target": execution_target,
        "configuration": config,
        "configuration_sha256": config_sha256,
        "source_audits": source_audits,
        "source_final_recheck": final_source_recheck,
        "parser_audits": {
            "pennsylvania": pa_parser_audit,
            "nova_scotia_population_names": population_audit,
            "nova_scotia_genepop": genepop_audit,
            "nova_scotia_introgression": introgression_audit,
        },
        "selection_audits": {
            "pennsylvania": pa_selection_audit,
            "nova_scotia": ns_selection_audit,
            "loci": locus_selection_audit,
        },
        "analysis": analysis,
    }
    with structured.SingleWriterLease(
        args.result_dir, ".brook_trout_microsatellite_result.lock"
    ):
        output = args.result_dir / "results.json"
        output_audit = structured.write_json_atomic(output, result, indent=2)
    primary = analysis["representations"]["raw_all"][
        "descriptive_primary_candidate_concordance"
    ]
    print(json.dumps({
        "output": output_audit,
        "configuration_sha256": config_sha256,
        "raw_all_descriptive_candidate_C_calls": primary["C_calls"],
        "raw_all_descriptive_primary_views": primary["views"],
        "formal_accuracy_eligible": False,
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
