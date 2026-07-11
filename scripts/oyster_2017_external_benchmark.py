#!/usr/bin/env python3
"""Run a guarded Sydney rock oyster candidate-null crossing stress test.

Thompson et al. (2017) reported no detectable sustained introgression from the
selectively bred B2 line into wild Sydney rock oysters at two Georges River
sites.  The released workbook contains the same SNPs used to obtain that
conclusion. The public workbook derives from the same SNP-discovery data, but
already omits three Q B2-labelled oysters removed after DAPC on the paper's
1,189 neutral loci and does not identify that exact neutral subset. The source
result is therefore a power-limited, outcome-derived candidate null—not proof
of zero migration and not a gold specificity label.

This runner uses W as the primary comparison and Q as a post-hoc-cleaned
sensitivity.  Standard and within-population-polymorphic views share the exact
same loci across sites.  Every learned output remains an uncalibrated natural-
data extrapolation; severe OOD panels abstain, and no panel enters accuracy or
specificity counts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import posixpath
from pathlib import Path
import re
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import numpy as np


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from additional_external_benchmarks import add_gate_score, simulation_gate_head
from external_benchmarks import (
    MANIFEST_DIR,
    MAX_DEPTH,
    REPO,
    git_revision,
    open_text,
    prepare_vcf,
    read_manifest,
    score_panel,
    set_below_normal_priority,
    sha256_file,
    simulation_direction_head,
    subset_prepared_vcf,
)
from harpagifer_external_benchmark import frequency_geometry as _frequency_geometry
import tinkerbird_external_benchmark as runtime_helpers


DEFAULT_CACHE = REPO / "data" / "real" / "oyster_2017_external_benchmark"
DEFAULT_RESULTS = REPO / "results" / "oyster_2017_external_benchmark_2026_07_11"
DEFAULT_CAP = 1_200
PANEL_RECORD = MANIFEST_DIR / "oyster_2017" / "panel_populations.tsv"
SOURCES_RECORD = MANIFEST_DIR / "oyster_2017" / "sources.json"
PANEL_RECORD_SHA256 = "de9190e7abb65647d92e8f9064d0c63d549c641561d85282bf95b072fcefd1b2"

DRYAD_RECORD = "https://datadryad.org/dataset/doi:10.5061/dryad.32q80"
DRYAD_ARCHIVE = "https://datadryad.org/api/v2/versions/3411/download"
DRYAD_FILE_DOWNLOAD = "https://datadryad.org/api/v2/files/21290/download"
ARCHIVE = {
    "key": "doi_10_5061_dryad_32q80__version_3411.zip",
    "digest_policy": "not_pinned_generated_ZIP_wrapper_validate_inner_workbook_only",
}
FILE = {
    "id": 21_290,
    "key": "SNP_data_M12109.xlsx",
    "archive_member": "SNP_data_M12109.xlsx",
    "download": DRYAD_FILE_DOWNLOAD,
    "bytes": 729_706,
    "md5": "572a079597af8530b15aaffd07325b55",
    "sha256": "e0f6983f1a15c9d7a1aeb4a76e220f24b1d4c766600502413b2cb5c4fdde8029",
}
SOURCE_CONTRACT = {
    "sheet_name": "SNP_data_M12109",
    "dimension": "A1:CNJ93",
    "samples": 90,
    "loci": 1_200,
    "genotype_pairs": 108_000,
    "missing_genotype_pairs": 3_647,
    "ordered_sample_sha256": "1cb297a1bcebd42d09e8ecbe8b1a9805c8cc2e735237248d0475b35de69ea034",
    "sample_population_tsv_sha256": "3673f332f880de07355c5d3a1c0612140f068325339bbc85dccbc67480e0d03f",
    "ordered_locus_sha256": "48f1c20c0bb01bad52330eae6bf5775ffd3cc2bf74e9d7eae374165546614632",
    "genalex_semantic_sha256": "9f58cf09e53c8353ea5d1ec272b0af7ac7ae05af257b12b6acb014c392be825e",
    "missing_original_sample_ids": ["28", "29", "31"],
    "samples_below_0_95_call_rate": ["20", "24", "36", "38", "50", "57", "66", "70", "84"],
    "loci_below_0_95_call_rate": 387,
}
DERIVED_SOURCE_CONTRACT = {
    "bytes": 470_039,
    "sha256": "7d978cb745008e880a023f4c6347c54d50abd9c19cfb5daeba1f964fc829d756",
    "ordered_locus_id_sha256": SOURCE_CONTRACT["ordered_locus_sha256"],
    "samples": 90,
    "loci": 1_200,
}
WORKBOOK_POPULATION_COUNTS = {
    "PSHB2": 11,
    "PSHWT": 10,
    "QBB2": 9,
    "QBOC": 12,
    "QBWC": 12,
    "WBB2": 12,
    "WBOC": 12,
    "WBWC": 12,
}
PANEL_SPECS = {
    "W": {
        "population_order": ("WWC", "WOC", "WB2"),
        "workbook_populations": ("WBWC", "WBOC", "WBB2"),
        "population_counts": {"WWC": 12, "WOC": 12, "WB2": 12},
        "role": "primary_cleaner_near_null",
        "same_data_excluded_ids": [],
    },
    "Q": {
        "population_order": ("QWC", "QOC", "QB2"),
        "workbook_populations": ("QBWC", "QBOC", "QBB2"),
        "population_counts": {"QWC": 12, "QOC": 12, "QB2": 9},
        "role": "secondary_posthoc_cleaned_sensitivity",
        "same_data_excluded_ids": ["28", "29", "31"],
    },
}
EXPECTED_FILTERS = {
    "standard_contract": {
        "loci": 1_101,
        "ordered_locus_sha256": "edafecad96e334e40dfff485bb99ba6e1354a0d7841af7257851673c286a75ed",
        "ordered_locus_id_sha256": "f55a7c9d365de973394eaae962a165b09632b0bfe8eba85d5d09774e36756629",
        "insufficient_called_copies": 82,
        "not_polymorphic_in_every_panel": 17,
    },
    "within_population_polymorphism": {
        "loci": 589,
        "ordered_locus_sha256": "df6659235f9030e4ecec84c36be610239e0912d8986daf2f053144892d71ec26",
        "ordered_locus_id_sha256": "d79c788ba837504fa709ba1653157f3b03155b2c1cc165ddc5ee038028e6c810",
        "insufficient_called_copies": 82,
        "not_polymorphic_in_every_panel": 529,
    },
}

_MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_DOC_REL_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PKG_REL_NS = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_CELL_REFERENCE = re.compile(r"([A-Z]+)([1-9][0-9]*)\Z")


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DNNaic-audit/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_workbook(path: Path) -> dict:
    observed = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "md5": _md5_file(path),
        "sha256": sha256_file(path),
    }
    for key in ("bytes", "md5", "sha256"):
        if observed[key] != FILE[key]:
            raise AssertionError(f"oyster workbook {key} contract changed")
    return observed


def _extract_workbook(archive: Path, output: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = [name for name in bundle.namelist() if Path(name).name == FILE["archive_member"]]
        if members != [FILE["archive_member"]]:
            raise AssertionError(f"unexpected Dryad archive members for oyster workbook: {members}")
        if bundle.getinfo(members[0]).file_size != FILE["bytes"]:
            raise AssertionError("Dryad archive workbook byte count changed")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".part")
        with bundle.open(members[0]) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        temporary.replace(output)


def ensure_source(workbook: Path, archive: Path, download_missing: bool) -> dict:
    if not workbook.exists():
        if archive.exists():
            _extract_workbook(archive, workbook)
        elif not download_missing:
            raise FileNotFoundError(workbook)
        else:
            try:
                _download(DRYAD_FILE_DOWNLOAD, workbook)
                _verify_workbook(workbook)
            except (urllib.error.URLError, TimeoutError, AssertionError):
                workbook.unlink(missing_ok=True)
                _download(DRYAD_ARCHIVE, archive)
                _extract_workbook(archive, workbook)
    return {
        "workbook": _verify_workbook(workbook),
        "retrieval_contract": (
            "only the canonical inner workbook is recorded; acquisition route and generated ZIP-wrapper "
            "presence, bytes, timestamp, size, and digest are deliberately excluded from result identity"
        ),
    }


def _column_index(letters: str) -> int:
    value = 0
    if not letters or not letters.isalpha() or letters != letters.upper():
        raise ValueError(f"invalid OOXML column reference: {letters!r}")
    for letter in letters:
        value = value * 26 + ord(letter) - ord("A") + 1
    return value - 1


def _parse_cell_reference(reference: str) -> tuple[int, int]:
    match = _CELL_REFERENCE.fullmatch(reference)
    if match is None:
        raise ValueError(f"invalid OOXML cell reference: {reference!r}")
    return int(match.group(2)) - 1, _column_index(match.group(1))


def _parse_dimension(reference: str) -> tuple[int, int]:
    fields = reference.split(":")
    if len(fields) != 2 or fields[0] != "A1":
        raise ValueError(f"unsupported OOXML worksheet dimension: {reference!r}")
    row, column = _parse_cell_reference(fields[1])
    return row + 1, column + 1


def _shared_strings(bundle: zipfile.ZipFile) -> list[str]:
    member = "xl/sharedStrings.xml"
    if member not in bundle.namelist():
        return []
    root = ET.fromstring(bundle.read(member))
    return [
        "".join(node.text or "" for node in item.iter(_MAIN_NS + "t"))
        for item in root.findall(_MAIN_NS + "si")
    ]


def _resolve_sheet_member(bundle: zipfile.ZipFile, sheet_name: str) -> str:
    workbook_member = "xl/workbook.xml"
    relations_member = "xl/_rels/workbook.xml.rels"
    workbook = ET.fromstring(bundle.read(workbook_member))
    sheets = [
        sheet
        for sheet in workbook.findall(".//" + _MAIN_NS + "sheet")
        if sheet.attrib.get("name") == sheet_name
    ]
    if len(sheets) != 1:
        raise AssertionError(f"expected exactly one {sheet_name!r} worksheet")
    relation_id = sheets[0].attrib.get(_DOC_REL_NS + "id")
    relations = ET.fromstring(bundle.read(relations_member))
    matches = [
        relation
        for relation in relations.findall(_PKG_REL_NS + "Relationship")
        if relation.attrib.get("Id") == relation_id
    ]
    if len(matches) != 1 or matches[0].attrib.get("TargetMode") == "External":
        raise ValueError("unsafe or unresolved OOXML worksheet relationship")
    target = matches[0].attrib.get("Target", "")
    if target.startswith(("/", "\\")) or "\\" in target:
        raise ValueError("unsafe OOXML worksheet target")
    member = posixpath.normpath(posixpath.join("xl", target))
    if member.startswith("../") or not member.startswith("xl/") or member not in bundle.namelist():
        raise ValueError("unsafe or absent OOXML worksheet target")
    return member


def read_xlsx_sheet(path: Path, sheet_name: str) -> dict:
    """Read a bounded OOXML worksheet without evaluating formulas."""
    with zipfile.ZipFile(path) as bundle:
        shared = _shared_strings(bundle)
        member = _resolve_sheet_member(bundle, sheet_name)
        root = ET.fromstring(bundle.read(member))
    dimension_nodes = root.findall(_MAIN_NS + "dimension")
    if len(dimension_nodes) != 1:
        raise ValueError("worksheet must have exactly one declared dimension")
    dimension = dimension_nodes[0].attrib.get("ref", "")
    n_rows, n_columns = _parse_dimension(dimension)
    cells: dict[tuple[int, int], str] = {}
    for cell in root.findall(".//" + _MAIN_NS + "c"):
        if cell.find(_MAIN_NS + "f") is not None:
            raise ValueError("formulas are not permitted in the source workbook")
        coordinate = _parse_cell_reference(cell.attrib.get("r", ""))
        if coordinate in cells:
            raise ValueError(f"duplicate OOXML cell coordinate: {cell.attrib.get('r')}")
        if coordinate[0] >= n_rows or coordinate[1] >= n_columns:
            raise ValueError("OOXML cell lies outside declared worksheet dimension")
        cell_type = cell.attrib.get("t")
        value_node = cell.find(_MAIN_NS + "v")
        raw = "" if value_node is None else value_node.text or ""
        if cell_type == "s":
            try:
                value = shared[int(raw)]
            except (ValueError, IndexError) as exc:
                raise ValueError("invalid OOXML shared-string index") from exc
        elif cell_type == "inlineStr":
            inline = cell.find(_MAIN_NS + "is")
            value = "" if inline is None else "".join(
                node.text or "" for node in inline.iter(_MAIN_NS + "t")
            )
        elif cell_type in (None, "n", "str"):
            value = raw
        else:
            raise ValueError(f"unsupported OOXML cell type: {cell_type!r}")
        cells[coordinate] = value
    merged = root.find(_MAIN_NS + "mergeCells")
    return {
        "member": member,
        "dimension": dimension,
        "n_rows": n_rows,
        "n_columns": n_columns,
        "cells": cells,
        "merged_cell_ranges": 0 if merged is None else len(list(merged)),
    }


def _as_int(value: str, context: str) -> int:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(f"{context}: expected an integer, got {value!r}") from exc
    integer = int(number)
    if not np.isfinite(number) or number != integer:
        raise ValueError(f"{context}: expected an exact integer, got {value!r}")
    return integer


def _hash_lines(lines: list[str]) -> str:
    return hashlib.sha256("".join(f"{line}\n" for line in lines).encode("utf-8")).hexdigest()


def decode_genalex_pair(first: int, second: int) -> str:
    if first == 0 or second == 0:
        if (first, second) == (0, 0):
            return "./."
        raise ValueError("partial GenAlEx missing genotype is not permitted")
    if first not in (1, 2) or second not in (1, 2):
        raise ValueError("GenAlEx allele code must be 0, 1, or 2")
    return f"{first - 1}/{second - 1}" if first == second else "0/1"


def parse_genalex_workbook(path: Path) -> dict:
    sheet = read_xlsx_sheet(path, SOURCE_CONTRACT["sheet_name"])
    if sheet["dimension"] != SOURCE_CONTRACT["dimension"]:
        raise AssertionError("oyster worksheet dimension changed")
    if (sheet["n_rows"], sheet["n_columns"]) != (93, 2_402):
        raise AssertionError("oyster worksheet shape changed")
    if sheet["merged_cell_ranges"] != 0:
        raise AssertionError("oyster worksheet unexpectedly contains merged cells")
    cells = sheet["cells"]
    get = lambda row, column: cells.get((row, column), "")
    if [_as_int(get(0, index), f"header {index}") for index in range(3)] != [1_200, 90, 8]:
        raise AssertionError("GenAlEx leading header changed")
    if [_as_int(get(0, index), f"population count {index}") for index in range(3, 11)] != list(
        WORKBOOK_POPULATION_COUNTS.values()
    ):
        raise AssertionError("GenAlEx population-count header changed")
    if get(1, 0) != "Codominant data template" or get(2, 0) != "Sample" or get(2, 1) != "Pop":
        raise AssertionError("GenAlEx template header changed")
    header_populations = [get(1, index) for index in range(3, 11)]
    if header_populations != list(WORKBOOK_POPULATION_COUNTS):
        raise AssertionError("GenAlEx population order changed")

    loci = []
    for index in range(SOURCE_CONTRACT["loci"]):
        first = get(2, 2 + 2 * index)
        second = get(2, 3 + 2 * index)
        if first != f"Locus{index + 1}" or second != "":
            raise AssertionError(f"unexpected GenAlEx locus header at pair {index + 1}")
        loci.append(first)

    samples: list[str] = []
    populations: list[str] = []
    genotypes: list[list[tuple[int, int]]] = []
    reversed_heterozygotes = 0
    pair_counts: Counter[str] = Counter()
    for row_index in range(3, 93):
        sample = str(_as_int(get(row_index, 0), f"sample row {row_index + 1}"))
        population = get(row_index, 1)
        if population not in WORKBOOK_POPULATION_COUNTS:
            raise AssertionError(f"unexpected workbook population {population!r}")
        row_genotypes = []
        for locus_index in range(SOURCE_CONTRACT["loci"]):
            first = _as_int(get(row_index, 2 + 2 * locus_index), "GenAlEx allele")
            second = _as_int(get(row_index, 3 + 2 * locus_index), "GenAlEx allele")
            decode_genalex_pair(first, second)
            reversed_heterozygotes += (first, second) == (2, 1)
            pair_counts[f"{first}/{second}"] += 1
            row_genotypes.append((first, second))
        samples.append(sample)
        populations.append(population)
        genotypes.append(row_genotypes)
    if len(samples) != len(set(samples)):
        raise AssertionError("oyster workbook sample IDs are not unique")

    population_counts = dict(Counter(populations))
    missing_by_sample = [sum(pair == (0, 0) for pair in row) for row in genotypes]
    missing_by_locus = [
        sum(genotypes[sample][locus] == (0, 0) for sample in range(len(samples)))
        for locus in range(len(loci))
    ]
    globally_polymorphic = sum(
        len(
            {
                allele
                for sample in range(len(samples))
                for allele in genotypes[sample][locus]
                if allele in (1, 2)
            }
        )
        == 2
        for locus in range(len(loci))
    )
    sample_population_tsv = "sample\tpopulation\n" + "".join(
        f"{sample}\t{population}\n" for sample, population in zip(samples, populations)
    )
    semantic_rows = "".join(
        f"{sample}\t{population}\t"
        + "\t".join(str(allele) for pair in row for allele in pair)
        + "\n"
        for sample, population, row in zip(samples, populations, genotypes)
    )
    population_called_copy_counts = {}
    for population in WORKBOOK_POPULATION_COUNTS:
        sample_indexes = [index for index, label in enumerate(populations) if label == population]
        values = [
            sum(
                2
                for sample_index in sample_indexes
                if genotypes[sample_index][locus_index] != (0, 0)
            )
            for locus_index in range(len(loci))
        ]
        population_called_copy_counts[population] = {
            "individuals": len(sample_indexes),
            "minimum": min(values),
            "maximum": max(values),
            "mean": float(np.mean(values)),
        }

    missing_ids = sorted(
        set(map(str, range(1, 94))) - set(samples), key=int
    )
    low_call_samples = [
        sample for sample, missing in zip(samples, missing_by_sample) if missing > 0.05 * len(loci)
    ]
    audit = {
        "sheet_name": SOURCE_CONTRACT["sheet_name"],
        "sheet_member": sheet["member"],
        "dimension": sheet["dimension"],
        "samples": len(samples),
        "loci": len(loci),
        "genotype_pairs": len(samples) * len(loci),
        "missing_genotype_pairs": sum(missing_by_sample),
        "overall_call_rate": 1.0 - sum(missing_by_sample) / (len(samples) * len(loci)),
        "genotype_pair_counts": dict(sorted(pair_counts.items())),
        "partial_missing_pairs": 0,
        "reversed_heterozygote_pairs_2_slash_1": reversed_heterozygotes,
        "population_counts": population_counts,
        "population_called_copy_counts": population_called_copy_counts,
        "ordered_sample_sha256": _hash_lines(samples),
        "sample_population_tsv_sha256": hashlib.sha256(sample_population_tsv.encode("utf-8")).hexdigest(),
        "ordered_locus_sha256": _hash_lines(loci),
        "genalex_semantic_sha256": hashlib.sha256(semantic_rows.encode("utf-8")).hexdigest(),
        "missing_original_sample_ids": missing_ids,
        "sample_missingness": {
            "minimum_missing_pairs": min(missing_by_sample),
            "maximum_missing_pairs": max(missing_by_sample),
            "maximum_missing_fraction": max(missing_by_sample) / len(loci),
            "maximum_missing_sample": samples[int(np.argmax(missing_by_sample))],
            "samples_below_0_95_call_rate": low_call_samples,
        },
        "locus_missingness": {
            "minimum_called_individuals": len(samples) - max(missing_by_locus),
            "minimum_call_rate": (len(samples) - max(missing_by_locus)) / len(samples),
            "loci_below_0_95_call_rate": sum(missing > 0.05 * len(samples) for missing in missing_by_locus),
        },
        "globally_polymorphic_loci": globally_polymorphic,
        "paper_filter_discrepancy": (
            "the paper states individuals and loci were retained at >=95% call rate, but the final "
            "released workbook has 9 individuals and 387 loci below 95%; all individuals remain above 94%"
        ),
        "locus_identity_guardrail": (
            "the workbook labels loci only as Locus1..Locus1200 and does not identify which 11 were "
            "consensus selection outliers; the paper's 1,189 neutral-locus subset cannot be reconstructed"
        ),
    }
    exact = {
        "samples": audit["samples"],
        "loci": audit["loci"],
        "genotype_pairs": audit["genotype_pairs"],
        "missing_genotype_pairs": audit["missing_genotype_pairs"],
        "ordered_sample_sha256": audit["ordered_sample_sha256"],
        "sample_population_tsv_sha256": audit["sample_population_tsv_sha256"],
        "ordered_locus_sha256": audit["ordered_locus_sha256"],
        "genalex_semantic_sha256": audit["genalex_semantic_sha256"],
        "missing_original_sample_ids": audit["missing_original_sample_ids"],
        "samples_below_0_95_call_rate": audit["sample_missingness"]["samples_below_0_95_call_rate"],
        "loci_below_0_95_call_rate": audit["locus_missingness"]["loci_below_0_95_call_rate"],
    }
    if exact != {key: SOURCE_CONTRACT[key] for key in exact}:
        raise AssertionError("oyster GenAlEx semantic contract changed")
    if population_counts != WORKBOOK_POPULATION_COUNTS:
        raise AssertionError("oyster workbook population counts changed")
    if reversed_heterozygotes != 0 or globally_polymorphic != len(loci):
        raise AssertionError("oyster workbook allele-orientation/polymorphism contract changed")
    return {
        "samples": samples,
        "populations": populations,
        "loci": loci,
        "genotypes": genotypes,
        "audit": audit,
    }


def ordered_locus_id_sha256(vcf: Path) -> str:
    identifiers = []
    with open_text(vcf) as handle:
        for line in handle:
            if not line.startswith("#") and line.strip():
                identifiers.append(line.split("\t", 3)[2])
    return _hash_lines(identifiers)


def materialize_source_vcf(workbook: dict, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##source=Dryad_10.5061/dryad.32q80_GenAlEx_conversion\n")
        handle.write("##dnnaic_coordinate_guardrail=CHROM_0_and_POS_are_synthetic_workbook_order_not_physical_coordinates\n")
        handle.write("##dnnaic_allele_guardrail=REF_A_and_ALT_C_are_placeholders_for_GenAlEx_codes_1_and_2_not_sequence_or_ancestry\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(workbook["samples"]) + "\n")
        for locus_index, locus in enumerate(workbook["loci"]):
            genotypes = [
                decode_genalex_pair(row[locus_index][0], row[locus_index][1])
                for row in workbook["genotypes"]
            ]
            handle.write(
                "\t".join(
                    ["0", str(locus_index + 1), locus, "A", "C", ".", "PASS", ".", "GT"]
                    + genotypes
                )
                + "\n"
            )
    return {
        "path": str(output),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
        "ordered_locus_id_sha256": ordered_locus_id_sha256(output),
        "samples": len(workbook["samples"]),
        "loci": len(workbook["loci"]),
        "coordinate_status": "synthetic_CHROM_0_and_POS_equal_workbook_locus_order",
        "allele_status": "synthetic_A_C_encoding_of_GenAlEx_1_2",
    }


def validate_derived_source_audit(audit: dict) -> None:
    observed = {key: audit[key] for key in DERIVED_SOURCE_CONTRACT}
    if observed != DERIVED_SOURCE_CONTRACT:
        raise AssertionError("deterministic oyster GenAlEx-to-VCF contract changed")


def read_panel_record(path: Path = PANEL_RECORD) -> list[dict[str, str]]:
    if sha256_file(path) != PANEL_RECORD_SHA256:
        raise AssertionError("oyster population-record byte contract changed")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if [row["workbook_population"] for row in rows] != list(WORKBOOK_POPULATION_COUNTS):
        raise AssertionError("oyster population-record order changed")
    if [int(row["expected_n"]) for row in rows] != list(WORKBOOK_POPULATION_COUNTS.values()):
        raise AssertionError("oyster population-record counts changed")
    crosswalk = {row["workbook_population"]: row["paper_population"] for row in rows}
    expected = {
        "PSHB2": "HB2",
        "PSHWT": "HC",
        "QBB2": "QB2",
        "QBOC": "QOC",
        "QBWC": "QWC",
        "WBB2": "WB2",
        "WBOC": "WOC",
        "WBWC": "WWC",
    }
    if crosswalk != expected:
        raise AssertionError("oyster workbook-to-paper population crosswalk changed")
    return rows


def materialize_manifests(workbook: dict, output_dir: Path) -> tuple[dict[str, Path], dict]:
    rows = read_panel_record()
    crosswalk = {row["workbook_population"]: row["paper_population"] for row in rows}
    selected = {
        panel: [
            (sample, crosswalk[population])
            for sample, population in zip(workbook["samples"], workbook["populations"])
            if population in spec["workbook_populations"]
        ]
        for panel, spec in PANEL_SPECS.items()
    }
    union_samples = {sample for pairs in selected.values() for sample, _ in pairs}
    selected["union"] = [
        (sample, crosswalk[population])
        for sample, population in zip(workbook["samples"], workbook["populations"])
        if sample in union_samples
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    audits = {}
    for panel, pairs in selected.items():
        path = output_dir / f"oyster.{panel}.tsv"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("sample\tpopulation\n")
            for sample, population in pairs:
                handle.write(f"{sample}\t{population}\n")
        mapping = read_manifest(path, require_three=panel != "union")
        counts = dict(Counter(mapping.values()))
        if panel in PANEL_SPECS and counts != PANEL_SPECS[panel]["population_counts"]:
            raise AssertionError(f"unexpected oyster {panel} manifest counts")
        if panel == "union" and len(mapping) != 69:
            raise AssertionError("unexpected oyster union manifest count")
        paths[panel] = path
        audits[panel] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "samples": len(mapping),
            "population_counts": counts,
            "ordered_sample_sha256": _hash_lines(list(mapping)),
        }
    excluded = [
        {"sample": sample, "workbook_population": population, "paper_population": crosswalk[population]}
        for sample, population in zip(workbook["samples"], workbook["populations"])
        if sample not in union_samples
    ]
    if len(excluded) != 21 or len(union_samples) + len(excluded) != SOURCE_CONTRACT["samples"]:
        raise AssertionError("oyster manifest accounting changed")
    audits["excluded_nonbenchmark_reference_cohorts"] = {
        "samples": len(excluded),
        "population_counts": dict(Counter(row["paper_population"] for row in excluded)),
        "reason": (
            "HB2 and HC are useful stock/reference cohorts but form only a two-group hatchery "
            "comparison and do not supply a defensible third population for the DNNaic topology"
        ),
    }
    return paths, audits


def oyster_frequency_geometry(vcf: Path, manifest: Path, population_order: tuple[str, str, str]) -> dict:
    result = _frequency_geometry(vcf, manifest, pop_order=population_order)
    result["iid_locus_bootstrap"]["method"] = "naive IID resampling of anonymous DArT loci"
    result["iid_locus_bootstrap"]["guardrail"] = (
        "fixed-sample conditional sensitivity only; the workbook supplies no chromosome, physical "
        "position, or linkage map, so independence is unverified and this is not chromosome-block uncertainty"
    )
    result["interpretation"] = (
        "sample-frequency geometry only. Projection is reference-flip invariant but is not bounded ancestry, "
        "migration, or temporal direction. The finite-called-copy f3 subtraction assumes independent binomial "
        "called-copy sampling and is not generally unbiased; positive f3 does not prove a near-null and negative "
        "f3 would not supply direction. The prespecified 0.95 diagnostic threshold is never tuned post hoc."
    )
    return result


def adjudicate_panel(
    predicted_class: str,
    gate_score: float,
    direction_rms_z: float,
    gate_rms_z: float,
) -> dict:
    severe = max(direction_rms_z, gate_rms_z) > 10.0
    gate_relation = (
        "qualitatively_concordant_with_detection_limited_near_null"
        if gate_score < 0.5
        else "qualitatively_in_tension_with_detection_limited_near_null"
    )
    if gate_score < 0.5:
        direction_relation = "not_applicable_under_raw_near_null_gate"
    elif predicted_class == "C":
        direction_relation = "raw_class_matches_counterfactual_exposure_orientation_if_event_present"
    else:
        direction_relation = "raw_class_differs_from_counterfactual_exposure_orientation_if_event_present"
    return {
        "natural_data_call_status": "abstain_severe_OOD" if severe else "descriptive_only_no_gold_label",
        "severe_OOD": severe,
        "severe_OOD_rule": "max(direction RMS-z, gate RMS-z) > 10; heuristic, not calibrated support",
        "literature_gate_relation": gate_relation,
        "counterfactual_direction_relation": direction_relation,
        "candidate_class_if_event_present": "C",
        "direction_truth_available": False,
        "gate_truth_available": False,
        "accuracy_eligible": False,
        "specificity_eligible": False,
        "interpretation_guardrail": (
            "the raw gate is not a probability or migration estimate; concordance/tension is descriptive "
            "and cannot be called a true negative, false positive, correct, or incorrect"
        ),
    }


def run_panels(
    source_vcf: Path,
    manifests: dict[str, Path],
    cache: Path,
    cap: int,
    direction_head,
    gate_head,
) -> tuple[list[dict], dict]:
    panels = []
    shared_audits = {}
    for filter_name, strict in (
        ("standard_contract", False),
        ("within_population_polymorphism", True),
    ):
        shared_vcf = cache / f"oyster.shared.{filter_name}.vcf"
        shared_popmap = cache / f"oyster.shared.{filter_name}.popmap.tsv"
        shared_audit = prepare_vcf(
            source_vcf,
            manifests["union"],
            shared_vcf,
            shared_popmap,
            cap=cap,
            seed=20260711,
            min_called_copies=16,
            require_three_populations=False,
            polymorphic_panel_manifests=(manifests["W"], manifests["Q"]),
            polymorphic_within_each_population=strict,
        )
        expected = EXPECTED_FILTERS[filter_name]
        counts = shared_audit["counts"]
        if counts["retained_after_cap"] != expected["loci"]:
            raise AssertionError(f"unexpected oyster {filter_name} locus count")
        if shared_audit["ordered_locus_sha256"] != expected["ordered_locus_sha256"]:
            raise AssertionError(f"unexpected oyster {filter_name} ordered locus hash")
        if counts["insufficient_called_copies"] != expected["insufficient_called_copies"]:
            raise AssertionError(f"unexpected oyster {filter_name} called-copy exclusions")
        if counts["not_polymorphic_in_every_panel"] != expected["not_polymorphic_in_every_panel"]:
            raise AssertionError(f"unexpected oyster {filter_name} polymorphism exclusions")
        locus_id_hash = ordered_locus_id_sha256(shared_vcf)
        if locus_id_hash != expected["ordered_locus_id_sha256"]:
            raise AssertionError(f"unexpected oyster {filter_name} locus-ID hash")
        shared_audit["ordered_locus_id_sha256"] = locus_id_hash
        shared_audit["shared_site_contract"] = (
            "one exact ordered W/Q locus intersection; site outputs are directly paired within this filter"
        )
        shared_audit["joint_called_copy_guardrail"] = (
            "minimum 16 called copies in every one of six cohorts means QBB2 (n=9) may miss at most one "
            "diploid genotype per locus while n=12 cohorts may miss four; because loci are shared, W is "
            "also conditioned on Q missingness"
        )
        shared_audits[filter_name] = shared_audit

        for site, spec in PANEL_SPECS.items():
            panel_vcf = cache / f"oyster.{site}.{filter_name}.vcf"
            panel_popmap = cache / f"oyster.{site}.{filter_name}.popmap.tsv"
            panel_audit = subset_prepared_vcf(
                shared_vcf,
                manifests[site],
                panel_vcf,
                panel_popmap,
                shared_audit,
            )
            panel_audit["comparison_locus_contract"] = (
                "same exact ordered locus set as the other Georges River site within this filter"
            )
            panel_audit["ordered_locus_id_sha256"] = ordered_locus_id_sha256(panel_vcf)
            if panel_audit["ordered_locus_id_sha256"] != expected["ordered_locus_id_sha256"]:
                raise AssertionError("oyster panel subsetting changed locus IDs")
            expectation = {
                "benchmark_role": "same_release_literature_supported_candidate_null_crossing_sensitivity_stress_test",
                "site_role": spec["role"],
                "expected_gate": None,
                "literature_candidate_gate_state": "no_detected_sustained_B2_to_wild_introgression",
                "candidate_class_if_event_present": "C",
                "candidate_forward_direction_if_present": f"{spec['population_order'][2]} (P3) -> {spec['population_order'][1]} (P2)",
                "direction_truth_available": False,
                "gate_truth_available": False,
                "accuracy_eligible": False,
                "specificity_eligible": False,
                "source_label_reuse": (
                    "the released 1,200-locus workbook derives from the same SNP-discovery data used by "
                    "the paper, but already omits the three QBB2 individuals removed after DAPC and does "
                    "not identify the paper's 1,189-neutral-locus subset; the source label/exclusion is "
                    "outcome-derived from the same underlying data, not independent"
                ),
                "same_data_excluded_ids": spec["same_data_excluded_ids"],
                "joint_ascertainment": (
                    "W and Q share one DArT discovery/filtering process and one anonymous locus release"
                ),
                "joint_called_copy_guardrail": (
                    "the shared minimum of 16 called copies allows at most one missing QBB2 genotype "
                    "(n=9) versus four in n=12 cohorts, so the primary W panel is conditioned on Q missingness"
                ),
                "sample_scope": "all released samples for this site after author preprocessing",
                "locus_filter_variant": (
                    "both alleles observed within every P1/P2/P3 population at both W and Q; strong ascertainment"
                    if strict
                    else "both alleles observed across each complete W and Q panel on one shared intersection"
                ),
                "tree_contract_status": "operational_wild_WC_OC_cluster_vs_selected_B2_not_species_tree",
                "B2_founder_selection_guardrail": (
                    "B2 is a selectively bred sixth-generation stock descended from 140 founders sampled "
                    "across Georges River and three northern rivers; drift, pedigree, selection, and heterozygosity differ"
                ),
            }
            panel = score_panel(
                f"oyster_{site}_{filter_name}",
                panel_vcf,
                panel_popmap,
                spec["population_order"],
                panel_audit,
                direction_head[0],
                direction_head[1],
                expectation,
            )
            panel["population_order"]["tree_contract_status"] = expectation["tree_contract_status"]
            add_gate_score(panel, gate_head[0], gate_head[1])
            panel["simulation_head"]["interpretation"] = (
                "uncalibrated classifier scores on a natural-data input; not probabilities, posteriors, "
                "or OOD-detector scores"
            )
            panel["simulation_gate"]["interpretation"] = (
                "uncalibrated gate-classifier score on a natural-data input; not a probability, posterior, "
                "migration estimate, or OOD-detector score"
            )
            panel["model_free_comparator"] = oyster_frequency_geometry(
                panel_vcf, manifests[site], spec["population_order"]
            )
            panel["adjudication"] = adjudicate_panel(
                panel["simulation_head"]["predicted_class"],
                panel["simulation_gate"]["appreciable_score"],
                panel["simulation_feature_shift"]["rms_z"],
                panel["simulation_gate_feature_shift"]["rms_z"],
            )
            panels.append(panel)
    for filter_name in EXPECTED_FILTERS:
        site_panels = [panel for panel in panels if panel["panel_id"].endswith(filter_name)]
        if len(site_panels) != 2 or len({panel["input_audit"]["ordered_locus_sha256"] for panel in site_panels}) != 1:
            raise AssertionError("oyster W/Q panels do not share exact loci")
    return panels, shared_audits


def summarize_outcomes(panels: list[dict]) -> dict:
    if len(panels) != 4:
        raise AssertionError("oyster benchmark requires exactly four analytic sensitivities")
    return {
        "analytic_sensitivity_runs": 4,
        "correlated_site_comparisons": 2,
        "literature_near_null_candidate_comparisons": 2,
        "unique_biological_systems": 1,
        "independent_validation_panels": 0,
        "accuracy_available": False,
        "accuracy_estimate": None,
        "specificity_available": False,
        "specificity_estimate": None,
        "raw_gate_below_0_5": sum(
            panel["simulation_gate"]["appreciable_score"] < 0.5 for panel in panels
        ),
        "raw_gate_crossings_at_0_5": sum(
            panel["simulation_gate"]["appreciable_score"] >= 0.5 for panel in panels
        ),
        "raw_head_prediction_counts": {
            label: sum(panel["simulation_head"]["predicted_class"] == label for panel in panels)
            for label in ("A", "B", "C")
        },
        "raw_counterfactual_C_calls": sum(
            panel["simulation_head"]["predicted_class"] == "C" for panel in panels
        ),
        "severe_OOD_panels": sum(panel["adjudication"]["severe_OOD"] for panel in panels),
        "abstained_panels": sum(
            panel["adjudication"]["natural_data_call_status"] == "abstain_severe_OOD"
            for panel in panels
        ),
        "interpretation": (
            "two correlated Georges River sites crossed with two shared-locus ascertainment filters; "
            "run counts are descriptive sensitivities, not validation trials"
        ),
    }


def validate_sources_record(path: Path = SOURCES_RECORD) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["data_doi"] != "10.5061/dryad.32q80" or record["version_id"] != 3_411:
        raise AssertionError("oyster Dryad source identity changed")
    for key in ("id", "key", "archive_member", "download", "bytes", "md5", "sha256"):
        if record["file"][key] != FILE[key]:
            raise AssertionError(f"oyster source-record file {key} changed")
    if record["file"]["sha256"] != FILE["sha256"] or record["file"]["md5"] != FILE["md5"]:
        raise AssertionError("oyster source-record workbook hash changed")
    if record["archive"]["digest_policy"] != ARCHIVE["digest_policy"]:
        raise AssertionError("oyster source-record archive policy changed")
    workbook_record = record["workbook_contract"]
    for key in (
        "sheet_name",
        "dimension",
        "samples",
        "loci",
        "genotype_pairs",
        "missing_genotype_pairs",
        "ordered_sample_sha256",
        "sample_population_tsv_sha256",
        "ordered_locus_sha256",
        "genalex_semantic_sha256",
    ):
        if workbook_record[key] != SOURCE_CONTRACT[key]:
            raise AssertionError(f"oyster source-record workbook {key} changed")
    rows = read_panel_record()
    expected_crosswalk = {row["workbook_population"]: row["paper_population"] for row in rows}
    if record["population_mapping"]["workbook_to_paper"] != expected_crosswalk:
        raise AssertionError("oyster source-record population crosswalk changed")
    design = record["analysis_design"]
    if (
        design["expected_gate"] is not None
        or design["direction_truth_available"] is not False
        or design["gate_truth_available"] is not False
        or design["independent_validation_panels"] != 0
        or design["candidate_class_if_event_present"] != "C"
    ):
        raise AssertionError("oyster source record incorrectly claims independent validation")
    for filter_name, expected in EXPECTED_FILTERS.items():
        observed = design["shared_locus_filters"][filter_name]
        if observed["loci"] != expected["loci"] or observed["ordered_locus_sha256"] != expected["ordered_locus_sha256"]:
            raise AssertionError(f"oyster source-record {filter_name} contract changed")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    parser.add_argument("--source-xlsx")
    parser.add_argument("--archive")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    if args.cap != DEFAULT_CAP:
        parser.error(f"--cap is frozen at {DEFAULT_CAP} for the source contract")

    set_below_normal_priority()
    cache = Path(args.cache_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = Path(args.source_xlsx).resolve() if args.source_xlsx else cache / FILE["key"]
    archive_path = Path(args.archive).resolve() if args.archive else cache / ARCHIVE["key"]

    verified = ensure_source(workbook_path, archive_path, args.download_missing)
    sources_record = validate_sources_record()
    workbook = parse_genalex_workbook(workbook_path)
    source_vcf = cache / "oyster.synthetic_source.vcf"
    derived_source = materialize_source_vcf(workbook, source_vcf)
    validate_derived_source_audit(derived_source)
    population_record = read_panel_record()
    manifests, manifest_audit = materialize_manifests(workbook, cache / "manifests")

    data_root = Path(args.data_root).resolve()
    direction_head = simulation_direction_head(data_root, max_depth=MAX_DEPTH)
    gate_head = simulation_gate_head(data_root, max_depth=MAX_DEPTH)
    panels, shared_filter_audits = run_panels(
        source_vcf,
        manifests,
        cache,
        args.cap,
        direction_head,
        gate_head,
    )
    result = {
        "schema_version": "dnnaic-oyster-2017-external-benchmark-v1",
        "git": git_revision(),
        "runtime": runtime_helpers.runtime_audit(),
        "guardrail": (
            "same-release, power-limited candidate-null crossing-sensitivity stress test only; no panel is a proven zero, "
            "gold label, independent validation, accuracy trial, or specificity trial"
        ),
        "source": {
            "record": DRYAD_RECORD,
            "data_doi": "10.5061/dryad.32q80",
            "paper_doi": "10.3354/meps12109",
            "license": "CC0-1.0",
            "verified": verified,
            "sources_record": {
                "path": str(SOURCES_RECORD),
                "sha256": sha256_file(SOURCES_RECORD),
                "content": sources_record,
            },
            "population_record": {
                "path": str(PANEL_RECORD),
                "sha256": sha256_file(PANEL_RECORD),
                "rows": population_record,
            },
            "workbook_audit": workbook["audit"],
            "derived_source_vcf": derived_source,
            "manifest_audit": manifest_audit,
            "shared_filter_audits": shared_filter_audits,
            "ascertainment_guardrails": (
                "anonymous DArT genotype-by-sequencing release; 1,189 neutral plus 11 consensus selection "
                "outliers cannot be distinguished by the generic locus labels; W/Q share joint discovery; "
                "released QBB2 already omits three oysters removed after the paper's neutral-locus DAPC; "
                "physical linkage is unavailable"
            ),
        },
        "published_evidence": {
            "source_result": (
                "strong wild-versus-B2 partitioning and no detected sustained introgression at either site"
            ),
            "detection_limit": (
                "site sample sizes of 9-12 and same-data preprocessing may miss low gene flow, occasional "
                "hybrids, or gene flow that does not produce sustained introgression; the finding does not "
                "establish migration rate zero"
            ),
            "Q_same_data_exclusions": (
                "original IDs 28, 29, and 31 were omitted from released QBB2 after the paper's DAPC on "
                "its 1,189 neutral loci assigned them to the wild/control cluster; Q is outcome-conditioned "
                "and secondary"
            ),
            "topology_guardrail": (
                "WC and overcatch are nearby wild microhabitat cohorts, while B2 is selected hatchery stock; "
                "the operational ((WC,OC),B2) order is not a rooted species tree"
            ),
            "later_context": {
                "wild_population_study": (
                    "O'Hare et al. 2021 (10.1007/s10592-021-01343-4) analyzed a different 363-wild-"
                    "oyster/3,400-neutral-SNP question, contains no B2 or overcatch cohorts, and is not a "
                    "documented reanalysis; O'Hare is the same lead researcher as Thompson and Stow/Raftos "
                    "are shared coauthors, so it is not independent validation"
                ),
                "species_wide_generalization_guardrail": (
                    "the 2021 result of one highly connected wild stock, high effective population size, and no "
                    "recent bottleneck argues against generalizing the 2017 Georges River inference to compromised "
                    "species-wide wild-population resilience; it does not re-test or overturn the local "
                    "wild-versus-B2 diversity contrast"
                ),
                "review": (
                    "Bishop et al. 2023 (10.3389/fmars.2023.1162487) is a review that cites Thompson 2017 "
                    "among studies reporting little evidence of aquaculture-line introgression; it performs "
                    "no genotype reanalysis and supplies no independent truth label"
                ),
                "reanalysis_search_status": (
                    "Dryad/DataCite relations and exact DOI/filename searches located no later independent "
                    "reanalysis of the released 90-by-1,200 matrix as of 2026-07-11"
                ),
            },
        },
        "direction_head": direction_head[2],
        "gate_head": gate_head[2],
        "outcome_summary": summarize_outcomes(panels),
        "panels": panels,
    }
    output = result_dir / "results.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "outcome": result["outcome_summary"],
                "panels": [
                    {
                        "panel_id": panel["panel_id"],
                        "direction": panel["simulation_head"]["predicted_class"],
                        "gate": panel["simulation_gate"]["appreciable_score"],
                        "direction_rms_z": panel["simulation_feature_shift"]["rms_z"],
                        "gate_rms_z": panel["simulation_gate_feature_shift"]["rms_z"],
                        "status": panel["adjudication"]["natural_data_call_status"],
                        "projection": panel["model_free_comparator"][
                            "P2_projection_from_P1_toward_P3_all_loci"
                        ],
                        "loci": panel["padze"]["n_loci_kept"],
                    }
                    for panel in panels
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
