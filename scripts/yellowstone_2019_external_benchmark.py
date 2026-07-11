#!/usr/bin/env python3
"""Run a guarded Yellowstone cutthroat--rainbow trout transfer stress test.

The primary trio is frozen without looking at DNNaic scores: isolated
SFOwlCreek Yellowstone cutthroat trout are P1, Trout Creek is P2, and Story
Hatchery rainbow trout are P3. Trout Creek is the sampled site nearest the
stocked Buffalo Bill Reservoir, but it also received Yellowstone cutthroat
stocking and the references are modern proxies. Candidate class C therefore
encodes an ecological introduction-history prior, not pedigree truth or an
exclusive population-level edge.

The source VCF contains both hard GT calls and PL likelihoods. The runner
reports source-GT and unique-PL-argmin sensitivities, their locus overlap,
reference/target sensitivities, the complete population--library confounding,
and a defect in the released VCF-to-Entropy converter. Every learned score is
descriptive only; the severe-OOD threshold is a heuristic diagnostic and never
creates an accepted direction call or an accuracy estimate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import urllib.request


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from additional_external_benchmarks import add_gate_score, simulation_gate_head
from external_benchmarks import (
    MANIFEST_DIR,
    MAX_DEPTH,
    REPO,
    git_revision,
    prepare_vcf,
    score_panel,
    set_below_normal_priority,
    sha256_file,
    simulation_direction_head,
    verify_file,
)
from tinkerbird_external_benchmark import frequency_projection, runtime_audit


DEFAULT_CACHE = REPO / "data" / "real" / "yellowstone_2019_external_benchmark"
DEFAULT_RESULTS = REPO / "results" / "yellowstone_2019_external_benchmark_2026_07_11"
SOURCE_RECORD = MANIFEST_DIR / "yellowstone_cutthroat_2019" / "sources.json"
SOURCE_RECORD_CANONICAL_LF_CONTRACT = {
    "bytes": 4_133,
    "sha256": "3633e5aa098b2223a79361c239b87ed0d46b9239262280808ec9daac9fe2a62e",
}
DEFAULT_CAP = 15_000
SEVERE_OOD_RMS_Z = 10.0
DRYAD_VERSION = "https://datadryad.org/api/v2/versions/29052"
PAPER_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC6775767/"

FILES = {
    "vcf": {
        "key": "variants_0.6_maf0.05.recode.vcf.gz",
        "copied_file_id": 109_334,
        "created_file_id": 109_343,
        "url": "https://datadryad.org/api/v2/files/109343/download",
        "bytes": 47_687_900,
        "md5": "ddfa5ab0307957daaea5f9ad5c4afe19",
        "sha256": "c0341a3a9dc11206e460907bea3d618f535245c7993d9cfce39c12c2b9b8bc86",
    },
    "metadata": {
        "key": "YSC_RBT_allind_barcode_key_detailed.csv",
        "copied_file_id": 109_340,
        "created_file_id": 109_349,
        "url": "https://datadryad.org/api/v2/files/109349/download",
        "bytes": 93_423,
        "md5": "9dde36c8b70c0544b6f93f411b1bc204",
        "sha256": "05d71fb77dac9cf61f80a54f4ac8c8fdf79f52423a9fd1e5d9b21f299cd60f46",
    },
    "predictors": {
        "key": "predictors_var_26july18.csv",
        "copied_file_id": 109_339,
        "created_file_id": 109_348,
        "url": "https://datadryad.org/api/v2/files/109348/download",
        "bytes": 2_852,
        "md5": "603fa826890a786bd3d3b9c968b98dc7",
        "sha256": "a9663d2a8922e3fdace4b90598becf8991ca57a11be626bc353e87bc89d8abfb",
    },
    "response": {
        "key": "response_var_18jan18.csv",
        "copied_file_id": 109_338,
        "created_file_id": 109_347,
        "url": "https://datadryad.org/api/v2/files/109347/download",
        "bytes": 1_048,
        "md5": "3de33d11206953a4bb7fe9cc6fe3cc1b",
        "sha256": "01e4e84e18ae21aa88ed9c85ccd3ee2148b4e522b2170d3c110ad72433e3b9b1",
    },
    "converter": {
        "key": "vcf2mpgl.pl",
        "copied_file_id": 109_336,
        "created_file_id": 109_345,
        "url": "https://datadryad.org/api/v2/files/109345/download",
        "bytes": 3_268,
        "md5": "b479c763ebb11e0458f7c44d7f43b96d",
        "sha256": "f68cba59fc020cf59f97f021a7385f0cb03c0ade1c922aa3fbc95126e80fa0bc",
    },
}

METADATA_HEADER = [
    "plate",
    "well",
    "col",
    "row",
    "name",
    "barcode",
    "sourceplate",
    "FishID",
    "Tributary",
    "Priority",
    "Reach",
    "Length_mm",
    "Mass_g",
    "Library_number",
    "TroutPlate_number",
]
METADATA_ONLY_IDS = [
    "EGM16_0128",
    "EGM16_0139",
    "EGM16_0168",
    "EGM16_0169",
    "EGM16_0251",
    "EGM16_0274",
    "EGM16_0317",
    "EGM16_0380",
    "EGM16_0385",
    "EGM16_0505",
    "EGM16_0689",
    "EGM16_0714",
    "EGM16_0741",
    "EGM16_0849",
    "EGM16_0903",
    "EGM16_0920",
    "EGM16_1153",
    "EGM16_1270",
    "EGM1964_042",
]

PANEL_SPECS = {
    "main": {
        "groups": {"P1": "SFOwlCreek", "P2": "Trout", "P3": "StoryHatchery"},
        "counts": {"P1": 61, "P2": 58, "P3": 20},
        "manifest_bytes": 2_007,
        "manifest_sha256": "a6e1fad0047e4c0beedeae2dc165eba24acdf9853e4f57626677a7af90ff80bd",
        "candidate_class": "C",
        "role": "ecological_introduction_history_direction_stress_test",
        "candidate_basis": (
            "Trout is the nearest sampled site to Buffalo Bill Reservoir and had zero direct "
            "rainbow-trout stocking; the paper infers upstream movement from the stocked reservoir"
        ),
    },
    "tensleep_reference": {
        "groups": {"P1": "TensleepHatchery", "P2": "Trout", "P3": "StoryHatchery"},
        "counts": {"P1": 19, "P2": 58, "P3": 20},
        "manifest_bytes": 1_358,
        "manifest_sha256": "811d6236cb8a8c38150289e3bbd0ae50b25030e105dcf99bea9a7fbce93b2157",
        "candidate_class": "C",
        "role": "reference_proxy_sensitivity",
        "candidate_basis": "same ecological exposure prior as main, with a modern hatchery YCT proxy",
    },
    "big_direct_stock": {
        "groups": {"P1": "TensleepHatchery", "P2": "Big", "P3": "StoryHatchery"},
        "counts": {"P1": 19, "P2": 61, "P3": 20},
        "manifest_bytes": 1_400,
        "manifest_sha256": "fbd36f7cf328de7b099c14aeb2638940f1e164b42fada6b6d8b4743f524cbb7a",
        "candidate_class": "C",
        "role": "direct_stocking_sensitivity",
        "candidate_basis": "external stocking table records 50,000 RBT and 5,000 YCT stocked at Big",
    },
    "candidate_null": {
        "groups": {"P1": "TensleepHatchery", "P2": "SFOwlCreek", "P3": "StoryHatchery"},
        "counts": {"P1": 19, "P2": 61, "P3": 20},
        "manifest_bytes": 1_461,
        "manifest_sha256": "2f7d79812915b8307c5310301c3a5adcb0700afb92f2beb79b2a398357ba8715",
        "candidate_class": None,
        "role": "same_species_candidate_null_structure_diagnostic",
        "candidate_basis": "no independent no-flow label; the direction head has no no-event class",
    },
}

GROUP_ORDERED_ID_SHA256 = {
    "SFOwlCreek": "f820a0a0127c45ec82a2f1d120c5bec994ea49074ca94995d2e3105347641477",
    "Trout": "c6bcea211090a30dc90ef84ce77242229e2eb21dc23fda9cbdf4b44acae88053",
    "StoryHatchery": "36606b31e46813dd41cae47ddf609e4ae700172e5f2fe41545d5df3b8a162b32",
    "TensleepHatchery": "80ebcad303c69265cfb1c52a296487d3b878cae2ae0ffcc5b214547f69d97420",
    "Big": "e221a9ceb306bae7d305e55900b67bee76448981d546ed9229930992fc92c6ef",
}

SOURCE_VCF_CONTRACT = {
    "metadata_header_lines": 63,
    "samples": 1_286,
    "variants": 12_666,
    "chromosomes": 29,
    "raw_header_bytes": 105_559,
    "raw_header_sha256": "5f93ec5b124102c3b6866f69624127bfa7d571a7ae7c60e66a5d33f47553b037",
    "normalized_header_bytes": 14_253,
    "normalized_header_sha256": "b8e9014b0853f41ea07aa4dc52104af5a9436a7e8f275a2539e370621dd7dce9",
    "raw_sample_list_sha256": "8e437303110fdca8d903f8859c17f2c3c3a91ccba782991c03a5422bb0756092",
    "normalized_sample_list_sha256": "318548eb015dd6c32287bca8ada66b241bdb9b130592ebee8bf94d57891ac693",
    "ordered_locus_sha256": "12b9a0f39fe7a3a12db057ce78e913a6848e5a5e7b5819843a0e0169c6643f59",
    "variant_payload_sha256": "43e5110079f575881cdaa1e1ad76d92b96d063990973c79f49a21b862758d55b",
    "decompressed_sha256": "9de64f02fc892a5620852d5fb9d9531ac3a46c8c3b9a1567dcd14fc6a6011128",
}
NORMALIZED_REPRESENTATION_CONTRACT = {
    "source_GT": {
        "bytes": 65_783_829,
        "sha256": "9e06f9fb38347e4867cdd4e8b11678517e96df75f8e3f461271393fdcde0bbb3",
    },
    "unique_PL_argmin": {
        "bytes": 65_783_829,
        "sha256": "b5acf651bca6e842fa16e32038a6e7780813cad8f7070a3050400edc838385d9",
    },
}
CONVERTER_CELL_CONTRACT = {
    "cells": 16_288_476,
    "converter_zeroed_informative_PL_cells": 1_789_601,
    "source_GT_called": 11_399_315,
    "source_GT_missing": 4_889_161,
    "unique_PL_argmin_called": 13_170_198,
    "unique_PL_argmin_with_DP_positive": 13_170_198,
}

RUN_SPECS = [
    ("main", "source_GT", "standard_contract", False),
    ("main", "source_GT", "within_population_polymorphism", True),
    ("main", "unique_PL_argmin", "standard_contract", False),
    ("main", "unique_PL_argmin", "within_population_polymorphism", True),
    ("tensleep_reference", "source_GT", "standard_contract", False),
    ("big_direct_stock", "source_GT", "standard_contract", False),
    ("candidate_null", "source_GT", "standard_contract", False),
]
EXPECTED_PANEL_LOCI = {
    ("main", "source_GT", "standard_contract"): (
        11_758,
        "9ef1e1f65321ab60540f2c90e1aa12d4f78ac2449c8686a0a295d78c7e75a274",
    ),
    ("main", "source_GT", "within_population_polymorphism"): (
        2_591,
        "bad416e8a0ed253a1b87c9089453bd40cf7ed393b3aea5fd3b8eae96b52b7b42",
    ),
    ("main", "unique_PL_argmin", "standard_contract"): (
        12_170,
        "7ac1f0ef6d59f80bb1bc78a061aa691663c996a1d4f9630e7d40222824678050",
    ),
    ("main", "unique_PL_argmin", "within_population_polymorphism"): (
        3_653,
        "822b251fe3411f576331534511101e3ecd8c02194a8d8a785b57027b4c701bef",
    ),
    ("tensleep_reference", "source_GT", "standard_contract"): (
        9_642,
        "eb29d4d242b91ff443cf161a77b14575f4c024bbd955b7e5c91e8c9cdf7c5c42",
    ),
    ("big_direct_stock", "source_GT", "standard_contract"): (
        9_721,
        "4ad22ffff991cf8515ef1d0230e61f921b8c06b2331f72c98d4786526fc612d8",
    ),
    ("candidate_null", "source_GT", "standard_contract"): (
        7_563,
        "6965a13e3ee0f0f5706286e22f983657c99aedcf4a6cd51071700b2e3df576e2",
    ),
}
EXPECTED_PREPARED_VCF = {
    "yellowstone_main_source_GT_standard_contract": {
        "bytes": 6_985_231,
        "sha256": "15b75a25ea71df90e2c8717ba67f0c9439cb08124cc396d9ceba199a745046f6",
    },
    "yellowstone_main_source_GT_within_population_polymorphism": {
        "bytes": 1_540_619,
        "sha256": "a8e40b0a3aa493aee11fbfcf8a8bfe516422b77fa60794eddaac496a57884be9",
    },
    "yellowstone_main_unique_PL_argmin_standard_contract": {
        "bytes": 7_229_934,
        "sha256": "bfdcb9a8ba6614919606747517cd6e196efb9bdcb724531f640f20817d2df10b",
    },
    "yellowstone_main_unique_PL_argmin_within_population_polymorphism": {
        "bytes": 2_171_362,
        "sha256": "5fc1bcba5b0f47abe35eaa4c1df734294144fc52fe152aa799f27ad7fa3870cc",
    },
    "yellowstone_tensleep_reference_source_GT_standard_contract": {
        "bytes": 4_108_114,
        "sha256": "8b4cd4a189a59bde19493e6ea846e9778a6da36fcc1dcc3791f522bc3378a420",
    },
    "yellowstone_big_direct_stock_source_GT_standard_contract": {
        "bytes": 4_258_457,
        "sha256": "9d36721a33393a9f19deab039c148ad2102fc9954de2c2c21ebf7cc14df1e6b4",
    },
    "yellowstone_candidate_null_source_GT_standard_contract": {
        "bytes": 3_313_401,
        "sha256": "1b985601a2f1cfd096263854dcafb336849a1830fa78cceae0aae51313764b5f",
    },
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DNNaic external benchmark/1"})
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(output)


def ensure_source(path: Path, spec: dict, download_missing: bool) -> dict:
    if not path.exists():
        if not download_missing:
            raise FileNotFoundError(path)
        _download(spec["url"], path)
    verified = verify_file(path, spec["bytes"], spec["sha256"])
    observed_md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if observed_md5 != spec["md5"]:
        raise AssertionError(f"{path}: MD5 mismatch")
    return {
        **verified,
        "md5": observed_md5,
        "dryad_copied_file_id": spec["copied_file_id"],
        "dryad_created_file_id": spec["created_file_id"],
    }


def validate_sources_record() -> dict:
    raw = SOURCE_RECORD.read_bytes()
    canonical_lf = raw.replace(b"\r\n", b"\n")
    if b"\r" in canonical_lf:
        raise AssertionError("Yellowstone sources.json contains a non-CRLF carriage return")
    canonical_contract = {
        "bytes": len(canonical_lf),
        "sha256": hashlib.sha256(canonical_lf).hexdigest(),
    }
    if canonical_contract != SOURCE_RECORD_CANONICAL_LF_CONTRACT:
        raise AssertionError("Yellowstone sources.json canonical LF contract changed")
    record = json.loads(raw.decode("utf-8"))
    if record["schema_version"] != "dnnaic-yellowstone-2019-source-v1":
        raise AssertionError("unexpected Yellowstone source schema")
    if record["data_doi"] != "10.5061/dryad.6s7d02q" or record["version_id"] != 29_052:
        raise AssertionError("unexpected Yellowstone Dryad version")
    if record["files"] != FILES:
        raise AssertionError("Yellowstone source file contracts differ from runner")
    return {
        "path": str(SOURCE_RECORD),
        "canonical_lf": canonical_contract,
        "working_tree": {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "line_endings_normalized_for_contract": raw != canonical_lf,
        },
        "record": record,
    }


def normalize_vcf_sample(value: str) -> str:
    name = PurePosixPath(value.replace("\\", "/")).name
    if name.startswith("aln_"):
        name = name[4:]
    suffix = ".sorted.bam"
    if name.endswith(suffix):
        name = name[: -len(suffix)]
    if not name.startswith("EGM") or "_" not in name:
        raise ValueError(f"unexpected Yellowstone VCF sample label: {value!r}")
    return name


def read_metadata(path: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != METADATA_HEADER:
            raise AssertionError(f"unexpected metadata columns: {reader.fieldnames}")
        rows = list(reader)
    ids = [row["FishID"] for row in rows]
    if len(rows) != 1_305 or len(set(ids)) != len(ids):
        raise AssertionError("Yellowstone metadata must contain 1,305 unique FishIDs")
    return rows, {row["FishID"]: row for row in rows}


def decode_source_gt(value: str) -> str:
    value = value.replace("|", "/")
    if value in {".", "./."} or "." in value.split("/"):
        return "./."
    if value in {"0/0", "1/1"}:
        return value
    if value in {"0/1", "1/0"}:
        return "0/1"
    raise ValueError(f"unexpected biallelic diploid GT: {value!r}")


def unique_pl_argmin(pl_value: str) -> tuple[str, tuple[int, int, int] | None]:
    parts = pl_value.split(",")
    if len(parts) != 3 or any(part in {"", "."} for part in parts):
        return "./.", None
    try:
        values = tuple(int(part) for part in parts)
    except ValueError:
        return "./.", None
    minimum = min(values)
    if values.count(minimum) != 1:
        return "./.", values
    return ("0/0", "0/1", "1/1")[values.index(minimum)], values


def audit_converter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    required = (
        'if( $a1 eq "." && $a2 eq ".")',
        "$g1 = 0;",
        "$g2 = 0;",
        "$g3 = 0;",
    )
    if not all(token in text for token in required):
        raise AssertionError("released vcf2mpgl missing-GT behavior changed")
    return {
        "released_behavior": (
            "when GT is ./., overwrite the three parsed PL values with 0 0 0 before writing MPGL"
        ),
        "methodological_consequence": (
            "informative low-GQ PL values attached to masked GT cells are discarded; the archived "
            "Entropy input is therefore not a lossless genotype-likelihood representation"
        ),
    }


def audit_source_and_materialize_representations(
    source: Path,
    metadata_by_id: dict[str, dict[str, str]],
    raw_output: Path,
    pl_output: Path,
) -> tuple[list[str], dict]:
    raw_output.parent.mkdir(parents=True, exist_ok=True)
    meta_headers = 0
    variants = 0
    chromosomes = set()
    coordinates = set()
    filters = Counter()
    formats = Counter()
    ids = Counter()
    decompressed = hashlib.sha256()
    payload = hashlib.sha256()
    loci = hashlib.sha256()
    samples: list[str] | None = None
    original_samples: list[str] | None = None
    representation = {group: Counter() for group in GROUP_ORDERED_ID_SHA256}
    global_cells = Counter()
    converter_zeroed_gq = set()
    converter_zeroed_nonpositive_DP = 0

    with gzip.open(source, "rt", encoding="utf-8", newline="") as incoming, raw_output.open(
        "w", encoding="utf-8", newline="\n"
    ) as raw_handle, pl_output.open("w", encoding="utf-8", newline="\n") as pl_handle:
        for line in incoming:
            encoded = line.encode("utf-8")
            decompressed.update(encoded)
            if line.startswith("##"):
                meta_headers += 1
                normalized = line.rstrip("\r\n") + "\n"
                raw_handle.write(normalized)
                pl_handle.write(normalized)
                continue
            if line.startswith("#CHROM"):
                source_fields = line.rstrip("\r\n").split("\t")
                original_samples = source_fields[9:]
                samples = [normalize_vcf_sample(sample) for sample in original_samples]
                if len(samples) != 1_286 or len(set(samples)) != len(samples):
                    raise AssertionError("VCF sample normalization is not a 1:1 mapping")
                if any(sample not in metadata_by_id for sample in samples):
                    raise AssertionError("a normalized VCF sample is absent from metadata")
                normalized_header = "\t".join(source_fields[:9] + samples) + "\n"
                if len(encoded) != SOURCE_VCF_CONTRACT["raw_header_bytes"] or hashlib.sha256(encoded).hexdigest() != SOURCE_VCF_CONTRACT["raw_header_sha256"]:
                    raise AssertionError("raw #CHROM header contract changed")
                normalized_bytes = normalized_header.encode("utf-8")
                if len(normalized_bytes) != SOURCE_VCF_CONTRACT["normalized_header_bytes"] or hashlib.sha256(normalized_bytes).hexdigest() != SOURCE_VCF_CONTRACT["normalized_header_sha256"]:
                    raise AssertionError("normalized #CHROM header contract changed")
                raw_handle.write(normalized_header)
                pl_handle.write(normalized_header)
                continue
            if line.startswith("#") or not line.strip():
                raise AssertionError("unexpected nonstandard VCF header line")
            if samples is None:
                raise ValueError("VCF row precedes #CHROM")
            payload.update(encoded)
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 9 + len(samples):
                raise AssertionError("VCF row width changed")
            variants += 1
            chromosomes.add(fields[0])
            coordinate = (fields[0], fields[1])
            if coordinate in coordinates:
                raise AssertionError("duplicate Yellowstone CHROM/POS")
            coordinates.add(coordinate)
            if len(fields[3]) != 1 or len(fields[4]) != 1 or "," in fields[4]:
                raise AssertionError("source contains a non-biallelic single-base SNP")
            filters[fields[6]] += 1
            formats[fields[8]] += 1
            ids[fields[2]] += 1
            loci.update(("\t".join((fields[0], fields[1], fields[3], fields[4])) + "\n").encode("utf-8"))
            format_names = fields[8].split(":")
            if format_names != ["GT", "PL", "DP", "AD", "GQ"]:
                raise AssertionError(f"unexpected FORMAT: {fields[8]}")

            raw_calls = []
            pl_calls = []
            for sample, cell in zip(samples, fields[9:]):
                parts = cell.split(":")
                if len(parts) != len(format_names):
                    raise AssertionError("malformed genotype cell")
                raw_call = decode_source_gt(parts[0])
                pl_call, pl_values = unique_pl_argmin(parts[1])
                raw_calls.append(raw_call)
                pl_calls.append(pl_call)
                global_cells["cells"] += 1
                global_cells["source_GT_called"] += int(raw_call != "./.")
                global_cells["unique_PL_argmin_called"] += int(pl_call != "./.")
                informative_pl = pl_values is not None and len(set(pl_values)) > 1
                if raw_call == "./.":
                    global_cells["source_GT_missing"] += 1
                    global_cells["converter_zeroed_informative_PL_cells"] += int(informative_pl)
                    if informative_pl:
                        try:
                            converter_zeroed_gq.add(int(parts[4]))
                        except ValueError as error:
                            raise AssertionError("informative masked PL cell has nonnumeric GQ") from error
                        try:
                            converter_zeroed_nonpositive_DP += int(float(parts[2]) <= 0)
                        except ValueError:
                            converter_zeroed_nonpositive_DP += 1
                if pl_call != "./.":
                    try:
                        dp_positive = float(parts[2]) > 0
                    except ValueError:
                        dp_positive = False
                    global_cells["unique_PL_argmin_with_DP_positive"] += int(dp_positive)

                group = metadata_by_id[sample]["Tributary"]
                if group in representation:
                    counts = representation[group]
                    counts["cells"] += 1
                    counts["source_GT_called"] += int(raw_call != "./.")
                    counts["unique_PL_argmin_called"] += int(pl_call != "./.")
                    counts["PL_only"] += int(raw_call == "./." and pl_call != "./.")
                    counts["GT_only"] += int(raw_call != "./." and pl_call == "./.")
                    both = raw_call != "./." and pl_call != "./."
                    counts["both_called"] += int(both)
                    counts["concordant_when_both_called"] += int(both and raw_call == pl_call)
                    if pl_call != "./.":
                        try:
                            counts["unique_PL_argmin_with_DP_positive"] += int(float(parts[2]) > 0)
                        except ValueError:
                            pass

            base = fields[:9]
            base[8] = "GT"
            raw_handle.write("\t".join(base + raw_calls) + "\n")
            pl_handle.write("\t".join(base + pl_calls) + "\n")

    if samples is None or original_samples is None:
        raise AssertionError("source VCF has no #CHROM line")
    observed = {
        "metadata_header_lines": meta_headers,
        "samples": len(samples),
        "variants": variants,
        "chromosomes": len(chromosomes),
        "raw_sample_list_sha256": _sha256_text("".join(sample + "\n" for sample in original_samples)),
        "normalized_sample_list_sha256": _sha256_text("".join(sample + "\n" for sample in samples)),
        "ordered_locus_sha256": loci.hexdigest(),
        "variant_payload_sha256": payload.hexdigest(),
        "decompressed_sha256": decompressed.hexdigest(),
    }
    for key, expected in SOURCE_VCF_CONTRACT.items():
        if key in observed and observed[key] != expected:
            raise AssertionError(f"source VCF {key} changed: {observed[key]!r}")
    if filters != {"PASS": 12_666} or formats != {"GT:PL:DP:AD:GQ": 12_666}:
        raise AssertionError("source FILTER/FORMAT counts changed")
    if ids != {".": 12_666}:
        raise AssertionError("source locus IDs are no longer all dot")
    if dict(global_cells) != CONVERTER_CELL_CONTRACT:
        raise AssertionError(f"source genotype-cell audit changed: {global_cells}")
    if converter_zeroed_gq != set(range(1, 10)) or converter_zeroed_nonpositive_DP != 0:
        raise AssertionError("informative PL values masked by GT are no longer exactly GQ 1--9 with DP>0")
    if global_cells["unique_PL_argmin_called"] != global_cells["unique_PL_argmin_with_DP_positive"]:
        raise AssertionError("a unique PL argmin call has nonpositive or missing DP")
    normalized_contract = {
        "source_GT": {"bytes": raw_output.stat().st_size, "sha256": sha256_file(raw_output)},
        "unique_PL_argmin": {"bytes": pl_output.stat().st_size, "sha256": sha256_file(pl_output)},
    }
    if normalized_contract != NORMALIZED_REPRESENTATION_CONTRACT:
        raise AssertionError("normalized Yellowstone representation contract changed")

    representation_audit = {}
    for group, values in representation.items():
        both = values["both_called"]
        representation_audit[group] = {
            **dict(values),
            "source_GT_call_rate": values["source_GT_called"] / values["cells"],
            "unique_PL_argmin_call_rate": values["unique_PL_argmin_called"] / values["cells"],
            "GT_PL_concordance_given_both": (
                values["concordant_when_both_called"] / both if both else None
            ),
        }
    metadata_ids = set(metadata_by_id)
    metadata_only = sorted(metadata_ids - set(samples))
    if metadata_only != METADATA_ONLY_IDS:
        raise AssertionError("metadata-only FishID set changed")
    return samples, {
        **observed,
        "filter_counts": dict(filters),
        "format_counts": dict(formats),
        "all_locus_ID_values_are_dot": True,
        "unique_CHROM_POS": True,
        "metadata_only_samples": metadata_only,
        "vcf_only_samples": [],
        "representation_diagnostics": representation_audit,
        "converter_cell_audit": dict(global_cells),
        "converter_zeroed_informative_PL_GQ_range": [min(converter_zeroed_gq), max(converter_zeroed_gq)],
        "converter_zeroed_informative_PL_all_DP_positive": True,
        "normalized_representations": {
            "source_GT": {
                "path": str(raw_output),
                **normalized_contract["source_GT"],
            },
            "unique_PL_argmin": {
                "path": str(pl_output),
                **normalized_contract["unique_PL_argmin"],
                "rule": (
                    "unique minimum of the diploid biallelic PL triplet maps to 0/0, 0/1, or 1/1; "
                    "ties/missing/malformed become ./.; no extra DP/GQ filter"
                ),
                "observed_all_accepted_calls_have_DP_positive": True,
            },
        },
    }


def metadata_and_manifest_audit(
    metadata_rows: list[dict[str, str]],
    metadata_by_id: dict[str, dict[str, str]],
    source_samples: list[str],
    cache: Path,
) -> tuple[dict[str, Path], dict]:
    source_set = set(source_samples)
    retained_rows = [row for row in metadata_rows if row["FishID"] in source_set]
    retained_counts = dict(sorted(Counter(row["Tributary"] for row in retained_rows).items()))
    group_hashes = {}
    for group in GROUP_ORDERED_ID_SHA256:
        ordered = [sample for sample in source_samples if metadata_by_id[sample]["Tributary"] == group]
        group_hashes[group] = _sha256_text("".join(sample + "\n" for sample in ordered))
    if group_hashes != GROUP_ORDERED_ID_SHA256:
        raise AssertionError("ordered Yellowstone cohort identities changed")

    manifests = {}
    manifest_audits = {}
    panel_library_counts = {}
    for name, spec in PANEL_SPECS.items():
        tributary_to_role = {tributary: role for role, tributary in spec["groups"].items()}
        selected = [
            (sample, tributary_to_role[metadata_by_id[sample]["Tributary"]])
            for sample in source_samples
            if metadata_by_id[sample]["Tributary"] in tributary_to_role
        ]
        raw = "".join(f"{sample}\t{role}\n" for sample, role in selected).encode("utf-8")
        observed_counts = dict(sorted(Counter(role for _sample, role in selected).items()))
        if observed_counts != spec["counts"]:
            raise AssertionError(f"{name}: cohort counts changed")
        if len(raw) != spec["manifest_bytes"] or hashlib.sha256(raw).hexdigest() != spec["manifest_sha256"]:
            raise AssertionError(f"{name}: manifest contract changed")
        path = cache / f"yellowstone.{name}.manifest.tsv"
        path.write_bytes(raw)
        manifests[name] = path
        manifest_audits[name] = {
            "path": str(path),
            "samples": len(selected),
            "population_counts": observed_counts,
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        panel_library_counts[name] = {
            role: dict(
                sorted(
                    Counter(
                        metadata_by_id[sample]["Library_number"]
                        for sample, observed_role in selected
                        if observed_role == role
                    ).items()
                )
            )
            for role in ("P1", "P2", "P3")
        }

    main = PANEL_SPECS["main"]
    main_libraries = panel_library_counts["main"]
    library_sets = [set(main_libraries[role]) for role in ("P1", "P2", "P3")]
    if any(library_sets[i] & library_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("main populations are no longer fully library-confounded")
    main_role_rows = {
        role: [
            metadata_by_id[sample]
            for sample in source_samples
            if metadata_by_id[sample]["Tributary"] == tributary
        ]
        for role, tributary in main["groups"].items()
    }
    if {row["Priority"] for row in main_role_rows["P1"]} != {"Reference"} or {
        row["Reach"] for row in main_role_rows["P1"]
    } != {"NA"}:
        raise AssertionError("SFOwlCreek reference metadata changed")
    if {row["Priority"] for row in main_role_rows["P2"]} != {"1"} or dict(
        sorted(Counter(row["Reach"] for row in main_role_rows["P2"]).items())
    ) != {"1": 18, "2": 20, "3": 20}:
        raise AssertionError("Trout priority/reach metadata changed")
    if {row["Priority"] for row in main_role_rows["P3"]} != {"Reference"} or {
        row["Reach"] for row in main_role_rows["P3"]
    } != {"NA"}:
        raise AssertionError("StoryHatchery reference metadata changed")
    return manifests, {
        "rows": len(metadata_rows),
        "unique_FishID": len(metadata_by_id),
        "VCF_retained_rows": len(retained_rows),
        "retained_tributary_counts": retained_counts,
        "ordered_group_ID_sha256": group_hashes,
        "manifests": manifest_audits,
        "panel_library_counts": panel_library_counts,
        "main_population_library_sets_disjoint": True,
        "main_priority_reach_contract": {
            "P1": {"Priority": "Reference", "Reach": "NA"},
            "P2": {"Priority": "1", "Reach_counts": {"1": 18, "2": 20, "3": 20}},
            "P3": {"Priority": "Reference", "Reach": "NA"},
        },
        "guardrail": (
            "In the main trio every population uses a disjoint set of sequencing libraries; "
            "population, call rate, and batch are therefore inseparable."
        ),
    }


def audit_predictors_and_response(predictors: Path, response: Path) -> dict:
    with predictors.open(encoding="utf-8", newline="") as handle:
        predictor_rows = list(csv.DictReader(handle))
    with response.open(encoding="utf-8", newline="") as handle:
        response_rows = list(csv.DictReader(handle))
    if len(predictor_rows) != 27 or len(response_rows) != 27:
        raise AssertionError("predictor/response site counts changed")
    predictor = {row["trib"]: row for row in predictor_rows}
    responses = {row["trib"]: row for row in response_rows}
    nearest = sorted(predictor_rows, key=lambda row: float(row["distance_m_BBR"]))
    if [row["trib"] for row in nearest[:2]] != ["Trout", "Jim"]:
        raise AssertionError("Trout is no longer uniquely nearest Buffalo Bill Reservoir")
    if predictor["Trout"] != {
        "trib": "Trout", "GNIS_NAME": "Trout Creek", "ELEV": "1655.24", "CANOPY": "10.71",
        "SLOPE": "0.02334", "PRECIP": "321.3", "CUMDRAINAG": "125.5", "S1_93_11": "16.22",
        "aspect": "S", "distance_m_BBR": "3930", "nrainbow": "0", "ncutt": "76095",
        "duration": "73", "roadside": "1", "nfish": "102735", "prop.cutt": "0.740692071835304",
    }:
        raise AssertionError("Trout predictor row changed")
    if responses["Trout"] != {
        "trib": "Trout", "prop.ysc": "0.2618", "prop.ysc.ind": "0",
        "prop.rbt.ind": "0.3103", "prop.hyb.ind": "0.6897", "range.bc": "0.7503",
    }:
        raise AssertionError("Trout response row changed")
    if predictor["Big"]["nrainbow"] != "50000" or predictor["Big"]["ncutt"] != "5000":
        raise AssertionError("Big stocking row changed")
    return {
        "sites": len(predictor_rows),
        "frozen_target_selection_before_genomic_outcomes": {
            "rule": "minimum distance_m_BBR among all released predictor rows",
            "selected": "Trout",
            "distance_m_BBR": 3_930,
            "next_site": "Jim",
            "next_distance_m_BBR": 11_317,
        },
        "Trout_predictors": predictor["Trout"],
        "Big_predictors": predictor["Big"],
        "Trout_published_same_genotype_response": responses["Trout"],
        "Big_published_same_genotype_response": responses["Big"],
        "archive_gap": (
            "per-individual Entropy q/Q values and author ancestry-class labels are not archived; "
            "only site summaries are released"
        ),
        "same_data_guardrail": (
            "published response values were computed from these same SNPs and are descriptive "
            "comparators, never independent truth labels"
        ),
    }


def adjudicate_panel(panel: dict, candidate_class: str | None) -> dict:
    prediction = panel["simulation_head"]["predicted_class"]
    direction_rms = panel["simulation_feature_shift"]["rms_z"]
    gate_rms = panel["simulation_gate_feature_shift"]["rms_z"]
    severe = max(direction_rms, gate_rms) > SEVERE_OOD_RMS_Z
    return {
        "candidate_class": candidate_class,
        "raw_head_matches_candidate": None if candidate_class is None else prediction == candidate_class,
        "direction_truth_available": False,
        "exclusive_single_edge_truth_available": False,
        "gate_truth_available": False,
        "severe_OOD": severe,
        "severe_OOD_rule": (
            f"max(direction RMS-z, gate RMS-z) > {SEVERE_OOD_RMS_Z:g}; "
            "heuristic diagnostic, not calibrated support"
        ),
        "natural_data_call_status": (
            "abstain_severe_OOD" if severe else "descriptive_only_no_gold_label"
        ),
        "direction_call_accepted": False,
        "formal_direction_accuracy_eligible": False,
        "gate_accuracy_eligible": False,
        "guardrail": (
            "ecological exposure, reference swaps, representation changes, and the candidate-null "
            "are correlated stress tests; none supplies an independent direction or gate label"
        ),
    }


def run_panels(
    representations: dict[str, Path],
    manifests: dict[str, Path],
    metadata_audit: dict,
    cache: Path,
    cap: int,
    direction_head,
    gate_head,
) -> tuple[list[dict], dict[tuple[str, str, str], Path]]:
    panels = []
    prepared_paths = {}
    for panel_name, representation, filter_name, strict in RUN_SPECS:
        spec = PANEL_SPECS[panel_name]
        panel_id = f"yellowstone_{panel_name}_{representation}_{filter_name}"
        prepared_vcf = cache / f"{panel_id}.vcf"
        prepared_popmap = cache / f"{panel_id}.popmap.tsv"
        audit = prepare_vcf(
            representations[representation],
            manifests[panel_name],
            prepared_vcf,
            prepared_popmap,
            cap=cap,
            seed=20260711,
            polymorphic_within_each_population=strict,
        )
        expected_loci, expected_hash = EXPECTED_PANEL_LOCI[(panel_name, representation, filter_name)]
        if audit["counts"]["eligible_before_cap"] != expected_loci or audit["counts"]["retained_after_cap"] != expected_loci:
            raise AssertionError(f"{panel_id}: eligible loci changed")
        if audit["ordered_locus_sha256"] != expected_hash:
            raise AssertionError(f"{panel_id}: ordered locus contract changed")
        observed_prepared = {
            "bytes": audit["derived_vcf"]["bytes"],
            "sha256": audit["derived_vcf"]["sha256"],
        }
        if observed_prepared != EXPECTED_PREPARED_VCF[panel_id]:
            raise AssertionError(f"{panel_id}: prepared VCF byte contract changed")
        observed_popmap = {
            "bytes": audit["derived_popmap"]["bytes"],
            "sha256": audit["derived_popmap"]["sha256"],
        }
        if observed_popmap != {
            "bytes": spec["manifest_bytes"],
            "sha256": spec["manifest_sha256"],
        }:
            raise AssertionError(f"{panel_id}: prepared popmap contract changed")
        audit["representation"] = representation
        audit["locus_filter_variant"] = filter_name
        audit["panel_library_counts"] = metadata_audit["panel_library_counts"][panel_name]
        audit["population_library_confounding"] = (
            "complete for the frozen main trio" if panel_name == "main" else "recorded sensitivity-specific library allocation"
        )
        expectation = {
            "benchmark_role": spec["role"],
            "candidate_class": spec["candidate_class"],
            "candidate_forward_direction": (
                "StoryHatchery RBT proxy P3 -> target P2" if spec["candidate_class"] else None
            ),
            "candidate_basis": spec["candidate_basis"],
            "direction_truth_available": False,
            "exclusive_single_edge_truth_available": False,
            "formal_direction_accuracy_eligible": False,
            "gate_truth_available": False,
            "representation": representation,
            "locus_filter_variant": filter_name,
            "tree_contract_status": (
                "operational proxy order only; long-term hybridization, two-taxon stocking, modern "
                "references, and recurrent backcrossing do not establish a clean ((P1,P2),P3) event"
            ),
        }
        panel = score_panel(
            panel_id,
            prepared_vcf,
            prepared_popmap,
            ("P1", "P2", "P3"),
            audit,
            direction_head[0],
            direction_head[1],
            expectation,
        )
        add_gate_score(panel, gate_head[0], gate_head[1])
        panel["model_free_comparator"] = frequency_projection(
            prepared_vcf,
            manifests[panel_name],
            ("P1", "P2", "P3"),
            bootstrap_replicates=500,
        )
        panel["model_free_comparator"]["finite_sample_and_uncertainty_guardrail"] = (
            "f3 is a plug-in product of sample allele frequencies with no finite-sample correction; "
            "the 29-chromosome block bootstrap resamples chromosomes, not fish, and is descriptive "
            "rather than a formal admixture test"
        )
        panel["adjudication"] = adjudicate_panel(panel, spec["candidate_class"])
        panels.append(panel)
        prepared_paths[(panel_name, representation, filter_name)] = prepared_vcf
    return panels, prepared_paths


def representation_overlap(prepared: dict[tuple[str, str, str], Path]) -> dict:
    result = {}
    for filter_name in ("standard_contract", "within_population_polymorphism"):
        raw_path = prepared[("main", "source_GT", filter_name)]
        pl_path = prepared[("main", "unique_PL_argmin", filter_name)]
        raw_order = []
        pl_keys = set()
        with raw_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.rstrip("\r\n").split("\t")
                raw_order.append((fields[0], fields[1], fields[3], fields[4]))
        with pl_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or not line.strip():
                    continue
                fields = line.rstrip("\r\n").split("\t")
                key = (fields[0], fields[1], fields[3], fields[4])
                if key in pl_keys:
                    raise AssertionError("duplicate PL prepared locus")
                pl_keys.add(key)
        raw_keys = set(raw_order)
        intersection_order = [key for key in raw_order if key in pl_keys]
        intersection_text = "".join("\t".join(key) + "\n" for key in intersection_order)
        union = raw_keys | pl_keys
        result[filter_name] = {
            "source_GT_loci": len(raw_keys),
            "unique_PL_argmin_loci": len(pl_keys),
            "intersection": len(intersection_order),
            "union": len(union),
            "jaccard": len(intersection_order) / len(union),
            "fraction_source_GT_in_PL": len(intersection_order) / len(raw_keys),
            "ordered_intersection_sha256": _sha256_text(intersection_text),
            "guardrail": (
                "the independently filtered GT and PL panels are correlated but not fully paired; "
                "representation and locus ascertainment both change"
            ),
        }
    expected = {
        "standard_contract": (11_758, 12_170, 11_758, "9ef1e1f65321ab60540f2c90e1aa12d4f78ac2449c8686a0a295d78c7e75a274"),
        "within_population_polymorphism": (2_591, 3_653, 2_584, "17ee36a5d5213951e3851d7ece5be6f2a189b8cc97455c3d1248c8ae33c3c662"),
    }
    for name, (raw_n, pl_n, overlap_n, digest) in expected.items():
        observed = result[name]
        if (
            observed["source_GT_loci"],
            observed["unique_PL_argmin_loci"],
            observed["intersection"],
            observed["ordered_intersection_sha256"],
        ) != (raw_n, pl_n, overlap_n, digest):
            raise AssertionError(f"{name}: GT/PL overlap contract changed")
    return result


def summarize_outcomes(panels: list[dict]) -> dict:
    if len(panels) != len(RUN_SPECS):
        raise AssertionError("Yellowstone benchmark panel count changed")
    prediction_counts = Counter(panel["simulation_head"]["predicted_class"] for panel in panels)
    severe = sum(panel["adjudication"]["severe_OOD"] for panel in panels)
    candidate_panels = [panel for panel in panels if panel["adjudication"]["candidate_class"]]
    return {
        "analytic_correlated_sensitivity_rows": len(panels),
        "unique_biological_systems": 1,
        "independent_direction_truth_units": 0,
        "independent_gate_truth_units": 0,
        "correlated_sensitivities_not_trials": True,
        "raw_head_prediction_counts": dict(sorted(prediction_counts.items())),
        "raw_candidate_C_concordant_sensitivity_rows": sum(
            panel["adjudication"]["raw_head_matches_candidate"] is True
            for panel in candidate_panels
        ),
        "candidate_C_sensitivity_rows": len(candidate_panels),
        "accuracy_denominator": None,
        "severe_OOD_panels": severe,
        "abstained_panels": severe,
        "descriptive_nonsevere_panels": len(panels) - severe,
        "accepted_direction_calls": 0,
        "direction_accuracy_estimate": None,
        "gate_accuracy_estimate": None,
        "accuracy_guardrail": (
            "Seven representation/filter/reference/target rows reuse one two-species system and "
            "supply no accuracy denominator. Published ancestry summaries reuse the same SNPs."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    for name in FILES:
        parser.add_argument(f"--{name}")
    parser.add_argument("--source-dir")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    if args.cap < max(value[0] for value in EXPECTED_PANEL_LOCI.values()):
        parser.error(f"--cap must be at least {max(value[0] for value in EXPECTED_PANEL_LOCI.values())}")

    set_below_normal_priority()
    cache = Path(args.cache_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    source_dir = Path(args.source_dir).resolve() if args.source_dir else cache
    cache.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        name: Path(getattr(args, name)).resolve() if getattr(args, name) else source_dir / spec["key"]
        for name, spec in FILES.items()
    }
    verified = {
        name: ensure_source(paths[name], spec, args.download_missing)
        for name, spec in FILES.items()
    }
    sources_record = validate_sources_record()
    metadata_rows, metadata_by_id = read_metadata(paths["metadata"])
    converter_audit = audit_converter(paths["converter"])
    raw_vcf = cache / "yellowstone.normalized.source_GT.vcf"
    pl_vcf = cache / "yellowstone.normalized.unique_PL_argmin.vcf"
    source_samples, source_audit = audit_source_and_materialize_representations(
        paths["vcf"], metadata_by_id, raw_vcf, pl_vcf
    )
    manifests, metadata_audit = metadata_and_manifest_audit(
        metadata_rows, metadata_by_id, source_samples, cache
    )
    external_tables = audit_predictors_and_response(paths["predictors"], paths["response"])
    direction_head = simulation_direction_head(Path(args.data_root).resolve(), max_depth=MAX_DEPTH)
    gate_head = simulation_gate_head(Path(args.data_root).resolve(), max_depth=MAX_DEPTH)
    panels, prepared = run_panels(
        {"source_GT": raw_vcf, "unique_PL_argmin": pl_vcf},
        manifests,
        metadata_audit,
        cache,
        args.cap,
        direction_head,
        gate_head,
    )
    overlap = representation_overlap(prepared)
    result = {
        "schema_version": "dnnaic-yellowstone-2019-external-benchmark-v1",
        "git": git_revision(),
        "runtime": runtime_audit(),
        "guardrail": (
            "One historically admixed two-species system with modern proxy references, complete "
            "population--library confounding, same-data published ancestry summaries, and seven "
            "correlated sensitivity rows. No row is a formal direction or gate accuracy datum."
        ),
        "source": {
            "record": DRYAD_VERSION,
            "data_doi": "10.5061/dryad.6s7d02q",
            "version_id": 29_052,
            "license": "CC0-1.0",
            "paper": PAPER_URL,
            "paper_doi": "10.1111/mec.15175",
            "corrigendum_doi": "10.1111/mec.15381",
            "verified_files": verified,
            "sources_record": sources_record,
            "dryad_v2_archive_audit": {
                "observed_wrapper_bytes": 147_158_006,
                "observed_wrapper_sha256": "63d6767acdf4bffcf1e2a4e995c1ac26d664c34acc800ed97d259aa1d20ee864",
                "entries": 18,
                "unique_paths": 9,
                "duplicate_contract": "each path appears twice with identical size and MD5",
                "trust_boundary": "extracted file bytes, not dynamically generated ZIP wrapper",
            },
            "VCF_audit": source_audit,
            "metadata_audit": metadata_audit,
            "converter_audit": converter_audit,
            "predictor_response_audit": external_tables,
            "corrigendum_audit": (
                "DOI 10.1111/mec.15381 and its existence are confirmed, but the publisher-hosted "
                "correction content was unavailable to this audit; its substantive delta remains unaudited"
            ),
            "locus_ascertainment_guardrail": (
                "the released 12,666 SNPs were globally filtered at at least 60% data and MAF >5% "
                "using all 1,286 fish, including scored cohorts; DNNaic pooled/within-population "
                "filters then use panel labels, so loci are not prospectively held out"
            ),
        },
        "analysis_design": {
            "primary_population_order": PANEL_SPECS["main"]["groups"],
            "primary_sample_counts": PANEL_SPECS["main"]["counts"],
            "primary_candidate_class": "C",
            "candidate_basis": PANEL_SPECS["main"]["candidate_basis"],
            "direction_truth_available": False,
            "exclusive_single_edge_truth_available": False,
            "formal_direction_accuracy_eligible": False,
            "gate_truth_available": False,
            "run_matrix": [
                {
                    "panel": panel,
                    "representation": representation,
                    "locus_filter": filter_name,
                }
                for panel, representation, filter_name, _strict in RUN_SPECS
            ],
            "representation_overlap": overlap,
            "published_same_data_RBT_ancestry_comparator": 1 - 0.2618,
            "published_comparator_guardrail": (
                "1 - site mean YCT ancestry q; derived from these same SNPs, not independent validation"
            ),
            "locus_ascertainment_is_prospective_held_out": False,
        },
        "direction_head": direction_head[2],
        "gate_head": gate_head[2],
        "panels": panels,
        "outcome": summarize_outcomes(panels),
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
                "panels": [
                    {
                        "panel_id": panel["panel_id"],
                        "loci": panel["padze"]["n_loci_kept"],
                        "raw_direction": panel["simulation_head"]["predicted_class"],
                        "raw_gate": panel["simulation_gate"]["appreciable_score"],
                        "direction_rms_z": panel["simulation_feature_shift"]["rms_z"],
                        "gate_rms_z": panel["simulation_gate_feature_shift"]["rms_z"],
                        "projection": panel["model_free_comparator"]["P2_projection_from_P1_toward_P3_all_loci"],
                        "status": panel["adjudication"]["natural_data_call_status"],
                    }
                    for panel in panels
                ],
                "outcome": result["outcome"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
