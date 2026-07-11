#!/usr/bin/env python3
"""Run a guarded 2024 *Harpagifer antarcticus* transfer stress test.

Bernal-Duran et al. (2024) released a 143-sample neutral-SNP VCF and raw
Lagrangian particle matrices for the Western Antarctic Peninsula.  Two ROMS
directions align with biologically sensible three-site orders: Doumer Island
(DOI) to Foyn Harbour (FHA), with Horseshoe Island (HOS) as P3; and Adelaide
Island (AIS) to HOS, with DOI as P3.  Each forward edge exceeds its reciprocal
in every modelled season and defines a candidate DNNaic class-A orientation.

These are not introgression truth or accuracy trials.  ROMS estimates potential
passive-larval settlement in 2008--2012, not realized historical gene flow; the
system has many simultaneous edges; and the paper describes four population
groups rather than a rooted three-population species tree.  The released prose
and the labelled thesis figure also disagree about connectivity normalization,
so both raw settlement rate and destination-conditional share are reported.
Every severe-OOD learned output abstains.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import urllib.request
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
    called_alleles,
    git_revision,
    open_text,
    prepare_vcf,
    read_manifest,
    score_panel,
    set_below_normal_priority,
    sha256_file,
    simulation_direction_head,
    verify_file,
)
from harpagifer_external_benchmark import frequency_geometry
import tinkerbird_external_benchmark as runtime_helpers


DEFAULT_CACHE = REPO / "data" / "real" / "hantarcticus_2024_external_benchmark"
DEFAULT_RESULTS = REPO / "results" / "hantarcticus_2024_external_benchmark_2026_07_11"
DEFAULT_CAP = 20_778
PREFIX_RECORD = MANIFEST_DIR / "hantarcticus_2024" / "population_prefixes.tsv"
SOURCES_RECORD = MANIFEST_DIR / "hantarcticus_2024" / "sources.json"

DRYAD_RECORD = "https://datadryad.org/dataset/doi:10.5061/dryad.b5mkkwhjk"
ZENODO_RECORD = "https://zenodo.org/records/11109021"
FILES = {
    "vcf": {
        "name": "Hantarcticus_20778_143_neutralSNPs.vcf",
        "bytes": 12_581_530,
        "sha256": "48d832ade62ef3ad21ced7869e6f2a9e5c418593978e6260725be0ba02f998a5",
        "download": (
            "https://zenodo.org/api/records/11109021/files/"
            "Hantarcticus_20778_143_neutralSNPs.vcf/content"
        ),
    },
    "matrices": {
        "name": "biophysical_matrices_Hantarcticus.zip",
        "bytes": 3_705_582,
        "sha256": "3ac56229b68ff9c77de9517015e52dfa766bc3e5590cd4b5e502e8a6aefb3456",
        "download": (
            "https://zenodo.org/api/records/11109021/files/"
            "biophysical_matrices_Hantarcticus.zip/content"
        ),
    },
    "readme": {
        "name": "README.md",
        "bytes": 6_358,
        "sha256": "55464f867352d1f99db2fcabdd43e48efb524525667400396d99815cf8068bdc",
        "download": "https://zenodo.org/api/records/11109021/files/README.md/content",
    },
}
SOURCE_CONTRACT = {
    "samples": 143,
    "variant_rows": 20_778,
    "genotype_cells": 2_971_254,
    "fully_missing_genotype_cells": 344_218,
    "ordered_sample_sha256": "68545e930910390cf662179939afdc1c5073a42cce269ee485cfc93991be3bcd",
    "locus_semantic_sha256": "979472625148268308c23f592bb792b05955143252149af992a63f008782718e",
}
PAPER_TABLE1_COUNTS = {
    "SIG": 2,
    "FIB": 9,
    "CHB": 12,
    "DIS": 18,
    "BST": 20,
    "GRE": 2,
    "FHA": 18,
    "PLO": 13,
    "DOI": 10,
    "AIS": 20,
    "HOS": 9,
}
EXPECTED_SITE_COUNTS = dict(PAPER_TABLE1_COUNTS, FIB=19)
MODEL_AXIS_ORDER = (
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
PAPER_DISPLAY_ORIGIN_ORDER = (
    "NKG",
    "FIB",
    "CHB",
    "DIS",
    "BST",
    "FHA",
    "GRE",
    "DOI",
    "AIS",
    "HOS",
    "MOT",
    "MIN",
)
PAPER_DISPLAY_DESTINATION_TOP_TO_BOTTOM = (
    "MIN",
    "MOT",
    "HOS",
    "AIS",
    "DOI",
    "GRE",
    "FHA",
    "BST",
    "DIS",
    "CHB",
    "FIB",
    "NKG",
)
ARCHIVE_TO_PAPER_ORIGIN_INDICES = (0, 1, 2, 3, 4, 5, 10, 6, 7, 11, 8, 9)
RAW_ARCHIVE_TO_PAPER_ROW_INDICES = (2, 3, 0, 4, 5, 1, 6, 7, 8, 9, 10, 11)
SEASONS = ("Eday_1050", "Eday_1415", "Eday_1780", "Eday_2145")
ARCHIVE_SEMANTIC_CONTRACT = {
    "nonmetadata_member_name_size_crc_sha256": "77520ae3e34eec41ddd5273fc49481ac5dc9dac06113c66d3c6f1bff0714f189",
    "day100_inventory_sha256": "17215bc00c9e2544aa6f238c1420fe3b549a8e8cb5d6c7f88d56b70f7d699f36",
    "sum_40_day100_matrices_sha256": "4a02f1a77ff8b7630772e6969e959cf479cb44b6ecb5089fb7d913cd0f7133d0",
}
DAILY_MATRIX_RE = re.compile(
    r"^mat/(Eday_\d+)/float_connectivity/"
    r"float_connectivity_(r\d\d)_(\d{4}-\d{2}-\d{2})_Eday_\d+\.txt$"
)
MEAN_TRAJECTORY_RE = re.compile(
    r"^mat/(Eday_\d+)/float_connectivity/float_connectivity_mean_global_Eday_\d+\.txt$"
)

PANELS = {
    "doi_to_fha_hos": {
        "population_order": ("DOI", "FHA", "HOS"),
        "external_edge": ("DOI", "FHA"),
        "sample_counts": {"DOI": 10, "FHA": 18, "HOS": 9},
        "evidence_tier": "primary",
    },
    "ais_to_hos_doi": {
        "population_order": ("AIS", "HOS", "DOI"),
        "external_edge": ("AIS", "HOS"),
        "sample_counts": {"AIS": 20, "HOS": 9, "DOI": 10},
        "evidence_tier": "secondary_weaker_directional_sensitivity",
    },
}
EXPECTED_FILTERS = {
    "doi_to_fha_hos": {
        "standard_contract": {
            "loci": 16_299,
            "ordered_locus_sha256": "bd8019539f8bda8108a8a204fd7f610ba492ceabc3b4438fd7137bb1183cce86",
        },
        "within_population_polymorphism": {
            "loci": 12_074,
            "ordered_locus_sha256": "32bdc7739277a66b4f83faa33f8be768f9210a7eed2528ffc3d26872ac74c20f",
        },
    },
    "ais_to_hos_doi": {
        "standard_contract": {
            "loci": 16_301,
            "ordered_locus_sha256": "8a161bcfdf0d4ab393841f0167b111e74e1793543c793cd34a6ebe5f2504a31f",
        },
        "within_population_polymorphism": {
            "loci": 11_931,
            "ordered_locus_sha256": "23d34f029bebba494acd96795d78bf633c63a92aada49c4a8b5ef87387e65933",
        },
    },
}


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DNNaic-audit/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(output)


def ensure_sources(paths: dict[str, Path], download_missing: bool) -> dict:
    verified = {}
    for key, contract in FILES.items():
        path = paths[key]
        if not path.exists():
            if not download_missing:
                raise FileNotFoundError(path)
            _download(contract["download"], path)
        verified[key] = verify_file(path, contract["bytes"], contract["sha256"])
    return verified


def ordered_sample_sha256(samples: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample}\n" for sample in samples).encode("utf-8")
    ).hexdigest()


def read_prefix_record(path: Path = PREFIX_RECORD) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = ("HAS", "HA", "HAC", "HAP", "HAD", "HAO", "HGR", "HFH", "HAL", "HAY", "HAR", "HLT")
    if tuple(row["vcf_prefix"] for row in rows) != expected:
        raise AssertionError("H. antarcticus prefix-record order changed")
    if sum(int(row["VCF_n"]) for row in rows) != SOURCE_CONTRACT["samples"]:
        raise AssertionError("H. antarcticus prefix counts no longer sum to 143")
    return rows


def sample_site(sample: str, rows: list[dict[str, str]] | None = None) -> str:
    rows = read_prefix_record() if rows is None else rows
    matches = [row for row in rows if sample.startswith(row["vcf_prefix"] + "_")]
    if len(matches) != 1:
        raise ValueError(f"{sample}: expected exactly one H. antarcticus prefix match")
    return matches[0]["benchmark_site"]


def read_vcf_samples(path: Path) -> list[str]:
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                if len(samples) != len(set(samples)):
                    raise AssertionError("H. antarcticus VCF sample IDs are not unique")
                return samples
    raise ValueError(f"{path}: no #CHROM header")


def reconstruct_sample_mapping(source: Path) -> tuple[dict[str, str], dict]:
    rows = read_prefix_record()
    samples = read_vcf_samples(source)
    digest = ordered_sample_sha256(samples)
    if len(samples) != SOURCE_CONTRACT["samples"] or digest != SOURCE_CONTRACT[
        "ordered_sample_sha256"
    ]:
        raise AssertionError("released H. antarcticus sample-order contract changed")
    mapping = {sample: sample_site(sample, rows) for sample in samples}
    counts = {site: list(mapping.values()).count(site) for site in dict.fromkeys(mapping.values())}
    if counts != EXPECTED_SITE_COUNTS:
        raise AssertionError(f"unexpected H. antarcticus prefix counts: {counts}")
    return mapping, {
        "status": "prefix_mapping_with_cross_source_corrections",
        "ordered_sample_sha256": digest,
        "VCF_site_counts": counts,
        "paper_Table1_site_counts": PAPER_TABLE1_COUNTS,
        "paper_Table1_total": sum(PAPER_TABLE1_COUNTS.values()),
        "VCF_total": len(samples),
        "known_discrepancies": [
            "VCF HGR_01/HGR_02 correspond to Green Reef although the README says HGE.",
            "README HAR=Alexander conflicts with the paper/model AIS=Adelaide; matching GBIF project occurrences place HAR samples in the Adelaide/Rothera region.",
            "The VCF includes 10 HAC Fildes Bay samples omitted from the paper Table 1 total of 133.",
        ],
        "guardrail": (
            "prefix membership is reconstructed from the released VCF/README and independent "
            "same-project GBIF records; it is not an author-supplied VCF population map"
        ),
    }


def audit_source_vcf(path: Path, sites: dict[str, str]) -> dict:
    samples: list[str] | None = None
    site_columns: dict[str, list[int]] | None = None
    rows = 0
    genotype_cells = 0
    missing_cells = 0
    invalid_cells = 0
    positions: list[int] = []
    identifiers: set[str] = set()
    chromosomes: set[str] = set()
    formats: set[str] = set()
    filters: set[str] = set()
    qualities: set[str] = set()
    infos: set[str] = set()
    missing_by_sample = {sample: 0 for sample in sites}
    minimum_called = {site: 10**9 for site in EXPECTED_SITE_COUNTS}
    maximum_called = {site: 0 for site in EXPECTED_SITE_COUNTS}
    semantic = hashlib.sha256()
    reference_unknown_major_used = False
    allowed = {"./.", "0/0", "0/1", "1/0", "1/1"}

    with open_text(path) as handle:
        for line in handle:
            if line.startswith("##Tassel=") and "Reference allele is not known" in line:
                reference_unknown_major_used = True
            if line.startswith("#CHROM"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                if samples != list(sites):
                    raise AssertionError("H. antarcticus source order differs from mapping")
                site_columns = {
                    site: [9 + i for i, sample in enumerate(samples) if sites[sample] == site]
                    for site in EXPECTED_SITE_COUNTS
                }
                continue
            if line.startswith("#") or not line.strip():
                continue
            if samples is None or site_columns is None:
                raise ValueError("H. antarcticus variant before #CHROM")
            fields = line.rstrip("\r\n").split("\t")
            rows += 1
            if len(fields) != 9 + len(samples):
                raise AssertionError("H. antarcticus VCF row width changed")
            chromosomes.add(fields[0])
            position = int(fields[1])
            positions.append(position)
            identifiers.add(fields[2])
            semantic.update("\t".join(fields[:5]).encode("utf-8"))
            semantic.update(b"\n")
            qualities.add(fields[5])
            filters.add(fields[6])
            infos.add(fields[7])
            formats.add(fields[8])
            if (
                len(fields[3]) != 1
                or len(fields[4]) != 1
                or "," in fields[4]
                or fields[3] not in "ACGT"
                or fields[4] not in "ACGT"
                or fields[3] == fields[4]
            ):
                raise AssertionError("H. antarcticus source is not biallelic SNP-only")
            if fields[2] != f"TP{position}":
                raise AssertionError("H. antarcticus locus IDs no longer encode POS")
            for sample, cell in zip(samples, fields[9:]):
                genotype_cells += 1
                if cell not in allowed:
                    invalid_cells += 1
                elif cell == "./.":
                    missing_cells += 1
                    missing_by_sample[sample] += 1
            for site, columns in site_columns.items():
                copies = sum(len(called_alleles(fields[index])) for index in columns)
                minimum_called[site] = min(minimum_called[site], copies)
                maximum_called[site] = max(maximum_called[site], copies)

    if samples is None or rows != SOURCE_CONTRACT["variant_rows"]:
        raise AssertionError("unexpected H. antarcticus source dimensions")
    if genotype_cells != SOURCE_CONTRACT["genotype_cells"]:
        raise AssertionError("unexpected H. antarcticus genotype-cell count")
    if missing_cells != SOURCE_CONTRACT["fully_missing_genotype_cells"] or invalid_cells:
        raise AssertionError("unexpected H. antarcticus missing/invalid GT contract")
    if chromosomes != {"0"} or formats != {"GT"} or filters != {"PASS"}:
        raise AssertionError("unexpected H. antarcticus structural VCF fields")
    if qualities != {"."} or infos != {"."}:
        raise AssertionError("unexpected H. antarcticus QUAL/INFO fields")
    if positions != sorted(positions) or len(positions) != len(set(positions)):
        raise AssertionError("H. antarcticus positions are not sorted and unique")
    if len(identifiers) != rows:
        raise AssertionError("H. antarcticus locus IDs are not unique")
    if semantic.hexdigest() != SOURCE_CONTRACT["locus_semantic_sha256"]:
        raise AssertionError("H. antarcticus locus semantic digest changed")
    if not reference_unknown_major_used:
        raise AssertionError("H. antarcticus unknown-reference header guardrail changed")

    sample_missingness = {sample: missing_by_sample[sample] / rows for sample in samples}
    if any(value > 0.25 for value in sample_missingness.values()):
        raise AssertionError("unexpected H. antarcticus sample above 25% missingness")
    return {
        "samples": len(samples),
        "variant_rows": rows,
        "CHROM_values": sorted(chromosomes),
        "position_range": [min(positions), max(positions)],
        "unique_positions": len(set(positions)),
        "unique_IDs": len(identifiers),
        "ordered_CHROM_POS_ID_REF_ALT_sha256": semantic.hexdigest(),
        "FORMAT": sorted(formats),
        "FILTER": sorted(filters),
        "genotype_cells": genotype_cells,
        "fully_missing_genotype_cells": missing_cells,
        "missing_genotype_fraction": missing_cells / genotype_cells,
        "partial_or_invalid_genotype_cells": invalid_cells,
        "maximum_sample_missingness": max(sample_missingness.values()),
        "samples_above_0_25_missingness": [],
        "called_copy_range_by_site": {
            site: {"minimum": minimum_called[site], "maximum": maximum_called[site]}
            for site in EXPECTED_SITE_COUNTS
        },
        "linkage_guardrail": (
            "all released loci use CHROM=0 and no reference linkage map is supplied; "
            "physical independence cannot be verified"
        ),
        "allele_orientation_guardrail": (
            "the Tassel header states that the reference allele is unknown and the major allele "
            "was encoded as REF; REF must not be interpreted as ancestral"
        ),
    }


def _matrix_from_member(archive: zipfile.ZipFile, member: str) -> np.ndarray:
    matrix = np.loadtxt(io.BytesIO(archive.read(member)), delimiter=",")
    if matrix.shape != (12, 12):
        raise AssertionError(f"{member}: expected a 12x12 matrix")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise AssertionError(f"{member}: non-finite or negative particle count")
    if not np.all(matrix == np.floor(matrix)):
        raise AssertionError(f"{member}: daily particle matrix is not integer-valued")
    # The text files store the plotted y-axis from top to bottom (MIN to NKG),
    # whereas array row zero below denotes the first labelled destination NKG.
    return matrix[::-1, :]


def _edge_summary(
    season_D: dict[str, np.ndarray],
    day100_runs: dict[tuple[str, str], np.ndarray],
    origin: str,
    destination: str,
) -> dict:
    origin_index = MODEL_AXIS_ORDER.index(origin)
    destination_index = MODEL_AXIS_ORDER.index(destination)
    reciprocal_origin_index = destination_index
    reciprocal_destination_index = origin_index
    per_season = []
    for season in SEASONS:
        matrix = season_D[season]
        destination_total = float(matrix[destination_index, :].sum())
        reciprocal_total = float(matrix[reciprocal_destination_index, :].sum())
        forward = float(matrix[destination_index, origin_index])
        reverse = float(matrix[reciprocal_destination_index, reciprocal_origin_index])
        per_season.append(
            {
                "season": season,
                "mean_day100_settlers_forward": forward,
                "mean_day100_settlers_reciprocal": reverse,
                "forward_fraction_of_100_released": forward / 100.0,
                "reciprocal_fraction_of_100_released": reverse / 100.0,
                "forward_destination_conditional_share": (
                    forward / destination_total if destination_total > 0 else None
                ),
                "reciprocal_destination_conditional_share": (
                    reverse / reciprocal_total if reciprocal_total > 0 else None
                ),
                "destination_total_mean_settlers": destination_total,
                "reciprocal_destination_total_mean_settlers": reciprocal_total,
            }
        )
    mean_D = np.mean([season_D[season] for season in SEASONS], axis=0)
    forward = float(mean_D[destination_index, origin_index])
    reverse = float(mean_D[reciprocal_destination_index, reciprocal_origin_index])
    destination_total = float(mean_D[destination_index, :].sum())
    reciprocal_total = float(mean_D[reciprocal_destination_index, :].sum())
    run_values = [
        float(day100_runs[(season, run)][destination_index, origin_index])
        for season in SEASONS
        for run in (f"r{i:02d}" for i in range(1, 11))
    ]
    reciprocal_run_values = [
        float(day100_runs[(season, run)][reciprocal_destination_index, reciprocal_origin_index])
        for season in SEASONS
        for run in (f"r{i:02d}" for i in range(1, 11))
    ]
    conditional_forward = [
        row["forward_destination_conditional_share"]
        for row in per_season
        if row["forward_destination_conditional_share"] is not None
    ]
    conditional_reverse = [
        row["reciprocal_destination_conditional_share"]
        for row in per_season
        if row["reciprocal_destination_conditional_share"] is not None
    ]
    return {
        "origin": origin,
        "destination": destination,
        "candidate_direction": f"{origin} -> {destination}",
        "origin_axis_index_zero_based": origin_index,
        "destination_axis_index_zero_based": destination_index,
        "seasons": per_season,
        "all_four_seasons_forward_positive": all(
            row["mean_day100_settlers_forward"] > 0 for row in per_season
        ),
        "all_four_seasons_forward_exceeds_reciprocal": all(
            row["mean_day100_settlers_forward"]
            > row["mean_day100_settlers_reciprocal"]
            for row in per_season
        ),
        "all_four_seasons_forward_destination_share_exceeds_reciprocal": all(
            row["forward_destination_conditional_share"]
            > (row["reciprocal_destination_conditional_share"] or 0.0)
            for row in per_season
        ),
        "raw_day100_settlers_across_40_runs": int(sum(run_values)),
        "raw_reciprocal_day100_settlers_across_40_runs": int(sum(reciprocal_run_values)),
        "runs_with_forward_settlers": sum(value > 0 for value in run_values),
        "runs_with_reciprocal_settlers": sum(value > 0 for value in reciprocal_run_values),
        "four_season_mean_day100_settlers": forward,
        "four_season_mean_reciprocal_day100_settlers": reverse,
        "four_season_mean_fraction_of_100_released": forward / 100.0,
        "four_season_mean_reciprocal_fraction_of_100_released": reverse / 100.0,
        "mean_of_season_destination_conditional_shares": (
            float(np.mean(conditional_forward)) if conditional_forward else None
        ),
        "mean_of_season_reciprocal_destination_conditional_shares": (
            float(np.mean(conditional_reverse)) if conditional_reverse else None
        ),
        "pooled_destination_conditional_share": (
            forward / destination_total if destination_total > 0 else None
        ),
        "pooled_reciprocal_destination_conditional_share": (
            reverse / reciprocal_total if reciprocal_total > 0 else None
        ),
        "normalization_guardrail": (
            "mean settler counts divided by 100 reproduce the scale of the labelled thesis "
            "connectivity percentages, while the prose definition implies destination-conditional "
            "normalization; both are reported and neither is a genomic migration rate"
        ),
    }


def audit_biophysical_archive(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        daily: dict[tuple[str, str], list[tuple[str, str]]] = {}
        mean_trajectory_members = []
        content_digest = hashlib.sha256()
        for info in archive.infolist():
            if info.is_dir() or info.filename.startswith("__MACOSX/") or info.filename.endswith(".DS_Store"):
                continue
            content_digest.update(
                f"{info.filename}\t{info.file_size}\t{info.CRC}\n".encode("utf-8")
            )
            match = DAILY_MATRIX_RE.match(info.filename)
            if match:
                daily.setdefault((match.group(1), match.group(2)), []).append(
                    (match.group(3), info.filename)
                )
            elif MEAN_TRAJECTORY_RE.match(info.filename):
                mean_trajectory_members.append(info.filename)
            else:
                raise AssertionError(f"unexpected biophysical archive member: {info.filename}")
        if len(names) != 8_042:
            raise AssertionError("H. antarcticus ZIP entry count changed")
        if len(daily) != 40 or sum(len(values) for values in daily.values()) != 4_000:
            raise AssertionError("expected 4 seasons x 10 runs x 100 daily matrices")
        if len(mean_trajectory_members) != 4:
            raise AssertionError("expected four mean-global trajectory matrices")
        if tuple(sorted({season for season, _ in daily})) != SEASONS:
            raise AssertionError("H. antarcticus model seasons changed")

        day100_runs: dict[tuple[str, str], np.ndarray] = {}
        run_audit = []
        day100_inventory_digest = hashlib.sha256()
        for season in SEASONS:
            for number in range(1, 11):
                run = f"r{number:02d}"
                members = sorted(daily[(season, run)])
                if len(members) != 100 or len({date for date, _ in members}) != 100:
                    raise AssertionError(f"{season}/{run}: expected 100 unique daily matrices")
                # Parse every member so malformed unused days cannot hide in a pinned archive.
                parsed = [_matrix_from_member(archive, member) for _, member in members]
                day100_runs[(season, run)] = parsed[-1]
                day100_inventory_digest.update(
                    f"{season}\t{run}\t{members[-1][0]}\t{members[-1][1]}\n".encode("utf-8")
                )
                run_audit.append(
                    {
                        "season": season,
                        "run": run,
                        "first_date": members[0][0],
                        "day100_date": members[-1][0],
                        "daily_matrices": len(members),
                        "day100_member": members[-1][1],
                    }
                )

        season_D = {
            season: np.mean(
                [day100_runs[(season, f"r{number:02d}")] for number in range(1, 11)],
                axis=0,
            )
            for season in SEASONS
        }
        summed_day100 = np.sum(list(day100_runs.values()), axis=0).astype(np.int64)
        matrix_digest = hashlib.sha256()
        matrix_digest.update(
            ("destination\\origin\t" + "\t".join(MODEL_AXIS_ORDER) + "\n").encode("utf-8")
        )
        for destination, row in zip(MODEL_AXIS_ORDER, summed_day100):
            matrix_digest.update(
                (destination + "\t" + "\t".join(str(int(value)) for value in row) + "\n").encode(
                    "utf-8"
                )
            )
        edges = {
            panel_id: _edge_summary(
                season_D,
                day100_runs,
                *spec["external_edge"],
            )
            for panel_id, spec in PANELS.items()
        }

    semantic_observed = {
        "nonmetadata_member_name_size_crc_sha256": content_digest.hexdigest(),
        "day100_inventory_sha256": day100_inventory_digest.hexdigest(),
        "sum_40_day100_matrices_sha256": matrix_digest.hexdigest(),
    }
    if semantic_observed != ARCHIVE_SEMANTIC_CONTRACT:
        raise AssertionError(f"biophysical archive semantic contract changed: {semantic_observed}")
    expected_edge_contract = {
        "doi_to_fha_hos": (92, 10, 32, 7, 2.3, 0.25),
        "ais_to_hos_doi": (50, 13, 19, 7, 1.25, 0.325),
    }
    for panel_id, expected in expected_edge_contract.items():
        edge = edges[panel_id]
        observed = (
            edge["raw_day100_settlers_across_40_runs"],
            edge["raw_reciprocal_day100_settlers_across_40_runs"],
            edge["runs_with_forward_settlers"],
            edge["runs_with_reciprocal_settlers"],
            edge["four_season_mean_day100_settlers"],
            edge["four_season_mean_reciprocal_day100_settlers"],
        )
        if observed != expected:
            raise AssertionError(f"{panel_id}: biophysical edge contract changed: {observed}")
    return {
        "zip_entries_including_directories_and_macos_metadata": len(names),
        "daily_connectivity_matrices": 4_000,
        "mean_global_trajectory_matrices": 4,
        "seasons": list(SEASONS),
        "runs_per_season": 10,
        "daily_matrices_per_run": 100,
        **semantic_observed,
        "model_axis_order": list(MODEL_AXIS_ORDER),
        "paper_display_origin_order": list(PAPER_DISPLAY_ORIGIN_ORDER),
        "paper_display_destination_top_to_bottom": list(
            PAPER_DISPLAY_DESTINATION_TOP_TO_BOTTOM
        ),
        "archive_to_paper_origin_indices": list(ARCHIVE_TO_PAPER_ORIGIN_INDICES),
        "raw_archive_to_paper_row_indices": list(RAW_ARCHIVE_TO_PAPER_ROW_INDICES),
        "axis_mapping_status": (
            "reconstructed from the thesis's labelled old-ten axes plus exact current-paper "
            "Figure 3 cell/archive cross-checks; DOI/FHA use labelled old-ten positions, while "
            "secondary AIS/HOS additionally depends on the verified appended HOS crosswalk"
        ),
        "raw_row_orientation": (
            "stored rows are reverse plotted-destination order and are flipped before indexing; "
            "columns are origin in labelled x-axis order"
        ),
        "aggregation": (
            "take the chronologically last (day-100) matrix for each release run, average 10 run "
            "D matrices within each season, then average four seasonal summaries"
        ),
        "run_inventory": run_audit,
        "candidate_edges": edges,
        "source_implementation_discrepancy": {
            "status": "published_figure_implementation_resolved_but_prose_conflicts",
            "methods_semantics": (
                "C is described as the share of particles settling at destination j that originated at i"
            ),
            "figure_numeric_scale": (
                "every inspected current-paper Figure 3 cell exactly equals its seasonal mean D; "
                "with 100 particles released per source/run, this is percentage points of source "
                "releases and matches the explicitly percent-labelled thesis precursor"
            ),
            "benchmark_policy": (
                "report raw settlement rate, per-season and pooled destination shares separately; "
                "candidate direction is accepted only because forward exceeds reciprocal in every "
                "season under both representations"
            ),
        },
    }


def materialize_manifests(
    source: Path,
    sample_sites: dict[str, str],
    output_dir: Path,
) -> tuple[dict[str, Path], dict]:
    samples = read_vcf_samples(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    audits = {}
    for panel_id, spec in PANELS.items():
        order = spec["population_order"]
        selected = [sample for sample in samples if sample_sites[sample] in order]
        path = output_dir / f"{panel_id}.manifest.tsv"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("sample\tpopulation\n")
            for sample in selected:
                handle.write(f"{sample}\t{sample_sites[sample]}\n")
        mapping = read_manifest(path)
        counts = {population: list(mapping.values()).count(population) for population in order}
        if counts != spec["sample_counts"]:
            raise AssertionError(f"{panel_id}: unexpected manifest counts {counts}")
        paths[panel_id] = path
        audits[panel_id] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "samples": len(mapping),
            "population_counts": counts,
            "selection": "all released VCF columns from the three named prefix-defined sites",
        }
    return paths, audits


def run_panels(
    source: Path,
    manifests: dict[str, Path],
    biophysical_audit: dict,
    cache: Path,
    cap: int,
    direction_head,
    gate_head,
) -> list[dict]:
    panels = []
    for panel_id, spec in PANELS.items():
        manifest = manifests[panel_id]
        population_order = spec["population_order"]
        external_edge = biophysical_audit["candidate_edges"][panel_id]
        for filter_name, strict in (
            ("standard_contract", False),
            ("within_population_polymorphism", True),
        ):
            panel_vcf = cache / f"{panel_id}.{filter_name}.vcf"
            panel_popmap = cache / f"{panel_id}.{filter_name}.popmap.tsv"
            audit = prepare_vcf(
                source,
                manifest,
                panel_vcf,
                panel_popmap,
                cap=cap,
                seed=20260711,
                min_called_copies=16,
                polymorphic_within_each_population=strict,
            )
            expected = EXPECTED_FILTERS[panel_id][filter_name]
            if audit["counts"]["retained_after_cap"] != expected["loci"]:
                raise AssertionError(f"{panel_id}/{filter_name}: unexpected locus count")
            if audit["ordered_locus_sha256"] != expected["ordered_locus_sha256"]:
                raise AssertionError(f"{panel_id}/{filter_name}: unexpected locus hash")
            expectation = {
                "benchmark_role": "independent_biophysical_direction_stress_test",
                "candidate_class": "A",
                "candidate_forward_direction": (
                    f"{population_order[0]} (P1) -> {population_order[1]} (P2)"
                ),
                "expected_gate": None,
                "direction_basis": (
                    "an independently simulated ROMS passive-larval edge exceeds its reciprocal "
                    "in all four seasons under both source-release and destination-conditional "
                    "representations"
                ),
                "evidence_tier": spec["evidence_tier"],
                "external_edge_evidence": external_edge,
                "label_source_reuse": (
                    "the particle matrices are numerically independent of the SNPs, but geography, "
                    "site choice, and biological interpretation are shared with the source study"
                ),
                "temporal_guardrail": (
                    "ROMS seasons span 2008-2012 and represent potential 100-day passive larval "
                    "transport, whereas adults were sampled across years and SNPs integrate deeper history"
                ),
                "locus_filter_variant": (
                    "both alleles called within P1, P2, and P3; strong ascertainment"
                    if strict
                    else "both alleles called across the complete three-population panel"
                ),
                "tree_contract_status": (
                    "operational ((P1,P2),P3) order places central-WAP DOI/FHA or southern AIS/HOS "
                    "as the closer geographic/genetic pair; it is not a rooted species-tree contract"
                ),
                "multi_edge_guardrail": (
                    "the oceanographic system contains many simultaneous edges and is outside the "
                    "single-pulse training histories"
                ),
                "accuracy_eligible": False,
            }
            scored = score_panel(
                f"hantarcticus_2024_{panel_id}_{filter_name}",
                panel_vcf,
                panel_popmap,
                population_order,
                audit,
                direction_head[0],
                direction_head[1],
                expectation,
            )
            scored["population_order"]["tree_contract_status"] = expectation[
                "tree_contract_status"
            ]
            add_gate_score(scored, gate_head[0], gate_head[1])
            scored["model_free_comparator"] = frequency_geometry(
                panel_vcf,
                manifest,
                population_order,
            )
            direction_rms = scored["simulation_feature_shift"]["rms_z"]
            gate_rms = scored["simulation_gate_feature_shift"]["rms_z"]
            severe = max(direction_rms, gate_rms) > 10
            prediction = scored["simulation_head"]["predicted_class"]
            scored["adjudication"] = {
                "candidate_class": "A",
                "predicted_class": prediction,
                "matches_candidate_reference": prediction == "A",
                "gate_score": scored["simulation_gate"]["appreciable_score"],
                "accuracy_eligible": False,
                "severe_OOD": severe,
                "severe_OOD_rule": (
                    "max(direction RMS-z, gate RMS-z) > 10; heuristic, not calibrated support"
                ),
                "natural_data_call_status": (
                    "abstain_severe_OOD" if severe else "diagnostic_only"
                ),
            }
            panels.append(scored)
    return panels


def summarize_outcomes(panels: list[dict]) -> dict:
    abstained = sum(
        panel["adjudication"]["natural_data_call_status"] == "abstain_severe_OOD"
        for panel in panels
    )
    return {
        "analytic_sensitivity_runs": len(panels),
        "unique_biological_systems": 1,
        "correlated_candidate_comparisons": 2,
        "independent_validation_panels": 0,
        "accuracy_estimate": None,
        "accuracy_available": False,
        "candidate_class": "A",
        "candidate_label_status": "independent_biophysical_direction_not_genomic_truth",
        "severe_OOD_panels": sum(panel["adjudication"]["severe_OOD"] for panel in panels),
        "abstained_panels": abstained,
        "all_panels_abstain_due_to_severe_OOD": abstained == len(panels),
        "raw_OOD_head_matches_candidate_A": sum(
            panel["adjudication"]["matches_candidate_reference"] for panel in panels
        ),
        "raw_OOD_head_prediction_counts": {
            label: sum(
                panel["simulation_head"]["predicted_class"] == label for panel in panels
            )
            for label in ("A", "B", "C")
        },
        "raw_OOD_gate_threshold_crossings_at_0.5": sum(
            panel["simulation_gate"]["called_at_0.5"] for panel in panels
        ),
        "interpretation": (
            "two correlated oceanographic comparisons from one system, each repeated "
            "under two locus-ascertainment filters; run counts are not accuracy trials"
        ),
    }


def validate_sources_record(path: Path = SOURCES_RECORD) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["data_doi"] != "10.5061/dryad.b5mkkwhjk":
        raise AssertionError("H. antarcticus source DOI changed")
    for key, contract in FILES.items():
        record_key = "biophysical_matrices" if key == "matrices" else key
        if record["files"][record_key]["sha256"] != contract["sha256"]:
            raise AssertionError(f"H. antarcticus {key} source-record hash changed")
    if record["mapping_guardrails"]["sample_order_sha256"] != SOURCE_CONTRACT[
        "ordered_sample_sha256"
    ]:
        raise AssertionError("H. antarcticus source-record sample hash changed")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    parser.add_argument("--source-vcf")
    parser.add_argument("--matrices-zip")
    parser.add_argument("--source-readme")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    if args.cap < DEFAULT_CAP:
        parser.error(f"--cap must be at least {DEFAULT_CAP} to preserve frozen locus contracts")

    set_below_normal_priority()
    cache = Path(args.cache_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "vcf": Path(args.source_vcf).resolve() if args.source_vcf else cache / FILES["vcf"]["name"],
        "matrices": (
            Path(args.matrices_zip).resolve()
            if args.matrices_zip
            else cache / FILES["matrices"]["name"]
        ),
        "readme": (
            Path(args.source_readme).resolve()
            if args.source_readme
            else cache / FILES["readme"]["name"]
        ),
    }
    verified = ensure_sources(paths, args.download_missing)
    sources_record = validate_sources_record()
    sample_sites, mapping_audit = reconstruct_sample_mapping(paths["vcf"])
    source_audit = audit_source_vcf(paths["vcf"], sample_sites)
    biophysical_audit = audit_biophysical_archive(paths["matrices"])
    manifests, manifest_audit = materialize_manifests(
        paths["vcf"], sample_sites, cache / "manifests"
    )

    data_root = Path(args.data_root).resolve()
    direction_head = simulation_direction_head(data_root, max_depth=MAX_DEPTH)
    gate_head = simulation_gate_head(data_root, max_depth=MAX_DEPTH)
    panels = run_panels(
        paths["vcf"],
        manifests,
        biophysical_audit,
        cache,
        args.cap,
        direction_head,
        gate_head,
    )
    result = {
        "schema_version": "dnnaic-hantarcticus-2024-external-benchmark-v1",
        "git": git_revision(),
        "runtime": runtime_helpers.runtime_audit(),
        "guardrail": (
            "independent biophysical direction stress test only; passive-larval ROMS transport is "
            "not introgression truth, comparisons are correlated and multi-edge, no panel is "
            "accuracy-eligible, and severe-OOD outputs abstain"
        ),
        "source": {
            "record": DRYAD_RECORD,
            "access_mirror": ZENODO_RECORD,
            "data_doi": "10.5061/dryad.b5mkkwhjk",
            "publication_doi": "10.1111/mec.17360",
            "license": "CC0-1.0",
            "verified": verified,
            "sources_record": {
                "path": str(SOURCES_RECORD),
                "sha256": sha256_file(SOURCES_RECORD),
                "content": sources_record,
            },
            "prefix_record": {
                "path": str(PREFIX_RECORD),
                "sha256": sha256_file(PREFIX_RECORD),
            },
            "source_vcf_contract": source_audit,
            "mapping_audit": mapping_audit,
            "manifest_audit": manifest_audit,
            "biophysical_archive_audit": biophysical_audit,
            "ascertainment_guardrails": (
                "ApeKI GBS/UNEAK release; author filters include site call rate 0.75, minimum "
                "proportion of sites present 0.7, MAF 0.01, population-label-informed modified "
                "HWE/FDR filtering, and removal of pcadapt/BayeScan differentiated candidates; "
                "this may attenuate private/differentiated signal, and all loci have artificial CHROM=0"
            ),
        },
        "published_evidence": {
            "candidate_directions": [
                "DOI -> FHA exceeds reciprocal FHA -> DOI in each of four ROMS seasons.",
                "AIS -> HOS exceeds reciprocal HOS -> AIS in each of four ROMS seasons.",
            ],
            "independence_guardrail": (
                "ROMS particle paths are numerically independent of the SNP genotypes, but are "
                "potential physical dispersal rather than realized migration or introgression"
            ),
            "source_conclusion": (
                "the paper reports four regional genetic groups and concludes that geography plus "
                "ocean circulation explains structure better than either alone"
            ),
            "normalization_adjudication": (
                "current-paper and thesis figure cells reproduce mean day-100 D per 100 released "
                "particles; the prose instead describes destination-conditional C, so both are audited"
            ),
            "freshness_audit": (
                "as of 2026-07-11, this remains the latest peer-reviewed species-specific population-"
                "genomic plus biophysical connectivity study; no correction, retraction, or independent "
                "directional reanalysis was found. A 2026 assembly preprint is newer species genomics "
                "but does not reanalyse these populations or connectivity labels; a later cross-taxon "
                "WAP paper describes the cited structure as subtle/high-gene-flow context without "
                "validating a pairwise direction"
            ),
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
                        "projection": panel["model_free_comparator"][
                            "P2_projection_from_P1_toward_P3_all_loci"
                        ],
                        "loci": panel["padze"]["n_loci_kept"],
                    }
                    for panel in panels
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
