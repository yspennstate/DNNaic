#!/usr/bin/env python3
"""Run guarded 2024 tinkerbird holdout and near-null transfer panels.

The primary panel is sample-disjoint from the 95 females used by Sebastianelli
et al. (2024) to infer asymmetric mating/backcrossing.  It uses exact author
reference labels and nine Mpofu chrysoconus males selected by sex, locality,
and taxon rather than the direction statistic.  Candidate class C is still a
weak system-level majority label, not exclusive unidirectional truth: current
phylogenomics supports bidirectional introgression tails.

A direct 14-daughter panel deliberately reproduces the author-labelled cohort
and is circular.  Two same-source contrasts are gate-only near-nulls, never
claims of zero historical migration.  Every panel is autosome-only.  One
source-ordered SNP per Stacks RAD locus is the primary scope; the all-SNP scope
is a linked, intra-tag-pseudoreplicated sensitivity analysis.  Learned scores
are uncalibrated OOD diagnostics, not probabilities or migration magnitudes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import urllib.request

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
    subset_prepared_vcf,
    verify_file,
)
import tinkerbird_external_benchmark as legacy


DEFAULT_CACHE = REPO / "data" / "real" / "tinkerbird_2024_external_benchmark"
DEFAULT_RESULTS = REPO / "results" / "tinkerbird_2024_external_benchmark_2026_07_11"
DEFAULT_CAP = 15_000
PANEL_DIR = MANIFEST_DIR / "tinkerbird_2024"
DIRECTION_UNION = PANEL_DIR / "direction_union.tsv"
DIRECTION_HOLDOUT = PANEL_DIR / "direction_holdout.tsv"
DIRECTION_DIRECT = PANEL_DIR / "direction_direct.tsv"
GEOGRAPHIC_NEAR_NULL = PANEL_DIR / "geographic_near_null.tsv"
RECENT_CROSS_NEAR_NULL = PANEL_DIR / "recent_cross_near_null.tsv"
DIRECTION_FEMALES_95 = PANEL_DIR / "direction_females_95.tsv"
AUTOSOME_PATTERN = re.compile(r"^SUPER_(?:[1-9]|[1-3][0-9]|4[0-4])$")
STACKS_LOCUS_PATTERN = re.compile(r"^([0-9]+):([0-9]+):([+-])$")

SCOPE_ROLES = {
    "autosome_one_per_RAD_locus": (
        "primary: first source-ordered SNP per Stacks catalog RAD locus; removes "
        "within-tag pseudoreplication but not chromosome-scale linkage"
    ),
    "autosome_all_snps": (
        "sensitivity: linked multi-SNP-per-RAD source scope with a capped reservoir and "
        "intra-tag pseudoreplication"
    ),
}

FIGSHARE_RECORD = "https://figshare.com/articles/dataset/25308376"
FIGSHARE_FILES = "https://ndownloader.figshare.com/files"
FILES = {
    "vcf": {
        "key": "southern_africa_biallelic_snps_minDP4_MaxMiss20_MAF5.vcf.gz",
        "id": 44_729_950,
        "bytes": 148_766_627,
        "sha256": "6438a889ad91b865237e6a4e5169bfbe61e9eea884696fa7fdcfef83d68c7c30",
        "md5": "38294c4f2998dcf7dd62e337db5e90b3",
    },
    "female_metadata": {
        "key": "MS_SouthernAfrica_ddRADS_95SympF_14Mar23.xlsx",
        "id": 44_729_974,
        "bytes": 22_157,
        "sha256": "8e395902f7fb0c236287ddda85afc7f881cbc67421957c98a80c1eb10502d114",
        "md5": "c982c71fa9ebfa9c2dedf11e0d0877db",
    },
    "master_metadata": {
        "key": "MS_Tinkerbird138_Master_06Oct23.xlsx",
        "id": 44_729_977,
        "bytes": 53_230,
        "sha256": "c96ce8162f42b28d9470f9085be3378d6e54b3111139ef99541f6dcec5385c10",
        "md5": "8de687c8286e8feb387838be1ab06a52",
    },
    "supplement": {
        "key": "41467_2024_47305_MOESM1_ESM.pdf",
        "bytes": 1_640_095,
        "sha256": "c31b65cb94729f1ceaafb28ca4f76a8b80e431136573a15f001c2a4bcb7a7f93",
        "md5": "73edee4ebcac0572fd59fe31f08a2272",
        "download": (
            "https://static-content.springer.com/esm/art%3A10.1038%2F"
            "s41467-024-47305-5/MediaObjects/41467_2024_47305_MOESM1_ESM.pdf"
        ),
    },
}

SOURCE_CONTRACT = {
    "samples": 452,
    "variant_rows": 84_112,
    "anchored_autosomes": 82_309,
    "Z": 1_157,
    "W": 5,
    "S76_non_numbered": 31,
    "unplaced_scaffolds": 610,
    "autosomal_RAD_loci": 23_395,
    "one_per_RAD_ordered_key_sha256": (
        "a78ee8f7f57e81745d7f745d3c600435ed800c1611e5bf9a1fe57468253200ef"
    ),
    "paper_reported_SNPs": 82_950,
}


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(output)


def ensure_file(path: Path, spec: dict, download_missing: bool) -> dict:
    if not path.exists():
        if not download_missing:
            raise FileNotFoundError(path)
        url = spec.get("download") or f"{FIGSHARE_FILES}/{spec['id']}"
        _download(url, path)
    return verify_file(path, spec["bytes"], spec["sha256"])


def read_simple_manifest(path: Path) -> dict[str, str]:
    mapping = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if fields == ["sample", "population"]:
            continue
        if len(fields) != 2 or fields[0] in mapping:
            raise ValueError(f"{path}:{number}: invalid manifest row")
        mapping[fields[0]] = fields[1]
    if not mapping:
        raise ValueError(f"{path}: empty manifest")
    return mapping


def _samples(mapping: dict[str, str], population: str) -> set[str]:
    return {sample for sample, label in mapping.items() if label == population}


def audit_manifests() -> dict:
    union = read_simple_manifest(DIRECTION_UNION)
    holdout = read_manifest(DIRECTION_HOLDOUT)
    direct = read_manifest(DIRECTION_DIRECT)
    geographic = read_manifest(GEOGRAPHIC_NEAR_NULL)
    recent = read_manifest(RECENT_CROSS_NEAR_NULL)
    females = read_simple_manifest(DIRECTION_FEMALES_95)
    counts = {
        "direction_union": {
            label: list(union.values()).count(label) for label in set(union.values())
        },
        "direction_holdout": {
            label: list(holdout.values()).count(label) for label in set(holdout.values())
        },
        "direction_direct": {
            label: list(direct.values()).count(label) for label in set(direct.values())
        },
        "geographic_near_null": {
            label: list(geographic.values()).count(label)
            for label in set(geographic.values())
        },
        "recent_cross_near_null": {
            label: list(recent.values()).count(label) for label in set(recent.values())
        },
        "direction_females_95": len(females),
    }
    expected_counts = {
        "direction_union": {
            "ExtoniRefExact23": 23,
            "HoldoutContactMale9": 9,
            "DirectExtoniMotherDaughter14": 14,
            "PusillusRefExact8": 8,
        },
        "direction_holdout": {
            "ExtoniRefExact23": 23,
            "MpofuChrysoconusMale9": 9,
            "PusillusRefExact8": 8,
        },
        "direction_direct": {
            "ExtoniRefExact23": 23,
            "ExtoniMotherDaughter14": 14,
            "PusillusRefExact8": 8,
        },
        "geographic_near_null": {
            "SeringveldExtoni11": 11,
            "CullinanDeTweedeExtoni9": 9,
            "PusillusRefExact8": 8,
        },
        "recent_cross_near_null": {
            "PusillusRefExact8": 8,
            "NearPurePusillusDaughter12": 12,
            "ExtoniRefExact23": 23,
        },
        "direction_females_95": 95,
    }
    if counts != expected_counts:
        raise AssertionError(f"unexpected 2024 tinkerbird manifest counts: {counts}")

    holdout_p1 = _samples(holdout, "ExtoniRefExact23")
    holdout_p2 = _samples(holdout, "MpofuChrysoconusMale9")
    holdout_p3 = _samples(holdout, "PusillusRefExact8")
    direct_p1 = _samples(direct, "ExtoniRefExact23")
    direct_p2 = _samples(direct, "ExtoniMotherDaughter14")
    direct_p3 = _samples(direct, "PusillusRefExact8")
    female_set = set(females)
    if holdout_p1 != direct_p1 or holdout_p3 != direct_p3:
        raise AssertionError("direction panels do not share exact P1/P3 references")
    if set(holdout) & female_set:
        raise AssertionError("any primary holdout sample overlaps the 95-female direction cohort")
    if not direct_p2.issubset(female_set):
        raise AssertionError("direct direction panel is not contained in the labelled females")
    if not _samples(recent, "NearPurePusillusDaughter12").issubset(female_set):
        raise AssertionError("recent-cross near-null is not contained in labelled females")
    geographic_p1 = _samples(geographic, "SeringveldExtoni11")
    geographic_p2 = _samples(geographic, "CullinanDeTweedeExtoni9")
    geographic_p3 = _samples(geographic, "PusillusRefExact8")
    recent_p1 = _samples(recent, "PusillusRefExact8")
    recent_p2 = _samples(recent, "NearPurePusillusDaughter12")
    recent_p3 = _samples(recent, "ExtoniRefExact23")
    if geographic_p1 & geographic_p2:
        raise AssertionError("geographic near-null P1/P2 overlap")
    if geographic_p3 != holdout_p3 or not (
        geographic_p1 | geographic_p2
    ).issubset(holdout_p1):
        raise AssertionError("geographic control identities changed")
    if recent_p1 != holdout_p3 or recent_p3 != holdout_p1:
        raise AssertionError("recent-cross control reference identities changed")
    if direct_p2 & recent_p2:
        raise AssertionError("direct and recent-cross daughter pools overlap")
    if set(union) != set(holdout) | set(direct):
        raise AssertionError("direction union does not equal the two direction panels")
    union_contract = {
        "ExtoniRefExact23": holdout_p1,
        "HoldoutContactMale9": holdout_p2,
        "DirectExtoniMotherDaughter14": direct_p2,
        "PusillusRefExact8": holdout_p3,
    }
    for union_label, expected_samples in union_contract.items():
        if _samples(union, union_label) != expected_samples:
            raise AssertionError(
                f"direction union membership changed for {union_label}"
            )
    return {
        "counts": counts,
        "sample_disjoint_holdout": True,
        "direction_union_subgroup_contract": True,
        "direct_panel_subset_of_direction_females": True,
        "recent_cross_panel_subset_of_direction_females": True,
        "manifest_files": {
            path.name: {"path": str(path), "sha256": sha256_file(path)}
            for path in (
                DIRECTION_UNION,
                DIRECTION_HOLDOUT,
                DIRECTION_DIRECT,
                GEOGRAPHIC_NEAR_NULL,
                RECENT_CROSS_NEAR_NULL,
                DIRECTION_FEMALES_95,
            )
        },
    }


def is_autosome(chromosome: str) -> bool:
    return AUTOSOME_PATTERN.fullmatch(chromosome) is not None


def rad_locus_id(identifier: str, chromosome: str, position: str) -> str:
    match = STACKS_LOCUS_PATTERN.fullmatch(identifier)
    if match is None:
        raise ValueError(
            f"{chromosome}:{position}: expected Stacks <locus>:<column>:<strand> ID, "
            f"found {identifier!r}"
        )
    return match.group(1)


def audit_source_vcf(path: Path, required_samples: set[str]) -> dict:
    samples = None
    counts = {
        "anchored_autosomes": 0,
        "Z": 0,
        "W": 0,
        "S76_non_numbered": 0,
        "unplaced_scaffolds": 0,
    }
    rad_loci: set[str] = set()
    structural_failures = 0
    formats: set[str] = set()
    genotype_cells = 0
    missing_genotype_cells = 0
    partial_or_invalid_genotype_cells = 0
    previous_position: dict[str, int] = {}
    within_chromosome_position_decreases = 0
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                continue
            if line.startswith("#") or not line.strip():
                continue
            if samples is None:
                raise ValueError(f"{path}: data before #CHROM header")
            fields = line.rstrip("\r\n").split("\t")
            chromosome = fields[0]
            position = int(fields[1])
            if chromosome in previous_position and position < previous_position[chromosome]:
                within_chromosome_position_decreases += 1
            previous_position[chromosome] = position
            if is_autosome(chromosome):
                counts["anchored_autosomes"] += 1
                rad_loci.add(rad_locus_id(fields[2], chromosome, fields[1]))
            elif chromosome == "SUPER_Z":
                counts["Z"] += 1
            elif chromosome == "SUPER_W":
                counts["W"] += 1
            elif chromosome == "S76":
                counts["S76_non_numbered"] += 1
            else:
                counts["unplaced_scaffolds"] += 1
            structural_failures += int(
                len(fields[3]) != 1
                or len(fields[4]) != 1
                or "," in fields[4]
                or fields[6] != "PASS"
            )
            formats.add(fields[8])
            for cell in fields[9:]:
                genotype_cells += 1
                alleles = cell.split(":", 1)[0].replace("|", "/").split("/")
                if alleles == [".", "."]:
                    missing_genotype_cells += 1
                elif len(alleles) != 2 or any(allele not in {"0", "1"} for allele in alleles):
                    partial_or_invalid_genotype_cells += 1
    if samples is None or len(samples) != SOURCE_CONTRACT["samples"]:
        raise AssertionError("unexpected 2024 tinkerbird sample contract")
    if len(set(samples)) != len(samples):
        raise AssertionError("duplicate 2024 tinkerbird source samples")
    expected_counts = {
        key: SOURCE_CONTRACT[key]
        for key in (
            "anchored_autosomes",
            "Z",
            "W",
            "S76_non_numbered",
            "unplaced_scaffolds",
        )
    }
    if counts != expected_counts or sum(counts.values()) != SOURCE_CONTRACT["variant_rows"]:
        raise AssertionError(f"unexpected source chromosome counts: {counts}")
    if structural_failures or formats != {"GT:DP:AD:GQ:GL"}:
        raise AssertionError("unexpected source VCF structural contract")
    if (
        genotype_cells != SOURCE_CONTRACT["variant_rows"] * SOURCE_CONTRACT["samples"]
        or partial_or_invalid_genotype_cells
    ):
        raise AssertionError("source GT cells are not diploid biallelic calls or full missing")
    if within_chromosome_position_decreases != 32_348:
        raise AssertionError("unexpected source-order position-decrease count")
    if len(rad_loci) != SOURCE_CONTRACT["autosomal_RAD_loci"]:
        raise AssertionError("unexpected autosomal Stacks RAD-locus count")
    missing = sorted(required_samples - set(samples))
    if missing:
        raise AssertionError(f"manifest samples absent from source: {missing}")
    return {
        "samples": len(samples),
        "variant_rows": sum(counts.values()),
        "chromosome_assignment_counts": counts,
        "autosomal_RAD_loci": len(rad_loci),
        "all_rows_biallelic_PASS_SNPs": True,
        "FORMAT": "GT:DP:AD:GQ:GL",
        "genotype_cells": genotype_cells,
        "fully_missing_genotype_cells": missing_genotype_cells,
        "partial_or_invalid_genotype_cells": partial_or_invalid_genotype_cells,
        "within_chromosome_position_decreases": within_chromosome_position_decreases,
        "source_order_guardrail": (
            "Stacks rows are not genomic-position sorted; one-per-RAD selection is source-first "
            "and occurs before any sorting"
        ),
        "required_manifest_samples_present": len(required_samples),
        "source_paper_count_reconciliation": (
            "paper 82,950 ddRAD SNPs equals 82,309 numbered autosomal plus 641 "
            "S76/non-numbered or unplaced non-sex rows; source total 84,112 additionally "
            "includes 1,157 Z and 5 W"
        ),
    }


def subset_autosomes(source: Path, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    retained = 0
    saw_header = False
    with open_text(source) as incoming, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as outgoing:
        for line in incoming:
            if line.startswith("#"):
                outgoing.write(line.rstrip("\r\n") + "\n")
                saw_header = saw_header or line.startswith("#CHROM")
                continue
            if not line.strip():
                continue
            source_rows += 1
            if is_autosome(line.split("\t", 1)[0]):
                outgoing.write(line.rstrip("\r\n") + "\n")
                retained += 1
    if (
        not saw_header
        or source_rows != SOURCE_CONTRACT["variant_rows"]
        or retained != SOURCE_CONTRACT["anchored_autosomes"]
    ):
        raise AssertionError("2024 tinkerbird autosome extraction violated its contract")
    return {
        "source": str(source),
        "source_variant_rows": source_rows,
        "retained_variant_rows": retained,
        "selection": (
            "SUPER_1 through SUPER_44; Z, W, S76/non-numbered, and unplaced sequence excluded"
        ),
        "derived_vcf": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }


def thin_one_per_rad_locus(
    source: Path,
    output: Path,
    *,
    expected_source_rows: int | None = SOURCE_CONTRACT["anchored_autosomes"],
    expected_RAD_loci: int | None = SOURCE_CONTRACT["autosomal_RAD_loci"],
    expected_ordered_key_sha256: str | None = SOURCE_CONTRACT[
        "one_per_RAD_ordered_key_sha256"
    ],
) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    source_rows = 0
    retained = 0
    saw_header = False
    ordered_key_hash = hashlib.sha256()
    with open_text(source) as incoming, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as outgoing:
        for line in incoming:
            if line.startswith("#"):
                outgoing.write(line.rstrip("\r\n") + "\n")
                saw_header = saw_header or line.startswith("#CHROM")
                continue
            if not line.strip():
                continue
            source_rows += 1
            fields = line.rstrip("\r\n").split("\t")
            locus = rad_locus_id(fields[2], fields[0], fields[1])
            if locus in seen:
                continue
            seen.add(locus)
            outgoing.write("\t".join(fields) + "\n")
            ordered_key_hash.update(
                ("\t".join((fields[0], fields[1], fields[2], fields[3], fields[4])) + "\n").encode(
                    "utf-8"
                )
            )
            retained += 1
    if not saw_header:
        raise AssertionError("one-per-RAD-locus input lacks a VCF header")
    if expected_source_rows is not None and source_rows != expected_source_rows:
        raise AssertionError("one-per-RAD-locus source row count changed")
    if expected_RAD_loci is not None and retained != expected_RAD_loci:
        raise AssertionError("one-per-RAD-locus extraction violated its contract")
    ordered_key_sha256 = ordered_key_hash.hexdigest()
    if (
        expected_ordered_key_sha256 is not None
        and ordered_key_sha256 != expected_ordered_key_sha256
    ):
        raise AssertionError("one-per-RAD-locus ordered-key digest changed")
    return {
        "source": str(source),
        "source_variant_rows": source_rows,
        "selection": "first source-ordered SNP per Stacks catalog RAD-locus ID prefix",
        "retained_variant_rows": retained,
        "RAD_loci": len(seen),
        "ordered_key_contract": "CHROM<TAB>POS<TAB>ID<TAB>REF<TAB>ALT<LF>",
        "ordered_key_sha256": ordered_key_sha256,
        "derived_vcf": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }


def frequency_geometry(
    vcf: Path,
    manifest_path: Path,
    pop_order: tuple[str, str, str],
    diagnostic_threshold: float = 0.95,
    bootstrap_replicates: int = 500,
) -> dict:
    mapping = read_manifest(manifest_path)
    if set(mapping.values()) != set(pop_order):
        raise ValueError("manifest populations do not match the requested order")
    columns = None
    values = {population: [] for population in pop_order}
    called_copy_counts = {population: [] for population in pop_order}
    blocks: list[str] = []
    RAD_loci: list[str] = []
    with open_text(vcf) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                sample_column = {sample: 9 + index for index, sample in enumerate(samples)}
                missing = sorted(set(mapping) - set(samples))
                if missing:
                    raise ValueError(f"{vcf}: comparator samples absent: {missing}")
                columns = {
                    population: [
                        sample_column[sample]
                        for sample, label in mapping.items()
                        if label == population
                    ]
                    for population in pop_order
                }
                continue
            if line.startswith("#") or not line.strip():
                continue
            if columns is None:
                raise ValueError(f"{vcf}: data before #CHROM header")
            fields = line.rstrip("\r\n").split("\t")
            blocks.append(fields[0])
            RAD_loci.append(rad_locus_id(fields[2], fields[0], fields[1]))
            for population in pop_order:
                alleles = [
                    allele
                    for index in columns[population]
                    for allele in called_alleles(fields[index])
                ]
                if not alleles:
                    raise ValueError(f"{vcf}: locus has no called {population} alleles")
                values[population].append(
                    sum(allele == "1" for allele in alleles) / len(alleles)
                )
                called_copy_counts[population].append(len(alleles))
    if columns is None or not blocks:
        raise ValueError(f"{vcf}: no usable loci")
    p1, p2, p3 = (
        np.asarray(values[population], dtype=float) for population in pop_order
    )
    axis = p3 - p1
    numerator = (p2 - p1) * axis
    denominator = axis**2
    diagnostic = np.abs(axis) >= diagnostic_threshold
    f3_plugin = (p2 - p1) * (p2 - p3)
    n2 = np.asarray(called_copy_counts[pop_order[1]], dtype=float)
    if np.any(n2 <= 1):
        raise AssertionError("P2 requires at least two called gene copies per locus")
    p2_finite_called_copy_correction = p2 * (1.0 - p2) / (n2 - 1.0)
    f3_finite_called_copy_corrected = f3_plugin - p2_finite_called_copy_correction

    def project(mask: np.ndarray) -> float | None:
        total = float(denominator[mask].sum())
        if not np.any(mask) or total == 0.0:
            return None
        return float(numerator[mask].sum() / total)

    bootstrap = legacy.scaffold_block_bootstrap(
        blocks,
        numerator,
        denominator,
        f3_finite_called_copy_corrected,
        diagnostic,
        replicates=bootstrap_replicates,
    )
    bootstrap["method"] = (
        "nonparametric chromosome-block bootstrap over SUPER_1 through SUPER_44"
    )
    bootstrap["chromosome_blocks"] = bootstrap.pop("scaffold_blocks")
    bootstrap["maximum_SNPs_per_chromosome"] = bootstrap.pop(
        "maximum_loci_per_scaffold"
    )
    bootstrap["f3_finite_called_copy_corrected"] = bootstrap.pop("f3_P2_P1_P3")
    bootstrap["guardrail"] = (
        "fixed-panel descriptive chromosome-block uncertainty; sample ascertainment and "
        "chromosome-scale linkage remain, and these are not accuracy confidence intervals"
    )
    return {
        "n_loci": int(len(p1)),
        "chromosomes": len(set(blocks)),
        "RAD_loci": len(set(RAD_loci)),
        "P2_projection_from_P1_toward_P3_all_loci": project(
            np.ones(len(p1), dtype=bool)
        ),
        "diagnostic_threshold_abs_P3_minus_P1_frequency": diagnostic_threshold,
        "diagnostic_loci": int(diagnostic.sum()),
        "P2_projection_from_P1_toward_P3_diagnostic_loci": project(diagnostic),
        "f3_like_plugin_P2_P1_P3": float(np.mean(f3_plugin)),
        "mean_P2_finite_called_copy_correction": float(
            np.mean(p2_finite_called_copy_correction)
        ),
        "f3_P2_P1_P3_finite_called_copy_corrected": float(
            np.mean(f3_finite_called_copy_corrected)
        ),
        "mean_squared_frequency_distance": {
            "P1_P2": float(np.mean((p1 - p2) ** 2)),
            "P2_P3": float(np.mean((p2 - p3) ** 2)),
            "P1_P3": float(np.mean((p1 - p3) ** 2)),
        },
        "chromosome_block_bootstrap": bootstrap,
        "interpretation": (
            "sample-frequency geometry; projection is reference-flip invariant but has endpoint "
            "sampling error and is not bounded ancestry or temporal direction. Plug-in f3-like "
            "geometry and a finite-called-copy P2 correction are both reported. The correction "
            "is exact only under independent binomial chromosome sampling and is not labelled "
            "generally unbiased. The diagnostic-locus projection is endpoint-ascertained on "
            "these same sample frequencies and remains descriptive."
        ),
    }


def _score(
    panel_id: str,
    vcf: Path,
    popmap: Path,
    manifest: Path,
    pop_order: tuple[str, str, str],
    audit: dict,
    direction_head,
    gate_head,
    expectation: dict,
) -> dict:
    panel = score_panel(
        panel_id,
        vcf,
        popmap,
        pop_order,
        audit,
        direction_head[0],
        direction_head[1],
        expectation,
    )
    panel["population_order"]["tree_contract_status"] = expectation[
        "tree_contract_status"
    ]
    add_gate_score(panel, gate_head[0], gate_head[1])
    panel["model_free_comparator"] = frequency_geometry(vcf, manifest, pop_order)
    candidate = expectation.get("candidate_class")
    prediction = panel["simulation_head"]["predicted_class"]
    severe_OOD = max(
        panel["simulation_feature_shift"]["rms_z"],
        panel["simulation_gate_feature_shift"]["rms_z"],
    ) > 10
    panel["adjudication"] = {
        "candidate_class": candidate,
        "predicted_class": prediction,
        "matches_candidate_majority": prediction == candidate if candidate else None,
        "gate_score": panel["simulation_gate"]["appreciable_score"],
        "gate_near_null_contrast": expectation["benchmark_role"] == "gate_near_null",
        "accuracy_eligible": False,
        "severe_OOD": severe_OOD,
        "natural_data_call_status": "abstain_severe_OOD" if severe_OOD else "diagnostic_only",
    }
    return panel


def summarize_panel_outcomes(panels: list[dict]) -> dict:
    if not panels:
        raise ValueError("cannot summarize an empty panel set")
    direction = [
        panel
        for panel in panels
        if panel["external_expectation"].get("candidate_class") is not None
    ]
    near_null = [
        panel
        for panel in panels
        if panel["external_expectation"]["benchmark_role"] == "gate_near_null"
    ]
    predictions = {
        label: sum(panel["simulation_head"]["predicted_class"] == label for panel in direction)
        for label in ("A", "B", "C")
    }
    near_null_scores = [
        panel["simulation_gate"]["appreciable_score"] for panel in near_null
    ]
    severe = sum(panel["adjudication"]["severe_OOD"] for panel in panels)
    return {
        "panels": len(panels),
        "severe_OOD_panels": severe,
        "abstain_due_to_severe_OOD": severe == len(panels),
        "accuracy_estimate": None,
        "direction_candidate_panels": {
            "n": len(direction),
            "prediction_counts": predictions,
            "matches_candidate": sum(
                panel["adjudication"]["matches_candidate_majority"] for panel in direction
            ),
        },
        "gate_near_null_panels": {
            "n": len(near_null),
            "called_appreciable_at_0.5": sum(
                panel["simulation_gate"]["called_at_0.5"] for panel in near_null
            ),
            "minimum_score": min(near_null_scores) if near_null_scores else None,
            "maximum_score": max(near_null_scores) if near_null_scores else None,
        },
        "interpretation": (
            "severe-OOD direction-transfer and gate-specificity stress-test outcome; thresholded "
            "natural-data scores are abstained, not validated calls or accuracy observations"
        ),
    }


def _filter_description(strict: bool) -> str:
    return (
        "both alleles called within every population; strong ascertainment, not a quality filter"
        if strict
        else "both alleles called across every complete three-population panel"
    )


def run_direction_family(
    source: Path,
    cache: Path,
    scope: str,
    filter_name: str,
    strict: bool,
    cap: int,
    direction_head,
    gate_head,
) -> tuple[list[dict], Path]:
    shared_vcf = cache / f"direction.{scope}.{filter_name}.shared.vcf"
    shared_popmap = cache / f"direction.{scope}.{filter_name}.shared.popmap.tsv"
    shared_audit = prepare_vcf(
        source,
        DIRECTION_UNION,
        shared_vcf,
        shared_popmap,
        cap=cap,
        seed=20260711,
        require_three_populations=False,
        polymorphic_panel_manifests=(DIRECTION_HOLDOUT, DIRECTION_DIRECT),
        polymorphic_within_each_population=strict,
    )
    specs = (
        (
            "holdout",
            DIRECTION_HOLDOUT,
            ("ExtoniRefExact23", "MpofuChrysoconusMale9", "PusillusRefExact8"),
            {
                "benchmark_role": "sample_disjoint_direction_holdout",
                "candidate_class": "C",
                "candidate_forward_direction": "PusillusRefExact8 (P3) -> Mpofu contact males (P2)",
                "truth_strength": (
                    "weak system-level asymmetric/majority label; current evidence is bidirectional"
                ),
                "selection": (
                    "P2 selected by sex=M, location=Mpofu, author species=chrysoconus; "
                    "disjoint from all 95 direction females"
                ),
                "direct_label_source_reuse": False,
                "tree_contract_status": "operational_reference_order_not_proven_demographic_tree",
                "accuracy_eligible": False,
            },
        ),
        (
            "direct_circular",
            DIRECTION_DIRECT,
            ("ExtoniRefExact23", "ExtoniMotherDaughter14", "PusillusRefExact8"),
            {
                "benchmark_role": "direct_circular_direction_reproduction",
                "candidate_class": "C",
                "candidate_forward_direction": "PusillusRefExact8 (P3) -> labelled daughters (P2)",
                "truth_strength": (
                    "aggregate asymmetry: paternal ancestry spans the full range; not 14 exact positives"
                ),
                "selection": "P2 selected by inferred maternal ancestry from these same genotypes",
                "direct_label_source_reuse": True,
                "tree_contract_status": "violated_by_inferred-parent_hybrid_pool",
                "accuracy_eligible": False,
            },
        ),
    )
    panels = []
    for label, manifest, order, expectation in specs:
        vcf = cache / f"direction.{scope}.{filter_name}.{label}.vcf"
        popmap = cache / f"direction.{scope}.{filter_name}.{label}.popmap.tsv"
        audit = subset_prepared_vcf(shared_vcf, manifest, vcf, popmap, shared_audit)
        audit["comparison_locus_contract"] = (
            "same ordered ExtoniRef/PusillusRef/holdout/direct callable-site intersection"
        )
        expectation = {
            **expectation,
            "locus_filter_variant": _filter_description(strict),
            "scope": scope,
            "scope_role": SCOPE_ROLES[scope],
        }
        panels.append(
            _score(
                f"tinkerbird_2024_direction_{label}_{scope}_{filter_name}",
                vcf,
                popmap,
                manifest,
                order,
                audit,
                direction_head,
                gate_head,
                expectation,
            )
        )
    if panels[0]["input_audit"]["ordered_locus_sha256"] != panels[1]["input_audit"][
        "ordered_locus_sha256"
    ]:
        raise AssertionError("direction holdout/direct panels do not share ordered loci")
    return panels, shared_vcf


def run_single_panel(
    source: Path,
    cache: Path,
    scope: str,
    filter_name: str,
    strict: bool,
    cap: int,
    direction_head,
    gate_head,
    *,
    panel_name: str,
    manifest: Path,
    order: tuple[str, str, str],
    expectation: dict,
) -> tuple[dict, Path]:
    vcf = cache / f"{panel_name}.{scope}.{filter_name}.vcf"
    popmap = cache / f"{panel_name}.{scope}.{filter_name}.popmap.tsv"
    audit = prepare_vcf(
        source,
        manifest,
        vcf,
        popmap,
        cap=cap,
        seed=20260711,
        polymorphic_within_each_population=strict,
    )
    expectation = {
        **expectation,
        "locus_filter_variant": _filter_description(strict),
        "scope": scope,
        "scope_role": SCOPE_ROLES[scope],
        "accuracy_eligible": False,
    }
    return (
        _score(
            f"tinkerbird_2024_{panel_name}_{scope}_{filter_name}",
            vcf,
            popmap,
            manifest,
            order,
            audit,
            direction_head,
            gate_head,
            expectation,
        ),
        vcf,
    )


def cross_family_locus_overlap(family_paths: dict[str, Path]) -> dict:
    expected = {"direction", "geographic", "recent"}
    if set(family_paths) != expected:
        raise ValueError("cross-family overlap requires direction, geographic, and recent paths")
    loci = {family: legacy.locus_keys(path) for family, path in family_paths.items()}
    union = set().union(*loci.values())
    all_three = set.intersection(*loci.values())
    pairwise = {}
    for left, right in (
        ("direction", "geographic"),
        ("direction", "recent"),
        ("geographic", "recent"),
    ):
        intersection = loci[left] & loci[right]
        pair_union = loci[left] | loci[right]
        pairwise[f"{left}__{right}"] = {
            "intersection": len(intersection),
            "union": len(pair_union),
            "jaccard": len(intersection) / len(pair_union),
            f"fraction_of_{left}": len(intersection) / len(loci[left]),
            f"fraction_of_{right}": len(intersection) / len(loci[right]),
        }
    return {
        "family_loci": {family: len(keys) for family, keys in loci.items()},
        "pairwise": pairwise,
        "all_three_intersection": len(all_three),
        "all_three_union": len(union),
        "all_three_intersection_fraction_of_union": len(all_three) / len(union),
        "comparison_eligible": False,
        "interpretation": (
            "families use separately ascertained locus draws; overlaps document the mismatch, "
            "so cross-family gate-score contrasts remain qualitative"
        ),
    }


def pair_locus_overlap(
    left_path: Path,
    right_path: Path,
    left_label: str,
    right_label: str,
) -> dict:
    left = legacy.locus_keys(left_path)
    right = legacy.locus_keys(right_path)
    intersection = left & right
    union = left | right
    return {
        f"{left_label}_loci": len(left),
        f"{right_label}_loci": len(right),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": len(intersection) / len(union),
        f"fraction_of_{left_label}": len(intersection) / len(left),
        f"fraction_of_{right_label}": len(intersection) / len(right),
        "comparison_eligible": False,
        "interpretation": (
            "independently capped/reservoir-sampled scopes are not a paired locus analysis; "
            "the overlap is reported to expose that ascertainment difference"
        ),
    }


def run_all_panels(
    sources: dict[str, Path], cache: Path, cap: int, direction_head, gate_head
) -> tuple[list[dict], dict]:
    if set(sources) != set(SCOPE_ROLES):
        raise ValueError("analysis sources do not match the declared scope roles")
    panels = []
    paths = {}
    for scope, source in sources.items():
        for filter_name, strict in (
            ("standard_contract", False),
            ("within_population_polymorphism", True),
        ):
            direction_panels, shared_path = run_direction_family(
                source,
                cache,
                scope,
                filter_name,
                strict,
                cap,
                direction_head,
                gate_head,
            )
            panels.extend(direction_panels)
            paths[(scope, "direction", filter_name)] = shared_path
            geographic, geographic_path = run_single_panel(
                source,
                cache,
                scope,
                filter_name,
                strict,
                cap,
                direction_head,
                gate_head,
                panel_name="geographic_near_null",
                manifest=GEOGRAPHIC_NEAR_NULL,
                order=(
                    "SeringveldExtoni11",
                    "CullinanDeTweedeExtoni9",
                    "PusillusRefExact8",
                ),
                expectation={
                    "benchmark_role": "gate_near_null",
                    "candidate_class": None,
                    "expected_gate": (
                        "qualitative same-taxon diagnostic contrast; no quantitative score "
                        "expectation across separately ascertained loci"
                    ),
                    "truth_strength": "geographic same-taxon contrast, not zero historical migration",
                    "direct_label_source_reuse": False,
                    "tree_contract_status": "operational_same_taxon_sister_reference",
                    "comparison_scope": "separate locus draw; gate comparison is qualitative",
                },
            )
            panels.append(geographic)
            paths[(scope, "geographic", filter_name)] = geographic_path
            recent, recent_path = run_single_panel(
                source,
                cache,
                scope,
                filter_name,
                strict,
                cap,
                direction_head,
                gate_head,
                panel_name="recent_cross_near_null",
                manifest=RECENT_CROSS_NEAR_NULL,
                order=(
                    "PusillusRefExact8",
                    "NearPurePusillusDaughter12",
                    "ExtoniRefExact23",
                ),
                expectation={
                    "benchmark_role": "gate_near_null",
                    "candidate_class": None,
                    "expected_gate": (
                        "qualitative recent-cross diagnostic contrast; no quantitative score "
                        "expectation across separately ascertained loci"
                    ),
                    "truth_strength": (
                        "both inferred parents exceed 0.97 pusillus ancestry; not historical no-flow"
                    ),
                    "direct_label_source_reuse": True,
                    "tree_contract_status": "violated_by_inferred-parent_daughter_pool",
                    "comparison_scope": "separate locus draw; gate comparison is qualitative",
                },
            )
            panels.append(recent)
            paths[(scope, "recent", filter_name)] = recent_path
    within_family_filter_overlap = {
        scope: {
            family: legacy.locus_overlap(
                paths[(scope, family, "standard_contract")],
                paths[(scope, family, "within_population_polymorphism")],
            )
            for family in ("direction", "geographic", "recent")
        }
        for scope in sources
    }
    cross_family_overlap = {
        scope: {
            filter_name: cross_family_locus_overlap(
                {
                    family: paths[(scope, family, filter_name)]
                    for family in ("direction", "geographic", "recent")
                }
            )
            for filter_name in (
                "standard_contract",
                "within_population_polymorphism",
            )
        }
        for scope in sources
    }
    cross_scope_overlap = {
        family: {
            filter_name: pair_locus_overlap(
                paths[("autosome_one_per_RAD_locus", family, filter_name)],
                paths[("autosome_all_snps", family, filter_name)],
                "one_per_RAD",
                "all_SNP",
            )
            for filter_name in (
                "standard_contract",
                "within_population_polymorphism",
            )
        }
        for family in ("direction", "geographic", "recent")
    }
    return panels, {
        "within_family_filter_overlap": within_family_filter_overlap,
        "cross_family_locus_overlap": cross_family_overlap,
        "cross_scope_locus_overlap": cross_scope_overlap,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    parser.add_argument("--source-vcf")
    parser.add_argument("--female-metadata")
    parser.add_argument("--master-metadata")
    parser.add_argument("--supplement")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    if args.cap < 1:
        parser.error("--cap must be positive")

    set_below_normal_priority()
    cache = Path(args.cache_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    supplied = {
        "vcf": args.source_vcf,
        "female_metadata": args.female_metadata,
        "master_metadata": args.master_metadata,
        "supplement": args.supplement,
    }
    paths = {
        name: Path(supplied[name]).resolve()
        if supplied[name]
        else cache / FILES[name]["key"]
        for name in FILES
    }
    verified_files = {
        name: ensure_file(paths[name], FILES[name], args.download_missing) for name in FILES
    }
    manifest_audit = audit_manifests()
    required_samples = set()
    for manifest in (
        DIRECTION_UNION,
        DIRECTION_HOLDOUT,
        DIRECTION_DIRECT,
        GEOGRAPHIC_NEAR_NULL,
        RECENT_CROSS_NEAR_NULL,
        DIRECTION_FEMALES_95,
    ):
        required_samples.update(read_simple_manifest(manifest))
    source_audit = audit_source_vcf(paths["vcf"], required_samples)
    autosome_vcf = cache / "tinkerbird_2024.autosomes.vcf"
    autosome_audit = subset_autosomes(paths["vcf"], autosome_vcf)
    rad_vcf = cache / "tinkerbird_2024.autosomes.one_per_RAD_locus.vcf"
    rad_audit = thin_one_per_rad_locus(autosome_vcf, rad_vcf)

    direction_head = simulation_direction_head(
        Path(args.data_root).resolve(), max_depth=MAX_DEPTH
    )
    gate_head = simulation_gate_head(Path(args.data_root).resolve(), max_depth=MAX_DEPTH)
    panels, overlap = run_all_panels(
        {"autosome_one_per_RAD_locus": rad_vcf, "autosome_all_snps": autosome_vcf},
        cache,
        args.cap,
        direction_head,
        gate_head,
    )
    result = {
        "schema_version": "dnnaic-tinkerbird-2024-external-benchmark-v1",
        "git": git_revision(),
        "runtime": legacy.runtime_audit(),
        "guardrail": (
            "Sample-disjoint and circular direction stress tests plus qualitative gate "
            "near-null contrasts; no panel is gold unidirectional truth or accuracy-eligible."
        ),
        "source": {
            "record": FIGSHARE_RECORD,
            "data_doi": "10.6084/m9.figshare.25308376.v1",
            "paper_doi": "10.1038/s41467-024-47305-5",
            "current_phylogenomics_doi": "10.1093/sysbio/syaf033",
            "license": "CC-BY-4.0",
            "verified_files": verified_files,
            "source_vcf_contract": source_audit,
            "autosome_extraction": autosome_audit,
            "one_per_RAD_locus_extraction": rad_audit,
            "ascertainment_guardrails": (
                "ddRAD restriction-site sampling; global MAF>=5% and <=20% missingness over "
                "452 birds; mapping to a P. pusillus reference; these affect richness/private channels"
            ),
            "called_copy_guardrail": (
                "PADZE g=16 makes all eight PusillusRefExact8 birds completely called at every "
                "retained locus, while larger P1/P2 groups may retain missing calls; this is "
                "asymmetric missingness ascertainment"
            ),
        },
        "published_direction_evidence": {
            "main_text": {"model": "LM", "z": 6.949, "P": "<0.001"},
            "supplementary_table_S14": {
                "estimate": 0.582,
                "SE": 0.086,
                "z": 6.714,
                "P": "<0.001",
            },
            "numerical_discrepancy": (
                "main text z=6.949 differs from Supplementary Table S14 z=6.714"
            ),
            "direction_inference": (
                "paternal ancestry equals daughter Z ancestry; maternal ancestry is clipped "
                "2*autosomal minus Z ancestry"
            ),
            "candidate_class_C_meaning": (
                "predominant/asymmetric pusillus ancestry into extoni-background contact birds; "
                "not exclusive direction"
            ),
            "current_guardrail": (
                "Rancilhac et al. 2025 infer bidirectional introgression tails and core-range effects"
            ),
        },
        "manifest_audit": manifest_audit,
        "manifest_selection_provenance": {
            "runtime_checks": (
                "exact counts, subgroup/set relations, manifest hashes, and presence of every "
                "sample in the pinned 452-bird VCF"
            ),
            "construction_audit": (
                "frozen manifests were independently checked against the pinned author workbook "
                "fields during bundle construction; the runner deliberately does not dynamically "
                "parse Excel metadata"
            ),
            "holdout_limit": (
                "individual-disjoint from the 95 direction females, but not study-independent: "
                "global source filters used all 452 birds and the candidate label comes from the "
                "same biological study system"
            ),
        },
        "direction_head": direction_head[2],
        "gate_head": gate_head[2],
        "analysis_scope_roles": SCOPE_ROLES,
        "locus_overlap_audit": overlap,
        "direction_panel_locus_contract": (
            "holdout and direct panels share exact ordered loci within each scope/filter. "
            "Marginal chromosome-block summaries use the same seed, but no paired-difference "
            "statistic or paired inferential claim is reported"
        ),
        "outcome_summary": summarize_panel_outcomes(panels),
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
                "panels": [
                    {
                        "panel_id": panel["panel_id"],
                        "role": panel["external_expectation"]["benchmark_role"],
                        "direction": panel["simulation_head"]["predicted_class"],
                        "gate": panel["simulation_gate"]["appreciable_score"],
                        "direction_rms_z": panel["simulation_feature_shift"]["rms_z"],
                        "gate_rms_z": panel["simulation_gate_feature_shift"]["rms_z"],
                        "projection": panel["model_free_comparator"][
                            "P2_projection_from_P1_toward_P3_all_loci"
                        ],
                        "f3_finite_called_copy_corrected": panel[
                            "model_free_comparator"
                        ][
                            "f3_P2_P1_P3_finite_called_copy_corrected"
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
