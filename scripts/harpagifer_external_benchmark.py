#!/usr/bin/env python3
"""Run a guarded Harpagifer candidate-direction transfer benchmark.

Segovia et al. (2022) reported a dominant recent-migration estimate from their
northern Patagonia cluster to their southern Patagonia cluster.  With north as
P1, south as P2, and the Falklands/Malvinas cluster as P3, that is a candidate
DNNaic class-A sensitivity.

It is not gold truth or independent validation.  The label and clusters reuse
the same 2,993 released GBS SNPs, reciprocal and additional edges were reported,
and the sample-to-site mapping has to be reconstructed from VCF column order.
The four operational panels below are sensitivity views of one biological
comparison.  Every learned output is an uncalibrated OOD diagnostic; severe OOD
panels abstain and are excluded from accuracy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
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
    verify_file,
)
import tinkerbird_external_benchmark as runtime_helpers


DEFAULT_CACHE = REPO / "data" / "real" / "harpagifer_external_benchmark"
DEFAULT_RESULTS = REPO / "results" / "harpagifer_external_benchmark_2026_07_11"
DEFAULT_CAP = 15_000
BLOCKS_RECORD = MANIFEST_DIR / "harpagifer" / "reconstructed_blocks.tsv"
SOURCES_RECORD = MANIFEST_DIR / "harpagifer" / "sources.json"
POPULATION_ORDER = ("NorthPatagonia", "SouthPatagonia", "FalklandsMalvinas")

DRYAD_RECORD = "https://datadryad.org/dataset/doi:10.5061/dryad.jwstqjq9q"
DRYAD_ARCHIVE = "https://datadryad.org/api/v2/versions/156912/download"
DRYAD_FILE_DOWNLOAD = "https://datadryad.org/api/v2/files/1231632/download"
ARCHIVE = {
    "key": "doi_10_5061_dryad_jwstqjq9q__v20220102.zip",
    "observed_bytes": 1_517_885,
    "observed_cached_sha256": "7a663a52c391090199a6c20f85cd9dcd73cac97b1bb31ee4a6b98ea6373cedbe",
    "digest_policy": "not_pinned_observed_nondeterministic_generated_ZIP_wrapper",
}
FILE = {
    "id": 1_231_632,
    "key": "Hbi_Hpal_118_2993_6May19_GEO.vcf",
    "archive_member": "Hbi_Hpal_118_2993_6May19_GEO.vcf",
    "download": DRYAD_FILE_DOWNLOAD,
    "bytes": 1_517_411,
    "sha256": "7dbc3686e4a24ef36c2a358b55d3d95ff5f2f8d2340087099119335c9b0474a8",
}
SOURCE_CONTRACT = {
    "samples": 118,
    "H_bispinis_samples": 93,
    "H_palliolatus_samples": 25,
    "variant_rows": 2_993,
    "genotype_cells": 353_174,
    "fully_missing_genotype_cells": 34_619,
    "ordered_sample_sha256": "1282d61affd67c5d3e891928e1780ef4a991f907d82ec4a1a4b82bfba61fa86e",
    "locus_semantic_sha256": "d2e7393b4361c3a895453e8ef17a5b9b58c5787c171b03b22d935d10bb765263",
}
SUPPLEMENT_SNP_COUNTS = (2, 14, 15, 7, 11, 15, 15, 13, 25)
MAIN_TABLE3_SAMPLE_COUNTS = (3, 13, 13, 13, 12, 19, 14, 14, 27)
EXPECTED_HIGH_MISSING = {
    "HBi_014",
    "HBi_026",
    "HBi_029",
    "HBi_031",
    "HBi_034",
    "HBi_038",
    "HBi_040",
    "HBi_107",
    "HBi_115",
    "HBi_126",
    "HBi_313",
}
EXPECTED_SAMPLE_SCOPES = {
    "all_released_samples": {
        "NorthPatagonia": 50,
        "SouthPatagonia": 43,
        "FalklandsMalvinas": 25,
    },
    "sample_missingness_le_0_25": {
        "NorthPatagonia": 40,
        "SouthPatagonia": 42,
        "FalklandsMalvinas": 25,
    },
}
EXPECTED_FILTERS = {
    "standard_contract": {
        "loci": 2_993,
        "ordered_locus_sha256": "a72e91972c32a39363d2c6133f8822cde43ead12f97931b89d05abbee5ad76a6",
    },
    "within_population_polymorphism": {
        "loci": 2_977,
        "ordered_locus_sha256": "c4a2a74166e672091fd5bc3f17404f93369f6d495ea060016e47c7a316ba0ac5",
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


def ensure_source(source: Path, archive: Path, download_missing: bool) -> dict:
    if not source.exists():
        if not download_missing:
            raise FileNotFoundError(source)
        _download(DRYAD_FILE_DOWNLOAD, source)
    verified = {"vcf": verify_file(source, FILE["bytes"], FILE["sha256"])}
    verified["archive_wrapper_observation"] = (
        {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "verification_status": "observed_only_not_pinned",
            "guardrail": (
                "repeated official downloads produced generated ZIP wrappers with different member "
                "timestamps and SHA-256 values; only the byte-identical inner VCF is canonical and pinned"
            ),
        }
        if archive.exists()
        else None
    )
    return verified


def read_vcf_samples(path: Path) -> list[str]:
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                if len(samples) != len(set(samples)):
                    raise AssertionError("Harpagifer VCF sample IDs are not unique")
                return samples
    raise ValueError(f"{path}: no #CHROM header")


def ordered_sample_sha256(samples: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sample}\n" for sample in samples).encode("utf-8")
    ).hexdigest()


def read_blocks(path: Path = BLOCKS_RECORD) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected_sites = ("TEM", "FP", "CSB", "PB", "IC3", "PY", "FPI", "PW", "HP")
    if tuple(row["site"] for row in rows) != expected_sites:
        raise AssertionError("Harpagifer site-block order changed")
    if tuple(int(row["supplement_SNP_n"]) for row in rows) != SUPPLEMENT_SNP_COUNTS:
        raise AssertionError("Harpagifer supplement count contract changed")
    cursor = 0
    for row in rows:
        start = int(row["start_zero_based"])
        end = int(row["end_exclusive"])
        if start != cursor or end <= start:
            raise AssertionError("Harpagifer site blocks are not a contiguous partition")
        if row["analysis_population"] not in POPULATION_ORDER:
            raise AssertionError("unexpected Harpagifer analysis population")
        cursor = end
    if cursor != SOURCE_CONTRACT["samples"]:
        raise AssertionError("Harpagifer site blocks do not cover all released samples")
    return rows


def reconstruct_population_mapping(
    source: Path, blocks_path: Path = BLOCKS_RECORD
) -> tuple[dict[str, str], dict[str, str], dict]:
    samples = read_vcf_samples(source)
    digest = ordered_sample_sha256(samples)
    if len(samples) != SOURCE_CONTRACT["samples"] or digest != SOURCE_CONTRACT[
        "ordered_sample_sha256"
    ]:
        raise AssertionError("released Harpagifer sample-order contract changed")
    if not all(sample.startswith("HBi_") for sample in samples[:93]) or not all(
        sample.startswith("Hpal_") for sample in samples[93:]
    ):
        raise AssertionError("Harpagifer nominal-species sample blocks changed")

    populations: dict[str, str] = {}
    sites: dict[str, str] = {}
    block_audit = []
    for row in read_blocks(blocks_path):
        start = int(row["start_zero_based"])
        end = int(row["end_exclusive"])
        selected = samples[start:end]
        for sample in selected:
            populations[sample] = row["analysis_population"]
            sites[sample] = row["site"]
        block_audit.append(
            {
                "site": row["site"],
                "analysis_population": row["analysis_population"],
                "start_zero_based": start,
                "end_exclusive": end,
                "VCF_n": len(selected),
                "supplement_SNP_n": int(row["supplement_SNP_n"]),
                "first_sample": selected[0],
                "last_sample": selected[-1],
                "provenance_note": row["provenance_note"],
            }
        )
    counts = {
        population: list(populations.values()).count(population)
        for population in POPULATION_ORDER
    }
    if counts != EXPECTED_SAMPLE_SCOPES["all_released_samples"]:
        raise AssertionError(f"unexpected Harpagifer reconstructed counts: {counts}")
    return populations, sites, {
        "status": "reconstructed_from_VCF_column_order_using_published_contiguous_site_order",
        "ordered_sample_sha256": digest,
        "population_counts": counts,
        "site_blocks": block_audit,
        "known_discrepancies": [
            f"Supplementary Table S1 SNP counts {list(SUPPLEMENT_SNP_COUNTS)} sum to {sum(SUPPLEMENT_SNP_COUNTS)} and report TEM n=2, while the VCF has 118 samples and three leading TEM-like columns.",
            f"Main-text Table 3 sample sizes {list(MAIN_TABLE3_SAMPLE_COUNTS)} sum to {sum(MAIN_TABLE3_SAMPLE_COUNTS)} and therefore do not crosswalk one-to-one to the 118-column released VCF.",
            "The block record enumerates only VCF-positive reconstructed site blocks, not every collection locality; Supplementary Table S1 omits FM and sequence tables use inconsistent site acronyms.",
        ],
        "guardrail": (
            "the grouping is a strong order-and-count reconstruction used for sensitivity analysis, "
            "not an author-supplied individual crosswalk"
        ),
    }


def audit_source_vcf(
    path: Path, populations: dict[str, str], sites: dict[str, str]
) -> dict:
    samples: list[str] | None = None
    population_columns: dict[str, list[int]] | None = None
    rows = 0
    positions: list[int] = []
    identifiers: set[str] = set()
    genotype_cells = 0
    missing_cells = 0
    invalid_cells = 0
    missing_by_sample = {sample: 0 for sample in populations}
    missing_by_population = {population: 0 for population in POPULATION_ORDER}
    minimum_called = {population: 10**9 for population in POPULATION_ORDER}
    maximum_called = {population: 0 for population in POPULATION_ORDER}
    chromosomes: set[str] = set()
    formats: set[str] = set()
    filters: set[str] = set()
    qualities: set[str] = set()
    infos: set[str] = set()
    locus_digest = hashlib.sha256()
    reference_unknown_major_used = False
    allowed = {"./.", "0/0", "0/1", "1/0", "1/1"}

    with open_text(path) as handle:
        for line in handle:
            if line.startswith("##Tassel=") and "Reference allele is not known" in line:
                reference_unknown_major_used = True
            if line.startswith("#CHROM"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                if samples != list(populations):
                    raise AssertionError("Harpagifer source order differs from reconstructed mapping")
                population_columns = {
                    population: [
                        9 + index
                        for index, sample in enumerate(samples)
                        if populations[sample] == population
                    ]
                    for population in POPULATION_ORDER
                }
                continue
            if line.startswith("#") or not line.strip():
                continue
            if samples is None or population_columns is None:
                raise ValueError("Harpagifer variant before #CHROM")
            fields = line.rstrip("\r\n").split("\t")
            rows += 1
            if len(fields) != 9 + len(samples):
                raise AssertionError("Harpagifer VCF row width changed")
            chromosomes.add(fields[0])
            position = int(fields[1])
            positions.append(position)
            identifiers.add(fields[2])
            locus_digest.update("\t".join(fields[:5]).encode("utf-8"))
            locus_digest.update(b"\n")
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
                raise AssertionError("Harpagifer source is not biallelic SNP-only")
            if fields[2] != f"T0_{position}":
                raise AssertionError("Harpagifer locus IDs no longer encode POS")
            for sample, cell in zip(samples, fields[9:]):
                genotype_cells += 1
                if cell not in allowed:
                    invalid_cells += 1
                    continue
                if cell == "./.":
                    missing_cells += 1
                    missing_by_sample[sample] += 1
                    missing_by_population[populations[sample]] += 1
            for population, columns in population_columns.items():
                copies = sum(len(called_alleles(fields[index])) for index in columns)
                minimum_called[population] = min(minimum_called[population], copies)
                maximum_called[population] = max(maximum_called[population], copies)

    if samples is None or rows != SOURCE_CONTRACT["variant_rows"]:
        raise AssertionError("unexpected Harpagifer source dimensions")
    if genotype_cells != SOURCE_CONTRACT["genotype_cells"]:
        raise AssertionError("unexpected Harpagifer genotype-cell count")
    if missing_cells != SOURCE_CONTRACT["fully_missing_genotype_cells"] or invalid_cells:
        raise AssertionError("unexpected Harpagifer missing/invalid GT contract")
    if chromosomes != {"0"} or formats != {"GT"} or filters != {"PASS"}:
        raise AssertionError("unexpected Harpagifer structural VCF fields")
    if qualities != {"."} or infos != {"."}:
        raise AssertionError("unexpected Harpagifer QUAL/INFO fields")
    if len(positions) != len(set(positions)) or positions != sorted(positions):
        raise AssertionError("Harpagifer positions are not unique and sorted")
    if len(identifiers) != rows:
        raise AssertionError("Harpagifer locus IDs are not unique")
    semantic_digest = locus_digest.hexdigest()
    if semantic_digest != SOURCE_CONTRACT["locus_semantic_sha256"]:
        raise AssertionError("Harpagifer ordered locus semantic digest changed")
    if not reference_unknown_major_used:
        raise AssertionError("Harpagifer unknown-reference header guardrail changed")

    sample_missingness = {
        sample: missing_by_sample[sample] / rows for sample in samples
    }
    observed_high_missing = {
        sample for sample, fraction in sample_missingness.items() if fraction > 0.25
    }
    if observed_high_missing != EXPECTED_HIGH_MISSING:
        raise AssertionError("Harpagifer >25% missing-sample contract changed")
    return {
        "samples": len(samples),
        "nominal_species_counts": {
            "H_bispinis_prefix": sum(sample.startswith("HBi_") for sample in samples),
            "H_palliolatus_prefix": sum(sample.startswith("Hpal_") for sample in samples),
        },
        "variant_rows": rows,
        "CHROM_values": sorted(chromosomes),
        "position_range": [min(positions), max(positions)],
        "unique_positions": len(set(positions)),
        "unique_IDs": len(identifiers),
        "ordered_CHROM_POS_ID_REF_ALT_sha256": semantic_digest,
        "FORMAT": sorted(formats),
        "FILTER": sorted(filters),
        "genotype_cells": genotype_cells,
        "fully_missing_genotype_cells": missing_cells,
        "missing_genotype_fraction": missing_cells / genotype_cells,
        "partial_or_invalid_genotype_cells": invalid_cells,
        "missing_cells_by_population": missing_by_population,
        "sample_missingness": sample_missingness,
        "samples_above_0_25_missingness": [
            sample for sample in samples if sample in observed_high_missing
        ],
        "called_copy_range_by_population": {
            population: {
                "minimum": minimum_called[population],
                "maximum": maximum_called[population],
            }
            for population in POPULATION_ORDER
        },
        "site_counts_from_reconstruction": {
            site: list(sites.values()).count(site) for site in dict.fromkeys(sites.values())
        },
        "linkage_guardrail": (
            "all released loci use CHROM=0 and no reference assembly or linkage map is supplied; "
            "physical independence cannot be verified"
        ),
        "allele_orientation_guardrail": (
            "the Tassel header states that the reference allele is unknown and the major allele "
            "was encoded as REF; REF must not be interpreted as ancestral"
        ),
    }


def materialize_manifests(
    source: Path,
    populations: dict[str, str],
    sites: dict[str, str],
    source_audit: dict,
    output_dir: Path,
) -> tuple[dict[str, Path], dict]:
    samples = read_vcf_samples(source)
    missingness = source_audit["sample_missingness"]
    selected = {
        "all_released_samples": samples,
        "sample_missingness_le_0_25": [
            sample for sample in samples if missingness[sample] <= 0.25
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    audits = {}
    for scope, scope_samples in selected.items():
        path = output_dir / f"harpagifer.{scope}.tsv"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("sample\tpopulation\n")
            for sample in scope_samples:
                handle.write(f"{sample}\t{populations[sample]}\n")
        mapping = read_manifest(path)
        counts = {
            population: list(mapping.values()).count(population)
            for population in POPULATION_ORDER
        }
        if counts != EXPECTED_SAMPLE_SCOPES[scope]:
            raise AssertionError(f"unexpected {scope} Harpagifer counts: {counts}")
        paths[scope] = path
        audits[scope] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "samples": len(mapping),
            "population_counts": counts,
            "reconstructed_site_counts": {
                site: sum(sites[sample] == site for sample in mapping)
                for site in dict.fromkeys(sites.values())
            },
            "selection": (
                "all released VCF columns"
                if scope == "all_released_samples"
                else "prespecified sample-level sensitivity retaining missing-GT fraction <=0.25"
            ),
        }
    audits["sample_missingness_sensitivity"] = {
        "threshold": 0.25,
        "comparison": "missing GT cells / 2,993 released loci",
        "excluded_samples_in_VCF_order": [
            sample for sample in samples if sample in EXPECTED_HIGH_MISSING
        ],
        "guardrail": (
            "not an outcome-based or direction-model-based sample filter, but strongly "
            "group-reweighting: removes 10/50 P1, 1/43 P2, and 0/25 P3"
        ),
    }
    return paths, audits


def _summary(values: list[float]) -> dict | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "replicates_retained": int(len(array)),
        "bootstrap_sd": float(np.std(array, ddof=1)) if len(array) > 1 else 0.0,
        "percentile_95_interval": [
            float(np.percentile(array, 2.5)),
            float(np.percentile(array, 97.5)),
        ],
    }


def iid_locus_bootstrap(
    numerator: np.ndarray,
    denominator: np.ndarray,
    f3_values: np.ndarray,
    diagnostic: np.ndarray,
    *,
    seed: int = 20260711,
    replicates: int = 500,
) -> dict:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    rng = np.random.default_rng(seed)
    projections: list[float] = []
    diagnostic_projections: list[float] = []
    f3_estimates: list[float] = []
    n = len(numerator)
    for _ in range(replicates):
        draw = rng.integers(0, n, size=n)
        total = float(denominator[draw].sum())
        if total > 0:
            projections.append(float(numerator[draw].sum() / total))
        diagnostic_total = float(
            np.where(diagnostic[draw], denominator[draw], 0.0).sum()
        )
        if diagnostic_total > 0:
            diagnostic_projections.append(
                float(
                    np.where(diagnostic[draw], numerator[draw], 0.0).sum()
                    / diagnostic_total
                )
            )
        f3_estimates.append(float(np.mean(f3_values[draw])))
    return {
        "method": "naive IID resampling of released GBS loci",
        "seed": seed,
        "requested_replicates": replicates,
        "loci": n,
        "projection_all_loci": _summary(projections),
        "projection_diagnostic_loci": _summary(diagnostic_projections),
        "f3_finite_called_copy_corrected": _summary(f3_estimates),
        "guardrail": (
            "fixed-sample conditional sensitivity only; all source CHROM values are 0 and no "
            "linkage map exists, so independence is unverified and this is not chromosome-block uncertainty"
        ),
    }


def frequency_geometry(
    vcf: Path,
    manifest_path: Path,
    pop_order: tuple[str, str, str] = POPULATION_ORDER,
    diagnostic_threshold: float = 0.95,
    bootstrap_replicates: int = 500,
) -> dict:
    mapping = read_manifest(manifest_path)
    if set(mapping.values()) != set(pop_order):
        raise ValueError("manifest populations do not match requested order")
    columns = None
    values = {population: [] for population in pop_order}
    called_copy_counts = {population: [] for population in pop_order}
    with open_text(vcf) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                sample_column = {sample: 9 + index for index, sample in enumerate(samples)}
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
                raise ValueError("Harpagifer panel variant before #CHROM")
            fields = line.rstrip("\r\n").split("\t")
            for population in pop_order:
                alleles = [
                    allele
                    for index in columns[population]
                    for allele in called_alleles(fields[index])
                ]
                if not alleles:
                    raise ValueError(f"locus has no called {population} alleles")
                values[population].append(
                    sum(allele == "1" for allele in alleles) / len(alleles)
                )
                called_copy_counts[population].append(len(alleles))
    if columns is None or not values[pop_order[0]]:
        raise ValueError("Harpagifer panel has no usable loci")

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
        raise AssertionError("P2 requires at least two called gene copies")
    correction = p2 * (1.0 - p2) / (n2 - 1.0)
    f3_corrected = f3_plugin - correction

    def project(mask: np.ndarray) -> float | None:
        total = float(denominator[mask].sum())
        if not np.any(mask) or total == 0.0:
            return None
        return float(numerator[mask].sum() / total)

    return {
        "n_loci": int(len(p1)),
        "P2_projection_from_P1_toward_P3_all_loci": project(
            np.ones(len(p1), dtype=bool)
        ),
        "diagnostic_threshold_abs_P3_minus_P1_frequency": diagnostic_threshold,
        "maximum_abs_P3_minus_P1_frequency": float(np.max(np.abs(axis))),
        "diagnostic_loci": int(diagnostic.sum()),
        "P2_projection_from_P1_toward_P3_diagnostic_loci": project(diagnostic),
        "f3_like_plugin_P2_P1_P3": float(np.mean(f3_plugin)),
        "mean_P2_finite_called_copy_correction": float(np.mean(correction)),
        "f3_P2_P1_P3_finite_called_copy_corrected": float(np.mean(f3_corrected)),
        "mean_squared_frequency_distance": {
            "P1_P2": float(np.mean((p1 - p2) ** 2)),
            "P2_P3": float(np.mean((p2 - p3) ** 2)),
            "P1_P3": float(np.mean((p1 - p3) ** 2)),
        },
        "iid_locus_bootstrap": iid_locus_bootstrap(
            numerator,
            denominator,
            f3_corrected,
            diagnostic,
            replicates=bootstrap_replicates,
        ),
        "interpretation": (
            "sample-frequency geometry only; projection is reference-flip invariant but is not "
            "bounded ancestry or temporal direction. The finite-copy f3 correction assumes "
            "independent binomial called-copy sampling and is not generally unbiased. The "
            "prespecified 0.95 diagnostic threshold "
            "is retained even when it yields zero loci and is not tuned post hoc."
        ),
    }


def run_panels(
    source: Path,
    manifests: dict[str, Path],
    cache: Path,
    cap: int,
    direction_head,
    gate_head,
) -> list[dict]:
    panels = []
    observed_locus_hashes = {name: set() for name in EXPECTED_FILTERS}
    for sample_scope, manifest in manifests.items():
        for filter_name, strict in (
            ("standard_contract", False),
            ("within_population_polymorphism", True),
        ):
            panel_vcf = cache / f"harpagifer.{sample_scope}.{filter_name}.vcf"
            panel_popmap = cache / f"harpagifer.{sample_scope}.{filter_name}.popmap.tsv"
            audit = prepare_vcf(
                source,
                manifest,
                panel_vcf,
                panel_popmap,
                cap=cap,
                seed=20260711,
                polymorphic_within_each_population=strict,
            )
            expected_filter = EXPECTED_FILTERS[filter_name]
            if audit["counts"]["retained_after_cap"] != expected_filter["loci"]:
                raise AssertionError(f"unexpected Harpagifer {filter_name} locus count")
            if audit["ordered_locus_sha256"] != expected_filter[
                "ordered_locus_sha256"
            ]:
                raise AssertionError(f"unexpected Harpagifer {filter_name} locus hash")
            observed_locus_hashes[filter_name].add(audit["ordered_locus_sha256"])
            expectation = {
                "benchmark_role": "candidate_direction_sensitivity",
                "candidate_class": "A",
                "candidate_forward_direction": "NorthPatagonia (P1) -> SouthPatagonia (P2)",
                "expected_gate": None,
                "direction_basis": (
                    "dominant same-data BayesAss estimate reported by the source study; "
                    "not a clean one-edge history or external label"
                ),
                "label_source_reuse": (
                    "the source paper inferred clusters and migration on the same 2,993 GBS SNPs; "
                    "no held-out loci or samples define the candidate label"
                ),
                "sample_scope": sample_scope,
                "sample_filter": (
                    "all released samples"
                    if sample_scope == "all_released_samples"
                    else "prespecified <=25% sample-missingness sensitivity"
                ),
                "locus_filter_variant": (
                    "both alleles called within P1, P2, and P3; strong ascertainment"
                    if strict
                    else "both alleles called across the complete three-population panel"
                ),
                "tree_contract_status": (
                    "operational three-cluster order, not a rooted species tree; source study "
                    "argued for one evolutionary unit"
                ),
                "accuracy_eligible": False,
            }
            panel = score_panel(
                f"harpagifer_{sample_scope}_{filter_name}",
                panel_vcf,
                panel_popmap,
                POPULATION_ORDER,
                audit,
                direction_head[0],
                direction_head[1],
                expectation,
            )
            panel["population_order"]["tree_contract_status"] = expectation[
                "tree_contract_status"
            ]
            add_gate_score(panel, gate_head[0], gate_head[1])
            panel["model_free_comparator"] = frequency_geometry(panel_vcf, manifest)
            direction_rms = panel["simulation_feature_shift"]["rms_z"]
            gate_rms = panel["simulation_gate_feature_shift"]["rms_z"]
            severe = max(direction_rms, gate_rms) > 10
            prediction = panel["simulation_head"]["predicted_class"]
            panel["adjudication"] = {
                "candidate_class": "A",
                "predicted_class": prediction,
                "matches_candidate_reference": prediction == "A",
                "gate_score": panel["simulation_gate"]["appreciable_score"],
                "accuracy_eligible": False,
                "severe_OOD": severe,
                "severe_OOD_rule": "max(direction RMS-z, gate RMS-z) > 10; heuristic, not calibrated support",
                "natural_data_call_status": (
                    "abstain_severe_OOD" if severe else "diagnostic_only"
                ),
            }
            panels.append(panel)
    if any(len(hashes) != 1 for hashes in observed_locus_hashes.values()):
        raise AssertionError("Harpagifer sample scopes do not share exact locus sets")
    return panels


def summarize_outcomes(panels: list[dict]) -> dict:
    abstained = sum(
        panel["adjudication"]["natural_data_call_status"] == "abstain_severe_OOD"
        for panel in panels
    )
    return {
        "analytic_sensitivity_runs": len(panels),
        "unique_biological_systems": 1,
        "candidate_comparisons": 1,
        "independent_validation_panels": 0,
        "accuracy_estimate": None,
        "accuracy_available": False,
        "candidate_class": "A",
        "candidate_label_status": "literature_dominant_same_SNP_sensitivity",
        "severe_OOD_panels": sum(
            panel["adjudication"]["severe_OOD"] for panel in panels
        ),
        "abstained_panels": abstained,
        "all_panels_abstain_due_to_severe_OOD": abstained == len(panels),
        "raw_OOD_head_matches_candidate_A": sum(
            panel["adjudication"]["matches_candidate_reference"] for panel in panels
        ),
        "raw_OOD_head_prediction_counts": {
            label: sum(
                panel["simulation_head"]["predicted_class"] == label
                for panel in panels
            )
            for label in ("A", "B", "C")
        },
        "raw_OOD_gate_threshold_crossings_at_0.5": sum(
            panel["simulation_gate"]["called_at_0.5"] for panel in panels
        ),
        "interpretation": (
            "one same-data, mixed-direction Harpagifer comparison repeated over two sample scopes "
            "and two locus-ascertainment filters; panel counts are not accuracy trials"
        ),
    }


def validate_sources_record(path: Path = SOURCES_RECORD) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["data_doi"] != "10.5061/dryad.jwstqjq9q":
        raise AssertionError("Harpagifer source DOI changed")
    if record["archive"]["digest_policy"] != ARCHIVE["digest_policy"]:
        raise AssertionError("Harpagifer source-record archive policy changed")
    if record["file"]["sha256"] != FILE["sha256"]:
        raise AssertionError("Harpagifer source-record VCF hash changed")
    if record["mapping_provenance"]["ordered_sample_sha256"] != SOURCE_CONTRACT[
        "ordered_sample_sha256"
    ]:
        raise AssertionError("Harpagifer source-record sample hash changed")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    parser.add_argument("--source-vcf")
    parser.add_argument("--archive")
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
    source = (
        Path(args.source_vcf).resolve()
        if args.source_vcf
        else cache / FILE["key"]
    )
    archive = (
        Path(args.archive).resolve()
        if args.archive
        else cache / ARCHIVE["key"]
    )
    verified = ensure_source(source, archive, args.download_missing)
    sources_record = validate_sources_record()
    populations, sites, mapping_audit = reconstruct_population_mapping(source)
    source_audit = audit_source_vcf(source, populations, sites)
    manifests, manifest_audit = materialize_manifests(
        source, populations, sites, source_audit, cache / "manifests"
    )

    data_root = Path(args.data_root).resolve()
    direction_head = simulation_direction_head(data_root, max_depth=MAX_DEPTH)
    gate_head = simulation_gate_head(data_root, max_depth=MAX_DEPTH)
    panels = run_panels(
        source,
        manifests,
        cache,
        args.cap,
        direction_head,
        gate_head,
    )
    result = {
        "schema_version": "dnnaic-harpagifer-external-benchmark-v1",
        "git": git_revision(),
        "runtime": runtime_helpers.runtime_audit(),
        "guardrail": (
            "candidate-class-A same-data sensitivity only; no panel is independent, gold truth, "
            "or accuracy-eligible, and severe OOD outputs abstain"
        ),
        "source": {
            "record": DRYAD_RECORD,
            "data_doi": "10.5061/dryad.jwstqjq9q",
            "license": "CC0-1.0",
            "verified": verified,
            "sources_record": {
                "path": str(SOURCES_RECORD),
                "sha256": sha256_file(SOURCES_RECORD),
                "content": sources_record,
            },
            "source_vcf_contract": source_audit,
            "mapping_audit": mapping_audit,
            "manifest_audit": manifest_audit,
            "ascertainment_guardrails": (
                "ApeKI GBS/UNEAK neutral release, source MAF >=0.05 and site call-rate >=0.75, "
                "removal of loci with HWE deviation in at least 60% of localities after FDR, "
                "removal of 68 Bayescan differentiated candidates from 3,061 SNPs, "
                "same-data clustering and migration labels, reconstructed site membership, and unknown linkage"
            ),
        },
        "published_evidence": {
            "candidate_direction": (
                "source-study BayesAss means reported 18.8% north-to-south Patagonia versus "
                "2.03% reciprocal migration"
            ),
            "other_reported_edges": (
                "north-to-Falklands/Malvinas 1.15% and south-to-Falklands/Malvinas 2.54%; "
                "reverse island-to-Patagonia estimates were negligible and not plotted"
            ),
            "method_guardrail": (
                "BayesAss used 10,000 iterations with 10% burn-in in the source paper; reported "
                "fractions are not per-generation msprime migration rates"
            ),
            "biological_guardrail": (
                "the source paper rejected incipient speciation and described one evolutionary "
                "unit despite three SNP clusters; P3 is a population cluster, not a species outgroup"
            ),
            "newer_related_work": (
                "Bernal-Duran et al. 2024 (10.1111/mec.17360) studies H. antarcticus with a "
                "different 143-sample/20,778-SNP Western Antarctic Peninsula dataset; it informs "
                "geographic-connectivity nuisance but does not update or validate this label"
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
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
