#!/usr/bin/env python3
"""Run a guarded tinkerbird asymmetric-backcrossing transfer benchmark.

Kirschel et al. (2020) inferred a majority cross between extoni-enriched
females and relatively pusillus-enriched males from autosomal, Z-chromosome,
and mitochondrial ancestry.  With the source-era extoni reference pool as P1,
their 14 admixed offspring as P2, and the source-era pusillus reference pool as
P3, that majority orientation is a candidate DNNaic class C label.

It is not exact gold-standard direction or external validation.  Nine birds
support the majority cross, three the reciprocal, and two are equal.  P2 was
selected as admixed from this same ddRAD dataset, and its label reuses ancestry
estimated from these data.  The pooled hybrids also violate the panmictic
tree-leaf assumption.  Every result is therefore an explicitly circular,
selection-enriched OOD failure diagnostic and is excluded from accuracy.

The source encodes female Z genotypes as diploid, including heterozygotes.
Until sex-aware ploidy is implemented, this runner uses only source scaffolds
anchored to autosomes.  It reports both the ordinary SNP-level contract and a
one-source-SNP-per-scaffold sensitivity, with scaffold-blocked uncertainty.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
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
    verify_file,
)


DEFAULT_CACHE = REPO / "data" / "real" / "tinkerbird_external_benchmark"
DEFAULT_RESULTS = REPO / "results" / "tinkerbird_external_benchmark_2026_07_11"
DEFAULT_CAP = 15_000
MANIFEST = MANIFEST_DIR / "tinkerbird_contact.tsv"
POPULATION_ORDER = ("LegacyExtoniReference", "Admixed14", "LegacyPusillusReference")
AUTOSOME_PATTERN = re.compile(r"^scaf_(?:\d{2}|01A|04A)_.+$")

SOURCE = {
    "key": "revision.recode.vcf.gz",
    "bytes": 3_223_094,
    "sha256": "51144aabaddac820269af2f8ff5648393b69a20be3c0398a72ca4d9c83756a51",
    "zenodo_md5": "a8258458fba61a3ebbfa33fea09a347e",
    "download": (
        "https://zenodo.org/api/records/4430805/files/"
        "revision.recode.vcf.gz/content"
    ),
    "record": "https://zenodo.org/records/4430805",
    "data_doi": "10.5061/dryad.jm63xsj87",
    "paper_doi": "10.1111/mec.15691",
    "license": "CC0-1.0",
    "expected_samples": 85,
    "expected_variants": 104_933,
    "expected_anchored_autosome_variants": 57_913,
    "expected_Z_variants": 1_533,
    "expected_unplaced_variants": 45_487,
    "paper_reported_Z_variants": 1_532,
    "assay": "linked ddRAD SNPs aligned to a draft P. pusillus genome; Beagle-imputed VCF",
}

EXTONI_REFERENCE = (
    "AR93101b",
    "AR93102b",
    "AR93104",
    "AR93105",
    "AR93107",
    "AR93109",
    "AR93110",
    "AR93141",
    "AR93142",
    "AR93143",
    "AR93144",
    "AR93145",
)

TABLE2_NAMED_ADMIXED = (
    "AR93112",
    "AR93115",
    "AR93118",
    "AR93119",
    "AR93122",
    "AR93153",
    "AR93157",
    "AR93159",
    "AR93167",
    "AR93168",
    "AR93170",
    "AR93172",
    "K69365",
)

# Table 2 contains only 13 rows, but the Discussion explicitly identifies
# AR93163 as the second pusillus-haplotype female with an extoni father.
ADMIXED14 = TABLE2_NAMED_ADMIXED + ("AR93163",)
ADMIXED_FEMALES = (
    "AR93115",
    "AR93122",
    "AR93153",
    "AR93163",
    "AR93167",
    "AR93168",
    "AR93172",
    "K69365",
)

PUSILLUS_REFERENCE = (
    "AR93128",
    "AR93129",
    "AR93130",
    "AR93131",
    "AR93132",
    "AR93133",
    "AR93134",
    "AR93135",
    "AR93138",
    "AR93139",
    "AR93140",
    "AR93176",
    "AR93177",
    "AR93178",
    "AR93179",
    "AR93180",
    "AR93181",
)


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(output)


def ensure_source(path: Path, download_missing: bool) -> dict:
    if not path.exists():
        if not download_missing:
            raise FileNotFoundError(path)
        _download(SOURCE["download"], path)
    return verify_file(path, SOURCE["bytes"], SOURCE["sha256"])


def is_anchored_autosome(scaffold: str) -> bool:
    return AUTOSOME_PATTERN.fullmatch(scaffold) is not None


def validate_manifest(path: Path = MANIFEST) -> dict[str, str]:
    mapping = read_manifest(path)
    expected = {
        **{sample: POPULATION_ORDER[0] for sample in EXTONI_REFERENCE},
        **{sample: POPULATION_ORDER[1] for sample in ADMIXED14},
        **{sample: POPULATION_ORDER[2] for sample in PUSILLUS_REFERENCE},
    }
    if mapping != expected:
        raise AssertionError("tinkerbird manifest differs from the pinned source-derived design")
    return mapping


def audit_source_vcf(path: Path, mapping: dict[str, str]) -> dict:
    samples: list[str] | None = None
    sample_columns: dict[str, int] = {}
    counts = {"anchored_autosome": 0, "Z": 0, "unplaced": 0}
    female_Z_cells = 0
    female_Z_diploid_cells = 0
    female_Z_heterozygous_cells = 0
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                if samples is not None:
                    raise ValueError(f"{path}: duplicate #CHROM header")
                samples = line.rstrip("\r\n").split("\t")[9:]
                sample_columns = {sample: 9 + index for index, sample in enumerate(samples)}
                continue
            if line.startswith("#") or not line.strip():
                continue
            if samples is None:
                raise ValueError(f"{path}: data before #CHROM header")
            fields = line.rstrip("\r\n").split("\t")
            scaffold = fields[0]
            if is_anchored_autosome(scaffold):
                counts["anchored_autosome"] += 1
            elif scaffold.startswith("scaf_Z_"):
                counts["Z"] += 1
                for sample in ADMIXED_FEMALES:
                    genotype = (
                        fields[sample_columns[sample]]
                        .split(":", 1)[0]
                        .replace("|", "/")
                    )
                    alleles = [allele for allele in genotype.split("/") if allele != "."]
                    female_Z_cells += 1
                    if len(alleles) == 2:
                        female_Z_diploid_cells += 1
                        female_Z_heterozygous_cells += int(len(set(alleles)) == 2)
            else:
                counts["unplaced"] += 1
    if samples is None:
        raise ValueError(f"{path}: missing #CHROM header")
    if len(samples) != SOURCE["expected_samples"] or len(set(samples)) != len(samples):
        raise AssertionError("unexpected tinkerbird source sample contract")
    expected_counts = {
        "anchored_autosome": SOURCE["expected_anchored_autosome_variants"],
        "Z": SOURCE["expected_Z_variants"],
        "unplaced": SOURCE["expected_unplaced_variants"],
    }
    if counts != expected_counts or sum(counts.values()) != SOURCE["expected_variants"]:
        raise AssertionError(f"unexpected tinkerbird source scaffold counts: {counts}")
    missing = sorted(set(mapping) - set(samples))
    if missing:
        raise AssertionError(f"manifest samples absent from source VCF: {missing}")
    if female_Z_cells != 12_264 or female_Z_diploid_cells != female_Z_cells:
        raise AssertionError("unexpected female Z genotype encoding")
    if female_Z_heterozygous_cells != 1_302:
        raise AssertionError("unexpected female Z heterozygote count")
    return {
        "samples": len(samples),
        "variant_rows": sum(counts.values()),
        "scaffold_assignment_counts": counts,
        "manifest_samples_present": len(mapping),
        "manifest_population_counts": {
            population: list(mapping.values()).count(population)
            for population in POPULATION_ORDER
        },
        "female_Z_ploidy_audit": {
            "female_samples": list(ADMIXED_FEMALES),
            "Z_variant_rows": counts["Z"],
            "female_by_Z_cells": female_Z_cells,
            "cells_encoded_with_two_called_alleles": female_Z_diploid_cells,
            "heterozygous_cells": female_Z_heterozygous_cells,
            "conclusion": (
                "female birds are ZW, so these calls cannot be consumed as ordinary diploid "
                "autosomal genotypes; every scored panel is anchored-autosome-only"
            ),
        },
        "source_paper_count_discrepancy": (
            "source VCF has 1,533 scaf_Z variants; paper section 3.3 reports 1,532"
        ),
    }


def subset_anchored_autosomes(source: Path, output: Path) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    retained_rows = 0
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
            if is_anchored_autosome(line.split("\t", 1)[0]):
                outgoing.write(line.rstrip("\r\n") + "\n")
                retained_rows += 1
    if not saw_header or retained_rows != SOURCE["expected_anchored_autosome_variants"]:
        raise AssertionError("anchored-autosome extraction violated its source contract")
    return {
        "source": str(source),
        "selection": (
            "scaffold name matches scaf_<two-digit>_*, scaf_01A_*, or scaf_04A_*; "
            "Z and unplaced scaffolds excluded"
        ),
        "source_variant_rows": source_rows,
        "retained_variant_rows": retained_rows,
        "derived_vcf": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }


def thin_one_per_scaffold(
    source: Path,
    output: Path,
    *,
    expected_source_rows: int | None = SOURCE["expected_anchored_autosome_variants"],
    expected_source_scaffolds: int | None = 8_815,
    expected_retained_scaffolds: int | None = 8_744,
) -> dict:
    """Keep the first structurally eligible source-ordered SNP per scaffold."""
    output.parent.mkdir(parents=True, exist_ok=True)
    source_scaffolds: set[str] = set()
    retained_scaffolds: set[str] = set()
    source_rows = 0
    retained_rows = 0
    structurally_ineligible_rows = 0
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
            fields = line.rstrip("\r\n").split("\t")
            scaffold = fields[0]
            source_scaffolds.add(scaffold)
            if scaffold in retained_scaffolds:
                continue
            if (
                len(fields) < 7
                or len(fields[3]) != 1
                or len(fields[4]) != 1
                or "," in fields[4]
                or fields[6] not in ("PASS", ".")
            ):
                structurally_ineligible_rows += 1
                continue
            retained_scaffolds.add(scaffold)
            outgoing.write("\t".join(fields) + "\n")
            retained_rows += 1
    if not saw_header:
        raise AssertionError("one-per-scaffold input lacks a VCF header")
    if expected_source_rows is not None and source_rows != expected_source_rows:
        raise AssertionError("one-per-scaffold input is not the anchored-autosome source")
    if (
        expected_source_scaffolds is not None
        and len(source_scaffolds) != expected_source_scaffolds
    ):
        raise AssertionError(
            f"unexpected source scaffold count: {len(source_scaffolds)}"
        )
    if (
        expected_retained_scaffolds is not None
        and retained_rows != expected_retained_scaffolds
    ):
        raise AssertionError(
            f"unexpected structurally eligible scaffold count: {retained_rows}"
        )
    return {
        "source": str(source),
        "source_variant_rows": source_rows,
        "selection": (
            "first source-ordered single-base biallelic PASS/dot SNP per anchored-autosomal "
            "scaffold before panel genotype/polymorphism filtering"
        ),
        "retained_variant_rows": retained_rows,
        "source_scaffolds": len(source_scaffolds),
        "structurally_ineligible_rows_seen_before_first_eligible_SNP": (
            structurally_ineligible_rows
        ),
        "scaffolds_without_structurally_eligible_SNP": (
            len(source_scaffolds) - len(retained_scaffolds)
        ),
        "retained_scaffolds": len(retained_scaffolds),
        "derived_vcf": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }


def published_sample_audit() -> dict:
    return {
        "study_contact_zone_samples": 49,
        "prose_reported_admixed_samples": 14,
        "prose_reported_parental_direction_counts": {
            "more_extoni_mother_more_pusillus_father": 9,
            "reverse": 3,
            "equal_autosomal_and_Z_ancestry": 2,
        },
        "table_2_named_samples": len(TABLE2_NAMED_ADMIXED),
        "table_2_sample_ids": list(TABLE2_NAMED_ADMIXED),
        "fourteenth_sample": {
            "sample": "AR93163",
            "basis": (
                "omitted from Table 2 but explicitly named in the Discussion as a "
                "pusillus-haplotype female with an extoni father"
            ),
        },
        "candidate_direction_mapping": (
            "majority cross has an extoni-enriched mother and relatively pusillus-enriched "
            "father; with extoni=P1, admixed=P2, pusillus=P3 this is candidate class C"
        ),
        "truth_strength": (
            "heterogeneous majority heuristic: 9/14 candidate C, 3/14 reciprocal, 2/14 equal"
        ),
        "direct_label_source_reuse": (
            "P2 was selected by fastSTRUCTURE ancestry from this deposited ddRAD dataset, and "
            "the direction label reuses autosomal-versus-Z ancestry in these same birds"
        ),
        "accuracy_eligible": False,
    }


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


def scaffold_block_bootstrap(
    scaffolds: list[str],
    projection_numerator: np.ndarray,
    projection_denominator: np.ndarray,
    f3_values: np.ndarray,
    diagnostic: np.ndarray,
    *,
    seed: int = 20260711,
    replicates: int = 500,
) -> dict:
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    unique, inverse = np.unique(np.asarray(scaffolds, dtype="U"), return_inverse=True)
    block_n = np.bincount(inverse).astype(float)
    block_num = np.bincount(inverse, weights=projection_numerator)
    block_den = np.bincount(inverse, weights=projection_denominator)
    block_f3 = np.bincount(inverse, weights=f3_values)
    block_diag_num = np.bincount(
        inverse, weights=np.where(diagnostic, projection_numerator, 0.0)
    )
    block_diag_den = np.bincount(
        inverse, weights=np.where(diagnostic, projection_denominator, 0.0)
    )
    rng = np.random.default_rng(seed)
    projections: list[float] = []
    diagnostic_projections: list[float] = []
    f3_estimates: list[float] = []
    for _ in range(replicates):
        draw = rng.integers(0, len(unique), size=len(unique))
        denominator = float(block_den[draw].sum())
        if denominator > 0:
            projections.append(float(block_num[draw].sum() / denominator))
        diagnostic_denominator = float(block_diag_den[draw].sum())
        if diagnostic_denominator > 0:
            diagnostic_projections.append(
                float(block_diag_num[draw].sum() / diagnostic_denominator)
            )
        f3_estimates.append(float(block_f3[draw].sum() / block_n[draw].sum()))
    return {
        "method": "nonparametric scaffold-block bootstrap with equal scaffold resampling",
        "seed": seed,
        "requested_replicates": replicates,
        "scaffold_blocks": int(len(unique)),
        "maximum_loci_per_scaffold": int(block_n.max()),
        "projection_all_loci": _summary(projections),
        "projection_diagnostic_loci": _summary(diagnostic_projections),
        "f3_P2_P1_P3": _summary(f3_estimates),
        "guardrail": (
            "descriptive linked-locus sensitivity only; case ascertainment and label reuse remain"
        ),
    }


def frequency_projection(
    vcf: Path,
    manifest_path: Path,
    pop_order: tuple[str, str, str] = POPULATION_ORDER,
    diagnostic_threshold: float = 0.95,
    bootstrap_replicates: int = 500,
) -> dict:
    """Reference-invariant projection and f3 with scaffold-blocked uncertainty."""
    mapping = read_manifest(manifest_path)
    if set(mapping.values()) != set(pop_order):
        raise ValueError("manifest populations do not match the requested order")
    columns: dict[str, list[int]] | None = None
    values = {population: [] for population in pop_order}
    scaffolds: list[str] = []
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
            scaffolds.append(fields[0])
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
    if columns is None or not scaffolds:
        raise ValueError(f"{vcf}: no usable loci")
    p1, p2, p3 = (
        np.asarray(values[population], dtype=float) for population in pop_order
    )
    axis = p3 - p1
    numerator = (p2 - p1) * axis
    denominator = axis**2
    diagnostic = np.abs(axis) >= diagnostic_threshold
    f3_values = (p2 - p1) * (p2 - p3)

    def project(mask: np.ndarray) -> float | None:
        total_denominator = float(denominator[mask].sum())
        if not np.any(mask) or total_denominator == 0.0:
            return None
        return float(numerator[mask].sum() / total_denominator)

    return {
        "n_loci": int(len(p1)),
        "scaffolds": int(len(set(scaffolds))),
        "P2_projection_from_P1_toward_P3_all_loci": project(
            np.ones(len(p1), dtype=bool)
        ),
        "diagnostic_threshold_abs_P3_minus_P1_frequency": diagnostic_threshold,
        "diagnostic_loci": int(diagnostic.sum()),
        "P2_projection_from_P1_toward_P3_diagnostic_loci": project(diagnostic),
        "f3_P2_P1_P3": float(np.mean(f3_values)),
        "mean_squared_frequency_distance": {
            "P1_P2": float(np.mean((p1 - p2) ** 2)),
            "P2_P3": float(np.mean((p2 - p3) ** 2)),
            "P1_P3": float(np.mean((p1 - p3) ** 2)),
        },
        "scaffold_block_bootstrap": scaffold_block_bootstrap(
            scaffolds,
            numerator,
            denominator,
            f3_values,
            diagnostic,
            replicates=bootstrap_replicates,
        ),
        "interpretation": (
            "reference-invariant descriptive geometry; projection is not bounded ancestry, and "
            "neither projection nor f3 estimates temporal direction"
        ),
    }


def locus_keys(vcf: Path) -> set[tuple[str, str, str, str]]:
    keys = set()
    with open_text(vcf) as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            key = (fields[0], fields[1], fields[3], fields[4])
            if key in keys:
                raise AssertionError(f"duplicate locus key in {vcf}: {key}")
            keys.add(key)
    return keys


def locus_overlap(standard_vcf: Path, strict_vcf: Path) -> dict:
    standard = locus_keys(standard_vcf)
    strict = locus_keys(strict_vcf)
    intersection = standard & strict
    union = standard | strict
    return {
        "standard_loci": len(standard),
        "within_population_polymorphism_loci": len(strict),
        "intersection": len(intersection),
        "union": len(union),
        "jaccard": len(intersection) / len(union),
        "fraction_of_standard_in_intersection": len(intersection) / len(standard),
        "fraction_of_strict_in_intersection": len(intersection) / len(strict),
        "interpretation": (
            "the filters are separate ascertainment analyses; any independently capped sets "
            "also differ through reservoir sampling"
        ),
    }


def runtime_audit() -> dict:
    packages = {}
    for package in ("numpy", "scikit-learn", "padze"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "argv": sys.argv,
        "thread_environment": {name: os.environ.get(name) for name in thread_variables},
        "process_priority": "BelowNormal requested at runner start on Windows",
    }


def run_panels(
    scope_sources: dict[str, tuple[Path, str]],
    cache: Path,
    cap: int,
    direction_head,
    gate_head,
) -> tuple[list[dict], dict]:
    panels = []
    paths: dict[tuple[str, str], Path] = {}
    for scope, (source, linkage_contract) in scope_sources.items():
        for filter_name, strict in (
            ("standard_contract", False),
            ("within_population_polymorphism", True),
        ):
            panel_vcf = cache / f"tinkerbird.{scope}.{filter_name}.vcf"
            panel_popmap = cache / f"tinkerbird.{scope}.{filter_name}.popmap.tsv"
            audit = prepare_vcf(
                source,
                MANIFEST,
                panel_vcf,
                panel_popmap,
                cap=cap,
                seed=20260711,
                polymorphic_within_each_population=strict,
            )
            audit["scope"] = scope
            audit["linkage_contract"] = linkage_contract
            expectation = {
                "candidate_class": "C",
                "candidate_forward_direction": (
                    "LegacyPusillusReference (P3) ancestry -> Admixed14 offspring (P2)"
                ),
                "direction_truth_strength": (
                    "heterogeneous majority heuristic: 9/14 candidate C, 3/14 reciprocal, "
                    "2/14 equal"
                ),
                "direction_basis": (
                    "published parental-cross inference from autosomal, Z-chromosome, and "
                    "mitochondrial ancestry"
                ),
                "direct_label_source_reuse": True,
                "selection_guardrail": (
                    "P2 was selected for intermediate ancestry in this same ddRAD dataset"
                ),
                "accuracy_eligible": False,
                "locus_filter_variant": (
                    "both alleles called within each of P1, P2, and P3; strong ascertainment, "
                    "not a quality filter"
                    if strict
                    else "both alleles called across the complete three-population panel"
                ),
                "linkage_scope": linkage_contract,
            }
            panel = score_panel(
                f"tinkerbird_{scope}_{filter_name}",
                panel_vcf,
                panel_popmap,
                POPULATION_ORDER,
                audit,
                direction_head[0],
                direction_head[1],
                expectation,
            )
            panel["population_order"].update(
                {
                    "tree_contract_status": "violated_by_case_ascertained_hybrid_pool",
                    "tree_contract": (
                        "coordinate order imposed for feature extraction; P2 is not a "
                        "panmictic tree leaf"
                    ),
                }
            )
            add_gate_score(panel, gate_head[0], gate_head[1])
            panel["model_free_comparator"] = frequency_projection(
                panel_vcf, MANIFEST, POPULATION_ORDER
            )
            predicted = panel["simulation_head"]["predicted_class"]
            direction_rms = panel["simulation_feature_shift"]["rms_z"]
            gate_rms = panel["simulation_gate_feature_shift"]["rms_z"]
            panel["adjudication"] = {
                "candidate_class": "C",
                "predicted_class": predicted,
                "matches_candidate_majority": predicted == "C",
                "gate_score": panel["simulation_gate"]["appreciable_score"],
                "severe_OOD": max(direction_rms, gate_rms) > 10,
                "accuracy_eligible": False,
                "outcome": (
                    "candidate-majority direction concordance"
                    if predicted == "C"
                    else "candidate-majority direction discordance"
                ),
            }
            panels.append(panel)
            paths[(scope, filter_name)] = panel_vcf
    overlap = {
        scope: locus_overlap(
            paths[(scope, "standard_contract")],
            paths[(scope, "within_population_polymorphism")],
        )
        for scope in scope_sources
    }
    return panels, overlap


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    parser.add_argument("--source-vcf", help="pinned revision.recode.vcf.gz")
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
        else cache / SOURCE["key"]
    )
    verified_file = ensure_source(source, args.download_missing)
    mapping = validate_manifest()
    source_contract = audit_source_vcf(source, mapping)
    autosome_vcf = cache / "tinkerbird.anchored_autosomes.vcf"
    autosome_audit = subset_anchored_autosomes(source, autosome_vcf)
    thinned_vcf = cache / "tinkerbird.anchored_autosomes.one_per_scaffold.vcf"
    thinning_audit = thin_one_per_scaffold(autosome_vcf, thinned_vcf)

    direction_head = simulation_direction_head(
        Path(args.data_root).resolve(), max_depth=MAX_DEPTH
    )
    gate_head = simulation_gate_head(Path(args.data_root).resolve(), max_depth=MAX_DEPTH)
    panels, filter_overlap = run_panels(
        {
            "autosome_all_snps": (
                autosome_vcf,
                "linked SNP-level panel capped by fixed-seed reservoir",
            ),
            "autosome_one_per_scaffold": (
                thinned_vcf,
                "at most one source-ordered structurally eligible SNP per scaffold before "
                "panel genotype/polymorphism filtering",
            ),
        },
        cache,
        args.cap,
        direction_head,
        gate_head,
    )
    result = {
        "schema_version": "dnnaic-tinkerbird-external-benchmark-v2",
        "git": git_revision(),
        "runtime": runtime_audit(),
        "guardrail": (
            "Circular, selection-enriched asymmetric-backcrossing OOD failure panel; never an "
            "external accuracy estimate. Candidate C is a heterogeneous majority label, not truth."
        ),
        "matched_control": None,
        "source": {
            **SOURCE,
            "verified_file": verified_file,
            "source_vcf_contract": source_contract,
            "anchored_autosome_extraction": autosome_audit,
            "one_per_scaffold_extraction": thinning_audit,
        },
        "population_design": {
            "P1": {
                "label": POPULATION_ORDER[0],
                "n": len(EXTONI_REFERENCE),
                "samples": list(EXTONI_REFERENCE),
            },
            "P2": {
                "label": POPULATION_ORDER[1],
                "n": len(ADMIXED14),
                "samples": list(ADMIXED14),
            },
            "P3": {
                "label": POPULATION_ORDER[2],
                "n": len(PUSILLUS_REFERENCE),
                "samples": list(PUSILLUS_REFERENCE),
            },
            "reference_basis": (
                "legacy source-era taxon/locality reference pools from the 2019 supplement; "
                "AR93101/AR93102 appear as AR93101b/AR93102b in the released VCF"
            ),
            "later_metadata_warning": (
                "Sebastianelli et al. 2024 metadata classify AR93110 and AR93178 as sympatric, "
                "so these legacy pools must not be relabelled as exact current allopatric sets"
            ),
            "panmictic_tree_leaf_contract_satisfied": False,
        },
        "published_sample_audit": published_sample_audit(),
        "direction_head": direction_head[2],
        "gate_head": gate_head[2],
        "filter_overlap": filter_overlap,
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
                        "direction": panel["simulation_head"]["predicted_class"],
                        "direction_scores": panel["simulation_head"]["scores"],
                        "gate": panel["simulation_gate"]["appreciable_score"],
                        "direction_rms_z": panel["simulation_feature_shift"]["rms_z"],
                        "gate_rms_z": panel["simulation_gate_feature_shift"]["rms_z"],
                        "projection": panel["model_free_comparator"][
                            "P2_projection_from_P1_toward_P3_all_loci"
                        ],
                        "f3": panel["model_free_comparator"]["f3_P2_P1_P3"],
                        "diagnostic_loci": panel["model_free_comparator"][
                            "diagnostic_loci"
                        ],
                        "loci": panel["padze"]["n_loci_kept"],
                    }
                    for panel in result["panels"]
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
