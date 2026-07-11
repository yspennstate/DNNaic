#!/usr/bin/env python3
"""Run a guarded corkwing-wrasse candidate-direction sensitivity benchmark.

Faust et al. (2018) document transport of southern corkwing wrasse into
Flatanger, with escapees and hybrid descendants.  Three separate southern
reference populations therefore provide candidate-class-C donor sensitivities
for the same complete 40-fish Flatanger recipient population. Role-swapped
western contrasts are same-locus descriptive comparators, never matched
controls or no-flow populations.

The primary scope removes both 15 author HWE exclusions and the 200 loci used
to generate NewHybrids labels.  An all-released-locus scope is explicitly
circular sensitivity.  Every learned score is an uncalibrated OOD diagnostic;
no panel is an independent validation or accuracy observation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
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

from additional_external_benchmarks import (
    add_gate_score,
    simulation_gate_head,
)
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
import tinkerbird_external_benchmark as runtime_helpers


DEFAULT_CACHE = REPO / "data" / "real" / "wrasse_external_benchmark"
DEFAULT_RESULTS = REPO / "results" / "wrasse_external_benchmark_2026_07_11"
DEFAULT_CAP = 15_000
PANEL_CONFIG = MANIFEST_DIR / "wrasse" / "panel_populations.tsv"
SOURCES_RECORD = MANIFEST_DIR / "wrasse" / "sources.json"

DRYAD_RECORD = "https://datadryad.org/dataset/doi:10.5061/dryad.tv553"
DRYAD_ARCHIVE = "https://datadryad.org/api/v2/versions/21981/download"
ARCHIVE = {
    "key": "dryad_tv553_v1.zip",
    "bytes": 24_227_679,
    "sha256": "15474df5ba0808db77f403f56076a17b7c3821e31a62a9c996e0a81468bd1620",
    "md5": "e042b4c3cbe108d2f2f743dfa640c303",
}
FILES = {
    "vcf": {
        "key": "west.filt.maf0.01.recode.vcf",
        "bytes": 18_733_292,
        "sha256": "c05741f03ecdb2403f173cf249eb910281632f65bccad27aaaaaf848ffb2e21a",
    },
    "metadata": {
        "key": "Sampleinfo_metadata.txt",
        "bytes": 15_251,
        "sha256": "5d7d1fba38095e4231d94c6b34b65d5ea6f392d742d3d4b660453e7b5cd769ad",
    },
    "hwe_genepop": {
        "key": "west_genepop4357ID.txt",
        "bytes": 5_282_897,
        "sha256": "6c5e923a7c97ce4fd08d87372ca19363beefe6eb337068ed0037c9858c4b038a",
    },
    "newhybrid": {
        "key": "newhybrid200SNPs.dat",
        "bytes": 192_097,
        "sha256": "77c6ba3bc26a88dcee518b971bd4b96c4235911b32daa27a224c9e75d81929e5",
    },
}
SOURCE_CONTRACT = {
    "samples": 240,
    "populations": 6,
    "samples_per_population": 40,
    "variant_rows": 4_372,
    "HWE_retained_loci": 4_357,
    "HWE_excluded_loci": 15,
    "NewHybrids_label_loci": 200,
    "primary_source_loci": 4_157,
}
SITE_CODES = {
    "Flatanger": "FKH",
    "Austevoll": "SMAU",
    "Stavanger": "SMID",
    "Kristiansand": "SMTF",
    "Strömstad": "SMST",
    "Kungsbacka": "SMKB",
}
EXPECTED_PANELS = {
    "candidate_SMTF": (
        "candidate_direction_sensitivity",
        "SMAU",
        "FKH",
        "SMTF",
        "C",
    ),
    "candidate_SMST": (
        "candidate_direction_sensitivity",
        "SMAU",
        "FKH",
        "SMST",
        "C",
    ),
    "candidate_SMKB": (
        "candidate_direction_sensitivity",
        "SMAU",
        "FKH",
        "SMKB",
        "C",
    ),
    "comparator_SMTF": (
        "role_swapped_comparator",
        "SMID",
        "SMAU",
        "SMTF",
        None,
    ),
    "comparator_SMST": (
        "role_swapped_comparator",
        "SMID",
        "SMAU",
        "SMST",
        None,
    ),
    "comparator_SMKB": (
        "role_swapped_comparator",
        "SMID",
        "SMAU",
        "SMKB",
        None,
    ),
}
SCOPE_ROLES = {
    "primary_HWE_and_label_loci_excluded": (
        "primary: excludes 15 author HWE loci plus all 200 NewHybrids label-generating loci"
    ),
    "all_released_loci_sensitivity": (
        "sensitivity: includes same-data NewHybrids label loci and remains circular"
    ),
}


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DNNaic-audit/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(output)


def ensure_sources(
    paths: dict[str, Path], archive_path: Path, download_missing: bool
) -> dict:
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        if not download_missing:
            raise FileNotFoundError(f"missing wrasse source files: {missing}")
        if not archive_path.exists():
            _download(DRYAD_ARCHIVE, archive_path)
        verify_file(archive_path, ARCHIVE["bytes"], ARCHIVE["sha256"])
        with zipfile.ZipFile(archive_path) as archive:
            members = set(archive.namelist())
            for name in missing:
                key = FILES[name]["key"]
                if key not in members:
                    raise AssertionError(f"Dryad archive lacks {key}")
                paths[name].parent.mkdir(parents=True, exist_ok=True)
                with archive.open(key) as source, paths[name].open("wb") as target:
                    while chunk := source.read(1024 * 1024):
                        target.write(chunk)
    verified = {
        name: verify_file(path, FILES[name]["bytes"], FILES[name]["sha256"])
        for name, path in paths.items()
    }
    archive_audit = None
    if archive_path.exists():
        archive_audit = verify_file(
            archive_path, ARCHIVE["bytes"], ARCHIVE["sha256"]
        )
    return {"files": verified, "archive": archive_audit}


def read_panel_specs(path: Path = PANEL_CONFIG) -> dict[str, dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    specs = {}
    for row in rows:
        panel = row["panel"]
        if panel in specs:
            raise ValueError(f"duplicate wrasse panel {panel}")
        candidate = None if row["candidate_class"] == "." else row["candidate_class"]
        specs[panel] = {
            "benchmark_role": row["benchmark_role"],
            "P1": row["P1"],
            "P2": row["P2"],
            "P3": row["P3"],
            "candidate_class": candidate,
        }
    observed = {
        name: (
            spec["benchmark_role"],
            spec["P1"],
            spec["P2"],
            spec["P3"],
            spec["candidate_class"],
        )
        for name, spec in specs.items()
    }
    if observed != EXPECTED_PANELS:
        raise AssertionError(f"wrasse panel contract changed: {observed}")
    return specs


def read_vcf_samples(path: Path) -> list[str]:
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                if len(samples) != len(set(samples)):
                    raise AssertionError("wrasse VCF sample IDs are not unique")
                return samples
    raise ValueError(f"{path}: no #CHROM header")


def metadata_sample_contract(metadata: Path) -> tuple[list[str], dict[str, str], int]:
    with metadata.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != SOURCE_CONTRACT["samples"]:
        raise AssertionError("unexpected wrasse metadata row count")
    expected_samples = []
    populations = {}
    technical_replicates = 0
    for row in rows:
        site = row["Sample_ID"]
        if site not in SITE_CODES:
            raise AssertionError(f"unexpected wrasse site {site}")
        population = SITE_CODES[site]
        base = row["Individual_ID"]
        if not re.fullmatch(rf"{population}\d{{2}}", base):
            raise AssertionError(f"metadata ID/site mismatch: {base} {site}")
        technical = row["Technical_Replicates"].lower()
        if technical not in {"yes", "no"}:
            raise AssertionError("unexpected technical-replicate flag")
        sample = base + ("a" if technical == "yes" else "")
        technical_replicates += technical == "yes"
        if sample in populations:
            raise AssertionError(f"duplicate metadata-derived sample {sample}")
        expected_samples.append(sample)
        populations[sample] = population
    return expected_samples, populations, technical_replicates


def materialize_manifests(
    vcf: Path,
    metadata: Path,
    output_dir: Path,
    specs: dict[str, dict],
) -> tuple[Path, dict[str, Path], dict]:
    source_samples = read_vcf_samples(vcf)
    expected_samples, populations, technical_replicates = metadata_sample_contract(
        metadata
    )
    if set(source_samples) != set(expected_samples):
        raise AssertionError("VCF IDs do not equal metadata-derived exact sample IDs")
    counts = {
        population: list(populations.values()).count(population)
        for population in SITE_CODES.values()
    }
    if set(counts.values()) != {SOURCE_CONTRACT["samples_per_population"]}:
        raise AssertionError(f"unexpected population counts: {counts}")
    if technical_replicates != 48:
        raise AssertionError("unexpected technical-replicate sample count")

    output_dir.mkdir(parents=True, exist_ok=True)
    union = output_dir / "wrasse.union.tsv"
    with union.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("sample\tpopulation\n")
        for sample in source_samples:
            handle.write(f"{sample}\t{populations[sample]}\n")
    panel_paths = {}
    for panel, spec in specs.items():
        selected_populations = (spec["P1"], spec["P2"], spec["P3"])
        path = output_dir / f"wrasse.{panel}.tsv"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("sample\tpopulation\n")
            for sample in source_samples:
                population = populations[sample]
                if population in selected_populations:
                    handle.write(f"{sample}\t{population}\n")
        mapping = read_manifest(path)
        if len(mapping) != 120 or {
            label: list(mapping.values()).count(label) for label in set(mapping.values())
        } != {label: 40 for label in selected_populations}:
            raise AssertionError(f"unexpected materialized panel {panel}")
        panel_paths[panel] = path
    return union, panel_paths, {
        "metadata_rows": len(expected_samples),
        "VCF_samples": len(source_samples),
        "exact_VCF_metadata_ID_set_match": True,
        "technical_replicate_suffix_samples": technical_replicates,
        "population_counts": counts,
        "union_manifest": {"path": str(union), "sha256": sha256_file(union)},
        "panel_manifests": {
            panel: {"path": str(path), "sha256": sha256_file(path)}
            for panel, path in panel_paths.items()
        },
    }


def audit_source_vcf(path: Path, union_manifest: Path) -> dict:
    mapping = read_manifest(union_manifest, require_three=False)
    samples = None
    population_columns = None
    rows = 0
    chromosomes = []
    positions = []
    formats = set()
    genotype_cells = 0
    missing_cells = 0
    invalid_cells = 0
    filters = set()
    ids = set()
    AF_colon_rows = 0
    NS_max = 0
    minimum_called = {population: 10**9 for population in set(mapping.values())}
    maximum_called = {population: 0 for population in set(mapping.values())}
    with open_text(path) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                if samples != list(mapping):
                    raise AssertionError("source sample order differs from union manifest")
                population_columns = {
                    population: [
                        9 + index
                        for index, sample in enumerate(samples)
                        if mapping[sample] == population
                    ]
                    for population in set(mapping.values())
                }
                continue
            if line.startswith("#") or not line.strip():
                continue
            if samples is None or population_columns is None:
                raise ValueError("wrasse source variant before #CHROM")
            fields = line.rstrip("\r\n").split("\t")
            rows += 1
            chromosomes.append(fields[0])
            positions.append(int(fields[1]))
            ids.add(fields[2])
            filters.add(fields[6])
            formats.add(fields[8])
            info = dict(
                item.split("=", 1) for item in fields[7].split(";") if "=" in item
            )
            AF_colon_rows += ":" in info.get("AF", "")
            NS_max = max(NS_max, int(info["NS"]))
            if len(fields) != 9 + len(samples):
                raise AssertionError("wrasse source row width changed")
            if (
                len(fields[3]) != 1
                or len(fields[4]) != 1
                or "," in fields[4]
                or fields[3] not in "ACGT"
                or fields[4] not in "ACGT"
            ):
                raise AssertionError("wrasse source is not biallelic SNP-only")
            for cell in fields[9:]:
                genotype_cells += 1
                alleles = cell.split(":", 1)[0].replace("|", "/").split("/")
                if alleles == [".", "."]:
                    missing_cells += 1
                elif len(alleles) != 2 or any(a not in {"0", "1"} for a in alleles):
                    invalid_cells += 1
            for population, columns in population_columns.items():
                copies = sum(len(called_alleles(fields[index])) for index in columns)
                minimum_called[population] = min(minimum_called[population], copies)
                maximum_called[population] = max(maximum_called[population], copies)
    if samples is None or len(samples) != SOURCE_CONTRACT["samples"]:
        raise AssertionError("unexpected wrasse sample count")
    if rows != SOURCE_CONTRACT["variant_rows"]:
        raise AssertionError("unexpected wrasse source variant count")
    if len(chromosomes) != len(set(chromosomes)):
        raise AssertionError("wrasse CHROM tags are not unique one-row loci")
    if ids != {"."} or filters != {"."} or formats != {"GT:GQ:PS:AD:DP"}:
        raise AssertionError("unexpected wrasse VCF structural fields")
    if invalid_cells or genotype_cells != rows * len(samples):
        raise AssertionError("wrasse GT cells are not diploid biallelic/full missing")
    if min(minimum_called.values()) < 28:
        raise AssertionError("wrasse source no longer satisfies observed called-copy floor")
    if AF_colon_rows != rows or NS_max <= len(samples):
        raise AssertionError("expected source metadata quirks were not observed")
    return {
        "samples": len(samples),
        "variant_rows": rows,
        "unique_2bRAD_tag_CHROM_values": len(set(chromosomes)),
        "position_range_within_tag": [min(positions), max(positions)],
        "FORMAT": sorted(formats),
        "genotype_cells": genotype_cells,
        "fully_missing_genotype_cells": missing_cells,
        "partial_or_invalid_genotype_cells": invalid_cells,
        "called_copy_range_by_population": {
            population: {
                "minimum": minimum_called[population],
                "maximum": maximum_called[population],
            }
            for population in sorted(minimum_called)
        },
        "metadata_quirks": {
            "AF_colon_delimited_rows": AF_colon_rows,
            "maximum_source_NS": NS_max,
            "source_samples": len(samples),
            "guardrail": (
                "source INFO/FORMAT declarations are nonstandard; downstream VCFs are "
                "sanitized to GT-only and do not consume INFO, AD, PS, DP, or GQ"
            ),
        },
        "linkage_guardrail": (
            "each CHROM is a unique 2bRAD tag and POS is within-tag position; no physical "
            "chromosomes or linkage map are available"
        ),
    }


def parse_exclusion_contract(
    source_vcf: Path, hwe_genepop: Path, newhybrid: Path, metadata: Path
) -> tuple[set[str], dict]:
    source_loci = []
    with open_text(source_vcf) as handle:
        for line in handle:
            if not line.startswith("#") and line.strip():
                source_loci.append(line.split("\t", 1)[0])
    genepop_lines = hwe_genepop.read_text(encoding="utf-8-sig").splitlines()
    try:
        first_pop = next(
            index for index, line in enumerate(genepop_lines) if line.strip().lower() == "pop"
        )
    except StopIteration as error:
        raise AssertionError("Genepop source has no POP delimiter") from error
    retained = [line.strip() for line in genepop_lines[1:first_pop] if line.strip()]
    if len(retained) != SOURCE_CONTRACT["HWE_retained_loci"] or len(set(retained)) != len(
        retained
    ):
        raise AssertionError("unexpected Genepop HWE-retained locus contract")
    hwe_excluded = set(source_loci) - set(retained)
    if len(hwe_excluded) != SOURCE_CONTRACT["HWE_excluded_loci"]:
        raise AssertionError("unexpected HWE exclusion count")
    if retained != [locus for locus in source_loci if locus not in hwe_excluded]:
        raise AssertionError("Genepop HWE loci do not preserve VCF source order")

    newhybrid_lines = [
        line.strip()
        for line in newhybrid.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    expected_headers = [
        ["NumIndivs", "240"],
        ["NumLoci", "200"],
        ["Digits", "2"],
        ["Format", "Lumped"],
    ]
    if [line.split() for line in newhybrid_lines[:4]] != expected_headers:
        raise AssertionError("unexpected NewHybrids header contract")
    newhybrid_names = None
    locus_line_index = None
    for index, line in enumerate(newhybrid_lines):
        if line.startswith("LocusNames "):
            newhybrid_names = line.split()[1:]
            locus_line_index = index
            break
    if newhybrid_names is None:
        raise AssertionError("NewHybrids source has no LocusNames row")
    if (
        len(newhybrid_names) != SOURCE_CONTRACT["NewHybrids_label_loci"]
        or len(set(newhybrid_names)) != len(newhybrid_names)
        or not set(newhybrid_names).issubset(retained)
        or hwe_excluded & set(newhybrid_names)
    ):
        raise AssertionError("unexpected NewHybrids label-locus contract")
    label_digest = hashlib.sha256(
        "".join(f"{locus}\n" for locus in newhybrid_names).encode("utf-8")
    ).hexdigest()
    if label_digest != "117ae0bd1683164e1237114826c7d29af99d35cfd4a9f2913e14206bc2a93099":
        raise AssertionError("NewHybrids ordered label-locus digest changed")

    metadata_samples, metadata_populations, _ = metadata_sample_contract(metadata)
    source_samples = None
    sample_columns = None
    expected_codes: dict[str, dict[str, str]] = {}
    with open_text(source_vcf) as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                source_samples = line.rstrip("\r\n").split("\t")[9:]
                sample_columns = {
                    sample: 9 + index for index, sample in enumerate(source_samples)
                }
                continue
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if fields[0] not in set(newhybrid_names):
                continue
            if sample_columns is None:
                raise ValueError("wrasse source variant before #CHROM")
            per_sample = {}
            for sample, column in sample_columns.items():
                alleles = fields[column].split(":", 1)[0].replace("|", "/").split("/")
                if alleles == [".", "."]:
                    code = "0"
                elif len(alleles) == 2 and all(allele in {"0", "1"} for allele in alleles):
                    code = {("0", "0"): "101", ("0", "1"): "102", ("1", "1"): "202"}[
                        tuple(sorted(alleles))
                    ]
                else:
                    raise AssertionError("invalid source GT at NewHybrids locus")
                per_sample[sample] = code
            expected_codes[fields[0]] = per_sample
    if source_samples is None or set(expected_codes) != set(newhybrid_names):
        raise AssertionError("could not reconstruct all NewHybrids source genotypes")
    body = newhybrid_lines[locus_line_index + 1 :]
    if len(body) != SOURCE_CONTRACT["samples"]:
        raise AssertionError("unexpected NewHybrids body row count")
    allowed_codes = {"0", "101", "102", "202"}
    z_counts = {"none": 0, "z0s": 0, "z1s": 0}
    genotype_cells = 0
    for expected_index, (sample, line) in enumerate(
        zip(metadata_samples, body), start=1
    ):
        fields = line.split()
        if fields[0] != str(expected_index):
            raise AssertionError("NewHybrids individual indices are not 1..240")
        z_token = fields[1] if fields[1] in {"z0s", "z1s"} else None
        codes = fields[2:] if z_token else fields[1:]
        z_counts[z_token or "none"] += 1
        population = metadata_populations[sample]
        expected_z = None if population == "FKH" else (
            "z0s" if population in {"SMAU", "SMID"} else "z1s"
        )
        if z_token != expected_z:
            raise AssertionError("NewHybrids z token does not match metadata population")
        if len(codes) != len(newhybrid_names) or not set(codes).issubset(allowed_codes):
            raise AssertionError("unexpected NewHybrids genotype row")
        expected = [expected_codes[locus][sample] for locus in newhybrid_names]
        if codes != expected:
            raise AssertionError(f"NewHybrids/VCF genotype mismatch for {sample}")
        genotype_cells += len(codes)
    if z_counts != {"none": 40, "z0s": 80, "z1s": 120}:
        raise AssertionError("unexpected NewHybrids z-token counts")
    excluded = hwe_excluded | set(newhybrid_names)
    return excluded, {
        "source_loci": len(source_loci),
        "HWE_retained_loci": len(retained),
        "HWE_excluded_loci": len(hwe_excluded),
        "HWE_excluded_in_VCF_order": [
            locus for locus in source_loci if locus in hwe_excluded
        ],
        "NewHybrids_label_loci": len(newhybrid_names),
        "NewHybrids_ordered_locus_sha256": label_digest,
        "NewHybrids_body_rows": len(body),
        "NewHybrids_genotype_cells_exactly_match_VCF": genotype_cells,
        "NewHybrids_z_token_counts": z_counts,
        "NewHybrids_sample_order": "metadata order (not VCF column order)",
        "exclusion_sets_disjoint": True,
        "primary_source_loci": len(source_loci) - len(excluded),
        "guardrail": (
            "the 200 loci used to generate same-data NewHybrids labels are excluded from the "
            "primary source before any benchmark filtering"
        ),
    }


def filter_primary_source(source: Path, output: Path, excluded: set[str]) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    retained = 0
    with open_text(source) as incoming, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as outgoing:
        for line in incoming:
            if line.startswith("#"):
                outgoing.write(line.rstrip("\r\n") + "\n")
                continue
            if not line.strip():
                continue
            source_rows += 1
            if line.split("\t", 1)[0] not in excluded:
                outgoing.write(line.rstrip("\r\n") + "\n")
                retained += 1
    if (
        source_rows != SOURCE_CONTRACT["variant_rows"]
        or retained != SOURCE_CONTRACT["primary_source_loci"]
    ):
        raise AssertionError("primary wrasse locus exclusion violated its contract")
    return {
        "source_variant_rows": source_rows,
        "excluded_loci": len(excluded),
        "retained_variant_rows": retained,
        "derived_vcf": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
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
    projections = []
    diagnostic_projections = []
    f3_estimates = []
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
        "method": "naive IID resampling of released 2bRAD loci",
        "seed": seed,
        "requested_replicates": replicates,
        "loci": n,
        "projection_all_loci": _summary(projections),
        "projection_diagnostic_loci": _summary(diagnostic_projections),
        "f3_finite_called_copy_corrected": _summary(f3_estimates),
        "guardrail": (
            "fixed-sample conditional sensitivity only; no physical chromosomes or linkage map "
            "exist, so independence is unverified and this is not chromosome-block uncertainty"
        ),
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
                raise ValueError("wrasse panel variant before #CHROM")
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
        raise ValueError("wrasse panel has no usable loci")
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
            "bounded ancestry or temporal direction. The f3-like correction assumes independent "
            "binomial called-copy sampling and is not generally unbiased."
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
    severe = max(
        panel["simulation_feature_shift"]["rms_z"],
        panel["simulation_gate_feature_shift"]["rms_z"],
    ) > 10
    panel["adjudication"] = {
        "candidate_class": candidate,
        "predicted_class": prediction,
        "matches_candidate_reference": prediction == candidate if candidate else None,
        "gate_score": panel["simulation_gate"]["appreciable_score"],
        "accuracy_eligible": False,
        "severe_OOD": severe,
        "natural_data_call_status": "abstain_severe_OOD" if severe else "diagnostic_only",
    }
    return panel


def run_all_panels(
    sources: dict[str, Path],
    specs: dict[str, dict],
    union_manifest: Path,
    panel_manifests: dict[str, Path],
    cache: Path,
    cap: int,
    direction_head,
    gate_head,
) -> tuple[list[dict], dict]:
    if set(sources) != set(SCOPE_ROLES):
        raise ValueError("wrasse sources do not match declared scopes")
    panels = []
    shared_audits = {}
    for scope, source in sources.items():
        for filter_name, strict in (
            ("standard_contract", False),
            ("within_population_polymorphism", True),
        ):
            shared_vcf = cache / f"wrasse.{scope}.{filter_name}.shared.vcf"
            shared_popmap = cache / f"wrasse.{scope}.{filter_name}.shared.popmap.tsv"
            shared_audit = prepare_vcf(
                source,
                union_manifest,
                shared_vcf,
                shared_popmap,
                cap=cap,
                seed=20260711,
                require_three_populations=False,
                polymorphic_panel_manifests=tuple(panel_manifests.values()),
                polymorphic_within_each_population=strict,
            )
            shared_audit["global_comparison_contract"] = (
                "one exact ordered locus set shared by all six candidate/comparator panels"
            )
            shared_audits[f"{scope}__{filter_name}"] = shared_audit
            hashes = set()
            for panel_name, spec in specs.items():
                manifest = panel_manifests[panel_name]
                vcf = cache / f"wrasse.{scope}.{filter_name}.{panel_name}.vcf"
                popmap = cache / f"wrasse.{scope}.{filter_name}.{panel_name}.popmap.tsv"
                audit = subset_prepared_vcf(
                    shared_vcf, manifest, vcf, popmap, shared_audit
                )
                audit["comparison_locus_contract"] = (
                    "same exact globally filtered ordered loci as all wrasse panels in this "
                    "scope/filter"
                )
                hashes.add(audit["ordered_locus_sha256"])
                positive = spec["benchmark_role"] == "candidate_direction_sensitivity"
                expectation = {
                    "benchmark_role": spec["benchmark_role"],
                    "candidate_class": spec["candidate_class"],
                    "candidate_forward_direction": (
                        f"{spec['P3']} southern reference (P3) -> FKH Flatanger (P2)"
                        if positive
                        else None
                    ),
                    "expected_gate": None,
                    "gate_comparison_status": (
                        "candidate panel; no calibrated natural-data gate expectation"
                        if positive
                        else "exploratory role-swapped comparator; no null threshold or ordering"
                    ),
                    "evidence_and_caveats": (
                        "documented anthropogenic southern transport plus author hybrid assignments; "
                        "older background flow and post-admixture selection complicate present frequencies"
                        if positive
                        else "role-swapped western comparator; Stavanger contains reported hybrids"
                    ),
                    "selection": (
                        "all 40 Flatanger fish; no outcome-based individual selection"
                        if positive
                        else "all 40 Austevoll fish as P2 and all 40 Stavanger fish as P1"
                    ),
                    "label_source_reuse": (
                        "primary removes the 200 NewHybrids label loci; study-level label and "
                        "population selection still reuse this source dataset"
                    ),
                    "tree_contract_status": (
                        "operational western-sister/southern-reference order; not a rooted demographic tree"
                    ),
                    "donor_reference": spec["P3"],
                    "scope": scope,
                    "scope_role": SCOPE_ROLES[scope],
                    "locus_filter_variant": (
                        "both alleles called within all six source populations; strong ascertainment"
                        if strict
                        else "both alleles called across every declared three-population panel"
                    ),
                    "accuracy_eligible": False,
                }
                panels.append(
                    _score(
                        f"wrasse_{panel_name}_{scope}_{filter_name}",
                        vcf,
                        popmap,
                        manifest,
                        (spec["P1"], spec["P2"], spec["P3"]),
                        audit,
                        direction_head,
                        gate_head,
                        expectation,
                    )
                )
            if len(hashes) != 1:
                raise AssertionError("wrasse panels do not share exact ordered loci")
    return panels, shared_audits


def same_locus_role_changed_contrasts(panels: list[dict]) -> list[dict]:
    indexed = {}
    for panel in panels:
        expectation = panel["external_expectation"]
        key = (
            expectation["scope"],
            "within_population_polymorphism"
            if panel["panel_id"].endswith("within_population_polymorphism")
            else "standard_contract",
            expectation["donor_reference"],
            expectation["benchmark_role"],
        )
        indexed[key] = panel
    comparisons = []
    for scope in SCOPE_ROLES:
        for filter_name in (
            "standard_contract",
            "within_population_polymorphism",
        ):
            for donor in ("SMTF", "SMST", "SMKB"):
                positive = indexed[
                    (scope, filter_name, donor, "candidate_direction_sensitivity")
                ]
                control = indexed[(scope, filter_name, donor, "role_swapped_comparator")]
                if positive["input_audit"]["ordered_locus_sha256"] != control[
                    "input_audit"
                ]["ordered_locus_sha256"]:
                    raise AssertionError("paired wrasse panels do not share ordered loci")
                positive_gate = positive["simulation_gate"]["appreciable_score"]
                control_gate = control["simulation_gate"]["appreciable_score"]
                comparisons.append(
                    {
                        "scope": scope,
                        "filter": filter_name,
                        "donor_reference": donor,
                        "ordered_locus_sha256": positive["input_audit"][
                            "ordered_locus_sha256"
                        ],
                        "loci": positive["padze"]["n_loci_kept"],
                        "candidate_panel_raw_OOD_gate_score": positive_gate,
                        "role_swapped_comparator_raw_OOD_gate_score": control_gate,
                        "candidate_minus_comparator_raw_OOD_gate_score": (
                            positive_gate - control_gate
                        ),
                        "gate_probability_ceiling_tie": positive_gate == control_gate == 1.0,
                        "raw_OOD_head_direction_candidate_panel": positive[
                            "simulation_head"
                        ]["predicted_class"],
                        "candidate_direction": "C",
                        "raw_OOD_head_direction_role_swapped_comparator": control[
                            "simulation_head"
                        ]["predicted_class"],
                        "comparison_role": (
                            "same-locus descriptive role-changed contrast that confounds both P1 "
                            "and P2 identities; no null threshold, prespecified ordering, causal "
                            "matching, independence, or accuracy claim"
                        ),
                    }
                )
    return comparisons


def summarize_outcomes(panels: list[dict], paired: list[dict]) -> dict:
    positive = [
        panel
        for panel in panels
        if panel["external_expectation"]["benchmark_role"]
        == "candidate_direction_sensitivity"
    ]
    controls = [
        panel
        for panel in panels
        if panel["external_expectation"]["benchmark_role"] == "role_swapped_comparator"
    ]
    deltas = [
        comparison["candidate_minus_comparator_raw_OOD_gate_score"]
        for comparison in paired
    ]
    abstained = sum(
        panel["adjudication"]["natural_data_call_status"] == "abstain_severe_OOD"
        for panel in panels
    )
    return {
        "panels": len(panels),
        "severe_OOD_panels": sum(panel["adjudication"]["severe_OOD"] for panel in panels),
        "abstained_panels": abstained,
        "abstain_due_to_severe_OOD": abstained == len(panels),
        "accuracy_estimate": None,
        "independent_validation_panels": 0,
        "unique_biological_recipient_cohorts": 1,
        "unique_candidate_recipient": "FKH",
        "candidate_direction_sensitivities": {
            "n": len(positive),
            "abstained_panels": sum(
                panel["adjudication"]["natural_data_call_status"]
                == "abstain_severe_OOD"
                for panel in positive
            ),
            "raw_OOD_head_matches_candidate_C": sum(
                panel["adjudication"]["matches_candidate_reference"] for panel in positive
            ),
            "raw_OOD_head_prediction_counts": {
                label: sum(
                    panel["simulation_head"]["predicted_class"] == label
                    for panel in positive
                )
                for label in ("A", "B", "C")
            },
        },
        "role_swapped_comparators": {
            "n": len(controls),
            "abstained_panels": sum(
                panel["adjudication"]["natural_data_call_status"]
                == "abstain_severe_OOD"
                for panel in controls
            ),
            "raw_OOD_gate_threshold_crossings_at_0.5": sum(
                panel["simulation_gate"]["called_at_0.5"] for panel in controls
            ),
        },
        "same_locus_role_changed_contrasts": {
            "n": len(paired),
            "gate_probability_ceiling_ties": sum(
                comparison["gate_probability_ceiling_tie"] for comparison in paired
            ),
            "raw_gate_delta_range": [min(deltas), max(deltas)],
        },
        "interpretation": (
            "one mixed-history Flatanger recipient cohort repeated across donor references, "
            "scopes, filters, and exploratory comparators; counts are not independent accuracy trials"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--source-vcf")
    parser.add_argument("--metadata")
    parser.add_argument("--hwe-genepop")
    parser.add_argument("--newhybrid")
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
    supplied = {
        "vcf": args.source_vcf,
        "metadata": args.metadata,
        "hwe_genepop": args.hwe_genepop,
        "newhybrid": args.newhybrid,
    }
    paths = {
        name: Path(supplied[name]).resolve()
        if supplied[name]
        else cache / FILES[name]["key"]
        for name in FILES
    }
    archive_path = (
        Path(args.archive).resolve() if args.archive else cache / ARCHIVE["key"]
    )
    source_files = ensure_sources(paths, archive_path, args.download_missing)
    specs = read_panel_specs()
    manifest_dir = cache / "manifests"
    union_manifest, panel_manifests, metadata_audit = materialize_manifests(
        paths["vcf"], paths["metadata"], manifest_dir, specs
    )
    source_audit = audit_source_vcf(paths["vcf"], union_manifest)
    excluded, exclusion_audit = parse_exclusion_contract(
        paths["vcf"], paths["hwe_genepop"], paths["newhybrid"], paths["metadata"]
    )
    primary_source = cache / "wrasse.primary_HWE_and_label_loci_excluded.vcf"
    primary_audit = filter_primary_source(paths["vcf"], primary_source, excluded)

    direction_head = simulation_direction_head(
        Path(args.data_root).resolve(), max_depth=MAX_DEPTH
    )
    gate_head = simulation_gate_head(Path(args.data_root).resolve(), max_depth=MAX_DEPTH)
    panels, shared_audits = run_all_panels(
        {
            "primary_HWE_and_label_loci_excluded": primary_source,
            "all_released_loci_sensitivity": paths["vcf"],
        },
        specs,
        union_manifest,
        panel_manifests,
        cache,
        args.cap,
        direction_head,
        gate_head,
    )
    paired = same_locus_role_changed_contrasts(panels)
    result = {
        "schema_version": "dnnaic-wrasse-external-benchmark-v1",
        "git": git_revision(),
        "runtime": runtime_helpers.runtime_audit(),
        "guardrail": (
            "candidate-C donor-reference sensitivities and same-locus role-changed comparators; "
            "no panel is independent, gold truth, or accuracy-eligible"
        ),
        "source": {
            "record": DRYAD_RECORD,
            "data_doi": "10.5061/dryad.tv553",
            "license": "CC0-1.0",
            "papers": {
                "original": "10.1098/rsos.171752",
                "background_gene_flow": "10.1111/mec.15310",
                "larger_confirmation": "10.1111/eva.13220",
                "selection_update_2026": "10.1111/eva.70214",
            },
            "sources_record": {
                "path": str(SOURCES_RECORD),
                "sha256": sha256_file(SOURCES_RECORD),
            },
            "verified": source_files,
            "source_vcf_contract": source_audit,
            "metadata_and_manifest_audit": metadata_audit,
            "label_locus_exclusion_audit": exclusion_audit,
            "primary_source_extraction": primary_audit,
            "ascertainment_guardrails": (
                "2bRAD loci, pooled MAF>1% release, same-data population/label inference, "
                "unknown linkage, founder/range-edge structure, and post-admixture selection"
            ),
        },
        "published_evidence": {
            "recent_event": (
                "southern fish transported to Flatanger; FKH48a/FKH50a clear southern-genotype "
                "escapees, FKH67 an F1, and twelve potential western backcrosses reported"
            ),
            "sampling_guardrail": "all 40 FKH fish retained; no hybrid-only cherry-picking",
            "background_direction_caveat": (
                "2020 work inferred ongoing bidirectional contact with older/background asymmetry "
                "predominantly west to south, superimposed on the recent anthropogenic "
                "southern-to-Flatanger event"
            ),
            "larger_2021_confirmation": (
                "a survey of 1,766 fish reported six high-probability southern-origin Flatanger "
                "fish and 70 potential hybrids there; escapees/hybrids were concentrated at the "
                "northern edge and reached about 20 percent locally"
            ),
            "latest_caveat": (
                "2026 mesocosm evidence found strong selective winter mortality in hybrid "
                "offspring and suggested potential assortative mating"
            ),
            "other_contact_zone_assignments": (
                "the 2018 article reports seven/eight potential backcrosses in Stavanger "
                "inconsistently across sections, plus two in Kristiansand, reinforcing that "
                "comparators are not nulls"
            ),
            "mixed_history_classifier_caveat": (
                "raw class A means Austevoll P1 to Flatanger P2 and can reflect dominant western "
                "or range-expansion background in a cohort with superimposed recent southern "
                "introgression; a mutually exclusive one-edge head cannot isolate class C"
            ),
        },
        "panel_config": {
            "path": str(PANEL_CONFIG),
            "sha256": sha256_file(PANEL_CONFIG),
            "specs": specs,
        },
        "analysis_scope_roles": SCOPE_ROLES,
        "direction_head": direction_head[2],
        "gate_head": gate_head[2],
        "shared_locus_audits": shared_audits,
        "same_locus_role_changed_contrasts": paired,
        "outcome_summary": summarize_outcomes(panels, paired),
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
