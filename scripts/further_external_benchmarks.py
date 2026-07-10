#!/usr/bin/env python3
"""Run two further natural-data transfer diagnostics.

The scrub-jay panel is an exact near-zero Dsuite specificity benchmark.  The
Lake Malawi panel has significant excess sharing between mbuna and the basal
pelagic clade, but the source evidence does not orient donor and recipient.
Accordingly, its gate has a positive expectation while its A/B/C score is
descriptive only.

Both source VCFs are immutable public artifacts.  This runner verifies their
hashes, applies the same depth/callability/locus-cap contract as the other
external bundles, and refits both released linear heads on the identical
g=2..16 simulation grid.  Every model output is an uncalibrated OOD score, not
a probability or a biological posterior.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from additional_external_benchmarks import add_gate_score, simulation_gate_head
from external_benchmarks import (
    MANIFEST_DIR,
    MAX_DEPTH,
    MIN_CALLED_COPIES,
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


DEFAULT_CACHE = REPO / "data" / "real" / "further_external_benchmarks"
DEFAULT_RESULTS = REPO / "results" / "further_external_benchmarks_2026_07_11"
DEFAULT_CAP = 15_000

SCRUB_JAY = {
    "bytes": 11_132_546,
    "sha256": "04e297ecfe3b5509c9419f0e14f1f7cba16ee493caebf5f7f46a5dcf8faa431a",
    "download": (
        "https://raw.githubusercontent.com/DevonDeRaad/aph.rad/"
        "7de8067fee4d3bdc0fce65e55f0e17b8c53f078d/unzipped.filtered.vcf.gz"
    ),
    "repository_commit": "7de8067fee4d3bdc0fce65e55f0e17b8c53f078d",
    "data_doi": "10.5061/dryad.8sf7m0cph",
    "paper_doi": "10.1093/sysbio/syac034",
    "license": "CC0-1.0 (Dryad data record; the GitHub repository has no separate license)",
    "author_popmap_sha256": "c6deb1c92867fc512cb9747d420d27c9cdcff38298414bffde972949f847e007",
    "author_dsuite_sha256": "395bf4de612ccea1582a34fcecd28dcd82f72b87c30d2e4d9543a1e43eb079c8",
    "author_dsuite_row": {
        "P1": "iw",
        "P2": "mw",
        "P3": "s",
        "D": 0.00619049,
        "Z": 0.205384,
        "p": 0.418636,
        "f4_ratio": 0.00466407,
        "BBAA": 163.283,
        "ABBA": 93.5538,
        "BABA": 92.4027,
    },
}

LAKE_MALAWI = {
    "bytes": 49_824_023,
    "sha256": "8132246ce809f4f4efa77d174c595a469aa7534c2cb8ee9fbf5472f67202c2b7",
    "md5": "fed3ca4c8d984417ee02de535f98e644",
    "download": (
        "https://zenodo.org/api/records/4134522/files/"
        "Malinsky_et_al_2018_LakeMalawiCichlids_scaffold_0.vcf.gz/content"
    ),
    "record": "https://zenodo.org/records/4134522",
    "data_doi": "10.5281/zenodo.4134522",
    "paper_doi": "10.1038/s41559-018-0717-x",
    "license": "CC0-1.0",
    "author_popmap_sha256": "07a0b6e345150b64ffc6e8383572635b0a40663b25b4a78e5145466922accf58",
    "outgroup_sample": "Nbrichardi",
}


def _allele_frequency(fields: list[str], indices: list[int]) -> tuple[float, list[str]]:
    alleles = [allele for index in indices for allele in called_alleles(fields[index])]
    if not alleles:
        return float("nan"), alleles
    return float(sum(allele == "1" for allele in alleles) / len(alleles)), alleles


def d_site_components(p1: float, p2: float, p3: float, outgroup: float) -> tuple[float, float]:
    """Expected ABBA/BABA contributions for biallelic population frequencies."""
    abba = (1 - p1) * p2 * p3 * (1 - outgroup)
    abba += p1 * (1 - p2) * (1 - p3) * outgroup
    baba = p1 * (1 - p2) * p3 * (1 - outgroup)
    baba += (1 - p1) * p2 * (1 - p3) * outgroup
    return float(abba), float(baba)


def four_population_d(
    vcf: Path,
    manifest: Path,
    pop_order: tuple[str, str, str],
    outgroup_sample: str,
    block_size: int = 1_000_000,
    filter_manifest: Path | None = None,
    polymorphic_panel_manifests: tuple[Path, ...] = (),
    polymorphic_within_each_population: bool = False,
) -> dict:
    """Compute a transparent frequency-based D and delete-one-block jackknife."""
    mapping = read_manifest(manifest)
    filter_mapping = (
        read_manifest(filter_manifest, require_three=False)
        if filter_manifest is not None
        else mapping
    )
    if not set(mapping).issubset(filter_mapping):
        raise ValueError("D-scoring manifest must be a subset of the filter manifest")
    panel_maps = (
        [read_manifest(path) for path in polymorphic_panel_manifests]
        if polymorphic_panel_manifests
        else [filter_mapping]
    )
    if any(not set(panel).issubset(filter_mapping) for panel in panel_maps):
        raise ValueError("polymorphism panel must be a subset of the filter manifest")
    population_columns: dict[str, list[int]] | None = None
    filter_population_columns: dict[str, list[int]] | None = None
    polymorphic_columns: list[list[int]] | None = None
    outgroup_column: int | None = None
    totals = np.zeros(2, dtype=float)
    blocks: dict[tuple[str, int], np.ndarray] = defaultdict(lambda: np.zeros(2, dtype=float))
    counters: Counter[str] = Counter()

    with open_text(vcf) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\r\n").split("\t")
                samples = header[9:]
                missing = sorted(set(filter_mapping) - set(samples))
                if missing:
                    raise ValueError(f"{vcf}: filter-manifest samples absent from VCF: {missing}")
                if outgroup_sample not in samples:
                    raise ValueError(f"{vcf}: outgroup sample absent: {outgroup_sample}")
                sample_column = {sample: 9 + index for index, sample in enumerate(samples)}
                population_columns = {
                    population: [sample_column[sample] for sample, label in mapping.items() if label == population]
                    for population in pop_order
                }
                filter_population_columns = {
                    population: [
                        sample_column[sample]
                        for sample, label in filter_mapping.items()
                        if label == population
                    ]
                    for population in sorted(set(filter_mapping.values()))
                }
                if polymorphic_within_each_population:
                    polymorphic_columns = [
                        [sample_column[sample] for sample, label in panel.items() if label == population]
                        for panel in panel_maps
                        for population in sorted(set(panel.values()))
                    ]
                else:
                    polymorphic_columns = [
                        [sample_column[sample] for sample in panel]
                        for panel in panel_maps
                    ]
                outgroup_column = sample_column[outgroup_sample]
                continue
            if line.startswith("#") or not line.strip():
                continue
            counters["source_rows"] += 1
            if (
                population_columns is None
                or filter_population_columns is None
                or polymorphic_columns is None
                or outgroup_column is None
            ):
                raise ValueError(f"{vcf}: data row before #CHROM header")
            fields = line.rstrip("\r\n").split("\t")
            if (
                len(fields) <= outgroup_column
                or len(fields[3]) != 1
                or len(fields[4]) != 1
                or "," in fields[4]
                or fields[6] not in ("PASS", ".")
            ):
                counters["variant_or_filter_rejected"] += 1
                continue

            if any(
                sum(len(called_alleles(fields[index])) for index in indices)
                < MIN_CALLED_COPIES
                for indices in filter_population_columns.values()
            ):
                counters["insufficient_called_copies"] += 1
                continue
            if any(
                len(
                    {
                        allele
                        for index in indices
                        for allele in called_alleles(fields[index])
                    }
                )
                < 2
                for indices in polymorphic_columns
            ):
                counters["not_polymorphic_in_filter_contract"] += 1
                continue
            frequencies = [
                _allele_frequency(fields, population_columns[population])[0]
                for population in pop_order
            ]
            outgroup_alleles = called_alleles(fields[outgroup_column])
            if not outgroup_alleles:
                counters["outgroup_missing"] += 1
                continue
            outgroup_frequency = float(
                sum(allele == "1" for allele in outgroup_alleles) / len(outgroup_alleles)
            )
            abba, baba = d_site_components(*frequencies, outgroup_frequency)
            contribution = np.array([abba, baba], dtype=float)
            totals += contribution
            key = (fields[0], (int(fields[1]) - 1) // block_size)
            blocks[key] += contribution
            counters["eligible_sites"] += 1

    denominator = float(totals.sum())
    if denominator <= 0 or len(blocks) < 2:
        raise ValueError("insufficient ABBA/BABA weight or jackknife blocks")
    estimate = float((totals[0] - totals[1]) / denominator)
    delete_one = []
    for contribution in blocks.values():
        remainder = totals - contribution
        delete_one.append(float((remainder[0] - remainder[1]) / remainder.sum()))
    values = np.asarray(delete_one, dtype=float)
    n_blocks = len(values)
    se = float(np.sqrt((n_blocks - 1) / n_blocks * np.sum((values - values.mean()) ** 2)))
    return {
        "D": estimate,
        "SE": se,
        "Z": float(estimate / se),
        "ABBA": float(totals[0]),
        "BABA": float(totals[1]),
        "n_blocks": n_blocks,
        "block_size_bp": block_size,
        "counts": dict(counters),
        "formula": "frequency-weighted ABBA/BABA with the named outgroup; delete-one 1-Mb blocks",
        "filter_contract": {
            "filter_manifest": str(filter_manifest or manifest),
            "polymorphic_panel_manifests": [str(path) for path in polymorphic_panel_manifests],
            "polymorphic_within_each_population": polymorphic_within_each_population,
            "minimum_called_copies_per_filter_population": MIN_CALLED_COPIES,
        },
    }


def _score_filter_variants(
    *,
    dataset: str,
    source_vcf: Path,
    manifest: Path,
    cache: Path,
    cap: int,
    pop_order: tuple[str, str, str],
    expectation: dict,
    direction_head,
    gate_head,
) -> list[dict]:
    direction_scaler, direction_model, _ = direction_head
    gate_scaler, gate_model, _ = gate_head
    panels = []
    for suffix, strict in (
        ("standard_contract", False),
        ("within_population_polymorphism", True),
    ):
        derived_vcf = cache / f"{dataset}.{suffix}.filtered.vcf"
        derived_popmap = cache / f"{dataset}.{suffix}.filtered.popmap.tsv"
        audit = prepare_vcf(
            source_vcf,
            manifest,
            derived_vcf,
            derived_popmap,
            cap=cap,
            seed=20260711,
            polymorphic_within_each_population=strict,
        )
        panel_expectation = dict(expectation)
        panel_expectation["locus_filter_variant"] = (
            "both alleles observed within each population"
            if strict
            else "both alleles observed across the complete trio"
        )
        panel = score_panel(
            f"{dataset}_{suffix}",
            derived_vcf,
            derived_popmap,
            pop_order,
            audit,
            direction_scaler,
            direction_model,
            panel_expectation,
        )
        panels.append(add_gate_score(panel, gate_scaler, gate_model))
    return panels


def run_scrub_jay(vcf: Path, cache: Path, cap: int, direction_head, gate_head):
    source = verify_file(vcf, SCRUB_JAY["bytes"], SCRUB_JAY["sha256"])
    panels = _score_filter_variants(
        dataset="scrub_jay_exact_D_null",
        source_vcf=vcf,
        manifest=MANIFEST_DIR / "scrub_jay_null.tsv",
        cache=cache,
        cap=cap,
        pop_order=("iw", "mw", "s"),
        expectation={
            "expected_gate": "no appreciable event for this exact sampled trio",
            "direction_truth": None,
            "author_dsuite_row": SCRUB_JAY["author_dsuite_row"],
            "guardrail": (
                "exact near-zero D specificity benchmark, not proof of species-wide isolation; "
                "a narrow phenotypic contact zone is known near Mexico City"
            ),
        },
        direction_head=direction_head,
        gate_head=gate_head,
    )
    return panels, source


def run_lake_malawi(vcf: Path, cache: Path, cap: int, direction_head, gate_head):
    source = verify_file(vcf, LAKE_MALAWI["bytes"], LAKE_MALAWI["sha256"])
    positive_manifest = MANIFEST_DIR / "lake_malawi_mbuna_pelagic.tsv"
    negative_manifest = MANIFEST_DIR / "lake_malawi_deep_benthic.tsv"
    shared_manifest = MANIFEST_DIR / "lake_malawi_shared.tsv"
    direction_scaler, direction_model, _ = direction_head
    gate_scaler, gate_model, _ = gate_head
    panels = []
    classical_by_filter = {}
    for suffix, strict in (
        ("standard_shared_contract", False),
        ("within_population_polymorphism", True),
    ):
        classical_positive = four_population_d(
            vcf,
            positive_manifest,
            ("A_calliptera", "mbuna", "pelagic"),
            LAKE_MALAWI["outgroup_sample"],
            filter_manifest=shared_manifest,
            polymorphic_panel_manifests=(positive_manifest, negative_manifest),
            polymorphic_within_each_population=strict,
        )
        classical_negative = four_population_d(
            vcf,
            negative_manifest,
            ("A_calliptera", "mbuna", "deep_benthic"),
            LAKE_MALAWI["outgroup_sample"],
            filter_manifest=shared_manifest,
            polymorphic_panel_manifests=(positive_manifest, negative_manifest),
            polymorphic_within_each_population=strict,
        )
        classical_by_filter[suffix] = {
            "mbuna_pelagic_positive": classical_positive,
            "mbuna_deep_benthic_negative": classical_negative,
        }
        shared_vcf = cache / f"lake_malawi.{suffix}.shared.filtered.vcf"
        shared_popmap = cache / f"lake_malawi.{suffix}.shared.filtered.popmap.tsv"
        shared_audit = prepare_vcf(
            vcf,
            shared_manifest,
            shared_vcf,
            shared_popmap,
            cap=cap,
            seed=20260711,
            require_three_populations=False,
            polymorphic_panel_manifests=(positive_manifest, negative_manifest),
            polymorphic_within_each_population=strict,
        )
        panel_specs = (
            (
                "positive",
                positive_manifest,
                ("A_calliptera", "mbuna", "pelagic"),
                {
                    "expected_gate": "positive excess sharing between mbuna (P2) and pelagic taxa (P3)",
                    "direction_truth": None,
                    "classical_comparator": classical_positive,
                    "direction_guardrail": (
                        "published f-branch/excess-sharing evidence does not orient donor and recipient; "
                        "the A/B/C score is descriptive and must not count as directional accuracy"
                    ),
                    "topology_basis": (
                        "published topology places A. calliptera and mbuna as sisters relative "
                        "to basal pelagic taxa"
                    ),
                },
            ),
            (
                "negative",
                negative_manifest,
                ("A_calliptera", "mbuna", "deep_benthic"),
                {
                    "expected_gate": "no appreciable event for mbuna versus the deep-benthic clade",
                    "direction_truth": None,
                    "classical_comparator": classical_negative,
                    "source_result": (
                        "the source study reports no substantial genome-wide exchange between "
                        "deep-benthic taxa and mbuna"
                    ),
                    "guardrail": "matched specificity control, not proof of absolute historical isolation",
                },
            ),
        )
        for label, manifest, pop_order, expectation in panel_specs:
            panel_vcf = cache / f"lake_malawi.{suffix}.{label}.filtered.vcf"
            panel_popmap = cache / f"lake_malawi.{suffix}.{label}.filtered.popmap.tsv"
            audit = subset_prepared_vcf(
                shared_vcf,
                manifest,
                panel_vcf,
                panel_popmap,
                shared_audit,
            )
            audit["comparison_locus_contract"] = (
                "same ordered four-group callable-site intersection and cap as the matched "
                "Lake Malawi positive/negative panel"
            )
            expectation = dict(expectation)
            expectation["locus_filter_variant"] = (
                "both alleles observed within every population of both matched panels"
                if strict
                else "both matched three-population panels polymorphic on one shared callable-site intersection"
            )
            panel = score_panel(
                f"lake_malawi_{label}_{suffix}",
                panel_vcf,
                panel_popmap,
                pop_order,
                audit,
                direction_scaler,
                direction_model,
                expectation,
            )
            panels.append(add_gate_score(panel, gate_scaler, gate_model))
    return panels, source, classical_by_filter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    parser.add_argument("--scrub-vcf", help="pinned unzipped.filtered.vcf.gz")
    parser.add_argument("--malawi-vcf", help="Lake Malawi scaffold_0 VCF.gz")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    args = parser.parse_args()
    if not args.scrub_vcf and not args.malawi_vcf:
        parser.error("provide --scrub-vcf and/or --malawi-vcf")
    if args.cap < 1:
        parser.error("--cap must be positive")

    set_below_normal_priority()
    data_root = Path(args.data_root).resolve()
    cache = Path(args.cache_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    direction_head = simulation_direction_head(data_root, max_depth=MAX_DEPTH)
    gate_head = simulation_gate_head(data_root, max_depth=MAX_DEPTH)
    result = {
        "schema_version": "dnnaic-further-external-benchmarks-v1",
        "git": git_revision(),
        "guardrail": (
            "Scrub jay is an exact sampled-trio near-zero specificity benchmark. Lake Malawi "
            "has a positive sharing benchmark but no directional truth. All scores are uncalibrated OOD diagnostics."
        ),
        "direction_head": direction_head[2],
        "gate_head": gate_head[2],
        "sources": {},
        "classical_comparators": {},
        "panels": [],
    }
    if args.scrub_vcf:
        panels, source = run_scrub_jay(
            Path(args.scrub_vcf).resolve(), cache, args.cap, direction_head, gate_head
        )
        result["panels"].extend(panels)
        result["sources"]["scrub_jay"] = {**SCRUB_JAY, "verified_file": source}
        result["classical_comparators"]["scrub_jay"] = SCRUB_JAY["author_dsuite_row"]
    if args.malawi_vcf:
        panels, source, classical = run_lake_malawi(
            Path(args.malawi_vcf).resolve(), cache, args.cap, direction_head, gate_head
        )
        result["panels"].extend(panels)
        result["sources"]["lake_malawi"] = {**LAKE_MALAWI, "verified_file": source}
        result["classical_comparators"]["lake_malawi"] = classical

    output = result_dir / "results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "panels": [
                    {
                        "panel_id": panel["panel_id"],
                        "direction": panel["simulation_head"]["predicted_class"],
                        "gate": panel["simulation_gate"]["appreciable_score"],
                        "direction_rms_z": panel["simulation_feature_shift"]["rms_z"],
                        "gate_rms_z": panel["simulation_gate_feature_shift"]["rms_z"],
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
