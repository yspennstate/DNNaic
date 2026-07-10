#!/usr/bin/env python3
"""Run paired natural-data direction stress tests with biological labels.

The first dataset is the Ciona contact-zone system.  The source study supports
recent C. robusta -> C. intestinalis introgression in Southampton, concentrated
in a chromosome-5 hotspot.  Jersey is the fixed sister reference and Poole is
the matched site control; both are absent from the source HMM table and the
independent chromosome-ancestry positive-site list.  Both panels are evaluated
on one shared four-population callable-site intersection.

The VCF was purpose-ascertained and the hotspot is author-selected, so these
are mechanistic feature-transfer diagnostics rather than held-out classifier
validation.  All learned scores remain uncalibrated out-of-distribution scores.
"""
from __future__ import annotations

import argparse
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


DEFAULT_CACHE = REPO / "data" / "real" / "directional_external_benchmarks"
DEFAULT_RESULTS = REPO / "results" / "directional_external_benchmarks_2026_07_11"
DEFAULT_CAP = 15_000

CIONA = {
    "bytes": 109_974_779,
    "sha256": "e0a3586c11a65f5d0419b08d14827fd6cf61d2e2705dd74340137dee310a761c",
    "md5": "5523378c6493dd835ad639963f2ad0fa",
    "download": (
        "https://zenodo.org/api/records/5346932/files/"
        "Ciona_data3_introgression_mac2.vcf.gz/content"
    ),
    "record": "https://zenodo.org/records/5346932",
    "data_doi": "10.5281/zenodo.5346932",
    "paper_doi": "10.1111/mec.16189",
    "preprint_doi": "10.1101/2021.08.05.455260",
    "license": "CC-BY-4.0",
    "figure3_window": {
        "chromosome": "chromosome5",
        "start": 500_000,
        "end": 2_000_000,
        "basis": "chromosome-5 interval used for the source Figure 3 ancestry/tree analysis",
    },
    "southampton_exact_interval": {
        "chromosome": "chromosome5",
        "start": 661_065,
        "end": 1_174_846,
        "basis": "pooled Southampton HMM-positive interval reported in source Table S4",
    },
    "ascertainment_caveat": (
        "the source removed SNPs private to C. robusta as uninformative for the study's "
        "C. intestinalis structure/introgression analysis"
    ),
    "mapping_caveat": "reads were mapped to a C. robusta reference and mapping rates differed by species",
    "ddrad_contract": "51,141 released SNPs derive from 5,599 linked RAD loci",
    "expected_filter_contracts": {
        "genomewide_standard_shared_contract": {"eligible": 27_418},
        "genomewide_within_population_polymorphism": {
            "eligible": 475,
            "ordered_locus_sha256": "ddbbb659403421a6de61cb34793e07269aea275eb8a4f224fc603e51f878e2d8",
        },
        "figure3_window_standard_shared_contract": {
            "eligible": 544,
            "ordered_locus_sha256": "0c57d534a9d009b3ef035f2ea6918776cafd1d524910b1179ea5eb08b46b0c30",
            "one_per_rad_tag": 79,
        },
        "southampton_exact_interval_standard_shared_contract": {
            "eligible": 262,
            "ordered_locus_sha256": "147e4ba4eaa2ee22aff40f479f4aed90aa0e433654d744a0f26edd38727bddd4",
            "one_per_rad_tag": 40,
        },
    },
}


def subset_region(
    source: Path,
    output: Path,
    chromosome: str,
    start: int,
    end: int,
) -> dict:
    """Write one inclusive VCF interval while preserving the complete header."""
    if start < 1 or end < start:
        raise ValueError("invalid inclusive genomic interval")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_rows = 0
    retained_rows = 0
    saw_header = False
    with open_text(source) as incoming, output.open("w", encoding="utf-8", newline="\n") as outgoing:
        for line in incoming:
            if line.startswith("#"):
                outgoing.write(line.rstrip("\r\n") + "\n")
                saw_header = saw_header or line.startswith("#CHROM")
                continue
            if not line.strip():
                continue
            source_rows += 1
            fields = line.split("\t", 2)
            if len(fields) < 2:
                continue
            if fields[0] == chromosome and start <= int(fields[1]) <= end:
                outgoing.write(line.rstrip("\r\n") + "\n")
                retained_rows += 1
    if not saw_header or retained_rows == 0:
        raise ValueError(f"{source}: interval produced no usable VCF")
    return {
        "source": str(source),
        "source_rows": source_rows,
        "chromosome": chromosome,
        "start_inclusive": start,
        "end_inclusive": end,
        "retained_rows": retained_rows,
        "derived_vcf": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }


def thin_one_per_id_prefix(source: Path, output: Path, separator: str = ":") -> dict:
    """Keep the first source-ordered SNP for each VCF ID prefix."""
    output.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    source_rows = 0
    retained_rows = 0
    saw_header = False
    with open_text(source) as incoming, output.open("w", encoding="utf-8", newline="\n") as outgoing:
        for line in incoming:
            if line.startswith("#"):
                outgoing.write(line.rstrip("\r\n") + "\n")
                saw_header = saw_header or line.startswith("#CHROM")
                continue
            if not line.strip():
                continue
            source_rows += 1
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 5:
                continue
            identifier = fields[2]
            prefix = identifier.split(separator, 1)[0] if identifier != "." else f"{fields[0]}:{fields[1]}"
            if prefix in seen:
                continue
            seen.add(prefix)
            outgoing.write("\t".join(fields) + "\n")
            retained_rows += 1
    if not saw_header or retained_rows == 0:
        raise ValueError(f"{source}: ID-prefix thinning produced no usable VCF")
    return {
        "source": str(source),
        "source_rows": source_rows,
        "separator": separator,
        "selection": "first eligible source-ordered SNP per VCF ID prefix",
        "retained_rows": retained_rows,
        "unique_prefixes": len(seen),
        "derived_vcf": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }


def frequency_sharing_comparator(vcf: Path, manifest: Path) -> dict:
    """Reference-invariant four-population frequency contrasts."""
    mapping = read_manifest(manifest, require_three=False)
    required = ("Jer12", "Sth12", "Poo12", "CioAB")
    if set(mapping.values()) != set(required):
        raise ValueError("unexpected Ciona shared-manifest populations")
    columns: dict[str, list[int]] | None = None
    values = {population: [] for population in required}
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
                    population: [sample_column[sample] for sample, label in mapping.items() if label == population]
                    for population in required
                }
                continue
            if line.startswith("#") or not line.strip():
                continue
            if columns is None:
                raise ValueError(f"{vcf}: data before #CHROM header")
            fields = line.rstrip("\r\n").split("\t")
            for population in required:
                alleles = [
                    allele
                    for index in columns[population]
                    for allele in called_alleles(fields[index])
                ]
                if not alleles:
                    raise ValueError(f"{vcf}: missing comparator population at {fields[0]}:{fields[1]}")
                values[population].append(sum(allele == "1" for allele in alleles) / len(alleles))
    arrays = {population: np.asarray(values[population], dtype=float) for population in required}
    n_loci = len(arrays["Jer12"])
    if n_loci == 0 or any(len(array) != n_loci for array in arrays.values()):
        raise ValueError("empty or mismatched comparator arrays")
    ciona = arrays["CioAB"]
    return {
        "n_loci": n_loci,
        "unpolarized_reference_invariant_f4": float(
            np.mean((arrays["Jer12"] - arrays["Sth12"]) * (ciona - arrays["Poo12"]))
        ),
        "mean_squared_frequency_distance_to_CioAB": {
            population: float(np.mean((arrays[population] - ciona) ** 2))
            for population in ("Jer12", "Sth12", "Poo12")
        },
        "interpretation": (
            "negative f4 and a smaller Sth12-CioAB distance support localized excess "
            "Southampton sharing; no ancestral allele or defensible Patterson-D outgroup is available"
        ),
    }


def run_ciona(vcf: Path, cache: Path, cap: int, direction_head, gate_head):
    source = verify_file(vcf, CIONA["bytes"], CIONA["sha256"])
    positive_manifest = MANIFEST_DIR / "ciona_jersey_southampton_robusta.tsv"
    negative_manifest = MANIFEST_DIR / "ciona_jersey_poole_robusta.tsv"
    shared_manifest = MANIFEST_DIR / "ciona_shared.tsv"
    figure3_window = CIONA["figure3_window"]
    figure3_vcf = cache / "ciona.figure3_window.source.vcf"
    figure3_audit = subset_region(
        vcf,
        figure3_vcf,
        figure3_window["chromosome"],
        figure3_window["start"],
        figure3_window["end"],
    )
    exact_interval = CIONA["southampton_exact_interval"]
    exact_vcf = cache / "ciona.southampton_exact_interval.source.vcf"
    exact_audit = subset_region(
        vcf,
        exact_vcf,
        exact_interval["chromosome"],
        exact_interval["start"],
        exact_interval["end"],
    )

    direction_scaler, direction_model, _ = direction_head
    gate_scaler, gate_model, _ = gate_head
    scopes = (
        {
            "name": "genomewide_standard_shared_contract",
            "vcf": vcf,
            "strict": False,
            "scope_audit": {
                "kind": "purpose-ascertained whole released VCF",
                "source_sha256": CIONA["sha256"],
            },
        },
        {
            "name": "genomewide_within_population_polymorphism",
            "vcf": vcf,
            "strict": True,
            "scope_audit": {
                "kind": "whole released VCF with both alleles required within every population",
                "source_sha256": CIONA["sha256"],
            },
        },
        {
            "name": "figure3_window_standard_shared_contract",
            "vcf": figure3_vcf,
            "strict": False,
            "tag_sensitivity": True,
            "scope_audit": {
                "kind": "source Figure 3 chromosome-5 analysis window; target-selected mechanistic panel",
                **figure3_audit,
            },
        },
        {
            "name": "southampton_exact_interval_standard_shared_contract",
            "vcf": exact_vcf,
            "strict": False,
            "tag_sensitivity": True,
            "scope_audit": {
                "kind": "source Table S4 Southampton HMM-positive interval; target-selected mechanistic panel",
                **exact_audit,
            },
        },
    )
    panels = []
    scope_records = {}
    for scope in scopes:
        name = scope["name"]
        strict = scope["strict"]
        shared_vcf = cache / f"ciona.{name}.shared.filtered.vcf"
        shared_popmap = cache / f"ciona.{name}.shared.filtered.popmap.tsv"
        shared_audit = prepare_vcf(
            scope["vcf"],
            shared_manifest,
            shared_vcf,
            shared_popmap,
            cap=cap,
            seed=20260711,
            require_three_populations=False,
            polymorphic_panel_manifests=(positive_manifest, negative_manifest),
            polymorphic_within_each_population=strict,
        )
        shared_audit["source_scope"] = scope["scope_audit"]
        expected = CIONA["expected_filter_contracts"][name]
        if shared_audit["counts"]["eligible_before_cap"] != expected["eligible"]:
            raise AssertionError(
                f"{name}: expected {expected['eligible']} eligible loci, found "
                f"{shared_audit['counts']['eligible_before_cap']}"
            )
        if expected.get("ordered_locus_sha256") and shared_audit["ordered_locus_sha256"] != expected["ordered_locus_sha256"]:
            raise AssertionError(f"{name}: ordered locus hash differs from independently audited contract")
        variants = [(name, shared_vcf, shared_audit, scope["scope_audit"])]
        if scope.get("tag_sensitivity", False):
            thinned_source = cache / f"ciona.{name}.one_per_rad_tag.source.vcf"
            thinning_audit = thin_one_per_id_prefix(shared_vcf, thinned_source)
            if thinning_audit["retained_rows"] != expected["one_per_rad_tag"]:
                raise AssertionError(
                    f"{name}: expected {expected['one_per_rad_tag']} RAD tags, found "
                    f"{thinning_audit['retained_rows']}"
                )
            thinned_name = f"{name}_one_per_rad_tag"
            thinned_shared_vcf = cache / f"ciona.{thinned_name}.shared.filtered.vcf"
            thinned_shared_popmap = cache / f"ciona.{thinned_name}.shared.filtered.popmap.tsv"
            thinned_audit = prepare_vcf(
                thinned_source,
                shared_manifest,
                thinned_shared_vcf,
                thinned_shared_popmap,
                cap=cap,
                seed=20260711,
                require_three_populations=False,
                polymorphic_panel_manifests=(positive_manifest, negative_manifest),
            )
            thinned_scope = {
                **scope["scope_audit"],
                "linkage_sensitivity": thinning_audit,
            }
            thinned_audit["source_scope"] = thinned_scope
            variants.append((thinned_name, thinned_shared_vcf, thinned_audit, thinned_scope))

        specifications = (
            (
                "southampton_positive",
                positive_manifest,
                ("Jer12", "Sth12", "CioAB"),
                {
                    "expected_class": "C",
                    "expected_forward_direction": "CioAB C. robusta (P3) -> Sth12 C. intestinalis (P2)",
                    "direction_basis": (
                        "source HMM ancestry tracts and contact-zone analysis establish recent, "
                        "asymmetric C. robusta ancestry in Southampton C. intestinalis"
                    ),
                    "event_localization": {
                        "figure3_window": CIONA["figure3_window"],
                        "southampton_exact_interval": CIONA["southampton_exact_interval"],
                    },
                    "background_matching": (
                        "source Table S2 Fst: Jer-Sth=0.005, Jer-Poo=0.005, Sth-Poo=0.002; "
                        "main-text Sth/Poo comparisons are nonsignificant"
                    ),
                    "sampling_caveat": "Jersey was sampled in 2014; Southampton and Poole in 2012",
                    "guardrail": (
                        "purpose-ascertained VCF; the hotspot scope is additionally target-selected "
                        "and is a mechanistic stress test, not held-out validation"
                    ),
                },
            ),
            (
                "poole_negative",
                negative_manifest,
                ("Jer12", "Poo12", "CioAB"),
                {
                    "expected_gate": "no source-detected C. robusta tract in Poole",
                    "direction_truth": None,
                    "negative_basis": (
                        "Jersey and Poole are absent from both the source Table S4 HMM-positive "
                        "tracts and Figure S5 chromosome-ancestry positive-site list"
                    ),
                    "background_matching": (
                        "source Table S2 Fst: Jer-Sth=0.005, Jer-Poo=0.005, Sth-Poo=0.002; "
                        "main-text Sth/Poo comparisons are nonsignificant"
                    ),
                    "sampling_caveat": "Jersey was sampled in 2014; Southampton and Poole in 2012",
                    "guardrail": (
                        "site-level source-negative control, not proof of zero historical exchange"
                    ),
                },
            ),
        )
        for variant_name, variant_vcf, variant_audit, variant_scope in variants:
            comparator = frequency_sharing_comparator(variant_vcf, shared_manifest)
            scope_records[variant_name] = {
                "source_scope": variant_scope,
                "shared_filter": variant_audit,
                "model_free_comparator": comparator,
            }
            for label, manifest, pop_order, expectation in specifications:
                panel_vcf = cache / f"ciona.{variant_name}.{label}.filtered.vcf"
                panel_popmap = cache / f"ciona.{variant_name}.{label}.filtered.popmap.tsv"
                audit = subset_prepared_vcf(
                    variant_vcf,
                    manifest,
                    panel_vcf,
                    panel_popmap,
                    variant_audit,
                )
                audit["comparison_locus_contract"] = (
                    "same ordered four-population callable-site intersection and cap as the matched "
                    "Ciona positive/negative panel in this scope"
                )
                panel_expectation = dict(expectation)
                panel_expectation["model_free_scope_comparator"] = comparator
                panel = score_panel(
                    f"ciona_{label}_{variant_name}",
                    panel_vcf,
                    panel_popmap,
                    pop_order,
                    audit,
                    direction_scaler,
                    direction_model,
                    panel_expectation,
                )
                panels.append(add_gate_score(panel, gate_scaler, gate_model))
    return panels, source, scope_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    parser.add_argument("--ciona-vcf", required=True, help="Ciona_data3_introgression_mac2.vcf.gz")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    args = parser.parse_args()
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
    panels, source, scopes = run_ciona(
        Path(args.ciona_vcf).resolve(), cache, args.cap, direction_head, gate_head
    )
    result = {
        "schema_version": "dnnaic-directional-external-benchmarks-v1",
        "git": git_revision(),
        "guardrail": (
            "Ciona supplies a biologically labelled Southampton direction and a source-negative "
            "Poole site control on matched backgrounds. Its released VCF is purpose-ascertained "
            "and both hotspot scopes are target-selected. Scores are OOD diagnostics, not held-out validation."
        ),
        "direction_head": direction_head[2],
        "gate_head": gate_head[2],
        "sources": {"ciona": {**CIONA, "verified_file": source}},
        "scopes": scopes,
        "panels": panels,
    }
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
                    for panel in panels
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
