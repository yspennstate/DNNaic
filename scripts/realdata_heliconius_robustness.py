#!/usr/bin/env python3
"""Exploratory Heliconius diagnostics under explicit simulation-to-data shift.

Five pre-existing race panels are analyzed: four geographically motivated
cydno/timareta/melpomene trios and one allopatric mel_ros control.  They are not
independent biological validation experiments and the exact race panels do not
have externally established donor/recipient labels.  Current species-level
full-likelihood work estimates predominantly timareta -> melpomene (class B),
whereas melpomene -> timareta (class C) is documented at particular adaptive
wing-pattern loci.  Genome-wide calls are therefore reported only as an
exploratory discordance diagnostic.

Each panel is represented once by the paper's primary 54-D curve summary and
is scored by the canonical all-positive multinomial logistic head.  Softmax
outputs on these natural panels are uncalibrated scores, not probabilities.
Raw and diversity-normalized sharing ratios are generated directly with depth
cutoff and denominator-clipping sensitivity.  The source genotype/popmap files,
sample identities, filters, chromosomes, reservoir seeds, and checksums are
recorded in the result JSON.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import random
import shutil
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn

from dnnaic import build_matrix
from realdata_mouse_diversity import (
    CLASSES,
    DEFAULT_CUTOFFS,
    DEFAULT_RESULTS,
    REPO,
    canonical_direction_head,
    direction_features,
    package_version,
    ratio_diagnostics,
    sha256_file,
)


SOURCE_BASE = "https://github.com/simonhmartin/tutorials/raw/master/ABBA_BABA_whole_genome/data"
GENO_URL = f"{SOURCE_BASE}/hel92.DP8MP4BIMAC2HET75dist250.geno.gz"
POPMAP_URL = f"{SOURCE_BASE}/hel92.pop.txt"
DEFAULT_CACHE = REPO / "data" / "real" / "heliconius"
TARGET_SNP = 15_000
MIN_CALLED_COPIES = 16

PANELS = (
    {
        "id": "race_trio_1",
        "kind": "geographic_trio",
        "races": {"cydno": "cyd_chi", "timareta": "tim_txn", "melpomene": "mel_ama"},
        "reservoir_seed": 100,
    },
    {
        "id": "race_trio_2",
        "kind": "geographic_trio",
        "races": {"cydno": "cyd_zel", "timareta": "tim_flo", "melpomene": "mel_mel"},
        "reservoir_seed": 101,
    },
    {
        "id": "race_trio_3",
        "kind": "geographic_trio",
        "races": {"cydno": "cyd_chi", "timareta": "tim_flo", "melpomene": "mel_mal"},
        "reservoir_seed": 102,
    },
    {
        "id": "race_trio_4",
        "kind": "geographic_trio",
        "races": {"cydno": "cyd_zel", "timareta": "tim_txn", "melpomene": "mel_ama"},
        "reservoir_seed": 103,
    },
    {
        "id": "allopatric_control_1",
        "kind": "allopatric_control",
        "races": {"cydno": "cyd_zel", "timareta": "tim_flo", "melpomene": "mel_ros"},
        "reservoir_seed": 104,
    },
)
POPULATION_ORDER = ("cydno", "timareta", "melpomene")


def cache_source(source: Path | None, destination: Path, url: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        source = source.resolve()
        if source != destination.resolve():
            shutil.copy2(source, destination)
    elif not destination.exists():
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    if not destination.exists() or destination.stat().st_size == 0:
        raise FileNotFoundError(f"source cache unavailable: {destination}")
    return destination


def load_race_map(path: Path) -> dict[str, str]:
    mapping = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            mapping[fields[0]] = fields[1]
    return mapping


def parse_genotype(cell: str) -> tuple[str, str]:
    genotype = cell.split(":", 1)[0].replace("|", "/")
    alleles = genotype.split("/")
    if len(alleles) != 2:
        return ("N", "N")
    return alleles[0], alleles[1]


def chromosome_sort_key(chromosome: str) -> tuple[int, str]:
    suffix = chromosome.removeprefix("chr")
    return (int(suffix), "") if suffix.isdigit() else (10_000, suffix)


def make_panel_records(
    geno_path: Path, race_map: dict[str, str], panel: dict, target: int
) -> tuple[list[str], list[tuple[str, int, str, str, list[tuple[str, str]]]], dict]:
    """Reservoir-sample one panel after applying its declared filters."""
    with gzip.open(geno_path, "rt", encoding="latin-1") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        column = {name: index for index, name in enumerate(header)}
        sample_to_population = {
            sample: population
            for population, race in panel["races"].items()
            for sample, observed_race in race_map.items()
            if observed_race == race and sample in column
        }
        selected_names = [sample for sample in header if sample in sample_to_population]
        by_population = {
            population: [sample for sample in selected_names if sample_to_population[sample] == population]
            for population in POPULATION_ORDER
        }
        if any(not samples for samples in by_population.values()):
            raise RuntimeError(f"{panel['id']}: missing selected samples: {by_population}")

        rng = random.Random(panel["reservoir_seed"])
        reservoir: list[tuple[str, int, str, str, list[tuple[str, str]]]] = []
        source_records = 0
        eligible = 0
        for line in handle:
            source_records += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                continue
            genotypes = {sample: parse_genotype(fields[column[sample]]) for sample in selected_names}
            base_counts: Counter[str] = Counter(
                allele
                for genotype in genotypes.values()
                for allele in genotype
                if allele in {"A", "C", "G", "T"}
            )
            if len(base_counts) != 2:
                continue
            ref, alt = sorted(base_counts, key=lambda base: (-base_counts[base], base))
            called_by_population = {
                population: sum(
                    allele in (ref, alt)
                    for sample in samples
                    for allele in genotypes[sample]
                )
                for population, samples in by_population.items()
            }
            if any(value < MIN_CALLED_COPIES for value in called_by_population.values()):
                continue
            eligible += 1
            record = (
                fields[0], int(fields[1]), ref, alt,
                [genotypes[sample] for sample in selected_names],
            )
            if len(reservoir) < target:
                reservoir.append(record)
            else:
                replace = rng.randint(0, eligible - 1)
                if replace < target:
                    reservoir[replace] = record
    reservoir.sort(key=lambda row: (chromosome_sort_key(row[0]), row[1]))
    sample_position = {sample: index for index, sample in enumerate(selected_names)}
    called_copy_audit = {}
    for population, population_samples in by_population.items():
        positions = [sample_position[sample] for sample in population_samples]
        values = [
            sum(
                allele in {"A", "C", "G", "T"}
                for position in positions
                for allele in record[4][position]
            )
            for record in reservoir
        ]
        called_copy_audit[population] = {
            "minimum": int(min(values)),
            "maximum": int(max(values)),
            "mean": float(np.mean(values)),
        }
    audit = {
        "source_records_scanned": source_records,
        "eligible_after_filters": eligible,
        "reservoir_target": target,
        "retained_records": len(reservoir),
        "reservoir_seed": panel["reservoir_seed"],
        "sample_ids_by_population": by_population,
        "individual_counts": {population: len(samples) for population, samples in by_population.items()},
        "gene_copies_if_complete": {population: 2 * len(samples) for population, samples in by_population.items()},
        "called_gene_copies_per_retained_locus": called_copy_audit,
        "chromosome_counts": dict(Counter(record[0] for record in reservoir)),
    }
    return selected_names, reservoir, audit


def write_panel_vcf(
    panel_id: str, selected_names: list[str], records, race_map: dict[str, str], panel: dict,
    cache_dir: Path,
) -> tuple[Path, Path]:
    vcf_path = cache_dir / f"{panel_id}.vcf"
    popmap_path = cache_dir / f"{panel_id}.popmap.tsv"
    race_to_population = {race: population for population, race in panel["races"].items()}
    with vcf_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(selected_names) + "\n")
        for chromosome, position, ref, alt, genotypes in records:
            cells = [
                "/".join("0" if allele == ref else ("1" if allele == alt else ".") for allele in genotype)
                for genotype in genotypes
            ]
            handle.write(
                f"{chromosome}\t{position}\t.\t{ref}\t{alt}\t.\tPASS\t.\tGT\t"
                + "\t".join(cells) + "\n"
            )
    with popmap_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in selected_names:
            handle.write(f"{sample}\t{race_to_population[race_map[sample]]}\n")
    return vcf_path, popmap_path


def score_vcf(vcf_path: Path, popmap_path: Path, scaler, model) -> tuple[dict, np.ndarray, list[str]]:
    X, columns, loci = build_matrix(
        str(vcf_path), str(popmap_path), max_depth=100, pop_order=list(POPULATION_ORDER)
    )
    X = np.asarray(X, float)
    feature = direction_features(X[None, :, :])
    score = model.predict_proba(scaler.transform(feature))[0]
    ratios = ratio_diagnostics(X, columns, cutoffs=DEFAULT_CUTOFFS)
    primary = ratios["by_minimum_depth"].get("8")
    result = {
        "padze_retained_loci": int(loci.metadata.n_loci_kept),
        "depths": X[:, 0].astype(int).tolist(),
        "direction_call": str(CLASSES[int(np.argmax(score))]),
        "uncalibrated_softmax_scores": {
            str(label): float(value) for label, value in zip(CLASSES, score)
        },
        "primary_raw_ratio_g_ge_8": primary["raw_ratio_of_depth_means"] if primary else None,
        "primary_normalized_ratio_g_ge_8_clip_1e-12": (
            primary["normalized_ratio_of_depth_means_by_alpha_clip"]["1e-12"] if primary else None
        ),
        "sharing_ratio_sensitivity": ratios,
    }
    return result, X, columns


def leave_one_chromosome_out(
    selected_names, records, race_map, panel, cache_dir, scaler, model
) -> list[dict]:
    chromosomes = sorted({record[0] for record in records}, key=chromosome_sort_key)
    output = []
    for chromosome in chromosomes:
        subset = [record for record in records if record[0] != chromosome]
        if not subset:
            continue
        vcf_path, popmap_path = write_panel_vcf(
            "race_trio_1_leave_one_chromosome_out", selected_names, subset,
            race_map, panel, cache_dir,
        )
        score, _, _ = score_vcf(vcf_path, popmap_path, scaler, model)
        output.append({
            "dropped_chromosome": chromosome,
            "input_loci": len(subset),
            "padze_retained_loci": score["padze_retained_loci"],
            "direction_call": score["direction_call"],
            "uncalibrated_softmax_scores": score["uncalibrated_softmax_scores"],
        })
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geno", type=Path, default=None, help="existing source .geno.gz to cache")
    parser.add_argument("--popmap", type=Path, default=None, help="existing source race popmap to cache")
    parser.add_argument(
        "--data-root", default=os.environ.get("DNNAIC_DATA", str(REPO / "data" / "simulation_data"))
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--target-snps", type=int, default=TARGET_SNP)
    parser.add_argument("--skip-jackknife", action="store_true")
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    geno_path = cache_source(args.geno, cache_dir / "hel.geno.gz", GENO_URL)
    popmap_path = cache_source(args.popmap, cache_dir / "hel.pop.txt", POPMAP_URL)
    race_map = load_race_map(popmap_path)
    scaler, model, head_audit = canonical_direction_head(Path(args.data_root).resolve())

    panel_results = []
    canonical_records = canonical_names = canonical_panel = None
    for panel in PANELS:
        selected_names, records, audit = make_panel_records(
            geno_path, race_map, panel, args.target_snps
        )
        vcf_path, focal_popmap = write_panel_vcf(
            panel["id"], selected_names, records, race_map, panel, cache_dir
        )
        score, _, _ = score_vcf(vcf_path, focal_popmap, scaler, model)
        if score["padze_retained_loci"] != len(records):
            raise AssertionError(
                f"{panel['id']}: PADZE retained {score['padze_retained_loci']} of {len(records)} records"
            )
        panel_results.append({
            "panel_id": panel["id"],
            "kind": panel["kind"],
            "races": panel["races"],
            "filter_and_sample_audit": audit,
            "vcf_cache": str(vcf_path.relative_to(REPO)),
            "vcf_sha256": sha256_file(vcf_path),
            "popmap_cache": str(focal_popmap.relative_to(REPO)),
            **score,
        })
        print(
            f"[{panel['id']}] loci={len(records)} call={score['direction_call']} "
            f"raw={score['primary_raw_ratio_g_ge_8']:.3f} "
            f"norm={score['primary_normalized_ratio_g_ge_8_clip_1e-12']:.3f}",
            flush=True,
        )
        if panel["id"] == "race_trio_1":
            canonical_records, canonical_names, canonical_panel = records, selected_names, panel

    jackknife = []
    if not args.skip_jackknife and canonical_records is not None:
        jackknife = leave_one_chromosome_out(
            canonical_names, canonical_records, race_map, canonical_panel,
            cache_dir, scaler, model,
        )
        print(
            f"[jackknife] {len(jackknife)} chromosomes; calls="
            f"{dict(Counter(row['direction_call'] for row in jackknife))}",
            flush=True,
        )

    result = {
        "schema_version": "natural-diagnostics-heliconius-v2",
        "status": "exploratory_discordance_not_validation",
        "interpretation": {
            "class_mapping": {
                "B": "timareta -> melpomene",
                "C": "melpomene -> timareta",
            },
            "external_context": (
                "Species-level full-likelihood analysis estimates predominantly class B genome-wide; "
                "class C is documented at particular adaptive loci. Exact race-panel directions are not ground truth."
            ),
            "independence": (
                "Panels reuse races/individuals and linked loci; panel and chromosome checks are not independent experiments."
            ),
        },
        "source": {
            "genotype_url": GENO_URL,
            "popmap_url": POPMAP_URL,
            "genotype_cache": str(geno_path.relative_to(REPO)),
            "genotype_bytes": geno_path.stat().st_size,
            "genotype_sha256": sha256_file(geno_path),
            "popmap_cache": str(popmap_path.relative_to(REPO)),
            "popmap_bytes": popmap_path.stat().st_size,
            "popmap_sha256": sha256_file(popmap_path),
            "format": "Simon Martin ABBA_BABA_whole_genome tutorial .geno matrix and race map",
        },
        "filters": [
            "exactly two observed A/C/G/T alleles among selected focal samples",
            f"at least {MIN_CALLED_COPIES} called gene copies in each focal population",
            f"uniform reservoir sample of at most {args.target_snps} eligible SNPs per panel",
            "source allele bases converted once to stable 0/1 coding within each panel",
        ],
        "direction_head": head_audit,
        "panels": panel_results,
        "canonical_race_trio_1_leave_one_chromosome_out": jackknife,
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "padze": package_version("padze"),
        },
        "analysis_script": {
            "path": str(Path(__file__).resolve().relative_to(REPO)),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    result_path = result_dir / "heliconius.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (cache_dir / "heliconius_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (cache_dir / "robustness_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
