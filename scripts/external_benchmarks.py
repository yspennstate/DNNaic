#!/usr/bin/env python3
"""Run reproducible, explicitly OOD diagnostics on two external benchmarks.

This script adds one small targeted dataset (Andean-duck globin loci) and one
larger genome-wide dataset (scarlet runner bean).  Neither is treated as an
independent validation of the simulation-trained classifier.  The script
verifies source hashes, uses fixed author-derived population manifests,
filters to biallelic PASS/`.` SNPs with at least 16 called copies in every
population, requires polymorphism in every declared positive/null panel, and
deterministically caps the common eligible loci before PADZE extraction.

The 54-D simulation head is refit on the external panels' identical g=2..16
depth grid and is used only for uncalibrated OOD scores.
Published positive and negative/control expectations are recorded alongside
raw sharing ratios and feature-space z diagnostics so discordance is visible.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dnnaic import build_matrix
from realdata_mouse_diversity import (
    CLASSES,
    direction_features,
    package_version,
    ratio_diagnostics,
    sha256_file,
)


MANIFEST_DIR = REPO / "data" / "external_benchmarks"
DEFAULT_CACHE = REPO / "data" / "real" / "external_benchmarks"
DEFAULT_RESULTS = REPO / "results" / "external_benchmarks_2026_07_10"
DEFAULT_CAP = 15_000
MIN_CALLED_COPIES = 16
MAX_DEPTH = 16

DRYAD_VERSION = 249308
DRYAD_URL = f"https://datadryad.org/api/v2/versions/{DRYAD_VERSION}/download"
DUCK_FILES = {
    "beta": {
        "name": "ST_YBP_Hb_no_indels_tags.0.inds.missing.no.invariable.recode.vcf",
        "bytes": 1_816_093,
        "sha256": "f816bc2fb2c8b9dca971a3d5bd3d9262ca75c6e307e4a62d7aae61f3d03997b6",
    },
    "alpha": {
        "name": "ST_YBP_HBA_no_indels_tags.0.inds.missing.no.invariable.recode.vcf",
        "bytes": 385_713,
        "sha256": "23cf5420d4af20fa0892ba7b76815ad0ffb2a362de3715ebaf8c7b4506057922",
    },
}

RUNNER_URL = "https://osf.io/download/qjh75/"
RUNNER_NAME = "coccineus.recode.vcf"
RUNNER_BYTES = 454_599_072
RUNNER_SHA256 = "597ac8fb7fc0f74e5713a2faa34aae34f551283c02809c5ff78f450739e796ae"


def set_below_normal_priority() -> None:
    """Yield to the owner's interactive work on Windows."""
    if os.name != "nt":
        return
    try:
        import ctypes

        below_normal_priority_class = 0x00004000
        process = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(process, below_normal_priority_class)
    except Exception:
        pass


def open_text(path: Path):
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} bytes, found {size}")
    digest = sha256_file(path)
    if digest.lower() != expected_sha256.lower():
        raise ValueError(f"{path}: SHA-256 mismatch: {digest}")
    return {"path": str(path), "bytes": size, "sha256": digest}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DNNaic external benchmark/1"})
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
        shutil.copyfileobj(response, handle, length=1024 * 1024)
    temporary.replace(destination)


def ensure_duck_root(root: Path, download_missing: bool) -> Path:
    missing = [spec["name"] for spec in DUCK_FILES.values() if not (root / spec["name"]).exists()]
    if missing and not download_missing:
        raise FileNotFoundError(f"missing duck files under {root}: {missing}")
    if missing:
        root.mkdir(parents=True, exist_ok=True)
        archive = root.parent / f"dryad_bnzs7h4b4_v{DRYAD_VERSION}.zip"
        if not archive.exists():
            download(DRYAD_URL, archive)
        with zipfile.ZipFile(archive) as handle:
            resolved_root = root.resolve()
            for member in handle.infolist():
                target = (root / member.filename).resolve()
                if target != resolved_root and resolved_root not in target.parents:
                    raise ValueError(f"unsafe ZIP member: {member.filename}")
            handle.extractall(root)
    return root


def ensure_runner_vcf(path: Path, download_missing: bool) -> Path:
    if not path.exists() and download_missing:
        download(RUNNER_URL, path)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_manifest(path: Path, require_three: bool = True) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if fields == ["sample", "population"]:
            continue
        if len(fields) != 2:
            raise ValueError(f"{path}:{number}: expected sample and population")
        sample, population = fields
        if sample in mapping:
            raise ValueError(f"{path}:{number}: duplicate sample {sample}")
        mapping[sample] = population
    if require_three and len(set(mapping.values())) != 3:
        raise ValueError(f"{path}: expected exactly three populations")
    if len(set(mapping.values())) < 3:
        raise ValueError(f"{path}: expected at least three populations")
    return mapping


def called_alleles(cell: str) -> list[str]:
    genotype = cell.split(":", 1)[0].replace("|", "/")
    return [allele for allele in genotype.split("/") if allele in ("0", "1")]


def ordered_locus_sha256(rows: list[tuple[int, list[str]]]) -> str:
    digest = hashlib.sha256()
    for _, fields in rows:
        digest.update("\t".join((fields[0], fields[1], fields[3], fields[4])).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def simulation_direction_head(data_root: Path, max_depth: int = MAX_DEPTH):
    """Fit the simulation head on exactly the same depth grid as external panels."""
    directory = data_root / "regen_full"
    paths = {
        name: directory / name
        for name in ("X.npy", "direction.npy", "groups.npy")
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    X = np.load(paths["X.npy"], mmap_mode="r")
    direction = np.load(paths["direction.npy"], mmap_mode="r").astype("U2")
    groups = np.load(paths["groups.npy"], mmap_mode="r").astype("U80")
    unique, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    order = np.lexsort((np.asarray(X[:, 0]), inverse))
    per = np.bincount(inverse)
    if per.min() != 198 or per.max() != 198:
        raise AssertionError("regen_full curves must contain g=2..199")
    table = np.asarray(X[order]).reshape(len(unique), 198, X.shape[1])
    depth_grid = np.arange(2, max_depth + 1, dtype=float)
    use = table[0, :, 0] <= max_depth
    if not np.array_equal(table[0, use, 0], depth_grid):
        raise AssertionError(f"simulation depth grid does not contain g=2..{max_depth}")
    if not np.all(table[:, use, 0] == depth_grid[None, :]):
        raise AssertionError("simulation replicates do not share one depth grid")
    features = direction_features(table[:, use, :])
    labels = direction[first]
    positive = labels != "D"
    y = np.searchsorted(CLASSES, labels[positive])
    scaler = StandardScaler().fit(features[positive])
    model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs").fit(
        scaler.transform(features[positive]), y
    )
    arrays = {
        name: {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    audit = {
        "training_dataset": "regen_full",
        "training_selection": "all positive canonical replicates at every migration rate",
        "training_n": int(positive.sum()),
        "training_class_counts": {
            label: int((labels[positive] == label).sum()) for label in CLASSES
        },
        "depth_grid": depth_grid.astype(int).tolist(),
        "representation": (
            f"mean and population SD over g=2..{max_depth} for each of 27 non-depth coordinates; "
            "identical to the external-panel depth grid"
        ),
        "dimension": int(features.shape[1]),
        "model": "StandardScaler plus multinomial logistic regression (lbfgs, C=1)",
        "score_interpretation": "uncalibrated out-of-distribution score; not a probability or posterior",
        "fit_iterations": model.n_iter_.astype(int).tolist(),
        "training_array_audit": arrays,
    }
    return scaler, model, audit


def prepare_vcf(
    source: Path,
    manifest: Path,
    output_vcf: Path,
    output_popmap: Path,
    cap: int,
    seed: int,
    min_called_copies: int = MIN_CALLED_COPIES,
    require_three_populations: bool = True,
    polymorphic_panel_manifests: tuple[Path, ...] = (),
    polymorphic_within_each_population: bool = False,
) -> dict:
    """Filter a source VCF and retain a deterministic reservoir of eligible loci."""
    mapping = read_manifest(manifest, require_three=require_three_populations)
    panel_maps = []
    if polymorphic_panel_manifests:
        for panel_manifest in polymorphic_panel_manifests:
            panel_mapping = read_manifest(panel_manifest)
            absent = sorted(set(panel_mapping) - set(mapping))
            if absent:
                raise ValueError(
                    f"{panel_manifest}: samples absent from shared manifest: {absent}"
                )
            panel_maps.append((panel_manifest, panel_mapping))
    else:
        panel_maps.append((manifest, mapping))
    populations = sorted(set(mapping.values()))
    population_columns: dict[str, list[int]] = defaultdict(list)
    counters: Counter[str] = Counter()
    reservoir: list[tuple[int, list[str]]] = []
    rng = random.Random(seed)
    selected_samples: list[str] | None = None
    selected_source_columns: list[int] | None = None
    polymorphic_columns: list[list[int]] | None = None

    with open_text(source) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\r\n").split("\t")
                source_samples = header[9:]
                missing = sorted(set(mapping) - set(source_samples))
                if missing:
                    raise ValueError(f"{source}: manifest samples absent from VCF: {missing}")
                selected_samples = [sample for sample in source_samples if sample in mapping]
                selected_source_columns = [9 + source_samples.index(sample) for sample in selected_samples]
                for output_index, sample in enumerate(selected_samples):
                    population_columns[mapping[sample]].append(9 + output_index)
                output_column = {
                    sample: 9 + output_index
                    for output_index, sample in enumerate(selected_samples)
                }
                if polymorphic_within_each_population:
                    polymorphic_columns = [
                        [
                            output_column[sample]
                            for sample, population in panel_mapping.items()
                            if population == required_population
                        ]
                        for _, panel_mapping in panel_maps
                        for required_population in sorted(set(panel_mapping.values()))
                    ]
                else:
                    polymorphic_columns = [
                        [output_column[sample] for sample in panel_mapping]
                        for _, panel_mapping in panel_maps
                    ]
                continue
            if line.startswith("#") or not line.strip():
                continue
            if (
                selected_samples is None
                or selected_source_columns is None
                or polymorphic_columns is None
            ):
                raise ValueError(f"{source}: variant before #CHROM header")
            counters["source_variant_rows"] += 1
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 10:
                counters["malformed"] += 1
                continue
            if len(fields[3]) != 1 or len(fields[4]) != 1 or "," in fields[4]:
                counters["not_biallelic_snp"] += 1
                continue
            if fields[6] not in ("PASS", "."):
                counters["not_pass_or_dot"] += 1
                continue
            genotypes = [fields[index].split(":", 1)[0].replace("|", "/") for index in selected_source_columns]
            simplified = fields[:5] + [".", "PASS", ".", "GT"] + genotypes
            insufficient = False
            for population in populations:
                copies = sum(len(called_alleles(simplified[index])) for index in population_columns[population])
                if copies < min_called_copies:
                    insufficient = True
                    break
            if insufficient:
                counters["insufficient_called_copies"] += 1
                continue
            if any(
                len(
                    {
                        allele
                        for index in columns
                        for allele in called_alleles(simplified[index])
                    }
                )
                < 2
                for columns in polymorphic_columns
            ):
                counters["not_polymorphic_in_every_panel"] += 1
                continue

            counters["eligible_before_cap"] += 1
            record = (counters["source_variant_rows"], simplified)
            if len(reservoir) < cap:
                reservoir.append(record)
            else:
                replacement = rng.randrange(counters["eligible_before_cap"])
                if replacement < cap:
                    reservoir[replacement] = record

    if selected_samples is None:
        raise ValueError(f"{source}: no #CHROM header")
    reservoir.sort(key=lambda item: item[0])
    if not reservoir:
        raise ValueError(f"{source}: no loci passed filters")
    counters["retained_after_cap"] = len(reservoir)

    output_vcf.parent.mkdir(parents=True, exist_ok=True)
    with output_vcf.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(selected_samples) + "\n")
        for _, fields in reservoir:
            handle.write("\t".join(fields) + "\n")
    with output_popmap.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in selected_samples:
            handle.write(f"{sample}\t{mapping[sample]}\n")

    copy_counts = {}
    for population in populations:
        values = [
            sum(len(called_alleles(fields[index])) for index in population_columns[population])
            for _, fields in reservoir
        ]
        copy_counts[population] = {
            "individuals": len(population_columns[population]),
            "minimum": min(values),
            "maximum": max(values),
            "mean": float(np.mean(values)),
        }
    return {
        "source": str(source),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "filter_contract": {
            "variant": "single-base REF and single-base ALT; no multiallelic ALT",
            "filter": "PASS or dot (sources are author-filtered releases)",
            "minimum_called_copies_per_population": min_called_copies,
            "polymorphism": (
                "both REF and ALT must be called within every population of every declared panel"
                if polymorphic_within_each_population
                else "both REF and ALT must be called across every declared three-population panel"
            ),
            "polymorphism_scope": (
                "within each population"
                if polymorphic_within_each_population
                else "across the complete panel"
            ),
            "polymorphic_panel_manifests": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
                for path, _ in panel_maps
            ],
            "locus_cap": cap,
            "cap_method": "fixed-seed reservoir over eligible loci, restored to source order",
            "reservoir_seed": seed,
        },
        "counts": dict(counters),
        "population_called_copy_counts": copy_counts,
        "ordered_locus_sha256": ordered_locus_sha256(reservoir),
        "derived_vcf": {
            "path": str(output_vcf),
            "bytes": output_vcf.stat().st_size,
            "sha256": sha256_file(output_vcf),
        },
        "derived_popmap": {
            "path": str(output_popmap),
            "bytes": output_popmap.stat().st_size,
            "sha256": sha256_file(output_popmap),
        },
    }


def subset_prepared_vcf(
    source_vcf: Path,
    manifest: Path,
    output_vcf: Path,
    output_popmap: Path,
    shared_audit: dict,
) -> dict:
    """Subset a prepared multi-population VCF without changing its fixed loci."""
    mapping = read_manifest(manifest)
    selected_samples: list[str] | None = None
    selected_columns: list[int] | None = None
    rows: list[list[str]] = []
    with open_text(source_vcf) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                header = line.rstrip("\r\n").split("\t")
                source_samples = header[9:]
                missing = sorted(set(mapping) - set(source_samples))
                if missing:
                    raise ValueError(f"{source_vcf}: panel samples absent from prepared VCF: {missing}")
                selected_samples = [sample for sample in source_samples if sample in mapping]
                selected_columns = [9 + source_samples.index(sample) for sample in selected_samples]
                continue
            if line.startswith("#") or not line.strip():
                continue
            if selected_samples is None or selected_columns is None:
                raise ValueError(f"{source_vcf}: variant before #CHROM header")
            fields = line.rstrip("\r\n").split("\t")
            rows.append(fields[:9] + [fields[index] for index in selected_columns])
    if selected_samples is None or not rows:
        raise ValueError(f"{source_vcf}: no prepared panel rows")

    output_vcf.parent.mkdir(parents=True, exist_ok=True)
    with output_vcf.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(selected_samples) + "\n")
        for fields in rows:
            handle.write("\t".join(fields) + "\n")
    with output_popmap.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in selected_samples:
            handle.write(f"{sample}\t{mapping[sample]}\n")

    panel_locus_hash = ordered_locus_sha256(
        [(index, fields) for index, fields in enumerate(rows)]
    )
    if len(rows) != shared_audit["counts"]["retained_after_cap"]:
        raise AssertionError("panel subsetting changed the shared locus count")
    if panel_locus_hash != shared_audit["ordered_locus_sha256"]:
        raise AssertionError("panel subsetting changed ordered locus identities")

    populations = sorted(set(mapping.values()))
    population_columns = {
        population: [9 + index for index, sample in enumerate(selected_samples) if mapping[sample] == population]
        for population in populations
    }
    copy_counts = {}
    for population in populations:
        values = [
            sum(len(called_alleles(fields[index])) for index in population_columns[population])
            for fields in rows
        ]
        copy_counts[population] = {
            "individuals": len(population_columns[population]),
            "minimum": min(values),
            "maximum": max(values),
            "mean": float(np.mean(values)),
        }

    audit = copy.deepcopy(shared_audit)
    audit["manifest"] = str(manifest)
    audit["manifest_sha256"] = sha256_file(manifest)
    audit["population_called_copy_counts"] = copy_counts
    audit["ordered_locus_sha256"] = panel_locus_hash
    audit["comparison_locus_contract"] = (
        "same ordered locus intersection and cap as every runner-bean positive/null panel"
    )
    audit["derived_vcf"] = {
        "path": str(output_vcf),
        "bytes": output_vcf.stat().st_size,
        "sha256": sha256_file(output_vcf),
    }
    audit["derived_popmap"] = {
        "path": str(output_popmap),
        "bytes": output_popmap.stat().st_size,
        "sha256": sha256_file(output_popmap),
    }
    return audit


def score_panel(
    panel_id: str,
    vcf: Path,
    popmap: Path,
    pop_order: tuple[str, str, str],
    filter_audit: dict,
    scaler,
    model,
    expectation: dict,
) -> dict:
    X, columns, loci = build_matrix(
        str(vcf), str(popmap), max_depth=MAX_DEPTH, pop_order=list(pop_order)
    )
    X = np.asarray(X, dtype=float)
    if not np.isfinite(X).all():
        raise AssertionError(f"{panel_id}: non-finite PADZE feature")
    expected_loci = filter_audit["counts"]["retained_after_cap"]
    if loci.metadata.n_loci_kept != expected_loci:
        raise AssertionError(
            f"{panel_id}: PADZE retained {loci.metadata.n_loci_kept}; expected {expected_loci}"
        )
    feature = direction_features(X[None, :, :])
    z = scaler.transform(feature)[0]
    scores = model.predict_proba(scaler.transform(feature))[0]
    prediction = str(CLASSES[int(np.argmax(scores))])
    ratios = ratio_diagnostics(X, columns)
    return {
        "panel_id": panel_id,
        "population_order": {
            "P1": pop_order[0],
            "P2": pop_order[1],
            "P3": pop_order[2],
            "tree_contract": "((P1,P2),P3)",
        },
        "external_expectation": expectation,
        "padze": {
            "shape": list(X.shape),
            "columns": columns,
            "depths": X[:, 0].astype(int).tolist(),
            "n_loci_kept": int(loci.metadata.n_loci_kept),
            "all_finite": True,
        },
        "simulation_head": {
            "predicted_class": prediction,
            "scores": {label: float(value) for label, value in zip(CLASSES, scores)},
            "interpretation": "uncalibrated OOD scores; not probabilities or posteriors",
        },
        "simulation_feature_shift": {
            "reference": "StandardScaler fitted to 2,700 positive regen_full summaries on the identical g=2..16 grid",
            "rms_z": float(np.sqrt(np.mean(z**2))),
            "max_abs_z": float(np.max(np.abs(z))),
            "coordinates_abs_z_gt_3": int(np.sum(np.abs(z) > 3)),
            "coordinates_abs_z_gt_5": int(np.sum(np.abs(z) > 5)),
            "z_values": z.tolist(),
        },
        "sharing_ratio_diagnostics": ratios,
        "feature_matrix": X.tolist(),
        "input_audit": filter_audit,
    }


def git_revision() -> dict:
    completed = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    dirty = subprocess.run(
        ["git", "-C", str(REPO), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"commit": completed.stdout.strip(), "dirty_at_run": bool(dirty.stdout.strip())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        default=os.environ.get("DNNAIC_DATA", str(REPO / "data" / "simulation_data")),
        help="directory containing regen_full",
    )
    parser.add_argument("--duck-root", default=str(DEFAULT_CACHE / "sources" / "andean_ducks"))
    parser.add_argument("--runner-vcf", default=str(DEFAULT_CACHE / "sources" / RUNNER_NAME))
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE / "derived"))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()

    if args.cap < 1:
        parser.error("--cap must be positive")
    set_below_normal_priority()
    duck_root = ensure_duck_root(Path(args.duck_root).resolve(), args.download_missing)
    runner_vcf = ensure_runner_vcf(Path(args.runner_vcf).resolve(), args.download_missing)
    cache_dir = Path(args.cache_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    data_root = Path(args.data_root).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    source_audit = {"andean_ducks": {}, "runner_bean": {}}
    for key, spec in DUCK_FILES.items():
        source_audit["andean_ducks"][key] = verify_file(
            duck_root / spec["name"], spec["bytes"], spec["sha256"]
        )
    source_audit["runner_bean"] = verify_file(runner_vcf, RUNNER_BYTES, RUNNER_SHA256)

    duck_manifest = MANIFEST_DIR / "andean_ducks.popmap.tsv"
    bean_union_manifest = MANIFEST_DIR / "runner_bean_union.popmap.tsv"
    bean_cdmx_manifest = MANIFEST_DIR / "runner_bean_cdmx.popmap.tsv"
    bean_tepoz_manifest = MANIFEST_DIR / "runner_bean_tepoz.popmap.tsv"
    bean_null_manifest = MANIFEST_DIR / "runner_bean_null.popmap.tsv"
    beta_vcf = cache_dir / "andean_ducks_beta.filtered.vcf"
    beta_popmap = cache_dir / "andean_ducks_beta.popmap.tsv"
    alpha_vcf = cache_dir / "andean_ducks_alpha.filtered.vcf"
    alpha_popmap = cache_dir / "andean_ducks_alpha.popmap.tsv"
    bean_union_vcf = cache_dir / "runner_bean_union.filtered.vcf"
    bean_union_popmap = cache_dir / "runner_bean_union.popmap.tsv"
    bean_cdmx_vcf = cache_dir / "runner_bean_cdmx.filtered.vcf"
    bean_cdmx_popmap = cache_dir / "runner_bean_cdmx.popmap.tsv"
    bean_tepoz_vcf = cache_dir / "runner_bean_tepoz.filtered.vcf"
    bean_tepoz_popmap = cache_dir / "runner_bean_tepoz.popmap.tsv"
    bean_null_vcf = cache_dir / "runner_bean_null.filtered.vcf"
    bean_null_popmap = cache_dir / "runner_bean_null.popmap.tsv"

    beta_audit = prepare_vcf(
        duck_root / DUCK_FILES["beta"]["name"],
        duck_manifest,
        beta_vcf,
        beta_popmap,
        args.cap,
        2026071001,
    )
    alpha_audit = prepare_vcf(
        duck_root / DUCK_FILES["alpha"]["name"],
        duck_manifest,
        alpha_vcf,
        alpha_popmap,
        args.cap,
        2026071002,
    )
    bean_union_audit = prepare_vcf(
        runner_vcf,
        bean_union_manifest,
        bean_union_vcf,
        bean_union_popmap,
        args.cap,
        2026071003,
        require_three_populations=False,
        polymorphic_panel_manifests=(
            bean_cdmx_manifest,
            bean_tepoz_manifest,
            bean_null_manifest,
        ),
    )
    bean_cdmx_audit = subset_prepared_vcf(
        bean_union_vcf,
        bean_cdmx_manifest,
        bean_cdmx_vcf,
        bean_cdmx_popmap,
        bean_union_audit,
    )
    bean_tepoz_audit = subset_prepared_vcf(
        bean_union_vcf,
        bean_tepoz_manifest,
        bean_tepoz_vcf,
        bean_tepoz_popmap,
        bean_union_audit,
    )
    bean_null_audit = subset_prepared_vcf(
        bean_union_vcf,
        bean_null_manifest,
        bean_null_vcf,
        bean_null_popmap,
        bean_union_audit,
    )

    scaler, model, head_audit = simulation_direction_head(data_root, MAX_DEPTH)
    panels = [
        score_panel(
            "andean_duck_beta_positive",
            beta_vcf,
            beta_popmap,
            ("YBP_low", "YBP_high", "ST_high"),
            beta_audit,
            scaler,
            model,
            {
                "kind": "published positive topology/domain-shift benchmark",
                "expected_class_if_transferable": "C (ST_high -> YBP_high)",
                "published": {"D": 0.78, "fD": 0.68, "z": 4.586, "p": "<2e-6"},
                "direction_source": "separate IMa2 HBB analysis; D sign alone is not directional",
                "topology_caveat": "true quartet is ((YBP_high,YBP_low),(ST_high,ST_low)), not a pectinate tree",
            },
        ),
        score_panel(
            "andean_duck_alpha_negative",
            alpha_vcf,
            alpha_popmap,
            ("YBP_low", "YBP_high", "ST_high"),
            alpha_audit,
            scaler,
            model,
            {
                "kind": "published negative control",
                "expected": "no introgression direction call",
                "published": {"D": -0.04, "fD": -0.005, "p": 0.477},
                "head_caveat": "the direction head has no no-event class and must emit A, B, or C",
            },
        ),
        score_panel(
            "runner_bean_cdmx_wild_to_cultivar_positive",
            bean_cdmx_vcf,
            bean_cdmx_popmap,
            ("Cult_SUR_CH", "Cult_TMV_B", "Wild_TMV_B_CDMX"),
            bean_cdmx_audit,
            scaler,
            model,
            {
                "kind": "published positive D/f4 benchmark",
                "expected_class_if_transferable": "C (Wild_TMV_B_CDMX -> Cult_TMV_B)",
                "published": {"D": 0.109339, "f4_ratio": 0.421888, "z": 8.37171, "p": 0},
                "published_scenario": "Table S4 / vcf_scenario1.4_subset.txt",
                "replay_caveat": "canonical metadata uses FrijCol_11; the released scenario substitutes excluded FrijCol_10",
                "direction_caveat": "D/f4 establish excess sharing, not direction by themselves; class C follows the paper's wild-to-crop interpretation",
            },
        ),
        score_panel(
            "runner_bean_tepoz_wild_to_cultivar_replicate",
            bean_tepoz_vcf,
            bean_tepoz_popmap,
            ("Cult_SUR_CH", "Cult_TMV_B", "Wild_TMV_B_Tepoz"),
            bean_tepoz_audit,
            scaler,
            model,
            {
                "kind": "published positive donor replicate",
                "expected_class_if_transferable": "C (Wild_TMV_B_Tepoz -> Cult_TMV_B)",
                "published": {"D": 0.107835, "f4_ratio": 0.156145, "z": 6.95236, "p": 0},
                "published_scenario": "Table S4 / vcf_scenario1.5_subset.txt",
                "replay_caveat": "canonical metadata uses FrijCol_11; the released scenario substitutes excluded FrijCol_10",
                "direction_caveat": "D/f4 establish excess sharing, not direction by themselves; class C follows the paper's wild-to-crop interpretation",
            },
        ),
        score_panel(
            "runner_bean_published_null",
            bean_null_vcf,
            bean_null_popmap,
            ("Cult_SMOCC", "Cult_TMV_B_Spain", "Cult_TMV_B"),
            bean_null_audit,
            scaler,
            model,
            {
                "kind": "published null D/f4 benchmark",
                "expected": "abstention / no directional evidence",
                "published": {"D": 0.0008093, "f4_ratio": 0, "z": 0.0439146, "p": 0.96497},
                "published_scenario": "Table S4 / vcf_scenario1.8_subset.txt",
                "replay_caveat": "canonical metadata uses FrijCol_11; the released scenario substitutes excluded FrijCol_10",
                "head_caveat": "the direction head has no no-event class and must emit A, B, or C",
            },
        ),
    ]

    result = {
        "schema_version": 1,
        "status": "exploratory external OOD diagnostics; not classifier validation",
        "source_manifest": json.loads((MANIFEST_DIR / "sources.json").read_text(encoding="utf-8")),
        "source_file_audit": source_audit,
        "simulation_head_audit": head_audit,
        "panels": panels,
        "reproducibility": {
            "repository": git_revision(),
            "command": " ".join(sys.argv),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "padze": package_version("padze"),
            "max_depth": MAX_DEPTH,
            "locus_cap": args.cap,
            "process_priority": "BelowNormal requested on Windows",
        },
    }
    destination = result_dir / "results.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {destination}")
    for panel in panels:
        head = panel["simulation_head"]
        shift = panel["simulation_feature_shift"]
        print(
            f"{panel['panel_id']}: {head['predicted_class']} {head['scores']} "
            f"rms_z={shift['rms_z']:.3g} max_abs_z={shift['max_abs_z']:.3g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
