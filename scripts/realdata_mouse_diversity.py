#!/usr/bin/env python3
"""Exploratory mouse Vkorc1 diagnostics with a fixed callable-site panel.

This analysis does not validate an introgression-direction classifier.  It asks
how a descriptive P2--P3 versus P1--P3 sharing contrast changes when P1 is
M. m. musculus or M. m. castaneus.  Both trios are constructed from one
four-population callable-site intersection, capped only after intersection, so
the reference swap cannot change the loci, REF/ALT coding, or locus order.

The direction score uses the paper's primary model: one 54-dimensional vector
per canonical simulation replicate (mean and population SD over g=2..199 for
the 27 non-depth PADZE coordinates), a StandardScaler, and multinomial logistic
regression fit to all 2,700 positive canonical replicates.  Natural softmax
values are explicitly uncalibrated scores, not probabilities or posteriors.

The remote Harr et al. VCF region and its tabix index are cached under
``data/real/mouse``.  Configure the canonical arrays with ``--data-root`` or
``DNNAIC_DATA``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dnnaic import build_matrix
from dnnaic.tabix import http_get, read_tbi


VCF_URL = (
    "https://wwwuser.gwdguser.de/~evolbio/evolgen/wildmouse/vcf/"
    "AllMouse.vcf_90_recalibrated_snps_raw_indels_reheader_PopSorted.PASS.vcf.gz"
)
TBI_URL = VCF_URL + ".tbi"
GENOME_BUILD = "GRCm38/mm10 (as documented by the Harr et al. wild-mouse resource)"
REGION = ("chr7", 126_000_000, 129_000_000)
POPULATION_PREFIX = {
    "Mmm": "musculus",
    "Mmc": "castaneus",
    "Mmd": "domesticus",
    "Ms": "spretus",
}
FOUR_POPULATIONS = ("musculus", "castaneus", "domesticus", "spretus")
CLASSES = np.array(["A", "B", "C"])
DEFAULT_CAP = 15_000
MIN_CALLED_COPIES = 16
DEFAULT_CUTOFFS = (2, 4, 6, 8, 10, 12, 14)
DEFAULT_CLIPS = (0.0, 1e-12, 1e-9, 1e-6, 1e-4)

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = REPO / "data" / "real" / "mouse"
DEFAULT_RESULTS = REPO / "results" / "natural_diagnostics_2026_07_09"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_locus_sha256(rows: list[list[str]]) -> str:
    """Hash the ordered CHROM/POS/REF/ALT identity shared by both panels."""
    digest = hashlib.sha256()
    for fields in rows:
        digest.update("\t".join((fields[0], fields[1], fields[3], fields[4])).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"


def called_alleles(cell: str) -> list[str]:
    genotype = cell.split(":", 1)[0].replace("|", "/")
    return [allele for allele in genotype.split("/") if allele in ("0", "1")]


def fetch_complete_region(names, refs) -> tuple[list[str], int]:
    """Fetch complete BGZF coverage, increasing the byte range until end is crossed."""
    chromosome, begin, end = REGION
    intervals = refs[names.index(chromosome)]
    window = begin >> 14
    if window >= len(intervals):
        raise RuntimeError(f"tabix index has no interval for {REGION}")
    compressed_offset = intervals[window] >> 16
    for extra in (22_000_000, 44_000_000, 88_000_000, 176_000_000):
        data = http_get(VCF_URL, rng=f"bytes={compressed_offset}-{compressed_offset + extra}")
        lines = []
        crossed_end = False
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as handle:
                for raw in handle:
                    line = raw.decode("latin-1")
                    if line.startswith("#"):
                        continue
                    fields = line.split("\t", 2)
                    if fields[0] != chromosome:
                        if lines:
                            crossed_end = True
                            break
                        continue
                    position = int(fields[1])
                    if position < begin:
                        continue
                    if position > end:
                        crossed_end = True
                        break
                    lines.append(line)
        except (EOFError, OSError):
            pass
        if crossed_end:
            return lines, extra
    raise RuntimeError(
        f"HTTP ranges up to 176 MB did not cross {chromosome}:{end}; refusing a truncated cache"
    )


def source_cache(cache_dir: Path) -> tuple[list[str], list[str], dict]:
    """Return source sample names, cached region lines, and source provenance."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    tbi_path = cache_dir / "wildmouse_source.vcf.gz.tbi"
    header_path = cache_dir / "wildmouse_header_prefix.bin"
    region_path = cache_dir / "wildmouse_chr7_126000000_129000000.lines.gz"
    region_meta_path = cache_dir / "wildmouse_chr7_126000000_129000000.fetch.json"

    if not tbi_path.exists():
        tbi_path.write_bytes(http_get(TBI_URL))
    names, refs = read_tbi(tbi_path.read_bytes())

    if not header_path.exists():
        header_path.write_bytes(http_get(VCF_URL, rng="bytes=0-1500000"))
    with gzip.GzipFile(fileobj=io.BytesIO(header_path.read_bytes())) as handle:
        samples = None
        for raw in handle:
            line = raw.decode("latin-1")
            if line.startswith("#CHROM"):
                samples = line.rstrip("\n").split("\t")[9:]
                break
    if samples is None:
        raise RuntimeError("cached VCF header does not contain #CHROM")

    if not region_path.exists() or not region_meta_path.exists():
        lines, byte_range = fetch_complete_region(names, refs)
        with region_path.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as zipped:
                zipped.write("".join(lines).encode("latin-1"))
        region_meta_path.write_text(
            json.dumps({
                "complete_to_requested_end": True,
                "http_range_extra_bytes": byte_range,
                "last_retained_record": (
                    lines[-1].split("\t", 2)[:2] if lines else None
                ),
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    with gzip.open(region_path, "rt", encoding="latin-1") as handle:
        lines = handle.readlines()

    provenance = {
        "vcf_url": VCF_URL,
        "tbi_url": TBI_URL,
        "genome_build": GENOME_BUILD,
        "region_1_based_inclusive": [REGION[0], REGION[1], REGION[2]],
        "cached_header_prefix": {
            "path": str(header_path.relative_to(REPO)),
            "bytes": header_path.stat().st_size,
            "sha256": sha256_file(header_path),
        },
        "cached_tabix_index": {
            "path": str(tbi_path.relative_to(REPO)),
            "bytes": tbi_path.stat().st_size,
            "sha256": sha256_file(tbi_path),
        },
        "cached_region_lines": {
            "path": str(region_path.relative_to(REPO)),
            "compressed_bytes": region_path.stat().st_size,
            "sha256": sha256_file(region_path),
            "source_records": len(lines),
            "fetch_audit": json.loads(region_meta_path.read_text(encoding="utf-8")),
        },
    }
    return samples, lines, provenance


def sample_map(samples: list[str]) -> tuple[dict[str, str], dict[str, list[int]]]:
    mapping = {
        sample: population
        for sample in samples
        for prefix, population in POPULATION_PREFIX.items()
        if sample.startswith(prefix + "_")
    }
    indices = {
        population: [index for index, sample in enumerate(samples) if mapping.get(sample) == population]
        for population in FOUR_POPULATIONS
    }
    if any(not values for values in indices.values()):
        raise RuntimeError(f"missing a required population in VCF header: { {k: len(v) for k, v in indices.items()} }")
    return mapping, indices


def fixed_intersection(
    lines: list[str], indices: dict[str, list[int]], cap: int
) -> tuple[list[list[str]], dict]:
    """Filter once across all four populations, then apply one shared cap.

    If the intersection exceeds the cap, deterministic evenly spaced indices
    cover the full ordered region instead of taking a position-biased prefix.
    """
    eligible: list[list[str]] = []
    counts = {
        "source_records": len(lines),
        "single_base_biallelic": 0,
        "callable_all_four": 0,
        "polymorphic_in_both_three_population_panels": 0,
    }
    for line in lines:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 10 or len(fields[3]) != 1 or len(fields[4]) != 1:
            continue
        if fields[3] not in "ACGT" or fields[4] not in "ACGT" or "," in fields[4]:
            continue
        counts["single_base_biallelic"] += 1
        alleles = {
            population: [
                allele
                for sample_index in indices[population]
                for allele in called_alleles(fields[9 + sample_index])
            ]
            for population in FOUR_POPULATIONS
        }
        if any(len(alleles[population]) < MIN_CALLED_COPIES for population in FOUR_POPULATIONS):
            continue
        counts["callable_all_four"] += 1
        panel_musculus = alleles["musculus"] + alleles["domesticus"] + alleles["spretus"]
        panel_castaneus = alleles["castaneus"] + alleles["domesticus"] + alleles["spretus"]
        if set(panel_musculus) != {"0", "1"} or set(panel_castaneus) != {"0", "1"}:
            continue
        counts["polymorphic_in_both_three_population_panels"] += 1
        eligible.append(fields)
    if not eligible:
        raise RuntimeError("the four-population callable intersection is empty")
    if len(eligible) > cap:
        selected = np.linspace(0, len(eligible) - 1, cap, dtype=np.int64)
        retained = [eligible[int(index)] for index in selected]
        selection = "deterministic evenly spaced indices across the ordered full intersection"
    else:
        retained = eligible
        selection = "all eligible loci (intersection did not exceed cap)"
    counts["cap_after_intersection"] = cap
    counts["eligible_before_cap"] = len(eligible)
    counts["cap_selection"] = selection
    counts["retained_identical_loci"] = len(retained)
    counts["first_locus"] = f"{retained[0][0]}:{retained[0][1]}"
    counts["last_locus"] = f"{retained[-1][0]}:{retained[-1][1]}"
    return retained, counts


def write_panel(
    rows: list[list[str]], samples: list[str], mapping: dict[str, str], indices: dict[str, list[int]],
    reference: str, cache_dir: Path,
) -> tuple[Path, Path, list[str]]:
    populations = (reference, "domesticus", "spretus")
    selected = [index for population in populations for index in indices[population]]
    selected_names = [samples[index] for index in selected]
    vcf_path = cache_dir / f"mouse_vkorc1_same_loci_p1_{reference}.vcf"
    popmap_path = cache_dir / f"mouse_vkorc1_same_loci_p1_{reference}.popmap.tsv"
    with vcf_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(selected_names) + "\n")
        for fields in rows:
            cells = [fields[9 + index].split(":", 1)[0].replace("|", "/") for index in selected]
            handle.write(
                "\t".join((fields[0], fields[1], ".", fields[3], fields[4], ".", "PASS", ".", "GT"))
                + "\t" + "\t".join(cells) + "\n"
            )
    with popmap_path.open("w", encoding="utf-8", newline="\n") as handle:
        for sample in selected_names:
            handle.write(f"{sample}\t{mapping[sample]}\n")
    return vcf_path, popmap_path, selected_names


def direction_features(table: np.ndarray) -> np.ndarray:
    values = np.asarray(table, float)[:, :, 1:]
    return np.concatenate((values.mean(axis=1), values.std(axis=1)), axis=1)


def canonical_direction_head(data_root: Path):
    directory = data_root / "regen_full"
    X = np.load(directory / "X.npy", mmap_mode="r")
    direction = np.load(directory / "direction.npy", mmap_mode="r").astype("U2")
    groups = np.load(directory / "groups.npy", mmap_mode="r").astype("U80")
    unique, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    order = np.lexsort((np.asarray(X[:, 0]), inverse))
    per = np.bincount(inverse)
    if per.min() != 198 or per.max() != 198:
        raise AssertionError("canonical curves must contain g=2..199")
    table = np.asarray(X[order]).reshape(len(unique), 198, X.shape[1])
    features = direction_features(table)
    labels = direction[first]
    positive = labels != "D"
    y = np.searchsorted(CLASSES, labels[positive])
    scaler = StandardScaler().fit(features[positive])
    model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs").fit(
        scaler.transform(features[positive]), y
    )
    audit = {
        "training_dataset": "regen_full",
        "training_selection": "all positive canonical replicates at every migration rate",
        "training_n": int(positive.sum()),
        "training_class_counts": {label: int((labels[positive] == label).sum()) for label in CLASSES},
        "representation": "mean and population SD over available depths for each of 27 non-depth coordinates",
        "dimension": int(features.shape[1]),
        "model": "StandardScaler plus multinomial logistic regression (lbfgs, C=1)",
        "softmax_interpretation": "uncalibrated out-of-distribution model score; not a probability or posterior",
        "fit_iterations": model.n_iter_.astype(int).tolist(),
    }
    return scaler, model, audit


def ratio_diagnostics(
    X: np.ndarray, columns: list[str], cutoffs=DEFAULT_CUTOFFS, clips=DEFAULT_CLIPS
) -> dict:
    X = np.asarray(X, float)
    ix = {column: index for index, column in enumerate(columns)}
    depth = X[:, ix["g"]]
    alpha1 = X[:, ix["alpha_1_mean"]] - 1.0
    alpha2 = X[:, ix["alpha_2_mean"]] - 1.0
    p13 = X[:, ix["pihat_13_mean"]]
    p23 = X[:, ix["pihat_23_mean"]]
    by_cutoff = {}
    for cutoff in cutoffs:
        use = depth >= cutoff
        if not use.any():
            continue
        raw_denominator = float(np.mean(p13[use]))
        raw = float(np.mean(p23[use]) / raw_denominator) if raw_denominator > 0 else None
        normalized = {}
        for clip in clips:
            key = f"{clip:.0e}" if clip else "no_clip"
            if clip == 0 and (np.any(alpha1[use] <= 0) or np.any(alpha2[use] <= 0)):
                normalized[key] = None
                continue
            a1 = np.maximum(alpha1[use], clip) if clip else alpha1[use]
            a2 = np.maximum(alpha2[use], clip) if clip else alpha2[use]
            denominator = float(np.mean(p13[use] / a1))
            normalized[key] = float(np.mean(p23[use] / a2) / denominator) if denominator > 0 else None
        by_cutoff[str(cutoff)] = {
            "n_depths": int(use.sum()),
            "raw_ratio_of_depth_means": raw,
            "normalized_ratio_of_depth_means_by_alpha_clip": normalized,
            "min_alpha1_minus_one": float(alpha1[use].min()),
            "min_alpha2_minus_one": float(alpha2[use].min()),
        }
    return {
        "definition": {
            "raw": "mean_g(pihat_23_mean) / mean_g(pihat_13_mean)",
            "normalized": "mean_g[pihat_23_mean/max(alpha_2_mean-1,clip)] / mean_g[pihat_13_mean/max(alpha_1_mean-1,clip)]",
        },
        "primary_cutoff": 8,
        "primary_clip": 1e-12,
        "by_minimum_depth": by_cutoff,
    }


def panel_copy_counts(rows: list[list[str]], indices: dict[str, list[int]]) -> dict:
    result = {}
    for population in FOUR_POPULATIONS:
        values = [
            sum(len(called_alleles(fields[9 + index])) for index in indices[population])
            for fields in rows
        ]
        result[population] = {
            "individuals_in_source": len(indices[population]),
            "diploid_gene_copies_if_complete": 2 * len(indices[population]),
            "called_gene_copies_per_retained_locus_min": int(min(values)),
            "called_gene_copies_per_retained_locus_max": int(max(values)),
            "called_gene_copies_per_retained_locus_mean": float(np.mean(values)),
        }
    return result


def heterozygosity_on_rows(rows: list[list[str]], sample_indices: list[int]) -> float:
    values = []
    for fields in rows:
        alleles = [
            allele
            for index in sample_indices
            for allele in called_alleles(fields[9 + index])
        ]
        if not alleles:
            continue
        frequency = alleles.count("1") / len(alleles)
        values.append(2 * frequency * (1 - frequency))
    return float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", default=os.environ.get("DNNAIC_DATA", str(REPO / "data" / "simulation_data"))
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    samples, source_lines, source = source_cache(cache_dir)
    mapping, indices = sample_map(samples)
    rows, filter_audit = fixed_intersection(source_lines, indices, args.cap)
    common_locus_hash = ordered_locus_sha256(rows)
    scaler, model, head_audit = canonical_direction_head(data_root)

    panels = []
    for reference in ("musculus", "castaneus"):
        populations = [reference, "domesticus", "spretus"]
        vcf_path, popmap_path, selected_names = write_panel(
            rows, samples, mapping, indices, reference, cache_dir
        )
        X, columns, loci = build_matrix(
            str(vcf_path), str(popmap_path), max_depth=100, pop_order=populations
        )
        X = np.asarray(X, float)
        if loci.metadata.n_loci_kept != len(rows):
            raise AssertionError(
                f"{reference}: PADZE retained {loci.metadata.n_loci_kept}, expected {len(rows)} identical loci"
            )
        feature = direction_features(X[None, :, :])
        score = model.predict_proba(scaler.transform(feature))[0]
        ratios = ratio_diagnostics(X, columns)
        primary = ratios["by_minimum_depth"]["8"]
        panels.append(
            {
                "p1_reference": reference,
                "population_order": populations,
                "ordered_chrom_pos_ref_alt_sha256": common_locus_hash,
                "vcf_sha256": sha256_file(vcf_path),
                "vcf_cache": str(vcf_path.relative_to(REPO)),
                "popmap_cache": str(popmap_path.relative_to(REPO)),
                "sample_ids": selected_names,
                "individual_counts": {population: len(indices[population]) for population in populations},
                "gene_copies_if_complete": {population: 2 * len(indices[population]) for population in populations},
                "retained_loci": int(loci.metadata.n_loci_kept),
                "depths": X[:, 0].astype(int).tolist(),
                "direction_call": str(CLASSES[int(np.argmax(score))]),
                "uncalibrated_softmax_scores": {
                    str(label): float(value) for label, value in zip(CLASSES, score)
                },
                "primary_raw_ratio_g_ge_8": primary["raw_ratio_of_depth_means"],
                "primary_normalized_ratio_g_ge_8_clip_1e-12": primary[
                    "normalized_ratio_of_depth_means_by_alpha_clip"
                ]["1e-12"],
                "sharing_ratio_sensitivity": ratios,
                "reference_heterozygosity_on_identical_loci": heterozygosity_on_rows(
                    rows, indices[reference]
                ),
                "domesticus_heterozygosity_on_identical_loci": heterozygosity_on_rows(
                    rows, indices["domesticus"]
                ),
            }
        )

    result = {
        "schema_version": "natural-diagnostics-mouse-v2",
        "status": "exploratory_not_validation",
        "biological_context": (
            "The chr7 window overlaps Vkorc1, where spretus-to-domesticus adaptive introgression "
            "has been documented; the Harr panel and sharing ratios are not an independent direction label."
        ),
        "source": source,
        "filter": {
            "sequence": [
                "single-base biallelic REF/ALT",
                f"at least {MIN_CALLED_COPIES} called gene copies in every one of four populations",
                "polymorphic in each three-population panel after either P1 is omitted",
                "one common ordered locus list selected before the cap",
            ],
            **filter_audit,
        },
        "source_population_samples": {
            population: [samples[index] for index in indices[population]]
            for population in FOUR_POPULATIONS
        },
        "copy_count_audit": panel_copy_counts(rows, indices),
        "direction_head": head_audit,
        "panels": panels,
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
    result_path = result_dir / "mouse.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (cache_dir / "mouse_reference_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "result": str(result_path),
        "identical_loci": len(rows),
        "panels": [
            {
                "p1": panel["p1_reference"],
                "call": panel["direction_call"],
                "scores": panel["uncalibrated_softmax_scores"],
                "raw_g8": panel["primary_raw_ratio_g_ge_8"],
                "normalized_g8": panel["primary_normalized_ratio_g_ge_8_clip_1e-12"],
            }
            for panel in panels
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
