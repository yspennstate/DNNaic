#!/usr/bin/env python3
"""Run a paired European-seabass directional transfer benchmark.

Robinet et al. (2020) report a localized Mediterranean-ancestry excess in the
Atlantic SINE sample.  With PENI as a fixed Atlantic sister reference and a
pooled Mediterranean donor reference, the published direction is P3 -> P2
(DNNaic class C).  VIGO is a same-source contrast whose Mediterranean ancestry
is slightly below PENI, so it must not be described as a pristine no-flow null.

The released source is a sparse 1,012-marker SNP-chip/WGS merge with
recipient-conditioned ascertainment.  Results are therefore mechanistic OOD
stress tests, not an unbiased external accuracy estimate.  Every learned score
is uncalibrated and is not a probability or biological posterior.
"""
from __future__ import annotations

import argparse
import csv
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
    subset_prepared_vcf,
    verify_file,
)


DEFAULT_CACHE = REPO / "data" / "real" / "seabass_external_benchmark"
DEFAULT_RESULTS = REPO / "results" / "seabass_external_benchmark_2026_07_11"
DEFAULT_CAP = 15_000
ZENODO_RECORD = "https://zenodo.org/records/3989825"
ZENODO_API = "https://zenodo.org/api/records/3989825/files"

FILES = {
    "ped": {
        "key": "Merged_827_ChipsATL_10_GenomesMED_1012_SNPs_NoRep_23reg.ped",
        "bytes": 3_412_429,
        "sha256": "9e199dc329e6bd822888cb14ff9563a18d0a0ef2ca241f9c827cfb623ce670e1",
    },
    "map": {
        "key": "Merged_827_ChipsATL_10_GenomesMED_1012_SNPs_NoRep_23reg.map",
        "bytes": 26_661,
        "sha256": "1554fb442b9450e37ace6d1ceb898bab5f21da741103416104ecc27a8f38d9b2",
    },
    "metadata": {
        "key": "noms_837DLAB_ATLMED_ICESNAME_sreg_xy.csv",
        "bytes": 38_105,
        "sha256": "e2f89878e8954a17d760c967d2a72f8c2af1caabd9664e6ece71ed7e9f1280df",
    },
    "ancestry": {
        "key": "Merged_827_Chips_10_Genomes_1012_SNPs_NoRep.2.Q",
        "bytes": 29_166,
        "sha256": "48ca6265657f30a2c518103e68237c7d3f9086c1d2ca3e13056a10c181e0bfb6",
    },
    "summary": {
        "key": "mean_ancestrality_distSINE.csv",
        "bytes": 1_518,
        "sha256": "38026d593814732cdafff0f5132fbf4d2a13dbcd460636946ed53f86c4a48e0f",
    },
}

PANEL_DIR = MANIFEST_DIR / "seabass"
UNION_MANIFEST = PANEL_DIR / "union.tsv"
POSITIVE_MANIFEST = PANEL_DIR / "positive.tsv"
CONTROL_MANIFEST = PANEL_DIR / "control.tsv"
MED_PED_IDS = (
    "T1F_WMED", "T2F_WMED", "T5F_WMED", "T6F_WMED", "T7F_WMED", "T8F_WMED",
    "T5M_EMED", "T6M_EMED", "T7M_EMED", "T8M_EMED",
)
MED_METADATA_IDS = (
    "suppl1", "suppl3", "suppl7", "suppl9", "suppl11", "suppl13",
    "suppl8", "suppl10", "suppl12", "suppl14",
)


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(output)


def ensure_source(path: Path, spec: dict, download_missing: bool) -> dict:
    if not path.exists():
        if not download_missing:
            raise FileNotFoundError(path)
        _download(f"{ZENODO_API}/{spec['key']}/content", path)
    return verify_file(path, spec["bytes"], spec["sha256"])


def convert_ped_map_to_vcf(
    ped: Path,
    map_path: Path,
    output: Path,
    *,
    expected_samples: int = 837,
    expected_markers: int = 1_012,
    expected_sha256: str | None = "6fc291d99cac1b2d389928caa6edbd328e15e199b36776259d9a117dc8bb4bf6",
) -> dict:
    """Convert the source PED/MAP with deterministic nominal allele coding."""
    map_rows = [line.split() for line in map_path.read_text(encoding="ascii").splitlines() if line.strip()]
    ped_rows = [line.split() for line in ped.read_text(encoding="ascii").splitlines() if line.strip()]
    n_markers = len(map_rows)
    if n_markers != expected_markers or len(ped_rows) != expected_samples:
        raise AssertionError(
            f"source must contain {expected_samples} samples and {expected_markers} markers"
        )
    if any(len(row) != 6 + 2 * n_markers for row in ped_rows):
        raise AssertionError("PED row width does not match MAP marker count")
    iids = [row[1] for row in ped_rows]
    if len(set(iids)) != len(iids) or any(row[0] != row[1] for row in ped_rows):
        raise AssertionError("PED requires unique equal FID/IID values")
    if len({row[1] for row in map_rows}) != n_markers:
        raise AssertionError("MAP marker IDs are not unique")
    if len({(row[0], row[3]) for row in map_rows}) != n_markers:
        raise AssertionError("MAP chromosome/position pairs are not unique")

    allele_pairs = []
    missing_genotypes = 0
    for marker_index in range(n_markers):
        alleles = {
            row[6 + 2 * marker_index + offset]
            for row in ped_rows
            for offset in (0, 1)
        }
        if not alleles.issubset({"0", "A", "C", "G", "T"}):
            raise AssertionError("unexpected PED allele code")
        alleles.discard("0")
        if len(alleles) != 2:
            raise AssertionError("every source marker must be biallelic")
        allele_pairs.append(tuple(sorted(alleles)))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##source=dnnaic_seabass_pedmap_contract_v1\n")
        handle.write("##reference=nominal-allele-coding-no-reference-genome\n")
        handle.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Unphased genotype">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t")
        handle.write("\t".join(iids) + "\n")
        for marker_index, (chromosome, marker_id, _genetic_distance, position) in enumerate(map_rows):
            ref, alt = allele_pairs[marker_index]
            genotypes = []
            for row in ped_rows:
                a, b = row[6 + 2 * marker_index: 8 + 2 * marker_index]
                if (a == "0") != (b == "0"):
                    raise AssertionError("partial PED missing genotype")
                if a == "0":
                    missing_genotypes += 1
                    genotypes.append("./.")
                    continue
                codes = sorted(("0" if a == ref else "1", "0" if b == ref else "1"))
                genotypes.append("/".join(codes))
            handle.write(
                "\t".join(
                    [chromosome, position, marker_id, ref, alt, ".", "PASS", ".", "GT", *genotypes]
                )
                + "\n"
            )
    digest = sha256_file(output)
    if expected_sha256 is not None and digest != expected_sha256:
        raise AssertionError(f"converted VCF digest {digest} != {expected_sha256}")
    return {
        "contract": "dnnaic_seabass_pedmap_contract_v1",
        "samples": len(iids),
        "markers": n_markers,
        "missing_genotypes": missing_genotypes,
        "allele_orientation": "lexicographically smaller nominal allele REF; not reference-genome orientation",
        "derived_vcf": {"path": str(output), "bytes": output.stat().st_size, "sha256": digest},
    }


def _normalise_id(value: str) -> str:
    return value.replace("_", "").upper()


def author_ancestry_audit(
    ped: Path,
    metadata_path: Path,
    ancestry_path: Path,
    summary_path: Path,
) -> dict:
    ped_ids = [line.split()[1] for line in ped.read_text(encoding="ascii").splitlines() if line.strip()]
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    if len(metadata) != len(ped_ids):
        raise AssertionError("metadata/PED row mismatch")
    metadata_atlantic = {
        _normalise_id(row["id_specimen"]): row["id_specimen"] for row in metadata[:827]
    }
    ped_atlantic = {_normalise_id(sample): sample for sample in ped_ids[:827]}
    ped_only = sorted(set(ped_atlantic) - set(metadata_atlantic))
    metadata_only = sorted(set(metadata_atlantic) - set(ped_atlantic))
    if ped_only != ["DLAB0082", "DLAB0135", "DLAB0808"]:
        raise AssertionError(f"unexpected PED-only Atlantic samples: {ped_only}")
    if metadata_only != ["DLAB0076", "DLAB0133", "DLAB0805"]:
        raise AssertionError(f"unexpected metadata/Q-only Atlantic samples: {metadata_only}")
    if tuple(row["id_specimen"] for row in metadata[827:]) != MED_METADATA_IDS:
        raise AssertionError("Mediterranean metadata labels changed")
    if tuple(ped_ids[827:]) != MED_PED_IDS:
        raise AssertionError("Mediterranean PED order changed")

    ped_by_normalised = {_normalise_id(sample): sample for sample in ped_ids[:827]}
    ancestry: dict[str, float] = {}
    med_index = 0
    for line in ancestry_path.read_text(encoding="ascii").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 5:
            q1, q2, sample = float(fields[1]), float(fields[2]), fields[3]
            normalised = _normalise_id(sample)
            ped_sample = ped_by_normalised.get(normalised, sample)
        elif len(fields) == 2 and med_index < len(MED_PED_IDS):
            q1, q2 = map(float, fields)
            ped_sample = MED_PED_IDS[med_index]
            med_index += 1
        else:
            raise AssertionError(f"unexpected Q row: {fields}")
        if not np.isclose(q1 + q2, 1.0, atol=2e-6):
            raise AssertionError("Q ancestry columns do not sum to one")
        if ped_sample in ancestry:
            raise AssertionError(f"duplicate Q sample: {ped_sample}")
        ancestry[ped_sample] = q2
    if len(ancestry) != 837 or med_index != len(MED_PED_IDS):
        raise AssertionError("Q/PED row mismatch")

    union = read_manifest(UNION_MANIFEST, require_three=False)
    missing_benchmark_q = sorted(set(union) - set(ancestry))
    if missing_benchmark_q:
        raise AssertionError(f"benchmark samples absent from Q file: {missing_benchmark_q}")
    stats = {}
    for population in ("PENI", "SINE", "VIGO", "MED"):
        values = np.asarray(
            [ancestry[sample] for sample, label in union.items() if label == population],
            dtype=float,
        )
        stats[population] = {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "n_ge_0.15": int(np.sum(values >= 0.15)),
        }
    expected_means = {"PENI": 0.0448902069, "SINE": 0.1118777778, "VIGO": 0.0365005}
    for population, expected in expected_means.items():
        if not np.isclose(stats[population]["mean"], expected, atol=6e-8):
            raise AssertionError(f"unexpected {population} author ancestry mean")

    with summary_path.open(encoding="utf-8-sig", newline="") as handle:
        summary = {row["LOC"]: row for row in csv.DictReader(handle)}
    vigo_min = float(summary["VIGO"]["min_anc_MED"])
    vigo_max = float(summary["VIGO"]["max_anc_MED"])
    return {
        "Q_column": "Q2 (Mediterranean ancestry; chosen by the author labels and MED references)",
        "source_sample_set_mismatch": {
            "PED_only": [ped_atlantic[value] for value in ped_only],
            "metadata_and_Q_only": [metadata_atlantic[value] for value in metadata_only],
            "benchmark_samples_affected": [],
            "guardrail": (
                "the released PED and Q/metadata bundles differ by three non-benchmark Atlantic fish; "
                "ancestry benchmarks use explicit IDs, never positional alignment"
            ),
        },
        "population_statistics": stats,
        "contrasts": {
            "SINE_minus_PENI_mean": stats["SINE"]["mean"] - stats["PENI"]["mean"],
            "VIGO_minus_PENI_mean": stats["VIGO"]["mean"] - stats["PENI"]["mean"],
        },
        "published_result": (
            "SINE mean Mediterranean ancestry 11.2%, significantly above every other locality; "
            "six SINE fish at 15-40% were interpreted as late-generation backcrosses"
        ),
        "published_distance_test": "p < 2e-16",
        "source_summary_inconsistency": {
            "field": "VIGO min_anc_MED",
            "reported": vigo_min,
            "reported_max": vigo_max,
            "raw_Q_minimum": stats["VIGO"]["minimum"],
            "raw_Q_maximum": stats["VIGO"]["maximum"],
            "interpretation": "the reported minimum is impossible; raw Q values are authoritative here",
        },
    }


def frequency_projection(vcf: Path, manifest_path: Path, pop_order: tuple[str, str, str]) -> dict:
    mapping = read_manifest(manifest_path)
    columns = None
    values = {population: [] for population in pop_order}
    with open_text(vcf) as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                sample_column = {sample: 9 + index for index, sample in enumerate(samples)}
                columns = {
                    population: [sample_column[sample] for sample, label in mapping.items() if label == population]
                    for population in pop_order
                }
                continue
            if line.startswith("#") or not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            for population in pop_order:
                alleles = [allele for index in columns[population] for allele in called_alleles(fields[index])]
                values[population].append(sum(allele == "1" for allele in alleles) / len(alleles))
    p1, p2, p3 = (np.asarray(values[population], dtype=float) for population in pop_order)
    axis = p3 - p1
    denominator = float(np.mean(axis**2))
    return {
        "n_loci": int(len(p1)),
        "P2_projection_from_P1_toward_P3": float(np.mean((p2 - p1) * axis) / denominator),
        "mean_squared_frequency_distance": {
            "P1_P2": float(np.mean((p1 - p2) ** 2)),
            "P2_P3": float(np.mean((p2 - p3) ** 2)),
            "P1_P3": denominator,
        },
        "interpretation": "reference-invariant frequency projection; not a temporal-direction estimator",
    }


def run_panels(vcf: Path, cache: Path, cap: int, direction_head, gate_head) -> list[dict]:
    panels = []
    for suffix, strict in (("standard_contract", False), ("within_population_polymorphism", True)):
        shared_vcf = cache / f"seabass.{suffix}.shared.vcf"
        shared_popmap = cache / f"seabass.{suffix}.shared.popmap.tsv"
        shared_audit = prepare_vcf(
            vcf,
            UNION_MANIFEST,
            shared_vcf,
            shared_popmap,
            cap=cap,
            seed=20260711,
            require_three_populations=False,
            polymorphic_panel_manifests=(POSITIVE_MANIFEST, CONTROL_MANIFEST),
            polymorphic_within_each_population=strict,
        )
        specs = (
            (
                "positive",
                POSITIVE_MANIFEST,
                ("PENI", "SINE", "MED"),
                {
                    "expected_class": "C",
                    "expected_forward_direction": "MED (P3) -> SINE (P2)",
                    "published_benchmark": "SINE mean Mediterranean ancestry 0.111878 versus PENI 0.044890",
                    "direction_basis": "source demographic/ancestry interpretation; the ancestry contrast alone is not directional",
                },
            ),
            (
                "control",
                CONTROL_MANIFEST,
                ("PENI", "VIGO", "MED"),
                {
                    "expected_gate": "must not reproduce the SINE-positive contrast",
                    "direction_truth": None,
                    "published_benchmark": "VIGO mean Mediterranean ancestry 0.036501 versus PENI 0.044890",
                    "guardrail": "reversed low-contrast specificity panel, not proof of absolute no flow",
                },
            ),
        )
        for label, manifest, order, expectation in specs:
            panel_vcf = cache / f"seabass.{suffix}.{label}.vcf"
            panel_popmap = cache / f"seabass.{suffix}.{label}.popmap.tsv"
            audit = subset_prepared_vcf(shared_vcf, manifest, panel_vcf, panel_popmap, shared_audit)
            audit["comparison_locus_contract"] = "same ordered PENI/SINE/VIGO/MED callable-site intersection"
            expectation = {
                **expectation,
                "locus_filter_variant": (
                    "both alleles observed within PENI, SINE, VIGO, and MED"
                    if strict
                    else "both matched trios polymorphic on one shared callable-site intersection"
                ),
            }
            panel = score_panel(
                f"seabass_{label}_{suffix}",
                panel_vcf,
                panel_popmap,
                order,
                audit,
                direction_head[0],
                direction_head[1],
                expectation,
            )
            add_gate_score(panel, gate_head[0], gate_head[1])
            panel["model_free_comparator"] = frequency_projection(panel_vcf, manifest, order)
            panels.append(panel)
    return panels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    parser.add_argument("--ped")
    parser.add_argument("--map")
    parser.add_argument("--metadata")
    parser.add_argument("--ancestry")
    parser.add_argument("--summary")
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
        "ped": args.ped,
        "map": args.map,
        "metadata": args.metadata,
        "ancestry": args.ancestry,
        "summary": args.summary,
    }
    paths = {
        name: Path(value).resolve() if value else cache / FILES[name]["key"]
        for name, value in supplied.items()
    }
    verified = {
        name: ensure_source(paths[name], FILES[name], args.download_missing)
        for name in FILES
    }
    converted_vcf = cache / "seabass_837.nominal.vcf"
    conversion = convert_ped_map_to_vcf(paths["ped"], paths["map"], converted_vcf)
    ancestry = author_ancestry_audit(
        paths["ped"], paths["metadata"], paths["ancestry"], paths["summary"]
    )
    direction_head = simulation_direction_head(Path(args.data_root).resolve(), max_depth=MAX_DEPTH)
    gate_head = simulation_gate_head(Path(args.data_root).resolve(), max_depth=MAX_DEPTH)
    result = {
        "schema_version": "dnnaic-seabass-external-benchmark-v1",
        "git": git_revision(),
        "guardrail": (
            "Sparse recipient-ascertained SNP-chip/WGS transfer diagnostic; not an unbiased accuracy estimate. "
            "The VIGO contrast is not a pristine no-flow null."
        ),
        "source": {
            "record": ZENODO_RECORD,
            "data_doi": "10.5281/zenodo.3989825",
            "paper_doi": "10.1111/mec.15611",
            "license": "CC-BY-4.0",
            "author_code_commit": "dab84b0f936703ac5da0fac1e301cfef926cc9df",
            "verified_files": verified,
        },
        "conversion": conversion,
        "author_ancestry_benchmark": ancestry,
        "direction_head": direction_head[2],
        "gate_head": gate_head[2],
        "panels": run_panels(converted_vcf, cache, args.cap, direction_head, gate_head),
    }
    output = result_dir / "results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
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
                        "projection": panel["model_free_comparator"]["P2_projection_from_P1_toward_P3"],
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
