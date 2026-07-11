#!/usr/bin/env python3
"""Run a guarded pedigree-anchored dingo--dog direction stress test.

Weeks et al. (2025) include eight captive dingo-dog-hybrid x dingo
backcrosses.  Their breeding history supplies SNP-independent evidence for a
dog-introgressing component in the backcrosses.  With Alpine dingoes as P1,
the backcross cohort as P2, and domestic dogs as P3, the candidate orientation
for that component is DNNaic class C (P3 -> P2).

The eight P2 animals provide exactly 16 diploid copies at full call rate, so
missing genotypes remove loci at the g=16 contract.  The references are proxy
endpoints rather than the documented parents, and the cohort may contain
relatives.  The cross is not an exclusive population-level single-edge
history, so it cannot by itself yield a formal direction-accuracy estimate.
Filter variants are correlated sensitivities, not independent validation
trials.  Learned natural-data scores are always checked against the
prespecified severe-OOD rule before candidate concordance is interpreted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import urllib.request


SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from additional_external_benchmarks import add_gate_score, simulation_gate_head
from external_benchmarks import (
    MANIFEST_DIR,
    MAX_DEPTH,
    REPO,
    git_revision,
    prepare_vcf,
    score_panel,
    set_below_normal_priority,
    sha256_file,
    simulation_direction_head,
    verify_file,
)
from tinkerbird_external_benchmark import frequency_projection, runtime_audit


DEFAULT_CACHE = REPO / "data" / "real" / "dingo_weeks_2025_external_benchmark"
DEFAULT_RESULTS = REPO / "results" / "dingo_weeks_2025_external_benchmark_2026_07_11"
SOURCE_RECORD = MANIFEST_DIR / "dingo_weeks_2025" / "sources.json"
SOURCE_RECORD_CONTRACT = {
    "bytes": 3_480,
    "sha256": "30e8c5088aaaea37099d2be64b4b5868ec081b59f3372ba3ae7d7ee0ec9a16ad",
}
DEFAULT_CAP = 15_000
FIGSHARE_RECORD = "https://figshare.com/articles/online_resource/27022555"
PAPER_URL = "https://academic.oup.com/evlett/article/9/1/1/7828091"

FILES = {
    "vcf": {
        "id": 49_199_059,
        "key": "Weeks_etal_434.vcf.gz",
        "url": "https://ndownloader.figshare.com/files/49199059",
        "bytes": 5_266_043,
        "md5": "cc3a6f753726c289cbcb84842ba1ed80",
        "sha256": "42620f03b3768dc71617f198372a09ff4d42e4654d30c79fc960b972ac7b8125",
    },
    "metadata": {
        "id": 49_199_065,
        "key": "Weeks_meta_434.txt",
        "url": "https://ndownloader.figshare.com/files/49199065",
        "bytes": 9_848,
        "md5": "3e77e058c1f3dd7aee1ad832ef382e2f",
        "sha256": "d7cc4fe11dc36d56a52221d648ecb4da1cab878db42e0abfc324ffc139315acb",
    },
    "geolocation": {
        "id": 49_199_062,
        "key": "Weeks_meta_geolocation_434.txt",
        "url": "https://ndownloader.figshare.com/files/49199062",
        "bytes": 16_212,
        "md5": "d6ac2d30ed054378f4098d6b8f19c516",
        "sha256": "4ee68f300e14674f771a85264cbdd281033b9ec0292862738c6b993a736ce821",
    },
}

METADATA_COUNTS = {
    "alpine": 248,
    "back": 8,
    "desert": 58,
    "dog": 39,
    "hybrid": 7,
    "mallee": 74,
}
PAPER_COUNTS = {
    "alpine": 248,
    "back": 8,
    "desert": 74,
    "dog": 39,
    "hybrid": 7,
    "mallee": 58,
}
GEOLOCATION_COUNTS = {"Alpine": 248, "Desert": 77, "Mallee": 55, "NIL": 3}
PANEL_ROLES = {"alpine": "P1", "back": "P2", "dog": "P3"}
PANEL_COUNTS = {"P1": 248, "P2": 8, "P3": 39}
PANEL_MANIFEST_BYTES = 3_245
PANEL_MANIFEST_SHA256 = "12eed285869fa3426b82442ebbb44e866af77cf6013c7853776d421d20f23d45"
METADATA_SEMANTIC_SHA256 = "a0e8f0ecdcf1ad29c741db1ef8b66d1bc679276906a4a1a593b4e7c9d25bc2e8"
GROUP_ORDERED_ID_SHA256 = {
    "alpine": "5e10710f9193b5f288f1947b411b436355b7287f6daac27b6256830983602402",
    "back": "b9c8c24878f29e187318cac07e0bc9591ed6c9899c345127274ac4f209879678",
    "dog": "ae8b18da92a76ad4fe818564f9f31288e1a4626955838b28375a01b36e1e985a",
}
SOURCE_FILTER_COUNTS = {
    "PASS": 2_233,
    "VQSRTrancheSNP99.00to99.90": 125,
    "VQSRTrancheSNP99.90to100.00": 108,
}
SOURCE_ELIGIBILITY = {
    "all_release_rows": {
        "minimum_16_called_copies": 2_201,
        "pooled_polymorphic": 2_193,
        "within_each_population_polymorphic": 1_594,
    },
    "vcf_PASS_only": {
        "minimum_16_called_copies": 1_999,
        "pooled_polymorphic": 1_992,
        "within_each_population_polymorphic": 1_551,
    },
}
NORMALISED_ALLPASS = {
    "bytes": 39_485_888,
    "sha256": "c15274389e2601ae230e3092684f65aa01d158f2b33d00ea7961f2dd345ac9c6",
}
EXPECTED_FILTER_LOCI = {
    ("paper_release_all_rows", "standard_contract"): 2_193,
    ("paper_release_all_rows", "within_population_polymorphism"): 1_594,
    ("vcf_PASS_only", "standard_contract"): 1_992,
    ("vcf_PASS_only", "within_population_polymorphism"): 1_551,
}
EXPECTED_DERIVED_PANELS = {
    "dingo_backcross_paper_release_all_rows_standard_contract": {
        "vcf_bytes": 2_689_302,
        "vcf_sha256": "05130efd130b6375c0949b59d2ced8bfc68d5c0a833b56f0ebb9c2aa0dcd6519",
        "ordered_locus_sha256": "d5b469c2e5f3f62ca0703dc931aa84bb58bab3ddf0485b5b01e3b134b12ca00c",
    },
    "dingo_backcross_paper_release_all_rows_within_population_polymorphism": {
        "vcf_bytes": 1_956_357,
        "vcf_sha256": "e674ab8994aa25235d3e3e434d2cd0dac9341bcb89001cc6a65292b407bf4aed",
        "ordered_locus_sha256": "b931307dff9a253c3cec54c101c68bd5e8398640c6264a1a454e559363715e84",
    },
    "dingo_backcross_vcf_PASS_only_standard_contract": {
        "vcf_bytes": 2_443_577,
        "vcf_sha256": "2f400d4c3e30ef28ba3572fdd042c20383658971cb79cf1ddd90eb0967d8303b",
        "ordered_locus_sha256": "79696ea682909b94d4a64f1eb84d989367aa58948fa66d11bbe453019ce0be6f",
    },
    "dingo_backcross_vcf_PASS_only_within_population_polymorphism": {
        "vcf_bytes": 1_903_775,
        "vcf_sha256": "98f44ca8522aeb6815f27f95e4ddeadaf557cfa0fe6f5f2bf5be5fd27f3938fb",
        "ordered_locus_sha256": "6dd0696039348e18ff2e5db4f8c08f4760cefe59f1c3e15c5eaffcdbe16b75dd",
    },
}
PUBLISHED_FST = {"P1_P2": 0.07, "P2_P3": 0.17, "P1_P3": 0.27}
SEVERE_OOD_RMS_Z = 10.0


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DNNaic external benchmark/1"})
    with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(output)


def ensure_source(path: Path, spec: dict, download_missing: bool) -> dict:
    if not path.exists():
        if not download_missing:
            raise FileNotFoundError(path)
        _download(spec["url"], path)
    verified = verify_file(path, spec["bytes"], spec["sha256"])
    observed_md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if observed_md5 != spec["md5"]:
        raise AssertionError(f"{path}: MD5 mismatch")
    return {**verified, "md5": observed_md5, "figshare_file_id": spec["id"]}


def validate_sources_record() -> dict:
    raw = SOURCE_RECORD.read_bytes()
    observed_contract = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    if observed_contract != SOURCE_RECORD_CONTRACT:
        raise AssertionError("dingo sources.json byte contract changed")
    record = json.loads(SOURCE_RECORD.read_text(encoding="utf-8"))
    if record["schema_version"] != "dnnaic-dingo-weeks-2025-source-v1":
        raise AssertionError("unexpected dingo source schema")
    if record["data_doi"] != "10.6084/m9.figshare.27022555.v1":
        raise AssertionError("unexpected dingo data DOI")
    for name, spec in FILES.items():
        if record["files"][name] != spec:
            raise AssertionError(f"sources.json {name} contract differs from runner")
    return {
        "path": str(SOURCE_RECORD),
        **observed_contract,
        "record": record,
    }


def read_metadata(path: Path) -> tuple[list[tuple[str, str, str]], dict]:
    rows: list[tuple[str, str, str]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"metadata line {line_number} does not have three fields")
        rows.append((fields[0], fields[1], fields[2]))
    if len(rows) != 434 or len({row[0] for row in rows}) != len(rows):
        raise AssertionError("metadata must contain 434 unique sample IDs")
    if any(first != second for first, second, _ in rows):
        raise AssertionError("metadata DID and IID columns differ")
    counts = dict(sorted(Counter(row[2] for row in rows).items()))
    if counts != METADATA_COUNTS:
        raise AssertionError(f"metadata population counts changed: {counts}")
    semantic = "".join("\t".join(row) + "\n" for row in rows)
    if _sha256_text(semantic) != METADATA_SEMANTIC_SHA256:
        raise AssertionError("metadata semantic digest changed")
    group_hashes = {}
    for label in PANEL_ROLES:
        text = "".join(row[0] + "\n" for row in rows if row[2] == label)
        group_hashes[label] = _sha256_text(text)
    if group_hashes != GROUP_ORDERED_ID_SHA256:
        raise AssertionError("ordered panel sample identities changed")
    return rows, {
        "rows": len(rows),
        "unique_sample_ids": len(rows),
        "DID_equals_IID": True,
        "population_counts": counts,
        "semantic_sha256": METADATA_SEMANTIC_SHA256,
        "ordered_panel_group_id_sha256": group_hashes,
    }


def geolocation_audit(path: Path, metadata_rows: list[tuple[str, str, str]]) -> dict:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows or list(rows[0]) != ["DID", "IID", "X", "Y", "pop"]:
        raise AssertionError("unexpected geolocation columns")
    counts = dict(sorted(Counter(row["pop"] for row in rows).items()))
    if counts != GEOLOCATION_COUNTS:
        raise AssertionError(f"geolocation population counts changed: {counts}")
    geo_ids = {row["DID"] for row in rows}
    metadata_ids = {row[0] for row in metadata_rows}
    metadata_only = sorted(metadata_ids - geo_ids)
    geolocation_only = sorted(geo_ids - metadata_ids)
    if geolocation_only != ["2467566", "2777065", "2777079"]:
        raise AssertionError("geolocation-only IDs changed")
    metadata_label = {row[0]: row[2] for row in metadata_rows}
    metadata_only_counts = dict(sorted(Counter(metadata_label[sample] for sample in metadata_only).items()))
    if metadata_only_counts != {"back": 8, "dog": 39, "hybrid": 7}:
        raise AssertionError("metadata-only samples are no longer exactly the captive groups")
    return {
        "rows": len(rows),
        "unique_DID": len(geo_ids),
        "population_counts": counts,
        "metadata_only_samples": len(metadata_only),
        "metadata_only_population_counts": metadata_only_counts,
        "geolocation_only_samples": geolocation_only,
        "release_discrepancy": (
            "The headerless analysis metadata labels desert=58 and mallee=74, while paper Table 2 "
            "reports Desert=74 and Mallee=58; the geolocation file has Desert=77, Mallee=55, "
            "and three NIL rows absent from the analysis metadata. The Alpine/backcross/dog "
            "positive panel is unaffected; Desert and Mallee controls are excluded."
        ),
    }


def materialize_manifest(
    metadata_rows: list[tuple[str, str, str]], output: Path
) -> dict:
    text = "".join(
        f"{sample}\t{PANEL_ROLES[label]}\n"
        for sample, _iid, label in metadata_rows
        if label in PANEL_ROLES
    )
    raw = text.encode("utf-8")
    if len(raw) != PANEL_MANIFEST_BYTES or hashlib.sha256(raw).hexdigest() != PANEL_MANIFEST_SHA256:
        raise AssertionError("materialized dingo panel manifest changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    counts = Counter(line.split("\t")[1] for line in text.splitlines())
    if dict(sorted(counts.items())) != PANEL_COUNTS:
        raise AssertionError("materialized dingo panel counts changed")
    return {
        "path": str(output),
        "bytes": len(raw),
        "sha256": PANEL_MANIFEST_SHA256,
        "samples": sum(counts.values()),
        "population_counts": dict(sorted(counts.items())),
    }


def source_vcf_audit(path: Path, metadata_rows: list[tuple[str, str, str]]) -> dict:
    labels = {row[0]: row[2] for row in metadata_rows}
    samples: list[str] | None = None
    panel_columns: dict[str, list[int]] | None = None
    variants = 0
    filters: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    chromosomes: Counter[str] = Counter()
    missing = Counter()
    locus_ids: set[str] = set()
    coordinates: set[tuple[str, str]] = set()
    eligibility = {
        "all_release_rows": Counter(),
        "vcf_PASS_only": Counter(),
    }
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                samples = line.rstrip("\r\n").split("\t")[9:]
                expected = [row[0] for row in metadata_rows]
                if samples != expected:
                    raise AssertionError("VCF sample order differs from analysis metadata")
                panel_columns = {
                    label: [9 + index for index, sample in enumerate(samples) if labels[sample] == label]
                    for label in PANEL_ROLES
                }
                continue
            if line.startswith("#") or not line.strip():
                continue
            if samples is None or panel_columns is None:
                raise ValueError("VCF variant precedes #CHROM")
            fields = line.rstrip("\r\n").split("\t")
            variants += 1
            if len(fields[3]) != 1 or len(fields[4]) != 1 or "," in fields[4]:
                raise AssertionError("source contains a non-biallelic SNP")
            key = (fields[0], fields[1])
            if fields[2] in locus_ids or key in coordinates:
                raise AssertionError("source has duplicate locus identity")
            locus_ids.add(fields[2])
            coordinates.add(key)
            filters[fields[6]] += 1
            formats[fields[8]] += 1
            chromosomes[fields[0]] += 1
            population_alleles = {}
            for label, columns in panel_columns.items():
                alleles = []
                for index in columns:
                    genotype = fields[index].split(":", 1)[0].replace("|", "/")
                    parts = genotype.split("/")
                    if genotype in {".", "./."} or "." in parts:
                        missing[label] += 1
                        continue
                    if len(parts) != 2 or not set(parts).issubset({"0", "1"}):
                        raise AssertionError(f"unexpected diploid genotype {genotype!r}")
                    alleles.extend(parts)
                population_alleles[label] = alleles
            enough = all(len(alleles) >= 16 for alleles in population_alleles.values())
            pooled = enough and len(set(sum(population_alleles.values(), []))) == 2
            strict = enough and all(len(set(alleles)) == 2 for alleles in population_alleles.values())
            for scope in ("all_release_rows", "vcf_PASS_only"):
                if scope == "vcf_PASS_only" and fields[6] != "PASS":
                    continue
                eligibility[scope]["minimum_16_called_copies"] += int(enough)
                eligibility[scope]["pooled_polymorphic"] += int(pooled)
                eligibility[scope]["within_each_population_polymorphic"] += int(strict)
    if variants != 2_466 or samples is None:
        raise AssertionError("VCF must contain 434 samples and 2,466 variants")
    if dict(sorted(filters.items())) != SOURCE_FILTER_COUNTS:
        raise AssertionError(f"VCF FILTER counts changed: {filters}")
    observed_eligibility = {
        scope: dict(values) for scope, values in eligibility.items()
    }
    if observed_eligibility != SOURCE_ELIGIBILITY:
        raise AssertionError(f"source eligibility changed: {observed_eligibility}")
    if set(chromosomes) != {f"chr{value}" for value in range(1, 39)}:
        raise AssertionError("source is no longer autosomes chr1--chr38 only")
    return {
        "samples": len(samples),
        "variants": variants,
        "biallelic_autosomal_SNPs": True,
        "chromosomes": len(chromosomes),
        "source_filter_counts": dict(sorted(filters.items())),
        "format_counts": dict(sorted(formats.items())),
        "panel_missing_genotypes": dict(sorted(missing.items())),
        "eligibility_before_DNNaic_filtering": observed_eligibility,
        "sample_order_matches_metadata": True,
        "unique_locus_ids_and_coordinates": True,
    }


def normalise_release_filters(source: Path, output: Path) -> dict:
    filters: Counter[str] = Counter()
    variants = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(source, "rt", encoding="utf-8", newline="") as incoming, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as outgoing:
        for line in incoming:
            if line.startswith("#"):
                outgoing.write(line.rstrip("\r\n") + "\n")
                continue
            fields = line.rstrip("\r\n").split("\t")
            filters[fields[6]] += 1
            fields[6] = "PASS"
            variants += 1
            outgoing.write("\t".join(fields) + "\n")
    if variants != 2_466 or dict(sorted(filters.items())) != SOURCE_FILTER_COUNTS:
        raise AssertionError("normalization source contract changed")
    if output.stat().st_size != NORMALISED_ALLPASS["bytes"] or sha256_file(output) != NORMALISED_ALLPASS["sha256"]:
        raise AssertionError("normalized author-release VCF digest changed")
    return {
        "operation": "set FILTER=PASS on every released data row; no other field changed",
        "rationale": (
            "The paper reports all 2,466 released SNPs as its Australian dataset. This scope "
            "reproduces that row set; the separate vcf_PASS_only scope retains the source FILTER tags."
        ),
        "original_filter_counts": dict(sorted(filters.items())),
        "derived_vcf": {
            "path": str(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        },
    }


def adjudicate_panel(panel: dict) -> dict:
    prediction = panel["simulation_head"]["predicted_class"]
    direction_rms = panel["simulation_feature_shift"]["rms_z"]
    gate_rms = panel["simulation_gate_feature_shift"]["rms_z"]
    severe = max(direction_rms, gate_rms) > SEVERE_OOD_RMS_Z
    return {
        "pedigree_dog_introgression_component_available": True,
        "exclusive_single_edge_truth_available": False,
        "pedigree_candidate_class_for_dog_component": "C",
        "raw_head_matches_candidate_C": prediction == "C",
        "severe_OOD": severe,
        "severe_OOD_rule": (
            f"max(direction RMS-z, gate RMS-z) > {SEVERE_OOD_RMS_Z:g}; "
            "heuristic diagnostic, not calibrated support"
        ),
        "natural_data_call_status": (
            "abstain_severe_OOD" if severe else "descriptive_candidate_concordance_only"
        ),
        "direction_call_accepted": False,
        "formal_direction_accuracy_eligible": False,
        "gate_truth_available": False,
        "gate_accuracy_eligible": False,
        "guardrail": (
            "Pedigree establishes dog ancestry independently of SNPs, but this backcross is not an "
            "exclusive population-level single-edge history. The potentially related cohort is one "
            "biological unit, uses proxy references, and filter scopes are not independent trials."
        ),
    }


def summarize_outcomes(panels: list[dict]) -> dict:
    if len(panels) != 4:
        raise AssertionError("dingo benchmark must contain exactly four nested sensitivities")
    prediction_counts = Counter(panel["simulation_head"]["predicted_class"] for panel in panels)
    severe = sum(panel["adjudication"]["severe_OOD"] for panel in panels)
    return {
        "analytic_filter_sensitivity_runs": len(panels),
        "unique_biological_systems": 1,
        "independent_pedigree_dog_component_units": 1,
        "exclusive_single_edge_truth_units": 0,
        "independent_sample_level_units": 0,
        "correlated_filter_sensitivities_not_trials": True,
        "raw_head_prediction_counts": dict(sorted(prediction_counts.items())),
        "raw_head_matches_candidate_C": sum(
            panel["adjudication"]["raw_head_matches_candidate_C"] for panel in panels
        ),
        "severe_OOD_panels": severe,
        "abstained_panels": severe,
        "descriptive_nonsevere_panels": len(panels) - severe,
        "accepted_direction_calls": 0,
        "direction_accuracy_estimate": None,
        "gate_accuracy_estimate": None,
        "accuracy_guardrail": (
            "The pedigree anchors the dog-introgressing component, not an exclusive one-edge "
            "population history or an accuracy rate; four filters reuse the same eight P2 animals "
            "and reference samples."
        ),
    }


def run_panels(
    sources: dict[str, Path],
    manifest: Path,
    cache: Path,
    cap: int,
    direction_head,
    gate_head,
) -> list[dict]:
    panels = []
    for source_scope in ("paper_release_all_rows", "vcf_PASS_only"):
        source = sources[source_scope]
        for filter_name, strict in (
            ("standard_contract", False),
            ("within_population_polymorphism", True),
        ):
            panel_id = f"dingo_backcross_{source_scope}_{filter_name}"
            prepared_vcf = cache / f"{panel_id}.vcf"
            prepared_popmap = cache / f"{panel_id}.popmap.tsv"
            audit = prepare_vcf(
                source,
                manifest,
                prepared_vcf,
                prepared_popmap,
                cap=cap,
                seed=20260711,
                polymorphic_within_each_population=strict,
            )
            expected_loci = EXPECTED_FILTER_LOCI[(source_scope, filter_name)]
            if audit["counts"]["eligible_before_cap"] != expected_loci or audit["counts"]["retained_after_cap"] != expected_loci:
                raise AssertionError(f"{panel_id}: expected {expected_loci} loci")
            expected_derived = EXPECTED_DERIVED_PANELS[panel_id]
            observed_derived = {
                "vcf_bytes": audit["derived_vcf"]["bytes"],
                "vcf_sha256": audit["derived_vcf"]["sha256"],
                "ordered_locus_sha256": audit["ordered_locus_sha256"],
            }
            if observed_derived != expected_derived:
                raise AssertionError(f"{panel_id}: prepared VCF contract changed")
            observed_popmap = {
                "bytes": audit["derived_popmap"]["bytes"],
                "sha256": audit["derived_popmap"]["sha256"],
            }
            if observed_popmap != {
                "bytes": PANEL_MANIFEST_BYTES,
                "sha256": PANEL_MANIFEST_SHA256,
            }:
                raise AssertionError(f"{panel_id}: prepared popmap contract changed")
            audit["source_filter_scope"] = source_scope
            audit["source_filter_interpretation"] = (
                "all 2,466 rows reported as the paper's Australian SNP dataset; inherited FILTER tags normalized"
                if source_scope == "paper_release_all_rows"
                else "only the 2,233 source rows whose VCF FILTER field is PASS"
            )
            audit["P2_complete_case_ascertainment"] = {
                "P2_samples": PANEL_COUNTS["P2"],
                "maximum_diploid_copies": 16,
                "minimum_called_copies_contract": 16,
                "all_retained_loci_complete_case_in_P2": True,
                "guardrail": (
                    "Because P2 has exactly eight diploids, the frozen g=16 requirement removes "
                    "every locus with any missing P2 genotype; the analyzed row set is therefore "
                    "conditioned on complete P2 calls."
                ),
            }
            expectation = {
                "benchmark_role": "pedigree_anchored_dog_component_direction_stress_test",
                "candidate_class": "C",
                "candidate_forward_direction": "domestic dog P3 -> captive backcross P2",
                "pedigree": "(dingo x dog) F1 hybrid x dingo",
                "nominal_dog_ancestry": 0.25,
                "candidate_direction_basis": (
                    "documented dog ancestry in the captive backcross, independent of SNP genotypes"
                ),
                "exclusive_single_edge_truth_available": False,
                "published_pairwise_FST": PUBLISHED_FST,
                "reference_caveat": (
                    "Alpine dingoes and pooled dogs are proxy endpoints, not documented individual parents"
                ),
                "relatedness_caveat": "the release does not establish eight independent families",
                "tree_contract_status": (
                    "operational proxy order only: P2 is an admixed backcross cohort and P1/P3 "
                    "are proxy references rather than established tree leaves or documented parents"
                ),
                "locus_filter_variant": filter_name,
                "source_filter_scope": source_scope,
            }
            panel = score_panel(
                panel_id,
                prepared_vcf,
                prepared_popmap,
                ("P1", "P2", "P3"),
                audit,
                direction_head[0],
                direction_head[1],
                expectation,
            )
            add_gate_score(panel, gate_head[0], gate_head[1])
            panel["model_free_comparator"] = frequency_projection(
                prepared_vcf,
                manifest,
                ("P1", "P2", "P3"),
                bootstrap_replicates=500,
            )
            panel["adjudication"] = adjudicate_panel(panel)
            panels.append(panel)
    return panels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    parser.add_argument("--vcf")
    parser.add_argument("--metadata")
    parser.add_argument("--geolocation")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    if args.cap < max(EXPECTED_FILTER_LOCI.values()):
        parser.error(f"--cap must be at least {max(EXPECTED_FILTER_LOCI.values())}")

    set_below_normal_priority()
    cache = Path(args.cache_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    supplied = {"vcf": args.vcf, "metadata": args.metadata, "geolocation": args.geolocation}
    paths = {
        name: Path(value).resolve() if value else cache / FILES[name]["key"]
        for name, value in supplied.items()
    }
    verified = {
        name: ensure_source(paths[name], FILES[name], args.download_missing)
        for name in FILES
    }
    sources_record = validate_sources_record()
    metadata_rows, metadata = read_metadata(paths["metadata"])
    geolocation = geolocation_audit(paths["geolocation"], metadata_rows)
    vcf_audit = source_vcf_audit(paths["vcf"], metadata_rows)
    manifest = cache / "dingo_backcross_panel.tsv"
    manifest_audit = materialize_manifest(metadata_rows, manifest)
    normalized = cache / "Weeks_etal_434.author_release_allpass.vcf"
    normalization = normalise_release_filters(paths["vcf"], normalized)
    direction_head = simulation_direction_head(Path(args.data_root).resolve(), max_depth=MAX_DEPTH)
    gate_head = simulation_gate_head(Path(args.data_root).resolve(), max_depth=MAX_DEPTH)
    panels = run_panels(
        {"paper_release_all_rows": normalized, "vcf_PASS_only": paths["vcf"]},
        manifest,
        cache,
        args.cap,
        direction_head,
        gate_head,
    )
    result = {
        "schema_version": "dnnaic-dingo-weeks-2025-external-benchmark-v1",
        "git": git_revision(),
        "guardrail": (
            "One pedigree-anchored, eight-animal dog-component stress-test cohort with proxy endpoints; "
            "the cross is not an exclusive population-level single-edge history. "
            "Four source/locus filters are correlated sensitivities. Raw severe-OOD scores are "
            "not accepted calls, probabilities, ancestry estimates, or migration rates."
        ),
        "source": {
            "record": FIGSHARE_RECORD,
            "data_doi": "10.6084/m9.figshare.27022555.v1",
            "paper": PAPER_URL,
            "paper_doi": "10.1093/evlett/qrae057",
            "license": "CC-BY-4.0",
            "verified_files": verified,
            "sources_record": sources_record,
            "metadata_audit": metadata,
            "geolocation_audit": geolocation,
            "vcf_audit": vcf_audit,
            "release_filter_normalization": normalization,
        },
        "analysis_design": {
            "population_order": {"P1": "alpine", "P2": "back", "P3": "dog"},
            "sample_counts": PANEL_COUNTS,
            "manifest": manifest_audit,
            "candidate_class": "C",
            "candidate_direction_basis": (
                "known dog ancestry in the captive backcross, independent of SNPs"
            ),
            "pedigree_dog_introgression_component_available": True,
            "exclusive_single_edge_truth_available": False,
            "formal_direction_accuracy_eligible": False,
            "gate_truth_available": False,
            "nominal_dog_ancestry": 0.25,
            "published_pairwise_FST": PUBLISHED_FST,
            "excluded_known_F1_group": {
                "label": "hybrid",
                "samples": 7,
                "reason": "14 maximum diploid copies cannot support the frozen g=2..16 grid",
            },
            "excluded_desert_mallee_controls": geolocation["release_discrepancy"],
        },
        "direction_head": direction_head[2],
        "gate_head": gate_head[2],
        "runtime": runtime_audit(),
        "panels": panels,
        "outcome": summarize_outcomes(panels),
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
                        "loci": panel["padze"]["n_loci_kept"],
                        "raw_direction": panel["simulation_head"]["predicted_class"],
                        "raw_gate": panel["simulation_gate"]["appreciable_score"],
                        "direction_rms_z": panel["simulation_feature_shift"]["rms_z"],
                        "gate_rms_z": panel["simulation_gate_feature_shift"]["rms_z"],
                        "projection": panel["model_free_comparator"]["P2_projection_from_P1_toward_P3_all_loci"],
                        "status": panel["adjudication"]["natural_data_call_status"],
                    }
                    for panel in panels
                ],
                "outcome": result["outcome"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
