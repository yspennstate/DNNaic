#!/usr/bin/env python3
"""Run a guarded Fitzpatrick 2020 guppy genetic-rescue transfer test.

In March 2009, guppies were experimentally introduced above waterfall barriers
in the Caigual and Taylor drainages. Subsequent downstream flow into established
recipient populations fixes the biological source-to-recipient orientation
independently of the SNPs. The released VCFs sample pre-flow recipients,
post-flow recipients, and an SGS mainstem source proxy. This is therefore a
strong candidate-direction stress test, but not the exact contemporaneous
three-population tree or an exclusive single-edge history.

The five author-deposited VCFs have no explicit repository license, so the
runner fetches hash-pinned bytes at runtime and never vendors them. Caigual and
Taylor are two ecological recipient units sharing one donor proxy. Standard and
strict locus filters are correlated sensitivities; the strict panels contain
only 30 and 22 loci. The author release and both runner filters use post-flow
samples during locus selection, so this is not prospective held-out validation.
No row becomes a formal accuracy estimate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
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


DEFAULT_CACHE = REPO / "data" / "real" / "guppy_2020_external_benchmark"
DEFAULT_RESULTS = REPO / "results" / "guppy_2020_external_benchmark_2026_07_11"
SOURCE_RECORD = MANIFEST_DIR / "guppy_2020" / "sources.json"
SOURCE_RECORD_CANONICAL_LF_CONTRACT = {
    "bytes": 4_765,
    "sha256": "2691009ccc844526c254c06795e5f6855bf49ffd781325fa4e1c4ea0ad805b60",
}
DEFAULT_CAP = 15_000
SEVERE_OOD_RMS_Z = 10.0
COMMIT = "ac8ec0cdf29dec539494b49d8bdf32ff6f0197f2"
TREE = "eac1fe39081906b691f857e4493864db66361b02"
REPOSITORY = "https://github.com/gbradburd/guppy_seln"
PAPER_URL = "https://www.sciencedirect.com/science/article/pii/S0960982219315325"
ORIGINAL_EXPERIMENT_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4947150/"

FILES = {
    "NCA": {
        "key": "nchr_NCA_mapped.vcf",
        "url": f"https://raw.githubusercontent.com/gbradburd/guppy_seln/{COMMIT}/1_data/nchr_NCA_mapped.vcf",
        "bytes": 4_967_968,
        "sha256": "4cf713eac4243808c85800b5b192ff19b8132d66b240c6cc9f48f5658ab2c940",
        "git_blob": "471ab9e4952cfd72d9dd53e298393ef73b51632b",
    },
    "NTY": {
        "key": "nchr_NTY_mapped.vcf",
        "url": f"https://raw.githubusercontent.com/gbradburd/guppy_seln/{COMMIT}/1_data/nchr_NTY_mapped.vcf",
        "bytes": 4_829_947,
        "sha256": "041de93e3fb4361d029dc272a633ff6af8415959297730e34ba6e10ac88fc0b0",
        "git_blob": "d5282c035bd85fe48d475df7dfa895f50696f61f",
    },
    "PCA": {
        "key": "nchr_PCA_mapped.vcf",
        "url": f"https://raw.githubusercontent.com/gbradburd/guppy_seln/{COMMIT}/1_data/nchr_PCA_mapped.vcf",
        "bytes": 5_276_051,
        "sha256": "e4eddf5b9917db4fed4ed80ee42317fd8faebc8246472ab420815f78abadd682",
        "git_blob": "9e767e652387657fea2aad8bd23eb951eb6684fc",
    },
    "PTY": {
        "key": "nchr_PTY_mapped.vcf",
        "url": f"https://raw.githubusercontent.com/gbradburd/guppy_seln/{COMMIT}/1_data/nchr_PTY_mapped.vcf",
        "bytes": 6_252_249,
        "sha256": "75945f56e2ffd9360df5173db9fc050f0b0ac684c3bacdc380a46a3310ba4ff3",
        "git_blob": "464d42c9e88c6ffe4bc0d6676b4914e10c25f2b8",
    },
    "SGS": {
        "key": "nchr_SGS_mapped.vcf",
        "url": f"https://raw.githubusercontent.com/gbradburd/guppy_seln/{COMMIT}/1_data/nchr_SGS_mapped.vcf",
        "bytes": 2_870_728,
        "sha256": "88fe8b9e17493a9d8f5b930a76ee71eb6e6c11c1455cacd6c4a42b72adca7692",
        "git_blob": "0a63f9b8bab4607f45dca6ddf7ac66ae203d476b",
    },
}
FORMAT_SCRIPT = {
    "key": "format_guppy_data.R",
    "url": f"https://raw.githubusercontent.com/gbradburd/guppy_seln/{COMMIT}/1_data/format_guppy_data.R",
    "bytes": 3_657,
    "sha256": "ce3ea9823bb21d284bec2552ffa544f57353c24f5d9e8ea031aa0a6783ff3236",
    "git_blob": "dd2463fe0beb5de95d42a1cb46df31684c8b3617",
}

SAMPLE_IDS = {
    "NCA": [
        "NCA-01", "NCA-02", "NCA-03", "NCA-04", "NCA-05", "NCA-06", "NCA-07",
        "NCA-08", "NCA-09", "NCA-10", "NCA-12", "NCA-13", "NCA-14", "NCA-16",
        "NCA-17", "NCA-18", "NCA-19", "NCA-20",
    ],
    "NTY": [
        "NTY-01", "NTY-02", "NTY-03", "NTY-04", "NTY-05", "NTY-06", "NTY-07",
        "NTY-08", "NTY-09", "NTY-10", "NTY-11", "NTY-12", "NTY-13", "NTY-14",
        "NTY-15", "NTY-17", "NTY-18",
    ],
    "PCA": [
        "PCA-01", "PCA-02", "PCA-03", "PCA-06", "PCA-07", "PCA-09", "PCA-10",
        "PCA-11", "PCA-12", "PCA-13", "PCA-14", "PCA-15", "PCA-17", "PCA-18",
        "PCA-19", "PCA-20", "PCA-21", "PCA-22", "PCA-23",
    ],
    "PTY": [f"PTY-{value:02d}" for value in range(1, 24)],
    "SGS": [f"SGS-{value:02d}" for value in range(2, 11)],
}

PANEL_SPECS = {
    "caigual": {
        "groups": {"P1": "NCA", "P2": "PCA", "P3": "SGS"},
        "counts": {"P1": 18, "P2": 19, "P3": 9},
        "manifest_bytes": 460,
        "manifest_sha256": "51164cdf7692f98a035371f8ce5c482bfda2de2be7b85139143ef05c41bb71ec",
        "published_FST_pre_to_post": [0.29, 0.01],
        "published_monomorphic_pre_to_post": [0.95, 0.22],
    },
    "taylor": {
        "groups": {"P1": "NTY", "P2": "PTY", "P3": "SGS"},
        "counts": {"P1": 17, "P2": 23, "P3": 9},
        "manifest_bytes": 490,
        "manifest_sha256": "1197e3fc32abb5593c1edca40d1d22cc95c0eb43325b88eb9bdc8ee9440936d7",
        "published_FST_pre_to_post": [0.31, 0.02],
        "published_monomorphic_pre_to_post": [0.96, 0.24],
    },
}

SOURCE_VARIANT_CONTRACT = {
    "variants": 11_417,
    "chromosomes": 23,
    "ordered_locus_sha256": "3daefba0c6ac6dd3f8b285633bf09be2329ad0436b7528f5be7e6b9bc12e7c73",
}
EXPECTED_PANEL_LOCI = {
    ("caigual", "standard_contract"): (
        6_877,
        "e2133443750113e175a463df5490d32d2262c30c46f95f317cc90b124da7d338",
    ),
    ("caigual", "within_population_polymorphism"): (
        30,
        "caaa48c0fd276886937e210ae2d8f62a5fc7e070dc7cf9d8105b4d7a6b744c39",
    ),
    ("taylor", "standard_contract"): (
        6_696,
        "e9dbd8c73fec4e2d5261f9a58479b46e3b3314a060b31570033d916e4bf1ae88",
    ),
    ("taylor", "within_population_polymorphism"): (
        22,
        "b333cecf5dccc51b0b1984d1dfce01c7192cb0865ee6379a0223e24c7488cae9",
    ),
}
COMBINED_VCF_CONTRACT = {
    "bytes": 4_305_263,
    "sha256": "83a00efa9836673e0a894b794ea9b3d2820defd3ff453bafa7d55a830b53b49f",
}
EXPECTED_PREPARED_VCF = {
    "guppy_caigual_standard_contract": {
        "bytes": 1_492_537,
        "sha256": "4683a85f0c81810728c85e13b541ba0e9b5a35e7e229c3a489d7987464ea9920",
    },
    "guppy_caigual_within_population_polymorphism": {
        "bytes": 6_962,
        "sha256": "865418741f8b0247d38fa94ea4d5ed8363c912598008f9e21bae1ab1409f6373",
    },
    "guppy_taylor_standard_contract": {
        "bytes": 1_533_642,
        "sha256": "30ec1717e7e251db0812018600d19da8eb2c08309f513892a06ad2e3afb324d1",
    },
    "guppy_taylor_within_population_polymorphism": {
        "bytes": 5_513,
        "sha256": "b06ef1b7ccd944b1bc343d4b6d7e7537b09f4786c24019eb4f4892dad6605d92",
    },
}


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _download(url: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "DNNaic external benchmark/1"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, temporary.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def git_blob_sha1(path: Path) -> str:
    """Compute the Git blob object ID for the exact file bytes."""
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(path: Path, spec: dict) -> dict:
    verified = verify_file(path, spec["bytes"], spec["sha256"])
    observed_blob = git_blob_sha1(path)
    if observed_blob != spec["git_blob"]:
        raise ValueError(f"{path}: Git blob SHA-1 mismatch: {observed_blob}")
    return {**verified, "git_blob_sha1": observed_blob}


def ensure_source(path: Path, spec: dict, download_missing: bool) -> dict:
    if path.exists():
        try:
            return _verify_source(path, spec)
        except (OSError, ValueError):
            if not download_missing:
                raise
            path.unlink(missing_ok=True)
    elif not download_missing:
        raise FileNotFoundError(path)

    _download(spec["url"], path)
    try:
        return _verify_source(path, spec)
    except (OSError, ValueError):
        path.unlink(missing_ok=True)
        raise


def validate_sources_record() -> dict:
    raw = SOURCE_RECORD.read_bytes()
    canonical_lf = raw.replace(b"\r\n", b"\n")
    if b"\r" in canonical_lf:
        raise AssertionError("guppy sources.json contains a non-CRLF carriage return")
    canonical = {"bytes": len(canonical_lf), "sha256": hashlib.sha256(canonical_lf).hexdigest()}
    if canonical != SOURCE_RECORD_CANONICAL_LF_CONTRACT:
        raise AssertionError("guppy sources.json canonical LF contract changed")
    record = json.loads(raw.decode("utf-8"))
    if record["schema_version"] != "dnnaic-guppy-2020-source-v1":
        raise AssertionError("unexpected guppy source schema")
    if record["repository_commit"] != COMMIT or record["repository_tree"] != TREE:
        raise AssertionError("unexpected guppy repository revision")
    if record["files"] != FILES or record["format_script"] != FORMAT_SCRIPT:
        raise AssertionError("guppy source contracts differ from runner")
    return {
        "path": str(SOURCE_RECORD),
        "canonical_lf": canonical,
        "working_tree": {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "line_endings_normalized_for_contract": raw != canonical_lf,
        },
        "record": record,
    }


def decode_gt(value: str) -> str:
    value = value.replace("|", "/")
    if value in {".", "./."} or "." in value.split("/"):
        return "./."
    if value in {"0/0", "1/1"}:
        return value
    if value in {"0/1", "1/0"}:
        return "0/1"
    raise ValueError(f"unexpected biallelic diploid GT: {value!r}")


def _read_header(handle, population: str) -> tuple[list[str], list[str], list[str]]:
    meta = []
    for line in handle:
        if line.startswith("##"):
            meta.append(line.rstrip("\r\n"))
            continue
        if line.startswith("#CHROM"):
            fields = line.rstrip("\r\n").split("\t")
            samples = fields[9:]
            if samples != SAMPLE_IDS[population]:
                raise AssertionError(f"{population}: source sample IDs changed")
            return meta, fields[:9], samples
        raise AssertionError(f"{population}: malformed VCF header")
    raise AssertionError(f"{population}: no #CHROM header")


def materialize_combined_vcf(paths: dict[str, Path], output: Path) -> dict:
    populations = ("NCA", "NTY", "PCA", "PTY", "SGS")
    output.parent.mkdir(parents=True, exist_ok=True)
    locus_digest = hashlib.sha256()
    chromosomes = set()
    variants = 0
    missing_by_population = Counter()
    combined_samples = []
    with ExitStack() as stack:
        handles = [stack.enter_context(paths[population].open(encoding="utf-8", newline="")) for population in populations]
        headers = [_read_header(handle, population) for handle, population in zip(handles, populations)]
        if any(len(meta) != 10 for meta, _fixed, _samples in headers):
            raise AssertionError("guppy VCF metadata-header count changed")
        for _meta, _fixed, samples in headers:
            combined_samples.extend(samples)
        if len(combined_samples) != 86 or len(set(combined_samples)) != 86:
            raise AssertionError("combined guppy sample identities are not 86 unique IDs")
        with output.open("w", encoding="utf-8", newline="\n") as outgoing:
            for line in headers[0][0]:
                outgoing.write(line + "\n")
            outgoing.write("##DNNaic_join=exact shared locus key; source INFO reset because AF/NS are not population-specific\n")
            outgoing.write("\t".join(headers[0][1] + combined_samples) + "\n")
            while True:
                lines = [handle.readline() for handle in handles]
                if not any(lines):
                    break
                if not all(lines):
                    raise AssertionError("guppy source VCFs have unequal variant-row counts")
                rows = [line.rstrip("\r\n").split("\t") for line in lines]
                if any(len(row) != 9 + len(SAMPLE_IDS[population]) for row, population in zip(rows, populations)):
                    raise AssertionError("guppy VCF row width changed")
                keys = [tuple(row[:5]) for row in rows]
                if any(key != keys[0] for key in keys[1:]):
                    raise AssertionError("guppy VCF locus keys/order differ")
                variants += 1
                chrom, _pos, _id, ref, alt = keys[0]
                if len(ref) != 1 or len(alt) != 1 or "," in alt:
                    raise AssertionError("guppy source contains a non-biallelic SNP")
                chromosomes.add(chrom)
                locus_digest.update(("\t".join((chrom, rows[0][1], ref, alt)) + "\n").encode("utf-8"))
                genotypes = []
                for row, population in zip(rows, populations):
                    if row[6] != "PASS" or row[8] != "GT:DP:AD:GL":
                        raise AssertionError("guppy source FILTER/FORMAT changed")
                    for cell in row[9:]:
                        gt = decode_gt(cell.split(":", 1)[0])
                        genotypes.append(gt)
                        missing_by_population[population] += int(gt == "./.")
                fixed = rows[0][:8]
                fixed[6] = "PASS"
                fixed[7] = "."
                outgoing.write("\t".join(fixed + ["GT"] + genotypes) + "\n")
    observed = {
        "variants": variants,
        "chromosomes": len(chromosomes),
        "ordered_locus_sha256": locus_digest.hexdigest(),
    }
    if observed != SOURCE_VARIANT_CONTRACT:
        raise AssertionError(f"guppy source variant contract changed: {observed}")
    combined = {"bytes": output.stat().st_size, "sha256": sha256_file(output)}
    if combined != COMBINED_VCF_CONTRACT:
        raise AssertionError("combined guppy VCF byte contract changed")
    return {
        **observed,
        "source_files_share_exact_ordered_keys": True,
        "samples": len(combined_samples),
        "population_sample_counts": {population: len(SAMPLE_IDS[population]) for population in populations},
        "missing_genotypes": dict(sorted(missing_by_population.items())),
        "combined_vcf": {"path": str(output), **combined},
        "INFO_policy": "reset to dot; source AF/NS are from a combined dataset and not population-specific",
    }


def audit_format_script(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    wrong = '#\tNCA - post gene flow headwater Caigual'
    if wrong not in text or 'PCA - post gene flow headwater Caigual' not in text:
        raise AssertionError("format_guppy_data.R population-comment discrepancy changed")
    return {
        "discrepancy": (
            "the population-code comment labels NCA as post-flow even though PCA is also labelled "
            "post-flow and all downstream analysis/paper semantics use NCA as pre-flow"
        ),
        "benchmark_mapping": "NCA pre-flow P1; PCA post-flow P2",
    }


def materialize_manifests(output_dir: Path) -> tuple[dict[str, Path], dict]:
    manifests = {}
    audits = {}
    for panel, spec in PANEL_SPECS.items():
        population_to_role = {population: role for role, population in spec["groups"].items()}
        rows = []
        for population in ("NCA", "NTY", "PCA", "PTY", "SGS"):
            if population not in population_to_role:
                continue
            rows.extend((sample, population_to_role[population]) for sample in SAMPLE_IDS[population])
        raw = "".join(f"{sample}\t{role}\n" for sample, role in rows).encode("utf-8")
        counts = dict(sorted(Counter(role for _sample, role in rows).items()))
        if counts != spec["counts"]:
            raise AssertionError(f"{panel}: manifest counts changed")
        observed = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if observed != {
            "bytes": spec["manifest_bytes"],
            "sha256": spec["manifest_sha256"],
        }:
            raise AssertionError(f"{panel}: manifest byte contract changed")
        path = output_dir / f"guppy.{panel}.manifest.tsv"
        path.write_bytes(raw)
        manifests[panel] = path
        audits[panel] = {"path": str(path), "samples": len(rows), "population_counts": counts, **observed}
    return manifests, audits


def adjudicate_panel(panel: dict) -> dict:
    prediction = panel["simulation_head"]["predicted_class"]
    direction_rms = panel["simulation_feature_shift"]["rms_z"]
    gate_rms = panel["simulation_gate_feature_shift"]["rms_z"]
    severe = max(direction_rms, gate_rms) > SEVERE_OOD_RMS_Z
    return {
        "experimental_flow_direction_available": True,
        "candidate_class": "C",
        "raw_head_matches_candidate_C": prediction == "C",
        "exclusive_single_edge_truth_available": False,
        "formal_direction_accuracy_eligible": False,
        "gate_truth_available": False,
        "severe_OOD": severe,
        "severe_OOD_rule": (
            f"max(direction RMS-z, gate RMS-z) > {SEVERE_OOD_RMS_Z:g}; "
            "heuristic diagnostic, not calibrated support"
        ),
        "natural_data_call_status": (
            "abstain_severe_OOD" if severe else "descriptive_candidate_concordance_only"
        ),
        "direction_call_accepted": False,
        "gate_accuracy_eligible": False,
        "guardrail": (
            "the manipulation fixes downstream flow, but SGS is a proxy, P1/P2 are serial samples, "
            "and filter scopes are correlated; no formal accuracy rate is defined"
        ),
    }


def run_panels(
    combined_vcf: Path,
    manifests: dict[str, Path],
    cache: Path,
    cap: int,
    direction_head,
    gate_head,
) -> list[dict]:
    panels = []
    for panel_name, spec in PANEL_SPECS.items():
        for filter_name, strict in (
            ("standard_contract", False),
            ("within_population_polymorphism", True),
        ):
            panel_id = f"guppy_{panel_name}_{filter_name}"
            prepared_vcf = cache / f"{panel_id}.vcf"
            prepared_popmap = cache / f"{panel_id}.popmap.tsv"
            audit = prepare_vcf(
                combined_vcf,
                manifests[panel_name],
                prepared_vcf,
                prepared_popmap,
                cap=cap,
                seed=20260711,
                polymorphic_within_each_population=strict,
            )
            expected_loci, expected_hash = EXPECTED_PANEL_LOCI[(panel_name, filter_name)]
            if audit["counts"]["eligible_before_cap"] != expected_loci or audit["counts"]["retained_after_cap"] != expected_loci:
                raise AssertionError(f"{panel_id}: eligible loci changed")
            if audit["ordered_locus_sha256"] != expected_hash:
                raise AssertionError(f"{panel_id}: ordered locus contract changed")
            observed_prepared = {
                "bytes": audit["derived_vcf"]["bytes"],
                "sha256": audit["derived_vcf"]["sha256"],
            }
            if observed_prepared != EXPECTED_PREPARED_VCF[panel_id]:
                raise AssertionError(f"{panel_id}: prepared VCF contract changed")
            observed_popmap = {
                "bytes": audit["derived_popmap"]["bytes"],
                "sha256": audit["derived_popmap"]["sha256"],
            }
            if observed_popmap != {
                "bytes": spec["manifest_bytes"],
                "sha256": spec["manifest_sha256"],
            }:
                raise AssertionError(f"{panel_id}: prepared popmap contract changed")
            audit["locus_filter_variant"] = filter_name
            audit["strict_low_locus_sensitivity"] = strict
            expectation = {
                "benchmark_role": "experimentally_directed_genetic_rescue_stress_test",
                "ecological_unit": panel_name,
                "candidate_class": "C",
                "candidate_forward_direction": f"SGS source proxy P3 -> post-flow {spec['groups']['P2']} P2",
                "direction_basis": (
                    "2009 upstream translocation plus waterfall barriers imposed subsequent downstream flow"
                ),
                "experimental_flow_direction_available": True,
                "exclusive_single_edge_truth_available": False,
                "formal_direction_accuracy_eligible": False,
                "gate_truth_available": False,
                "tree_contract_status": (
                    "operational proxy order only; SGS proxies the introduced source and P1/P2 are "
                    "pre/post serial samples of one recipient lineage"
                ),
                "locus_ascertainment_outcome_blind": False,
                "locus_ascertainment_guardrail": (
                    "the author release was filtered across pre/post populations and both runner "
                    "filters inspect post-flow P2; experimental direction is SNP-independent, but "
                    "locus inclusion is not prospective held-out"
                ),
                "published_same_data_sanity_checks": {
                    "recipient_source_FST_pre_to_post": spec["published_FST_pre_to_post"],
                    "recipient_monomorphic_fraction_pre_to_post": spec["published_monomorphic_pre_to_post"],
                    "guardrail": "support strong genomic change but reuse the same experiment/data",
                },
                "locus_filter_variant": filter_name,
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
                manifests[panel_name],
                ("P1", "P2", "P3"),
                bootstrap_replicates=500,
            )
            panel["model_free_comparator"]["finite_sample_and_uncertainty_guardrail"] = (
                "projection/f3 are plug-in sample-frequency geometry; the chromosome-block "
                "bootstrap resamples linkage blocks, not fish, and supplies no temporal direction"
            )
            panel["adjudication"] = adjudicate_panel(panel)
            panels.append(panel)
    return panels


def summarize_outcomes(panels: list[dict]) -> dict:
    if len(panels) != 4:
        raise AssertionError("guppy benchmark must contain four correlated filter rows")
    predictions = Counter(panel["simulation_head"]["predicted_class"] for panel in panels)
    severe = sum(panel["adjudication"]["severe_OOD"] for panel in panels)
    return {
        "analytic_correlated_sensitivity_rows": 4,
        "ecological_recipient_units": 2,
        "shared_source_proxy": True,
        "independent_formal_accuracy_units": 0,
        "correlated_filters_not_trials": True,
        "raw_head_prediction_counts": dict(sorted(predictions.items())),
        "raw_candidate_C_concordant_sensitivity_rows": sum(
            panel["adjudication"]["raw_head_matches_candidate_C"] for panel in panels
        ),
        "accuracy_denominator": None,
        "severe_OOD_panels": severe,
        "abstained_panels": severe,
        "descriptive_nonsevere_panels": len(panels) - severe,
        "accepted_direction_calls": 0,
        "direction_accuracy_estimate": None,
        "gate_accuracy_estimate": None,
        "guardrail": (
            "Two drainage manipulations share SGS and each contributes standard/strict views; "
            "four rows are not four independent validations or an accuracy denominator."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    for population in FILES:
        parser.add_argument(f"--{population.lower()}")
    parser.add_argument("--format-script")
    parser.add_argument("--source-dir")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()
    if args.cap < max(value[0] for value in EXPECTED_PANEL_LOCI.values()):
        parser.error(f"--cap must be at least {max(value[0] for value in EXPECTED_PANEL_LOCI.values())}")

    set_below_normal_priority()
    cache = Path(args.cache_dir).resolve()
    result_dir = Path(args.result_dir).resolve()
    source_dir = Path(args.source_dir).resolve() if args.source_dir else cache
    cache.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        population: (
            Path(getattr(args, population.lower())).resolve()
            if getattr(args, population.lower())
            else source_dir / spec["key"]
        )
        for population, spec in FILES.items()
    }
    format_path = (
        Path(args.format_script).resolve()
        if args.format_script
        else source_dir / FORMAT_SCRIPT["key"]
    )
    verified = {
        population: ensure_source(paths[population], spec, args.download_missing)
        for population, spec in FILES.items()
    }
    verified["format_script"] = ensure_source(format_path, FORMAT_SCRIPT, args.download_missing)
    source_record = validate_sources_record()
    format_audit = audit_format_script(format_path)
    combined_vcf = cache / "guppy_2020.combined_GT.vcf"
    combined_audit = materialize_combined_vcf(paths, combined_vcf)
    manifests, manifest_audit = materialize_manifests(cache)
    direction_head = simulation_direction_head(Path(args.data_root).resolve(), max_depth=MAX_DEPTH)
    gate_head = simulation_gate_head(Path(args.data_root).resolve(), max_depth=MAX_DEPTH)
    panels = run_panels(combined_vcf, manifests, cache, args.cap, direction_head, gate_head)
    result = {
        "schema_version": "dnnaic-guppy-2020-external-benchmark-v1",
        "git": git_revision(),
        "runtime": runtime_audit(),
        "guardrail": (
            "Two experimentally directed recipient drainages sharing an SGS proxy, each scored "
            "under two correlated locus filters. The source repository has no explicit license; "
            "raw bytes are runtime-only. No formal accuracy or gate truth is claimed."
        ),
        "source": {
            "repository": REPOSITORY,
            "repository_commit": COMMIT,
            "repository_tree_metadata_pin": TREE,
            "repository_tree_verification": (
                "metadata pin from the source audit; runtime verifies each fetched payload by "
                "byte count, SHA-256, and computed Git blob SHA-1 but does not reconstruct the tree"
            ),
            "repository_license": None,
            "data_use_policy": "runtime fetch only; do not vendor source bytes or author code",
            "paper": PAPER_URL,
            "paper_doi": "10.1016/j.cub.2019.11.062",
            "paper_license": "CC-BY-NC-ND-4.0",
            "original_experiment": ORIGINAL_EXPERIMENT_URL,
            "original_experiment_doi": "10.1111/eva.12356",
            "verified_files": verified,
            "sources_record": source_record,
            "format_script_audit": format_audit,
            "combined_VCF_audit": combined_audit,
            "paper_release_discrepancy": (
                "paper reports 12,407 SNPs, while each of the five runtime-verified VCFs contains 11,417"
            ),
            "locus_ascertainment_guardrail": (
                "the author release was filtered across pre/post populations and both runner "
                "filters inspect post-flow P2; direction is experimentally SNP-independent, but "
                "this is not prospective held-out validation"
            ),
        },
        "analysis_design": {
            "panels": {name: spec["groups"] for name, spec in PANEL_SPECS.items()},
            "sample_counts": {name: spec["counts"] for name, spec in PANEL_SPECS.items()},
            "manifests": manifest_audit,
            "candidate_class": "C",
            "experimental_flow_direction_available": True,
            "exclusive_single_edge_truth_available": False,
            "formal_direction_accuracy_eligible": False,
            "gate_truth_available": False,
            "ecological_recipient_units": 2,
            "shared_source_proxy": "SGS",
            "strict_locus_counts": {"caigual": 30, "taylor": 22},
            "release_locus_ascertainment_outcome_blind": False,
            "benchmark_locus_filters_outcome_blind": False,
        },
        "direction_head": direction_head[2],
        "gate_head": gate_head[2],
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
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
