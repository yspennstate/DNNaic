#!/usr/bin/env python3
"""Run additional positive and null external DNNaic diagnostics.

The panels in this script deliberately complement the first external bundle:

* Kenyan giraffe is a genome-wide positive benchmark with an author-supplied
  Dsuite statistic and a direction supported by the fitted demographic graph.
* Rhode Island brook trout supplies two hatchery-versus-wild specificity
  stress tests from a study that found no significant captive-to-wild
  introgression overall.

All source VCFs remain outside Git.  The script checks immutable source hashes,
uses committed sample manifests, applies the same biallelic/call-depth/locus-cap
contract as ``external_benchmarks.py``, and fits both heads on the simulation
data using exactly the external panels' g=2..16 depth grid.  Scores are
uncalibrated out-of-distribution diagnostics, not biological posteriors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from external_benchmarks import (
    CLASSES,
    MANIFEST_DIR,
    MAX_DEPTH,
    REPO,
    git_revision,
    prepare_vcf,
    score_panel,
    set_below_normal_priority,
    sha256_file,
    simulation_direction_head,
    subset_prepared_vcf,
    verify_file,
)


DEFAULT_CACHE = REPO / "data" / "real" / "additional_external_benchmarks"
DEFAULT_RESULTS = REPO / "results" / "additional_external_benchmarks_2026_07_11"
DEFAULT_CAP = 15_000
APPRECIABLE = 2.5e-4

GIRAFFE = {
    "bytes": 1_334_251_753,
    "sha256": "7e7f4345df0129329f99db5f05e7cd120d86e0027d3afa1bc7df80c457fb95b1",
    "archive_bytes": 169_856_604,
    "archive_md5": "cdb1089460fb7aa8ca4bd8bf525d8a37",
    "archive_sha256": "531533164589477edecca055b1ff77afe1041e45f393841ee11bda82eb41e30d",
    "record": "https://zenodo.org/records/8381750",
    "download": "https://zenodo.org/api/records/8381750/files/dsuite_introgression.tar.gz/content",
    "data_doi": "10.5281/zenodo.8381750",
    "paper_doi": "10.1186/s12915-023-01722-y",
}

BROOK_TROUT = {
    "bytes": 240_467_834,
    "sha256": "ea3754560e62c9ae22c6d1ad988c75ce31a0d0382f7aa233217e41c1fac7b69c",
    "md5": "999484d17d08a0da16431762c002bff8",
    "record": "https://figshare.com/articles/dataset/28628804",
    "download": "https://ndownloader.figshare.com/files/53115473",
    "data_doi": "10.6084/m9.figshare.28628804.v1",
    "paper_doi": "10.1111/fwb.70033",
}


def depth_matched_gate_features(table: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Summarize one or more g=2..16 curves for a matched gate fit."""
    table = np.asarray(table, dtype=float)
    if table.ndim != 3 or table.shape[2] != 28:
        raise ValueError("expected (replicate, depth, 28) PADZE table")
    if table.shape[1] < 2:
        raise ValueError("at least two rarefaction depths are required")
    if not np.all(table[:, :, 0] == table[0, :, 0]):
        raise ValueError("replicates do not share one rarefaction-depth grid")
    indices = np.unique(
        np.round(np.geomspace(1, table.shape[1] - 1, 8)).astype(int)
    )
    curves = table[:, :, 1:]
    features = np.concatenate(
        [curves[:, indices, :].reshape(len(table), -1), curves.mean(axis=1)],
        axis=1,
    )
    return features, indices.astype(int).tolist()


def simulation_gate_head(data_root: Path, max_depth: int = MAX_DEPTH):
    """Fit an appreciable-vs-weak/control gate on the external depth grid."""
    directory = data_root / "regen_full"
    paths = {
        name: directory / name
        for name in ("X.npy", "direction.npy", "groups.npy", "magnitude.npy")
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    X = np.load(paths["X.npy"], mmap_mode="r")
    direction = np.load(paths["direction.npy"], mmap_mode="r").astype("U2")
    groups = np.load(paths["groups.npy"], mmap_mode="r").astype("U80")
    magnitude = np.load(paths["magnitude.npy"], mmap_mode="r").astype(float)
    unique, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    order = np.lexsort((np.asarray(X[:, 0]), inverse))
    per = np.bincount(inverse)
    if per.min() != 198 or per.max() != 198:
        raise AssertionError("regen_full curves must contain g=2..199")
    table = np.asarray(X[order]).reshape(len(unique), 198, X.shape[1])
    use = table[0, :, 0] <= max_depth
    expected_depths = np.arange(2, max_depth + 1, dtype=float)
    if not np.array_equal(table[0, use, 0], expected_depths):
        raise AssertionError(f"simulation depth grid does not contain g=2..{max_depth}")
    matched = table[:, use, :]
    features, indices = depth_matched_gate_features(matched)
    labels = direction[first]
    rates = magnitude[first]
    target = ((labels != "D") & (rates >= APPRECIABLE)).astype(int)
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs").fit(
        scaler.transform(features), target
    )
    audit = {
        "training_dataset": "regen_full",
        "training_target": "positive migration with rate >=2.5e-4 versus weak positives plus control",
        "training_n": int(len(target)),
        "training_class_counts": {
            "appreciable": int(target.sum()),
            "other": int((target == 0).sum()),
        },
        "depth_grid": expected_depths.astype(int).tolist(),
        "selected_depth_indices_zero_based": indices,
        "selected_depths": matched[0, indices, 0].astype(int).tolist(),
        "representation": "selected depth rows plus across-depth mean of 27 non-depth coordinates",
        "dimension": int(features.shape[1]),
        "model": "StandardScaler plus binary logistic regression (lbfgs, C=1)",
        "score_interpretation": "depth-matched uncalibrated OOD score; not the canonical 243-D gate probability",
        "fit_iterations": model.n_iter_.astype(int).tolist(),
        "training_array_audit": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    return scaler, model, audit


def add_gate_score(panel: dict, scaler, model) -> dict:
    matrix = np.asarray(panel["feature_matrix"], dtype=float)
    features, indices = depth_matched_gate_features(matrix[None, :, :])
    transformed = scaler.transform(features)
    score = float(model.predict_proba(transformed)[0, 1])
    z = transformed[0]
    panel["simulation_gate"] = {
        "appreciable_score": score,
        "called_at_0.5": bool(score >= 0.5),
        "selected_depth_indices_zero_based": indices,
        "interpretation": "depth-matched uncalibrated OOD score; not a probability or posterior",
    }
    panel["simulation_gate_feature_shift"] = {
        "rms_z": float(np.sqrt(np.mean(z**2))),
        "max_abs_z": float(np.max(np.abs(z))),
        "coordinates_abs_z_gt_3": int(np.sum(np.abs(z) > 3)),
        "coordinates_abs_z_gt_5": int(np.sum(np.abs(z) > 5)),
        "z_values": z.tolist(),
    }
    return panel


def run_giraffe(vcf: Path, cache: Path, cap: int, direction_head, gate_head):
    source = verify_file(vcf, GIRAFFE["bytes"], GIRAFFE["sha256"])
    manifest = MANIFEST_DIR / "giraffe_nubian_reticulated.tsv"
    scaler, model, _ = direction_head
    gate_scaler, gate_model, _ = gate_head
    panels = []
    for suffix, strict in (
        ("standard_contract", False),
        ("within_population_polymorphism", True),
    ):
        derived_vcf = cache / f"giraffe.{suffix}.filtered.vcf"
        derived_popmap = cache / f"giraffe.{suffix}.filtered.popmap.tsv"
        audit = prepare_vcf(
            vcf,
            manifest,
            derived_vcf,
            derived_popmap,
            cap=cap,
            seed=20260711,
            polymorphic_within_each_population=strict,
        )
        panel = score_panel(
            f"giraffe_nubian_to_laikipia_reticulated_{suffix}",
            derived_vcf,
            derived_popmap,
            ("Reticulated_14-18", "Reticulated_8-13", "Nubian_3"),
            audit,
            scaler,
            model,
            {
                "expected_class": "C",
                "expected_forward_direction": "Nubian_3 (P3) -> Reticulated_8-13 (P2)",
                "published_D": 0.211001,
                "published_Z": 25.4689,
                "published_p": 2.3e-16,
                "published_f4_ratio": 0.199169,
                "direction_basis": "OrientAGraph migration edge and asymmetric Nubian-ancestry analysis; D alone is not directional",
                "guardrail": "ancient natural event and OOD; contemporary migration was not significant",
                "locus_filter_variant": (
                    "stricter robustness panel: both alleles observed within each population"
                    if strict
                    else "standard external contract: both alleles observed across the complete trio"
                ),
            },
        )
        panels.append(add_gate_score(panel, gate_scaler, gate_model))
    return panels, source


def run_brook_trout(vcf: Path, cache: Path, cap: int, direction_head, gate_head):
    source = verify_file(vcf, BROOK_TROUT["bytes"], BROOK_TROUT["sha256"])
    shared_manifest = MANIFEST_DIR / "brook_trout_shared.tsv"
    panel_manifests = (
        MANIFEST_DIR / "brook_trout_lfa_null.tsv",
        MANIFEST_DIR / "brook_trout_lfr_null.tsv",
    )
    shared_vcf = cache / "brook_trout.shared.filtered.vcf"
    shared_popmap = cache / "brook_trout.shared.filtered.popmap.tsv"
    shared_audit = prepare_vcf(
        vcf,
        shared_manifest,
        shared_vcf,
        shared_popmap,
        cap=cap,
        seed=20260711,
        require_three_populations=False,
        polymorphic_panel_manifests=panel_manifests,
    )
    panels = []
    scaler, model, _ = direction_head
    gate_scaler, gate_model, _ = gate_head
    for donor, manifest in zip(("LFA", "LFR"), panel_manifests):
        panel_vcf = cache / f"brook_trout.{donor.lower()}.filtered.vcf"
        panel_popmap = cache / f"brook_trout.{donor.lower()}.filtered.popmap.tsv"
        audit = subset_prepared_vcf(
            shared_vcf, manifest, panel_vcf, panel_popmap, shared_audit
        )
        panel = score_panel(
            f"brook_trout_{donor.lower()}_to_baker_brook_null",
            panel_vcf,
            panel_popmap,
            ("AFP", "BAK", donor),
            audit,
            scaler,
            model,
            {
                "expected_gate": "no appreciable hatchery-to-wild event",
                "candidate_forward_direction_if_present": f"{donor} (P3) -> BAK (P2), class C",
                "study_result": "no significant introgression from captive-bred individuals overall",
                "panel_context": "AFP is non-stocked; BAK is stocked; LFA/LFR are the two hatchery strains",
                "guardrail": "specificity stress test, not a site-specific D=0 gold standard; one UTA individual showed hatchery ancestry",
            },
        )
        panels.append(add_gate_score(panel, gate_scaler, gate_model))
    return panels, source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="directory containing regen_full")
    parser.add_argument("--giraffe-vcf", help="extracted snps.sampled.vcf")
    parser.add_argument("--brook-vcf", help="BT_ALL.vcf")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULTS))
    parser.add_argument("--cap", type=int, default=DEFAULT_CAP)
    args = parser.parse_args()
    if not args.giraffe_vcf and not args.brook_vcf:
        parser.error("provide --giraffe-vcf and/or --brook-vcf")
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
        "schema_version": "dnnaic-additional-external-benchmarks-v1",
        "git": git_revision(),
        "guardrail": (
            "All scores are explicitly OOD diagnostics. Giraffe is a directional positive "
            "benchmark; brook trout is a study-level null/specificity stress test."
        ),
        "direction_head": direction_head[2],
        "gate_head": gate_head[2],
        "sources": {},
        "panels": [],
    }
    if args.giraffe_vcf:
        panels, source = run_giraffe(
            Path(args.giraffe_vcf).resolve(), cache, args.cap, direction_head, gate_head
        )
        result["panels"].extend(panels)
        result["sources"]["giraffe"] = {**GIRAFFE, "verified_file": source}
    if args.brook_vcf:
        panels, source = run_brook_trout(
            Path(args.brook_vcf).resolve(), cache, args.cap, direction_head, gate_head
        )
        result["panels"].extend(panels)
        result["sources"]["brook_trout"] = {**BROOK_TROUT, "verified_file": source}

    output = result_dir / "results.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "panels": [
            {
                "panel_id": panel["panel_id"],
                "direction": panel["simulation_head"]["predicted_class"],
                "gate": panel["simulation_gate"]["appreciable_score"],
                "rms_z": panel["simulation_feature_shift"]["rms_z"],
                "loci": panel["padze"]["n_loci_kept"],
            }
            for panel in result["panels"]
        ],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
