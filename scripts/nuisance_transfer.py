#!/usr/bin/env python3
"""CPU-only held-out nuisance-transfer grid for the DNNaic 54-D direction model.

Five explicitly parameterized simulation families are crossed with two exact
migration rates, three backwards-time directions, and independent replicates.
The canonical all-positive logistic model is fit and frozen *before* any new
replicate is scored.  A canonical-appreciable fit is retained as a sensitivity
analysis.  The only within-family fitting is repeated stratified five-fold OOF
evaluation, clearly labeled as a diagnostic rather than external transfer.

The simulation checkpoint is compact and resumable: only one 54-D mean+SD
PADZE aggregate per replicate is retained.  No GPU library or background
process is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import msprime
import numpy as np
import padze
import sklearn
from padze import LociData, Metadata, compute_features
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Reuse the already validated canonical aggregation/model machinery without
# changing it.  This script supplies only the nuisance-demography layer.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import matched_exposure as validated  # noqa: E402


SCHEMA_VERSION = "nuisance-transfer-v2-exact-prespecified-grid"
NE_CANONICAL = 10_000
SEQUENCE_LENGTH = 1_000_000
RECOMBINATION_RATE = 1.78e-8
MUTATION_RATE = 2e-8
RATES = np.array([5e-5, 2.5e-4], dtype=np.float64)
CLASSES = np.array(["A", "B", "C"])
CLASS_MIGRATION = {
    "A": ("P2", "P1"),
    "B": ("P3", "P2"),
    "C": ("P2", "P3"),
}
MIGRATION_END = 500
MOMENTS = ("mean", "variance", "se")
CURVE_COLUMNS = list(validated.CURVE_COLUMNS)
FEATURE_COLUMNS = list(validated.FEATURE_COLUMNS)
APPRECIABLE = validated.APPRECIABLE

POPULATION_ORDER = ("P1", "P2", "P3", "P4", "P12", "P123", "P1234")
CANONICAL_SIZES = {name: NE_CANONICAL for name in POPULATION_ORDER}

# Values are intentionally concrete and material, but each family is only one
# perturbation—not a distribution over demographic uncertainty.
FAMILIES = {
    "canonical_positive_control": {
        "description": (
            "Current canonical msprime/PADZE generator: haploid coalescent size "
            "10,000 throughout (diploid equivalent 5,000), splits 500/1000/2000, "
            "200 haploid gene copies, depths 2..199."
        ),
        "population_initial_sizes": dict(CANONICAL_SIZES),
        "size_changes": (),
        "split_times": (500, 1_000, 2_000),
        "gene_copies_per_population": 200,
        "depth_min": 2,
        "depth_max": 199,
    },
    "unequal_descendant_ne": {
        "description": (
            "Prespecified population-specific haploid coalescent sizes: P1=6,000, "
            "P2=14,000, P3=24,000, P4=12,000, P12=10,000, P123=16,000, "
            "P1234=18,000 (diploid equivalents are one half of each value)."
        ),
        "population_initial_sizes": {
            "P1": 6_000, "P2": 14_000, "P3": 24_000, "P4": 12_000,
            "P12": 10_000, "P123": 16_000, "P1234": 18_000,
        },
        "size_changes": (),
        "split_times": (500, 1_000, 2_000),
        "gene_copies_per_population": 200,
        "depth_min": 2,
        "depth_max": 199,
    },
    "deeper_splits": {
        "description": (
            "Prespecified composite deep-history family: all haploid coalescent "
            "sizes=20,000 (diploid equivalent 10,000) and splits 800/1600/3200 "
            "generations. Migration is explicitly switched off at t=500. This "
            "changes both population size and split depth."
        ),
        "population_initial_sizes": {name: 20_000 for name in POPULATION_ORDER},
        "size_changes": (),
        "split_times": (800, 1_600, 3_200),
        "gene_copies_per_population": 200,
        "depth_min": 2,
        "depth_max": 199,
    },
    "recent_p2_bottleneck": {
        "description": (
            "Prespecified staggered recent bottlenecks on a canonical haploid "
            "coalescent-size 10,000 background (diploid equivalent 5,000): P2 "
            "size=2,500 (diploid 1,250) during t=100..300 and P3 size=4,000 "
            "(diploid 2,000) during t=150..350 generations backwards in time, "
            "returning to 10,000 outside those intervals. Canonical splits."
        ),
        "population_initial_sizes": dict(CANONICAL_SIZES),
        "size_changes": (
            (100, "P2", 2_500), (150, "P3", 4_000),
            (300, "P2", 10_000), (350, "P3", 10_000),
        ),
        "split_times": (500, 1_000, 2_000),
        "gene_copies_per_population": 200,
        "depth_min": 2,
        "depth_max": 199,
    },
    "half_sample_depth": {
        "description": (
            "Canonical haploid-coalescent-size demography and rates with 100 "
            "haploid gene copies per P1/P2/P3 and the supported PADZE depth grid "
            "2..99; aggregation remains 54-D."
        ),
        "population_initial_sizes": dict(CANONICAL_SIZES),
        "size_changes": (),
        "split_times": (500, 1_000, 2_000),
        "gene_copies_per_population": 100,
        "depth_min": 2,
        "depth_max": 99,
    },
}

CONFIG_PAYLOAD = {
    "schema_version": SCHEMA_VERSION,
    "families": FAMILIES,
    "rates": RATES.tolist(),
    "class_migration": CLASS_MIGRATION,
    "migration_end": MIGRATION_END,
    "sequence_length": SEQUENCE_LENGTH,
    "recombination_rate": RECOMBINATION_RATE,
    "mutation_rate": MUTATION_RATE,
    "feature_columns": FEATURE_COLUMNS,
}
CONFIG_SHA256 = hashlib.sha256(
    json.dumps(CONFIG_PAYLOAD, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()
# The first complete 600-row checkpoint used the same numerical design but an
# earlier prose-only description fingerprint. Accept it once and rewrite the
# archive with the release fingerprint on the next save.
COMPATIBLE_CONFIG_SHA256 = {
    CONFIG_SHA256,
    "7503d6f7585ea68da52cc2dd4cf54bf5d2fc28e407332014bc165d69c96fcf6f",
}

FIXED_CURVE_RATES = np.array([5e-7, 2.5e-6, 5e-5, 2.5e-4], dtype=np.float64)
CURVE_SCHEMA_VERSION = "current-engine-disjoint-stream-fixed-rate-v1"
CURVE_CONFIG_SHA256 = hashlib.sha256(json.dumps({
    "schema_version": CURVE_SCHEMA_VERSION,
    "canonical_family_numerical_spec": {
        key: value for key, value in FAMILIES["canonical_positive_control"].items()
        if key != "description"
    },
    "rates": FIXED_CURVE_RATES.tolist(),
    "directions": CLASSES.tolist(),
    "reps_per_rate_direction": 30,
    "ancestry_and_mutation_streams": "distinct integer seeds",
}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def make_demography(family: str, label: str, rate: float) -> msprime.Demography:
    spec = FAMILIES[family]
    d = msprime.Demography()
    for name in POPULATION_ORDER:
        d.add_population(name=name, initial_size=spec["population_initial_sizes"][name])
    for event_time, population, size in spec["size_changes"]:
        d.add_population_parameters_change(
            time=event_time, population=population, initial_size=size
        )
    split_p12, split_p123, split_root = spec["split_times"]
    d.add_population_split(time=split_p12, derived=["P1", "P2"], ancestral="P12")
    d.add_population_split(time=split_p123, derived=["P12", "P3"], ancestral="P123")
    d.add_population_split(time=split_root, derived=["P123", "P4"], ancestral="P1234")
    source, dest = CLASS_MIGRATION[label]
    d.set_migration_rate(source=source, dest=dest, rate=float(rate))
    # Canonical split at 500 deactivates P1/P2 and automatically zeros every
    # requested mapping.  For a deeper first split we must stop it explicitly.
    if split_p12 > MIGRATION_END:
        d.add_migration_rate_change(
            time=MIGRATION_END, source=source, dest=dest, rate=0.0
        )
    d.sort_events()
    return d


def audit_demographies() -> dict:
    """Record actual epoch matrices and prove every design has exposure rate*500."""
    audit = {}
    population_names = list(POPULATION_ORDER)
    for family, spec in FAMILIES.items():
        family_audit = {
            "specification": {
                "description": spec["description"],
                "population_initial_sizes_haploid_coalescent": spec["population_initial_sizes"],
                "population_initial_sizes_diploid_equivalent": {
                    name: size / 2 for name, size in spec["population_initial_sizes"].items()
                },
                "size_changes_backward_time_haploid_coalescent": [
                    list(change) for change in spec["size_changes"]
                ],
                "size_changes_backward_time_diploid_equivalent": [
                    [event_time, population, size / 2]
                    for event_time, population, size in spec["size_changes"]
                ],
                "split_times_generations": list(spec["split_times"]),
                "gene_copies_per_sampled_population": spec["gene_copies_per_population"],
                "depths_inclusive": [spec["depth_min"], spec["depth_max"]],
            },
            "rate_direction_audits": {},
        }
        for rate in RATES:
            rate_key = f"{rate:.1e}"
            family_audit["rate_direction_audits"][rate_key] = {}
            for label in CLASSES:
                d = make_demography(family, str(label), float(rate))
                epochs = []
                integrated_hazard = 0.0
                nonzero_duration = 0.0
                migration_intervals = []
                for epoch in d.debug().epochs:
                    start = float(epoch.start_time)
                    end = float(epoch.end_time)
                    finite_end = math.isfinite(end)
                    matrix = np.asarray(epoch.migration_matrix, dtype=float)
                    nonzero = np.argwhere(matrix > 0)
                    entries = []
                    for source_index, dest_index in nonzero:
                        value = float(matrix[source_index, dest_index])
                        source = population_names[int(source_index)]
                        dest = population_names[int(dest_index)]
                        entries.append({"source": source, "dest": dest, "rate": value})
                        if finite_end:
                            duration = end - start
                            integrated_hazard += duration * value
                            nonzero_duration += duration
                    populations = []
                    for pop in epoch.populations:
                        populations.append({
                            "name": pop.name,
                            "active": bool(pop.active),
                            "start_size_haploid_coalescent": float(pop.start_size),
                            "end_size_haploid_coalescent": float(pop.end_size),
                            "start_size_diploid_equivalent": float(pop.start_size) / 2,
                            "end_size_diploid_equivalent": float(pop.end_size) / 2,
                        })
                    active_names = {
                        population["name"] for population in populations if population["active"]
                    }
                    for entry in entries:
                        if entry["source"] not in active_names or entry["dest"] not in active_names:
                            raise AssertionError(
                                f"{family}/{rate_key}/{label}: migration touches inactive "
                                f"population in epoch {start}..{end}: {entry}"
                            )
                        migration_intervals.append((start, end))
                    epochs.append({
                        "start": start,
                        "end": end if finite_end else None,
                        "populations": populations,
                        "nonzero_migration": entries,
                        "migration_matrix_population_order": population_names,
                        "migration_matrix": matrix.tolist(),
                    })

                requested_source, requested_dest = CLASS_MIGRATION[str(label)]
                for epoch in epochs:
                    for entry in epoch["nonzero_migration"]:
                        if (entry["source"], entry["dest"]) != (
                            requested_source, requested_dest
                        ):
                            raise AssertionError(
                                f"{family}/{rate_key}/{label} unexpected mapping {entry}"
                            )
                        if epoch["end"] is None or epoch["end"] > MIGRATION_END:
                            raise AssertionError(
                                f"{family}/{rate_key}/{label} migration persists beyond t=500"
                            )
                migration_intervals.sort()
                cursor = 0.0
                for start, end in migration_intervals:
                    if not np.isclose(start, cursor, rtol=0, atol=1e-12):
                        raise AssertionError(
                            f"{family}/{rate_key}/{label}: migration interval gap/overlap at "
                            f"{cursor}, next={start}..{end}"
                        )
                    cursor = end
                if not np.isclose(cursor, MIGRATION_END, rtol=0, atol=1e-12):
                    raise AssertionError(
                        f"{family}/{rate_key}/{label}: migration coverage ends at {cursor}"
                    )
                expected_hazard = float(rate) * MIGRATION_END
                if not np.isclose(integrated_hazard, expected_hazard, rtol=0, atol=1e-12):
                    raise AssertionError(
                        f"{family}/{rate_key}/{label}: hazard {integrated_hazard} "
                        f"!= {expected_hazard}"
                    )
                if not np.isclose(nonzero_duration, MIGRATION_END, rtol=0, atol=1e-12):
                    raise AssertionError(
                        f"{family}/{rate_key}/{label}: duration {nonzero_duration} != 500"
                    )
                family_audit["rate_direction_audits"][rate_key][str(label)] = {
                    "backward_mapping": f"{requested_source}->{requested_dest}",
                    "verified_nonzero_duration_generations": nonzero_duration,
                    "verified_integrated_single_lineage_hazard": integrated_hazard,
                    "conditional_single_lineage_probability_at_least_one_migration": (
                        -math.expm1(-integrated_hazard)
                    ),
                    "epochs": epochs,
                }
        audit[family] = family_audit
    return audit


def tree_sequence_to_loci(ts, gene_copies: int) -> LociData:
    populations = ["P1", "P2", "P3"]
    samples = ts.samples()
    sample_population = ts.tables.nodes.population[samples]
    masks = [sample_population == index for index in range(3)]
    counts_per_population = np.array([int(mask.sum()) for mask in masks])
    expected = np.full(3, gene_copies)
    if not np.array_equal(counts_per_population, expected):
        raise AssertionError(
            f"haploid samples {counts_per_population.tolist()} != {expected.tolist()}"
        )

    genotype = ts.genotype_matrix()
    count_matrices = []
    sizes = []
    if genotype.shape[0]:
        for row in genotype:
            maximum = int(row.max())
            if maximum < 1:
                continue
            matrix = np.stack([
                np.bincount(row[mask], minlength=maximum + 1) for mask in masks
            ]).astype(np.int64, copy=False)
            if int((matrix.sum(axis=0) > 0).sum()) < 2:
                continue
            count_matrices.append(matrix)
            sizes.append(matrix.sum(axis=1))
    sample_sizes = (
        np.vstack(sizes).astype(np.int64, copy=False)
        if sizes else np.zeros((0, 3), dtype=np.int64)
    )
    metadata = Metadata(
        source="msprime",
        populations=populations,
        sample_ids={population: [] for population in populations},
        ploidy={population: 1 for population in populations},
        n_loci_read=int(ts.num_sites),
        n_loci_kept=len(count_matrices),
        filters_applied=["polymorphic across P1/P2/P3"],
        missing_fraction=0.0,
    )
    return LociData(
        populations,
        count_matrices,
        sample_sizes,
        [f"s{i}" for i in range(len(count_matrices))],
        metadata,
    )


def simulate_feature(job: tuple[str, int, float, str, int, int, int]) -> dict:
    family, rate_index, rate, label, replicate, ancestry_seed, mutation_seed = job
    spec = FAMILIES[family]
    gene_copies = int(spec["gene_copies_per_population"])
    depths = np.arange(spec["depth_min"], spec["depth_max"] + 1, dtype=np.int64)
    started = time.perf_counter()
    ts = msprime.sim_ancestry(
        samples={"P1": gene_copies, "P2": gene_copies, "P3": gene_copies},
        sequence_length=SEQUENCE_LENGTH,
        recombination_rate=RECOMBINATION_RATE,
        ploidy=1,
        demography=make_demography(family, label, rate),
        random_seed=ancestry_seed,
    )
    ts = msprime.sim_mutations(
        ts, rate=MUTATION_RATE, random_seed=mutation_seed, keep=True
    )
    loci = tree_sequence_to_loci(ts, gene_copies)
    feature_table = compute_features(
        loci,
        depths=depths,
        pihat_sizes=(2,),
        moments=MOMENTS,
        bias_corrected=True,
    )
    matrix, columns = feature_table.to_frame()
    index = {column: i for i, column in enumerate(columns)}
    try:
        matrix = matrix[:, [index[column] for column in CURVE_COLUMNS]].astype(np.float64)
    except KeyError as exc:
        raise RuntimeError(f"PADZE contract changed; missing {exc}") from exc
    if matrix.shape != (len(depths), 28):
        raise AssertionError(
            f"{family}: PADZE shape {matrix.shape} != ({len(depths)}, 28)"
        )
    if not np.array_equal(matrix[:, 0], depths):
        raise AssertionError(f"{family}: PADZE depth grid mismatch")
    curve = matrix[:, 1:]
    feature = np.concatenate([curve.mean(axis=0), curve.std(axis=0)], axis=0)
    if feature.shape != (54,) or not np.isfinite(feature).all():
        raise AssertionError(f"{family}: invalid 54-D aggregate")
    return {
        "family": family,
        "rate_index": int(rate_index),
        "rate": float(rate),
        "label": label,
        "replicate": int(replicate),
        "ancestry_seed": int(ancestry_seed),
        "mutation_seed": int(mutation_seed),
        "gene_copies": gene_copies,
        "depth_min": int(depths[0]),
        "depth_max": int(depths[-1]),
        "num_sites": int(ts.num_sites),
        "num_loci": int(loci.metadata.n_loci_kept),
        "elapsed_seconds": float(time.perf_counter() - started),
        "feature": feature.astype(np.float32),
    }


def make_jobs(reps: int, seed_base: int) -> list[tuple[str, int, float, str, int, int, int]]:
    """Round-robin ordering keeps a partial checkpoint balanced across families."""
    jobs = []
    for replicate in range(reps):
        for family_index, family in enumerate(FAMILIES):
            for rate_index, rate in enumerate(RATES):
                for class_index, label in enumerate(CLASSES):
                    ancestry_seed = (
                        seed_base
                        + family_index * 1_000_000
                        + rate_index * 100_000
                        + class_index * 10_000
                        + replicate * 2
                    )
                    mutation_seed = ancestry_seed + 1
                    if not (0 < ancestry_seed < 2**32 and mutation_seed < 2**32):
                        raise ValueError("msprime seed outside [1, 2**32-1]")
                    jobs.append((
                        family, rate_index, float(rate), str(label), replicate,
                        ancestry_seed, mutation_seed,
                    ))
    return jobs


def record_key(record: dict) -> tuple[str, int, str, int]:
    return record["family"], int(record["rate_index"]), record["label"], record["replicate"]


def sorted_records(records: list[dict]) -> list[dict]:
    family_index = {family: index for index, family in enumerate(FAMILIES)}
    return sorted(
        records,
        key=lambda record: (
            family_index[record["family"]], record["rate"],
            record["label"], record["replicate"],
        ),
    )


def record_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    records = sorted_records(records)
    return {
        "X": np.vstack([record["feature"] for record in records]).astype(np.float32),
        "family": np.array([record["family"] for record in records], dtype="U40"),
        "provenance": np.array(
            [record.get("provenance", "new_nuisance_simulation") for record in records],
            dtype="U60",
        ),
        "rate_index": np.array([record["rate_index"] for record in records], dtype=np.int8),
        "rate": np.array([record["rate"] for record in records], dtype=np.float64),
        "labels": np.array([record["label"] for record in records], dtype="U1"),
        "replicate": np.array([record["replicate"] for record in records], dtype=np.int32),
        "ancestry_seed": np.array(
            [record["ancestry_seed"] for record in records], dtype=np.int64
        ),
        "mutation_seed": np.array(
            [record["mutation_seed"] for record in records], dtype=np.int64
        ),
        "gene_copies": np.array([record["gene_copies"] for record in records], dtype=np.int32),
        "depth_min": np.array([record["depth_min"] for record in records], dtype=np.int32),
        "depth_max": np.array([record["depth_max"] for record in records], dtype=np.int32),
        "num_sites": np.array([record["num_sites"] for record in records], dtype=np.int32),
        "num_loci": np.array([record["num_loci"] for record in records], dtype=np.int32),
        "elapsed_seconds": np.array(
            [record["elapsed_seconds"] for record in records], dtype=np.float64
        ),
    }


def save_archive(
    records: list[dict],
    path: Path,
    all_positive_probability: np.ndarray | None = None,
    appreciable_probability: np.ndarray | None = None,
    within_family_probability: np.ndarray | None = None,
    frozen_binary_bc_probability: np.ndarray | None = None,
    within_family_binary_bc_probability: np.ndarray | None = None,
) -> None:
    arrays = record_arrays(records)
    arrays.update({
        "schema_version": np.array(SCHEMA_VERSION),
        "config_sha256": np.array(CONFIG_SHA256),
        "feature_columns": np.array(FEATURE_COLUMNS, dtype="U60"),
        "classes_probability_order": CLASSES,
    })
    if all_positive_probability is not None:
        arrays["frozen_all_positive_probability"] = all_positive_probability.astype(np.float32)
    if appreciable_probability is not None:
        arrays["frozen_appreciable_probability"] = appreciable_probability.astype(np.float32)
    if within_family_probability is not None:
        arrays["within_family_oof_probability"] = within_family_probability.astype(np.float32)
    if frozen_binary_bc_probability is not None:
        arrays["frozen_binary_bc_probability_C_nan_for_A"] = (
            frozen_binary_bc_probability.astype(np.float32)
        )
    if within_family_binary_bc_probability is not None:
        arrays["within_family_binary_bc_oof_probability_C_nan_for_A"] = (
            within_family_binary_bc_probability.astype(np.float32)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_archive(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"].item()) != SCHEMA_VERSION:
            raise RuntimeError("nuisance checkpoint schema mismatch")
        if str(archive["config_sha256"].item()) not in COMPATIBLE_CONFIG_SHA256:
            raise RuntimeError("nuisance checkpoint configuration fingerprint mismatch")
        if list(archive["feature_columns"].astype(str)) != FEATURE_COLUMNS:
            raise RuntimeError("nuisance checkpoint feature contract mismatch")
        records = []
        for i in range(len(archive["family"])):
            records.append({
                "family": str(archive["family"][i]),
                "provenance": (
                    str(archive["provenance"][i])
                    if "provenance" in archive.files else "new_nuisance_simulation"
                ),
                "rate_index": int(archive["rate_index"][i]),
                "rate": float(archive["rate"][i]),
                "label": str(archive["labels"][i]),
                "replicate": int(archive["replicate"][i]),
                "ancestry_seed": int(archive["ancestry_seed"][i]),
                "mutation_seed": int(archive["mutation_seed"][i]),
                "gene_copies": int(archive["gene_copies"][i]),
                "depth_min": int(archive["depth_min"][i]),
                "depth_max": int(archive["depth_max"][i]),
                "num_sites": int(archive["num_sites"][i]),
                "num_loci": int(archive["num_loci"][i]),
                "elapsed_seconds": float(archive["elapsed_seconds"][i]),
                "feature": archive["X"][i].astype(np.float32),
            })
    return records


def save_curve_archive(
    records: list[dict],
    path: Path,
    frozen_threeway_probability: np.ndarray | None = None,
    frozen_binary_bc_probability: np.ndarray | None = None,
    within_curve_threeway_probability: np.ndarray | None = None,
    within_curve_binary_bc_probability: np.ndarray | None = None,
) -> None:
    arrays = record_arrays(records)
    arrays.update({
        "schema_version": np.array(CURVE_SCHEMA_VERSION),
        "config_sha256": np.array(CURVE_CONFIG_SHA256),
        "feature_columns": np.array(FEATURE_COLUMNS, dtype="U60"),
        "classes_probability_order": CLASSES,
    })
    if frozen_threeway_probability is not None:
        arrays["frozen_legacy_threeway_probability"] = frozen_threeway_probability.astype(
            np.float32
        )
    if frozen_binary_bc_probability is not None:
        arrays["frozen_legacy_binary_bc_probability_C_nan_for_A"] = (
            frozen_binary_bc_probability.astype(np.float32)
        )
    if within_curve_threeway_probability is not None:
        arrays["within_curve_threeway_oof_probability"] = (
            within_curve_threeway_probability.astype(np.float32)
        )
    if within_curve_binary_bc_probability is not None:
        arrays["within_curve_binary_bc_oof_probability_C_nan_for_A"] = (
            within_curve_binary_bc_probability.astype(np.float32)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_curve_archive(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"].item()) != CURVE_SCHEMA_VERSION:
            raise RuntimeError("independent-stream curve checkpoint schema mismatch")
        if str(archive["config_sha256"].item()) != CURVE_CONFIG_SHA256:
            raise RuntimeError("independent-stream curve configuration fingerprint mismatch")
        if list(archive["feature_columns"].astype(str)) != FEATURE_COLUMNS:
            raise RuntimeError("independent-stream curve feature contract mismatch")
        records = []
        for i in range(len(archive["family"])):
            records.append({
                "family": str(archive["family"][i]),
                "provenance": str(archive["provenance"][i]),
                "rate_index": int(archive["rate_index"][i]),
                "rate": float(archive["rate"][i]),
                "label": str(archive["labels"][i]),
                "replicate": int(archive["replicate"][i]),
                "ancestry_seed": int(archive["ancestry_seed"][i]),
                "mutation_seed": int(archive["mutation_seed"][i]),
                "gene_copies": int(archive["gene_copies"][i]),
                "depth_min": int(archive["depth_min"][i]),
                "depth_max": int(archive["depth_max"][i]),
                "num_sites": int(archive["num_sites"][i]),
                "num_loci": int(archive["num_loci"][i]),
                "elapsed_seconds": float(archive["elapsed_seconds"][i]),
                "feature": archive["X"][i].astype(np.float32),
            })
    return records


def prepare_curve_records_and_jobs(
    nuisance_records: list[dict], seed_base: int
) -> tuple[list[dict], list[tuple[str, int, float, str, int, int, int]]]:
    reused = []
    for record in nuisance_records:
        if record["family"] != "canonical_positive_control":
            continue
        curve_record = dict(record)
        matches = np.where(np.isclose(FIXED_CURVE_RATES, record["rate"], rtol=0, atol=0))[0]
        if len(matches) != 1:
            raise AssertionError(f"upper-rate control is not on fixed curve: {record['rate']}")
        curve_record["rate_index"] = int(matches[0])
        curve_record["provenance"] = "reused_complete_nuisance_canonical_control"
        reused.append(curve_record)
    if len(reused) != 120:
        raise AssertionError(f"expected 120 reusable upper-rate rows, found {len(reused)}")

    jobs = []
    for rate_index, rate in enumerate(FIXED_CURVE_RATES):
        replicate_range = range(30) if rate_index < 2 else range(20, 30)
        for class_index, label in enumerate(CLASSES):
            for replicate in replicate_range:
                ancestry_seed = (
                    seed_base + rate_index * 100_000 + class_index * 10_000 + 2 * replicate
                )
                mutation_seed = ancestry_seed + 1
                jobs.append((
                    "canonical_positive_control", rate_index, float(rate), str(label),
                    replicate, ancestry_seed, mutation_seed,
                ))
    if len(jobs) != 240:
        raise AssertionError(f"expected 240 new curve jobs, found {len(jobs)}")
    return reused, jobs


def simulate_curve_missing(
    records: list[dict],
    jobs: list[tuple[str, int, float, str, int, int, int]],
    workers: int,
    archive_path: Path,
) -> list[dict]:
    completed = {record_key(record) for record in records}
    missing = [job for job in jobs if (job[0], job[1], job[3], job[4]) not in completed]
    print(
        f"[curve] {len(records)} checkpointed (including reused); {len(missing)} "
        f"new simulations missing; {workers} foreground CPU workers",
        flush=True,
    )
    if not missing:
        return records
    started = time.perf_counter()
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(simulate_feature, job): job for job in missing}
        for completed_new, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            try:
                record = future.result()
            except Exception as exc:
                errors.append((job, repr(exc)))
                print(f"[curve] ERROR {job}: {exc!r}", flush=True)
                continue
            record["provenance"] = "new_current_engine_disjoint_stream_simulation"
            records.append(record)
            save_curve_archive(records, archive_path)
            if completed_new % 5 == 0 or completed_new == len(missing):
                print(
                    f"[curve] {completed_new}/{len(missing)} new; total={len(records)}; "
                    f"rate={record['rate']:.1e} {record['label']}{record['replicate']:02d}; "
                    f"wall={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
    if errors:
        raise RuntimeError(f"{len(errors)} curve simulations failed; first={errors[0]}")
    return records


def simulate_missing(
    records: list[dict],
    jobs: list[tuple[str, int, float, str, int, int, int]],
    workers: int,
    archive_path: Path,
) -> list[dict]:
    completed = {record_key(record) for record in records}
    missing = [
        job for job in jobs if (job[0], int(job[1]), job[3], job[4]) not in completed
    ]
    print(
        f"[simulate] {len(records)} checkpointed; {len(missing)} missing; "
        f"{workers} foreground CPU workers",
        flush=True,
    )
    if not missing:
        return records
    started = time.perf_counter()
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(simulate_feature, job): job for job in missing}
        for completed_new, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            try:
                record = future.result()
            except Exception as exc:
                errors.append((job, repr(exc)))
                print(f"[simulate] ERROR {job}: {exc!r}", flush=True)
                continue
            records.append(record)
            save_archive(records, archive_path)
            print(
                f"[simulate] {completed_new}/{len(missing)} new; total={len(records)}; "
            f"{record['family']} {record['rate']:.1e} {record['label']}"
                f"{record['replicate']:02d}; loci={record['num_loci']}; "
                f"worker={record['elapsed_seconds']:.1f}s; "
                f"wall={time.perf_counter() - started:.1f}s",
                flush=True,
            )
    if errors:
        raise RuntimeError(f"{len(errors)} jobs failed; first={errors[0]}")
    return records


def encode(labels: np.ndarray) -> np.ndarray:
    return validated.encode(labels)


def fit_frozen_models(canonical_root: Path):
    X, labels, rates = validated.load_canonical(canonical_root)
    positive = labels != "D"
    appreciable = positive & (rates >= APPRECIABLE)
    all_scale, all_model = validated.fit_frozen(X[positive], labels[positive])
    app_scale, app_model = validated.fit_frozen(X[appreciable], labels[appreciable])
    bc = np.isin(labels, ["B", "C"])
    bc_y = (labels[bc] == "C").astype(np.int64)
    bc_scale = StandardScaler().fit(X[bc])
    bc_model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs").fit(
        bc_scale.transform(X[bc]), bc_y
    )
    metadata = {
        "canonical_dataset": {
            "id": canonical_root.name,
            "required_files_sha256": {
                name: sha256_file(canonical_root / name)
                for name in ("X.npy", "direction.npy", "groups.npy", "magnitude.npy")
            },
            "path_omitted": True,
        },
        "all_positive": {
            "selection": "direction A/B/C at every positive rate",
            "n": int(positive.sum()),
            "class_counts": {
                str(label): int(((labels == label) & positive).sum()) for label in CLASSES
            },
            "iterations": all_model.n_iter_.astype(int).tolist(),
        },
        "appreciable_sensitivity": {
            "selection": "direction A/B/C and rate >= 2.5e-4",
            "n": int(appreciable.sum()),
            "class_counts": {
                str(label): int(((labels == label) & appreciable).sum()) for label in CLASSES
            },
            "iterations": app_model.n_iter_.astype(int).tolist(),
        },
        "binary_reversed_pair_primary": {
            "selection": "canonical B and C replicates at every positive rate",
            "label_encoding": "B=0, C=1",
            "n": int(bc.sum()),
            "class_counts": {
                "B": int((labels[bc] == "B").sum()),
                "C": int((labels[bc] == "C").sum()),
            },
            "iterations": bc_model.n_iter_.astype(int).tolist(),
        },
    }
    return (
        all_scale, all_model, app_scale, app_model, bc_scale, bc_model, metadata
    )


def save_models(
    path: Path,
    all_scale: StandardScaler,
    all_model: LogisticRegression,
    app_scale: StandardScaler,
    app_model: LogisticRegression,
    bc_scale: StandardScaler,
    bc_model: LogisticRegression,
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.array(SCHEMA_VERSION),
            config_sha256=np.array(CONFIG_SHA256),
            classes=CLASSES,
            feature_columns=np.array(FEATURE_COLUMNS, dtype="U60"),
            all_positive_scaler_mean=all_scale.mean_,
            all_positive_scaler_scale=all_scale.scale_,
            all_positive_coef=all_model.coef_,
            all_positive_intercept=all_model.intercept_,
            appreciable_scaler_mean=app_scale.mean_,
            appreciable_scaler_scale=app_scale.scale_,
            appreciable_coef=app_model.coef_,
            appreciable_intercept=app_model.intercept_,
            binary_bc_scaler_mean=bc_scale.mean_,
            binary_bc_scaler_scale=bc_scale.scale_,
            binary_bc_coef=bc_model.coef_,
            binary_bc_intercept=bc_model.intercept_,
            binary_bc_classes=np.array(["B", "C"]),
        )
    os.replace(temporary, path)


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n <= 0:
        return [float("nan"), float("nan")]
    proportion = successes / n
    denominator = 1.0 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    half = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def score_subset(y: np.ndarray, probability: np.ndarray) -> dict:
    prediction = probability.argmax(axis=1)
    confusion = np.zeros((3, 3), dtype=int)
    for truth, call in zip(y, prediction):
        confusion[int(truth), int(call)] += 1
    correct = int(np.trace(confusion))
    n = int(len(y))
    per_class = {}
    for class_index, label in enumerate(CLASSES):
        row = confusion[class_index]
        support = int(row.sum())
        class_correct = int(row[class_index])
        per_class[str(label)] = {
            "support": support,
            "predicted_counts_A_B_C": row.tolist(),
            "recall": class_correct / support if support else None,
            "recall_wilson_95_ci": wilson_interval(class_correct, support),
        }
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "accuracy_wilson_95_ci": wilson_interval(correct, n),
        "confusion_rows_true_columns_predicted_A_B_C": confusion.tolist(),
        "per_class": per_class,
    }


def rate_key(rate: float) -> str:
    return f"{rate:.1e}"


def structured_scores(
    families: np.ndarray,
    rates: np.ndarray,
    y: np.ndarray,
    probability: np.ndarray,
) -> dict:
    report = {
        "overall": score_subset(y, probability),
        "by_rate": {},
        "by_family": {},
    }
    for rate in RATES:
        use = np.isclose(rates, rate, rtol=0, atol=0)
        report["by_rate"][rate_key(float(rate))] = score_subset(y[use], probability[use])
    for family in FAMILIES:
        family_use = families == family
        family_report = {
            "overall": score_subset(y[family_use], probability[family_use]),
            "by_rate": {},
        }
        for rate in RATES:
            use = family_use & np.isclose(rates, rate, rtol=0, atol=0)
            family_report["by_rate"][rate_key(float(rate))] = score_subset(
                y[use], probability[use]
            )
        report["by_family"][family] = family_report
    return report


def score_binary_subset(y: np.ndarray, probability_c: np.ndarray) -> dict:
    """Score a separately trained B/C classifier; y is encoded B=0, C=1."""
    prediction = (probability_c >= 0.5).astype(np.int64)
    confusion = np.zeros((2, 2), dtype=int)
    for truth, call in zip(y, prediction):
        confusion[int(truth), int(call)] += 1
    correct = int(np.trace(confusion))
    n = int(len(y))
    per_class = {}
    for class_index, label in enumerate(("B", "C")):
        row = confusion[class_index]
        support = int(row.sum())
        class_correct = int(row[class_index])
        per_class[label] = {
            "support": support,
            "predicted_counts_B_C": row.tolist(),
            "recall": class_correct / support if support else None,
            "recall_wilson_95_ci": wilson_interval(class_correct, support),
        }
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else None,
        "accuracy_wilson_95_ci": wilson_interval(correct, n),
        "confusion_rows_true_columns_predicted_B_C": confusion.tolist(),
        "per_class": per_class,
    }


def structured_binary_scores(
    families: np.ndarray,
    rates: np.ndarray,
    labels: np.ndarray,
    probability_c_full: np.ndarray,
    rate_values: np.ndarray = RATES,
) -> dict:
    bc = np.isin(labels, ["B", "C"])
    y = (labels[bc] == "C").astype(np.int64)
    bc_families = families[bc]
    bc_rates = rates[bc]
    probability = probability_c_full[bc]
    report = {
        "overall": score_binary_subset(y, probability),
        "by_rate": {},
        "by_family": {},
    }
    for rate in rate_values:
        use = np.isclose(bc_rates, rate, rtol=0, atol=0)
        report["by_rate"][rate_key(float(rate))] = score_binary_subset(
            y[use], probability[use]
        )
    for family in dict.fromkeys(bc_families.tolist()):
        family_use = bc_families == family
        family_report = {
            "overall": score_binary_subset(y[family_use], probability[family_use]),
            "by_rate": {},
        }
        for rate in rate_values:
            use = family_use & np.isclose(bc_rates, rate, rtol=0, atol=0)
            family_report["by_rate"][rate_key(float(rate))] = score_binary_subset(
                y[use], probability[use]
            )
        report["by_family"][family] = family_report
    return report


def exact_rate_threeway_scores(
    rates: np.ndarray,
    y: np.ndarray,
    probability: np.ndarray,
    rate_values: np.ndarray,
) -> dict:
    return {
        "overall": score_subset(y, probability),
        "by_rate": {
            rate_key(float(rate)): score_subset(
                y[np.isclose(rates, rate, rtol=0, atol=0)],
                probability[np.isclose(rates, rate, rtol=0, atol=0)],
            )
            for rate in rate_values
        },
    }


def repeated_stratified_curve_threeway(
    X: np.ndarray,
    rates: np.ndarray,
    y: np.ndarray,
    repeat_seeds: tuple[int, ...],
):
    rate_index = np.searchsorted(FIXED_CURVE_RATES, rates)
    joint_stratum = 3 * rate_index + y
    repeat_probabilities = []
    repeat_accuracies = []
    fold_audit = []
    for repeat_seed in repeat_seeds:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat_seed)
        oof = np.full((len(y), 3), np.nan, dtype=np.float64)
        for fold_index, (train, test) in enumerate(splitter.split(X, joint_stratum)):
            scale = StandardScaler().fit(X[train])
            model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs").fit(
                scale.transform(X[train]), y[train]
            )
            oof[test] = model.predict_proba(scale.transform(X[test]))
            fold_audit.append({
                "repeat_seed": int(repeat_seed),
                "fold": int(fold_index),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_test_overlap": int(len(np.intersect1d(train, test))),
                "test_joint_rate_class_counts": {
                    f"{rate_key(float(rate))}_{label}": int(
                        ((rates[test] == rate) & (y[test] == class_index)).sum()
                    )
                    for rate in FIXED_CURVE_RATES
                    for class_index, label in enumerate(CLASSES)
                },
            })
        if np.isnan(oof).any():
            raise RuntimeError("incomplete current-engine curve 3-way OOF predictions")
        repeat_probabilities.append(oof)
        repeat_accuracies.append(float((oof.argmax(axis=1) == y).mean()))
    return (
        np.mean(repeat_probabilities, axis=0),
        repeat_accuracies,
        fold_audit,
    )


def repeated_stratified_curve_binary(
    X: np.ndarray,
    rates: np.ndarray,
    labels: np.ndarray,
    repeat_seeds: tuple[int, ...],
):
    bc_indices = np.where(np.isin(labels, ["B", "C"]))[0]
    bc_X = X[bc_indices]
    bc_rates = rates[bc_indices]
    y = (labels[bc_indices] == "C").astype(np.int64)
    rate_index = np.searchsorted(FIXED_CURVE_RATES, bc_rates)
    joint_stratum = 2 * rate_index + y
    repeat_probabilities = []
    repeat_accuracies = []
    fold_audit = []
    for repeat_seed in repeat_seeds:
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat_seed)
        oof = np.full(len(y), np.nan, dtype=np.float64)
        for fold_index, (train, test) in enumerate(splitter.split(bc_X, joint_stratum)):
            scale = StandardScaler().fit(bc_X[train])
            model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs").fit(
                scale.transform(bc_X[train]), y[train]
            )
            oof[test] = model.predict_proba(scale.transform(bc_X[test]))[:, 1]
            fold_audit.append({
                "repeat_seed": int(repeat_seed),
                "fold": int(fold_index),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_test_overlap": int(len(np.intersect1d(train, test))),
                "test_joint_rate_class_counts": {
                    f"{rate_key(float(rate))}_{label}": int(
                        ((bc_rates[test] == rate) & (y[test] == class_index)).sum()
                    )
                    for rate in FIXED_CURVE_RATES
                    for class_index, label in enumerate(("B", "C"))
                },
            })
        if np.isnan(oof).any():
            raise RuntimeError("incomplete current-engine curve binary B/C OOF predictions")
        repeat_probabilities.append(oof)
        repeat_accuracies.append(float(((oof >= 0.5) == y).mean()))
    full_probability = np.full(len(labels), np.nan, dtype=np.float64)
    full_probability[bc_indices] = np.mean(repeat_probabilities, axis=0)
    return full_probability, repeat_accuracies, fold_audit


def repeated_stratified_within_family(
    X: np.ndarray,
    families: np.ndarray,
    rates: np.ndarray,
    y: np.ndarray,
    repeat_seeds: tuple[int, ...],
):
    probability = np.full((len(y), 3), np.nan, dtype=np.float64)
    diagnostics = {}
    fold_audit = []
    for family in FAMILIES:
        global_indices = np.where(families == family)[0]
        family_X = X[global_indices]
        family_y = y[global_indices]
        rate_index = np.searchsorted(RATES, rates[global_indices])
        joint_stratum = 3 * rate_index + family_y
        repeat_probabilities = []
        repeat_accuracies = []
        for repeat_seed in repeat_seeds:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat_seed)
            oof = np.full((len(global_indices), 3), np.nan, dtype=np.float64)
            for fold_index, (train, test) in enumerate(
                splitter.split(family_X, joint_stratum)
            ):
                scale = StandardScaler().fit(family_X[train])
                model = LogisticRegression(
                    C=1.0, max_iter=3000, solver="lbfgs"
                ).fit(scale.transform(family_X[train]), family_y[train])
                oof[test] = model.predict_proba(scale.transform(family_X[test]))
                fold_audit.append({
                    "family": family,
                    "repeat_seed": int(repeat_seed),
                    "fold": int(fold_index),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "train_test_overlap": int(len(np.intersect1d(train, test))),
                    "test_joint_rate_class_counts": {
                        f"{rate_key(float(rate))}_{label}": int(
                            ((rates[global_indices[test]] == rate) &
                             (family_y[test] == class_index)).sum()
                        )
                        for rate in RATES
                        for class_index, label in enumerate(CLASSES)
                    },
                })
            if np.isnan(oof).any():
                raise RuntimeError(f"incomplete within-family OOF predictions for {family}")
            repeat_probabilities.append(oof)
            repeat_accuracies.append(float((oof.argmax(axis=1) == family_y).mean()))
        mean_probability = np.mean(repeat_probabilities, axis=0)
        probability[global_indices] = mean_probability
        diagnostics[family] = {
            "repeat_accuracies": repeat_accuracies,
            "repeat_accuracy_mean": float(np.mean(repeat_accuracies)),
            "repeat_accuracy_sd": float(np.std(repeat_accuracies)),
            "scores": structured_scores(
                np.repeat(family, len(global_indices)),
                rates[global_indices], family_y, mean_probability,
            )["by_family"][family],
        }
    if np.isnan(probability).any():
        raise RuntimeError("incomplete combined within-family probabilities")
    return probability, diagnostics, fold_audit


def repeated_stratified_binary_within_family(
    X: np.ndarray,
    families: np.ndarray,
    rates: np.ndarray,
    labels: np.ndarray,
    repeat_seeds: tuple[int, ...],
):
    """Separate B/C OOF models, never a restriction of the 3-way argmax."""
    full_probability_c = np.full(len(labels), np.nan, dtype=np.float64)
    diagnostics = {}
    fold_audit = []
    for family in FAMILIES:
        global_indices = np.where(
            (families == family) & np.isin(labels, ["B", "C"])
        )[0]
        family_X = X[global_indices]
        family_y = (labels[global_indices] == "C").astype(np.int64)
        rate_index = np.searchsorted(RATES, rates[global_indices])
        joint_stratum = 2 * rate_index + family_y
        repeat_probabilities = []
        repeat_accuracies = []
        for repeat_seed in repeat_seeds:
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=repeat_seed)
            oof = np.full(len(global_indices), np.nan, dtype=np.float64)
            for fold_index, (train, test) in enumerate(
                splitter.split(family_X, joint_stratum)
            ):
                scale = StandardScaler().fit(family_X[train])
                model = LogisticRegression(
                    C=1.0, max_iter=3000, solver="lbfgs"
                ).fit(scale.transform(family_X[train]), family_y[train])
                oof[test] = model.predict_proba(scale.transform(family_X[test]))[:, 1]
                fold_audit.append({
                    "family": family,
                    "repeat_seed": int(repeat_seed),
                    "fold": int(fold_index),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "train_test_overlap": int(len(np.intersect1d(train, test))),
                    "test_joint_rate_class_counts": {
                        f"{rate_key(float(rate))}_{label}": int(
                            ((rates[global_indices[test]] == rate) &
                             (family_y[test] == class_index)).sum()
                        )
                        for rate in RATES
                        for class_index, label in enumerate(("B", "C"))
                    },
                })
            if np.isnan(oof).any():
                raise RuntimeError(f"incomplete binary B/C OOF predictions for {family}")
            repeat_probabilities.append(oof)
            repeat_accuracies.append(float(((oof >= 0.5) == family_y).mean()))
        mean_probability = np.mean(repeat_probabilities, axis=0)
        full_probability_c[global_indices] = mean_probability
        one_family = np.repeat(family, len(global_indices))
        one_full_probability = np.full(len(global_indices), np.nan)
        one_full_probability[:] = mean_probability
        diagnostics[family] = {
            "repeat_accuracies": repeat_accuracies,
            "repeat_accuracy_mean": float(np.mean(repeat_accuracies)),
            "repeat_accuracy_sd": float(np.std(repeat_accuracies)),
            "scores": structured_binary_scores(
                one_family,
                rates[global_indices],
                labels[global_indices],
                one_full_probability,
            )["by_family"][family],
        }
    if np.isnan(full_probability_c[np.isin(labels, ["B", "C"])]).any():
        raise RuntimeError("incomplete combined binary B/C OOF probabilities")
    return full_probability_c, diagnostics, fold_audit


def misclassification_ledger(
    arrays: dict[str, np.ndarray], y: np.ndarray, probability: np.ndarray
) -> list[dict]:
    prediction = probability.argmax(axis=1)
    errors = []
    for i in np.where(prediction != y)[0]:
        errors.append({
            "family": str(arrays["family"][i]),
            "rate": float(arrays["rate"][i]),
            "true": str(CLASSES[y[i]]),
            "predicted": str(CLASSES[prediction[i]]),
            "replicate": int(arrays["replicate"][i]),
            "probabilities_A_B_C": probability[i].tolist(),
        })
    return errors


def binary_misclassification_ledger(
    arrays: dict[str, np.ndarray], probability_c_full: np.ndarray
) -> list[dict]:
    labels = arrays["labels"]
    bc_indices = np.where(np.isin(labels, ["B", "C"]))[0]
    y = (labels[bc_indices] == "C").astype(np.int64)
    prediction = (probability_c_full[bc_indices] >= 0.5).astype(np.int64)
    errors = []
    for local_index in np.where(prediction != y)[0]:
        i = int(bc_indices[local_index])
        errors.append({
            "family": str(arrays["family"][i]),
            "rate": float(arrays["rate"][i]),
            "true": str(labels[i]),
            "predicted": "C" if prediction[local_index] else "B",
            "replicate": int(arrays["replicate"][i]),
            "probability_C": float(probability_c_full[i]),
        })
    return errors


def simulation_diagnostics(arrays: dict[str, np.ndarray]) -> dict:
    report = {}
    for family in FAMILIES:
        use = arrays["family"] == family
        report[family] = {
            "n": int(use.sum()),
            "num_mutated_sites_mean": float(arrays["num_sites"][use].mean()),
            "num_mutated_sites_sd": float(arrays["num_sites"][use].std()),
            "num_PADZE_loci_mean": float(arrays["num_loci"][use].mean()),
            "num_PADZE_loci_sd": float(arrays["num_loci"][use].std()),
            "worker_seconds_mean": float(arrays["elapsed_seconds"][use].mean()),
        }
    return report


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_summary(path: Path, result: dict) -> None:
    primary = result["frozen_canonical_all_positive_transfer"]["scores"]
    binary = result["frozen_canonical_binary_BC_transfer"]["scores"]
    curve = result["current_engine_disjoint_stream_fixed_rate_replication"]
    lines = [
        "# Post hoc frozen-model nuisance-transfer grid",
        "",
        "The canonical all-positive 54-D logistic model was frozen before scoring "
        f"{result['simulation_grid']['n_replicates']} new CPU-only, disjoint-seed-stream "
        "replicates. This is a prespecified stress grid analyzed post hoc, not confirmatory "
        "empirical validation.",
        "",
        "## Frozen transfer by family and rate",
        "",
        "| Family | m=5e-5 | m=2.5e-4 |",
        "|---|---:|---:|",
    ]
    for family in FAMILIES:
        by_rate = primary["by_family"][family]["by_rate"]
        low = by_rate[rate_key(float(RATES[0]))]
        high = by_rate[rate_key(float(RATES[1]))]
        lines.append(
            f"| {family} | {low['correct']}/{low['n']} ({low['accuracy']:.3f}) | "
            f"{high['correct']}/{high['n']} ({high['accuracy']:.3f}) |"
        )
    lines.extend([
        "",
        "## Separately trained frozen binary B/C transfer",
        "",
        "| Family | m=5e-5 | m=2.5e-4 |",
        "|---|---:|---:|",
    ])
    for family in FAMILIES:
        by_rate = binary["by_family"][family]["by_rate"]
        low = by_rate[rate_key(float(RATES[0]))]
        high = by_rate[rate_key(float(RATES[1]))]
        lines.append(
            f"| {family} | {low['correct']}/{low['n']} ({low['accuracy']:.3f}) | "
            f"{high['correct']}/{high['n']} ({high['accuracy']:.3f}) |"
        )
    lines.extend([
        "",
        "## Current-engine disjoint-stream canonical replication",
        "",
        "| Rate | frozen 3-way | frozen binary B/C |",
        "|---:|---:|---:|",
    ])
    curve_three = curve["frozen_legacy_threeway_transfer"]["scores"]["by_rate"]
    curve_binary = curve["frozen_legacy_binary_BC_transfer"]["scores"]["by_rate"]
    for rate in FIXED_CURVE_RATES:
        key = rate_key(float(rate))
        three = curve_three[key]
        bc = curve_binary[key]
        lines.append(
            f"| {key} | {three['correct']}/{three['n']} ({three['accuracy']:.3f}) | "
            f"{bc['correct']}/{bc['n']} ({bc['accuracy']:.3f}) |"
        )
    lines.extend([
        "",
        "Wilson intervals, exact class-confusion rows, every msprime epoch/migration matrix, "
        "and the misclassification ledger are in `results.json`.",
        "",
        "Interpretation must remain narrow: each nuisance family is one prespecified perturbation "
        "with 20 replicates per rate and direction. Frozen-model transfer measures robustness to "
        "those perturbations only. Repeated within-family CV asks whether the same 54-D features "
        "remain learnable after matched retraining; it is not external validation and cannot rescue "
        "a failed frozen transfer claim.",
        "",
        "The four-rate curve is a current-engine design replication with distinct ancestry "
        "and mutation seed streams. It resolves the practical reproducibility vulnerability "
        "but does not isolate a causal seed effect because the archived generator's engine "
        "version is unknown. Population sizes are haploid coalescent sizes; diploid-equivalent "
        "values are one half and are explicit in `results.json`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=370_709_001)
    parser.add_argument("--curve-seed-base", type=int, default=470_709_001)
    parser.add_argument("--cv-seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path(os.environ.get("DNNAIC_DATA", "data/simulation_data")) / "regen_full",
        help="regen_full directory (default: $DNNAIC_DATA/regen_full)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/nuisance_transfer_2026_07_09"),
    )
    parser.add_argument("--simulate-only", action="store_true")
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    if not args.simulate_only and args.reps < 20:
        parser.error("final analysis requires at least 20 replicates per cell")
    if args.workers < 1:
        parser.error("--workers must be positive")

    print("[audit] validating actual demographic epochs and migration matrices", flush=True)
    demographic_audit = audit_demographies()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Freeze both canonical models before reading/scoring the nuisance batch.
    print("[model] fitting and freezing canonical models before nuisance scoring", flush=True)
    (
        all_scale, all_model, app_scale, app_model, bc_scale, bc_model, model_metadata
    ) = fit_frozen_models(args.canonical_root)
    model_path = args.out_dir / "frozen_canonical_models.npz"
    save_models(
        model_path, all_scale, all_model, app_scale, app_model, bc_scale, bc_model
    )

    archive_path = args.out_dir / "nuisance_features.npz"
    records = load_archive(archive_path)
    jobs = make_jobs(args.reps, args.seed_base)
    expected_seeds = {
        (family, rate_index, label, replicate): (ancestry_seed, mutation_seed)
        for family, rate_index, rate, label, replicate, ancestry_seed, mutation_seed in jobs
    }
    for record in records:
        key = record_key(record)
        if key in expected_seeds and expected_seeds[key] != (
            record["ancestry_seed"], record["mutation_seed"]
        ):
            raise RuntimeError(f"checkpoint seed mismatch for {key}")
    records = simulate_missing(records, jobs, min(args.workers, len(jobs)), archive_path)
    if args.simulate_only:
        print(f"[done] simulation checkpoint -> {archive_path}", flush=True)
        return 0

    requested = {
        (family, rate_index, label, replicate)
        for family, rate_index, rate, label, replicate, _, _ in jobs
    }
    selected = [record for record in records if record_key(record) in requested]
    selected = sorted_records(selected)
    expected_n = len(FAMILIES) * len(RATES) * len(CLASSES) * args.reps
    if len(selected) != expected_n:
        raise RuntimeError(f"selected {len(selected)} records, expected {expected_n}")
    arrays = record_arrays(selected)
    X = arrays["X"].astype(np.float64)
    y = encode(arrays["labels"])

    print("[score] frozen transfer and repeated stratified within-family diagnostics", flush=True)
    all_probability = all_model.predict_proba(all_scale.transform(X))
    app_probability = app_model.predict_proba(app_scale.transform(X))
    bc_probability = np.full(len(arrays["labels"]), np.nan, dtype=np.float64)
    bc_use = np.isin(arrays["labels"], ["B", "C"])
    bc_probability[bc_use] = bc_model.predict_proba(
        bc_scale.transform(X[bc_use])
    )[:, 1]
    repeat_seeds = tuple(int(seed) for seed in args.cv_seeds.split(",") if seed != "")
    if not repeat_seeds:
        raise ValueError("at least one CV repeat seed is required")
    within_probability, within_diagnostics, fold_audit = repeated_stratified_within_family(
        X, arrays["family"], arrays["rate"], y, repeat_seeds
    )
    (
        within_binary_probability,
        within_binary_diagnostics,
        binary_fold_audit,
    ) = repeated_stratified_binary_within_family(
        X, arrays["family"], arrays["rate"], arrays["labels"], repeat_seeds
    )

    save_archive(
        selected,
        archive_path,
        all_positive_probability=all_probability,
        appreciable_probability=app_probability,
        within_family_probability=within_probability,
        frozen_binary_bc_probability=bc_probability,
        within_family_binary_bc_probability=within_binary_probability,
    )

    primary_scores = structured_scores(
        arrays["family"], arrays["rate"], y, all_probability
    )
    sensitivity_scores = structured_scores(
        arrays["family"], arrays["rate"], y, app_probability
    )
    binary_scores = structured_binary_scores(
        arrays["family"], arrays["rate"], arrays["labels"], bc_probability
    )

    # Extend the current-engine canonical control to a complete four-rate curve.
    curve_archive_path = args.out_dir / "independent_seed_fixed_rate_features.npz"
    reusable_curve_records, curve_jobs = prepare_curve_records_and_jobs(
        selected, args.curve_seed_base
    )
    curve_records = load_curve_archive(curve_archive_path)
    if not curve_records:
        curve_records = reusable_curve_records
        save_curve_archive(curve_records, curve_archive_path)
    curve_records = simulate_curve_missing(
        curve_records, curve_jobs, min(args.workers, len(curve_jobs)), curve_archive_path
    )
    curve_expected_keys = {record_key(record) for record in reusable_curve_records}
    curve_expected_keys.update(
        (job[0], job[1], job[3], job[4]) for job in curve_jobs
    )
    curve_selected = sorted_records([
        record for record in curve_records if record_key(record) in curve_expected_keys
    ])
    if len(curve_selected) != 360:
        raise RuntimeError(
            f"independent-stream fixed-rate curve has {len(curve_selected)} rows, expected 360"
        )
    curve_arrays = record_arrays(curve_selected)
    curve_X = curve_arrays["X"].astype(np.float64)
    curve_y = encode(curve_arrays["labels"])
    curve_frozen_threeway_probability = all_model.predict_proba(
        all_scale.transform(curve_X)
    )
    curve_bc_probability = np.full(len(curve_y), np.nan, dtype=np.float64)
    curve_bc_use = np.isin(curve_arrays["labels"], ["B", "C"])
    curve_bc_probability[curve_bc_use] = bc_model.predict_proba(
        bc_scale.transform(curve_X[curve_bc_use])
    )[:, 1]
    (
        curve_within_threeway_probability,
        curve_threeway_repeat_accuracy,
        curve_threeway_fold_audit,
    ) = repeated_stratified_curve_threeway(
        curve_X, curve_arrays["rate"], curve_y, repeat_seeds
    )
    (
        curve_within_binary_probability,
        curve_binary_repeat_accuracy,
        curve_binary_fold_audit,
    ) = repeated_stratified_curve_binary(
        curve_X, curve_arrays["rate"], curve_arrays["labels"], repeat_seeds
    )
    save_curve_archive(
        curve_selected,
        curve_archive_path,
        frozen_threeway_probability=curve_frozen_threeway_probability,
        frozen_binary_bc_probability=curve_bc_probability,
        within_curve_threeway_probability=curve_within_threeway_probability,
        within_curve_binary_bc_probability=curve_within_binary_probability,
    )
    curve_frozen_threeway_scores = exact_rate_threeway_scores(
        curve_arrays["rate"], curve_y, curve_frozen_threeway_probability,
        FIXED_CURVE_RATES,
    )
    curve_frozen_binary_scores = structured_binary_scores(
        curve_arrays["family"], curve_arrays["rate"], curve_arrays["labels"],
        curve_bc_probability, FIXED_CURVE_RATES,
    )
    curve_within_threeway_scores = exact_rate_threeway_scores(
        curve_arrays["rate"], curve_y, curve_within_threeway_probability,
        FIXED_CURVE_RATES,
    )
    curve_within_binary_scores = structured_binary_scores(
        curve_arrays["family"], curve_arrays["rate"], curve_arrays["labels"],
        curve_within_binary_probability, FIXED_CURVE_RATES,
    )
    seed_pairs_distinct = bool(np.all(
        curve_arrays["ancestry_seed"] != curve_arrays["mutation_seed"]
    ))
    all_seed_streams_disjoint = bool(
        len(np.unique(curve_arrays["ancestry_seed"])) == len(curve_arrays["ancestry_seed"])
        and len(np.unique(curve_arrays["mutation_seed"])) == len(curve_arrays["mutation_seed"])
        and len(np.intersect1d(
            curve_arrays["ancestry_seed"], curve_arrays["mutation_seed"]
        )) == 0
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "configuration_sha256": CONFIG_SHA256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "cpu_only": True,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "msprime": msprime.__version__,
            "padze": getattr(padze, "__version__", "unknown"),
            "scikit_learn": sklearn.__version__,
            "platform": platform.platform(),
        },
        "invariants": {
            "population_size_convention": (
                "All population-size values are haploid coalescent sizes under the "
                "project convention; diploid-equivalent sizes are one half and are "
                "reported explicitly in the family audit."
            ),
            "sequence_length_bp": SEQUENCE_LENGTH,
            "recombination_rate_per_bp_per_generation": RECOMBINATION_RATE,
            "mutation_rate_per_bp_per_generation": MUTATION_RATE,
            "backward_direction_mappings": {
                label: f"{mapping[0]}->{mapping[1]}"
                for label, mapping in CLASS_MIGRATION.items()
            },
            "migration_active_interval_generations": [0, MIGRATION_END],
            "exact_rates": RATES.tolist(),
            "feature_dimension": 54,
            "feature_definition": (
                "mean and population SD across the supported PADZE depth grid for each "
                "of 27 non-depth coordinates"
            ),
        },
        "demographic_epoch_and_migration_matrix_audit": demographic_audit,
        "simulation_grid": {
            "families": list(FAMILIES),
            "rates": RATES.tolist(),
            "directions": CLASSES.tolist(),
            "reps_per_family_rate_direction": int(args.reps),
            "n_replicates": int(expected_n),
            "fixed_seed_base": int(args.seed_base),
            "diagnostics_by_family": simulation_diagnostics(arrays),
            "pulse_family": {
                "included": False,
                "reason": (
                    "A pulse is a distinct mass-migration parameterization, not a "
                    "single nuisance perturbation of this continuous-migration grid; "
                    "it was kept separate rather than conflated."
                ),
            },
        },
        "frozen_model_training": model_metadata,
        "frozen_canonical_all_positive_transfer": {
            "role": "primary external transfer analysis",
            "model": (
                "canonical-only StandardScaler plus multinomial logistic regression "
                "(lbfgs, C=1, max_iter=3000), frozen before nuisance scoring"
            ),
            "interval_method": "Wilson score 95% intervals over independent replicates",
            "scores": primary_scores,
            "misclassifications": misclassification_ledger(arrays, y, all_probability),
        },
        "frozen_canonical_appreciable_sensitivity": {
            "role": "prespecified model-training-band sensitivity; not the primary analysis",
            "scores": sensitivity_scores,
            "misclassifications": misclassification_ledger(arrays, y, app_probability),
        },
        "frozen_canonical_binary_BC_transfer": {
            "role": (
                "Primary clean donor-recipient reversal endpoint; separately trained "
                "binary B-versus-C scaler and classifier, never a subset of 3-way argmax calls."
            ),
            "model": (
                "canonical B/C-only StandardScaler plus binary logistic regression "
                "(B=0, C=1; lbfgs, C=1, max_iter=3000), frozen before nuisance scoring"
            ),
            "interval_method": "Wilson score 95% intervals over independent replicates",
            "scores": binary_scores,
            "misclassifications": binary_misclassification_ledger(arrays, bc_probability),
        },
        "repeated_stratified_within_family_cv_diagnostic": {
            "warning": (
                "Diagnostic matched retraining only; this is not frozen-model transfer "
                "and not external validation."
            ),
            "n_folds": 5,
            "repeat_seeds": list(repeat_seeds),
            "stratification": "joint exact-rate x direction (six equal strata per family)",
            "probability_aggregation": "mean OOF probabilities across repeats, then argmax",
            "all_fold_train_test_overlap_zero": bool(
                all(entry["train_test_overlap"] == 0 for entry in fold_audit)
            ),
            "by_family": within_diagnostics,
            "fold_audit": fold_audit,
        },
        "repeated_stratified_within_family_binary_BC_cv_diagnostic": {
            "warning": (
                "Separately fit binary B/C diagnostic matched retraining only; this is "
                "not frozen-model transfer and not external validation."
            ),
            "n_folds": 5,
            "repeat_seeds": list(repeat_seeds),
            "stratification": "joint exact-rate x B/C class (four equal strata per family)",
            "probability_aggregation": "mean held-out probability_C across repeats",
            "all_fold_train_test_overlap_zero": bool(
                all(entry["train_test_overlap"] == 0 for entry in binary_fold_audit)
            ),
            "by_family": within_binary_diagnostics,
            "fold_audit": binary_fold_audit,
        },
        "current_engine_disjoint_stream_fixed_rate_replication": {
            "role": (
                "Independent current-engine design replication addressing the legacy "
                "same-integer seed-stream vulnerability; not a causal isolation of seed "
                "coupling because the archived generator version is unknown."
            ),
            "schema_version": CURVE_SCHEMA_VERSION,
            "configuration_sha256": CURVE_CONFIG_SHA256,
            "rates": FIXED_CURVE_RATES.tolist(),
            "directions": CLASSES.tolist(),
            "reps_per_rate_direction": 30,
            "n_replicates": 360,
            "provenance_counts": {
                str(value): int((curve_arrays["provenance"] == value).sum())
                for value in np.unique(curve_arrays["provenance"])
            },
            "seed_policy": {
                "description": (
                    "Every new genealogy uses an ancestry seed and the immediately "
                    "following distinct mutation seed; seed sets are globally unique "
                    "and disjoint. Upper-rate rows reuse the completed nuisance control, "
                    "which used the identical disjoint-stream policy."
                ),
                "new_curve_seed_base": int(args.curve_seed_base),
                "ancestry_seed_not_equal_mutation_seed_for_every_row": seed_pairs_distinct,
                "ancestry_and_mutation_seed_sets_globally_unique_and_disjoint": (
                    all_seed_streams_disjoint
                ),
            },
            "frozen_legacy_threeway_transfer": {
                "warning": (
                    "The frozen model was trained on the archived canonical dataset, whose "
                    "generator reused the same integer for ancestry and mutation and whose "
                    "engine version is not recorded."
                ),
                "scores": curve_frozen_threeway_scores,
                "misclassifications": misclassification_ledger(
                    curve_arrays, curve_y, curve_frozen_threeway_probability
                ),
            },
            "frozen_legacy_binary_BC_transfer": {
                "model": "separately trained canonical B/C binary scaler and logistic classifier",
                "scores": curve_frozen_binary_scores,
                "misclassifications": binary_misclassification_ledger(
                    curve_arrays, curve_bc_probability
                ),
            },
            "within_current_design_repeated_OOF_threeway": {
                "warning": "diagnostic retraining, not frozen transfer",
                "repeat_seeds": list(repeat_seeds),
                "repeat_accuracies": curve_threeway_repeat_accuracy,
                "repeat_accuracy_mean": float(np.mean(curve_threeway_repeat_accuracy)),
                "repeat_accuracy_sd": float(np.std(curve_threeway_repeat_accuracy)),
                "scores": curve_within_threeway_scores,
                "all_fold_train_test_overlap_zero": bool(all(
                    entry["train_test_overlap"] == 0
                    for entry in curve_threeway_fold_audit
                )),
                "fold_audit": curve_threeway_fold_audit,
            },
            "within_current_design_repeated_OOF_binary_BC": {
                "warning": (
                    "separately fit binary B/C diagnostic retraining, not a restriction "
                    "of 3-way predictions and not frozen transfer"
                ),
                "repeat_seeds": list(repeat_seeds),
                "repeat_accuracies": curve_binary_repeat_accuracy,
                "repeat_accuracy_mean": float(np.mean(curve_binary_repeat_accuracy)),
                "repeat_accuracy_sd": float(np.std(curve_binary_repeat_accuracy)),
                "scores": curve_within_binary_scores,
                "all_fold_train_test_overlap_zero": bool(all(
                    entry["train_test_overlap"] == 0
                    for entry in curve_binary_fold_audit
                )),
                "fold_audit": curve_binary_fold_audit,
            },
            "checkpoint": curve_archive_path.name,
        },
        "scope_limit": (
            "Each family is one prespecified nuisance parameterization with finite Monte "
            "Carlo size. Results establish transfer behavior only for these settings, not "
            "uniform demographic robustness or empirical-species validity."
        ),
    }

    summary_path = args.out_dir / "SUMMARY.md"
    write_summary(summary_path, result)
    script_path = Path(__file__).resolve()
    result["artifacts"] = {
        archive_path.name: {"sha256": sha256_file(archive_path)},
        curve_archive_path.name: {"sha256": sha256_file(curve_archive_path)},
        model_path.name: {"sha256": sha256_file(model_path)},
        summary_path.name: {"sha256": sha256_file(summary_path)},
        "scripts/nuisance_transfer.py": {"sha256": sha256_file(script_path)},
    }
    result_path = args.out_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    concise = {
        family: {
            rate_key(float(rate)): primary_scores["by_family"][family]["by_rate"]
            [rate_key(float(rate))]["accuracy"]
            for rate in RATES
        }
        for family in FAMILIES
    }
    print(json.dumps({
        "primary_frozen_transfer_accuracy_by_family_rate": concise,
        "artifacts": [
            archive_path.name, curve_archive_path.name, model_path.name,
            result_path.name, summary_path.name,
        ],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
