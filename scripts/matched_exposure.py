#!/usr/bin/env python3
"""Matched-migration-exposure direction experiment (CPU only).

The canonical caterpillar demography gives the sister-population pair P1/P2 a
500-generation branch and the P12/P3 pair a 1,000-generation branch.  This
experiment holds the integrated single-lineage migration hazard ``m * T`` at
0.25 in all three direction classes:

    A: P2 -> P1, m=5e-4, T=500
    B: P3 -> P2, m=2.5e-4, T=1000
    C: P2 -> P3, m=2.5e-4, T=1000

Mappings are backwards in time, as required by msprime.  At generation 500 the
P2 branch becomes P12.  B and C therefore continue as P3 -> P12 and
P12 -> P3, respectively, until the P12/P3 split at generation 1000.  Merely
setting a P2/P3 migration matrix once would *not* produce T=1000: msprime zeros
that matrix when P2 becomes inactive at generation 500.

Each independent coalescent replicate is reduced to the exact 54-D
representation in ``scripts/direction_curve.py``: the mean and population-SD
across PADZE depths g=2..199 of each of the 27 non-depth coordinates.  The
script then reports two strictly separate evaluations:

1. Transfer from a standardized multinomial logistic model fit once on the
   canonical regen_full appreciable band (m >= 2.5e-4) and frozen before this
   matched batch is scored.  A secondary all-positive-rate canonical fit is
   included to reproduce direction_curve.py's training scope.
2. Leakage-free repeated five-fold out-of-fold logistic evaluation within the
   matched batch, with each simulation replicate appearing in exactly one test
   fold per repeat.

Simulation is resumable.  ``matched_features.npz`` is atomically updated after
every completed replicate, so extending ``--reps`` from 30 to 50 does not redo
the first 30.  No tree sequences or per-depth matrices are retained.
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

# This experiment has no GPU implementation.  Hiding devices also prevents an
# accidental GPU-aware dependency from claiming memory in worker processes.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import msprime
import numpy as np
import padze
import sklearn
from padze import LociData, Metadata, compute_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


NE = 10_000
SPLIT_P12 = 500
SPLIT_P123 = 1_000
SPLIT_ROOT = 2_000
SEQUENCE_LENGTH = 1_000_000
RECOMBINATION_RATE = 1.78e-8
MUTATION_RATE = 2e-8
GENE_COPIES_PER_POP = 200
DEPTHS = np.arange(2, 200, dtype=np.int64)
MOMENTS = ("mean", "variance", "se")
CLASSES = np.array(["A", "B", "C"])
APPRECIABLE = 2.5e-4
SCHEMA_VERSION = "matched-exposure-v1"

BLOCKS = (
    "alpha_1", "alpha_2", "alpha_3",
    "pi_1", "pi_2", "pi_3",
    "pihat_12", "pihat_13", "pihat_23",
)
CURVE_COLUMNS = ["g"] + [f"{block}_{moment}" for block in BLOCKS for moment in MOMENTS]
FEATURE_COLUMNS = (
    [f"depth_mean__{column}" for column in CURVE_COLUMNS[1:]]
    + [f"depth_sd__{column}" for column in CURVE_COLUMNS[1:]]
)

# Every tuple is (start, end, backwards-time source, backwards-time destination).
# P2's ancestral continuation is P12 after generation 500.
CLASS_DESIGN = {
    "A": {
        "rate": 5e-4,
        "duration": 500,
        "requested_mapping": ("P2", "P1"),
        "segments": ((0, 500, "P2", "P1"),),
    },
    "B": {
        "rate": 2.5e-4,
        "duration": 1_000,
        "requested_mapping": ("P3", "P2"),
        "segments": ((0, 500, "P3", "P2"), (500, 1_000, "P3", "P12")),
    },
    "C": {
        "rate": 2.5e-4,
        "duration": 1_000,
        "requested_mapping": ("P2", "P3"),
        "segments": ((0, 500, "P2", "P3"), (500, 1_000, "P12", "P3")),
    },
}


def make_demography(label: str) -> msprime.Demography:
    """Build one class, including ancestral continuation of B/C migration."""
    spec = CLASS_DESIGN[label]
    d = msprime.Demography()
    # P1/P2/P3 are deliberately first: their population IDs are 0,1,2 in the
    # genotype-to-locus conversion below.  P4 is present but is not sampled.
    for name in ("P1", "P2", "P3", "P4", "P12", "P123", "P1234"):
        d.add_population(name=name, initial_size=NE)
    d.add_population_split(time=SPLIT_P12, derived=["P1", "P2"], ancestral="P12")
    d.add_population_split(time=SPLIT_P123, derived=["P12", "P3"], ancestral="P123")
    d.add_population_split(time=SPLIT_ROOT, derived=["P123", "P4"], ancestral="P1234")

    segments = spec["segments"]
    _, _, source, dest = segments[0]
    d.set_migration_rate(source=source, dest=dest, rate=spec["rate"])
    # Population-split events were inserted first.  At a shared timestamp they
    # activate P12 before this rate change is applied.
    for start, _, source, dest in segments[1:]:
        d.add_migration_rate_change(
            time=start, source=source, dest=dest, rate=spec["rate"]
        )
    d.sort_events()
    return d


def validate_demographies() -> dict:
    """Assert actual msprime epochs implement the stated m*T design."""
    report = {}
    for label, spec in CLASS_DESIGN.items():
        d = make_demography(label)
        epochs = []
        observed = []
        for epoch in d.debug().epochs:
            start = float(epoch.start_time)
            end = float(epoch.end_time)
            nonzero = []
            for source, dest in np.argwhere(epoch.migration_matrix > 0):
                entry = {
                    "source": d.populations[int(source)].name,
                    "dest": d.populations[int(dest)].name,
                    "rate": float(epoch.migration_matrix[source, dest]),
                }
                nonzero.append(entry)
                if math.isfinite(end):
                    observed.append((start, end, entry["source"], entry["dest"], entry["rate"]))
            epochs.append({
                "start": start,
                "end": None if not math.isfinite(end) else end,
                "nonzero_migration": nonzero,
            })

        expected = [
            (float(start), float(end), source, dest, float(spec["rate"]))
            for start, end, source, dest in spec["segments"]
        ]
        if observed != expected:
            raise AssertionError(f"{label}: observed migration epochs {observed} != {expected}")
        hazard = sum((end - start) * rate for start, end, _, _, rate in observed)
        if not np.isclose(hazard, 0.25, rtol=0, atol=1e-12):
            raise AssertionError(f"{label}: integrated hazard is {hazard}, not 0.25")
        report[label] = {
            "requested_backward_mapping": "->".join(spec["requested_mapping"]),
            "rate": float(spec["rate"]),
            "duration_generations": int(spec["duration"]),
            "integrated_single_lineage_hazard_m_times_T": float(hazard),
            "single_lineage_probability_at_least_one_migration": float(-math.expm1(-hazard)),
            "actual_msprime_epochs": epochs,
        }
    return report


def tree_sequence_to_loci(ts: msprime.TreeSequence) -> LociData:
    """Convert P1/P2/P3 genotypes to PADZE allele-count matrices."""
    populations = ["P1", "P2", "P3"]
    samples = ts.samples()
    sample_population = ts.tables.nodes.population[samples]
    masks = [sample_population == j for j in range(3)]
    n = np.array([int(mask.sum()) for mask in masks], dtype=np.int64)
    if not np.array_equal(n, np.full(3, GENE_COPIES_PER_POP)):
        raise AssertionError(f"haploid sample counts are {n.tolist()}, expected 200 each")

    genotype = ts.genotype_matrix()
    count_matrices = []
    sample_sizes = []
    if genotype.shape[0]:
        max_allele = genotype.max(axis=1)
        for site_index, maximum in enumerate(max_allele):
            maximum = int(maximum)
            if maximum < 1:
                continue
            row = genotype[site_index]
            counts = np.stack([
                np.bincount(row[mask], minlength=maximum + 1)
                for mask in masks
            ]).astype(np.int64, copy=False)
            # Exclude sites monomorphic across the three experimental samples.
            if int((counts.sum(axis=0) > 0).sum()) < 2:
                continue
            count_matrices.append(counts)
            sample_sizes.append(counts.sum(axis=1))

    sizes = (
        np.vstack(sample_sizes).astype(np.int64, copy=False)
        if sample_sizes else np.zeros((0, 3), dtype=np.int64)
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
        sizes,
        [f"s{i}" for i in range(len(count_matrices))],
        metadata,
    )


def simulate_feature(job: tuple[str, int, int, int]) -> dict:
    """Worker: simulate one independent genealogy and return one 54-D vector."""
    label, replicate, ancestry_seed, mutation_seed = job
    started = time.perf_counter()
    ts = msprime.sim_ancestry(
        samples={
            "P1": GENE_COPIES_PER_POP,
            "P2": GENE_COPIES_PER_POP,
            "P3": GENE_COPIES_PER_POP,
        },
        sequence_length=SEQUENCE_LENGTH,
        recombination_rate=RECOMBINATION_RATE,
        ploidy=1,
        demography=make_demography(label),
        random_seed=ancestry_seed,
    )
    ts = msprime.sim_mutations(
        ts, rate=MUTATION_RATE, random_seed=mutation_seed, keep=True
    )
    loci = tree_sequence_to_loci(ts)
    table = compute_features(
        loci,
        depths=DEPTHS,
        pihat_sizes=(2,),
        moments=MOMENTS,
        bias_corrected=True,
    )
    matrix, columns = table.to_frame()
    index = {name: i for i, name in enumerate(columns)}
    try:
        matrix = matrix[:, [index[name] for name in CURVE_COLUMNS]].astype(np.float64)
    except KeyError as exc:
        raise RuntimeError(f"PADZE feature contract changed; missing {exc}") from exc
    if matrix.shape != (198, 28):
        raise AssertionError(f"PADZE matrix shape is {matrix.shape}, expected (198, 28)")
    if not np.array_equal(matrix[:, 0], DEPTHS):
        raise AssertionError("PADZE depth grid is not exactly g=2..199")
    curve = matrix[:, 1:]
    feature = np.concatenate([curve.mean(axis=0), curve.std(axis=0)], axis=0)
    if feature.shape != (54,) or not np.isfinite(feature).all():
        raise AssertionError("non-finite or incorrectly shaped 54-D aggregate")

    spec = CLASS_DESIGN[label]
    return {
        "label": label,
        "replicate": int(replicate),
        "ancestry_seed": int(ancestry_seed),
        "mutation_seed": int(mutation_seed),
        "rate": float(spec["rate"]),
        "duration": int(spec["duration"]),
        "hazard": float(spec["rate"] * spec["duration"]),
        "num_sites": int(ts.num_sites),
        "num_loci": int(loci.metadata.n_loci_kept),
        "elapsed_seconds": float(time.perf_counter() - started),
        "feature": feature.astype(np.float32),
    }


def make_jobs(reps_per_class: int, seed_base: int) -> list[tuple[str, int, int, int]]:
    jobs = []
    for class_index, label in enumerate(CLASSES):
        for replicate in range(reps_per_class):
            ancestry_seed = seed_base + class_index * 100_000 + 2 * replicate
            mutation_seed = ancestry_seed + 1
            if not (0 < ancestry_seed < 2**32 and 0 < mutation_seed < 2**32):
                raise ValueError("msprime seeds must lie in [1, 2**32-1]")
            jobs.append((str(label), replicate, ancestry_seed, mutation_seed))
    return jobs


def _record_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    records = sorted(records, key=lambda r: (r["label"], r["replicate"]))
    return {
        "X": np.vstack([r["feature"] for r in records]).astype(np.float32),
        "labels": np.array([r["label"] for r in records], dtype="U1"),
        "replicate": np.array([r["replicate"] for r in records], dtype=np.int32),
        "ancestry_seed": np.array([r["ancestry_seed"] for r in records], dtype=np.int64),
        "mutation_seed": np.array([r["mutation_seed"] for r in records], dtype=np.int64),
        "rate": np.array([r["rate"] for r in records], dtype=np.float64),
        "duration": np.array([r["duration"] for r in records], dtype=np.int32),
        "hazard": np.array([r["hazard"] for r in records], dtype=np.float64),
        "num_sites": np.array([r["num_sites"] for r in records], dtype=np.int32),
        "num_loci": np.array([r["num_loci"] for r in records], dtype=np.int32),
        "elapsed_seconds": np.array([r["elapsed_seconds"] for r in records], dtype=np.float64),
    }


def save_feature_archive(
    records: list[dict],
    path: Path,
    frozen_appreciable_probability: np.ndarray | None = None,
    frozen_all_positive_probability: np.ndarray | None = None,
    matched_cv_probability: np.ndarray | None = None,
) -> None:
    arrays = _record_arrays(records)
    arrays.update({
        "schema_version": np.array(SCHEMA_VERSION),
        "curve_depths": DEPTHS,
        "curve_columns": np.array(CURVE_COLUMNS, dtype="U40"),
        "feature_columns": np.array(FEATURE_COLUMNS, dtype="U60"),
    })
    if frozen_appreciable_probability is not None:
        arrays["frozen_appreciable_probability"] = frozen_appreciable_probability.astype(np.float32)
    if frozen_all_positive_probability is not None:
        arrays["frozen_all_positive_probability"] = frozen_all_positive_probability.astype(np.float32)
    if matched_cv_probability is not None:
        arrays["matched_cv_probability"] = matched_cv_probability.astype(np.float32)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_feature_archive(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as archive:
        schema = str(archive["schema_version"].item())
        if schema != SCHEMA_VERSION:
            raise RuntimeError(f"checkpoint schema {schema!r} != {SCHEMA_VERSION!r}")
        if list(archive["feature_columns"].astype(str)) != FEATURE_COLUMNS:
            raise RuntimeError("checkpoint 54-D feature contract differs from this script")
        records = []
        for i in range(len(archive["labels"])):
            records.append({
                "label": str(archive["labels"][i]),
                "replicate": int(archive["replicate"][i]),
                "ancestry_seed": int(archive["ancestry_seed"][i]),
                "mutation_seed": int(archive["mutation_seed"][i]),
                "rate": float(archive["rate"][i]),
                "duration": int(archive["duration"][i]),
                "hazard": float(archive["hazard"][i]),
                "num_sites": int(archive["num_sites"][i]),
                "num_loci": int(archive["num_loci"][i]),
                "elapsed_seconds": float(archive["elapsed_seconds"][i]),
                "feature": archive["X"][i].astype(np.float32),
            })
    return records


def simulate_missing(
    records: list[dict], jobs: list[tuple[str, int, int, int]], workers: int, archive: Path
) -> list[dict]:
    completed = {(r["label"], r["replicate"]) for r in records}
    missing = [job for job in jobs if (job[0], job[1]) not in completed]
    print(
        f"[simulate] {len(records)} checkpointed; {len(missing)} missing; "
        f"{workers} CPU worker(s)",
        flush=True,
    )
    if not missing:
        return records

    start = time.perf_counter()
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_job = {executor.submit(simulate_feature, job): job for job in missing}
        for done, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:  # keep successful work before failing loudly
                errors.append((job, repr(exc)))
                print(f"[simulate] ERROR {job}: {exc!r}", flush=True)
                continue
            records.append(result)
            save_feature_archive(records, archive)
            total_done = len(records)
            print(
                f"[simulate] {done}/{len(missing)} new; total={total_done}; "
                f"{result['label']}{result['replicate']:03d}; "
                f"loci={result['num_loci']}; worker={result['elapsed_seconds']:.1f}s; "
                f"wall={time.perf_counter() - start:.1f}s",
                flush=True,
            )
    if errors:
        raise RuntimeError(f"{len(errors)} simulations failed; first={errors[0]}")
    return records


def replicate_table(X: np.ndarray, direction: np.ndarray, groups: np.ndarray, magnitude: np.ndarray):
    """Exact 54-D aggregation used by scripts/direction_curve.py."""
    unique, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    order = np.lexsort((X[:, 0], inverse))
    per = np.bincount(inverse)
    if per.min() != per.max() or int(per[0]) != 198:
        raise ValueError("canonical replicates must each have depths g=2..199")
    table = X[order].reshape(len(unique), int(per[0]), X.shape[1])[:, :, 1:]
    features = np.concatenate([table.mean(axis=1), table.std(axis=1)], axis=1)
    return features, direction[first].astype("U1"), magnitude[first]


def load_canonical(canonical_root: Path):
    required = ("X.npy", "direction.npy", "groups.npy", "magnitude.npy")
    missing = [name for name in required if not (canonical_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"canonical root {canonical_root} lacks {missing}")
    X = np.load(canonical_root / "X.npy", mmap_mode="r")
    direction = np.load(canonical_root / "direction.npy", mmap_mode="r")
    groups = np.load(canonical_root / "groups.npy", mmap_mode="r")
    magnitude = np.load(canonical_root / "magnitude.npy", mmap_mode="r")
    features, labels, rates = replicate_table(X, direction, groups, magnitude)
    if features.shape[1] != 54:
        raise AssertionError(f"canonical aggregate is {features.shape}, not (*,54)")
    return features, labels, rates


def encode(labels: np.ndarray) -> np.ndarray:
    y = np.searchsorted(CLASSES, labels)
    if np.any(y >= len(CLASSES)) or not np.array_equal(CLASSES[y], labels):
        raise ValueError(f"unexpected direction labels: {np.unique(labels).tolist()}")
    return y.astype(np.int64)


def fit_frozen(X: np.ndarray, labels: np.ndarray):
    y = encode(labels)
    scale = StandardScaler().fit(X)
    model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs").fit(scale.transform(X), y)
    return scale, model


def unstratified_folds(n: int, seed: int, n_splits: int = 5):
    """Same deterministic fold constructor as scripts/direction_curve.py."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    fold = np.empty(n, dtype=np.int64)
    for f in range(n_splits):
        fold[permutation[f::n_splits]] = f
    for f in range(n_splits):
        yield np.where(fold != f)[0], np.where(fold == f)[0]


def repeated_oof(X: np.ndarray, y: np.ndarray, seeds: tuple[int, ...]):
    all_probability = []
    repeat_accuracy = []
    fold_audit = []
    for seed in seeds:
        probability = np.full((len(y), len(CLASSES)), np.nan, dtype=np.float64)
        for fold_index, (train, test) in enumerate(unstratified_folds(len(y), seed)):
            scale = StandardScaler().fit(X[train])
            model = LogisticRegression(C=1.0, max_iter=3000, solver="lbfgs").fit(
                scale.transform(X[train]), y[train]
            )
            probability[test] = model.predict_proba(scale.transform(X[test]))
            fold_audit.append({
                "repeat_seed": int(seed),
                "fold": int(fold_index),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_test_overlap": int(len(np.intersect1d(train, test))),
                "test_class_counts": {
                    str(CLASSES[k]): int((y[test] == k).sum()) for k in range(len(CLASSES))
                },
            })
        if np.isnan(probability).any():
            raise RuntimeError(f"repeat {seed} has incomplete OOF predictions")
        all_probability.append(probability)
        repeat_accuracy.append(float((probability.argmax(axis=1) == y).mean()))
    return np.mean(all_probability, axis=0), repeat_accuracy, fold_audit


def percentile_interval(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def wilson_interval(correct: int, n: int, z: float = 1.959963984540054) -> list[float]:
    p = correct / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [float(max(0.0, center - half)), float(min(1.0, center + half))]


def classification_report(
    y: np.ndarray,
    probability: np.ndarray,
    bootstrap_draws: int,
    bootstrap_seed: int,
) -> dict:
    prediction = probability.argmax(axis=1)
    rng = np.random.default_rng(bootstrap_seed)
    correct = prediction == y
    n = len(y)
    acc_boot = np.empty(bootstrap_draws, dtype=np.float64)
    balanced_boot = np.empty(bootstrap_draws, dtype=np.float64)
    class_boot = {k: np.empty(bootstrap_draws, dtype=np.float64) for k in range(3)}
    class_indices = {k: np.where(y == k)[0] for k in range(3)}
    for draw in range(bootstrap_draws):
        sample = rng.integers(0, n, n)
        acc_boot[draw] = correct[sample].mean()
        recalls = []
        for k in range(3):
            indices = class_indices[k]
            within = indices[rng.integers(0, len(indices), len(indices))]
            recall = correct[within].mean()
            class_boot[k][draw] = recall
            recalls.append(recall)
        balanced_boot[draw] = np.mean(recalls)

    per_class = {}
    confusion = np.zeros((3, 3), dtype=int)
    for truth, call in zip(y, prediction):
        confusion[truth, call] += 1
    for k, label in enumerate(CLASSES):
        use = y == k
        per_class[str(label)] = {
            "n": int(use.sum()),
            "correct": int(correct[use].sum()),
            "recall": float(correct[use].mean()),
            "recall_wilson_95_ci": wilson_interval(int(correct[use].sum()), int(use.sum())),
            "replicate_bootstrap_95_ci": percentile_interval(class_boot[k]),
        }
    recalls = [per_class[str(label)]["recall"] for label in CLASSES]
    return {
        "n": int(n),
        "accuracy": float(correct.mean()),
        "accuracy_wilson_95_ci": wilson_interval(int(correct.sum()), int(n)),
        "accuracy_replicate_bootstrap_95_ci": percentile_interval(acc_boot),
        "balanced_accuracy": float(np.mean(recalls)),
        "balanced_accuracy_stratified_bootstrap_95_ci": percentile_interval(balanced_boot),
        "per_class": per_class,
        "confusion_rows_true_columns_predicted_A_B_C": confusion.tolist(),
        "predicted_class_counts": {
            str(label): int((prediction == k).sum()) for k, label in enumerate(CLASSES)
        },
    }


def fit_and_score(
    canonical_X: np.ndarray,
    canonical_labels: np.ndarray,
    matched_X: np.ndarray,
    matched_y: np.ndarray,
    bootstrap_draws: int,
    bootstrap_seed: int,
):
    scale, model = fit_frozen(canonical_X, canonical_labels)
    probability = model.predict_proba(scale.transform(matched_X))
    report = classification_report(matched_y, probability, bootstrap_draws, bootstrap_seed)
    report.update({
        "canonical_training_n": int(len(canonical_labels)),
        "canonical_training_class_counts": {
            str(label): int((canonical_labels == label).sum()) for label in CLASSES
        },
        "fit_converged_iterations": model.n_iter_.astype(int).tolist(),
    })
    return scale, model, probability, report


def save_models(
    path: Path,
    appreciable_scale: StandardScaler,
    appreciable_model: LogisticRegression,
    all_scale: StandardScaler,
    all_model: LogisticRegression,
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.array(SCHEMA_VERSION),
            classes=CLASSES,
            feature_columns=np.array(FEATURE_COLUMNS, dtype="U60"),
            appreciable_scaler_mean=appreciable_scale.mean_,
            appreciable_scaler_scale=appreciable_scale.scale_,
            appreciable_coef=appreciable_model.coef_,
            appreciable_intercept=appreciable_model.intercept_,
            all_positive_scaler_mean=all_scale.mean_,
            all_positive_scaler_scale=all_scale.scale_,
            all_positive_coef=all_model.coef_,
            all_positive_intercept=all_model.intercept_,
        )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_diagnostics(arrays: dict[str, np.ndarray]) -> dict:
    report = {}
    for label in CLASSES:
        use = arrays["labels"] == label
        report[str(label)] = {
            "n": int(use.sum()),
            "rate": float(np.unique(arrays["rate"][use]).item()),
            "duration_generations": int(np.unique(arrays["duration"][use]).item()),
            "hazard": float(np.unique(arrays["hazard"][use]).item()),
            "num_mutated_sites_mean": float(arrays["num_sites"][use].mean()),
            "num_mutated_sites_sd": float(arrays["num_sites"][use].std()),
            "num_PADZE_loci_mean": float(arrays["num_loci"][use].mean()),
            "num_PADZE_loci_sd": float(arrays["num_loci"][use].std()),
        }
    return report


def write_summary(path: Path, result: dict) -> None:
    transfer = result["frozen_canonical_appreciable_transfer"]["metrics"]
    cv = result["matched_design_repeated_fivefold_cv"]["metrics"]
    lines = [
        "# Matched migration-exposure experiment",
        "",
        f"Completed {result['matched_batch']['n_replicates']} independent CPU-only msprime/PADZE "
        f"replicates ({result['matched_batch']['reps_per_class']} per class). Actual msprime epoch "
        "matrices were checked to give mT=0.25 for A, B, and C.",
        "",
        "## Main results",
        "",
        f"- Canonical-appreciable logistic, scored without refitting: {transfer['accuracy']:.3f} "
        f"(conditional Wilson 95% CI {transfer['accuracy_wilson_95_ci'][0]:.3f}--"
        f"{transfer['accuracy_wilson_95_ci'][1]:.3f}).",
        f"- Leakage-free repeated five-fold accuracy within the matched design: {cv['accuracy']:.3f} "
        f"(conditional Wilson 95% CI {cv['accuracy_wilson_95_ci'][0]:.3f}--"
        f"{cv['accuracy_wilson_95_ci'][1]:.3f}).",
        "",
        "Per-class recall (transfer / matched-design CV):",
        "",
    ]
    for label in CLASSES:
        t = transfer["per_class"][str(label)]
        c = cv["per_class"][str(label)]
        lines.append(f"- {label}: {t['recall']:.3f} / {c['recall']:.3f}")
    lines.extend([
        "",
        "The first score uses a scaler and logistic coefficients fit only to the existing canonical "
        "regen_full appreciable band, then applies them without refitting. It is a targeted joint "
        "rate--duration--ancestral-continuation sensitivity, not a preregistered external validation. "
        "The within-design result is based only on held-out predictions; no replicate appears in both "
        "train and test in any fold.",
        "",
        "See `results.json` for design epochs, confusion matrices, per-class bootstrap intervals, "
        "fold audits, software versions, and file hashes.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=30, help="replicates per class (minimum 30)")
    parser.add_argument("--workers", type=int, default=3, help="CPU worker processes")
    parser.add_argument("--seed-base", type=int, default=260_709_001)
    parser.add_argument("--cv-seeds", default="0,1,2,3,4")
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path(os.environ.get("DNNAIC_DATA", "data/simulation_data")) / "regen_full",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/matched_exposure_2026_07_09"),
    )
    parser.add_argument(
        "--simulate-only", action="store_true", help="checkpoint features but skip model evaluation"
    )
    args = parser.parse_args()
    if args.reps < 1:
        parser.error("--reps must be positive")
    if not args.simulate_only and args.reps < 30:
        parser.error("final evaluation requires at least 30 replicates per class")
    if args.workers < 1:
        parser.error("--workers must be positive")

    design_validation = validate_demographies()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.out_dir / "matched_features.npz"
    records = load_feature_archive(archive_path)
    jobs = make_jobs(args.reps, args.seed_base)
    expected_seed = {(label, replicate): (aseed, mseed) for label, replicate, aseed, mseed in jobs}
    for record in records:
        key = (record["label"], record["replicate"])
        if key in expected_seed and expected_seed[key] != (
            record["ancestry_seed"], record["mutation_seed"]
        ):
            raise RuntimeError(f"checkpoint seed mismatch for {key}")
    records = simulate_missing(records, jobs, min(args.workers, len(jobs)), archive_path)
    if args.simulate_only:
        print(f"[done] simulation checkpoint -> {archive_path}", flush=True)
        return 0

    wanted = {(label, replicate) for label, replicate, _, _ in jobs}
    selected = [r for r in records if (r["label"], r["replicate"]) in wanted]
    selected = sorted(selected, key=lambda r: (r["label"], r["replicate"]))
    if len(selected) != 3 * args.reps:
        raise RuntimeError(f"have {len(selected)} requested records, expected {3 * args.reps}")
    arrays = _record_arrays(selected)
    matched_X = arrays["X"].astype(np.float64)
    matched_y = encode(arrays["labels"])

    print("[model] aggregating canonical regen_full and fitting frozen models", flush=True)
    canonical_X, canonical_labels, canonical_rates = load_canonical(args.canonical_root)
    positive = canonical_labels != "D"
    appreciable = positive & (canonical_rates >= APPRECIABLE)
    app_scale, app_model, app_probability, app_report = fit_and_score(
        canonical_X[appreciable], canonical_labels[appreciable],
        matched_X, matched_y, args.bootstrap_draws, 2026070901,
    )
    all_scale, all_model, all_probability, all_report = fit_and_score(
        canonical_X[positive], canonical_labels[positive],
        matched_X, matched_y, args.bootstrap_draws, 2026070902,
    )

    cv_seeds = tuple(int(value) for value in args.cv_seeds.split(",") if value != "")
    if not cv_seeds:
        raise ValueError("at least one CV seed is required")
    cv_probability, repeat_accuracy, fold_audit = repeated_oof(matched_X, matched_y, cv_seeds)
    cv_report = classification_report(
        matched_y, cv_probability, args.bootstrap_draws, 2026070903
    )

    # Rewrite the compact archive in the exact selected order, now including
    # all three probability matrices.
    save_feature_archive(
        selected,
        archive_path,
        frozen_appreciable_probability=app_probability,
        frozen_all_positive_probability=all_probability,
        matched_cv_probability=cv_probability,
    )
    model_path = args.out_dir / "frozen_canonical_models.npz"
    save_models(model_path, app_scale, app_model, all_scale, all_model)

    result = {
        "schema_version": SCHEMA_VERSION,
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
        "design": {
            "haploid_coalescent_population_size": NE,
            "diploid_equivalent_effective_size": NE // 2,
            "global_ploidy": 1,
            "split_generations": [SPLIT_P12, SPLIT_P123, SPLIT_ROOT],
            "sequence_length_bp": SEQUENCE_LENGTH,
            "recombination_rate_per_bp_per_generation": RECOMBINATION_RATE,
            "mutation_rate_per_bp_per_generation": MUTATION_RATE,
            "haploid_gene_copies_per_sampled_population": GENE_COPIES_PER_POP,
            "sampled_populations": ["P1", "P2", "P3"],
            "unsampled_outgroup_present_in_demography": "P4",
            "rarefaction_depths_inclusive": [int(DEPTHS[0]), int(DEPTHS[-1])],
            "class_epoch_validation": design_validation,
        },
        "feature_representation": {
            "dimension": 54,
            "definition": (
                "population mean and population SD across g=2..199 for each of the 27 "
                "non-depth PADZE coordinates; identical to scripts/direction_curve.py"
            ),
            "curve_columns": CURVE_COLUMNS,
            "feature_columns": FEATURE_COLUMNS,
            "compact_archive": archive_path.name,
        },
        "matched_batch": {
            "n_replicates": int(len(selected)),
            "reps_per_class": int(args.reps),
            "fixed_seed_base": int(args.seed_base),
            "diagnostics_by_class": class_diagnostics(arrays),
        },
        "frozen_canonical_appreciable_transfer": {
            "canonical_dataset": {
                "id": args.canonical_root.name,
                "input_arrays": {
                    name: {"sha256": sha256_file(args.canonical_root / name)}
                    for name in ("X.npy", "direction.npy", "groups.npy", "magnitude.npy")
                },
            },
            "training_selection": "direction in A/B/C and migration rate >= 2.5e-4",
            "model": (
                "StandardScaler fit on canonical training only, then multinomial logistic "
                "regression (lbfgs, C=1, max_iter=3000); frozen before matched scoring"
            ),
            "metrics": app_report,
        },
        "frozen_canonical_all_positive_sensitivity": {
            "purpose": "training-scope sensitivity matching scripts/direction_curve.py full fit",
            "training_selection": "direction in A/B/C at every positive migration rate",
            "metrics": all_report,
        },
        "matched_design_repeated_fivefold_cv": {
            "model": "fold-local StandardScaler plus multinomial logistic (lbfgs, C=1)",
            "unit": "independent coalescent replicate (one 54-D row)",
            "n_folds": 5,
            "repeat_seeds": list(cv_seeds),
            "probability_aggregation": "mean OOF probability over repeats, then argmax",
            "per_repeat_accuracy": repeat_accuracy,
            "all_fold_train_test_overlap_zero": bool(
                all(entry["train_test_overlap"] == 0 for entry in fold_audit)
            ),
            "fold_audit": fold_audit,
            "metrics": cv_report,
        },
        "uncertainty": {
            "method": "nonparametric bootstrap over independent simulation replicates",
            "draws": int(args.bootstrap_draws),
            "interval": "percentile 95%",
            "per_class_recall_resampling": "within true class",
        },
        "artifacts": {
            archive_path.name: {"sha256": sha256_file(archive_path)},
            model_path.name: {"sha256": sha256_file(model_path)},
        },
    }
    result_path = args.out_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_path = args.out_dir / "SUMMARY.md"
    write_summary(summary_path, result)
    print(json.dumps({
        "frozen_canonical_appreciable_transfer": app_report,
        "matched_design_repeated_fivefold_cv": cv_report,
        "artifacts": [str(archive_path), str(model_path), str(result_path), str(summary_path)],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
