#!/usr/bin/env python3
"""Four-population, all-12-direction PADZE benchmark (CPU only).

This experiment extends the three-direction DNNaic study to every ordered edge among four
sampled populations.  The primary design is deliberately exchangeable: P1--P4 split from one
ancestor at the same time, have equal effective sizes, and experience the same integrated
migration exposure.  A class label ``Pi->Pj`` is a *forward-time* donor-to-recipient direction;
msprime therefore receives the reverse backwards-time migration matrix entry ``Pj -> Pi``.

Each independent coalescent replicate is summarized by the direct four-population PADZE
extension: alpha and pi for each population plus all six pair-private pihat statistics, each
with mean/variance/SE across loci.  Mean and population SD across rarefaction depth give one
84-D vector per replicate.  Evaluation is strictly replicate-level and includes:

* multinomial logistic regression;
* permutation-equivariant logistic regression (all S4 relabelings, training folds only);
* an RMT LDA whose within-class covariance bulk is shrunk at the Marchenko--Pastur edge;
* an RBF kernel, a small MLP, and an RBF kernel on the MLP's penultimate features;
* label-permutation and population-label-removal negative controls;
* frozen transfer to weak-signal, unequal-Ne, balanced-tree, and half-sequence shifts.

Simulation checkpoints are atomically updated in small batches and at normal exit.  No GPU
library is used.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import multiprocessing
import os
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import msprime
import numpy as np
import padze
import sklearn
from padze import LociData, Metadata, compute_features
from scipy.integrate import cumulative_trapezoid
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


SCHEMA_VERSION = "dnnaic-four-pop-twelve-direction-v1"
POPULATIONS = ("P1", "P2", "P3", "P4")
N_POPULATIONS = len(POPULATIONS)
DIRECTIONS = tuple(
    f"{donor}->{recipient}"
    for donor in POPULATIONS
    for recipient in POPULATIONS
    if donor != recipient
)
CLASS_INDEX = {label: index for index, label in enumerate(DIRECTIONS)}
PERMUTATIONS = tuple(itertools.permutations(range(N_POPULATIONS)))

ANCESTRAL_NE = 10_000
MIGRATION_DURATION = 250
RECOMBINATION_RATE = 1.78e-8
MUTATION_RATE = 2e-8
GENE_COPIES_PER_POP = 48
DEPTHS = np.arange(2, 41, dtype=np.int64)
MOMENTS = ("mean", "variance", "se")

PAIR_KEYS = tuple(
    f"pihat_{i}{j}" for i in range(1, N_POPULATIONS + 1)
    for j in range(i + 1, N_POPULATIONS + 1)
)
STAT_KEYS = (
    tuple(f"alpha_{i}" for i in range(1, N_POPULATIONS + 1))
    + tuple(f"pi_{i}" for i in range(1, N_POPULATIONS + 1))
    + PAIR_KEYS
)
CURVE_COLUMNS = tuple(
    f"{stat}_{moment}" for stat in STAT_KEYS for moment in MOMENTS
)
FEATURE_COLUMNS = tuple(
    f"{aggregation}__{column}"
    for aggregation in ("depth_mean", "depth_sd")
    for column in CURVE_COLUMNS
)


FAMILY_SPECS = {
    "baseline": {
        "tree": "star",
        "leaf_ne": (10_000, 10_000, 10_000, 10_000),
        "hazard": 0.25,
        "sequence_length": 250_000,
        "description": "exchangeable four-tip star, equal Ne, mT=0.25",
    },
    "weak_signal": {
        "tree": "star",
        "leaf_ne": (10_000, 10_000, 10_000, 10_000),
        "hazard": 0.05,
        "sequence_length": 250_000,
        "description": "baseline demography with fivefold weaker integrated exposure",
    },
    "unequal_ne": {
        "tree": "star",
        "leaf_ne": (5_000, 10_000, 20_000, 8_000),
        "hazard": 0.25,
        "sequence_length": 250_000,
        "description": "fixed population-specific Ne imbalance",
    },
    "balanced_tree": {
        "tree": "balanced",
        "leaf_ne": (10_000, 10_000, 10_000, 10_000),
        "hazard": 0.25,
        "sequence_length": 250_000,
        "description": "((P1,P2),(P3,P4)) topology instead of a star",
    },
    "half_sequence": {
        "tree": "star",
        "leaf_ne": (10_000, 10_000, 10_000, 10_000),
        "hazard": 0.25,
        "sequence_length": 125_000,
        "description": "half the sequence length and therefore fewer loci",
    },
}


def split_direction(label: str) -> tuple[str, str]:
    donor, recipient = label.split("->")
    if donor not in POPULATIONS or recipient not in POPULATIONS or donor == recipient:
        raise ValueError(f"invalid direction {label!r}")
    return donor, recipient


def make_demography(family: str, label: str) -> msprime.Demography:
    """Return a demography with an explicitly stopped forward donor->recipient edge."""
    spec = FAMILY_SPECS[family]
    donor, recipient = split_direction(label)
    demography = msprime.Demography()
    for population, size in zip(POPULATIONS, spec["leaf_ne"]):
        demography.add_population(name=population, initial_size=size)
    if spec["tree"] == "star":
        demography.add_population(name="P0", initial_size=ANCESTRAL_NE)
        demography.add_population_split(
            time=500, derived=list(POPULATIONS), ancestral="P0"
        )
    elif spec["tree"] == "balanced":
        for ancestor in ("P12", "P34", "P0"):
            demography.add_population(name=ancestor, initial_size=ANCESTRAL_NE)
        demography.add_population_split(
            time=400, derived=["P1", "P2"], ancestral="P12"
        )
        demography.add_population_split(
            time=400, derived=["P3", "P4"], ancestral="P34"
        )
        demography.add_population_split(
            time=800, derived=["P12", "P34"], ancestral="P0"
        )
    else:
        raise ValueError(f"unknown tree {spec['tree']!r}")

    rate = float(spec["hazard"] / MIGRATION_DURATION)
    # Forward donor->recipient is backwards recipient->donor in msprime.
    demography.set_migration_rate(source=recipient, dest=donor, rate=rate)
    demography.add_migration_rate_change(
        time=MIGRATION_DURATION, source=recipient, dest=donor, rate=0.0
    )
    demography.sort_events()
    return demography


def validate_demography(family: str, label: str) -> dict:
    """Check the actual epoch matrices, not just the requested API calls."""
    demography = make_demography(family, label)
    donor, recipient = split_direction(label)
    expected_rate = float(FAMILY_SPECS[family]["hazard"] / MIGRATION_DURATION)
    observed = []
    epochs = []
    for epoch in demography.debug().epochs:
        start = float(epoch.start_time)
        end = float(epoch.end_time)
        nonzero = []
        for source, dest in np.argwhere(epoch.migration_matrix > 0):
            entry = {
                "source": demography.populations[int(source)].name,
                "dest": demography.populations[int(dest)].name,
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
    expected = [(0.0, float(MIGRATION_DURATION), recipient, donor, expected_rate)]
    if observed != expected:
        raise AssertionError(f"{family}/{label}: epochs {observed!r} != {expected!r}")
    hazard = sum((end - start) * rate for start, end, _, _, rate in observed)
    target = float(FAMILY_SPECS[family]["hazard"])
    if not np.isclose(hazard, target, rtol=0, atol=1e-12):
        raise AssertionError(f"{family}/{label}: hazard {hazard} != {target}")
    return {
        "forward_time_label": label,
        "backwards_time_msprime_mapping": f"{recipient}->{donor}",
        "rate_per_generation": expected_rate,
        "duration_generations": MIGRATION_DURATION,
        "integrated_single_lineage_hazard_m_times_T": hazard,
        "probability_at_least_one_migration": float(-math.expm1(-hazard)),
        "actual_epochs": epochs,
    }


def validate_all_demographies(families: Iterable[str]) -> dict:
    return {
        family: {label: validate_demography(family, label) for label in DIRECTIONS}
        for family in families
    }


def tree_sequence_to_loci(ts: msprime.TreeSequence) -> LociData:
    samples = ts.samples()
    sample_population = ts.tables.nodes.population[samples]
    masks = [sample_population == index for index in range(N_POPULATIONS)]
    sizes = np.array([int(mask.sum()) for mask in masks], dtype=np.int64)
    expected = np.full(N_POPULATIONS, GENE_COPIES_PER_POP, dtype=np.int64)
    if not np.array_equal(sizes, expected):
        raise AssertionError(f"gene-copy counts {sizes.tolist()} != {expected.tolist()}")

    genotype = ts.genotype_matrix()
    count_matrices = []
    sample_sizes = []
    for row in genotype:
        maximum = int(row.max())
        if maximum < 1:
            continue
        counts = np.stack([
            np.bincount(row[mask], minlength=maximum + 1) for mask in masks
        ]).astype(np.int64, copy=False)
        if int((counts.sum(axis=0) > 0).sum()) < 2:
            continue
        count_matrices.append(counts)
        sample_sizes.append(counts.sum(axis=1))
    size_matrix = (
        np.vstack(sample_sizes).astype(np.int64, copy=False)
        if sample_sizes else np.zeros((0, N_POPULATIONS), dtype=np.int64)
    )
    metadata = Metadata(
        source="msprime",
        populations=list(POPULATIONS),
        sample_ids={population: [] for population in POPULATIONS},
        ploidy={population: 1 for population in POPULATIONS},
        n_loci_read=int(ts.num_sites),
        n_loci_kept=len(count_matrices),
        filters_applied=["polymorphic across P1/P2/P3/P4"],
        missing_fraction=0.0,
    )
    return LociData(
        list(POPULATIONS), count_matrices, size_matrix,
        [f"s{index}" for index in range(len(count_matrices))], metadata,
    )


def feature_from_loci(loci: LociData) -> np.ndarray:
    table = compute_features(
        loci,
        depths=DEPTHS,
        pihat_sizes=(2,),
        moments=MOMENTS,
        bias_corrected=True,
    )
    matrix, columns = table.to_frame()
    index = {name: i for i, name in enumerate(columns)}
    wanted = [index[column] for column in CURVE_COLUMNS]
    curve = matrix[:, wanted].astype(np.float64)
    if curve.shape != (len(DEPTHS), len(CURVE_COLUMNS)):
        raise AssertionError(f"unexpected curve shape {curve.shape}")
    feature = np.concatenate([curve.mean(axis=0), curve.std(axis=0)], axis=0)
    if feature.shape != (len(FEATURE_COLUMNS),) or not np.isfinite(feature).all():
        raise AssertionError("non-finite or incorrectly shaped four-population feature")
    return feature


@dataclass(frozen=True)
class Job:
    family: str
    label: str
    replicate: int
    ancestry_seed: int
    mutation_seed: int


def make_jobs(
    families: Iterable[str], baseline_reps: int, shift_reps: int, seed_base: int
) -> list[Job]:
    jobs = []
    for family_index, family in enumerate(FAMILY_SPECS):
        if family not in families:
            continue
        reps = baseline_reps if family == "baseline" else shift_reps
        for class_index, label in enumerate(DIRECTIONS):
            for replicate in range(reps):
                ancestry_seed = seed_base + family_index * 2_000_000 + class_index * 10_000 + 2 * replicate
                mutation_seed = ancestry_seed + 1
                if not (0 < ancestry_seed < 2**32 and 0 < mutation_seed < 2**32):
                    raise ValueError("msprime seeds must lie in [1, 2**32-1]")
                jobs.append(Job(family, label, replicate, ancestry_seed, mutation_seed))
    return jobs


def simulate_feature(job: Job) -> dict:
    started = time.perf_counter()
    spec = FAMILY_SPECS[job.family]
    ts = msprime.sim_ancestry(
        samples={population: GENE_COPIES_PER_POP for population in POPULATIONS},
        sequence_length=int(spec["sequence_length"]),
        recombination_rate=RECOMBINATION_RATE,
        ploidy=1,
        demography=make_demography(job.family, job.label),
        random_seed=job.ancestry_seed,
    )
    ts = msprime.sim_mutations(
        ts, rate=MUTATION_RATE, random_seed=job.mutation_seed, keep=True
    )
    loci = tree_sequence_to_loci(ts)
    if loci.metadata.n_loci_kept < 2:
        raise RuntimeError(f"{job.family}/{job.label}/{job.replicate}: too few loci")
    donor, recipient = split_direction(job.label)
    return {
        "family": job.family,
        "label": job.label,
        "donor": donor,
        "recipient": recipient,
        "replicate": job.replicate,
        "ancestry_seed": job.ancestry_seed,
        "mutation_seed": job.mutation_seed,
        "hazard": float(spec["hazard"]),
        "sequence_length": int(spec["sequence_length"]),
        "num_sites": int(ts.num_sites),
        "num_loci": int(loci.metadata.n_loci_kept),
        "elapsed_seconds": float(time.perf_counter() - started),
        "feature": feature_from_loci(loci).astype(np.float32),
    }


def record_key(record: dict) -> tuple[str, str, int]:
    return record["family"], record["label"], int(record["replicate"])


def record_arrays(records: list[dict]) -> dict[str, np.ndarray]:
    records = sorted(records, key=record_key)
    if not records:
        return {}
    return {
        "X": np.vstack([record["feature"] for record in records]).astype(np.float32),
        "family": np.array([record["family"] for record in records], dtype="U24"),
        "label": np.array([record["label"] for record in records], dtype="U8"),
        "donor": np.array([record["donor"] for record in records], dtype="U2"),
        "recipient": np.array([record["recipient"] for record in records], dtype="U2"),
        "replicate": np.array([record["replicate"] for record in records], dtype=np.int32),
        "ancestry_seed": np.array([record["ancestry_seed"] for record in records], dtype=np.int64),
        "mutation_seed": np.array([record["mutation_seed"] for record in records], dtype=np.int64),
        "hazard": np.array([record["hazard"] for record in records], dtype=np.float64),
        "sequence_length": np.array([record["sequence_length"] for record in records], dtype=np.int64),
        "num_sites": np.array([record["num_sites"] for record in records], dtype=np.int32),
        "num_loci": np.array([record["num_loci"] for record in records], dtype=np.int32),
        "elapsed_seconds": np.array([record["elapsed_seconds"] for record in records], dtype=np.float64),
    }


def save_records(records: list[dict], path: Path) -> None:
    arrays = record_arrays(records)
    arrays.update({
        "schema_version": np.array(SCHEMA_VERSION),
        "directions": np.array(DIRECTIONS, dtype="U8"),
        "depths": DEPTHS,
        "curve_columns": np.array(CURVE_COLUMNS, dtype="U40"),
        "feature_columns": np.array(FEATURE_COLUMNS, dtype="U64"),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as archive:
        if str(archive["schema_version"].item()) != SCHEMA_VERSION:
            raise RuntimeError(f"checkpoint {path} has an incompatible schema")
        records = []
        for index in range(len(archive["label"])):
            records.append({
                "family": str(archive["family"][index]),
                "label": str(archive["label"][index]),
                "donor": str(archive["donor"][index]),
                "recipient": str(archive["recipient"][index]),
                "replicate": int(archive["replicate"][index]),
                "ancestry_seed": int(archive["ancestry_seed"][index]),
                "mutation_seed": int(archive["mutation_seed"][index]),
                "hazard": float(archive["hazard"][index]),
                "sequence_length": int(archive["sequence_length"][index]),
                "num_sites": int(archive["num_sites"][index]),
                "num_loci": int(archive["num_loci"][index]),
                "elapsed_seconds": float(archive["elapsed_seconds"][index]),
                "feature": archive["X"][index].astype(np.float32),
            })
    return records


def simulate_missing(
    records: list[dict], jobs: list[Job], workers: int, checkpoint: Path
) -> list[dict]:
    by_key = {record_key(record): record for record in records}
    missing = [job for job in jobs if (job.family, job.label, job.replicate) not in by_key]
    expected_seeds = {
        (job.family, job.label, job.replicate): (job.ancestry_seed, job.mutation_seed)
        for job in jobs
    }
    for key, record in by_key.items():
        if key in expected_seeds and expected_seeds[key] != (
            record["ancestry_seed"], record["mutation_seed"]
        ):
            raise RuntimeError(f"checkpoint seed mismatch for {key}")
    if not missing:
        print(f"[simulate] checkpoint already contains all {len(jobs)} requested replicates")
        return list(by_key.values())

    print(f"[simulate] {len(missing)} missing of {len(jobs)} requested; workers={workers}", flush=True)
    completed = 0
    checkpoint_every = 12
    if workers == 1:
        for job in missing:
            record = simulate_feature(job)
            by_key[record_key(record)] = record
            completed += 1
            if completed % checkpoint_every == 0:
                save_records(list(by_key.values()), checkpoint)
            print(
                f"  [{completed}/{len(missing)}] {job.family} {job.label} r{job.replicate} "
                f"loci={record['num_loci']} {record['elapsed_seconds']:.2f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(simulate_feature, job): job for job in missing}
            for future in as_completed(futures):
                job = futures[future]
                record = future.result()
                by_key[record_key(record)] = record
                completed += 1
                if completed % checkpoint_every == 0:
                    save_records(list(by_key.values()), checkpoint)
                print(
                    f"  [{completed}/{len(missing)}] {job.family} {job.label} r{job.replicate} "
                    f"loci={record['num_loci']} {record['elapsed_seconds']:.2f}s",
                    flush=True,
                )
    save_records(list(by_key.values()), checkpoint)
    return list(by_key.values())


def encode(labels: np.ndarray) -> np.ndarray:
    try:
        return np.array([CLASS_INDEX[str(label)] for label in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"unexpected direction label {exc}") from exc


def _transform_stat_key(stat: str, permutation: tuple[int, ...]) -> str:
    if stat.startswith("alpha_") or stat.startswith("pi_"):
        prefix, raw = stat.rsplit("_", 1)
        return f"{prefix}_{permutation[int(raw) - 1] + 1}"
    if stat.startswith("pihat_"):
        raw = stat.split("_", 1)[1]
        mapped = sorted(permutation[int(value) - 1] + 1 for value in raw)
        return "pihat_" + "".join(str(value) for value in mapped)
    raise ValueError(stat)


def feature_target_map(permutation: tuple[int, ...]) -> np.ndarray:
    source_to_target = np.empty(len(FEATURE_COLUMNS), dtype=np.int64)
    lookup = {column: index for index, column in enumerate(FEATURE_COLUMNS)}
    for source, column in enumerate(FEATURE_COLUMNS):
        aggregation, rest = column.split("__", 1)
        moment = next(moment for moment in MOMENTS if rest.endswith("_" + moment))
        stat = rest[: -(len(moment) + 1)]
        target_column = f"{aggregation}__{_transform_stat_key(stat, permutation)}_{moment}"
        source_to_target[source] = lookup[target_column]
    if len(np.unique(source_to_target)) != len(source_to_target):
        raise AssertionError("population relabeling is not a feature permutation")
    return source_to_target


PERM_FEATURE_TARGETS = tuple(feature_target_map(permutation) for permutation in PERMUTATIONS)
PERM_LABEL_TARGETS = tuple(
    np.array([
        CLASS_INDEX[
            f"P{permutation[int(split_direction(label)[0][1:]) - 1] + 1}"
            f"->P{permutation[int(split_direction(label)[1][1:]) - 1] + 1}"
        ]
        for label in DIRECTIONS
    ], dtype=np.int64)
    for permutation in PERMUTATIONS
)


def permute_features(X: np.ndarray, source_to_target: np.ndarray) -> np.ndarray:
    out = np.empty_like(X)
    out[:, source_to_target] = X
    return out


def equivariant_augmentation(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = []
    labels = []
    for target_map, label_map in zip(PERM_FEATURE_TARGETS, PERM_LABEL_TARGETS):
        features.append(permute_features(X, target_map))
        labels.append(label_map[y])
    return np.vstack(features), np.concatenate(labels)


def label_invariant_features(X: np.ndarray) -> np.ndarray:
    """Remove population identities by sorting each exchangeable statistic orbit."""
    out = X.copy()
    lookup = {column: index for index, column in enumerate(FEATURE_COLUMNS)}
    for aggregation in ("depth_mean", "depth_sd"):
        for moment in MOMENTS:
            groups = [
                [lookup[f"{aggregation}__alpha_{i}_{moment}"] for i in range(1, 5)],
                [lookup[f"{aggregation}__pi_{i}_{moment}"] for i in range(1, 5)],
                [lookup[f"{aggregation}__{key}_{moment}"] for key in PAIR_KEYS],
            ]
            for indices in groups:
                out[:, indices] = np.sort(out[:, indices], axis=1)
    return out


def mp_median(aspect: float) -> float:
    """Numerical median of the unit-scale Marchenko--Pastur continuous law, q<1."""
    if not 0 < aspect < 1:
        raise ValueError("MP median estimator requires 0 < q < 1")
    lower = (1.0 - math.sqrt(aspect)) ** 2
    upper = (1.0 + math.sqrt(aspect)) ** 2
    grid = np.linspace(lower + 1e-10, upper - 1e-10, 20_001)
    density = np.sqrt((upper - grid) * (grid - lower)) / (2.0 * math.pi * aspect * grid)
    cdf = cumulative_trapezoid(density, grid, initial=0.0)
    cdf /= cdf[-1]
    return float(np.interp(0.5, cdf, grid))


class MPBulkLDA:
    """Equal-prior LDA with a Marchenko--Pastur bulk covariance shrinker."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MPBulkLDA":
        self.classes_ = np.unique(y)
        self.means_ = np.vstack([X[y == cls].mean(axis=0) for cls in self.classes_])
        residual = np.vstack([X[y == cls] - self.means_[i] for i, cls in enumerate(self.classes_)])
        degrees = max(1, len(X) - len(self.classes_))
        covariance = residual.T @ residual / degrees
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, np.finfo(float).eps)
        aspect = X.shape[1] / degrees
        if aspect >= 1:
            # The primary benchmark has q<1.  This conservative fallback keeps small-shift
            # within-family audits finite without pretending the zero-eigenvalue atom is data.
            positive = eigenvalues[eigenvalues > np.finfo(float).eps * 100]
            noise = float(np.median(positive)) if len(positive) else 1.0
            threshold = noise * (1.0 + math.sqrt(aspect)) ** 2
        else:
            noise = float(np.median(eigenvalues) / mp_median(aspect))
            threshold = noise * (1.0 + math.sqrt(aspect)) ** 2
        shrunk = np.full_like(eigenvalues, noise)
        spikes = eigenvalues > threshold
        for index in np.where(spikes)[0]:
            sample = float(eigenvalues[index])
            discriminant = (sample - (aspect - 1.0) * noise) ** 2 - 4.0 * sample * noise
            if discriminant > 0:
                population = 0.5 * (
                    sample - (aspect - 1.0) * noise + math.sqrt(discriminant)
                )
                shrunk[index] = max(noise, population)
            else:
                shrunk[index] = sample
        self.inverse_covariance_ = (eigenvectors / shrunk) @ eigenvectors.T
        self.whitener_ = (eigenvectors / np.sqrt(shrunk)) @ eigenvectors.T
        self.aspect_ratio_ = float(aspect)
        self.noise_variance_ = float(noise)
        self.bulk_edge_ = float(threshold)
        self.n_spikes_ = int(spikes.sum())
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        linear = X @ self.inverse_covariance_ @ self.means_.T
        offset = 0.5 * np.einsum(
            "ki,ij,kj->k", self.means_, self.inverse_covariance_, self.means_
        )
        return linear - offset

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[self.decision_function(X).argmax(axis=1)]


def hidden_features(model: MLPClassifier, X_scaled: np.ndarray) -> np.ndarray:
    hidden = X_scaled
    for weights, bias in zip(model.coefs_[:-1], model.intercepts_[:-1]):
        hidden = np.maximum(0.0, hidden @ weights + bias)
    return hidden


def fit_mlp(X: np.ndarray, y: np.ndarray, seed: int) -> MLPClassifier:
    model = MLPClassifier(
        hidden_layer_sizes=(48, 24), activation="relu", solver="lbfgs",
        alpha=1e-2, max_iter=600, random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    return model


MODEL_NAMES = (
    "logistic",
    "equivariant_logistic",
    "mp_bulk_lda",
    "rbf_kernel",
    "mlp",
    "deep_feature_rbf",
    "deep_mp_rbf",
    "label_invariant_logistic",
)


def fit_predict_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    model_names: Iterable[str] = MODEL_NAMES,
) -> tuple[dict[str, np.ndarray], dict]:
    names = set(model_names)
    predictions: dict[str, np.ndarray] = {}
    diagnostics: dict = {}
    scale = StandardScaler().fit(X_train)
    train_scaled = scale.transform(X_train)
    test_scaled = scale.transform(X_test)

    if "logistic" in names:
        model = LogisticRegression(C=1.0, max_iter=3_000, solver="lbfgs")
        model.fit(train_scaled, y_train)
        predictions["logistic"] = model.predict(test_scaled)

    if "mp_bulk_lda" in names:
        model = MPBulkLDA().fit(train_scaled, y_train)
        predictions["mp_bulk_lda"] = model.predict(test_scaled)
        diagnostics["mp_bulk_lda"] = {
            "aspect_ratio": model.aspect_ratio_,
            "noise_variance": model.noise_variance_,
            "bulk_edge": model.bulk_edge_,
            "n_spikes": model.n_spikes_,
        }

    if "rbf_kernel" in names:
        model = SVC(C=1.0, kernel="rbf", gamma="scale")
        model.fit(train_scaled, y_train)
        predictions["rbf_kernel"] = model.predict(test_scaled)

    need_mlp = bool({"mlp", "deep_feature_rbf", "deep_mp_rbf"} & names)
    if need_mlp:
        mlp = fit_mlp(train_scaled, y_train, seed)
        if "mlp" in names:
            predictions["mlp"] = mlp.predict(test_scaled)
        if {"deep_feature_rbf", "deep_mp_rbf"} & names:
            train_hidden = hidden_features(mlp, train_scaled)
            test_hidden = hidden_features(mlp, test_scaled)
            hidden_scale = StandardScaler().fit(train_hidden)
            train_hidden_scaled = hidden_scale.transform(train_hidden)
            test_hidden_scaled = hidden_scale.transform(test_hidden)
        if "deep_feature_rbf" in names:
            kernel = SVC(C=1.0, kernel="rbf", gamma="scale")
            kernel.fit(train_hidden_scaled, y_train)
            predictions["deep_feature_rbf"] = kernel.predict(
                test_hidden_scaled
            )
        if "deep_mp_rbf" in names:
            rmt = MPBulkLDA().fit(train_hidden_scaled, y_train)
            kernel = SVC(C=1.0, kernel="rbf", gamma="scale")
            kernel.fit(train_hidden_scaled @ rmt.whitener_, y_train)
            predictions["deep_mp_rbf"] = kernel.predict(
                test_hidden_scaled @ rmt.whitener_
            )
            diagnostics["deep_mp_rbf"] = {
                "aspect_ratio": rmt.aspect_ratio_,
                "noise_variance": rmt.noise_variance_,
                "bulk_edge": rmt.bulk_edge_,
                "n_spikes": rmt.n_spikes_,
            }
        diagnostics["mlp"] = {"iterations": int(mlp.n_iter_), "loss": float(mlp.loss_)}

    if "equivariant_logistic" in names:
        augmented_X, augmented_y = equivariant_augmentation(X_train, y_train)
        augmented_scale = StandardScaler().fit(augmented_X)
        model = LogisticRegression(C=1.0, max_iter=3_000, solver="lbfgs")
        model.fit(augmented_scale.transform(augmented_X), augmented_y)
        predictions["equivariant_logistic"] = model.predict(
            augmented_scale.transform(X_test)
        )

    if "label_invariant_logistic" in names:
        invariant_train = label_invariant_features(X_train)
        invariant_test = label_invariant_features(X_test)
        invariant_scale = StandardScaler().fit(invariant_train)
        model = LogisticRegression(C=1.0, max_iter=3_000, solver="lbfgs")
        model.fit(invariant_scale.transform(invariant_train), y_train)
        predictions["label_invariant_logistic"] = model.predict(
            invariant_scale.transform(invariant_test)
        )
    return predictions, diagnostics


def majority_vote(predictions: np.ndarray, n_classes: int) -> tuple[np.ndarray, int]:
    voted = np.empty(predictions.shape[1], dtype=np.int64)
    ties = 0
    for column in range(predictions.shape[1]):
        counts = np.bincount(predictions[:, column], minlength=n_classes)
        maxima = np.flatnonzero(counts == counts.max())
        if len(maxima) > 1:
            ties += 1
            first = int(predictions[0, column])
            voted[column] = first if first in maxima else int(maxima[0])
        else:
            voted[column] = int(maxima[0])
    return voted, ties


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return [max(0.0, center - half), min(1.0, center + half)]


def direction_report(y: np.ndarray, prediction: np.ndarray) -> dict:
    exact = prediction == y
    truth_labels = np.array(DIRECTIONS, dtype="U8")[y]
    pred_labels = np.array(DIRECTIONS, dtype="U8")[prediction]
    truth_parts = [split_direction(label) for label in truth_labels]
    pred_parts = [split_direction(label) for label in pred_labels]
    truth_donor = np.array([part[0] for part in truth_parts])
    truth_recipient = np.array([part[1] for part in truth_parts])
    pred_donor = np.array([part[0] for part in pred_parts])
    pred_recipient = np.array([part[1] for part in pred_parts])
    pair_correct = np.array([
        frozenset(truth) == frozenset(pred)
        for truth, pred in zip(truth_parts, pred_parts)
    ])
    per_class = {}
    for index, label in enumerate(DIRECTIONS):
        use = y == index
        successes = int(exact[use].sum())
        per_class[label] = {
            "n": int(use.sum()),
            "correct": successes,
            "recall": float(exact[use].mean()),
            "wilson_95": wilson_interval(successes, int(use.sum())),
        }
    correct = int(exact.sum())
    pair_n = int(pair_correct.sum())
    return {
        "n": int(len(y)),
        "exact_12way_accuracy": float(exact.mean()),
        "exact_correct": correct,
        "exact_wilson_95": wilson_interval(correct, len(y)),
        "unordered_pair_accuracy": float(pair_correct.mean()),
        "orientation_accuracy_conditional_on_correct_pair": (
            float(exact[pair_correct].mean()) if pair_n else float("nan")
        ),
        "donor_accuracy": float((truth_donor == pred_donor).mean()),
        "recipient_accuracy": float((truth_recipient == pred_recipient).mean()),
        "macro_recall": float(np.mean([entry["recall"] for entry in per_class.values()])),
        "per_class": per_class,
        "confusion_rows_true_columns_predicted": confusion_matrix(
            y, prediction, labels=np.arange(len(DIRECTIONS))
        ).astype(int).tolist(),
    }


def repeated_cv(
    X: np.ndarray,
    y: np.ndarray,
    seeds: tuple[int, ...],
    folds: int,
    model_names: Iterable[str] = MODEL_NAMES,
) -> dict:
    names = tuple(model_names)
    class_counts = np.bincount(y, minlength=len(DIRECTIONS))
    if np.any(class_counts < folds):
        raise ValueError(f"each class needs >= {folds} rows; counts={class_counts.tolist()}")
    repeat_predictions = {
        name: np.full((len(seeds), len(y)), -1, dtype=np.int64) for name in names
    }
    repeat_accuracy = {name: [] for name in names}
    fold_audit = []
    diagnostic_rows = []
    for repeat_index, seed in enumerate(seeds):
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(X, y)):
            predictions, diagnostics = fit_predict_models(
                X[train], y[train], X[test], seed * 100 + fold, names
            )
            for name in names:
                repeat_predictions[name][repeat_index, test] = predictions[name]
            fold_audit.append({
                "repeat_seed": int(seed),
                "fold": int(fold),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "train_test_overlap": int(len(np.intersect1d(train, test))),
                "test_class_counts": np.bincount(
                    y[test], minlength=len(DIRECTIONS)
                ).astype(int).tolist(),
            })
            if diagnostics:
                diagnostic_rows.append({
                    "repeat_seed": int(seed), "fold": int(fold), **diagnostics
                })
        for name in names:
            if np.any(repeat_predictions[name][repeat_index] < 0):
                raise RuntimeError(f"incomplete OOF predictions for {name}, seed {seed}")
            repeat_accuracy[name].append(float(
                (repeat_predictions[name][repeat_index] == y).mean()
            ))
    reports = {}
    for name in names:
        voted, ties = majority_vote(repeat_predictions[name], len(DIRECTIONS))
        reports[name] = {
            "majority_vote": direction_report(y, voted),
            "per_repeat_exact_accuracy": repeat_accuracy[name],
            "mean_repeat_exact_accuracy": float(np.mean(repeat_accuracy[name])),
            "sd_repeat_exact_accuracy": float(np.std(repeat_accuracy[name])),
            "majority_vote_ties": int(ties),
        }
    return {
        "folds": int(folds),
        "repeat_seeds": list(seeds),
        "all_train_test_overlaps_zero": bool(
            all(entry["train_test_overlap"] == 0 for entry in fold_audit)
        ),
        "fold_audit": fold_audit,
        "model_diagnostics": diagnostic_rows,
        "models": reports,
    }


def reversed_pair_cv(
    X: np.ndarray, y: np.ndarray, seeds: tuple[int, ...], folds: int
) -> dict:
    """Orient each edge when its unordered population pair is supplied by an oracle."""
    repeat_prediction = np.full((len(seeds), len(y)), -1, dtype=np.int64)
    fold_audit = []
    pairs = tuple(itertools.combinations(POPULATIONS, 2))
    for first, second in pairs:
        forward = CLASS_INDEX[f"{first}->{second}"]
        reverse = CLASS_INDEX[f"{second}->{first}"]
        use = np.where((y == forward) | (y == reverse))[0]
        binary = (y[use] == reverse).astype(np.int64)
        for repeat_index, seed in enumerate(seeds):
            splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
            for fold, (train_local, test_local) in enumerate(splitter.split(X[use], binary)):
                train = use[train_local]
                test = use[test_local]
                scale = StandardScaler().fit(X[train])
                model = LogisticRegression(C=1.0, max_iter=3_000, solver="lbfgs").fit(
                    scale.transform(X[train]), binary[train_local]
                )
                call = model.predict(scale.transform(X[test]))
                repeat_prediction[repeat_index, test] = np.where(call == 0, forward, reverse)
                fold_audit.append({
                    "pair": f"{first}<->{second}",
                    "repeat_seed": int(seed),
                    "fold": int(fold),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "train_test_overlap": int(len(np.intersect1d(train, test))),
                })
    if np.any(repeat_prediction < 0):
        raise RuntimeError("incomplete reversed-pair OOF predictions")
    voted, ties = majority_vote(repeat_prediction, len(DIRECTIONS))
    correct = voted == y
    per_pair = {}
    for first, second in pairs:
        labels = {
            CLASS_INDEX[f"{first}->{second}"], CLASS_INDEX[f"{second}->{first}"]
        }
        use = np.array([label in labels for label in y])
        successes = int(correct[use].sum())
        per_pair[f"{first}<->{second}"] = {
            "n": int(use.sum()),
            "accuracy": float(correct[use].mean()),
            "correct": successes,
            "wilson_95": wilson_interval(successes, int(use.sum())),
        }
    successes = int(correct.sum())
    return {
        "task": "binary donor-versus-recipient orientation conditional on the true unordered pair",
        "chance": 0.5,
        "n": int(len(y)),
        "accuracy": float(correct.mean()),
        "correct": successes,
        "wilson_95": wilson_interval(successes, len(y)),
        "per_repeat_accuracy": [
            float((repeat_prediction[index] == y).mean()) for index in range(len(seeds))
        ],
        "per_pair": per_pair,
        "majority_vote_ties": int(ties),
        "all_train_test_overlaps_zero": bool(
            all(entry["train_test_overlap"] == 0 for entry in fold_audit)
        ),
        "fold_audit": fold_audit,
    }


def fit_full_models(X: np.ndarray, y: np.ndarray, seed: int = 20260710) -> dict:
    models = {}
    scale = StandardScaler().fit(X)
    scaled = scale.transform(X)
    logistic = LogisticRegression(C=1.0, max_iter=3_000, solver="lbfgs").fit(scaled, y)
    models["logistic"] = (scale, logistic)
    models["mp_bulk_lda"] = (scale, MPBulkLDA().fit(scaled, y))
    models["rbf_kernel"] = (scale, SVC(C=1.0, kernel="rbf", gamma="scale").fit(scaled, y))
    mlp = fit_mlp(scaled, y, seed)
    models["mlp"] = (scale, mlp)
    hidden = hidden_features(mlp, scaled)
    hidden_scale = StandardScaler().fit(hidden)
    hidden_kernel = SVC(C=1.0, kernel="rbf", gamma="scale").fit(
        hidden_scale.transform(hidden), y
    )
    models["deep_feature_rbf"] = (scale, mlp, hidden_scale, hidden_kernel)
    hidden_scaled = hidden_scale.transform(hidden)
    hidden_rmt = MPBulkLDA().fit(hidden_scaled, y)
    hidden_mp_kernel = SVC(C=1.0, kernel="rbf", gamma="scale").fit(
        hidden_scaled @ hidden_rmt.whitener_, y
    )
    models["deep_mp_rbf"] = (
        scale, mlp, hidden_scale, hidden_rmt.whitener_, hidden_mp_kernel
    )
    augmented_X, augmented_y = equivariant_augmentation(X, y)
    augmented_scale = StandardScaler().fit(augmented_X)
    augmented_model = LogisticRegression(C=1.0, max_iter=3_000, solver="lbfgs").fit(
        augmented_scale.transform(augmented_X), augmented_y
    )
    models["equivariant_logistic"] = (augmented_scale, augmented_model)
    return models


def predict_full_model(name: str, model: tuple, X: np.ndarray) -> np.ndarray:
    if name == "deep_feature_rbf":
        scale, mlp, hidden_scale, kernel = model
        hidden = hidden_features(mlp, scale.transform(X))
        return kernel.predict(hidden_scale.transform(hidden))
    if name == "deep_mp_rbf":
        scale, mlp, hidden_scale, whitener, kernel = model
        hidden = hidden_features(mlp, scale.transform(X))
        return kernel.predict(hidden_scale.transform(hidden) @ whitener)
    scale, estimator = model
    return estimator.predict(scale.transform(X))


def select_records(records: list[dict], jobs: list[Job]) -> list[dict]:
    wanted = {(job.family, job.label, job.replicate) for job in jobs}
    selected = [record for record in records if record_key(record) in wanted]
    selected.sort(key=record_key)
    if len(selected) != len(wanted):
        raise RuntimeError(f"selected {len(selected)} records, expected {len(wanted)}")
    return selected


def family_arrays(records: list[dict], family: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    selected = [record for record in records if record["family"] == family]
    selected.sort(key=lambda record: (CLASS_INDEX[record["label"]], record["replicate"]))
    X = np.vstack([record["feature"] for record in selected]).astype(np.float64)
    y = encode(np.array([record["label"] for record in selected]))
    return X, y, selected


def family_diagnostics(records: list[dict], family: str) -> dict:
    selected = [record for record in records if record["family"] == family]
    sites = np.array([record["num_sites"] for record in selected], dtype=float)
    loci = np.array([record["num_loci"] for record in selected], dtype=float)
    elapsed = np.array([record["elapsed_seconds"] for record in selected], dtype=float)
    return {
        "n": int(len(selected)),
        "n_per_direction": {
            label: int(sum(record["label"] == label for record in selected))
            for label in DIRECTIONS
        },
        "num_sites_mean": float(sites.mean()),
        "num_sites_sd": float(sites.std()),
        "num_PADZE_loci_mean": float(loci.mean()),
        "num_PADZE_loci_sd": float(loci.std()),
        "simulation_seconds_sum": float(elapsed.sum()),
        "simulation_seconds_median": float(np.median(elapsed)),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_summary(path: Path, result: dict) -> None:
    models = result["baseline_repeated_cv"]["models"]
    lines = [
        "# Four-population, 12-direction extension",
        "",
        f"The primary exchangeable benchmark contains {result['sample_sizes']['baseline']} independent "
        "coalescent replicates across all 12 forward-time ordered edges.",
        "",
        "## Leakage-free baseline",
        "",
        "| model | exact 12-way | unordered pair | orientation given pair | donor | recipient |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, entry in models.items():
        report = entry["majority_vote"]
        lines.append(
            f"| {name} | {report['exact_12way_accuracy']:.3f} | "
            f"{report['unordered_pair_accuracy']:.3f} | "
            f"{report['orientation_accuracy_conditional_on_correct_pair']:.3f} | "
            f"{report['donor_accuracy']:.3f} | {report['recipient_accuracy']:.3f} |"
        )
    lines.extend([
        "",
        "Chance levels are 1/12 for the exact edge, 1/6 for the unordered pair, 1/2 for "
        "orientation conditional on the right pair, and 1/4 for donor or recipient.",
        f"A separate binary reversal audit, given the true unordered pair, reaches "
        f"{result['baseline_reversed_pair_cv']['accuracy']:.3f} "
        f"({result['baseline_reversed_pair_cv']['correct']}/"
        f"{result['baseline_reversed_pair_cv']['n']}).",
        "",
        "## Frozen nuisance transfer",
        "",
        "| family | logistic | equivariant logistic | MP-bulk LDA | RBF | MLP | deep-feature RBF | deep+MP+RBF |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    order = (
        "logistic", "equivariant_logistic", "mp_bulk_lda", "rbf_kernel", "mlp",
        "deep_feature_rbf", "deep_mp_rbf",
    )
    for family, reports in result["frozen_nuisance_transfer"].items():
        values = " | ".join(f"{reports[name]['exact_12way_accuracy']:.3f}" for name in order)
        lines.append(f"| {family} | {values} |")
    lines.extend([
        "",
        "Frozen transfer uses models fit once on the baseline family. Failures are part of the "
        "result; within-family refits are reported in `results.json` only to distinguish loss of "
        "transfer from loss of information.",
        "",
        "All intervals, classwise counts, confusion matrices, epoch audits, seed ledgers, "
        "negative controls, software versions, and artifact hashes are in `results.json`.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("results/twelve_direction_2026_07_10"))
    parser.add_argument("--baseline-reps", type=int, default=20)
    parser.add_argument("--shift-reps", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed-base", type=int, default=270_710_001)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument("--cv-seeds", default="0,1,2")
    parser.add_argument(
        "--families", default=",".join(FAMILY_SPECS),
        help="comma-separated subset of baseline,weak_signal,unequal_ne,balanced_tree,half_sequence",
    )
    parser.add_argument("--simulate-only", action="store_true")
    args = parser.parse_args()
    if args.baseline_reps < 1 or args.shift_reps < 1:
        parser.error("replicate counts must be positive")
    if args.workers < 1 or args.workers > 2:
        parser.error("workers must be 1 or 2 on this laptop")
    families = tuple(value for value in args.families.split(",") if value)
    unknown = sorted(set(families) - set(FAMILY_SPECS))
    if unknown:
        parser.error(f"unknown families: {unknown}")
    if "baseline" not in families:
        parser.error("baseline is required")
    cv_seeds = tuple(int(value) for value in args.cv_seeds.split(",") if value)
    if not cv_seeds:
        parser.error("at least one CV seed is required")

    epoch_validation = validate_all_demographies(families)
    jobs = make_jobs(families, args.baseline_reps, args.shift_reps, args.seed_base)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.out_dir / "features.npz"
    records = simulate_missing(load_records(checkpoint), jobs, args.workers, checkpoint)
    selected = select_records(records, jobs)
    if args.simulate_only:
        print(f"[done] simulation checkpoint -> {checkpoint}")
        return 0

    if args.baseline_reps < args.cv_folds or args.shift_reps < args.cv_folds:
        parser.error("evaluation requires reps per class >= cv-folds")
    baseline_X, baseline_y, _ = family_arrays(selected, "baseline")
    print("[evaluate] repeated replicate-level CV on the exchangeable baseline", flush=True)
    baseline_cv = repeated_cv(baseline_X, baseline_y, cv_seeds, args.cv_folds)
    reversed_pair = reversed_pair_cv(baseline_X, baseline_y, cv_seeds, args.cv_folds)

    print("[evaluate] replicate-label permutation negative control", flush=True)
    rng = np.random.default_rng(20260710)
    permuted_y = rng.permutation(baseline_y)
    label_permutation = repeated_cv(
        baseline_X, permuted_y, cv_seeds, args.cv_folds, model_names=("logistic",)
    )

    print("[evaluate] fitting frozen baseline models and scoring nuisance shifts", flush=True)
    frozen = fit_full_models(baseline_X, baseline_y)
    transfer = {}
    within_family = {}
    for family in families:
        if family == "baseline":
            continue
        X, y, _ = family_arrays(selected, family)
        transfer[family] = {
            name: direction_report(y, predict_full_model(name, model, X))
            for name, model in frozen.items()
        }
        within_family[family] = repeated_cv(
            X, y, cv_seeds, args.cv_folds,
            model_names=("logistic", "equivariant_logistic", "mp_bulk_lda"),
        )

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
        "direction_convention": {
            "reported_labels": "forward-time donor->recipient",
            "msprime_mapping": "backwards-time recipient->donor",
            "n_ordered_edges": len(DIRECTIONS),
            "labels": DIRECTIONS,
        },
        "design": {
            "populations": POPULATIONS,
            "gene_copies_per_population": GENE_COPIES_PER_POP,
            "migration_duration_generations": MIGRATION_DURATION,
            "recombination_rate_per_bp_per_generation": RECOMBINATION_RATE,
            "mutation_rate_per_bp_per_generation": MUTATION_RATE,
            "family_specs": FAMILY_SPECS,
            "actual_msprime_epoch_validation": epoch_validation,
            "seed_base": args.seed_base,
        },
        "feature_representation": {
            "definition": (
                "four alpha + four pi + six pair-private pihat statistics; locus mean/variance/SE "
                "at g=2..40; depth mean and population SD"
            ),
            "dimension": len(FEATURE_COLUMNS),
            "curve_columns": CURVE_COLUMNS,
            "feature_columns": FEATURE_COLUMNS,
            "population_relabelings_checked": len(PERMUTATIONS),
        },
        "sample_sizes": {
            family: int(sum(record["family"] == family for record in selected))
            for family in families
        },
        "family_diagnostics": {
            family: family_diagnostics(selected, family) for family in families
        },
        "chance_levels": {
            "exact_12way": 1 / 12,
            "unordered_pair": 1 / 6,
            "orientation_given_pair": 1 / 2,
            "donor": 1 / 4,
            "recipient": 1 / 4,
        },
        "baseline_repeated_cv": baseline_cv,
        "baseline_reversed_pair_cv": reversed_pair,
        "negative_controls": {
            "population_labels_removed": baseline_cv["models"]["label_invariant_logistic"],
            "replicate_labels_permuted": label_permutation["models"]["logistic"],
        },
        "frozen_nuisance_transfer": transfer,
        "within_family_retrained_cv": within_family,
        "interpretation_guardrails": [
            "All accuracy intervals are conditional on the stated simulation families.",
            "Permutation augmentation creates no new independent replicates and is used only in training folds.",
            "Within-family refitting diagnoses retained information; it does not rescue failed frozen transfer.",
            "Deep-feature kernel performance is reported even when it fails to improve on the linear head.",
        ],
    }
    result_path = args.out_dir / "results.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary_path = args.out_dir / "SUMMARY.md"
    write_summary(summary_path, result)
    result["artifacts"] = {
        checkpoint.name: {"sha256": sha256_file(checkpoint)},
        result_path.name: {"sha256_before_artifact_block": sha256_file(result_path)},
        summary_path.name: {"sha256": sha256_file(summary_path)},
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline": {
            name: entry["majority_vote"]["exact_12way_accuracy"]
            for name, entry in baseline_cv["models"].items()
        },
        "frozen_transfer": {
            family: {name: report["exact_12way_accuracy"] for name, report in reports.items()}
            for family, reports in transfer.items()
        },
        "artifacts": [str(checkpoint), str(result_path), str(summary_path)],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
