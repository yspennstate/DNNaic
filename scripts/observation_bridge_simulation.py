#!/usr/bin/env python3
"""Generate paired observation-process views for a small DNNaic bridge bank.

The canonical simulation varies genealogy and migration rate but exposes every
genealogy through one idealized observation process: 200 complete haploid
copies, all globally polymorphic simulated sites, and roughly 4,500 loci.  The
external audit instead found 22--16,301 retained loci, population-correlated
missingness, MAF/call-rate/RAD ascertainment, and frequent filters requiring
polymorphism within populations.  Raw PADZE SE also equals sigma/sqrt(L), so a
classifier can learn assay size rather than biology.

This CPU-only retraining pilot creates a small independent bank of A/B/C
genealogy sets and, for every parent, emits the same g=2..16 PADZE curve under
ten paired observation views.  They cross a complete 200-copy control with
32/64-copy sampling, global MAF 1%/5%, RAD-like one-SNP-per-bin, 64/256-locus
caps, population-correlated missingness, and a within-each-population filter.
All views of one genealogy share ``parent_genealogy_id`` and can never cross a
CV fold.  Exact rates are also shared across A/B/C and receive a separate
``rate_family_id``.

Each parent contains 20 independent 50 kb contigs (1 Mb total), avoiding the
fiction that thousands of linked SNPs are independent loci.  The default 24
rate families produce 72 parents and at most 720 feature curves.  A compressed
checkpoint is updated atomically after every parent.
The runner checks the owner's compute governor before initial work and before
every simulation; distress aborts before the next job.  It is single-threaded,
BelowNormal on Windows, and stores no tree sequences.

The optional analysis is exploratory.  It evaluates the prespecified raw,
SE-free, and S3-orbit representations with parent-genealogy and rate-family
outer splits, plus leave-one-observation-view-out transfer.  Natural rows are
unlabeled OOD diagnostics only, never an accuracy denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import structured_transfer_pilot as structured
from scripts.simulate_demography import BACKWARD_MIGRATION, build_demography


SCHEMA_VERSION = "dnnaic-observation-bridge-v3"
CLASSES = np.array(["A", "B", "C"])
GENE_COPIES = 200
DEPTHS = np.arange(2, 17, dtype=np.int64)
CONTIG_COUNT = 20
CONTIG_LENGTH = 50_000
SEQUENCE_LENGTH = CONTIG_COUNT * CONTIG_LENGTH
RECOMBINATION_RATE = 1.78e-8
MUTATION_RATE = 2e-8
RATE_MIN = 5e-5
RATE_MAX = 5e-4
DEFAULT_RATE_FAMILIES = 24
DEFAULT_SEED_BASE = 607_110_001
MIN_CALLED_COPIES = 16
MIN_VIEW_LOCI = 16
MOMENTS = ("mean", "variance", "se")
BLOCKS = (
    "alpha_1", "alpha_2", "alpha_3",
    "pi_1", "pi_2", "pi_3",
    "pihat_12", "pihat_13", "pihat_23",
)
CURVE_COLUMNS = ["g"] + [f"{block}_{moment}" for block in BLOCKS for moment in MOMENTS]
DEFAULT_CACHE = (
    Path.home()
    / "Documents"
    / "Codex"
    / "2026-07-10"
    / "dnnaic-datasets2-data"
    / "observation_bridge_2026_07_11"
)
DEFAULT_RESULTS = REPO / "results" / "observation_bridge_simulation_2026_07_11"

# These views are prespecified and deliberately separate biological genealogy
# from the observation/ascertainment process.
VIEW_SPECS = {
    "complete_all_observed_alleles": {
        "family": "complete_control",
        "factor_tags": [],
        "sample_copies": 200,
        "global_maf": 0.0,
        "within_each_population": False,
        "rad_bin_bp": None,
        "cap": None,
        "missingness": False,
    },
    "sample_64_union": {
        "family": "sample_size",
        "factor_tags": ["sample_downsampling"],
        "sample_copies": 64,
        "global_maf": 0.0,
        "within_each_population": False,
        "rad_bin_bp": None,
        "cap": None,
        "missingness": False,
    },
    "sample_32_union": {
        "family": "sample_size",
        "factor_tags": ["sample_downsampling"],
        "sample_copies": 32,
        "global_maf": 0.0,
        "within_each_population": False,
        "rad_bin_bp": None,
        "cap": None,
        "missingness": False,
    },
    "sample_64_global_maf_01": {
        "family": "maf",
        "factor_tags": ["sample_downsampling", "maf_filter"],
        "sample_copies": 64,
        "global_maf": 0.01,
        "within_each_population": False,
        "rad_bin_bp": None,
        "cap": None,
        "missingness": False,
    },
    "sample_64_global_maf_05": {
        "family": "maf",
        "factor_tags": ["sample_downsampling", "maf_filter"],
        "sample_copies": 64,
        "global_maf": 0.05,
        "within_each_population": False,
        "rad_bin_bp": None,
        "cap": None,
        "missingness": False,
    },
    "sample_64_rad_like_maf_01": {
        "family": "rad",
        "factor_tags": ["sample_downsampling", "maf_filter", "rad_thinning"],
        "sample_copies": 64,
        "global_maf": 0.01,
        "within_each_population": False,
        "rad_bin_bp": 10_000,
        "cap": None,
        "missingness": False,
    },
    "sample_64_cap_64_maf_01": {
        "family": "locus_count",
        "factor_tags": ["sample_downsampling", "maf_filter", "locus_cap"],
        "sample_copies": 64,
        "global_maf": 0.01,
        "within_each_population": False,
        "rad_bin_bp": None,
        "cap": 64,
        "missingness": False,
    },
    "sample_64_cap_256_maf_01": {
        "family": "locus_count",
        "factor_tags": ["sample_downsampling", "maf_filter", "locus_cap"],
        "sample_copies": 64,
        "global_maf": 0.01,
        "within_each_population": False,
        "rad_bin_bp": None,
        "cap": 256,
        "missingness": False,
    },
    "sample_64_missingness_maf_01": {
        "family": "missingness",
        "factor_tags": ["sample_downsampling", "maf_filter", "missingness"],
        "sample_copies": 64,
        "global_maf": 0.01,
        "within_each_population": False,
        "rad_bin_bp": None,
        "cap": None,
        "missingness": True,
        "call_rate_range": [0.80, 0.98],
    },
    "sample_64_within_each_population": {
        "family": "within_population_ascertainment",
        "factor_tags": ["sample_downsampling", "within_population_filter"],
        "sample_copies": 64,
        "global_maf": 0.0,
        "within_each_population": True,
        "rad_bin_bp": None,
        "cap": None,
        "missingness": False,
    },
}


@dataclass(frozen=True)
class BridgeJob:
    rate_index: int
    rate: float
    label: str
    parent_genealogy_id: str
    rate_family_id: str
    ancestry_seed: int
    mutation_seed: int
    nuisance_profile_seed: int
    observation_seed: int


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def derived_seed(base_seed: int, *parts: str) -> int:
    """Derive a stable nonzero uint32 seed without order-dependent offsets."""
    payload = "|".join((str(int(base_seed)), *map(str, parts))).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return int(value % (2**32 - 1) + 1)


def configuration(
    rate_families: int,
    seed_base: int,
    *,
    jobs: Sequence[BridgeJob] | None = None,
    rates: Sequence[float] | None = None,
) -> dict:
    if jobs is None and rates is None:
        jobs, generated_rates = make_jobs(rate_families, seed_base)
        rates = generated_rates
    elif jobs is None or rates is None:
        raise ValueError("jobs and rates must be supplied together")
    jobs = list(jobs)
    rates = np.asarray(rates, dtype=float)
    if len(jobs) != rate_families * len(CLASSES) or len(rates) != rate_families:
        raise ValueError("job/rate manifest does not match requested rate-family count")
    source_paths = {
        "observation_bridge": Path(__file__).resolve(),
        "structured_transfer": Path(structured.__file__).resolve(),
        "demography_generator": REPO / "scripts" / "simulate_demography.py",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "rate_families": int(rate_families),
        "seed_base": int(seed_base),
        "classes": CLASSES.tolist(),
        "gene_copies_per_population": GENE_COPIES,
        "depths": DEPTHS.tolist(),
        "sequence_design": {
            "independent_contigs": CONTIG_COUNT,
            "contig_length": CONTIG_LENGTH,
            "total_sequence_length": SEQUENCE_LENGTH,
        },
        "recombination_rate": RECOMBINATION_RATE,
        "mutation_rate": MUTATION_RATE,
        "rate_prior": {
            "kind": "log_uniform",
            "minimum": RATE_MIN,
            "maximum": RATE_MAX,
            "sharing": "one exact draw shared across A/B/C",
        },
        "exact_rate_draws": [
            {"rate_index": int(index), "rate_hex": float(rate).hex()}
            for index, rate in enumerate(rates)
        ],
        "job_manifest": [
            {
                **asdict(job),
                "rate_hex": float(job.rate).hex(),
            }
            for job in jobs
        ],
        "ancestry_model": "msprime.StandardCoalescent",
        "mutation_model": "msprime.JC69",
        "view_specs": VIEW_SPECS,
        "ordered_view_seed_contract": [
            {
                "view_index": int(index),
                "view": name,
                "nuisance_profile_seed": (
                    "derived_seed(job.nuisance_profile_seed, 'profile', view_name)"
                ),
                "downstream_filter_seed": (
                    "derived_seed(job.observation_seed, 'sampling', view_name); never selects "
                    "biological sample identities"
                ),
            }
            for index, name in enumerate(VIEW_SPECS)
        ],
        "fixed_sample_contract": {
            "seed": "job.nuisance_profile_seed (shared across A/B/C within a rate family)",
            "population_sampling": "one stable permutation per population, reused across loci and views",
            "nesting": "nested: the 32-copy sample is a strict subset of the 64-copy sample",
            "downstream_randomness": (
                "view-specific observation_seed streams apply only missingness, RAD thinning, "
                "and locus caps after the fixed biological sample is selected"
            ),
        },
        "contig_seed_contract": {
            "ancestry": "derived_seed(job.ancestry_seed, 'ancestry', contig_index)",
            "mutation": "derived_seed(job.mutation_seed, 'mutation', contig_index)",
        },
        "feature_contract": {
            "curve_columns": CURVE_COLUMNS,
            "moments": list(MOMENTS),
            "shape": [int(len(DEPTHS)), int(len(CURVE_COLUMNS))],
        },
        "dependency_versions": {
            name: importlib_metadata.version(name)
            for name in ("msprime", "tskit", "numpy", "padze", "scikit-learn")
        },
        "semantic_source_sha256": {
            name: structured.sha256_file(path)
            for name, path in source_paths.items()
        },
        "minimum_called_copies_per_population_locus": MIN_CALLED_COPIES,
        "minimum_view_loci": MIN_VIEW_LOCI,
        "direction_semantics": {
            label: {
                "forward": {
                    "A": "P1->P2",
                    "B": "P2->P3",
                    "C": "P3->P2",
                }[label],
                "backward_msprime": f"{BACKWARD_MIGRATION[label][0]}->{BACKWARD_MIGRATION[label][1]}",
            }
            for label in CLASSES
        },
    }


def configuration_sha256(config: dict) -> str:
    return hashlib.sha256(_canonical_json(config)).hexdigest()


def make_jobs(rate_families: int, seed_base: int) -> tuple[list[BridgeJob], np.ndarray]:
    if rate_families < 3:
        raise ValueError("at least three rate families are required")
    rng = np.random.default_rng(seed_base)
    rates = np.exp(
        rng.uniform(np.log(RATE_MIN), np.log(RATE_MAX), size=rate_families)
    ).astype(float)
    jobs = []
    for rate_index, rate in enumerate(rates):
        for class_index, label in enumerate(CLASSES):
            ordinal = rate_index * len(CLASSES) + class_index
            jobs.append(BridgeJob(
                rate_index=int(rate_index),
                rate=float(rate),
                label=str(label),
                parent_genealogy_id=(
                    f"bridge|rate{rate_index:03d}|{float(rate).hex()}|{label}"
                ),
                rate_family_id=f"bridge-rate|{rate_index:03d}|{float(rate).hex()}",
                ancestry_seed=int(seed_base + 100_000 + ordinal),
                mutation_seed=int(seed_base + 200_000 + ordinal),
                nuisance_profile_seed=int(seed_base + 300_000 + rate_index),
                observation_seed=int(seed_base + 400_000 + ordinal),
            ))
    independent_seeds = np.array([
        seed
        for job in jobs
        for seed in (job.ancestry_seed, job.mutation_seed, job.observation_seed)
    ])
    profile_seeds = np.array([job.nuisance_profile_seed for job in jobs])
    if len(np.unique(independent_seeds)) != len(independent_seeds):
        raise AssertionError("bridge seed streams overlap")
    if len(np.unique(profile_seeds)) != rate_families:
        raise AssertionError("nuisance profiles must be shared once per rate family")
    for rate_index in range(rate_families):
        current = profile_seeds[[job.rate_index == rate_index for job in jobs]]
        if len(current) != len(CLASSES) or len(np.unique(current)) != 1:
            raise AssertionError("A/B/C do not share one nuisance profile seed")
    realized_independent = [
        seed
        for job in jobs
        for seed in (
            *(
                derived_seed(job.ancestry_seed, "ancestry", str(index))
                for index in range(CONTIG_COUNT)
            ),
            *(
                derived_seed(job.mutation_seed, "mutation", str(index))
                for index in range(CONTIG_COUNT)
            ),
            *(
                derived_seed(job.observation_seed, "sampling", view)
                for view in VIEW_SPECS
            ),
        )
    ]
    if len(set(realized_independent)) != len(realized_independent):
        raise AssertionError("derived independent bridge seed streams collide")
    return jobs, rates


def _population_id_by_name(tree_sequence) -> dict[str, int]:
    result = {}
    for population_id in range(tree_sequence.num_populations):
        metadata = tree_sequence.population(population_id).metadata
        if isinstance(metadata, dict) and "name" in metadata:
            result[str(metadata["name"])] = int(population_id)
    missing = [name for name in ("P1", "P2", "P3") if name not in result]
    if missing:
        raise ValueError("tree sequence lacks named populations: " + ",".join(missing))
    return result


def extract_genotype_panel(
    tree_sequence,
    *,
    contig_index: int,
) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    """Return per-copy four-state genotypes for every observed polymorphic site."""
    population_ids = _population_id_by_name(tree_sequence)
    samples = tree_sequence.samples()
    sample_population = tree_sequence.tables.nodes.population[samples]
    masks = [sample_population == population_ids[name] for name in ("P1", "P2", "P3")]
    sample_sizes = np.array([int(mask.sum()) for mask in masks], dtype=int)
    if not np.array_equal(sample_sizes, np.full(3, GENE_COPIES)):
        raise AssertionError(f"bridge sample counts {sample_sizes.tolist()} != {[GENE_COPIES] * 3}")
    genotype_rows = []
    positions = []
    locus_ids = []
    allele_cardinalities = []
    for site_index, variant in enumerate(tree_sequence.variants()):
        genotype = np.asarray(variant.genotypes, dtype=int)
        if np.any(genotype < 0):
            raise AssertionError("msprime bridge unexpectedly produced missing genotypes")
        observed = np.unique(genotype[genotype >= 0])
        if len(observed) < 2:
            continue
        if len(variant.alleles) > 4 or int(observed.max()) >= 4:
            raise AssertionError("JC69 observation produced more than four allele states")
        population_genotypes = np.stack([
            genotype[mask] for mask in masks
        ]).astype(np.int8, copy=False)
        if population_genotypes.shape != (3, GENE_COPIES):
            raise AssertionError("bridge per-population genotype shape changed")
        genotype_rows.append(population_genotypes)
        local_position = float(variant.site.position)
        global_position = contig_index * CONTIG_LENGTH + local_position
        positions.append(global_position)
        locus_ids.append(f"contig{contig_index:02d}|site{site_index}@{local_position:.1f}")
        allele_cardinalities.append(int(len(observed)))
    genotypes = (
        np.stack(genotype_rows)
        if genotype_rows
        else np.zeros((0, 3, GENE_COPIES), dtype=np.int8)
    )
    audit = {
        "contig_index": int(contig_index),
        "sites": int(tree_sequence.num_sites),
        "polymorphic_sites": int(len(genotypes)),
        "biallelic_polymorphic_sites": int(sum(value == 2 for value in allele_cardinalities)),
        "multiallelic_polymorphic_sites": int(sum(value > 2 for value in allele_cardinalities)),
        "monomorphic_sites_excluded": int(tree_sequence.num_sites - len(genotypes)),
        "gene_copies": {name: int(value) for name, value in zip(("P1", "P2", "P3"), sample_sizes)},
    }
    return genotypes, np.asarray(positions, dtype=float), locus_ids, audit


def _multivariate_sample(
    colors: np.ndarray,
    n: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample allele counts without replacement using sequential hypergeometrics."""
    colors = np.asarray(colors, dtype=np.int64)
    total = int(colors.sum())
    if n < 0 or n > total:
        raise ValueError(f"cannot sample {n} copies from {total}")
    if n == total:
        return colors.copy()
    output = np.zeros_like(colors)
    remaining_total = total
    remaining_sample = int(n)
    for index in range(len(colors) - 1):
        good = int(colors[index])
        draw = int(rng.hypergeometric(good, remaining_total - good, remaining_sample))
        output[index] = draw
        remaining_total -= good
        remaining_sample -= draw
    output[-1] = remaining_sample
    return output


def sample_index_subsets(profile_seed: int) -> dict[int, np.ndarray]:
    """Create class-shared, cross-locus sample subsets with 32 nested in 64."""
    permutations = []
    for population in range(3):
        rng = np.random.default_rng(
            derived_seed(profile_seed, "base_sample_permutation", str(population))
        )
        permutations.append(rng.permutation(GENE_COPIES))
    result = {
        GENE_COPIES: np.tile(np.arange(GENE_COPIES, dtype=np.int64), (3, 1)),
        64: np.stack([np.sort(permutation[:64]) for permutation in permutations]),
        32: np.stack([np.sort(permutation[:32]) for permutation in permutations]),
    }
    for population in range(3):
        if not set(result[32][population]).issubset(set(result[64][population])):
            raise AssertionError("32-copy bridge sample is not nested inside 64 copies")
    return result


def sample_index_sha256(indices: np.ndarray) -> str:
    indices = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(
        _canonical_json({"shape": list(indices.shape)}) + indices.tobytes()
    ).hexdigest()


def genotype_counts(genotypes: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
    """Count aligned JC69 allele states for one fixed cross-locus sample."""
    genotypes = np.asarray(genotypes)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    if genotypes.ndim != 3 or genotypes.shape[1:] != (3, GENE_COPIES):
        raise ValueError("genotypes must have shape (locus, 3, GENE_COPIES)")
    if sample_indices.ndim != 2 or sample_indices.shape[0] != 3:
        raise ValueError("sample indices must have shape (3, copies)")
    if np.any((sample_indices < 0) | (sample_indices >= GENE_COPIES)):
        raise ValueError("sample index is outside the simulated population")
    if np.any((genotypes < 0) | (genotypes > 3)):
        raise ValueError("JC69 genotypes must use allele states 0..3")
    if any(len(np.unique(row)) != len(row) for row in sample_indices):
        raise ValueError("fixed biological samples cannot contain duplicate copy indices")
    counts = np.zeros((len(genotypes), 3, 4), dtype=np.int64)
    for population in range(3):
        selected = genotypes[:, population, sample_indices[population]]
        for allele in range(4):
            counts[:, population, allele] = np.sum(selected == allele, axis=1)
    expected = sample_indices.shape[1]
    if not np.all(counts.sum(axis=2) == expected):
        raise AssertionError("fixed-sample genotype counts do not conserve copies")
    return counts


def downsample_missingness(
    counts: np.ndarray,
    call_rates: Sequence[float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Hypergeometrically downsample called copies without changing genotypes."""
    counts = np.asarray(counts, dtype=np.int64)
    call_rates = np.asarray(call_rates, dtype=float)
    if counts.ndim != 3 or counts.shape[1] != 3 or counts.shape[2] < 2:
        raise ValueError("counts must have shape (locus, 3, at least 2 alleles)")
    if call_rates.shape != (3,) or np.any((call_rates <= 0) | (call_rates > 1)):
        raise ValueError("call rates must contain three values in (0,1]")
    output = np.zeros_like(counts)
    called = np.zeros((len(counts), 3), dtype=np.int64)
    for locus in range(len(counts)):
        for population in range(3):
            colors = counts[locus, population]
            total = int(colors.sum())
            n_called = int(rng.binomial(total, call_rates[population]))
            output[locus, population] = _multivariate_sample(colors, n_called, rng)
            called[locus, population] = n_called
    return output, called


def _ordered_locus_sha256(ids: Sequence[str]) -> str:
    return structured.sha256_text(map(str, ids))


def apply_view(
    counts: np.ndarray,
    positions: np.ndarray,
    locus_ids: Sequence[str],
    name: str,
    *,
    base_sample_sha256: str,
    profile_seed: int,
    sampling_seed: int,
) -> tuple[np.ndarray, list[str], dict]:
    """Apply one pinned observation/ascertainment view."""
    try:
        spec = VIEW_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown observation view {name!r}") from exc
    counts = np.asarray(counts, dtype=np.int64)
    expected_copies = int(spec["sample_copies"])
    if counts.ndim != 3 or counts.shape[1:] != (3, 4):
        raise ValueError("counts must have shape (locus, 3, 4)")
    if not np.all(counts.sum(axis=2) == expected_copies):
        raise ValueError(
            f"view {name} requires a fixed {expected_copies}-copy sample per population"
        )
    if not _is_sha256(base_sample_sha256):
        raise ValueError("base sample SHA-256 is required")
    profile_rng = np.random.default_rng(profile_seed)
    sampling_rng = np.random.default_rng(sampling_seed)
    current = counts.copy()
    called = current.sum(axis=2)
    call_rates = np.ones(3, dtype=float)
    if spec["missingness"]:
        lo, hi = spec["call_rate_range"]
        call_rates = profile_rng.uniform(lo, hi, size=3)
        current, called = downsample_missingness(current, call_rates, sampling_rng)
    keep = np.all(called >= MIN_CALLED_COPIES, axis=1)
    global_counts = current.sum(axis=1)
    total = global_counts.sum(axis=1)
    second_largest = np.sort(global_counts, axis=1)[:, -2]
    with np.errstate(divide="ignore", invalid="ignore"):
        minor = np.divide(
            second_largest,
            total,
            out=np.zeros_like(second_largest, dtype=float),
            where=total > 0,
        )
    # Every retained locus must remain polymorphic after missingness.  In
    # particular, a zero MAF threshold must not silently admit loci made
    # monomorphic by genotype downsampling.
    keep &= np.count_nonzero(global_counts, axis=1) >= 2
    keep &= minor >= float(spec["global_maf"])
    if spec["within_each_population"]:
        keep &= np.all(np.count_nonzero(current, axis=2) >= 2, axis=1)
    indices = np.flatnonzero(keep)
    if spec["rad_bin_bp"] is not None and len(indices):
        bins = np.floor(np.asarray(positions)[indices] / int(spec["rad_bin_bp"])).astype(np.int64)
        selected = []
        for value in np.unique(bins):
            candidates = indices[bins == value]
            selected.append(int(sampling_rng.choice(candidates)))
        indices = np.array(sorted(selected), dtype=int)
    eligible = int(len(indices))
    cap = spec["cap"]
    if cap is not None and len(indices) > int(cap):
        indices = np.sort(sampling_rng.choice(indices, size=int(cap), replace=False))
    selected_ids = [str(locus_ids[index]) for index in indices]
    selected_counts = current[indices].astype("<i8", copy=False)
    audit = {
        "view": name,
        "specification": spec,
        "input_loci": int(len(counts)),
        "eligible_before_cap": eligible,
        "retained_after_cap": int(len(indices)),
        "profile_seed": int(profile_seed),
        "sampling_seed": int(sampling_seed),
        "base_sample_sha256": base_sample_sha256,
        "fixed_sample_copies_per_population": expected_copies,
        "input_count_matrix_sha256": hashlib.sha256(
            counts.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "call_rates": {
            name: float(value)
            for name, value in zip(("P1", "P2", "P3"), call_rates)
        },
        "called_copy_counts": {
            name: {
                "minimum": int(called[indices, population].min()) if len(indices) else None,
                "mean": float(called[indices, population].mean()) if len(indices) else None,
                "maximum": int(called[indices, population].max()) if len(indices) else None,
            }
            for population, name in enumerate(("P1", "P2", "P3"))
        },
        "ordered_locus_sha256": _ordered_locus_sha256(selected_ids),
        "selected_count_matrix_sha256": hashlib.sha256(selected_counts.tobytes()).hexdigest(),
        "usable": bool(len(indices) >= MIN_VIEW_LOCI),
        "minimum_view_loci": MIN_VIEW_LOCI,
    }
    return current[indices], selected_ids, audit


def counts_to_curve(
    counts: np.ndarray,
    locus_ids: Sequence[str],
    *,
    expected_copies: int,
) -> tuple[np.ndarray, list[str]]:
    from padze import LociData, Metadata, compute_features

    counts = np.asarray(counts, dtype=np.int64)
    if counts.ndim != 3 or counts.shape[1] != 3 or counts.shape[2] < 2:
        raise ValueError("counts must have shape (locus, 3, at least 2 alleles)")
    sample_sizes = counts.sum(axis=2)
    if len(counts) < MIN_VIEW_LOCI or np.any(sample_sizes < MIN_CALLED_COPIES):
        raise ValueError("view does not meet locus/called-copy minimum")
    loci = LociData(
        populations=["P1", "P2", "P3"],
        count_matrices=[matrix[:, matrix.sum(axis=0) > 0] for matrix in counts],
        sample_sizes=sample_sizes,
        locus_ids=list(locus_ids),
        metadata=Metadata(
            source="DNNaic observation bridge",
            populations=["P1", "P2", "P3"],
            sample_ids={name: [] for name in ("P1", "P2", "P3")},
            ploidy={name: 1 for name in ("P1", "P2", "P3")},
            n_loci_read=int(len(counts)),
            n_loci_kept=int(len(counts)),
            filters_applied=["pinned observation bridge view"],
            missing_fraction=float(
                1.0
                - sample_sizes.sum()
                / (len(counts) * 3 * expected_copies)
            ),
        ),
    )
    table = compute_features(
        loci,
        depths=DEPTHS,
        pihat_sizes=(2,),
        moments=MOMENTS,
        bias_corrected=True,
    )
    matrix, columns = table.to_frame()
    index = {column: position for position, column in enumerate(columns)}
    try:
        ordered = matrix[:, [index[column] for column in CURVE_COLUMNS]].astype(float)
    except KeyError as exc:
        raise RuntimeError(f"PADZE feature contract changed: {exc}") from exc
    if ordered.shape != (len(DEPTHS), 28) or not np.isfinite(ordered).all():
        raise AssertionError("observation bridge produced an invalid PADZE curve")
    if not np.array_equal(ordered[:, 0], DEPTHS):
        raise AssertionError("observation bridge depth grid changed")
    return ordered, CURVE_COLUMNS


def simulate_job(
    job: BridgeJob,
    *,
    compute_state: Path | None = None,
) -> list[dict]:
    import msprime

    source, destination = BACKWARD_MIGRATION[job.label]
    genotype_parts = []
    position_parts = []
    locus_ids = []
    contig_audits = []
    realized_seeds = []
    for contig_index in range(CONTIG_COUNT):
        if compute_state is not None:
            structured.compute_gate(compute_state)
        ancestry_seed = derived_seed(job.ancestry_seed, "ancestry", str(contig_index))
        mutation_seed = derived_seed(job.mutation_seed, "mutation", str(contig_index))
        realized_seeds.extend((ancestry_seed, mutation_seed))
        tree_sequence = msprime.sim_ancestry(
            samples={name: GENE_COPIES for name in ("P1", "P2", "P3")},
            sequence_length=CONTIG_LENGTH,
            discrete_genome=True,
            recombination_rate=RECOMBINATION_RATE,
            ploidy=1,
            model=msprime.StandardCoalescent(),
            demography=build_demography(job.rate, source, destination),
            random_seed=ancestry_seed,
        )
        tree_sequence = msprime.sim_mutations(
            tree_sequence,
            rate=MUTATION_RATE,
            model=msprime.JC69(),
            discrete_genome=True,
            keep=True,
            random_seed=mutation_seed,
        )
        current_genotypes, current_positions, current_ids, current_audit = extract_genotype_panel(
            tree_sequence,
            contig_index=contig_index,
        )
        current_audit["ancestry_seed"] = ancestry_seed
        current_audit["mutation_seed"] = mutation_seed
        genotype_parts.append(current_genotypes)
        position_parts.append(current_positions)
        locus_ids.extend(current_ids)
        contig_audits.append(current_audit)
    if len(set(realized_seeds)) != len(realized_seeds):
        raise AssertionError("derived ancestry/mutation seeds collide within a bridge parent")
    genotypes = np.concatenate(genotype_parts, axis=0)
    positions = np.concatenate(position_parts)
    sample_subsets = sample_index_subsets(job.nuisance_profile_seed)
    counts_by_size = {
        int(size): genotype_counts(genotypes, indices)
        for size, indices in sample_subsets.items()
    }
    sample_hashes = {
        int(size): sample_index_sha256(indices)
        for size, indices in sample_subsets.items()
    }
    source_audit = {
        "independent_contigs": CONTIG_COUNT,
        "contig_length": CONTIG_LENGTH,
        "total_sequence_length": SEQUENCE_LENGTH,
        "total_sites": int(sum(current["sites"] for current in contig_audits)),
        "polymorphic_sites": int(len(genotypes)),
        "multiallelic_polymorphic_sites": int(
            sum(current["multiallelic_polymorphic_sites"] for current in contig_audits)
        ),
        "genotype_panel_sha256": hashlib.sha256(
            genotypes.astype(np.int8, copy=False).tobytes()
        ).hexdigest(),
        "full_count_matrix_sha256": hashlib.sha256(
            counts_by_size[GENE_COPIES].astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "fixed_sample_index_sha256_by_copies": {
            str(size): digest for size, digest in sorted(sample_hashes.items())
        },
        "contigs": contig_audits,
    }
    records = []
    for view_index, view in enumerate(VIEW_SPECS):
        if compute_state is not None:
            structured.compute_gate(compute_state)
        profile_seed = derived_seed(job.nuisance_profile_seed, "profile", view)
        sampling_seed = derived_seed(job.observation_seed, "sampling", view)
        sample_copies = int(VIEW_SPECS[view]["sample_copies"])
        observed, selected_ids, view_audit = apply_view(
            counts_by_size[sample_copies],
            positions,
            locus_ids,
            view,
            base_sample_sha256=sample_hashes[sample_copies],
            profile_seed=profile_seed,
            sampling_seed=sampling_seed,
        )
        record = {
            **asdict(job),
            "view": view,
            "view_index": int(view_index),
            "source_audit": source_audit,
            "view_audit": view_audit,
            "feature": None,
            "invalid_reason": None,
        }
        if view_audit["usable"]:
            feature, columns = counts_to_curve(
                observed,
                selected_ids,
                expected_copies=int(VIEW_SPECS[view]["sample_copies"]),
            )
            if columns != CURVE_COLUMNS:
                raise AssertionError("bridge PADZE columns changed")
            record["feature"] = feature.astype(np.float32)
        else:
            record["invalid_reason"] = (
                f"{view_audit['retained_after_cap']} loci < {MIN_VIEW_LOCI}"
            )
        records.append(record)
    return records


def record_key(record: dict) -> tuple[str, str]:
    return str(record["parent_genealogy_id"]), str(record["view"])


@contextmanager
def single_writer_lock(directory: Path):
    """Hold an OS advisory lock; crash exit releases it without stale recovery."""
    with structured.SingleWriterLease(
        directory,
        ".observation_bridge.lock",
    ) as lease:
        yield lease.path


def save_checkpoint(path: Path, records: Sequence[dict], config_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [record_key(record) for record in records]
    if len(set(keys)) != len(keys):
        raise RuntimeError("refusing to save duplicate observation-bridge records")
    for parent in sorted({key[0] for key in keys}):
        present = {view for current_parent, view in keys if current_parent == parent}
        if present != set(VIEW_SPECS):
            raise RuntimeError(f"refusing to save incomplete parent {parent}")
    metadata = []
    feature_rows = []
    for record in sorted(records, key=record_key):
        current = {key: value for key, value in record.items() if key != "feature"}
        feature = record.get("feature")
        if feature is None:
            current["feature_index"] = None
        else:
            feature = np.asarray(feature, dtype=np.float32)
            if feature.shape != (len(DEPTHS), len(CURVE_COLUMNS)) or not np.isfinite(feature).all():
                raise RuntimeError(f"invalid checkpoint feature for {record_key(record)}")
            current["feature_index"] = len(feature_rows)
            feature_rows.append(feature)
        metadata.append(current)
    features = (
        np.stack(feature_rows)
        if feature_rows
        else np.zeros((0, len(DEPTHS), 28), dtype=np.float32)
    )
    temporary = path.with_suffix(path.suffix + ".part")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.array(SCHEMA_VERSION),
            configuration_sha256=np.array(config_sha256),
            metadata_json=np.array(json.dumps(metadata, sort_keys=True, allow_nan=False)),
            features=features,
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_checkpoint(
    path: Path,
    expected_config_sha256: str,
    expected_jobs: Sequence[BridgeJob] | None = None,
) -> list[dict]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as archive:
        schema = str(archive["schema_version"].item())
        config_sha = str(archive["configuration_sha256"].item())
        if schema != SCHEMA_VERSION or config_sha != expected_config_sha256:
            raise RuntimeError("observation bridge checkpoint configuration changed")
        metadata = json.loads(str(archive["metadata_json"].item()))
        features = np.asarray(archive["features"], dtype=np.float32)
    if features.ndim != 3 or features.shape[1:] != (len(DEPTHS), len(CURVE_COLUMNS)):
        raise RuntimeError(f"observation bridge checkpoint feature shape changed: {features.shape}")
    if not np.isfinite(features).all():
        raise RuntimeError("observation bridge checkpoint contains non-finite features")
    feature_indices = [
        current.get("feature_index") for current in metadata
        if current.get("feature_index") is not None
    ]
    if any(isinstance(index, bool) or not isinstance(index, int) for index in feature_indices):
        raise RuntimeError("observation bridge checkpoint has a non-integer feature index")
    if sorted(feature_indices) != list(range(len(features))):
        raise RuntimeError("observation bridge checkpoint feature indices are not a bijection")
    records = []
    for current in metadata:
        index = current.pop("feature_index")
        current["feature"] = None if index is None else features[int(index)]
        if current["feature"] is not None and not np.array_equal(current["feature"][:, 0], DEPTHS):
            raise RuntimeError("observation bridge checkpoint depth grid changed")
        usable = bool(current.get("view_audit", {}).get("usable", False))
        if usable != (current["feature"] is not None):
            raise RuntimeError("checkpoint feature presence disagrees with view usability")
        records.append(current)
    if len({record_key(record) for record in records}) != len(records):
        raise RuntimeError("observation bridge checkpoint contains duplicate records")
    if expected_jobs is not None:
        jobs = {job.parent_genealogy_id: job for job in expected_jobs}
        view_indices = {name: index for index, name in enumerate(VIEW_SPECS)}
        for record in records:
            parent = str(record.get("parent_genealogy_id"))
            if parent not in jobs:
                raise RuntimeError(f"checkpoint contains unexpected parent {parent}")
            expected = asdict(jobs[parent])
            for field, value in expected.items():
                observed = record.get(field)
                equal = (
                    float(observed).hex() == float(value).hex()
                    if field == "rate"
                    else observed == value
                )
                if not equal:
                    raise RuntimeError(f"checkpoint {parent} metadata field {field} changed")
            view = str(record.get("view"))
            if view not in view_indices or record.get("view_index") != view_indices[view]:
                raise RuntimeError(f"checkpoint {parent} has an invalid view contract")
            view_audit = record.get("view_audit", {})
            expected_profile_seed = derived_seed(
                jobs[parent].nuisance_profile_seed,
                "profile",
                view,
            )
            expected_sampling_seed = derived_seed(
                jobs[parent].observation_seed,
                "sampling",
                view,
            )
            expected_sample_hash = sample_index_sha256(
                sample_index_subsets(jobs[parent].nuisance_profile_seed)[
                    int(VIEW_SPECS[view]["sample_copies"])
                ]
            )
            if (
                view_audit.get("specification") != VIEW_SPECS[view]
                or view_audit.get("profile_seed") != expected_profile_seed
                or view_audit.get("sampling_seed") != expected_sampling_seed
                or view_audit.get("base_sample_sha256") != expected_sample_hash
                or view_audit.get("fixed_sample_copies_per_population")
                != int(VIEW_SPECS[view]["sample_copies"])
                or not _is_sha256(view_audit.get("input_count_matrix_sha256"))
            ):
                raise RuntimeError(f"checkpoint {parent} view metadata changed for {view}")
        for parent in sorted({str(record["parent_genealogy_id"]) for record in records}):
            current = [record for record in records if str(record["parent_genealogy_id"]) == parent]
            for sample_copies in sorted({
                int(VIEW_SPECS[str(record["view"])]["sample_copies"])
                for record in current
            }):
                hashes = {
                    str(record["view_audit"]["input_count_matrix_sha256"])
                    for record in current
                    if int(VIEW_SPECS[str(record["view"])]["sample_copies"]) == sample_copies
                }
                if len(hashes) != 1:
                    raise RuntimeError(
                        f"checkpoint {parent} does not reuse one {sample_copies}-copy sample"
                    )
        complete = []
        for parent in sorted({record["parent_genealogy_id"] for record in records}):
            current = [record for record in records if record["parent_genealogy_id"] == parent]
            if {record["view"] for record in current} == set(VIEW_SPECS):
                complete.extend(current)
        records = complete
    return records


def checkpoint_audit(path: Path, records: Sequence[dict], config_sha256: str) -> dict:
    valid = [record for record in records if record["feature"] is not None]
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": structured.sha256_file(path),
        "configuration_sha256": config_sha256,
        "records": int(len(records)),
        "valid_feature_curves": int(len(valid)),
        "invalid_views": int(len(records) - len(valid)),
        "parent_genealogies": int(len({record["parent_genealogy_id"] for record in records})),
        "view_counts": dict(sorted(Counter(record["view"] for record in records).items())),
    }


def validity_audit(records: Sequence[dict]) -> dict:
    """Audit eligibility before exclusions, including complete A/B/C triplets."""
    records = list(records)
    if not records:
        raise ValueError("observation bridge has no records")

    def summarize(current: Sequence[dict]) -> dict:
        usable = sum(record.get("feature") is not None for record in current)
        return {
            "records": int(len(current)),
            "usable": int(usable),
            "invalid": int(len(current) - usable),
            "usable_fraction": None if not current else float(usable / len(current)),
        }

    parents = sorted({str(record["parent_genealogy_id"]) for record in records})
    complete_parents = []
    by_parent = {}
    for parent in parents:
        current = [record for record in records if record["parent_genealogy_id"] == parent]
        present = {record["view"] for record in current}
        complete = (
            present == set(VIEW_SPECS)
            and len(current) == len(VIEW_SPECS)
            and all(record.get("feature") is not None for record in current)
        )
        by_parent[parent] = {**summarize(current), "all_views_usable": bool(complete)}
        if complete:
            complete_parents.append(parent)
    complete_parent_set = set(complete_parents)
    class_summaries = {}
    for label in CLASSES:
        current = [record for record in records if record["label"] == label]
        class_parents = sorted({str(record["parent_genealogy_id"]) for record in current})
        class_complete = [parent for parent in class_parents if parent in complete_parent_set]
        class_summaries[str(label)] = {
            **summarize(current),
            "parents": int(len(class_parents)),
            "complete_parents": int(len(class_complete)),
            "complete_parent_fraction": (
                None if not class_parents else float(len(class_complete) / len(class_parents))
            ),
        }

    rate_families = sorted({str(record["rate_family_id"]) for record in records})
    by_rate_family = {}
    complete_rate_families = []
    complete_family_parent_ids = []
    expected_labels = set(map(str, CLASSES))
    for family in rate_families:
        current = [record for record in records if str(record["rate_family_id"]) == family]
        family_parents = sorted({str(record["parent_genealogy_id"]) for record in current})
        label_to_parents = {
            label: sorted({
                str(record["parent_genealogy_id"])
                for record in current
                if str(record["label"]) == label
            })
            for label in expected_labels
        }
        one_parent_per_class = (
            set(label_to_parents) == expected_labels
            and all(len(values) == 1 for values in label_to_parents.values())
        )
        rate_hexes = {float(record["rate"]).hex() for record in current}
        rate_indices = {int(record["rate_index"]) for record in current}
        profile_seeds = {int(record["nuisance_profile_seed"]) for record in current}
        sample_hashes_by_copies = {
            copies: {
                record.get("view_audit", {}).get("base_sample_sha256")
                for record in current
                if int(VIEW_SPECS[str(record["view"])]["sample_copies"]) == copies
            }
            for copies in (32, 64, GENE_COPIES)
        }
        metadata_consistent = (
            len(rate_hexes) == 1
            and len(rate_indices) == 1
            and len(profile_seeds) == 1
            and all(
                len(values) == 1 and all(_is_sha256(value) for value in values)
                for values in sample_hashes_by_copies.values()
            )
        )
        family_complete = (
            one_parent_per_class
            and len(family_parents) == len(CLASSES)
            and len(current) == len(CLASSES) * len(VIEW_SPECS)
            and metadata_consistent
            and all(parent in complete_parent_set for parent in family_parents)
        )
        by_rate_family[family] = {
            **summarize(current),
            "parents": int(len(family_parents)),
            "labels": sorted(label for label, values in label_to_parents.items() if values),
            "one_parent_per_class": bool(one_parent_per_class),
            "metadata_consistent": bool(metadata_consistent),
            "fixed_sample_hashes_by_copies": {
                str(copies): sorted(map(str, values))
                for copies, values in sample_hashes_by_copies.items()
            },
            "all_classes_all_views_usable": bool(family_complete),
        }
        if family_complete:
            complete_rate_families.append(family)
            complete_family_parent_ids.extend(family_parents)
    result = {
        "overall": summarize(records),
        "by_class": class_summaries,
        "by_view": {
            view: summarize([record for record in records if record["view"] == view])
            for view in VIEW_SPECS
        },
        "by_rate_family": by_rate_family,
        "parents": int(len(parents)),
        "complete_parents": int(len(complete_parents)),
        "complete_parent_fraction": float(len(complete_parents) / len(parents)),
        "complete_parent_ids": complete_parents,
        "rate_families": int(len(rate_families)),
        "complete_rate_families": int(len(complete_rate_families)),
        "complete_rate_family_fraction": float(
            len(complete_rate_families) / len(rate_families)
        ),
        "complete_rate_family_ids": complete_rate_families,
        "complete_family_parent_ids": sorted(complete_family_parent_ids),
        "by_parent": by_parent,
        "decision_contract": (
            "Invalid views are abstentions/failures. A rate family is modeled only when its "
            "A/B/C parents each have every prespecified view, preventing class-dependent "
            "survivorship and preserving paired exact-rate comparisons."
        ),
    }
    return result


def view_metrics(cv_result: dict, records: Sequence[dict]) -> dict:
    rows = cv_result["oof_predictions"]
    if len(rows) != len(records):
        raise AssertionError("OOF row/bridge record count differs")
    result = {}
    for view in VIEW_SPECS:
        use = [index for index, record in enumerate(records) if record["view"] == view]
        if not use:
            continue
        truth = np.array([np.searchsorted(CLASSES, records[index]["label"]) for index in use])
        prediction = np.array([np.searchsorted(CLASSES, rows[index]["mean_prediction"]) for index in use])
        result[view] = {
            "n": int(len(use)),
            "accuracy": float(np.mean(prediction == truth)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "macro_f1": float(
                f1_score(
                    truth,
                    prediction,
                    average="macro",
                    labels=np.arange(len(CLASSES)),
                    zero_division=0,
                )
            ),
            "median_rms_z": float(np.median([rows[index]["mean_rms_z"] for index in use])),
        }
    return result


def leave_view_out(
    features: np.ndarray,
    labels: np.ndarray,
    parents: np.ndarray,
    views: np.ndarray,
    *,
    C_grid: Sequence[float],
    outer_splits: int,
    inner_splits: int,
    seed: int,
    compute_state: Path | None = None,
) -> dict:
    """Test each observation view after excluding that view from every fit."""
    result = {}
    for target_view in sorted(np.unique(views)):
        probability = np.full((len(labels), len(CLASSES)), np.nan, dtype=float)
        scored = np.zeros(len(labels), dtype=bool)
        folds = structured.grouped_folds(
            labels,
            parents,
            n_splits=outer_splits,
            seed=seed,
        )
        fold_ledger = []
        for fold_index, (outer_train, outer_test) in enumerate(folds):
            train = outer_train[views[outer_train] != target_view]
            test = outer_test[views[outer_test] == target_view]
            if not len(test):
                continue
            selected_C, inner = structured._choose_C(
                features[train],
                labels[train],
                parents[train],
                grid=C_grid,
                seed=seed * 100 + fold_index,
                n_splits=inner_splits,
                compute_state=compute_state,
            )
            if compute_state is not None:
                structured.compute_gate(compute_state)
            scaler, model = structured._fit_model(features[train], labels[train], selected_C)
            probability[test] = model.predict_proba(scaler.transform(features[test]))
            scored[test] = True
            fold_ledger.append({
                "fold": int(fold_index),
                "selected_C": selected_C,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "target_view_absent_from_training": True,
                "parent_overlap": False,
                "inner_selection": inner,
            })
        target = (views == target_view)
        if not np.array_equal(scored, target):
            raise AssertionError(f"leave-view-out ledger incomplete for {target_view}")
        prediction = probability[target].argmax(axis=1)
        truth = labels[target]
        result[str(target_view)] = {
            "n": int(target.sum()),
            "accuracy": float(np.mean(prediction == truth)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "macro_f1": float(
                f1_score(
                    truth,
                    prediction,
                    average="macro",
                    labels=np.arange(len(CLASSES)),
                    zero_division=0,
                )
            ),
            "prediction_counts": {
                str(label): int(np.sum(prediction == index))
                for index, label in enumerate(CLASSES)
            },
            "fold_ledger": fold_ledger,
        }
    return result


def leave_factor_out(
    features: np.ndarray,
    labels: np.ndarray,
    parents: np.ndarray,
    views: np.ndarray,
    *,
    C_grid: Sequence[float],
    outer_splits: int,
    inner_splits: int,
    seed: int,
    compute_state: Path | None = None,
) -> dict:
    """Exclude every composite view containing one observation factor."""
    factors = sorted({
        factor
        for specification in VIEW_SPECS.values()
        for factor in specification["factor_tags"]
    })
    result = {}
    for factor in factors:
        target_views = {
            view
            for view, specification in VIEW_SPECS.items()
            if factor in specification["factor_tags"]
        }
        target = np.isin(views, sorted(target_views))
        probability = np.full((len(labels), len(CLASSES)), np.nan, dtype=float)
        scored = np.zeros(len(labels), dtype=bool)
        folds = structured.grouped_folds(
            labels,
            parents,
            n_splits=outer_splits,
            seed=seed,
        )
        fold_ledger = []
        for fold_index, (outer_train, outer_test) in enumerate(folds):
            train = outer_train[~np.isin(views[outer_train], sorted(target_views))]
            test = outer_test[np.isin(views[outer_test], sorted(target_views))]
            if not len(train) or not len(test):
                raise AssertionError(f"factor holdout {factor} produced an empty fold")
            selected_C, inner = structured._choose_C(
                features[train],
                labels[train],
                parents[train],
                grid=C_grid,
                seed=seed * 100 + fold_index,
                n_splits=inner_splits,
                compute_state=compute_state,
            )
            if compute_state is not None:
                structured.compute_gate(compute_state)
            scaler, model = structured._fit_model(features[train], labels[train], selected_C)
            probability[test] = model.predict_proba(scaler.transform(features[test]))
            scored[test] = True
            fold_ledger.append({
                "fold": int(fold_index),
                "selected_C": selected_C,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "target_views": sorted(target_views),
                "target_factor_absent_from_training": True,
                "parent_overlap": False,
                "inner_selection": inner,
            })
        if not np.array_equal(scored, target):
            raise AssertionError(f"factor holdout ledger incomplete for {factor}")
        prediction = probability[target].argmax(axis=1)
        truth = labels[target]
        result[factor] = {
            "n": int(target.sum()),
            "target_views": sorted(target_views),
            "accuracy": float(np.mean(prediction == truth)),
            "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
            "macro_f1": float(
                f1_score(
                    truth,
                    prediction,
                    average="macro",
                    labels=np.arange(len(CLASSES)),
                    zero_division=0,
                )
            ),
            "fold_ledger": fold_ledger,
        }
    return result


def adjudicate_bridge(validity: dict, variants: dict) -> dict:
    """Apply thresholds frozen before any bridge or natural result is inspected."""
    raw = variants["raw_all"]
    structured_variant = variants["orbit_composition_mean_variance"]

    def mean_appreciable(current: dict) -> float:
        return float(np.mean([
            repeat["appreciable"]["accuracy"]
            for repeat in current["per_repeat_metrics"]
        ]))

    raw_genealogy = mean_appreciable(raw["genealogy_cv"])
    structured_genealogy = mean_appreciable(structured_variant["genealogy_cv"])
    raw_rate = mean_appreciable(raw["rate_family_cv"])
    structured_rate = mean_appreciable(structured_variant["rate_family_cv"])
    raw_factor = float(np.mean([
        value["balanced_accuracy"]
        for value in raw["leave_one_observation_factor_out"].values()
    ]))
    structured_factor = float(np.mean([
        value["balanced_accuracy"]
        for value in structured_variant["leave_one_observation_factor_out"].values()
    ]))
    raw_bundles = raw["natural_transfer"]["coverage"][
        "bundle_balanced_prespecified"
    ]["by_bundle"]
    structured_bundles = structured_variant["natural_transfer"]["coverage"][
        "bundle_balanced_prespecified"
    ]["by_bundle"]
    if set(raw_bundles) != set(structured_bundles):
        raise AssertionError("bridge natural bundle sets differ between representations")
    bundle_ratios = {
        bundle: (
            structured_bundles[bundle]["median_rms_z"]
            / raw_bundles[bundle]["median_rms_z"]
        )
        for bundle in sorted(raw_bundles)
    }
    ratios = np.asarray(list(bundle_ratios.values()), dtype=float)
    minimum_class_coverage = min(
        value["usable_fraction"] for value in validity["by_class"].values()
    )
    minimum_view_coverage = min(
        value["usable_fraction"] for value in validity["by_view"].values()
    )
    minimum_class_complete_parent_fraction = min(
        value["complete_parent_fraction"] for value in validity["by_class"].values()
    )
    criteria = {
        "overall_view_usability_at_least_0_90": bool(
            validity["overall"]["usable_fraction"] >= 0.90
        ),
        "each_class_view_usability_at_least_0_85": bool(minimum_class_coverage >= 0.85),
        "each_view_usability_at_least_0_75": bool(minimum_view_coverage >= 0.75),
        "complete_parent_fraction_at_least_0_80": bool(
            validity["complete_parent_fraction"] >= 0.80
        ),
        "complete_rate_family_fraction_at_least_0_80": bool(
            validity["complete_rate_family_fraction"] >= 0.80
        ),
        "each_class_complete_parent_fraction_at_least_0_80": bool(
            minimum_class_complete_parent_fraction >= 0.80
        ),
        "structured_genealogy_appreciable_accuracy_at_least_0_90": bool(
            structured_genealogy >= 0.90
        ),
        "structured_rate_family_appreciable_accuracy_at_least_0_85": bool(
            structured_rate >= 0.85
        ),
        "structured_genealogy_loss_vs_raw_at_most_3_points": bool(
            structured_genealogy >= raw_genealogy - 0.03
        ),
        "structured_rate_family_loss_vs_raw_at_most_3_points": bool(
            structured_rate >= raw_rate - 0.03
        ),
        "structured_mean_factor_holdout_balanced_accuracy_at_least_0_60": bool(
            structured_factor >= 0.60
        ),
        "structured_factor_holdout_loss_vs_raw_at_most_3_points": bool(
            structured_factor >= raw_factor - 0.03
        ),
        "median_paired_natural_bundle_rms_z_ratio_at_most_0_80": bool(
            np.median(ratios) <= 0.80
        ),
        "at_least_70_percent_of_natural_bundles_reduce_rms_z": bool(
            np.mean(ratios < 1.0) >= 0.70
        ),
    }
    return {
        "prespecified_success_criteria": criteria,
        "all_criteria_pass": bool(all(criteria.values())),
        "validity_summary": {
            "overall_view_usability": validity["overall"]["usable_fraction"],
            "minimum_class_view_usability": minimum_class_coverage,
            "minimum_view_usability": minimum_view_coverage,
            "complete_parent_fraction": validity["complete_parent_fraction"],
            "complete_rate_family_fraction": validity["complete_rate_family_fraction"],
            "minimum_class_complete_parent_fraction": minimum_class_complete_parent_fraction,
        },
        "appreciable_accuracy": {
            "genealogy_cv": {"raw_all": raw_genealogy, "structured": structured_genealogy},
            "rate_family_cv": {"raw_all": raw_rate, "structured": structured_rate},
        },
        "mean_factor_holdout_balanced_accuracy": {
            "raw_all": raw_factor,
            "structured": structured_factor,
        },
        "natural_bundle_diagnostic": {
            "structured_over_raw_rms_z": bundle_ratios,
            "median_ratio": float(np.median(ratios)),
            "fraction_improved": float(np.mean(ratios < 1.0)),
            "accuracy_denominator": None,
        },
        "decision_rule": (
            "Passing only licenses a larger sealed bank spanning nulls, all six ordered edges, "
            "and held-out demography x observation families. Failure triggers simulation-design "
            "revision, never tuning on natural candidate labels."
        ),
    }


def analyze_records(
    records: Sequence[dict],
    *,
    seeds: Sequence[int],
    C_grid: Sequence[float],
    outer_splits: int,
    inner_splits: int,
    natural_paths: Sequence[Path],
    compute_state: Path | None = None,
) -> dict:
    validity = validity_audit(records)
    complete_families = set(validity["complete_rate_family_ids"])
    valid = [
        record
        for record in sorted(records, key=record_key)
        if record["rate_family_id"] in complete_families
    ]
    complete_family_count = len(complete_families)
    largest_outer_test = math.ceil(complete_family_count / outer_splits)
    smallest_outer_train = complete_family_count - largest_outer_test
    if complete_family_count < outer_splits or smallest_outer_train < inner_splits:
        raise RuntimeError(
            "too few complete A/B/C rate families for nested grouped analysis: "
            f"families={complete_family_count}, outer_splits={outer_splits}, "
            f"smallest_outer_train={smallest_outer_train}, inner_splits={inner_splits}"
        )
    expected_rows = complete_family_count * len(CLASSES) * len(VIEW_SPECS)
    expected_parents = complete_family_count * len(CLASSES)
    if len(valid) != expected_rows or len({record["parent_genealogy_id"] for record in valid}) != expected_parents:
        raise AssertionError("complete-family bridge row cardinality changed")
    table = np.stack([record["feature"] for record in valid]).astype(float)
    labels_text = np.array([record["label"] for record in valid])
    if set(map(str, labels_text)) != set(CLASSES):
        raise AssertionError("complete bridge parents lost a direction class")
    labels = np.searchsorted(CLASSES, labels_text)
    rates = np.array([record["rate"] for record in valid], dtype=float)
    designs = np.array(["bridge"] * len(valid))
    parents = np.array([record["parent_genealogy_id"] for record in valid])
    rate_families = np.array([record["rate_family_id"] for record in valid])
    views = np.array([record["view"] for record in valid])
    row_ids = np.array([f"{record['parent_genealogy_id']}|{record['view']}" for record in valid])
    variants = {}
    feature_bank = {}
    for name in structured.REPRESENTATIONS:
        if compute_state is not None:
            structured.compute_gate(compute_state)
        features = structured.representation_features(table, name)
        feature_bank[name] = features
        genealogy_cv = structured.nested_oof(
            features,
            labels,
            rates,
            designs,
            row_ids,
            parents,
            outer_name="new parent genealogy; every paired observation view blocked together",
            seeds=seeds,
            C_grid=C_grid,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            compute_state=compute_state,
        )
        rate_cv = structured.nested_oof(
            features,
            labels,
            rates,
            designs,
            row_ids,
            rate_families,
            outer_name="new exact rate family; all A/B/C genealogies and views blocked together",
            seeds=seeds,
            C_grid=C_grid,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            compute_state=compute_state,
        )
        variants[name] = {
            "feature_dimension": int(features.shape[1]),
            "genealogy_cv": genealogy_cv,
            "genealogy_cv_by_observation_view": view_metrics(genealogy_cv, valid),
            "rate_family_cv": rate_cv,
            "leave_one_observation_view_out": leave_view_out(
                features,
                labels,
                parents,
                views,
                C_grid=C_grid,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                seed=100_711,
                compute_state=compute_state,
            ),
            "leave_one_observation_factor_out": leave_factor_out(
                features,
                labels,
                parents,
                views,
                C_grid=C_grid,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                seed=110_711,
                compute_state=compute_state,
            ),
        }
    # Keep natural rows inaccessible until every simulation-only comparison and
    # nuisance-family holdout has completed.
    if compute_state is not None:
        structured.compute_gate(compute_state)
    panels, natural_audit = structured.load_natural_panels(natural_paths, max_depth=16)
    for name in structured.REPRESENTATIONS:
        variants[name]["natural_transfer"] = structured.final_natural_score(
            feature_bank[name],
            labels,
            rate_families,
            panels,
            representation=name,
            C_grid=C_grid,
            inner_splits=inner_splits,
            seed=90_711,
            compute_state=compute_state,
            verify_source_raw_head=False,
        )
    adjudication = adjudicate_bridge(validity, variants)
    return {
        "validity": validity,
        "modeled_rows": int(len(valid)),
        "excluded_rows_from_incomplete_rate_families": int(len(records) - len(valid)),
        "parent_genealogies": int(len(np.unique(parents))),
        "rate_families": int(len(np.unique(rate_families))),
        "views": dict(sorted(Counter(map(str, views)).items())),
        "natural_source_audit": natural_audit,
        "representations": variants,
        "bridge_adjudication": adjudication,
        "guardrail": (
            "This is a small A/B/C retraining bridge, not a causal decomposition or sealed final "
            "bank. It has no null and only three of six ordered edges. Invalid views are explicit "
            "abstentions; natural candidate labels are not used for selection or accuracy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-families", type=int, default=DEFAULT_RATE_FAMILIES)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--seeds", default="0,1")
    parser.add_argument("--C-grid", default="0.01,0.1,1,10")
    parser.add_argument("--outer-splits", type=int, default=4)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument(
        "--natural-result",
        type=Path,
        action="append",
        default=None,
        help="explicit results.json path(s); defaults to the pinned structured-pilot cohort",
    )
    parser.add_argument("--compute-state", type=Path, default=structured.DEFAULT_COMPUTE_STATE)
    parser.add_argument(
        "--compute-target",
        choices=("local", "azure"),
        default="local",
    )
    parser.add_argument(
        "--allow-stopped-trading-compute",
        action="store_true",
        help="use the same pressure-checked explicit stopped-trading authorization as the pilot",
    )
    args = parser.parse_args()
    os.environ[structured.COMPUTE_TARGET_ENV] = args.compute_target
    if args.allow_stopped_trading_compute:
        os.environ[structured.STOPPED_TRADING_AUTH_ENV] = "1"
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.outer_splits < 3 or args.inner_splits < 3:
        parser.error("outer/inner splits must be at least 3")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value != "")
    C_grid = tuple(float(value) for value in args.C_grid.split(",") if value != "")
    if not seeds or not C_grid or any(value <= 0 for value in C_grid):
        parser.error("seeds and positive C-grid values are required")

    initial_gate = structured.compute_gate(args.compute_state)
    priority = structured.set_below_normal_priority()
    revision = structured.git_revision(script=Path(__file__))
    jobs, rates = make_jobs(args.rate_families, args.seed_base)
    config = configuration(
        args.rate_families,
        args.seed_base,
        jobs=jobs,
        rates=rates,
    )
    config_sha = configuration_sha256(config)
    requested_jobs = jobs if args.limit is None else jobs[: args.limit]
    checkpoint = args.cache_dir / "observation_bridge_features.npz"
    with single_writer_lock(args.cache_dir):
        records = load_checkpoint(checkpoint, config_sha, jobs)
        existing = {record["parent_genealogy_id"] for record in records}
        for index, job in enumerate(requested_jobs, start=1):
            if job.parent_genealogy_id in existing:
                continue
            # A second hard gate makes a state transition stop before the next simulation.
            structured.compute_gate(args.compute_state)
            current = simulate_job(job, compute_state=args.compute_state)
            records.extend(current)
            existing.add(job.parent_genealogy_id)
            save_checkpoint(checkpoint, records, config_sha)
            print(
                f"[{index}/{len(requested_jobs)}] {job.parent_genealogy_id}: "
                f"{sum(record['feature'] is not None for record in current)}/{len(current)} usable views",
                flush=True,
            )
        wanted_parents = {job.parent_genealogy_id for job in requested_jobs}
        selected = [
            record for record in records
            if record["parent_genealogy_id"] in wanted_parents
        ]
        expected = len(requested_jobs) * len(VIEW_SPECS)
        if len(selected) != expected:
            raise RuntimeError(f"selected {len(selected)} checkpoint rows, expected {expected}")
    if args.simulate_only:
        print(json.dumps({
            "checkpoint": str(checkpoint.resolve()),
            "selected_records": len(selected),
            "configuration_sha256": config_sha,
        }, indent=2))
        return 0
    if args.limit is not None:
        parser.error("analysis is unavailable with --limit; use --simulate-only")

    result_lock = structured.SingleWriterLease(
        args.result_dir,
        ".observation_bridge_result.lock",
    ).acquire()
    # Analysis performs repeated model fitting, so re-check after the final
    # simulation/checkpoint write rather than relying on the last per-job gate.
    final_gate = structured.compute_gate(args.compute_state)
    natural_paths = (
        [path.resolve() for path in args.natural_result]
        if args.natural_result is not None
        else structured.pinned_natural_paths()
    )
    analysis = analyze_records(
        selected,
        seeds=seeds,
        C_grid=C_grid,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        natural_paths=natural_paths,
        compute_state=args.compute_state,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "exploratory_observation_bridge_not_paper_result",
        "git": revision,
        "initial_compute_gate": initial_gate,
        "pre_analysis_compute_gate": final_gate,
        "runtime": structured.runtime_audit(priority),
        "configuration": config,
        "configuration_sha256": config_sha,
        "exact_rate_draws": [
            {"rate_index": int(index), "rate": float(rate), "rate_hex": float(rate).hex()}
            for index, rate in enumerate(rates)
        ],
        "job_manifest": [asdict(job) for job in jobs],
        "checkpoint": checkpoint_audit(checkpoint, selected, config_sha),
        "analysis": analysis,
        "guardrail": (
            "Every observation view is grouped by parent and invalid views are explicit abstentions. "
            "This three-class/no-null retraining pilot cannot establish a general direction detector, "
            "a false-positive rate, or real-data accuracy. Natural rows remain unlabeled diagnostics."
        ),
        "next_step": (
            "Only if every prespecified bridge criterion passes, generate a sealed bank over held-out "
            "demography x ascertainment families, explicit nulls, and all six ordered edges."
        ),
    }
    args.result_dir.mkdir(parents=True, exist_ok=True)
    output = args.result_dir / "results.json"
    structured.write_json_atomic(output, result, indent=2)
    print(json.dumps({
        "output": str(output.resolve()),
        "checkpoint": result["checkpoint"],
        "analysis_rows": analysis["modeled_rows"],
        "all_criteria_pass": analysis["bridge_adjudication"]["all_criteria_pass"],
    }, indent=2, allow_nan=False))
    result_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
