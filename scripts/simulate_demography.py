#!/usr/bin/env python3
"""Regenerate the exact 3,200-replicate DNNaic coalescent design.

The archived ``regen_full`` arrays contain:

* 1,200 fixed-rate replicates: A/B/C x four rates x 100 replicates;
* 1,500 continuous-rate replicates: 100 rates drawn once from Exp(2.5e-4),
  shared across A/B/C, with five replicates per (rate, class); and
* 500 zero-migration controls (class D).

Each experimental population contributes 200 *gene copies* (``ploidy=1``), not
200 diploid individuals.  The global setting also fixes the continuous-time
coalescent scale: ``initial_size=10_000`` is a haploid coalescent size, equivalent
to diploid Ne=5,000.  The tree is ``((P1,P2),P3),P4`` with splits at
500/1,000/2,000 generations, a 1 Mb sequence, recombination rate
1.78e-8, and mutation rate 2e-8.

Migration labels describe forward-time donor -> recipient flow.  msprime follows
lineages backward in time, so its source and destination are reversed:

    A: P1 -> P2 forward   = source P2, destination P1 backward
    B: P2 -> P3 forward   = source P3, destination P2 backward
    C: P3 -> P2 forward   = source P2, destination P3 backward
    D: no migration

The command writes a deterministic JSON manifest (including its SHA-256) and CSV
metadata before simulating.  Corrected regeneration uses disjoint deterministic
ancestry and mutation seed streams.  ``--legacy-coupled-seeds`` records and replays
the historical same-integer seed policy for audit only.  ``--dry-run`` emits only
the plan.  The resulting
tree sequences are converted to the published 28-column PADZE arrays by
``scripts/extract_padze_from_trees.py``.

Usage:
    python scripts/simulate_demography.py --out-dir data/raw/trees --dry-run
    python scripts/simulate_demography.py --out-dir data/raw/trees
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np


MANIFEST_SCHEMA = "dnnaic-simulation-manifest-v2"

# Demographic constants used for the archived study.
NE = 10_000                      # haploid coalescent size; diploid-equivalent Ne=5,000
SPLIT_P12 = 500
SPLIT_P123 = 1_000
SPLIT_ROOT = 2_000
SEQ_LEN = 1_000_000
RECOMB = 1.78e-8
MUT = 2e-8
N_SAMPLE = 200                 # haploid gene copies per experimental population
PLOIDY = 1
EXPERIMENTAL_POPULATIONS = ("P1", "P2", "P3")

DISCRETE_RATES = (5e-7, 2.5e-6, 5e-5, 2.5e-4)
N_DISCRETE_REPS = 100
N_CONTINUOUS_RATES = 100
N_CONTINUOUS_REPS = 5
N_CONTROL_REPS = 500
CONTINUOUS_MEAN = 2.5e-4

DEFAULT_CONTINUOUS_SEED = 2026
DEFAULT_SHUFFLE_SEED = 12345
DEFAULT_SEED0 = 70000
DEFAULT_MUTATION_SEED0 = 170000

# Labels are forward-time donor -> recipient.  The msprime mapping is the
# corresponding backward-time lineage movement (recipient -> donor).
FORWARD_FLOW = {
    "A": ("P1", "P2"),
    "B": ("P2", "P3"),
    "C": ("P3", "P2"),
    "D": (None, None),
}
BACKWARD_MIGRATION = {
    case: ((recipient, donor) if donor is not None else (None, None))
    for case, (donor, recipient) in FORWARD_FLOW.items()
}
# Kept as a public alias for callers of the earlier script.
CASES = BACKWARD_MIGRATION


@dataclass(frozen=True)
class SimulationJob:
    """One true simulation replicate in manifest order."""

    ordinal: int
    case: str
    design: str
    rate: float
    rate_index: int | None
    replicate: int
    group: str
    ancestry_seed: int
    mutation_seed: int

    @property
    def seed(self) -> int:
        """Backward-compatible alias for the ancestry seed."""
        return self.ancestry_seed

    @property
    def tree_file(self) -> str:
        return f"{self.ordinal:05d}_{self.case}.trees"

    def manifest_record(self) -> dict[str, Any]:
        donor, recipient = FORWARD_FLOW[self.case]
        source, destination = BACKWARD_MIGRATION[self.case]
        return {
            "ordinal": self.ordinal,
            "class": self.case,
            "design": self.design,
            # Hex is the lossless, locale-independent representation used by the hash.
            "rate": self.rate,
            "rate_hex": self.rate.hex(),
            "rate_index": self.rate_index,
            "replicate": self.replicate,
            "group": self.group,
            "ancestry_seed": self.ancestry_seed,
            "mutation_seed": self.mutation_seed,
            "forward_donor": donor,
            "forward_recipient": recipient,
            "msprime_source": source,
            "msprime_destination": destination,
            "tree_file": self.tree_file,
        }


def study_config(
    *,
    continuous_seed: int = DEFAULT_CONTINUOUS_SEED,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    seed0: int = DEFAULT_SEED0,
    mutation_seed0: int = DEFAULT_MUTATION_SEED0,
    legacy_coupled_seeds: bool = False,
) -> dict[str, Any]:
    """Return the simulation settings covered by the manifest hash."""

    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "demography": {
            "haploid_coalescent_population_size": NE,
            "diploid_equivalent_effective_size": NE // 2,
            "split_generations": {
                "P1_P2": SPLIT_P12,
                "P12_P3": SPLIT_P123,
                "P123_P4": SPLIT_ROOT,
            },
            "sequence_length": SEQ_LEN,
            "recombination_rate": RECOMB,
            "mutation_rate": MUT,
            "ancestry_model": "StandardCoalescent",
            "mutation_model": "JC69",
            "discrete_genome": True,
            "experimental_populations": list(EXPERIMENTAL_POPULATIONS),
            "gene_copies_per_population": N_SAMPLE,
            "ploidy": PLOIDY,
        },
        "design": {
            "fixed_rates": list(DISCRETE_RATES),
            "fixed_replicates_per_rate_class": N_DISCRETE_REPS,
            "continuous_rate_count": N_CONTINUOUS_RATES,
            "continuous_replicates_per_rate_class": N_CONTINUOUS_REPS,
            "continuous_exponential_mean": CONTINUOUS_MEAN,
            "control_replicates": N_CONTROL_REPS,
        },
        "seeds": {
            "continuous_rates": continuous_seed,
            "job_shuffle": shuffle_seed,
            "ancestry_seed0": seed0,
            "mutation_seed0": seed0 if legacy_coupled_seeds else mutation_seed0,
            "ancestry_mutation_policy": (
                "legacy_same_integer" if legacy_coupled_seeds else "disjoint_deterministic_streams"
            ),
        },
        "forward_flow": {
            case: list(pair) for case, pair in FORWARD_FLOW.items()
        },
        "backward_msprime_migration": {
            case: list(pair) for case, pair in BACKWARD_MIGRATION.items()
        },
    }


def build_jobs(
    *,
    continuous_seed: int = DEFAULT_CONTINUOUS_SEED,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    seed0: int = DEFAULT_SEED0,
    mutation_seed0: int = DEFAULT_MUTATION_SEED0,
    legacy_coupled_seeds: bool = False,
) -> list[SimulationJob]:
    """Build the archived job set/order under the selected explicit seed policy."""

    # Raw tuple: case, design, rate, rate_index, replicate, stable group id.
    raw: list[tuple[str, str, float, int | None, int, str]] = []

    for case in ("A", "B", "C"):
        for rate_index, rate in enumerate(DISCRETE_RATES):
            for replicate in range(N_DISCRETE_REPS):
                group = f"{case}|fixed|{rate:.6e}|rep{replicate:03d}"
                raw.append((case, "fixed", float(rate), rate_index, replicate, group))

    # Draw once, then reuse every exact floating-point rate for all three classes.
    rng = np.random.Generator(np.random.PCG64(continuous_seed))
    continuous_rates = rng.exponential(
        scale=CONTINUOUS_MEAN, size=N_CONTINUOUS_RATES
    )
    for rate_index, raw_rate in enumerate(continuous_rates):
        rate = float(raw_rate)
        for case in ("A", "B", "C"):
            for replicate in range(N_CONTINUOUS_REPS):
                group = (
                    f"{case}|cont|r{rate_index:03d}_{rate:.6e}|"
                    f"rep{replicate:02d}"
                )
                raw.append(
                    (case, "continuous", rate, rate_index, replicate, group)
                )

    for replicate in range(N_CONTROL_REPS):
        raw.append(
            ("D", "control", 0.0, None, replicate,
             f"D|control|0|rep{replicate:03d}")
        )

    # This is the shuffle used by regenerate_true_replicates.py.  Seeds are assigned
    # after the shuffle: the first manifest job gets seed0 + 1.
    random.Random(shuffle_seed).shuffle(raw)
    return [
        SimulationJob(
            ordinal=ordinal,
            case=case,
            design=design,
            rate=rate,
            rate_index=rate_index,
            replicate=replicate,
            group=group,
            ancestry_seed=seed0 + ordinal,
            mutation_seed=(seed0 if legacy_coupled_seeds else mutation_seed0) + ordinal,
        )
        for ordinal, (case, design, rate, rate_index, replicate, group)
        in enumerate(raw, start=1)
    ]


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def compute_manifest_hash(
    config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> str:
    """Hash every design setting, label, rate, mapping, file, and replicate seed."""

    payload = {"config": config, "jobs": list(records)}
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def manifest_document(
    jobs: Sequence[SimulationJob], config: Mapping[str, Any]
) -> dict[str, Any]:
    records = [job.manifest_record() for job in jobs]
    return {
        "schema": MANIFEST_SCHEMA,
        "config": dict(config),
        "job_count": len(records),
        "manifest_hash": compute_manifest_hash(config, records),
        "jobs": records,
    }


def runtime_versions() -> dict[str, str]:
    """Record execution provenance without making the design hash environment-specific."""

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for package in ("msprime", "tskit", "numpy", "padze"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_manifest(
    out_dir: Path, jobs: Sequence[SimulationJob], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Write lossless JSON plus a human-readable CSV view of the same plan."""

    document = manifest_document(jobs, config)
    document["runtime"] = runtime_versions()
    (out_dir / "simulation_manifest.json").write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fields = list(document["jobs"][0]) if document["jobs"] else []
    with (out_dir / "sim_metadata.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(document["jobs"])
    return document


def build_demography(rate: float = 0.0, source: str | None = None,
                      destination: str | None = None):
    import msprime

    demography = msprime.Demography()
    for name in ("P1", "P2", "P3", "P4", "P12", "P123", "P1234"):
        demography.add_population(name=name, initial_size=NE)
    demography.add_population_split(
        time=SPLIT_P12, derived=["P1", "P2"], ancestral="P12"
    )
    demography.add_population_split(
        time=SPLIT_P123, derived=["P12", "P3"], ancestral="P123"
    )
    demography.add_population_split(
        time=SPLIT_ROOT, derived=["P123", "P4"], ancestral="P1234"
    )
    if source is not None and destination is not None and rate > 0:
        demography.set_migration_rate(
            source=source, dest=destination, rate=float(rate)
        )
    return demography


def simulate_one(job: SimulationJob, out_dir: Path) -> None:
    import msprime

    source, destination = BACKWARD_MIGRATION[job.case]
    demography = build_demography(job.rate, source, destination)
    tree_sequence = msprime.sim_ancestry(
        samples={population: N_SAMPLE for population in EXPERIMENTAL_POPULATIONS},
        sequence_length=SEQ_LEN,
        discrete_genome=True,
        recombination_rate=RECOMB,
        ploidy=PLOIDY,
        model=msprime.StandardCoalescent(),
        demography=demography,
        random_seed=job.ancestry_seed,
    )
    tree_sequence = msprime.sim_mutations(
        tree_sequence, rate=MUT, random_seed=job.mutation_seed,
        model=msprime.JC69(), discrete_genome=True,
    )
    tree_sequence.dump(str(out_dir / job.tree_file))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", default="data/raw/trees")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SHUFFLE_SEED,
        help="job-order shuffle seed (archive default: 12345)",
    )
    parser.add_argument(
        "--seed0", type=int, default=DEFAULT_SEED0,
        help="base for ancestry seeds; first job uses seed0+1 (archive: 70000)",
    )
    parser.add_argument(
        "--mutation-seed0", type=int, default=DEFAULT_MUTATION_SEED0,
        help="base for the corrected disjoint mutation seeds (default: 170000)",
    )
    parser.add_argument(
        "--legacy-coupled-seeds", action="store_true",
        help="audit-only replay of the archive's same ancestry/mutation integer seed",
    )
    parser.add_argument(
        "--cont-seed", type=int, default=DEFAULT_CONTINUOUS_SEED,
        help="PCG64 seed for the 100 shared continuous rates (archive: 2026)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="write and verify the manifest without running msprime",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="simulate only the first N manifest jobs (smoke testing only)",
    )
    args = parser.parse_args()

    full_jobs = build_jobs(
        continuous_seed=args.cont_seed,
        shuffle_seed=args.seed,
        seed0=args.seed0,
        mutation_seed0=args.mutation_seed0,
        legacy_coupled_seeds=args.legacy_coupled_seeds,
    )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    jobs = full_jobs if args.limit is None else full_jobs[:args.limit]
    config = study_config(
        continuous_seed=args.cont_seed,
        shuffle_seed=args.seed,
        seed0=args.seed0,
        mutation_seed0=args.mutation_seed0,
        legacy_coupled_seeds=args.legacy_coupled_seeds,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    document = write_manifest(out_dir, jobs, config)

    if not args.dry_run:
        try:
            import msprime  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "msprime is not installed; install DNNaic with the 'simulate' extra "
                "or use --dry-run"
            ) from exc
        for index, job in enumerate(jobs, start=1):
            simulate_one(job, out_dir)
            if index % 50 == 0 or index == len(jobs):
                print(
                    f"  [{index}/{len(jobs)}] {job.group} "
                    f"ancestry_seed={job.ancestry_seed} mutation_seed={job.mutation_seed}",
                    flush=True,
                )

    verb = "planned" if args.dry_run else "simulated"
    print(
        f"[simulate] {verb} {len(jobs)} replicates; "
        f"manifest sha256={document['manifest_hash']}"
    )
    print(f"[simulate] manifest -> {out_dir / 'simulation_manifest.json'}")
    if args.limit is not None:
        print("[simulate] NOTE: --limit produced a non-canonical smoke-test subset")
    elif len(jobs) != 3_200:
        raise RuntimeError(f"canonical manifest must contain 3,200 jobs, got {len(jobs)}")
    if not args.dry_run:
        print(
            "[simulate] next: python scripts/extract_padze_from_trees.py "
            f"--trees-dir {out_dir} --out data/simulation_data/regen_full"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
