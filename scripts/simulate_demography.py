#!/usr/bin/env python3
"""Seeded msprime simulator for the four-population, variable-migration-rate study.

Reproduces the coalescent design from the manuscript: a hierarchical divergence history

    P1234 -> P123 + P4 (outgroup)   at 2000 generations
    P123  -> P12  + P3              at 1000 generations
    P12   -> P1   + P2              at  500 generations

with Ne = 10,000 throughout, 200 diploid individuals per population, 1 Mbp sequences, and
recombination 1.78e-8 and mutation 2e-8 per base pair per generation. Four migration cases,
backwards in time -- A: P1->P2, B: P2->P3, C: P3->P2, and D: no migration -- are run at four fixed
rates {5e-7, 2.5e-6, 5e-5, 2.5e-4} (100 replicates each) plus a continuum of 400 rates drawn from
Exponential(mean 2.5e-4), five replicates each.

Each replicate is written as a tree sequence alongside a metadata row (class, replicate, rate,
seed, source, dest), so the per-replicate migration rate is recorded for a leakage-free feature
build. The outgroup P4 is sampled as well, so the same simulations feed both the rarefaction
features (from P1,P2,P3) and the classical D-statistic baselines (which need an outgroup).

Usage:  python scripts/simulate_demography.py --out-dir data/raw/trees --seed 12345

PADZE then extracts the rarefaction features from the tree sequences to build the arrays the
loaders read (see data/README.md). Requires msprime>=1.2; the full design is ~1,300 replicates.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path

# Demographic constants (manuscript Methods)
NE = 10_000
SPLIT_P12 = 500        # P12 -> P1, P2
SPLIT_P123 = 1000      # P123 -> P12, P3
SPLIT_ROOT = 2000      # P1234 -> P123, P4
SEQ_LEN = 1_000_000
RECOMB = 1.78e-8
MUT = 2e-8
N_SAMPLE = 200         # diploid individuals per population

DISCRETE_RATES = [5e-7, 2.5e-6, 5e-5, 2.5e-4]
N_DISCRETE_REPS = 100
N_EXP_RATES = 400
N_EXP_REPS = 5
EXP_MEAN = 2.5e-4

# case -> (source, dest) for msprime backwards-in-time migration
CASES = {"A": ("P1", "P2"), "B": ("P2", "P3"), "C": ("P3", "P2"), "D": (None, None)}


def build_demography(rate=0.0, source=None, dest=None):
    import msprime
    d = msprime.Demography()
    for name in ["P1", "P2", "P3", "P4", "P12", "P123", "P1234"]:
        d.add_population(name=name, initial_size=NE)
    d.add_population_split(time=SPLIT_P12, derived=["P1", "P2"], ancestral="P12")
    d.add_population_split(time=SPLIT_P123, derived=["P12", "P3"], ancestral="P123")
    d.add_population_split(time=SPLIT_ROOT, derived=["P123", "P4"], ancestral="P1234")
    if source and dest and rate > 0:
        # continuous (backwards-in-time) migration from `source` into `dest`
        d.set_migration_rate(source=source, dest=dest, rate=rate)
    d.sort_events()
    return d


def simulate_one(case, rate, seed, out_dir, rep_label):
    import msprime
    source, dest = CASES[case]
    demog = build_demography(rate=rate, source=source, dest=dest)
    ts = msprime.sim_ancestry(
        samples={"P1": N_SAMPLE, "P2": N_SAMPLE, "P3": N_SAMPLE, "P4": N_SAMPLE},
        sequence_length=SEQ_LEN, recombination_rate=RECOMB, ploidy=2,
        demography=demog, random_seed=seed,
    )
    ts = msprime.sim_mutations(ts, rate=MUT, random_seed=seed + 1, keep=True)
    path = Path(out_dir) / f"{case}_rep_{rep_label}.trees"
    ts.dump(str(path))
    return source, dest


def iter_jobs(rng_seed):
    """Yield (case, rep_label, rate, seed). Replicate labels are unique per case."""
    import numpy as np
    rng = np.random.default_rng(rng_seed)
    # discrete-rate replicates for the 3 migration cases
    rep = 0
    for case in ["A", "B", "C"]:
        rep = 0
        for rate in DISCRETE_RATES:
            for _ in range(N_DISCRETE_REPS):
                rep += 1
                yield case, rep, rate, int(rng.integers(1, 2**31 - 1))
        # exponential-draw replicates (continuous rates) appended to same case stream
        for _ in range(N_EXP_RATES // 3 + 1):
            rate = float(rng.exponential(EXP_MEAN))
            for _ in range(N_EXP_REPS):
                rep += 1
                yield case, rep, rate, int(rng.integers(1, 2**31 - 1))
    # control D (no migration)
    for r in range(1, N_DISCRETE_REPS + 1):
        yield "D", r, 0.0, int(rng.integers(1, 2**31 - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/raw/trees")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--dry-run", action="store_true",
                    help="only write the job/metadata plan; do not run msprime")
    ap.add_argument("--limit", type=int, default=None, help="cap #replicates (testing)")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "sim_metadata.csv"
    jobs = list(iter_jobs(args.seed))
    if args.limit:
        jobs = jobs[: args.limit]

    if not args.dry_run:
        try:
            import msprime  # noqa: F401
        except ImportError:
            raise SystemExit("msprime not installed. `pip install msprime` "
                             "or use --dry-run to emit the plan only.")

    with open(meta_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Class", "Replicate", "rate", "seed", "source", "dest"])
        for i, (case, rep, rate, seed) in enumerate(jobs, 1):
            src, dst = CASES[case]
            if not args.dry_run:
                src, dst = simulate_one(case, rate, seed, out, rep)
                if i % 50 == 0:
                    print(f"  [{i}/{len(jobs)}] simulated {case}_rep_{rep} rate={rate:.3e}")
            w.writerow([case, rep, rate, seed, src or "", dst or ""])

    print(f"[simulate] {'planned' if args.dry_run else 'simulated'} "
          f"{len(jobs)} replicates; metadata -> {meta_path}")
    print("[simulate] next: extract PADZE rarefaction features from the tree sequences "
          "(see data/README.md)")


if __name__ == "__main__":
    main()
