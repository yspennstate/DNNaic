#!/usr/bin/env python3
"""simulate_main_study.py  --  Seeded msprime simulator for the 4-population,
variable-migration-rate MAIN STUDY described in manuscript_draft.tex.

WHY THIS EXISTS
---------------
The committed `scripts/pipeline/simulation.py` is the *EDA* simulator only:
  * 3 populations (A,B,C), split times 100/200 generations,
  * migration rate HARD-CODED to 0.25 (the "extreme 25%" used for the PCA EDA),
  * no rate argument, and it does NOT record which rate a replicate used.
The manuscript's MAIN study instead describes:
  * 4 populations with a hierarchical Wright-Fisher history:
        P1234 -> P123 + P4 (outgroup)   at 2000 generations
        P123  -> P12  + P3              at 1000 generations
        P12   -> P1   + P2              at  500 generations
    Ne = 10,000 for all; 200 diploid individuals sampled per population.
  * sequence length 1 Mbp, recombination 1.78e-8/bp/gen, mutation 2e-8/bp/gen.
  * migration CASES (backwards in time):  A: P1->P2,  B: P2->P3,  C: P3->P2,  D: control.
  * RATES: four discrete classes {5e-7, 2.5e-6, 5e-5, 2.5e-4} x 100 replicates each,
    PLUS 400 rates drawn from Exponential(mean = 2.5e-4) x 5 replicates each.

This script reproduces that design WITH SEEDS and, crucially, writes a metadata CSV
(Class, Replicate, rate, seed, source, dest) so the per-replicate migration rate is
recorded -- which `build_dataset.py` then uses to build a leakage-free dataset.
It samples the outgroup P4 too, so the SAME simulations support both the DNNaic
features (P1,P2,P3 via ADZE) and classical D-statistic baselines (which need P4).

USAGE
-----
    python simulate_main_study.py --out-dir data/raw/trees --seed 12345
    # then run datagen.sh's downstream steps (tskit vcf -> VCFtoSTRU -> ADZE -> ADZEtoCSV)
    # then: build_dataset.py --rate-meta data/raw/trees/sim_metadata.csv ...

Requires: msprime>=1.2.  Heavy: ~1300 replicates; run on a server/cluster.
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

    print(f"[simulate_main_study] {'PLANNED' if args.dry_run else 'SIMULATED'} "
          f"{len(jobs)} replicates; metadata -> {meta_path}")
    print("[simulate_main_study] downstream: tskit vcf -> VCFtoSTRU.py -> ADZE "
          "-> ADZEtoCSV.py -> build_dataset.py --rate-meta sim_metadata.csv")


if __name__ == "__main__":
    main()
