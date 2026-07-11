#!/usr/bin/env python3
"""Run a published-demography, known-truth DNNaic transfer benchmark.

The benchmark uses stdpopsim 0.3.0's HomSap model
``OutOfAfricaExtendedNeandertalAdmixturePulse_3I21``.  Its only positive
backward-time migration edge is CEU -> NEA, so the forward-time event is
NEA -> CEU.  With P1=YRI, P2=CEU, and P3=NEA, that is DNNaic class C
(P3 -> P2).  A matched control deep-copies the catalog model and zeros only
the positive pulse rates.  It is a no-event/D condition and is never placed in
the A/B/C direction-accuracy denominator.

Every replicate is an independent one-megabase tree sequence with 100 diploid
individuals (200 gene copies) per population.  PADZE is computed on the full
g=2..199 curve.  The frozen direction head uses only g=2..16 and the current
54-dimensional raw mean/variance/SE summary.  A secondary frozen 243-D
appreciable-migration gate uses its historical eight-depth-plus-mean contract.

This is a single-direction simulation-transfer benchmark, not three-class
external accuracy and not natural-data validation.  The runner is
single-process, checkpoints after every replicate, checks the compute governor
before simulation and PADZE work, and requires source code identical to a
clean Git HEAD before producing publishable output.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import structured_transfer_pilot as structured


SCHEMA_VERSION = "dnnaic-stdpopsim-neanderthal-benchmark-v1"
CHECKPOINT_SCHEMA = "dnnaic-stdpopsim-neanderthal-checkpoint-v1"
SPECIES_ID = "HomSap"
MODEL_ID = "OutOfAfricaExtendedNeandertalAdmixturePulse_3I21"
PINNED_STDPOPSIM_VERSION = "0.3.0"
POPULATIONS = ("YRI", "CEU", "NEA")
DNNAIC_POPULATIONS = ("P1", "P2", "P3")
TRUE_DIRECTION = "C"
CONDITIONS = ("pulse", "control")
INDIVIDUALS_PER_POPULATION = 100
GENE_COPIES_PER_POPULATION = 200
SEQUENCE_LENGTH = 1_000_000
MUTATION_RATE = 2e-8
RECOMBINATION_RATE = 1.78e-8
FULL_DEPTHS = np.arange(2, 200, dtype=np.int64)
PRIMARY_MAX_DEPTH = 16
PRIMARY_DEPTHS = np.arange(2, PRIMARY_MAX_DEPTH + 1, dtype=np.int64)
MOMENTS = ("mean", "variance", "se")
BLOCKS = (
    "alpha_1", "alpha_2", "alpha_3",
    "pi_1", "pi_2", "pi_3",
    "pihat_12", "pihat_13", "pihat_23",
)
CURVE_COLUMNS = ["g"] + [f"{block}_{moment}" for block in BLOCKS for moment in MOMENTS]
DEFAULT_PAIRS = 30
DEFAULT_SEED_BASE = 711_300_001
DEFAULT_CACHE = (
    Path.home()
    / "Documents"
    / "Codex"
    / "2026-07-10"
    / "dnnaic-datasets2-data"
    / "stdpopsim_neanderthal_2026_07_11"
)
DEFAULT_RESULTS = REPO / "results" / "stdpopsim_neanderthal_benchmark_2026_07_11"


@dataclass(frozen=True)
class BenchmarkJob:
    family_index: int
    family_id: str
    condition: str
    job_id: str
    engine_seed: int


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(current) for key, current in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(current) for current in value]
    return value


def _canonical_json(value) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_array(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.tobytes()).hexdigest()


def make_jobs(pairs: int, seed_base: int) -> list[BenchmarkJob]:
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if not 0 < seed_base < 2**31 - 2 * pairs - 2:
        raise ValueError("seed base does not leave room for unique engine seeds")
    jobs = []
    for family_index in range(pairs):
        family_id = f"3I21-family-{family_index:04d}"
        for condition_index, condition in enumerate(CONDITIONS):
            seed = seed_base + 2 * family_index + condition_index
            jobs.append(BenchmarkJob(
                family_index=family_index,
                family_id=family_id,
                condition=condition,
                job_id=f"{family_id}__{condition}",
                engine_seed=seed,
            ))
    if len({job.job_id for job in jobs}) != len(jobs):
        raise AssertionError("benchmark job IDs collide")
    if len({job.engine_seed for job in jobs}) != len(jobs):
        raise AssertionError("benchmark engine seeds collide")
    return jobs


def _event_signature(event) -> dict:
    payload = {"event_type": type(event).__name__}
    # ``Demography.validate`` attaches a runtime back-reference named
    # ``demography`` to each event.  It is not model content and necessarily
    # differs between a deep copy and its source.
    payload.update({
        key: _jsonable(value)
        for key, value in vars(event).items()
        if key != "demography"
    })
    return payload


def _population_signature(population) -> dict:
    return {key: _jsonable(value) for key, value in vars(population).items()}


def _migration_audit(demography) -> dict:
    import msprime

    ceu = int(demography["CEU"].id)
    nea = int(demography["NEA"].id)
    positive = []
    all_positive_edges = {}
    pair_events = []
    global_rate_events = []
    mass_migrations = []
    for index, event in enumerate(demography.events):
        if isinstance(event, msprime.MassMigration):
            mass_migrations.append({
                "time": float(event.time),
                "source": int(event.source),
                "dest": int(event.dest),
                "proportion": float(event.proportion),
            })
        if not isinstance(event, msprime.MigrationRateChange):
            continue
        rate = float(event.rate)
        source = int(event.source)
        dest = int(event.dest)
        if rate > 0:
            edge = f"{source}->{dest}"
            all_positive_edges[edge] = all_positive_edges.get(edge, 0) + 1
            positive.append({
                "event_index": int(index),
                "time": float(event.time),
                "rate": rate,
                "source": source,
                "dest": dest,
            })
        if (source, dest) == (ceu, nea) or (source, dest) == (-1, -1):
            pair_events.append((float(event.time), index, rate, source, dest))
        if (source, dest) == (-1, -1):
            global_rate_events.append({"time": float(event.time), "rate": rate})

    # Integrate the realized backward CEU->NEA rate through the exact event ledger.
    current_rate = 0.0
    previous_time = 0.0
    hazard = 0.0
    for event_time, _index, rate, source, dest in sorted(pair_events):
        if event_time < previous_time:
            raise AssertionError("migration events are not time ordered")
        hazard += current_rate * (event_time - previous_time)
        if (source, dest) == (-1, -1) or (source, dest) == (ceu, nea):
            current_rate = rate
        previous_time = event_time
    if current_rate != 0:
        raise AssertionError("migration ledger ends with a nonzero rate")

    positive_payload = [
        [entry["time"], entry["rate"], entry["source"], entry["dest"]]
        for entry in positive
    ]
    return {
        "event_count": len(demography.events),
        "population_ids": {"CEU": ceu, "NEA": nea},
        "positive_event_count": len(positive),
        "positive_edge_counts": dict(sorted(all_positive_edges.items())),
        "positive_time_min": min((entry["time"] for entry in positive), default=None),
        "positive_time_max": max((entry["time"] for entry in positive), default=None),
        "positive_times_are_exact_856_through_2153": (
            [entry["time"] for entry in positive] == list(range(856, 2154))
        ),
        "sum_of_positive_event_rates": float(math.fsum(entry["rate"] for entry in positive)),
        "integrated_backward_CEU_to_NEA_hazard": float(hazard),
        "forward_NEA_to_CEU_single_lineage_probability": float(-math.expm1(-hazard)),
        "positive_event_ledger_sha256": hashlib.sha256(
            _canonical_json(positive_payload)
        ).hexdigest(),
        "global_rate_events": global_rate_events,
        "mass_migrations": mass_migrations,
    }


def _normalized_model_signature(model, *, zero_pulse: bool) -> dict:
    import msprime

    ceu = int(model.model["CEU"].id)
    nea = int(model.model["NEA"].id)
    events = []
    for event in model.model.events:
        payload = _event_signature(event)
        if (
            zero_pulse
            and isinstance(event, msprime.MigrationRateChange)
            and int(event.source) == ceu
            and int(event.dest) == nea
            and float(event.rate) > 0
        ):
            payload["rate"] = 0.0
        events.append(payload)
    return {
        "populations": [_population_signature(population) for population in model.model.populations],
        "events": events,
        "generation_time": float(model.generation_time),
        "mutation_rate": float(model.mutation_rate),
    }


def prepare_models() -> tuple[dict[str, object], dict]:
    import msprime
    import stdpopsim

    observed_version = importlib_metadata.version("stdpopsim")
    if observed_version != PINNED_STDPOPSIM_VERSION:
        raise RuntimeError(
            f"stdpopsim version is {observed_version}, expected {PINNED_STDPOPSIM_VERSION}"
        )
    species = stdpopsim.get_species(SPECIES_ID)
    pulse = species.get_demographic_model(MODEL_ID)
    if pulse.id != MODEL_ID:
        raise AssertionError("stdpopsim demographic model ID changed")
    if tuple(population.name for population in pulse.model.populations) != POPULATIONS:
        raise AssertionError("stdpopsim population order changed")
    if float(pulse.generation_time) != 29 or float(pulse.mutation_rate) != MUTATION_RATE:
        raise AssertionError("stdpopsim generation-time or mutation-rate contract changed")
    pulse.model.validate()

    pulse_audit = _migration_audit(pulse.model)
    expected_edge = (
        f"{pulse.model['CEU'].id}->{pulse.model['NEA'].id}"
    )
    if pulse_audit["positive_edge_counts"] != {expected_edge: 1298}:
        raise AssertionError(
            "catalog pulse no longer consists solely of 1,298 backward CEU->NEA events"
        )
    if not math.isclose(
        pulse_audit["integrated_backward_CEU_to_NEA_hazard"],
        0.03,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise AssertionError("catalog pulse integrated hazard is no longer 0.03")
    if not pulse_audit["positive_times_are_exact_856_through_2153"]:
        raise AssertionError("catalog pulse support is no longer every generation 856..2153")
    if not math.isclose(
        pulse_audit["sum_of_positive_event_rates"], 0.03, rel_tol=0, abs_tol=1e-14
    ):
        raise AssertionError("catalog pulse positive-rate sum is no longer 0.03")
    if pulse_audit["event_count"] != 1302:
        raise AssertionError("catalog model event count changed")
    if pulse_audit["global_rate_events"] != [
        {"time": 855.0, "rate": 0.0},
        {"time": 2154.0, "rate": 0.0},
    ]:
        raise AssertionError("catalog global pulse-boundary events changed")
    if pulse_audit["mass_migrations"] != [
        {"time": 2550.0, "source": 1, "dest": 0, "proportion": 1.0},
        {"time": 10000.0, "source": 2, "dest": 0, "proportion": 1.0},
    ]:
        raise AssertionError("catalog split mass migrations changed")

    control = copy.deepcopy(pulse)
    ceu = int(control.model["CEU"].id)
    nea = int(control.model["NEA"].id)
    zeroed = 0
    for event in control.model.events:
        if isinstance(event, msprime.MigrationRateChange) and float(event.rate) > 0:
            if (int(event.source), int(event.dest)) != (ceu, nea):
                raise AssertionError("refusing to zero a positive migration edge other than CEU->NEA")
            event.rate = 0.0
            zeroed += 1
    if zeroed != pulse_audit["positive_event_count"]:
        raise AssertionError("control did not zero every positive pulse event")
    control.model.sort_events()
    control.model.validate()
    control_audit = _migration_audit(control.model)
    if control_audit["positive_event_count"] != 0:
        raise AssertionError("matched control retains a positive migration event")
    if control_audit["integrated_backward_CEU_to_NEA_hazard"] != 0:
        raise AssertionError("matched control retains a nonzero migration hazard")

    normalized_pulse = _normalized_model_signature(pulse, zero_pulse=True)
    exact_control = _normalized_model_signature(control, zero_pulse=False)
    if normalized_pulse != exact_control:
        raise AssertionError("matched control differs from pulse model beyond zeroed pulse rates")
    pulse_signature = _normalized_model_signature(pulse, zero_pulse=False)
    audit = {
        "stdpopsim_version": observed_version,
        "species_id": SPECIES_ID,
        "model_id": MODEL_ID,
        "generation_time_years": float(pulse.generation_time),
        "model_mutation_rate": float(pulse.mutation_rate),
        "population_order": list(POPULATIONS),
        "dnnaic_mapping": {"P1": "YRI", "P2": "CEU", "P3": "NEA"},
        "backward_event": "CEU->NEA",
        "forward_event": "NEA->CEU",
        "dnnaic_true_direction": TRUE_DIRECTION,
        "pulse": pulse_audit,
        "control": control_audit,
        "zeroed_positive_event_count": zeroed,
        "pulse_model_signature_sha256": hashlib.sha256(
            _canonical_json(pulse_signature)
        ).hexdigest(),
        "control_model_signature_sha256": hashlib.sha256(
            _canonical_json(exact_control)
        ).hexdigest(),
        "normalized_pulse_matches_control": True,
    }
    return {"pulse": pulse, "control": control}, audit


def make_contig():
    import stdpopsim

    contig = stdpopsim.Contig.basic_contig(
        length=SEQUENCE_LENGTH,
        mutation_rate=MUTATION_RATE,
        recombination_rate=RECOMBINATION_RATE,
        ploidy=2,
    )
    if int(contig.length) != SEQUENCE_LENGTH or int(contig.ploidy) != 2:
        raise AssertionError("stdpopsim contig length/ploidy contract changed")
    if not math.isclose(float(contig.mutation_rate), MUTATION_RATE, rel_tol=0, abs_tol=0):
        raise AssertionError("stdpopsim contig mutation rate changed")
    if not math.isclose(
        float(contig.recombination_map.mean_rate), RECOMBINATION_RATE, rel_tol=0, abs_tol=1e-30
    ):
        raise AssertionError("stdpopsim contig recombination rate changed")
    return contig


def _engine_seeds(seed: int) -> list[int]:
    # stdpopsim 0.3.0 derives one ancestry and one mutation seed this way.
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.integers(1, 2**31 - 1, size=2)]


def tree_sequence_to_curve(
    tree_sequence,
    population_ids: dict[str, int],
    *,
    compute_state: Path | None = None,
) -> tuple[np.ndarray, dict]:
    from padze import LociData, Metadata, compute_features

    samples = tree_sequence.samples()
    sample_population = tree_sequence.tables.nodes.population[samples]
    masks = [sample_population == population_ids[name] for name in POPULATIONS]
    gene_copy_counts = [int(mask.sum()) for mask in masks]
    if gene_copy_counts != [GENE_COPIES_PER_POPULATION] * 3:
        raise AssertionError(
            f"tree sequence gene-copy counts are {gene_copy_counts}, expected 200 each"
        )
    if int(tree_sequence.num_samples) != 3 * GENE_COPIES_PER_POPULATION:
        raise AssertionError("tree sequence total sample count changed")
    if int(tree_sequence.num_individuals) != 3 * INDIVIDUALS_PER_POPULATION:
        raise AssertionError("tree sequence diploid individual count changed")

    count_matrices = []
    sample_sizes = []
    locus_ids = []
    count_hash = hashlib.sha256()
    multiallelic = 0
    for variant in tree_sequence.variants(copy=False):
        genotype = np.asarray(variant.genotypes)
        if genotype.shape != (3 * GENE_COPIES_PER_POPULATION,) or np.any(genotype < 0):
            raise AssertionError("simulated genotype vector is missing or incorrectly shaped")
        allele_count = int(genotype.max()) + 1
        if allele_count < 1:
            continue
        if allele_count > 4:
            raise AssertionError("stdpopsim's default JC69 mutation model emitted >4 alleles")
        counts = np.stack([
            np.bincount(genotype[mask], minlength=allele_count)
            for mask in masks
        ]).astype(np.int64, copy=False)
        present = counts.sum(axis=0) > 0
        if int(present.sum()) < 2:
            continue
        counts = counts[:, present]
        if counts.shape[1] > 2:
            multiallelic += 1
        if not np.array_equal(counts.sum(axis=1), np.full(3, GENE_COPIES_PER_POPULATION)):
            raise AssertionError("allele counts do not conserve population sample sizes")
        locus_id = f"site-{variant.site.id}@{float(variant.site.position):.17g}"
        count_matrices.append(counts)
        sample_sizes.append(counts.sum(axis=1))
        locus_ids.append(locus_id)
        count_hash.update(np.asarray(counts.shape, dtype="<i8").tobytes())
        count_hash.update(counts.astype("<i8", copy=False).tobytes())
        count_hash.update(locus_id.encode("utf-8"))
        count_hash.update(b"\0")
    if len(count_matrices) < 2:
        raise RuntimeError("simulation produced too few globally polymorphic sites for PADZE")
    sizes = np.vstack(sample_sizes).astype(np.int64, copy=False)
    loci = LociData(
        populations=list(DNNAIC_POPULATIONS),
        count_matrices=count_matrices,
        sample_sizes=sizes,
        locus_ids=locus_ids,
        metadata=Metadata(
            source=f"stdpopsim {PINNED_STDPOPSIM_VERSION} {SPECIES_ID}/{MODEL_ID}",
            populations=list(DNNAIC_POPULATIONS),
            sample_ids={name: [] for name in DNNAIC_POPULATIONS},
            ploidy={name: 2 for name in DNNAIC_POPULATIONS},
            n_loci_read=int(tree_sequence.num_sites),
            n_loci_kept=len(count_matrices),
            filters_applied=["globally polymorphic across YRI/CEU/NEA; all alleles retained"],
            missing_fraction=0.0,
        ),
    )
    if compute_state is not None:
        structured.compute_gate(compute_state)
    table = compute_features(
        loci,
        depths=FULL_DEPTHS,
        pihat_sizes=(2,),
        moments=MOMENTS,
        bias_corrected=True,
    )
    matrix, columns = table.to_frame()
    index = {column: position for position, column in enumerate(columns)}
    try:
        ordered = matrix[:, [index[column] for column in CURVE_COLUMNS]].astype(np.float64)
    except KeyError as exc:
        raise RuntimeError(f"PADZE feature contract changed; missing {exc}") from exc
    if ordered.shape != (len(FULL_DEPTHS), len(CURVE_COLUMNS)):
        raise AssertionError(f"PADZE curve shape is {ordered.shape}, expected (198, 28)")
    if not np.isfinite(ordered).all() or not np.array_equal(ordered[:, 0], FULL_DEPTHS):
        raise AssertionError("PADZE curve is nonfinite or has the wrong depth grid")
    audit = {
        "num_trees": int(tree_sequence.num_trees),
        "num_sites": int(tree_sequence.num_sites),
        "num_mutations": int(tree_sequence.num_mutations),
        "num_individuals": int(tree_sequence.num_individuals),
        "num_sample_nodes": int(tree_sequence.num_samples),
        "gene_copies_by_population": dict(zip(POPULATIONS, gene_copy_counts)),
        "globally_polymorphic_loci": len(count_matrices),
        "multiallelic_globally_polymorphic_loci": int(multiallelic),
        "ordered_count_ledger_sha256": count_hash.hexdigest(),
        "curve_sha256_float64": _sha256_array(ordered.astype("<f8", copy=False)),
    }
    return ordered, audit


def simulate_job(
    job: BenchmarkJob,
    models: dict[str, object],
    contig,
    *,
    compute_state: Path | None = None,
) -> dict:
    import stdpopsim

    if job.condition not in models:
        raise ValueError(f"unknown benchmark condition {job.condition!r}")
    if compute_state is not None:
        structured.compute_gate(compute_state)
    started = time.perf_counter()
    model = models[job.condition]
    engine = stdpopsim.get_engine("msprime")
    tree_sequence = engine.simulate(
        demographic_model=model,
        contig=contig,
        samples={name: INDIVIDUALS_PER_POPULATION for name in POPULATIONS},
        seed=job.engine_seed,
    )
    population_ids = {name: int(model.model[name].id) for name in POPULATIONS}
    curve, simulation_audit = tree_sequence_to_curve(
        tree_sequence,
        population_ids,
        compute_state=compute_state,
    )
    curve32 = curve.astype(np.float32)
    return {
        **asdict(job),
        "engine_derived_ancestry_seed": _engine_seeds(job.engine_seed)[0],
        "engine_derived_mutation_seed": _engine_seeds(job.engine_seed)[1],
        "elapsed_seconds": float(time.perf_counter() - started),
        "simulation_audit": simulation_audit,
        "curve_sha256_float32": _sha256_array(curve32.astype("<f4", copy=False)),
        "curve": curve32,
    }


def record_key(record: dict) -> tuple[int, int]:
    condition_index = CONDITIONS.index(str(record["condition"]))
    return int(record["family_index"]), condition_index


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part.{os.getpid()}.{time.time_ns()}")
    try:
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(path: Path, records: Sequence[dict], config_sha256: str) -> None:
    records = sorted(records, key=record_key)
    job_ids = [str(record["job_id"]) for record in records]
    if len(set(job_ids)) != len(job_ids):
        raise RuntimeError("refusing to save duplicate benchmark jobs")
    curves = np.stack([np.asarray(record["curve"], dtype=np.float32) for record in records])
    if curves.shape != (len(records), len(FULL_DEPTHS), len(CURVE_COLUMNS)):
        raise RuntimeError("refusing to save an incorrectly shaped benchmark checkpoint")
    if not np.isfinite(curves).all() or not np.all(curves[:, :, 0] == FULL_DEPTHS[None, :]):
        raise RuntimeError("refusing to save invalid benchmark curves")
    metadata = []
    for record, curve in zip(records, curves):
        current = {key: value for key, value in record.items() if key != "curve"}
        observed = _sha256_array(curve.astype("<f4", copy=False))
        if observed != current["curve_sha256_float32"]:
            raise RuntimeError(f"curve hash changed for {current['job_id']}")
        metadata.append(current)
    _atomic_npz(
        path,
        schema=np.asarray([CHECKPOINT_SCHEMA]),
        config_sha256=np.asarray([config_sha256]),
        metadata_json=np.asarray([_canonical_json(metadata).decode("ascii")]),
        curves=curves,
    )


def load_checkpoint(
    path: Path,
    config_sha256: str,
    jobs: Sequence[BenchmarkJob],
) -> list[dict]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema", "config_sha256", "metadata_json", "curves"}
        if set(archive.files) != required:
            raise RuntimeError("benchmark checkpoint member set changed")
        if archive["schema"].tolist() != [CHECKPOINT_SCHEMA]:
            raise RuntimeError("benchmark checkpoint schema changed")
        if archive["config_sha256"].tolist() != [config_sha256]:
            raise RuntimeError("benchmark checkpoint configuration changed; use a fresh cache")
        metadata = json.loads(str(archive["metadata_json"][0]))
        curves = np.asarray(archive["curves"], dtype=np.float32)
    if not isinstance(metadata, list) or len(metadata) != len(curves):
        raise RuntimeError("benchmark checkpoint metadata/curve cardinality changed")
    manifest = {job.job_id: job for job in jobs}
    records = []
    seen = set()
    for current, curve in zip(metadata, curves):
        if not isinstance(current, dict) or current.get("job_id") not in manifest:
            raise RuntimeError("benchmark checkpoint contains an unknown job")
        job = manifest[current["job_id"]]
        expected = asdict(job)
        if any(current.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"benchmark checkpoint job manifest changed for {job.job_id}")
        if job.job_id in seen:
            raise RuntimeError("benchmark checkpoint contains a duplicate job")
        seen.add(job.job_id)
        if curve.shape != (len(FULL_DEPTHS), len(CURVE_COLUMNS)):
            raise RuntimeError(f"benchmark checkpoint curve shape changed for {job.job_id}")
        if not np.isfinite(curve).all() or not np.array_equal(curve[:, 0], FULL_DEPTHS):
            raise RuntimeError(f"benchmark checkpoint curve is invalid for {job.job_id}")
        if _sha256_array(curve.astype("<f4", copy=False)) != current.get("curve_sha256_float32"):
            raise RuntimeError(f"benchmark checkpoint curve hash changed for {job.job_id}")
        records.append({**current, "curve": curve})
    return sorted(records, key=record_key)


def checkpoint_audit(path: Path, records: Sequence[dict], config_sha256: str) -> dict:
    completed_families = {}
    for record in records:
        completed_families.setdefault(str(record["family_id"]), set()).add(str(record["condition"]))
    complete = [family for family, conditions in completed_families.items() if conditions == set(CONDITIONS)]
    return {
        "path": str(path.resolve()),
        "bytes": int(path.stat().st_size),
        "sha256": structured.sha256_file(path),
        "schema_version": CHECKPOINT_SCHEMA,
        "configuration_sha256": config_sha256,
        "records": len(records),
        "complete_pulse_control_families": len(complete),
        "curve_shape": [len(records), len(FULL_DEPTHS), len(CURVE_COLUMNS)],
        "record_curve_hash_ledger_sha256": hashlib.sha256(_canonical_json([
            [record["job_id"], record["curve_sha256_float32"]]
            for record in sorted(records, key=record_key)
        ])).hexdigest(),
    }


def _wilson(success: Iterable[bool]) -> dict:
    values = np.asarray(list(success), dtype=bool)
    n = int(len(values))
    if n == 0:
        return {"successes": 0, "n": 0, "fraction": None, "wilson_95": None}
    count = int(values.sum())
    p = count / n
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return {
        "successes": count,
        "n": n,
        "fraction": float(p),
        "wilson_95": [float(max(0, center - half)), float(min(1, center + half))],
    }


def _gate_features(table: np.ndarray) -> tuple[np.ndarray, dict]:
    table = structured.validate_curve_table(table)
    if table.shape[1] != len(FULL_DEPTHS) or not np.array_equal(table[0, :, 0], FULL_DEPTHS):
        raise ValueError("appreciable-gate input must carry the full g=2..199 grid")
    depth_indices = np.unique(
        np.round(np.geomspace(1, table.shape[1] - 1, 8)).astype(int)
    )
    curves = table[:, :, 1:]
    features = np.concatenate(
        [curves[:, depth_indices, :].reshape(len(table), -1), curves.mean(axis=1)],
        axis=1,
    )
    if features.shape[1] != 243 or not np.isfinite(features).all():
        raise AssertionError("frozen appreciable-gate feature contract changed")
    return features, {
        "feature_dimension": 243,
        "selected_depth_row_indices_zero_based": depth_indices.tolist(),
        "selected_depths": table[0, depth_indices, 0].astype(int).tolist(),
        "description": "27 coordinates at eight historical log-spaced rows plus curve mean",
    }


def _model_payload(scaler, model, *, feature_columns: Sequence[str] | None = None) -> dict:
    return {
        "estimator": "sklearn.linear_model.LogisticRegression",
        "scaler": "sklearn.preprocessing.StandardScaler",
        "C": float(model.C),
        "solver": str(model.solver),
        "max_iter": int(model.max_iter),
        "classes": [_jsonable(value) for value in model.classes_],
        "n_iter": np.asarray(model.n_iter_).astype(int).tolist(),
        "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
        "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
        "coef": np.asarray(model.coef_, dtype=float).tolist(),
        "intercept": np.asarray(model.intercept_, dtype=float).tolist(),
        "feature_columns": None if feature_columns is None else list(feature_columns),
    }


def _distribution_summary(values: Sequence[float]) -> dict:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("distribution summary requires a nonempty finite vector")
    return {
        "n": int(len(values)),
        "minimum": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "maximum": float(values.max()),
    }


def analyze_records(
    records: Sequence[dict],
    canonical_root: Path,
    *,
    compute_state: Path | None = None,
) -> dict:
    records = sorted(records, key=record_key)
    families = {}
    for record in records:
        families.setdefault(str(record["family_id"]), set()).add(str(record["condition"]))
    incomplete = {family: sorted(conditions) for family, conditions in families.items() if conditions != set(CONDITIONS)}
    if incomplete:
        raise RuntimeError(f"analysis requires complete pulse/control families: {incomplete}")
    if len(records) != 2 * len(families):
        raise RuntimeError("analysis record cardinality is not two jobs per family")
    curves = np.stack([np.asarray(record["curve"], dtype=float) for record in records])
    conditions = np.asarray([record["condition"] for record in records])
    if compute_state is not None:
        structured.compute_gate(compute_state)
    canonical = structured.load_canonical(canonical_root, max_depth=199)
    canonical_table = np.asarray(canonical["table"], dtype=float)
    canonical_labels = np.asarray(canonical["labels"])
    canonical_rates = np.asarray(canonical["rates"], dtype=float)
    positive = np.isin(canonical_labels, ["A", "B", "C"])
    if int(positive.sum()) != 2700 or dict(zip(*np.unique(canonical_labels[positive], return_counts=True))) != {
        "A": 900, "B": 900, "C": 900
    }:
        raise AssertionError("canonical positive direction training set changed")

    primary_canonical = canonical_table[:, : len(PRIMARY_DEPTHS)]
    primary_external = curves[:, : len(PRIMARY_DEPTHS)]
    if not np.array_equal(primary_canonical[0, :, 0], PRIMARY_DEPTHS):
        raise AssertionError("canonical primary depth grid changed")
    direction_train = structured.representation_features(primary_canonical[positive], "raw_all")
    direction_external = structured.representation_features(primary_external, "raw_all")
    if direction_train.shape != (2700, 54) or direction_external.shape != (len(records), 54):
        raise AssertionError("frozen 54-D direction-head feature contract changed")
    if compute_state is not None:
        structured.compute_gate(compute_state)
    direction_scaler, direction_model = structured._fit_model(
        direction_train, canonical_labels[positive], C=1.0
    )
    direction_z = direction_scaler.transform(direction_external)
    direction_probability = direction_model.predict_proba(direction_z)
    direction_prediction = direction_model.classes_[np.argmax(direction_probability, axis=1)]
    direction_class_index = {str(label): index for index, label in enumerate(direction_model.classes_)}
    if set(direction_class_index) != {"A", "B", "C"}:
        raise AssertionError("frozen direction model classes changed")
    c_probability = direction_probability[:, direction_class_index[TRUE_DIRECTION]]
    next_probability = np.max(
        direction_probability[:, [direction_class_index["A"], direction_class_index["B"]]],
        axis=1,
    )

    gate_train, gate_contract = _gate_features(canonical_table)
    gate_external, external_gate_contract = _gate_features(curves)
    if gate_contract != external_gate_contract:
        raise AssertionError("canonical/external appreciable-gate contracts differ")
    gate_target = (
        positive & (canonical_rates >= structured.APPRECIABLE)
    ).astype(int)
    if len(np.unique(gate_target)) != 2:
        raise AssertionError("canonical appreciable-gate target lost a class")
    if compute_state is not None:
        structured.compute_gate(compute_state)
    gate_scaler, gate_model = structured._fit_model(gate_train, gate_target, C=1.0)
    gate_z = gate_scaler.transform(gate_external)
    gate_positive_index = int(np.flatnonzero(gate_model.classes_ == 1)[0])
    gate_probability = gate_model.predict_proba(gate_z)[:, gate_positive_index]

    ledger = []
    for index, record in enumerate(records):
        is_pulse = conditions[index] == "pulse"
        ledger.append({
            "job_id": record["job_id"],
            "family_id": record["family_id"],
            "family_index": int(record["family_index"]),
            "condition": record["condition"],
            "direction_truth": TRUE_DIRECTION if is_pulse else None,
            "included_in_direction_accuracy": bool(is_pulse),
            "direction_prediction": str(direction_prediction[index]),
            "direction_correct": bool(direction_prediction[index] == TRUE_DIRECTION) if is_pulse else None,
            "direction_probability": {
                label: float(direction_probability[index, direction_class_index[label]])
                for label in ("A", "B", "C")
            },
            "C_vs_next_best_margin": float(c_probability[index] - next_probability[index]),
            "direction_scaler_rms_z": float(np.sqrt(np.mean(direction_z[index] ** 2))),
            "direction_scaler_max_abs_z": float(np.max(np.abs(direction_z[index]))),
            "appreciable_gate_probability": float(gate_probability[index]),
            "appreciable_gate_call_at_0_5": bool(gate_probability[index] >= 0.5),
            "gate_scaler_rms_z": float(np.sqrt(np.mean(gate_z[index] ** 2))),
            "gate_scaler_max_abs_z": float(np.max(np.abs(gate_z[index]))),
        })

    pulse = conditions == "pulse"
    control = conditions == "control"
    direction_correct = direction_prediction[pulse] == TRUE_DIRECTION
    gate_truth = pulse.astype(int)
    gate_call = gate_probability >= 0.5
    paired_gate_differences = []
    for family_id in sorted(families):
        indices = [
            index for index, record in enumerate(records)
            if str(record["family_id"]) == family_id
        ]
        by_condition = {str(records[index]["condition"]): index for index in indices}
        if set(by_condition) != set(CONDITIONS):
            raise AssertionError(f"paired score ledger is incomplete for {family_id}")
        paired_gate_differences.append(
            float(
                gate_probability[by_condition["pulse"]]
                - gate_probability[by_condition["control"]]
            )
        )
    simulation_ledger = [
        {key: value for key, value in record.items() if key != "curve"}
        for record in records
    ]
    simulation_summary = {}
    for condition in CONDITIONS:
        current = [record for record in records if record["condition"] == condition]
        simulation_summary[condition] = {
            "globally_polymorphic_loci": _distribution_summary([
                record["simulation_audit"]["globally_polymorphic_loci"]
                for record in current
            ]),
            "multiallelic_globally_polymorphic_loci": _distribution_summary([
                record["simulation_audit"]["multiallelic_globally_polymorphic_loci"]
                for record in current
            ]),
            "elapsed_seconds": _distribution_summary([
                record["elapsed_seconds"] for record in current
            ]),
        }
    result = {
        "statistical_unit": "one independent one-megabase ancestry/mutation realization",
        "complete_pulse_control_families": len(families),
        "records": len(records),
        "direction_head": {
            "status": "single_true_direction_C_recall; controls excluded from accuracy",
            "feature_representation": "raw_all g=2..16 depth mean+SD (54-D)",
            "canonical_training_replicates": int(positive.sum()),
            "canonical_training_class_counts": {
                str(label): int(count)
                for label, count in zip(*np.unique(canonical_labels[positive], return_counts=True))
            },
            "pulse_class_C_recall": _wilson(direction_correct),
            "trivial_always_C_pulse_recall": 1.0,
            "baseline_guardrail": (
                "Because every positive row has class-C truth, always predicting C has 100% "
                "recall. This benchmark can falsify class-C transfer but cannot establish "
                "three-class skill even if recall is perfect."
            ),
            "pulse_predicted_class_counts": {
                str(label): int(count)
                for label, count in zip(*np.unique(direction_prediction[pulse], return_counts=True))
            },
            "control_forced_call_counts_diagnostic_only": {
                str(label): int(count)
                for label, count in zip(*np.unique(direction_prediction[control], return_counts=True))
            },
            "pulse_C_probability_mean": float(c_probability[pulse].mean()),
            "pulse_C_probability_median": float(np.median(c_probability[pulse])),
            "control_C_probability_mean_diagnostic_only": float(c_probability[control].mean()),
            "pulse_C_vs_next_best_margin_mean": float((c_probability - next_probability)[pulse].mean()),
            "model": _model_payload(
                direction_scaler,
                direction_model,
                feature_columns=structured.representation_columns("raw_all"),
            ),
        },
        "appreciable_gate": {
            "status": (
                "secondary frozen transfer diagnostic; canonical target is constant-migration "
                "rate >=2.5e-4, not the stdpopsim pulse hazard"
            ),
            "contract": gate_contract,
            "canonical_training_replicates": int(len(gate_target)),
            "canonical_target_counts": {
                "appreciable": int(gate_target.sum()),
                "other": int((gate_target == 0).sum()),
            },
            "pulse_sensitivity_at_0_5": _wilson(gate_call[pulse]),
            "control_specificity_at_0_5": _wilson(~gate_call[control]),
            "matched_pulse_control_roc_auc": float(roc_auc_score(gate_truth, gate_probability)),
            "pulse_score_mean": float(gate_probability[pulse].mean()),
            "control_score_mean": float(gate_probability[control].mean()),
            "paired_pulse_minus_control_score": {
                **_distribution_summary(paired_gate_differences),
                "fraction_positive": float(np.mean(np.asarray(paired_gate_differences) > 0)),
                "guardrail": (
                    "Pairs share configuration labels but use independent ancestry/mutation seeds; "
                    "this is a descriptive matched-design contrast, not a paired genealogy effect."
                ),
            },
            "model": _model_payload(gate_scaler, gate_model),
        },
        "external_support": {
            "direction_scaler_rms_z_median": float(np.median(np.sqrt(np.mean(direction_z**2, axis=1)))),
            "direction_scaler_rms_z_p95": float(np.quantile(np.sqrt(np.mean(direction_z**2, axis=1)), 0.95)),
            "direction_scaler_max_abs_z_p95": float(np.quantile(np.max(np.abs(direction_z), axis=1), 0.95)),
            "gate_scaler_rms_z_median": float(np.median(np.sqrt(np.mean(gate_z**2, axis=1)))),
        },
        "simulation_summary": simulation_summary,
        "simulation_record_ledger": simulation_ledger,
        "prediction_ledger": ledger,
        "canonical_source_audit": canonical["audit"],
        "guardrail": (
            "Only pulse rows have direction truth and all have class C. Control rows have no A/B/C "
            "truth, so forced calls are descriptive. This cannot estimate balanced three-class "
            "accuracy, natural-data accuracy, or performance on ancient empirical genomes."
        ),
    }
    return result


def configuration(
    pairs: int,
    seed_base: int,
    jobs: Sequence[BenchmarkJob],
    model_audit: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "pairs": int(pairs),
        "seed_base": int(seed_base),
        "job_manifest": [asdict(job) for job in jobs],
        "model": model_audit,
        "sequence": {
            "length_bp": SEQUENCE_LENGTH,
            "mutation_rate": MUTATION_RATE,
            "recombination_rate": RECOMBINATION_RATE,
            "independent_contigs_per_job": 1,
        },
        "sampling": {
            "individuals_per_population": INDIVIDUALS_PER_POPULATION,
            "gene_copies_per_population": GENE_COPIES_PER_POPULATION,
            "population_order": list(POPULATIONS),
            "ploidy": 2,
        },
        "padze": {
            "depths": FULL_DEPTHS.tolist(),
            "pihat_sizes": [2],
            "moments": list(MOMENTS),
            "bias_corrected": True,
            "columns": CURVE_COLUMNS,
            "site_filter": "globally polymorphic; all observed alleles retained",
        },
        "primary_direction_head": {
            "depths": PRIMARY_DEPTHS.tolist(),
            "representation": "raw_all",
            "feature_dimension": 54,
            "C": 1.0,
        },
        "pairing_guardrail": (
            "family_id pairs configuration only; pulse/control have distinct seeds and do not share a genealogy"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=DEFAULT_PAIRS)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--limit-pairs", type=int, default=None)
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path(os.environ.get("DNNAIC_DATA", "data/simulation_data")) / "regen_full",
    )
    parser.add_argument("--compute-state", type=Path, default=structured.DEFAULT_COMPUTE_STATE)
    parser.add_argument("--compute-target", choices=("local", "azure"), default="local")
    parser.add_argument("--allow-stopped-trading-compute", action="store_true")
    parser.add_argument("--allow-closing-owner-session", action="store_true")
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be positive")
    if args.limit_pairs is not None and not 1 <= args.limit_pairs <= args.pairs:
        parser.error("--limit-pairs must lie in [1, --pairs]")
    if args.limit_pairs is not None and not args.simulate_only:
        parser.error("--limit-pairs requires --simulate-only")

    os.environ[structured.COMPUTE_TARGET_ENV] = args.compute_target
    if args.allow_stopped_trading_compute:
        os.environ[structured.STOPPED_TRADING_AUTH_ENV] = "1"
    if args.allow_closing_owner_session:
        os.environ[structured.AZURE_CLOSING_OWNER_AUTH_ENV] = "1"
    initial_gate = structured.compute_gate(args.compute_state)
    priority = structured.set_below_normal_priority()
    revision = structured.git_revision(script=Path(__file__))
    structured.require_clean_tracked_revision(revision)

    models, model_audit = prepare_models()
    contig = make_contig()
    jobs = make_jobs(args.pairs, args.seed_base)
    config = configuration(args.pairs, args.seed_base, jobs, model_audit)
    config_sha256 = hashlib.sha256(_canonical_json(config)).hexdigest()
    requested_pairs = args.pairs if args.limit_pairs is None else args.limit_pairs
    requested_jobs = [job for job in jobs if job.family_index < requested_pairs]
    checkpoint = args.cache_dir / "stdpopsim_neanderthal_features.npz"

    with structured.SingleWriterLease(args.cache_dir, ".stdpopsim_neanderthal.lock"):
        records = load_checkpoint(checkpoint, config_sha256, jobs)
        completed = {str(record["job_id"]) for record in records}
        for index, job in enumerate(requested_jobs, start=1):
            if job.job_id in completed:
                continue
            structured.compute_gate(args.compute_state)
            current = simulate_job(job, models, contig, compute_state=args.compute_state)
            records.append(current)
            completed.add(job.job_id)
            save_checkpoint(checkpoint, records, config_sha256)
            print(
                f"[{index}/{len(requested_jobs)}] {job.job_id}: "
                f"{current['simulation_audit']['globally_polymorphic_loci']} loci, "
                f"{current['elapsed_seconds']:.2f}s",
                flush=True,
            )
        selected_ids = {job.job_id for job in requested_jobs}
        selected = [record for record in records if record["job_id"] in selected_ids]
        if len(selected) != len(requested_jobs):
            raise RuntimeError("checkpoint does not contain every requested benchmark job")

    if args.simulate_only:
        final_revision = structured.git_revision(script=Path(__file__))
        structured.require_revision_unchanged(revision, final_revision)
        print(json.dumps({
            "checkpoint": checkpoint_audit(checkpoint, selected, config_sha256),
            "configuration_sha256": config_sha256,
            "source_commit": final_revision["commit"],
            "requested_pairs": requested_pairs,
        }, indent=2, allow_nan=False))
        return 0

    if len(selected) != len(jobs):
        raise RuntimeError("full analysis requires every configured pulse/control job")
    result_lock = structured.SingleWriterLease(
        args.result_dir, ".stdpopsim_neanderthal_result.lock"
    ).acquire()
    pre_analysis_gate = structured.compute_gate(args.compute_state)
    analysis = analyze_records(
        selected,
        args.canonical_root,
        compute_state=args.compute_state,
    )
    final_revision = structured.git_revision(script=Path(__file__))
    structured.require_revision_unchanged(revision, final_revision)
    runtime = structured.runtime_audit(priority)
    runtime["packages"].update({
        name: importlib_metadata.version(name)
        for name in ("stdpopsim", "msprime", "tskit", "padze")
    })
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "known_truth_single_direction_simulation_transfer_not_external_three_class_accuracy",
        "git": revision,
        "final_source_recheck": final_revision,
        "initial_compute_gate": initial_gate,
        "pre_analysis_compute_gate": pre_analysis_gate,
        "runtime": runtime,
        "configuration": config,
        "configuration_sha256": config_sha256,
        "checkpoint": checkpoint_audit(checkpoint, selected, config_sha256),
        "analysis": analysis,
        "interpretation_guardrail": (
            "The published model supplies known forward NEA->CEU class-C truth and a matched "
            "no-pulse control. It tests transfer across demography, but only one direction; it "
            "does not validate natural samples or estimate balanced A/B/C accuracy."
        ),
    }
    output = args.result_dir / "results.json"
    output_audit = structured.write_json_atomic(output, result, indent=2)
    print(json.dumps({
        "output": output_audit,
        "checkpoint": result["checkpoint"],
        "pulse_class_C_recall": analysis["direction_head"]["pulse_class_C_recall"],
        "gate_roc_auc": analysis["appreciable_gate"]["matched_pulse_control_roc_auc"],
    }, indent=2, allow_nan=False))
    result_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
