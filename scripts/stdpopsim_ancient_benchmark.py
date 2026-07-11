#!/usr/bin/env python3
"""Run five focal-event transfer panels from two official stdpopsim models.

The bank uses three AncientEurope_4A21 population-formation events and two
AncientEurasia_9K19 pulses.  Each panel freezes a P1/P2/P3 order and therefore
has known B or C direction.  Its control changes only the focal event: Europe
renormalizes the retained admixture source to one, while Eurasia sets the focal
MassMigration proportion to zero.  Controls are focal-component-absent, not
globally no-admixture models.

Every job simulates one independent 1 Mb contig with 100 diploid individuals
per selected population at the catalog default sampling times, then computes
the full PADZE g=2..199 curve.  Positive/control families share a panel and
replicate index but use independent ancestry and mutation seeds.  The first
positive/control realization for every panel is retained as a tree sequence
for forensic reconstruction.  This is a synthetic transfer benchmark, not an
ancient-DNA sample-size reproduction or natural-data accuracy estimate.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import stdpopsim_neanderthal_benchmark as stdbench
from scripts import structured_transfer_pilot as structured


SCHEMA_VERSION = "dnnaic-stdpopsim-ancient-focal-benchmark-v1"
CHECKPOINT_SCHEMA = "dnnaic-stdpopsim-ancient-focal-checkpoint-v1"
PINNED_STDPOPSIM_VERSION = "0.3.0"
PINNED_MSPRIME_VERSION = "1.4.2"
PINNED_TSKIT_VERSION = "1.0.3"
HOMSAP_MODEL_SOURCE_SHA256 = "ca1adb03f251b7fc293323ef9fe4e77ec9e705a9ad38c21382946bed7c791e1c"
SPECIES_ID = "HomSap"
CONDITIONS = ("positive", "control")
REPRESENTATIONS = (
    "raw_all",
    "raw_mean_variance",
    "orbit_composition_mean_variance",
)
INDIVIDUALS_PER_POPULATION = 100
GENE_COPIES_PER_POPULATION = 200
SEQUENCE_LENGTH = 1_000_000
RECOMBINATION_RATE = 1.78e-8
DEFAULT_PAIRS_PER_PANEL = 30
DEFAULT_SEED_BASE = 71_100_000
DEFAULT_CACHE = (
    Path.home()
    / "Documents"
    / "Codex"
    / "2026-07-10"
    / "dnnaic-datasets2-data"
    / "stdpopsim_ancient_2026_07_11"
)
DEFAULT_RESULTS = REPO / "results" / "stdpopsim_ancient_benchmark_2026_07_11"


@dataclass(frozen=True)
class PanelSpec:
    panel_id: str
    model_id: str
    populations: tuple[str, str, str]
    direction_truth: str
    forward_event: str
    event_type: str
    event_time: float
    derived: str | None = None
    ancestral: tuple[str, ...] | None = None
    positive_proportions: tuple[float, ...] | None = None
    control_proportions: tuple[float, ...] | None = None
    source: str | None = None
    dest: str | None = None
    positive_proportion: float | None = None


PANELS = (
    PanelSpec(
        panel_id="europe_whg_to_neo",
        model_id="AncientEurope_4A21",
        populations=("EHG", "WHG", "NEO"),
        direction_truth="B",
        forward_event="WHG->NEO",
        event_type="Admixture",
        event_time=200.0,
        derived="NEO",
        ancestral=("WHG", "ANA"),
        positive_proportions=(0.25, 0.75),
        control_proportions=(0.0, 1.0),
    ),
    PanelSpec(
        panel_id="europe_chg_to_yam",
        model_id="AncientEurope_4A21",
        populations=("WHG", "YAM", "CHG"),
        direction_truth="C",
        forward_event="CHG->YAM",
        event_type="Admixture",
        event_time=180.0,
        derived="YAM",
        ancestral=("EHG", "CHG"),
        positive_proportions=(0.5, 0.5),
        control_proportions=(1.0, 0.0),
    ),
    PanelSpec(
        panel_id="europe_neo_to_bronze",
        model_id="AncientEurope_4A21",
        populations=("ANA", "NEO", "Bronze"),
        direction_truth="B",
        forward_event="NEO->Bronze",
        event_type="Admixture",
        event_time=140.0,
        derived="Bronze",
        ancestral=("YAM", "NEO"),
        positive_proportions=(0.5, 0.5),
        control_proportions=(1.0, 0.0),
    ),
    PanelSpec(
        panel_id="eurasia_neanderthal_to_han",
        model_id="AncientEurasia_9K19",
        populations=("Mbuti", "Han", "Neanderthal"),
        direction_truth="C",
        forward_event="Neanderthal->Han lineage",
        event_type="MassMigration",
        event_time=2272.0,
        source="Loschbour",
        dest="Neanderthal",
        positive_proportion=0.0296,
    ),
    PanelSpec(
        panel_id="eurasia_whg_to_sardinian",
        model_id="AncientEurasia_9K19",
        populations=("LBK", "Sardinian", "Loschbour"),
        direction_truth="C",
        forward_event="Loschbour/WHG->Sardinian",
        event_type="MassMigration",
        event_time=49.2,
        source="Sardinian",
        dest="Loschbour",
        positive_proportion=0.0317,
    ),
)


@dataclass(frozen=True)
class AncientJob:
    panel_index: int
    panel_id: str
    replicate_index: int
    family_id: str
    condition: str
    direction_truth: str
    job_id: str
    engine_seed: int


def panel_by_id(panel_id: str) -> PanelSpec:
    matches = [panel for panel in PANELS if panel.panel_id == panel_id]
    if len(matches) != 1:
        raise ValueError(f"unknown ancient panel {panel_id!r}")
    return matches[0]


def make_jobs(pairs_per_panel: int, seed_base: int) -> list[AncientJob]:
    if pairs_per_panel < 1:
        raise ValueError("pairs per panel must be positive")
    total = len(PANELS) * pairs_per_panel * len(CONDITIONS)
    if not 0 < seed_base < 2**31 - total - 1:
        raise ValueError("seed base does not leave room for all unique jobs")
    jobs = []
    offset = 0
    for panel_index, panel in enumerate(PANELS):
        for replicate_index in range(pairs_per_panel):
            family_id = f"{panel.panel_id}-family-{replicate_index:04d}"
            for condition in CONDITIONS:
                jobs.append(AncientJob(
                    panel_index=panel_index,
                    panel_id=panel.panel_id,
                    replicate_index=replicate_index,
                    family_id=family_id,
                    condition=condition,
                    direction_truth=panel.direction_truth,
                    job_id=f"{family_id}__{condition}",
                    engine_seed=seed_base + offset,
                ))
                offset += 1
    if len({job.job_id for job in jobs}) != len(jobs):
        raise AssertionError("ancient benchmark job IDs collide")
    if len({job.engine_seed for job in jobs}) != len(jobs):
        raise AssertionError("ancient benchmark engine seeds collide")
    return jobs


def _canonical_json(value) -> bytes:
    return stdbench._canonical_json(value)


def _event_signature(event) -> dict:
    return stdbench._event_signature(event)


def _model_signature(model) -> dict:
    return {
        "populations": [
            stdbench._population_signature(population)
            for population in model.model.populations
        ],
        "events": [_event_signature(event) for event in model.model.events],
        "generation_time": float(model.generation_time),
        "mutation_rate": float(model.mutation_rate),
    }


def _population_name(demography, reference) -> str:
    if isinstance(reference, str):
        return reference
    return str(demography.populations[int(reference)].name)


def _event_matches(demography, event, panel: PanelSpec) -> bool:
    if type(event).__name__ != panel.event_type or not math.isclose(
        float(event.time), panel.event_time, rel_tol=0, abs_tol=1e-12
    ):
        return False
    if panel.event_type == "Admixture":
        return (
            str(event.derived) == panel.derived
            and tuple(map(str, event.ancestral)) == panel.ancestral
            and np.allclose(
                np.asarray(event.proportions, dtype=float),
                np.asarray(panel.positive_proportions, dtype=float),
                rtol=0,
                atol=1e-15,
            )
        )
    return (
        _population_name(demography, event.source) == panel.source
        and _population_name(demography, event.dest) == panel.dest
        and math.isclose(
            float(event.proportion), float(panel.positive_proportion),
            rel_tol=0, abs_tol=1e-15,
        )
    )


def _locate_focal_event(model, panel: PanelSpec) -> tuple[int, object]:
    matches = [
        (index, event)
        for index, event in enumerate(model.model.events)
        if _event_matches(model.model, event, panel)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"{panel.panel_id}: expected one exact focal event, observed {len(matches)}"
        )
    return matches[0]


def prepare_models() -> tuple[dict[str, dict[str, object]], dict]:
    import stdpopsim

    versions = {
        "stdpopsim": importlib_metadata.version("stdpopsim"),
        "msprime": importlib_metadata.version("msprime"),
        "tskit": importlib_metadata.version("tskit"),
    }
    expected = {
        "stdpopsim": PINNED_STDPOPSIM_VERSION,
        "msprime": PINNED_MSPRIME_VERSION,
        "tskit": PINNED_TSKIT_VERSION,
    }
    if versions != expected:
        raise RuntimeError(f"ancient stdpopsim package contract changed: {versions}")
    source_module = importlib.import_module("stdpopsim.catalog.HomSap.demographic_models")
    source_path = Path(source_module.__file__).resolve()
    source_hash = structured.sha256_file(source_path)
    if source_hash != HOMSAP_MODEL_SOURCE_SHA256:
        raise RuntimeError(f"HomSap demographic-model source changed: {source_hash}")

    species = stdpopsim.get_species(SPECIES_ID)
    models: dict[str, dict[str, object]] = {}
    panel_audits = {}
    expected_model_contract = {
        "AncientEurope_4A21": (29.0, 1.25e-8),
        "AncientEurasia_9K19": (25.0, 1.22e-8),
    }
    for panel in PANELS:
        positive = species.get_demographic_model(panel.model_id)
        generation, mutation = expected_model_contract[panel.model_id]
        if positive.id != panel.model_id:
            raise AssertionError(f"{panel.panel_id}: catalog model ID changed")
        if not math.isclose(float(positive.generation_time), generation, rel_tol=0, abs_tol=0):
            raise AssertionError(f"{panel.panel_id}: generation time changed")
        if not math.isclose(float(positive.mutation_rate), mutation, rel_tol=0, abs_tol=0):
            raise AssertionError(f"{panel.panel_id}: mutation rate changed")
        positive.model.validate()
        population_names = tuple(population.name for population in positive.model.populations)
        if any(name not in population_names for name in panel.populations):
            raise AssertionError(f"{panel.panel_id}: selected population disappeared")
        positive_index, positive_event = _locate_focal_event(positive, panel)
        control = copy.deepcopy(positive)
        control_event = control.model.events[positive_index]
        if not _event_matches(control.model, control_event, panel):
            raise AssertionError(f"{panel.panel_id}: deep-copied event changed before ablation")
        if panel.event_type == "Admixture":
            control_event.proportions = list(panel.control_proportions)
        else:
            control_event.proportion = 0.0
        control.model.validate()

        positive_signature = _model_signature(positive)
        control_signature = _model_signature(control)
        positive_events = positive_signature["events"]
        control_events = control_signature["events"]
        changed_indices = [
            index
            for index, (left, right) in enumerate(zip(positive_events, control_events))
            if left != right
        ]
        if (
            positive_signature["populations"] != control_signature["populations"]
            or len(positive_events) != len(control_events)
            or changed_indices != [positive_index]
        ):
            raise AssertionError(
                f"{panel.panel_id}: control differs beyond the exact focal event"
            )
        allowed_field = "proportions" if panel.event_type == "Admixture" else "proportion"
        normalized_positive_event = dict(positive_events[positive_index])
        normalized_control_event = dict(control_events[positive_index])
        normalized_positive_event.pop(allowed_field)
        normalized_control_event.pop(allowed_field)
        if normalized_positive_event != normalized_control_event:
            raise AssertionError(f"{panel.panel_id}: focal event changed beyond {allowed_field}")

        sample_times = {
            name: float(positive.model[name].default_sampling_time)
            for name in panel.populations
        }
        if any(not math.isfinite(value) or value < 0 for value in sample_times.values()):
            raise AssertionError(f"{panel.panel_id}: selected population is not sampleable")
        models[panel.panel_id] = {"positive": positive, "control": control}
        panel_audits[panel.panel_id] = {
            "specification": asdict(panel),
            "population_order": list(panel.populations),
            "sample_times_generations": sample_times,
            "generation_time_years": float(positive.generation_time),
            "mutation_rate": float(positive.mutation_rate),
            "focal_event_index": int(positive_index),
            "positive_event": positive_events[positive_index],
            "control_event": control_events[positive_index],
            "positive_model_signature_sha256": hashlib.sha256(
                _canonical_json(positive_signature)
            ).hexdigest(),
            "control_model_signature_sha256": hashlib.sha256(
                _canonical_json(control_signature)
            ).hexdigest(),
            "control_differs_only_at_focal_event": True,
        }
    return models, {
        "versions": versions,
        "species_id": SPECIES_ID,
        "HomSap_demographic_models_source": str(source_path),
        "HomSap_demographic_models_source_sha256": source_hash,
        "panels": panel_audits,
        "excluded_primary_panel": (
            "BasalEurasian->LBK is excluded because the catalog ghost population has "
            "default_sampling_time=None"
        ),
        "engine_guardrail": (
            "msprime 1.4.2 dry_run aborts in native code for the unmodified "
            "AncientEurope_4A21 catalog model (lib/msprime.c line 6148), while actual "
            "simulations succeed. This runner never invokes dry_run; the pinned test suite "
            "uses tiny actual simulations for every panel and condition."
        ),
    }


def make_contig(model):
    import stdpopsim

    contig = stdpopsim.Contig.basic_contig(
        length=SEQUENCE_LENGTH,
        mutation_rate=float(model.mutation_rate),
        recombination_rate=RECOMBINATION_RATE,
        ploidy=2,
    )
    if int(contig.length) != SEQUENCE_LENGTH or int(contig.ploidy) != 2:
        raise AssertionError("ancient stdpopsim contig length/ploidy changed")
    return contig


def _dump_tree_sequence_atomic(tree_sequence, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.part.{os.getpid()}.{time.time_ns()}")
    try:
        tree_sequence.dump(str(temporary))
        with temporary.open("rb") as handle:
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
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": structured.sha256_file(path),
    }


def tree_sequence_to_curve(
    tree_sequence,
    panel: PanelSpec,
    population_ids: dict[str, int],
    expected_sample_times: dict[str, float],
    *,
    compute_state: Path | None = None,
) -> tuple[np.ndarray, dict]:
    from padze import LociData, Metadata, compute_features

    samples = tree_sequence.samples()
    sample_population = tree_sequence.tables.nodes.population[samples]
    sample_time = tree_sequence.tables.nodes.time[samples]
    masks = [sample_population == population_ids[name] for name in panel.populations]
    gene_copy_counts = [int(mask.sum()) for mask in masks]
    if gene_copy_counts != [GENE_COPIES_PER_POPULATION] * 3:
        raise AssertionError(
            f"{panel.panel_id}: observed gene-copy counts {gene_copy_counts}, expected 200 each"
        )
    observed_times = {}
    for name, mask in zip(panel.populations, masks):
        unique = np.unique(sample_time[mask])
        if unique.shape != (1,) or not math.isclose(
            float(unique[0]), expected_sample_times[name], rel_tol=0, abs_tol=1e-12
        ):
            raise AssertionError(f"{panel.panel_id}: realized sample time changed for {name}")
        observed_times[name] = float(unique[0])
    if int(tree_sequence.num_samples) != 3 * GENE_COPIES_PER_POPULATION:
        raise AssertionError("ancient tree sequence total sample count changed")
    if int(tree_sequence.num_individuals) != 3 * INDIVIDUALS_PER_POPULATION:
        raise AssertionError("ancient tree sequence diploid individual count changed")

    count_matrices = []
    sample_sizes = []
    locus_ids = []
    count_hash = hashlib.sha256()
    multiallelic = 0
    for variant in tree_sequence.variants(copy=False):
        genotype = np.asarray(variant.genotypes)
        if genotype.shape != (3 * GENE_COPIES_PER_POPULATION,) or np.any(genotype < 0):
            raise AssertionError("ancient simulated genotype vector is invalid")
        allele_count = int(genotype.max()) + 1
        if allele_count < 1:
            continue
        if allele_count > 4:
            raise AssertionError("ancient JC69 simulation emitted more than four alleles")
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
        if not np.array_equal(
            counts.sum(axis=1), np.full(3, GENE_COPIES_PER_POPULATION)
        ):
            raise AssertionError("ancient allele counts do not conserve sample sizes")
        locus_id = f"site-{variant.site.id}@{float(variant.site.position):.17g}"
        count_matrices.append(counts)
        sample_sizes.append(counts.sum(axis=1))
        locus_ids.append(locus_id)
        count_hash.update(np.asarray(counts.shape, dtype="<i8").tobytes())
        count_hash.update(counts.astype("<i8", copy=False).tobytes())
        count_hash.update(locus_id.encode("utf-8"))
        count_hash.update(b"\0")
    if len(count_matrices) < 2:
        raise RuntimeError(f"{panel.panel_id}: too few polymorphic sites for PADZE")
    sizes = np.vstack(sample_sizes).astype(np.int64, copy=False)
    loci = LociData(
        populations=["P1", "P2", "P3"],
        count_matrices=count_matrices,
        sample_sizes=sizes,
        locus_ids=locus_ids,
        metadata=Metadata(
            source=(
                f"stdpopsim {PINNED_STDPOPSIM_VERSION} {SPECIES_ID}/"
                f"{panel.model_id} panel {panel.panel_id}"
            ),
            populations=["P1", "P2", "P3"],
            sample_ids={name: [] for name in ("P1", "P2", "P3")},
            ploidy={name: 2 for name in ("P1", "P2", "P3")},
            n_loci_read=int(tree_sequence.num_sites),
            n_loci_kept=len(count_matrices),
            filters_applied=[
                "globally polymorphic across the frozen P1/P2/P3 panel; all alleles retained"
            ],
            missing_fraction=0.0,
        ),
    )
    if compute_state is not None:
        structured.compute_gate(compute_state)
    table = compute_features(
        loci,
        depths=stdbench.FULL_DEPTHS,
        pihat_sizes=(2,),
        moments=stdbench.MOMENTS,
        bias_corrected=True,
    )
    matrix, columns = table.to_frame()
    column_index = {column: index for index, column in enumerate(columns)}
    try:
        curve = matrix[:, [column_index[column] for column in stdbench.CURVE_COLUMNS]].astype(
            np.float64
        )
    except KeyError as exc:
        raise RuntimeError(f"ancient PADZE feature contract changed: {exc}") from exc
    if curve.shape != (198, 28) or not np.isfinite(curve).all():
        raise AssertionError("ancient PADZE curve is invalid")
    if not np.array_equal(curve[:, 0], stdbench.FULL_DEPTHS):
        raise AssertionError("ancient PADZE depth grid changed")
    return curve, {
        "num_trees": int(tree_sequence.num_trees),
        "num_sites": int(tree_sequence.num_sites),
        "num_mutations": int(tree_sequence.num_mutations),
        "num_individuals": int(tree_sequence.num_individuals),
        "num_sample_nodes": int(tree_sequence.num_samples),
        "gene_copies_by_population": dict(zip(panel.populations, gene_copy_counts)),
        "sample_times_generations": observed_times,
        "globally_polymorphic_loci": len(count_matrices),
        "multiallelic_globally_polymorphic_loci": int(multiallelic),
        "ordered_count_ledger_sha256": count_hash.hexdigest(),
        "curve_sha256_float64": stdbench._sha256_array(
            curve.astype("<f8", copy=False)
        ),
    }


def simulate_job(
    job: AncientJob,
    models: dict[str, dict[str, object]],
    model_audit: dict,
    cache_dir: Path,
    *,
    compute_state: Path | None = None,
) -> dict:
    import stdpopsim

    panel = panel_by_id(job.panel_id)
    model = models[job.panel_id][job.condition]
    if compute_state is not None:
        structured.compute_gate(compute_state)
    started = time.perf_counter()
    tree_sequence = stdpopsim.get_engine("msprime").simulate(
        demographic_model=model,
        contig=make_contig(model),
        samples={name: INDIVIDUALS_PER_POPULATION for name in panel.populations},
        seed=job.engine_seed,
    )
    raw_tree = None
    if job.replicate_index == 0:
        raw_tree = _dump_tree_sequence_atomic(
            tree_sequence,
            cache_dir / "raw_audit" / f"{job.job_id}.trees",
        )
    population_ids = {
        name: int(model.model[name].id)
        for name in panel.populations
    }
    curve, simulation_audit = tree_sequence_to_curve(
        tree_sequence,
        panel,
        population_ids,
        model_audit["panels"][panel.panel_id]["sample_times_generations"],
        compute_state=compute_state,
    )
    curve32 = curve.astype(np.float32)
    ancestry_seed, mutation_seed = stdbench._engine_seeds(job.engine_seed)
    return {
        **asdict(job),
        "engine_derived_ancestry_seed": ancestry_seed,
        "engine_derived_mutation_seed": mutation_seed,
        "elapsed_seconds": float(time.perf_counter() - started),
        "raw_tree_sequence": raw_tree,
        "simulation_audit": simulation_audit,
        "curve_sha256_float32": stdbench._sha256_array(
            curve32.astype("<f4", copy=False)
        ),
        "curve": curve32,
    }


def record_key(record: dict) -> tuple[int, int, int]:
    return (
        int(record["panel_index"]),
        int(record["replicate_index"]),
        CONDITIONS.index(str(record["condition"])),
    )


def save_checkpoint(path: Path, records: Sequence[dict], config_sha256: str) -> None:
    records = sorted(records, key=record_key)
    if len({record["job_id"] for record in records}) != len(records):
        raise RuntimeError("refusing to save duplicate ancient benchmark jobs")
    curves = np.stack([np.asarray(record["curve"], dtype=np.float32) for record in records])
    if curves.shape != (len(records), 198, 28) or not np.isfinite(curves).all():
        raise RuntimeError("refusing to save invalid ancient benchmark curves")
    metadata = []
    for record, curve in zip(records, curves):
        current = {key: value for key, value in record.items() if key != "curve"}
        if stdbench._sha256_array(curve.astype("<f4", copy=False)) != current[
            "curve_sha256_float32"
        ]:
            raise RuntimeError("ancient curve hash changed before checkpoint save")
        metadata.append(current)
    stdbench._atomic_npz(
        path,
        schema=np.asarray([CHECKPOINT_SCHEMA]),
        config_sha256=np.asarray([config_sha256]),
        metadata_json=np.asarray([_canonical_json(metadata).decode("ascii")]),
        curves=curves,
    )


def load_checkpoint(
    path: Path,
    config_sha256: str,
    jobs: Sequence[AncientJob],
) -> list[dict]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema", "config_sha256", "metadata_json", "curves"}
        if set(archive.files) != required:
            raise RuntimeError("ancient checkpoint member set changed")
        if archive["schema"].tolist() != [CHECKPOINT_SCHEMA]:
            raise RuntimeError("ancient checkpoint schema changed")
        if archive["config_sha256"].tolist() != [config_sha256]:
            raise RuntimeError("ancient checkpoint configuration changed")
        metadata = json.loads(str(archive["metadata_json"][0]))
        curves = np.asarray(archive["curves"], dtype=np.float32)
    manifest = {job.job_id: job for job in jobs}
    if len(metadata) != len(curves):
        raise RuntimeError("ancient checkpoint metadata/curve cardinality changed")
    records = []
    seen = set()
    for current, curve in zip(metadata, curves):
        job_id = current.get("job_id")
        if job_id not in manifest or job_id in seen:
            raise RuntimeError("ancient checkpoint has an unknown or duplicate job")
        seen.add(job_id)
        if any(current.get(key) != value for key, value in asdict(manifest[job_id]).items()):
            raise RuntimeError(f"ancient checkpoint manifest changed for {job_id}")
        if curve.shape != (198, 28) or not np.isfinite(curve).all():
            raise RuntimeError(f"ancient checkpoint curve is invalid for {job_id}")
        if stdbench._sha256_array(curve.astype("<f4", copy=False)) != current[
            "curve_sha256_float32"
        ]:
            raise RuntimeError(f"ancient checkpoint curve hash changed for {job_id}")
        raw = current.get("raw_tree_sequence")
        if raw is not None:
            raw_path = Path(raw["path"])
            if (
                not raw_path.is_file()
                or raw_path.stat().st_size != raw["bytes"]
                or structured.sha256_file(raw_path) != raw["sha256"]
            ):
                raise RuntimeError(f"ancient raw tree artifact changed for {job_id}")
        records.append({**current, "curve": curve})
    return sorted(records, key=record_key)


def record_selection_audit(records: Sequence[dict]) -> dict:
    families = {}
    panels = {}
    for record in records:
        families.setdefault(record["family_id"], set()).add(record["condition"])
        panels.setdefault(record["panel_id"], 0)
        panels[record["panel_id"]] += 1
    return {
        "records": len(records),
        "complete_positive_control_families": int(
            sum(conditions == set(CONDITIONS) for conditions in families.values())
        ),
        "records_by_panel": {key: int(value) for key, value in sorted(panels.items())},
        "record_curve_hash_ledger_sha256": hashlib.sha256(_canonical_json([
            [record["job_id"], record["curve_sha256_float32"]]
            for record in sorted(records, key=record_key)
        ])).hexdigest(),
    }


def checkpoint_audit(path: Path, records: Sequence[dict], config_sha256: str) -> dict:
    records = sorted(records, key=record_key)
    with np.load(path, allow_pickle=False) as archive:
        if archive["schema"].tolist() != [CHECKPOINT_SCHEMA]:
            raise RuntimeError("ancient checkpoint audit found a changed schema")
        if archive["config_sha256"].tolist() != [config_sha256]:
            raise RuntimeError("ancient checkpoint audit found a changed configuration")
        stored_metadata = json.loads(str(archive["metadata_json"][0]))
        stored_curve_shape = list(archive["curves"].shape)
    expected_metadata = [
        {key: value for key, value in record.items() if key != "curve"}
        for record in records
    ]
    if stored_metadata != expected_metadata:
        raise RuntimeError("ancient checkpoint bytes and audit records differ")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": structured.sha256_file(path),
        "schema_version": CHECKPOINT_SCHEMA,
        "configuration_sha256": config_sha256,
        "stored_curve_shape": stored_curve_shape,
        **record_selection_audit(records),
    }


def _confusion_payload(truth: np.ndarray, prediction: np.ndarray) -> dict:
    labels = ("B", "C", "A")
    return {
        "labels": list(labels),
        "matrix": confusion_matrix(truth, prediction, labels=list(labels)).astype(int).tolist(),
        "rows_are_truth_columns_are_predictions": True,
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
        families.setdefault(record["family_id"], set()).add(record["condition"])
    if len(families) != len(PANELS) * DEFAULT_PAIRS_PER_PANEL:
        # The caller also checks the configured manifest; this catches accidental
        # analysis with a nondefault or selected-only bank.
        raise RuntimeError("ancient analysis requires the full default 150-family bank")
    if any(conditions != set(CONDITIONS) for conditions in families.values()):
        raise RuntimeError("ancient analysis requires complete positive/control families")
    curves = np.stack([record["curve"] for record in records]).astype(float)
    conditions = np.asarray([record["condition"] for record in records])
    positive_rows = conditions == "positive"
    control_rows = conditions == "control"
    truth = np.asarray([record["direction_truth"] for record in records])
    panel_ids = np.asarray([record["panel_id"] for record in records])

    if compute_state is not None:
        structured.compute_gate(compute_state)
    canonical = structured.load_canonical(canonical_root, max_depth=199)
    canonical_table = np.asarray(canonical["table"], dtype=float)
    canonical_labels = np.asarray(canonical["labels"])
    canonical_rates = np.asarray(canonical["rates"], dtype=float)
    canonical_positive = np.isin(canonical_labels, ["A", "B", "C"])
    primary_rows = len(stdbench.PRIMARY_DEPTHS)
    representations = {}
    prediction_cache = {}
    for name in REPRESENTATIONS:
        train = structured.representation_features(
            canonical_table[canonical_positive, :primary_rows], name
        )
        external = structured.representation_features(curves[:, :primary_rows], name)
        scaler, model = structured._fit_model(
            train, canonical_labels[canonical_positive], C=1.0
        )
        z = scaler.transform(external)
        probability = model.predict_proba(z)
        class_index = {str(label): index for index, label in enumerate(model.classes_)}
        prediction = model.classes_[np.argmax(probability, axis=1)].astype(str)
        prediction_cache[name] = (prediction, probability, class_index, z)
        positive_truth = truth[positive_rows]
        positive_prediction = prediction[positive_rows]
        recalls = {
            label: float(np.mean(positive_prediction[positive_truth == label] == label))
            for label in ("B", "C")
        }
        per_panel = {}
        panel_accuracy = []
        for panel in PANELS:
            use = positive_rows & (panel_ids == panel.panel_id)
            correct = prediction[use] == panel.direction_truth
            panel_accuracy.append(float(np.mean(correct)))
            per_panel[panel.panel_id] = {
                "truth": panel.direction_truth,
                "forward_event": panel.forward_event,
                "recall": stdbench._wilson(correct),
                "predicted_class_counts": {
                    str(label): int(count)
                    for label, count in zip(*np.unique(prediction[use], return_counts=True))
                },
                "scaler_rms_z_median": float(
                    np.median(np.sqrt(np.mean(z[use] ** 2, axis=1)))
                ),
            }
        representations[name] = {
            "status": "target-blind fixed canonical C=1; raw_all is primary",
            "feature_dimension": int(train.shape[1]),
            "B_C_balanced_accuracy": float((recalls["B"] + recalls["C"]) / 2),
            "B_recall": stdbench._wilson(
                positive_prediction[positive_truth == "B"] == "B"
            ),
            "C_recall": stdbench._wilson(
                positive_prediction[positive_truth == "C"] == "C"
            ),
            "equal_panel_macro_accuracy": float(np.mean(panel_accuracy)),
            "confusion": _confusion_payload(positive_truth, positive_prediction),
            "per_panel": per_panel,
            "control_forced_call_counts_diagnostic_only": {
                str(label): int(count)
                for label, count in zip(*np.unique(prediction[control_rows], return_counts=True))
            },
            "scaler_rms_z_median": float(np.median(np.sqrt(np.mean(z**2, axis=1)))),
            "scaler_rms_z_p95": float(np.quantile(np.sqrt(np.mean(z**2, axis=1)), 0.95)),
            "scaler_max_abs_z_p95": float(
                np.quantile(np.max(np.abs(z), axis=1), 0.95)
            ),
            "model": stdbench._model_payload(
                scaler,
                model,
                feature_columns=structured.representation_columns(name),
            ),
            "guardrail": (
                "B and C are the only positive truths; a constant B or C predictor has 0.5 "
                "balanced accuracy. Wilson intervals are Monte Carlo summaries within fixed "
                "catalog panels, not across-demography generalization intervals."
            ),
        }

    gate_train, gate_contract = stdbench._gate_features(canonical_table)
    gate_external, external_gate_contract = stdbench._gate_features(curves)
    if gate_contract != external_gate_contract:
        raise AssertionError("ancient/canonical gate feature contracts differ")
    gate_target = (
        canonical_positive & (canonical_rates >= structured.APPRECIABLE)
    ).astype(int)
    gate_scaler, gate_model = structured._fit_model(gate_train, gate_target, C=1.0)
    gate_z = gate_scaler.transform(gate_external)
    gate_positive_index = int(np.flatnonzero(gate_model.classes_ == 1)[0])
    gate_score = gate_model.predict_proba(gate_z)[:, gate_positive_index]
    gate_call = gate_score >= 0.5
    gate_truth = positive_rows.astype(int)
    per_panel_gate = {}
    panel_aucs = []
    for panel in PANELS:
        use = panel_ids == panel.panel_id
        auc = float(roc_auc_score(gate_truth[use], gate_score[use]))
        panel_aucs.append(auc)
        per_panel_gate[panel.panel_id] = {
            "roc_auc": auc,
            "positive_sensitivity_at_0_5": stdbench._wilson(
                gate_call[use & positive_rows]
            ),
            "focal_absent_negative_call_fraction_at_0_5": stdbench._wilson(
                ~gate_call[use & control_rows]
            ),
            "positive_score": stdbench._distribution_summary(gate_score[use & positive_rows]),
            "control_score": stdbench._distribution_summary(gate_score[use & control_rows]),
        }

    primary_prediction, primary_probability, primary_class_index, primary_z = prediction_cache[
        "raw_all"
    ]
    prediction_ledger = []
    for index, record in enumerate(records):
        is_positive = bool(positive_rows[index])
        prediction_ledger.append({
            "job_id": record["job_id"],
            "panel_id": record["panel_id"],
            "replicate_index": int(record["replicate_index"]),
            "condition": record["condition"],
            "direction_truth": record["direction_truth"] if is_positive else None,
            "included_in_direction_accuracy": is_positive,
            "raw_all_prediction": str(primary_prediction[index]),
            "raw_all_correct": (
                bool(primary_prediction[index] == record["direction_truth"])
                if is_positive else None
            ),
            "raw_all_probability": {
                label: float(primary_probability[index, primary_class_index[label]])
                for label in ("A", "B", "C")
            },
            "raw_all_scaler_rms_z": float(
                np.sqrt(np.mean(primary_z[index] ** 2))
            ),
            "raw_all_scaler_max_abs_z": float(np.max(np.abs(primary_z[index]))),
            "appreciable_gate_score": float(gate_score[index]),
            "appreciable_gate_call_at_0_5": bool(gate_call[index]),
        })

    simulation_summary = {}
    for panel in PANELS:
        simulation_summary[panel.panel_id] = {}
        for condition in CONDITIONS:
            current = [
                record for record in records
                if record["panel_id"] == panel.panel_id
                and record["condition"] == condition
            ]
            simulation_summary[panel.panel_id][condition] = {
                "globally_polymorphic_loci": stdbench._distribution_summary([
                    record["simulation_audit"]["globally_polymorphic_loci"]
                    for record in current
                ]),
                "elapsed_seconds": stdbench._distribution_summary([
                    record["elapsed_seconds"] for record in current
                ]),
            }
    return {
        "statistical_unit": "one independent 1 Mb ancestry/mutation realization",
        "independent_focal_event_systems": len(PANELS),
        "families": len(families),
        "records": len(records),
        "direction_accuracy_rows": int(positive_rows.sum()),
        "focal_absent_controls_excluded_from_direction_accuracy": int(control_rows.sum()),
        "representations": representations,
        "appreciable_gate": {
            "status": (
                "secondary frozen transfer diagnostic; canonical target is sustained migration "
                "rate >=2.5e-4, not catalog admixture proportion"
            ),
            "contract": gate_contract,
            "overall_positive_control_roc_auc": float(
                roc_auc_score(gate_truth, gate_score)
            ),
            "equal_panel_macro_roc_auc": float(np.mean(panel_aucs)),
            "positive_sensitivity_at_0_5": stdbench._wilson(gate_call[positive_rows]),
            "focal_absent_negative_call_fraction_at_0_5": stdbench._wilson(
                ~gate_call[control_rows]
            ),
            "per_panel": per_panel_gate,
            "model": stdbench._model_payload(gate_scaler, gate_model),
            "control_guardrail": (
                "Controls remove only one focal component and may retain other catalog "
                "admixture; their negative-call fraction is not global no-gene-flow specificity."
            ),
        },
        "simulation_summary": simulation_summary,
        "simulation_record_ledger": [
            {key: value for key, value in record.items() if key != "curve"}
            for record in records
        ],
        "prediction_ledger": prediction_ledger,
        "canonical_source_audit": canonical["audit"],
        "guardrail": (
            "This bank contains two B and three C focal systems but no A system. The 30 Monte "
            "Carlo replicates per panel do not create more than five independent demographic "
            "event systems. Ancient sampling times and 25-50% formation ancestries or 3% pulses "
            "are severe transfer shifts relative to continuous-migration training."
        ),
    }


def configuration(
    pairs_per_panel: int,
    seed_base: int,
    jobs: Sequence[AncientJob],
    model_audit: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "pairs_per_panel": int(pairs_per_panel),
        "seed_base": int(seed_base),
        "job_manifest": [asdict(job) for job in jobs],
        "model_audit": model_audit,
        "sequence": {
            "length_bp": SEQUENCE_LENGTH,
            "mutation_rate": "catalog model-specific",
            "recombination_rate": RECOMBINATION_RATE,
            "independent_contigs_per_job": 1,
        },
        "sampling": {
            "individuals_per_population": INDIVIDUALS_PER_POPULATION,
            "gene_copies_per_population": GENE_COPIES_PER_POPULATION,
            "times": "catalog default sampling time for every selected population",
            "ploidy": 2,
        },
        "padze": {
            "depths": stdbench.FULL_DEPTHS.tolist(),
            "moments": list(stdbench.MOMENTS),
            "pihat_sizes": [2],
            "bias_corrected": True,
            "site_filter": "globally polymorphic in frozen triplet; all alleles retained",
        },
        "evaluation": {
            "representations": list(REPRESENTATIONS),
            "primary": "raw_all",
            "primary_depths": stdbench.PRIMARY_DEPTHS.tolist(),
            "C": 1.0,
            "truth_counts_per_representation": {
                "B": 2 * pairs_per_panel,
                "C": 3 * pairs_per_panel,
            },
        },
        "raw_retention": "first positive/control tree sequence for every panel",
        "pairing_guardrail": (
            "families match panel and replicate index only; conditions use independent seeds "
            "and never share a genealogy"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-panel", type=int, default=DEFAULT_PAIRS_PER_PANEL)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--limit-replicates", type=int, default=None)
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--compute-state", type=Path, default=structured.DEFAULT_COMPUTE_STATE)
    parser.add_argument("--compute-target", choices=("local", "azure"), default="local")
    parser.add_argument("--allow-stopped-trading-compute", action="store_true")
    parser.add_argument("--allow-closing-owner-session", action="store_true")
    args = parser.parse_args()
    if args.pairs_per_panel != DEFAULT_PAIRS_PER_PANEL:
        parser.error(
            f"--pairs-per-panel is frozen at {DEFAULT_PAIRS_PER_PANEL} for full analysis"
        )
    if args.limit_replicates is not None and not 1 <= args.limit_replicates <= args.pairs_per_panel:
        parser.error("--limit-replicates must lie in [1, --pairs-per-panel]")
    if args.limit_replicates is not None and not args.simulate_only:
        parser.error("--limit-replicates requires --simulate-only")

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
    jobs = make_jobs(args.pairs_per_panel, args.seed_base)
    config = configuration(args.pairs_per_panel, args.seed_base, jobs, model_audit)
    config_sha256 = hashlib.sha256(_canonical_json(config)).hexdigest()
    requested_replicates = (
        args.pairs_per_panel
        if args.limit_replicates is None
        else args.limit_replicates
    )
    requested_jobs = [
        job for job in jobs if job.replicate_index < requested_replicates
    ]
    checkpoint = args.cache_dir / "stdpopsim_ancient_features.npz"
    with structured.SingleWriterLease(args.cache_dir, ".stdpopsim_ancient.lock"):
        records = load_checkpoint(checkpoint, config_sha256, jobs)
        completed = {record["job_id"] for record in records}
        for index, job in enumerate(requested_jobs, start=1):
            if job.job_id in completed:
                continue
            structured.compute_gate(args.compute_state)
            record = simulate_job(
                job,
                models,
                model_audit,
                args.cache_dir,
                compute_state=args.compute_state,
            )
            records.append(record)
            completed.add(job.job_id)
            save_checkpoint(checkpoint, records, config_sha256)
            print(
                f"[{index}/{len(requested_jobs)}] {job.job_id}: "
                f"{record['simulation_audit']['globally_polymorphic_loci']} sites, "
                f"{record['elapsed_seconds']:.2f}s",
                flush=True,
            )
        requested_ids = {job.job_id for job in requested_jobs}
        selected = [record for record in records if record["job_id"] in requested_ids]
        if len(selected) != len(requested_jobs):
            raise RuntimeError("ancient checkpoint lacks a requested job")
    if args.simulate_only:
        final_revision = structured.git_revision(script=Path(__file__))
        structured.require_revision_unchanged(revision, final_revision)
        print(json.dumps({
            "checkpoint": checkpoint_audit(checkpoint, records, config_sha256),
            "requested_selection": record_selection_audit(selected),
            "configuration_sha256": config_sha256,
            "source_commit": final_revision["commit"],
            "requested_replicates_per_panel": requested_replicates,
        }, indent=2, allow_nan=False))
        return 0
    if len(selected) != len(jobs):
        raise RuntimeError("ancient full analysis requires all 300 jobs")

    result_lock = structured.SingleWriterLease(
        args.result_dir, ".stdpopsim_ancient_result.lock"
    ).acquire()
    try:
        pre_analysis_gate = structured.compute_gate(args.compute_state)
        analysis = analyze_records(
            selected,
            args.canonical_root,
            compute_state=args.compute_state,
        )
        final_revision = structured.git_revision(script=Path(__file__))
        structured.require_revision_unchanged(revision, final_revision)
        runtime = structured.runtime_audit(priority)
        for package in ("padze", "stdpopsim", "msprime", "tskit"):
            runtime["packages"][package] = importlib_metadata.version(package)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "known_truth_five_focal_event_synthetic_transfer_not_natural_accuracy",
            "git": revision,
            "final_source_recheck": final_revision,
            "initial_compute_gate": initial_gate,
            "pre_analysis_compute_gate": pre_analysis_gate,
            "runtime": runtime,
            "configuration": config,
            "configuration_sha256": config_sha256,
            "checkpoint": checkpoint_audit(checkpoint, selected, config_sha256),
            "analysis": analysis,
        }
        output = args.result_dir / "results.json"
        output_audit = structured.write_json_atomic(output, result, indent=2)
        print(json.dumps({
            "output": output_audit,
            "checkpoint": result["checkpoint"],
            "primary_balanced_accuracy": analysis["representations"]["raw_all"][
                "B_C_balanced_accuracy"
            ],
            "gate_roc_auc": analysis["appreciable_gate"][
                "overall_positive_control_roc_auc"
            ],
        }, indent=2, allow_nan=False))
    finally:
        result_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
