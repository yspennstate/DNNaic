#!/usr/bin/env python3
"""Run two independent, known-direction stdpopsim catalog panels.

The bank contains one human pulse (CEU -> western Ashkenazi) and one
nonhuman continuous edge (Israeli wolf -> golden jackal).  Each positive is
paired with an independently simulated focal-ablation control.  The control
changes exactly one event proportion or one initial migration-matrix cell;
all other catalog history is retained.

This is a synthetic transfer and focal-ablation benchmark.  It is not a
natural-data accuracy estimate, a reproduction of observed sample sizes, or
evidence that the frozen sustained-migration gate is calibrated for pulse
proportions or the very large dog migration rate.
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

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import stdpopsim_neanderthal_benchmark as stdbench
from scripts import structured_transfer_pilot as structured
from dnnaic.semantics import class_for_forward_edge


SCHEMA_VERSION = "dnnaic-stdpopsim-independent-catalog-benchmark-v1"
CHECKPOINT_SCHEMA = "dnnaic-stdpopsim-independent-catalog-checkpoint-v1"
PINNED_STDPOPSIM_VERSION = "0.3.0"
PINNED_MSPRIME_VERSION = "1.4.2"
PINNED_TSKIT_VERSION = "1.0.3"
SOURCE_CONTRACTS = {
    "HomSap": {
        "module": "stdpopsim.catalog.HomSap.demographic_models",
        "sha256": "ca1adb03f251b7fc293323ef9fe4e77ec9e705a9ad38c21382946bed7c791e1c",
    },
    "CanFam": {
        "module": "stdpopsim.catalog.CanFam.demographic_models",
        "sha256": "2eb9f4e525314717a139aa4a1585a573d2bd70255e07c1e16b5fff3bf844cc9b",
    },
}
CANONICAL_ARRAY_CONTRACTS = {
    "X.npy": {
        "bytes": 141_926_528,
        "sha256": "8a0a54b8d827301d47235ee196026687522180a9bcce07f2c52936e9d9bb56f5",
    },
    "design.npy": {
        "bytes": 25_344_128,
        "sha256": "beb06a522b59e10f311e5a130190159679b9c10595e30260e63c2f20a9c4500e",
    },
    "direction.npy": {
        "bytes": 2_534_528,
        "sha256": "a956a5bb90e147e3c0a4bf8527e0f8a3c8bd6d522fbc57f5e7a34742fdad7632",
    },
    "groups.npy": {
        "bytes": 76_032_128,
        "sha256": "e1a7c621e915615a178d44b4ce59c77da2d9c1f7549019acab587fe17da71a86",
    },
    "magnitude.npy": {
        "bytes": 5_068_928,
        "sha256": "417933cdad099ae4468253588ec9eb83ed323a34635c2e4dd0144cf13b59ee3c",
    },
}
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
DEFAULT_SEED_BASE = 711_500_001
DEFAULT_CACHE = (
    Path.home()
    / "Documents"
    / "Codex"
    / "2026-07-10"
    / "dnnaic-datasets2-data"
    / "stdpopsim_independent_catalog_2026_07_11"
)
DEFAULT_RESULTS = (
    REPO / "results" / "stdpopsim_independent_catalog_benchmark_2026_07_11"
)


@dataclass(frozen=True)
class PanelSpec:
    panel_id: str
    species_id: str
    model_id: str
    populations: tuple[str, str, str]
    direction_truth: str
    forward_event: str
    forward_donor: str
    forward_recipient: str
    generation_time: float
    mutation_rate: float
    focal_kind: str
    expected_population_order: tuple[str, ...]
    expected_event_count: int
    citation_doi: str
    catalog_url: str
    focal_parameter_semantics: str
    focal_event_index: int | None = None
    event_time: float | None = None
    event_source: str | None = None
    event_dest: str | None = None
    positive_proportion: float | None = None
    matrix_source: str | None = None
    matrix_dest: str | None = None
    positive_rate: float | None = None
    catalog_default_times: tuple[float | None, float | None, float | None] = (
        0.0,
        0.0,
        0.0,
    )
    expected_realized_times: tuple[float, float, float] = (0.0, 0.0, 0.0)


PANELS = (
    PanelSpec(
        panel_id="ashk_ceu_to_waj",
        species_id="HomSap",
        model_id="AshkSub_7G19",
        populations=("J", "WAJ", "CEU"),
        direction_truth="C",
        forward_event="CEU->WAJ lineage",
        forward_donor="CEU",
        forward_recipient="WAJ",
        generation_time=25.0,
        mutation_rate=2.5e-8,
        focal_kind="event_proportion",
        expected_population_order=("YRI", "CHB", "CEU", "ME", "J", "WAJ", "EAJ"),
        expected_event_count=10,
        citation_doi="https://doi.org/10.1093/molbev/msz047",
        catalog_url=(
            "https://popsim-consortium.github.io/stdpopsim-docs/latest/"
            "catalog.html#sec_catalog_homsap_models_ashksub_7g19"
        ),
        focal_parameter_semantics=(
            "backward MassMigration proportion; forward-time pulse ancestry component, "
            "not a continuous migration rate"
        ),
        focal_event_index=3,
        event_time=28.0,
        event_source="WAJ",
        event_dest="CEU",
        positive_proportion=0.17,
    ),
    PanelSpec(
        panel_id="canfam_isw_to_glj",
        species_id="CanFam",
        model_id="EarlyWolfAdmixture_6F14",
        populations=("CRW", "ISW", "GLJ"),
        direction_truth="B",
        forward_event="ISW->GLJ",
        forward_donor="ISW",
        forward_recipient="GLJ",
        generation_time=3.0,
        mutation_rate=1e-8,
        focal_kind="migration_matrix",
        expected_population_order=(
            "BSJ",
            "DNG",
            "CHW",
            "ISW",
            "CRW",
            "GLJ",
            "ancDOG",
            "ancWLF1",
            "ancWLF",
            "ancDW",
            "root",
        ),
        expected_event_count=8,
        citation_doi="https://doi.org/10.1371/journal.pgen.1004016",
        catalog_url=(
            "https://popsim-consortium.github.io/stdpopsim-docs/latest/"
            "catalog.html#sec_catalog_canfam_models_earlywolfadmixture_6f14"
        ),
        focal_parameter_semantics=(
            "backward-lineage continuous rate M[GLJ,ISW]; forward edge is ISW->GLJ, "
            "but 0.05 is not a forward migrant fraction"
        ),
        matrix_source="GLJ",
        matrix_dest="ISW",
        positive_rate=0.05,
        catalog_default_times=(None, None, None),
        expected_realized_times=(0.0, 0.0, 0.0),
    ),
)


@dataclass(frozen=True)
class CatalogJob:
    panel_index: int
    panel_id: str
    replicate_index: int
    family_id: str
    condition: str
    direction_truth: str
    job_id: str
    engine_seed: int


def _canonical_json(value) -> bytes:
    return stdbench._canonical_json(value)


def panel_by_id(panel_id: str) -> PanelSpec:
    matches = [panel for panel in PANELS if panel.panel_id == panel_id]
    if len(matches) != 1:
        raise ValueError(f"unknown independent catalog panel {panel_id!r}")
    return matches[0]


def derived_panel_direction(panel: PanelSpec) -> str:
    roles = {
        population: f"P{index}"
        for index, population in enumerate(panel.populations, start=1)
    }
    return class_for_forward_edge(
        roles[panel.forward_donor], roles[panel.forward_recipient]
    )


def audit_declared_focal_direction(panel: PanelSpec) -> None:
    """Bind declared forward truth to the reversed backward-time focal fields."""
    if panel.focal_kind == "event_proportion":
        backward_source = panel.event_source
        backward_dest = panel.event_dest
    elif panel.focal_kind == "migration_matrix":
        backward_source = panel.matrix_source
        backward_dest = panel.matrix_dest
    else:
        raise AssertionError(f"{panel.panel_id}: unknown focal kind")
    if (
        backward_source != panel.forward_recipient
        or backward_dest != panel.forward_donor
    ):
        raise AssertionError(
            f"{panel.panel_id}: backward focal {backward_source}->{backward_dest} does not "
            f"reverse to declared forward {panel.forward_donor}->{panel.forward_recipient}"
        )


def make_jobs(pairs_per_panel: int, seed_base: int) -> list[CatalogJob]:
    if pairs_per_panel < 1:
        raise ValueError("pairs per panel must be positive")
    total = len(PANELS) * pairs_per_panel * len(CONDITIONS)
    if not 0 < seed_base < 2**31 - total - 1:
        raise ValueError("seed base does not leave room for all jobs")
    jobs = []
    offset = 0
    for panel_index, panel in enumerate(PANELS):
        audit_declared_focal_direction(panel)
        truth = derived_panel_direction(panel)
        if truth != panel.direction_truth:
            raise AssertionError(
                f"{panel.panel_id}: semantic truth {truth} != {panel.direction_truth}"
            )
        for replicate_index in range(pairs_per_panel):
            family_id = f"{panel.panel_id}-family-{replicate_index:04d}"
            for condition in CONDITIONS:
                jobs.append(CatalogJob(
                    panel_index=panel_index,
                    panel_id=panel.panel_id,
                    replicate_index=replicate_index,
                    family_id=family_id,
                    condition=condition,
                    direction_truth=truth,
                    job_id=f"{family_id}__{condition}",
                    engine_seed=seed_base + offset,
                ))
                offset += 1
    if len({job.job_id for job in jobs}) != len(jobs):
        raise AssertionError("independent catalog job IDs collide")
    if len({job.engine_seed for job in jobs}) != len(jobs):
        raise AssertionError("independent catalog engine seeds collide")
    derived = [seed for job in jobs for seed in stdbench._engine_seeds(job.engine_seed)]
    if len(set(derived)) != 2 * len(jobs):
        raise AssertionError("derived ancestry/mutation seeds collide")
    return jobs


def _population_name(demography, reference) -> str:
    if isinstance(reference, str):
        return reference
    return str(demography.populations[int(reference)].name)


def _model_signature(model) -> dict:
    matrix = np.asarray(model.model.migration_matrix, dtype="<f8")
    return {
        "populations": [
            stdbench._population_signature(population)
            for population in model.model.populations
        ],
        "events": [stdbench._event_signature(event) for event in model.model.events],
        "migration_matrix": matrix.tolist(),
        "migration_matrix_sha256_float64": stdbench._sha256_array(matrix),
        "generation_time": float(model.generation_time),
        "mutation_rate": float(model.mutation_rate),
    }


def _focal_matrix_indices(demography, panel: PanelSpec) -> tuple[int, int]:
    if panel.matrix_source is None or panel.matrix_dest is None:
        raise ValueError(f"{panel.panel_id}: matrix focal names are absent")
    return int(demography[panel.matrix_source].id), int(demography[panel.matrix_dest].id)


def _audit_only_matrix_focal_change(
    positive_signature: dict,
    control_signature: dict,
    panel: PanelSpec,
) -> None:
    if any(
        positive_signature[key] != control_signature[key]
        for key in ("populations", "events", "generation_time", "mutation_rate")
    ):
        raise AssertionError(f"{panel.panel_id}: matrix control changed nonmatrix history")
    positive = np.asarray(positive_signature["migration_matrix"], dtype="<f8")
    control = np.asarray(control_signature["migration_matrix"], dtype="<f8")
    source = panel.expected_population_order.index(str(panel.matrix_source))
    dest = panel.expected_population_order.index(str(panel.matrix_dest))
    changed = np.argwhere(positive != control)
    if changed.tolist() != [[source, dest]]:
        raise AssertionError(
            f"{panel.panel_id}: matrix control changed cells {changed.tolist()}"
        )
    if not math.isclose(
        float(positive[source, dest]), float(panel.positive_rate), rel_tol=0, abs_tol=0
    ) or float(control[source, dest]) != 0.0:
        raise AssertionError(f"{panel.panel_id}: focal matrix values changed")


def _audit_ashk_context(model, panel: PanelSpec) -> None:
    events = model.model.events
    expected = (
        (2, "EAJ", "WAJ", 14.0, 1.0),
        (3, "WAJ", "CEU", 28.0, 0.17),
        (4, "WAJ", "J", 29.0, 1.0),
    )
    for index, source, dest, event_time, proportion in expected:
        event = events[index]
        if (
            type(event).__name__ != "MassMigration"
            or _population_name(model.model, event.source) != source
            or _population_name(model.model, event.dest) != dest
            or not math.isclose(float(event.time), event_time, rel_tol=0, abs_tol=0)
            or not math.isclose(
                float(event.proportion), proportion, rel_tol=0, abs_tol=0
            )
        ):
            raise AssertionError(f"{panel.panel_id}: Ashkenazi context event {index} changed")


def _audit_canfam_context(model, panel: PanelSpec) -> None:
    names = panel.expected_population_order
    matrix = np.asarray(model.model.migration_matrix, dtype=float)
    observed = {
        (names[row], names[column]): float(matrix[row, column])
        for row, column in np.argwhere(matrix != 0)
    }
    expected = {
        ("BSJ", "ISW"): 0.18,
        ("DNG", "CHW"): 0.03,
        ("CHW", "DNG"): 0.04,
        ("ISW", "BSJ"): 0.07,
        ("GLJ", "ISW"): 0.05,
    }
    if observed != expected or float(matrix[names.index("ISW"), names.index("GLJ")]) != 0:
        raise AssertionError(f"{panel.panel_id}: initial migration matrix changed")
    events = [stdbench._event_signature(event) for event in model.model.events]
    if events[2] != {
        "ancestral": "ancWLF1",
        "derived": ["ISW", "CRW"],
        "event_type": "PopulationSplit",
        "time": 13389,
    }:
        raise AssertionError(f"{panel.panel_id}: wolf split changed")
    for index, source, dest, rate in (
        (5, "ancDW", "GLJ", 0.02),
        (6, "GLJ", "ancDW", 0.99),
    ):
        event = events[index]
        if not (
            event.get("event_type") == "MigrationRateChange"
            and event.get("time") == 14874
            and event.get("source") == source
            and event.get("dest") == dest
            and event.get("rate") == rate
        ):
            raise AssertionError(f"{panel.panel_id}: deep migration event {index} changed")


def prepare_models() -> tuple[dict[str, dict[str, object]], dict]:
    import stdpopsim

    versions = {
        "stdpopsim": importlib_metadata.version("stdpopsim"),
        "msprime": importlib_metadata.version("msprime"),
        "tskit": importlib_metadata.version("tskit"),
    }
    expected_versions = {
        "stdpopsim": PINNED_STDPOPSIM_VERSION,
        "msprime": PINNED_MSPRIME_VERSION,
        "tskit": PINNED_TSKIT_VERSION,
    }
    if versions != expected_versions:
        raise RuntimeError(f"independent catalog package contract changed: {versions}")
    source_audit = {}
    for species_id, contract in SOURCE_CONTRACTS.items():
        module = importlib.import_module(contract["module"])
        path = Path(module.__file__).resolve()
        digest = structured.sha256_file(path)
        if digest != contract["sha256"]:
            raise RuntimeError(f"{species_id} demographic source changed: {digest}")
        source_audit[species_id] = {
            "path": str(path),
            "sha256": digest,
            "module": contract["module"],
        }

    models: dict[str, dict[str, object]] = {}
    panel_audits = {}
    for panel in PANELS:
        positive = stdpopsim.get_species(panel.species_id).get_demographic_model(
            panel.model_id
        )
        if positive.id != panel.model_id:
            raise AssertionError(f"{panel.panel_id}: model ID changed")
        if not math.isclose(
            float(positive.generation_time), panel.generation_time, rel_tol=0, abs_tol=0
        ) or not math.isclose(
            float(positive.mutation_rate), panel.mutation_rate, rel_tol=0, abs_tol=0
        ):
            raise AssertionError(f"{panel.panel_id}: generation/mutation contract changed")
        positive.model.validate()
        population_order = tuple(population.name for population in positive.model.populations)
        if population_order != panel.expected_population_order:
            raise AssertionError(f"{panel.panel_id}: population order changed")
        if len(positive.model.events) != panel.expected_event_count:
            raise AssertionError(f"{panel.panel_id}: event count changed")
        if any(name not in population_order for name in panel.populations):
            raise AssertionError(f"{panel.panel_id}: selected population disappeared")

        control = copy.deepcopy(positive)
        if panel.focal_kind == "event_proportion":
            _audit_ashk_context(positive, panel)
            positive_event = positive.model.events[int(panel.focal_event_index)]
            if (
                _population_name(positive.model, positive_event.source)
                != panel.forward_recipient
                or _population_name(positive.model, positive_event.dest)
                != panel.forward_donor
            ):
                raise AssertionError(f"{panel.panel_id}: actual event direction changed")
            control.model.events[int(panel.focal_event_index)].proportion = 0.0
        elif panel.focal_kind == "migration_matrix":
            _audit_canfam_context(positive, panel)
            source, dest = _focal_matrix_indices(control.model, panel)
            if (
                positive.model.populations[source].name != panel.forward_recipient
                or positive.model.populations[dest].name != panel.forward_donor
            ):
                raise AssertionError(f"{panel.panel_id}: actual matrix direction changed")
            control.model.migration_matrix[source, dest] = 0.0
        else:
            raise AssertionError(f"{panel.panel_id}: unknown focal kind")
        control.model.validate()

        positive_signature = _model_signature(positive)
        control_signature = _model_signature(control)
        if panel.focal_kind == "event_proportion":
            if (
                positive_signature["populations"] != control_signature["populations"]
                or positive_signature["migration_matrix"]
                != control_signature["migration_matrix"]
                or positive_signature["generation_time"]
                != control_signature["generation_time"]
                or positive_signature["mutation_rate"] != control_signature["mutation_rate"]
            ):
                raise AssertionError(f"{panel.panel_id}: event control changed other history")
            changed = [
                index
                for index, (left, right) in enumerate(zip(
                    positive_signature["events"], control_signature["events"]
                ))
                if left != right
            ]
            if changed != [panel.focal_event_index]:
                raise AssertionError(f"{panel.panel_id}: changed event set {changed}")
            left = dict(positive_signature["events"][int(panel.focal_event_index)])
            right = dict(control_signature["events"][int(panel.focal_event_index)])
            if left.pop("proportion") != panel.positive_proportion or right.pop(
                "proportion"
            ) != 0.0 or left != right:
                raise AssertionError(f"{panel.panel_id}: event control changed beyond proportion")
        else:
            _audit_only_matrix_focal_change(
                positive_signature, control_signature, panel
            )

        catalog_times = tuple(
            positive.model[name].default_sampling_time for name in panel.populations
        )
        if catalog_times != panel.catalog_default_times:
            raise AssertionError(f"{panel.panel_id}: catalog sampling defaults changed")
        sample_sets = positive.get_sample_sets(
            {name: INDIVIDUALS_PER_POPULATION for name in panel.populations},
            ploidy=2,
        )
        if len(sample_sets) != 3 or any(not sample_set.num_samples for sample_set in sample_sets):
            raise AssertionError(f"{panel.panel_id}: sample-set construction changed")
        if not any(citation.doi == panel.citation_doi for citation in positive.citations):
            raise AssertionError(f"{panel.panel_id}: demographic-model citation changed")
        citations = [
            {
                "author": citation.author,
                "year": int(citation.year),
                "doi": citation.doi,
                "reasons": sorted(map(str, citation.reasons)),
            }
            for citation in positive.citations
        ]

        models[panel.panel_id] = {"positive": positive, "control": control}
        panel_audits[panel.panel_id] = {
            "specification": asdict(panel),
            "population_order": list(panel.populations),
            "catalog_default_sampling_times": dict(zip(panel.populations, catalog_times)),
            "expected_realized_sampling_times": dict(
                zip(panel.populations, panel.expected_realized_times)
            ),
            "catalog_url": panel.catalog_url,
            "description": positive.description,
            "long_description_sha256": hashlib.sha256(
                positive.long_description.encode("utf-8")
            ).hexdigest(),
            "citations": citations,
            "positive_model_signature": positive_signature,
            "control_model_signature": control_signature,
            "positive_model_signature_sha256": hashlib.sha256(
                _canonical_json(positive_signature)
            ).hexdigest(),
            "control_model_signature_sha256": hashlib.sha256(
                _canonical_json(control_signature)
            ).hexdigest(),
            "control_differs_only_at_focal_component": True,
        }
    return models, {
        "versions": versions,
        "source_files": source_audit,
        "species_ids": sorted({panel.species_id for panel in PANELS}),
        "panels": panel_audits,
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
        raise AssertionError("independent catalog contig length/ploidy changed")
    if not math.isclose(
        float(contig.mutation_rate), float(model.mutation_rate), rel_tol=0, abs_tol=0
    ) or not math.isclose(
        float(contig.recombination_map.mean_rate),
        RECOMBINATION_RATE,
        rel_tol=0,
        abs_tol=1e-30,
    ):
        raise AssertionError("independent catalog contig rates changed")
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
        raise AssertionError("independent catalog total sample count changed")
    if int(tree_sequence.num_individuals) != 3 * INDIVIDUALS_PER_POPULATION:
        raise AssertionError("independent catalog diploid individual count changed")

    count_matrices = []
    sample_sizes = []
    locus_ids = []
    count_hash = hashlib.sha256()
    multiallelic = 0
    for variant in tree_sequence.variants(copy=False):
        genotype = np.asarray(variant.genotypes)
        if genotype.shape != (3 * GENE_COPIES_PER_POPULATION,) or np.any(genotype < 0):
            raise AssertionError("independent catalog genotype vector is invalid")
        allele_count = int(genotype.max()) + 1
        if allele_count < 1:
            continue
        if allele_count > 4:
            raise AssertionError("pinned JC69 simulation emitted more than four alleles")
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
            raise AssertionError("independent catalog allele counts do not conserve samples")
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

    loci = LociData(
        populations=["P1", "P2", "P3"],
        count_matrices=count_matrices,
        sample_sizes=np.vstack(sample_sizes).astype(np.int64, copy=False),
        locus_ids=locus_ids,
        metadata=Metadata(
            source=(
                f"stdpopsim {PINNED_STDPOPSIM_VERSION} {panel.species_id}/"
                f"{panel.model_id} panel {panel.panel_id}"
            ),
            populations=["P1", "P2", "P3"],
            sample_ids={name: [] for name in ("P1", "P2", "P3")},
            ploidy={name: 2 for name in ("P1", "P2", "P3")},
            n_loci_read=int(tree_sequence.num_sites),
            n_loci_kept=len(count_matrices),
            filters_applied=[
                "globally polymorphic across frozen P1/P2/P3; all alleles retained"
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
        raise RuntimeError(f"independent catalog PADZE contract changed: {exc}") from exc
    if curve.shape != (198, 28) or not np.isfinite(curve).all():
        raise AssertionError("independent catalog PADZE curve is invalid")
    if not np.array_equal(curve[:, 0], stdbench.FULL_DEPTHS):
        raise AssertionError("independent catalog PADZE depth grid changed")
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
    job: CatalogJob,
    models: dict[str, dict[str, object]],
    model_audit: dict,
    cache_dir: Path,
    *,
    compute_state: Path | None = None,
) -> dict:
    import msprime

    panel = panel_by_id(job.panel_id)
    model = models[job.panel_id][job.condition]
    if compute_state is not None:
        structured.compute_gate(compute_state)
    started = time.perf_counter()
    contig = make_contig(model)
    sample_sets = model.get_sample_sets(
        {name: INDIVIDUALS_PER_POPULATION for name in panel.populations},
        ploidy=2,
    )
    ancestry_seed, mutation_seed = stdbench._engine_seeds(job.engine_seed)
    tree_sequence = msprime.sim_ancestry(
        samples=sample_sets,
        recombination_rate=contig.recombination_map,
        demography=model.model,
        ploidy=2,
        random_seed=ancestry_seed,
        model=msprime.StandardCoalescent(),
        discrete_genome=True,
    )
    tree_sequence = msprime.sim_mutations(
        tree_sequence,
        random_seed=mutation_seed,
        rate=float(contig.mutation_rate),
        model=msprime.JC69(),
        discrete_genome=True,
    )
    raw_tree = None
    if job.replicate_index == 0:
        raw_tree = _dump_tree_sequence_atomic(
            tree_sequence,
            cache_dir / "raw_audit" / f"{job.job_id}.trees",
        )
    population_ids = {name: int(model.model[name].id) for name in panel.populations}
    curve, simulation_audit = tree_sequence_to_curve(
        tree_sequence,
        panel,
        population_ids,
        model_audit["panels"][panel.panel_id]["expected_realized_sampling_times"],
        compute_state=compute_state,
    )
    curve32 = curve.astype(np.float32)
    return {
        **asdict(job),
        "engine_derived_ancestry_seed": int(ancestry_seed),
        "engine_derived_mutation_seed": int(mutation_seed),
        "ancestry_model": "msprime.StandardCoalescent",
        "mutation_model": "msprime.JC69",
        "discrete_genome": True,
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
        raise RuntimeError("refusing to save duplicate independent catalog jobs")
    curves = np.stack([np.asarray(record["curve"], dtype=np.float32) for record in records])
    if curves.shape != (len(records), 198, 28) or not np.isfinite(curves).all():
        raise RuntimeError("refusing to save invalid independent catalog curves")
    metadata = []
    for record, curve in zip(records, curves):
        current = {key: value for key, value in record.items() if key != "curve"}
        if stdbench._sha256_array(curve.astype("<f4", copy=False)) != current[
            "curve_sha256_float32"
        ]:
            raise RuntimeError("independent catalog curve hash changed before checkpoint save")
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
    jobs: Sequence[CatalogJob],
) -> list[dict]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema", "config_sha256", "metadata_json", "curves"}
        if set(archive.files) != required:
            raise RuntimeError("independent catalog checkpoint member set changed")
        if archive["schema"].tolist() != [CHECKPOINT_SCHEMA]:
            raise RuntimeError("independent catalog checkpoint schema changed")
        if archive["config_sha256"].tolist() != [config_sha256]:
            raise RuntimeError("independent catalog checkpoint configuration changed")
        metadata = json.loads(str(archive["metadata_json"][0]))
        curves = np.asarray(archive["curves"], dtype=np.float32)
    manifest = {job.job_id: job for job in jobs}
    if len(metadata) != len(curves):
        raise RuntimeError("independent catalog checkpoint cardinality changed")
    records = []
    seen = set()
    for current, curve in zip(metadata, curves):
        job_id = current.get("job_id")
        if job_id not in manifest or job_id in seen:
            raise RuntimeError("independent catalog checkpoint has unknown/duplicate job")
        seen.add(job_id)
        if any(current.get(key) != value for key, value in asdict(manifest[job_id]).items()):
            raise RuntimeError(f"independent catalog manifest changed for {job_id}")
        expected_seeds = stdbench._engine_seeds(manifest[job_id].engine_seed)
        if [
            current.get("engine_derived_ancestry_seed"),
            current.get("engine_derived_mutation_seed"),
        ] != expected_seeds:
            raise RuntimeError(f"independent catalog derived seeds changed for {job_id}")
        if curve.shape != (198, 28) or not np.isfinite(curve).all():
            raise RuntimeError(f"independent catalog curve is invalid for {job_id}")
        if stdbench._sha256_array(curve.astype("<f4", copy=False)) != current[
            "curve_sha256_float32"
        ]:
            raise RuntimeError(f"independent catalog curve hash changed for {job_id}")
        raw = current.get("raw_tree_sequence")
        if raw is not None:
            raw_path = Path(raw["path"])
            if (
                not raw_path.is_file()
                or raw_path.stat().st_size != raw["bytes"]
                or structured.sha256_file(raw_path) != raw["sha256"]
            ):
                raise RuntimeError(f"independent catalog raw tree changed for {job_id}")
        records.append({**current, "curve": curve})
    return sorted(records, key=record_key)


def record_selection_audit(records: Sequence[dict]) -> dict:
    families = {}
    panels = {}
    for record in records:
        families.setdefault(record["family_id"], set()).add(record["condition"])
        panels[record["panel_id"]] = panels.get(record["panel_id"], 0) + 1
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
            raise RuntimeError("independent catalog checkpoint audit schema changed")
        if archive["config_sha256"].tolist() != [config_sha256]:
            raise RuntimeError("independent catalog checkpoint audit config changed")
        stored_metadata = json.loads(str(archive["metadata_json"][0]))
        stored_curves = np.asarray(archive["curves"], dtype=np.float32)
        stored_curve_shape = list(stored_curves.shape)
    expected_metadata = [
        {key: value for key, value in record.items() if key != "curve"}
        for record in records
    ]
    if stored_metadata != expected_metadata:
        raise RuntimeError("independent catalog checkpoint bytes/audit differ")
    if (
        stored_curves.shape != (len(records), 198, 28)
        or not np.isfinite(stored_curves).all()
        or not np.array_equal(
            stored_curves[:, :, 0], np.tile(stdbench.FULL_DEPTHS, (len(records), 1))
        )
    ):
        raise RuntimeError("independent catalog checkpoint stored curves are invalid")
    for record, curve in zip(records, stored_curves):
        if stdbench._sha256_array(curve.astype("<f4", copy=False)) != record[
            "curve_sha256_float32"
        ]:
            raise RuntimeError(
                f"independent catalog checkpoint stored curve changed for {record['job_id']}"
            )
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
    labels = ["B", "C", "A"]
    return {
        "labels": labels,
        "matrix": confusion_matrix(truth, prediction, labels=labels).astype(int).tolist(),
        "rows_are_truth_columns_are_predictions": True,
    }


def analyze_records(
    records: Sequence[dict],
    canonical_root: Path,
    *,
    compute_state: Path | None = None,
) -> dict:
    records = sorted(records, key=record_key)
    expected_records = len(PANELS) * DEFAULT_PAIRS_PER_PANEL * len(CONDITIONS)
    if len(records) != expected_records:
        raise RuntimeError(
            f"independent catalog analysis requires {expected_records} records"
        )
    if len({record["job_id"] for record in records}) != len(records):
        raise RuntimeError("independent catalog analysis has duplicate jobs")
    families = {}
    for record in records:
        families.setdefault(record["family_id"], set()).add(record["condition"])
    if (
        len(families) != len(PANELS) * DEFAULT_PAIRS_PER_PANEL
        or any(conditions != set(CONDITIONS) for conditions in families.values())
    ):
        raise RuntimeError("independent catalog analysis requires complete families")

    curves = np.stack([np.asarray(record["curve"], dtype=float) for record in records])
    if curves.shape != (expected_records, 198, 28) or not np.isfinite(curves).all():
        raise RuntimeError("independent catalog analysis curves are invalid")
    if not np.array_equal(curves[:, :, 0], np.tile(stdbench.FULL_DEPTHS, (len(records), 1))):
        raise RuntimeError("independent catalog analysis depth grids changed")
    conditions = np.asarray([record["condition"] for record in records])
    positive_rows = conditions == "positive"
    control_rows = conditions == "control"
    truth = np.asarray([record["direction_truth"] for record in records])
    panel_ids = np.asarray([record["panel_id"] for record in records])
    observed_truth_counts = {
        label: int(np.sum(truth[positive_rows] == label)) for label in ("B", "C")
    }
    if observed_truth_counts != {"B": 30, "C": 30}:
        raise RuntimeError(f"independent catalog truth balance changed: {observed_truth_counts}")

    if compute_state is not None:
        structured.compute_gate(compute_state)
    canonical = structured.load_canonical(canonical_root, max_depth=199)
    canonical_table = np.asarray(canonical["table"], dtype=float)
    canonical_labels = np.asarray(canonical["labels"])
    canonical_rates = np.asarray(canonical["rates"], dtype=float)
    canonical_positive = np.isin(canonical_labels, ["A", "B", "C"])
    canonical_audit = canonical["audit"]
    if (
        canonical_audit.get("replicates") != 3200
        or canonical_audit.get("rows") != 633600
        or canonical_audit.get("selected_curve_shape") != [3200, 198, 28]
        or canonical_audit.get("array_contracts") != CANONICAL_ARRAY_CONTRACTS
    ):
        raise RuntimeError("canonical training array contract changed")
    observed_class_counts = {
        str(label): int(count)
        for label, count in zip(*np.unique(canonical_labels, return_counts=True))
    }
    if observed_class_counts != {"A": 900, "B": 900, "C": 900, "D": 500}:
        raise RuntimeError(
            f"canonical training label counts changed: {observed_class_counts}"
        )

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
                "species_id": panel.species_id,
                "truth": panel.direction_truth,
                "forward_event": panel.forward_event,
                "interpretation": (
                    "compatibility with the declared focal direction under the retained full "
                    "catalog background; not causal proof that the focal edge was recovered"
                ),
                "recall": stdbench._wilson(correct),
                "predicted_class_counts": {
                    str(label): int(count)
                    for label, count in zip(*np.unique(prediction[use], return_counts=True))
                },
                "scaler_rms_z_median": float(
                    np.median(np.sqrt(np.mean(z[use] ** 2, axis=1)))
                ),
                "scaler_max_abs_z_p95": float(
                    np.quantile(np.max(np.abs(z[use]), axis=1), 0.95)
                ),
            }
        representations[name] = {
            "status": "target-blind frozen canonical C=1; raw_all is primary",
            "feature_dimension": int(train.shape[1]),
            "B_C_balanced_accuracy": float((recalls["B"] + recalls["C"]) / 2),
            "B_recall": stdbench._wilson(
                positive_prediction[positive_truth == "B"] == "B"
            ),
            "C_recall": stdbench._wilson(
                positive_prediction[positive_truth == "C"] == "C"
            ),
            "equal_panel_macro_accuracy": float(np.mean(panel_accuracy)),
            "equal_panel_macro_constant_baselines": {
                "always_B": 0.5,
                "always_C": 0.5,
            },
            "confusion": _confusion_payload(positive_truth, positive_prediction),
            "per_panel": per_panel,
            "control_forced_call_counts_diagnostic_only": {
                str(label): int(count)
                for label, count in zip(*np.unique(prediction[control_rows], return_counts=True))
            },
            "scaler_rms_z_median": float(np.median(np.sqrt(np.mean(z**2, axis=1)))),
            "scaler_rms_z_p95": float(
                np.quantile(np.sqrt(np.mean(z**2, axis=1)), 0.95)
            ),
            "scaler_max_abs_z_p95": float(
                np.quantile(np.max(np.abs(z), axis=1), 0.95)
            ),
            "model": stdbench._model_payload(
                scaler,
                model,
                feature_columns=structured.representation_columns(name),
            ),
            "guardrail": (
                "B and C each supply one panel and 30 positive Monte Carlo rows; constant-B "
                "and constant-C panel baselines are both 0.5. Wilson intervals are descriptive "
                "within fixed catalog panels, not across-demography confidence intervals."
            ),
        }

    gate_train, gate_contract = stdbench._gate_features(canonical_table)
    gate_external, external_gate_contract = stdbench._gate_features(curves)
    if gate_contract != external_gate_contract:
        raise AssertionError("independent catalog/canonical gate contracts differ")
    gate_target = (
        canonical_positive & (canonical_rates >= structured.APPRECIABLE)
    ).astype(int)
    gate_scaler, gate_model = structured._fit_model(gate_train, gate_target, C=1.0)
    gate_z = gate_scaler.transform(gate_external)
    gate_positive_index = int(np.flatnonzero(gate_model.classes_ == 1)[0])
    gate_score = gate_model.predict_proba(gate_z)[:, gate_positive_index]
    gate_truth = positive_rows.astype(int)
    per_panel_gate = {}
    panel_aucs = []
    for panel in PANELS:
        use = panel_ids == panel.panel_id
        auc = float(roc_auc_score(gate_truth[use], gate_score[use]))
        panel_aucs.append(auc)
        per_panel_gate[panel.panel_id] = {
            "species_id": panel.species_id,
            "positive_control_roc_auc": auc,
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
            "engine_seed": int(record["engine_seed"]),
            "engine_derived_ancestry_seed": int(record["engine_derived_ancestry_seed"]),
            "engine_derived_mutation_seed": int(record["engine_derived_mutation_seed"]),
            "curve_sha256_float32": record["curve_sha256_float32"],
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
            "raw_all_scaler_rms_z": float(np.sqrt(np.mean(primary_z[index] ** 2))),
            "raw_all_scaler_max_abs_z": float(np.max(np.abs(primary_z[index]))),
            "frozen_gate_score": float(gate_score[index]),
        })

    simulation_summary = {}
    for panel in PANELS:
        simulation_summary[panel.panel_id] = {}
        for condition in CONDITIONS:
            current = [
                record
                for record in records
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
        "species": len({panel.species_id for panel in PANELS}),
        "catalog_models": len({panel.model_id for panel in PANELS}),
        "focal_panels": len(PANELS),
        "families": len(families),
        "records": len(records),
        "direction_accuracy_rows": int(positive_rows.sum()),
        "focal_absent_controls_excluded_from_direction_accuracy": int(control_rows.sum()),
        "truth_counts": observed_truth_counts,
        "representations": representations,
        "frozen_gate_score_discrimination": {
            "status": (
                "score-only positive/control discrimination diagnostic; no threshold accuracy "
                "or appreciable-migration calibration claim"
            ),
            "canonical_contract": gate_contract,
            "pooled_positive_control_roc_auc": float(
                roc_auc_score(gate_truth, gate_score)
            ),
            "equal_panel_macro_positive_control_roc_auc": float(np.mean(panel_aucs)),
            "per_panel": per_panel_gate,
            "model": stdbench._model_payload(gate_scaler, gate_model),
            "guardrail": (
                "The frozen gate target is sustained migration rate >=2.5e-4. The human focal "
                "parameter is a 0.17 pulse proportion and the dog focal parameter is a 0.05 "
                "backward-lineage per-generation catalog rate. Scores are not comparable "
                "effect sizes."
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
            "This balanced bank adds two demographic systems, not 60 independent systems. "
            "Ashkenazi pulse timing is model-assumed; CanFam retains deep 0.99/0.02 ancestral "
            "jackal flow in both conditions and its focal rate is far outside training. The "
            "common recombination rate is a canonical-transfer contract, not a dog estimate."
            " Positive-row direction recall is compatibility with each declared direction "
            "under its complex retained background, not causal attribution to the focal edge."
        ),
    }


def _helper_source_audit() -> dict:
    paths = {
        "scripts/stdpopsim_neanderthal_benchmark.py": Path(stdbench.__file__).resolve(),
        "scripts/structured_transfer_pilot.py": Path(structured.__file__).resolve(),
        "dnnaic/semantics.py": (REPO / "dnnaic" / "semantics.py").resolve(),
    }
    return {
        name: {"path": str(path), "sha256": structured.sha256_file(path)}
        for name, path in paths.items()
    }


def configuration(
    pairs_per_panel: int,
    seed_base: int,
    jobs: Sequence[CatalogJob],
    model_audit: dict,
    revision: dict,
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source_revision": {
            key: revision.get(key)
            for key in (
                "commit",
                "script_sha256",
                "head_script_sha256",
                "head_blob_oid",
                "worktree_blob_oid",
                "tracked_diff_sha256",
                "tracked_dirty_at_snapshot",
            )
        },
        "helper_sources": _helper_source_audit(),
        "processing_dependencies": {
            "numpy": np.__version__,
            "scikit_learn": importlib_metadata.version("scikit-learn"),
            "padze": importlib_metadata.version("padze"),
            "stdpopsim": model_audit["versions"]["stdpopsim"],
            "msprime": model_audit["versions"]["msprime"],
            "tskit": model_audit["versions"]["tskit"],
        },
        "canonical_training_contract": {
            "replicates": 3200,
            "rows": 633600,
            "curve_shape": [3200, 198, 28],
            "label_counts": {"A": 900, "B": 900, "C": 900, "D": 500},
            "array_contracts": CANONICAL_ARRAY_CONTRACTS,
        },
        "pairs_per_panel": int(pairs_per_panel),
        "seed_base": int(seed_base),
        "job_manifest": [asdict(job) for job in jobs],
        "derived_seed_ledger": [
            {
                "job_id": job.job_id,
                "ancestry_seed": stdbench._engine_seeds(job.engine_seed)[0],
                "mutation_seed": stdbench._engine_seeds(job.engine_seed)[1],
            }
            for job in jobs
        ],
        "model_audit": model_audit,
        "sequence": {
            "length_bp": SEQUENCE_LENGTH,
            "mutation_rate": "catalog model-specific",
            "recombination_rate": RECOMBINATION_RATE,
            "recombination_guardrail": (
                "frozen canonical-training rate for cross-species comparability, not a "
                "species-faithful dog recombination estimate"
            ),
            "independent_contigs_per_job": 1,
            "ancestry_model": "msprime.StandardCoalescent",
            "mutation_model": "msprime.JC69",
            "discrete_genome": True,
        },
        "sampling": {
            "individuals_per_population": INDIVIDUALS_PER_POPULATION,
            "gene_copies_per_population": GENE_COPIES_PER_POPULATION,
            "ploidy": 2,
            "times": "panel-specific frozen expected realized times",
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
            "truth_counts": {"B": pairs_per_panel, "C": pairs_per_panel},
            "controls_excluded_from_direction_accuracy": True,
        },
        "raw_retention": "first positive/control tree sequence for each panel (four trees)",
        "checkpoint_portability_guardrail": (
            "source and retained-tree paths are absolute and revalidated; checkpoint is "
            "bound to its run layout"
        ),
        "pairing_guardrail": (
            "families match panel/replicate only; positive and control use independent "
            "ancestry and mutation seeds"
        ),
        "runtime_guardrail": (
            "each single msprime call is uninterruptible until that job returns; full runs are "
            "Azure-only at nice>=10 with the compute gate polled between simulation and PADZE jobs"
        ),
    }


def require_safe_execution_target(
    compute_target: str,
    *,
    simulate_only: bool,
    limit_replicates: int | None,
    operating_system: str = os.name,
) -> None:
    bounded_smoke = simulate_only and limit_replicates == 1
    if not bounded_smoke and (compute_target != "azure" or operating_system == "nt"):
        raise ValueError(
            "all independent-catalog execution except the one-replicate smoke is Azure-only; "
            "use a Linux Azure worker with --compute-target azure"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-per-panel", type=int, default=DEFAULT_PAIRS_PER_PANEL)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument("--limit-replicates", type=int, default=None)
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument(
        "--compute-state", type=Path, default=structured.DEFAULT_COMPUTE_STATE
    )
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
    try:
        require_safe_execution_target(
            args.compute_target,
            simulate_only=args.simulate_only,
            limit_replicates=args.limit_replicates,
        )
    except ValueError as exc:
        parser.error(str(exc))

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
    config = configuration(
        args.pairs_per_panel, args.seed_base, jobs, model_audit, revision
    )
    config_sha256 = hashlib.sha256(_canonical_json(config)).hexdigest()
    requested_replicates = (
        args.pairs_per_panel if args.limit_replicates is None else args.limit_replicates
    )
    requested_jobs = [
        job for job in jobs if job.replicate_index < requested_replicates
    ]
    checkpoint = args.cache_dir / "stdpopsim_independent_catalog_features.npz"
    with structured.SingleWriterLease(
        args.cache_dir, ".stdpopsim_independent_catalog.lock"
    ):
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
            raise RuntimeError("independent catalog checkpoint lacks a requested job")
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
        raise RuntimeError("independent catalog full analysis requires all 120 jobs")

    result_lock = structured.SingleWriterLease(
        args.result_dir, ".stdpopsim_independent_catalog_result.lock"
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
            "status": "known_truth_two_species_focal_ablation_synthetic_transfer",
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
            "gate_macro_roc_auc": analysis["frozen_gate_score_discrimination"][
                "equal_panel_macro_positive_control_roc_auc"
            ],
        }, indent=2, allow_nan=False))
    finally:
        result_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
