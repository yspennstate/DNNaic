from dataclasses import asdict, replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts import stdpopsim_independent_catalog_benchmark as benchmark


def test_panel_mapping_and_balanced_job_manifest():
    assert [
        (panel.panel_id, panel.positive_direction_truth) for panel in benchmark.PANELS
    ] == [
        ("ashk_ceu_to_waj", "C"),
        ("canfam_isw_to_glj", "B"),
    ]
    assert [benchmark.derived_panel_direction(panel) for panel in benchmark.PANELS] == [
        "C",
        "B",
    ]
    jobs = benchmark.make_jobs(benchmark.DEFAULT_PAIRS_PER_PANEL, benchmark.DEFAULT_SEED_BASE)
    assert len(jobs) == 120
    assert len({job.job_id for job in jobs}) == 120
    assert len({job.family_id for job in jobs}) == 60
    assert len({job.engine_seed for job in jobs}) == 120
    assert jobs[0].engine_seed == 711_500_001
    assert jobs[60].engine_seed == 711_500_061
    assert jobs[-1].engine_seed == 711_500_120
    assert all(job.panel_candidate_direction in {"B", "C"} for job in jobs)
    assert all(
        job.direction_truth == job.panel_candidate_direction
        for job in jobs
        if job.condition == "positive"
    )
    assert all(
        job.direction_truth is None for job in jobs if job.condition == "control"
    )
    derived = [
        seed
        for job in jobs
        for seed in benchmark.stdbench._engine_seeds(job.engine_seed)
    ]
    assert len(derived) == len(set(derived)) == 240
    assert benchmark.stdbench._engine_seeds(jobs[0].engine_seed) == [
        567_062_081,
        1_687_843_415,
    ]
    assert benchmark.stdbench._engine_seeds(jobs[1].engine_seed) == [
        252_020_544,
        103_844_422,
    ]
    assert benchmark.stdbench._engine_seeds(jobs[60].engine_seed) == [
        1_945_998_403,
        373_935_086,
    ]


def test_declared_forward_truth_is_bound_to_backward_focal_orientation():
    for panel in benchmark.PANELS:
        benchmark.audit_declared_focal_direction(panel)
    changed = replace(
        benchmark.panel_by_id("ashk_ceu_to_waj"),
        event_source="CEU",
        event_dest="WAJ",
    )
    with pytest.raises(AssertionError, match="does not reverse"):
        benchmark.audit_declared_focal_direction(changed)


def _matrix_signatures():
    panel = benchmark.panel_by_id("canfam_isw_to_glj")
    size = len(panel.expected_population_order)
    positive = np.zeros((size, size), dtype=float)
    source = panel.expected_population_order.index("GLJ")
    dest = panel.expected_population_order.index("ISW")
    positive[source, dest] = 0.05
    control = positive.copy()
    control[source, dest] = 0.0
    common = {
        "populations": [{"name": name} for name in panel.expected_population_order],
        "events": [{"event_type": "PopulationSplit"}],
        "generation_time": 3.0,
        "mutation_rate": 1e-8,
    }
    return panel, {
        **common,
        "migration_matrix": positive.tolist(),
    }, {
        **common,
        "migration_matrix": control.tolist(),
    }


def test_matrix_control_allows_exactly_the_focal_cell():
    panel, positive, control = _matrix_signatures()
    benchmark._audit_only_matrix_focal_change(positive, control, panel)


def test_matrix_control_rejects_a_second_changed_cell():
    panel, positive, control = _matrix_signatures()
    control["migration_matrix"][0][1] = 0.01
    with pytest.raises(AssertionError, match="changed cells"):
        benchmark._audit_only_matrix_focal_change(positive, control, panel)


def _record(job, curve):
    ancestry_seed, mutation_seed = benchmark.stdbench._engine_seeds(job.engine_seed)
    return {
        **asdict(job),
        "engine_derived_ancestry_seed": ancestry_seed,
        "engine_derived_mutation_seed": mutation_seed,
        "ancestry_model": "msprime.StandardCoalescent",
        "mutation_model": "msprime.JC69",
        "discrete_genome": True,
        "elapsed_seconds": 1.0,
        "raw_tree_sequence": None,
        "simulation_audit": {"globally_polymorphic_loci": 10},
        "curve_sha256_float32": benchmark.stdbench._sha256_array(
            curve.astype("<f4", copy=False)
        ),
        "curve": curve,
    }


def test_checkpoint_round_trip_binds_manifest_config_and_curve(tmp_path):
    jobs = benchmark.make_jobs(1, benchmark.DEFAULT_SEED_BASE)
    curve = np.zeros((198, 28), dtype=np.float32)
    curve[:, 0] = benchmark.stdbench.FULL_DEPTHS
    records = [_record(job, curve + index) for index, job in enumerate(jobs)]
    for record in records:
        record["curve"][:, 0] = benchmark.stdbench.FULL_DEPTHS
        record["curve_sha256_float32"] = benchmark.stdbench._sha256_array(
            record["curve"].astype("<f4", copy=False)
        )
    checkpoint = tmp_path / "checkpoint.npz"
    config_sha256 = "a" * 64
    benchmark.save_checkpoint(checkpoint, records, config_sha256)
    loaded = benchmark.load_checkpoint(checkpoint, config_sha256, jobs)
    assert [record["job_id"] for record in loaded] == [job.job_id for job in jobs]
    audit = benchmark.checkpoint_audit(checkpoint, loaded, config_sha256)
    assert audit["records"] == 4
    assert audit["complete_positive_control_families"] == 2
    assert audit["stored_curve_shape"] == [4, 198, 28]
    with pytest.raises(RuntimeError, match="configuration changed"):
        benchmark.load_checkpoint(checkpoint, "b" * 64, jobs)


def test_checkpoint_rejects_changed_derived_seed(tmp_path):
    jobs = benchmark.make_jobs(1, benchmark.DEFAULT_SEED_BASE)
    curve = np.zeros((198, 28), dtype=np.float32)
    curve[:, 0] = benchmark.stdbench.FULL_DEPTHS
    record = _record(jobs[0], curve)
    record["engine_derived_ancestry_seed"] += 1
    checkpoint = tmp_path / "checkpoint.npz"
    benchmark.save_checkpoint(checkpoint, [record], "a" * 64)
    with pytest.raises(RuntimeError, match="derived seeds changed"):
        benchmark.load_checkpoint(checkpoint, "a" * 64, jobs)


def test_analysis_rejects_incomplete_bank_before_loading_canonical(tmp_path):
    with pytest.raises(RuntimeError, match="requires 120 records"):
        benchmark.analyze_records([], tmp_path)


def test_full_execution_is_rejected_off_azure_but_bounded_smoke_is_allowed():
    with pytest.raises(ValueError, match="Azure-only"):
        benchmark.require_safe_execution_target(
            "local", simulate_only=False, limit_replicates=None, operating_system="nt"
        )
    with pytest.raises(ValueError, match="Azure-only"):
        benchmark.require_safe_execution_target(
            "local", simulate_only=True, limit_replicates=None, operating_system="posix"
        )
    with pytest.raises(ValueError, match="Azure-only"):
        benchmark.require_safe_execution_target(
            "azure", simulate_only=False, limit_replicates=None, operating_system="nt"
        )
    for limit in (2, 29, benchmark.DEFAULT_PAIRS_PER_PANEL):
        with pytest.raises(ValueError, match="Azure-only"):
            benchmark.require_safe_execution_target(
                "local",
                simulate_only=True,
                limit_replicates=limit,
                operating_system="nt",
            )
    benchmark.require_safe_execution_target(
        "azure", simulate_only=False, limit_replicates=None, operating_system="posix"
    )
    benchmark.require_safe_execution_target(
        "local", simulate_only=True, limit_replicates=1, operating_system="nt"
    )


def test_real_catalog_contracts_and_source_hashes():
    stdpopsim = pytest.importorskip("stdpopsim")
    pytest.importorskip("msprime")
    models, audit = benchmark.prepare_models()
    assert audit["versions"] == {
        "stdpopsim": "0.3.0",
        "msprime": "1.4.2",
        "tskit": "1.0.3",
    }
    assert {
        species: source["sha256"]
        for species, source in audit["source_files"].items()
    } == {
        species: contract["sha256"]
        for species, contract in benchmark.SOURCE_CONTRACTS.items()
    }
    ashk = benchmark.panel_by_id("ashk_ceu_to_waj")
    benchmark._audit_ashk_context(models[ashk.panel_id]["positive"], ashk)
    dog = benchmark.panel_by_id("canfam_isw_to_glj")
    benchmark._audit_canfam_context(models[dog.panel_id]["positive"], dog)
    benchmark._audit_only_matrix_focal_change(
        audit["panels"][dog.panel_id]["positive_model_signature"],
        audit["panels"][dog.panel_id]["control_model_signature"],
        dog,
    )
    assert stdpopsim.get_species("CanFam").id == "CanFam"


def test_canfam_tiny_ancestry_resolves_extant_sample_times_to_zero():
    msprime = pytest.importorskip("msprime")
    pytest.importorskip("stdpopsim")
    models, _ = benchmark.prepare_models()
    panel = benchmark.panel_by_id("canfam_isw_to_glj")
    model = models[panel.panel_id]["positive"]
    sample_sets = model.get_sample_sets(
        {name: 1 for name in panel.populations},
        ploidy=2,
    )
    assert [sample_set.time for sample_set in sample_sets] == [None, None, None]
    tree_sequence = msprime.sim_ancestry(
        samples=sample_sets,
        sequence_length=10,
        recombination_rate=0,
        demography=model.model,
        ploidy=2,
        random_seed=12345,
        model=msprime.StandardCoalescent(),
        discrete_genome=True,
    )
    sample_times = tree_sequence.tables.nodes.time[tree_sequence.samples()]
    assert np.array_equal(sample_times, np.zeros(6))


def test_source_file_has_no_unbound_random_seed_formula():
    source = Path(benchmark.__file__).read_bytes()
    assert hashlib.sha256(source).hexdigest() == benchmark.structured.sha256_file(
        Path(benchmark.__file__)
    )
    assert b"np.random.default_rng" not in source
