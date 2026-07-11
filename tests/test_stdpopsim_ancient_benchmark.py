from __future__ import annotations

from dataclasses import asdict
from importlib import util as importlib_util

import numpy as np
import pytest

from scripts import stdpopsim_ancient_benchmark as benchmark


def _curve(offset: float = 0.0) -> np.ndarray:
    curve = np.zeros((198, 28), dtype=np.float32)
    curve[:, 0] = benchmark.stdbench.FULL_DEPTHS
    curve[:, 1:] = offset
    return curve


def _record(job: benchmark.AncientJob, offset: float = 0.0) -> dict:
    curve = _curve(offset)
    ancestry_seed, mutation_seed = benchmark.stdbench._engine_seeds(job.engine_seed)
    return {
        **asdict(job),
        "engine_derived_ancestry_seed": ancestry_seed,
        "engine_derived_mutation_seed": mutation_seed,
        "elapsed_seconds": 0.1,
        "raw_tree_sequence": None,
        "simulation_audit": {
            "globally_polymorphic_loci": 10,
            "multiallelic_globally_polymorphic_loci": 0,
        },
        "curve_sha256_float32": benchmark.stdbench._sha256_array(
            curve.astype("<f4", copy=False)
        ),
        "curve": curve,
    }


def test_panel_contract_spans_two_B_and_three_C_focal_systems():
    assert len(benchmark.PANELS) == 5
    assert [panel.direction_truth for panel in benchmark.PANELS] == [
        "B", "C", "B", "C", "C"
    ]
    assert [benchmark.derived_panel_direction(panel) for panel in benchmark.PANELS] == [
        "B", "C", "B", "C", "C"
    ]
    assert [panel.populations for panel in benchmark.PANELS] == [
        ("EHG", "WHG", "NEO"),
        ("WHG", "YAM", "CHG"),
        ("ANA", "NEO", "Bronze"),
        ("Mbuti", "Han", "Neanderthal"),
        ("LBK", "Sardinian", "Loschbour"),
    ]
    assert benchmark.PANELS[0].control_proportions == (0.0, 1.0)
    assert benchmark.PANELS[1].control_proportions == (1.0, 0.0)
    assert benchmark.PANELS[2].control_proportions == (1.0, 0.0)
    assert all(panel.positive_proportion is not None for panel in benchmark.PANELS[3:])


def test_job_manifest_is_complete_balanced_and_uniquely_seeded():
    jobs = benchmark.make_jobs(30, benchmark.DEFAULT_SEED_BASE)
    assert len(jobs) == 300
    assert len({job.job_id for job in jobs}) == 300
    assert len({job.engine_seed for job in jobs}) == 300
    assert sum(job.direction_truth == "B" and job.condition == "positive" for job in jobs) == 60
    assert sum(job.direction_truth == "C" and job.condition == "positive" for job in jobs) == 90
    for panel in benchmark.PANELS:
        current = [job for job in jobs if job.panel_id == panel.panel_id]
        assert len(current) == 60
        assert {job.replicate_index for job in current} == set(range(30))
        for replicate in range(30):
            family = [job for job in current if job.replicate_index == replicate]
            assert {job.condition for job in family} == {"positive", "control"}
            assert len({job.family_id for job in family}) == 1
            assert len({job.engine_seed for job in family}) == 2


@pytest.mark.parametrize("pairs,seed", [(0, 10), (1, 0), (2, 2**31 - 1)])
def test_job_manifest_rejects_invalid_dimensions(pairs, seed):
    with pytest.raises(ValueError):
        benchmark.make_jobs(pairs, seed)


def test_checkpoint_round_trip_and_strict_full_record_audit(tmp_path):
    jobs = benchmark.make_jobs(1, 1000)
    records = [_record(job, index / 10) for index, job in enumerate(jobs)]
    path = tmp_path / "features.npz"
    benchmark.save_checkpoint(path, records, "a" * 64)
    loaded = benchmark.load_checkpoint(path, "a" * 64, jobs)
    assert len(loaded) == 10
    assert [record["job_id"] for record in loaded] == [job.job_id for job in jobs]
    assert np.array_equal(loaded[0]["curve"], records[0]["curve"])
    audit = benchmark.checkpoint_audit(path, loaded, "a" * 64)
    assert audit["records"] == 10
    assert audit["complete_positive_control_families"] == 5
    assert audit["stored_curve_shape"] == [10, 198, 28]
    with pytest.raises(RuntimeError, match="audit records differ"):
        benchmark.checkpoint_audit(path, loaded[:-1], "a" * 64)
    with pytest.raises(RuntimeError, match="configuration changed"):
        benchmark.load_checkpoint(path, "b" * 64, jobs)


def test_checkpoint_rejects_duplicates(tmp_path):
    job = benchmark.make_jobs(1, 1000)[0]
    record = _record(job)
    with pytest.raises(RuntimeError, match="duplicate"):
        benchmark.save_checkpoint(tmp_path / "duplicate.npz", [record, record], "a" * 64)


def test_configuration_records_truth_counts_and_independent_pairing():
    jobs = benchmark.make_jobs(30, benchmark.DEFAULT_SEED_BASE)
    revision = {
        "commit": "a" * 40,
        "script_sha256": "b" * 64,
        "head_script_sha256": "b" * 64,
        "head_blob_oid": "c" * 40,
        "worktree_blob_oid": "c" * 40,
        "tracked_diff_sha256": "d" * 64,
        "tracked_dirty_at_snapshot": False,
    }
    config = benchmark.configuration(
        30, benchmark.DEFAULT_SEED_BASE, jobs, {"panels": {}}, revision
    )
    assert config["evaluation"]["truth_counts_per_representation"] == {"B": 60, "C": 90}
    assert "independent seeds" in config["pairing_guardrail"]
    assert config["raw_retention"] == (
        "first positive/control tree sequence for every panel"
    )
    assert config["source_revision"]["commit"] == "a" * 40
    assert config["source_revision"]["script_sha256"] == "b" * 64
    assert config["canonical_training_contract"]["array_contracts"] == (
        benchmark.CANONICAL_ARRAY_CONTRACTS
    )


def test_pinned_source_hash_has_full_sha256_width():
    assert len(benchmark.HOMSAP_MODEL_SOURCE_SHA256) == 64
    int(benchmark.HOMSAP_MODEL_SOURCE_SHA256, 16)


@pytest.mark.skipif(
    importlib_util.find_spec("stdpopsim") is None,
    reason="the pinned stdpopsim catalog is exercised in the Azure environment",
)
def test_exact_catalog_events_and_controls_differ_only_at_focal_event():
    models, audit = benchmark.prepare_models()
    assert audit["versions"] == {
        "stdpopsim": "0.3.0",
        "msprime": "1.4.2",
        "tskit": "1.0.3",
    }
    assert audit["HomSap_demographic_models_source_sha256"] == (
        benchmark.HOMSAP_MODEL_SOURCE_SHA256
    )
    assert set(models) == {panel.panel_id for panel in benchmark.PANELS}
    for panel in benchmark.PANELS:
        current = audit["panels"][panel.panel_id]
        assert current["control_differs_only_at_focal_event"] is True
        assert current["population_order"] == list(panel.populations)
        if panel.event_type == "Admixture":
            assert current["positive_event"]["proportions"] == list(
                panel.positive_proportions
            )
            assert current["control_event"]["proportions"] == list(
                panel.control_proportions
            )
            assert sum(current["control_event"]["proportions"]) == 1.0
        else:
            assert current["positive_event"]["proportion"] == panel.positive_proportion
            assert current["control_event"]["proportion"] == 0.0


@pytest.mark.skipif(
    importlib_util.find_spec("stdpopsim") is None,
    reason="the pinned stdpopsim engine is exercised in the Azure environment",
)
def test_every_panel_condition_runs_tiny_actual_simulation_at_catalog_times():
    import stdpopsim

    models, audit = benchmark.prepare_models()
    engine = stdpopsim.get_engine("msprime")
    for panel in benchmark.PANELS:
        expected_times = audit["panels"][panel.panel_id]["sample_times_generations"]
        for condition in benchmark.CONDITIONS:
            model = models[panel.panel_id][condition]
            samples = model.get_sample_sets(
                {
                    name: benchmark.INDIVIDUALS_PER_POPULATION
                    for name in panel.populations
                },
                ploidy=2,
            )
            assert [sample.num_samples * sample.ploidy for sample in samples] == [
                200, 200, 200
            ]
            assert [sample.time for sample in samples] == [
                expected_times[name] for name in panel.populations
            ]
            tiny_contig = stdpopsim.Contig.basic_contig(
                length=1_000,
                mutation_rate=float(model.mutation_rate),
                recombination_rate=benchmark.RECOMBINATION_RATE,
                ploidy=2,
            )
            tree_sequence = engine.simulate(
                model,
                tiny_contig,
                {
                    name: benchmark.INDIVIDUALS_PER_POPULATION
                    for name in panel.populations
                },
                seed=12345,
            )
            assert tree_sequence.num_samples == 600
            assert tree_sequence.num_individuals == 300
            full_contig = benchmark.make_contig(model)
            assert full_contig.mutation_rate == model.mutation_rate
            assert full_contig.recombination_map.mean_rate == pytest.approx(
                benchmark.RECOMBINATION_RATE, abs=1e-30
            )
