from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib import util as importlib_util
import json

import numpy as np
import pytest

from scripts import stdpopsim_neanderthal_benchmark as benchmark


def _curve(offset: float = 0.0) -> np.ndarray:
    curve = np.zeros((len(benchmark.FULL_DEPTHS), len(benchmark.CURVE_COLUMNS)), dtype=np.float32)
    curve[:, 0] = benchmark.FULL_DEPTHS
    curve[:, 1:] = offset
    return curve


def _record(job: benchmark.BenchmarkJob, offset: float = 0.0) -> dict:
    curve = _curve(offset)
    return {
        **asdict(job),
        "engine_derived_ancestry_seed": job.engine_seed + 10,
        "engine_derived_mutation_seed": job.engine_seed + 11,
        "elapsed_seconds": 0.1,
        "simulation_audit": {"globally_polymorphic_loci": 10},
        "curve_sha256_float32": hashlib.sha256(
            curve.astype("<f4", copy=False).tobytes()
        ).hexdigest(),
        "curve": curve,
    }


def test_job_manifest_pairs_configuration_but_not_genealogies():
    jobs = benchmark.make_jobs(4, 1000)
    assert len(jobs) == 8
    assert len({job.job_id for job in jobs}) == 8
    assert len({job.engine_seed for job in jobs}) == 8
    for family_index in range(4):
        family = [job for job in jobs if job.family_index == family_index]
        assert {job.condition for job in family} == {"pulse", "control"}
        assert len({job.engine_seed for job in family}) == 2
        assert len({job.family_id for job in family}) == 1


@pytest.mark.parametrize("pairs,seed", [(0, 10), (1, 0), (2, 2**31 - 3)])
def test_job_manifest_rejects_invalid_size_or_seed(pairs, seed):
    with pytest.raises(ValueError):
        benchmark.make_jobs(pairs, seed)


def test_historical_gate_feature_contract_is_exactly_243d():
    table = np.stack([_curve(0.0), _curve(2.0)]).astype(float)
    features, audit = benchmark._gate_features(table)
    assert features.shape == (2, 243)
    assert audit["feature_dimension"] == 243
    assert len(audit["selected_depths"]) == 8
    assert audit["selected_depths"][0] == 3
    assert audit["selected_depths"][-1] == 199
    assert np.all(features[0] == 0)
    assert np.all(features[1] == 2)


def test_gate_rejects_short_primary_curve():
    table = np.stack([_curve()[:, :]])[:, : len(benchmark.PRIMARY_DEPTHS)]
    with pytest.raises(ValueError, match="full g=2..199"):
        benchmark._gate_features(table)


def test_checkpoint_round_trip_and_config_binding(tmp_path):
    jobs = benchmark.make_jobs(1, 1000)
    records = [_record(job, index / 10) for index, job in enumerate(jobs)]
    path = tmp_path / "features.npz"
    benchmark.save_checkpoint(path, records, "a" * 64)
    loaded = benchmark.load_checkpoint(path, "a" * 64, jobs)
    assert [record["job_id"] for record in loaded] == [job.job_id for job in jobs]
    assert np.array_equal(loaded[0]["curve"], records[0]["curve"])
    assert benchmark.checkpoint_audit(path, loaded, "a" * 64)[
        "complete_pulse_control_families"
    ] == 1
    with pytest.raises(RuntimeError, match="configuration changed"):
        benchmark.load_checkpoint(path, "b" * 64, jobs)


def test_checkpoint_rejects_duplicates(tmp_path):
    job = benchmark.make_jobs(1, 1000)[0]
    record = _record(job)
    with pytest.raises(RuntimeError, match="duplicate"):
        benchmark.save_checkpoint(tmp_path / "duplicate.npz", [record, record], "a" * 64)


def test_checkpoint_rejects_changed_manifest(tmp_path):
    jobs = benchmark.make_jobs(1, 1000)
    path = tmp_path / "features.npz"
    benchmark.save_checkpoint(path, [_record(jobs[0])], "a" * 64)
    changed_jobs = benchmark.make_jobs(1, 2000)
    with pytest.raises(RuntimeError, match="job manifest changed"):
        benchmark.load_checkpoint(path, "a" * 64, changed_jobs)


def test_wilson_interval_and_empty_contract():
    perfect = benchmark._wilson([True] * 30)
    assert perfect["successes"] == perfect["n"] == 30
    assert perfect["fraction"] == 1.0
    assert perfect["wilson_95"][0] == pytest.approx(0.8864866068)
    assert benchmark._wilson([]) == {
        "successes": 0,
        "n": 0,
        "fraction": None,
        "wilson_95": None,
    }


def test_canonical_json_is_sorted_compact_and_rejects_nan():
    assert benchmark._canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'
    with pytest.raises(ValueError):
        benchmark._canonical_json({"x": float("nan")})


@pytest.mark.skipif(
    importlib_util.find_spec("stdpopsim") is None,
    reason="stdpopsim is exercised in the pinned Azure environment",
)
def test_pinned_catalog_model_and_matched_control_are_exact():
    models, audit = benchmark.prepare_models()
    assert set(models) == {"pulse", "control"}
    assert audit["stdpopsim_version"] == "0.3.0"
    assert audit["population_order"] == ["YRI", "CEU", "NEA"]
    assert audit["dnnaic_true_direction"] == "C"
    assert audit["pulse"]["positive_event_count"] == 1298
    assert audit["pulse"]["positive_times_are_exact_856_through_2153"] is True
    assert audit["pulse"]["integrated_backward_CEU_to_NEA_hazard"] == pytest.approx(0.03)
    assert audit["pulse"]["forward_NEA_to_CEU_single_lineage_probability"] == pytest.approx(
        1 - np.exp(-0.03)
    )
    assert audit["control"]["positive_event_count"] == 0
    assert audit["control"]["integrated_backward_CEU_to_NEA_hazard"] == 0
    assert audit["normalized_pulse_matches_control"] is True
    assert audit["pulse"]["global_rate_events"] == audit["control"]["global_rate_events"]
    assert audit["pulse"]["mass_migrations"] == audit["control"]["mass_migrations"]


@pytest.mark.skipif(
    importlib_util.find_spec("stdpopsim") is None,
    reason="stdpopsim is exercised in the pinned Azure environment",
)
def test_contig_and_sampling_produce_200_gene_copies_per_population():
    import stdpopsim

    models, _audit = benchmark.prepare_models()
    contig = benchmark.make_contig()
    assert contig.length == 1_000_000
    assert contig.ploidy == 2
    assert contig.mutation_rate == 2e-8
    assert contig.recombination_map.mean_rate == pytest.approx(1.78e-8)
    samples = models["pulse"].get_sample_sets(
        {name: benchmark.INDIVIDUALS_PER_POPULATION for name in benchmark.POPULATIONS},
        ploidy=contig.ploidy,
    )
    assert [sample.num_samples for sample in samples] == [100, 100, 100]
    assert [sample.num_samples * sample.ploidy for sample in samples] == [200, 200, 200]
    engine = stdpopsim.get_engine("msprime")
    assert engine.simulate(
        models["pulse"],
        contig,
        {name: benchmark.INDIVIDUALS_PER_POPULATION for name in benchmark.POPULATIONS},
        seed=12345,
        dry_run=True,
    ) is None
    assert engine.simulate(
        models["control"],
        contig,
        {name: benchmark.INDIVIDUALS_PER_POPULATION for name in benchmark.POPULATIONS},
        seed=12346,
        dry_run=True,
    ) is None
