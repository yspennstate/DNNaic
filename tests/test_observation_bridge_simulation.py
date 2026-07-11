import copy
from pathlib import Path

import numpy as np
import pytest

from scripts import observation_bridge_simulation as bridge


SYNTHETIC_SAMPLE_SEED = 777


def _synthetic_genotypes(loci=300):
    genotypes = np.zeros((loci, 3, bridge.GENE_COPIES), dtype=np.int8)
    for locus in range(loci):
        for population in range(3):
            alternate = 30 + (locus + population) % 51
            genotypes[locus, population, :alternate] = 1
        if locus % 11 == 0:
            genotypes[locus, :, :5] = 2
    # A globally monomorphic site must be removed even by a zero-MAF view.
    genotypes[0] = 0
    # This site is globally polymorphic but monomorphic inside P1.
    genotypes[1, 0] = 0
    return genotypes


def _synthetic_counts(loci=300, sample_copies=bridge.GENE_COPIES):
    genotypes = _synthetic_genotypes(loci)
    indices = bridge.sample_index_subsets(SYNTHETIC_SAMPLE_SEED)[sample_copies]
    counts = bridge.genotype_counts(genotypes, indices)
    positions = np.linspace(1, bridge.SEQUENCE_LENGTH - 1, loci)
    ids = [f"synthetic-{index}" for index in range(loci)]
    return counts, positions, ids, bridge.sample_index_sha256(indices)


def test_job_manifest_is_deterministic_rate_shared_and_seed_separated():
    jobs, rates = bridge.make_jobs(5, 12345)
    again, again_rates = bridge.make_jobs(5, 12345)
    assert jobs == again
    assert np.array_equal(rates, again_rates)
    assert len(jobs) == 15
    for rate_index in range(5):
        current = [job for job in jobs if job.rate_index == rate_index]
        assert [job.label for job in current] == ["A", "B", "C"]
        assert len({job.rate.hex() for job in current}) == 1
        assert len({job.nuisance_profile_seed for job in current}) == 1
        assert len({job.observation_seed for job in current}) == 3
        assert all(job.rate.hex() in job.parent_genealogy_id for job in current)


def test_configuration_hash_covers_manifest_order_versions_and_sources():
    jobs, rates = bridge.make_jobs(3, 91)
    config = bridge.configuration(3, 91, jobs=jobs, rates=rates)
    assert len(config["job_manifest"]) == 9
    assert [row["view"] for row in config["ordered_view_seed_contract"]] == list(
        bridge.VIEW_SPECS
    )
    assert config["sequence_design"]["independent_contigs"] == bridge.CONTIG_COUNT
    assert "nested" in config["fixed_sample_contract"]["nesting"]
    assert set(config["dependency_versions"]) == {
        "msprime",
        "tskit",
        "numpy",
        "padze",
        "scikit-learn",
    }
    assert all(len(value) == 64 for value in config["semantic_source_sha256"].values())
    changed = copy.deepcopy(config)
    changed["job_manifest"][0]["rate_hex"] = float(1e-3).hex()
    assert bridge.configuration_sha256(config) != bridge.configuration_sha256(changed)


def test_seed_derivation_is_name_stable_and_nonzero_uint32():
    first = bridge.derived_seed(5, "sampling", "view-a")
    assert first == bridge.derived_seed(5, "sampling", "view-a")
    assert first != bridge.derived_seed(5, "sampling", "view-b")
    assert 1 <= first <= 2**32 - 1


def test_multiallelic_sampling_is_deterministic_and_conserves_counts():
    colors = np.array([100, 60, 30, 10])
    first = bridge._multivariate_sample(colors, 64, np.random.default_rng(7))
    second = bridge._multivariate_sample(colors, 64, np.random.default_rng(7))
    assert np.array_equal(first, second)
    assert first.sum() == 64
    assert np.all(first <= colors)
    counts, _, _, _ = _synthetic_counts(20, 32)
    missing, called = bridge.downsample_missingness(
        counts,
        [0.8, 0.9, 0.95],
        np.random.default_rng(8),
    )
    assert missing.shape == counts.shape
    assert np.all(called <= 32)
    assert np.all(missing <= counts)


def test_fixed_sample_indices_are_cross_locus_deterministic_and_nested():
    subsets = bridge.sample_index_subsets(123)
    again = bridge.sample_index_subsets(123)
    for copies in (32, 64, bridge.GENE_COPIES):
        assert np.array_equal(subsets[copies], again[copies])
        assert subsets[copies].shape == (3, copies)
        assert len(bridge.sample_index_sha256(subsets[copies])) == 64
        assert np.all((subsets[copies] >= 0) & (subsets[copies] < bridge.GENE_COPIES))
        assert all(len(np.unique(row)) == copies for row in subsets[copies])
    different = bridge.sample_index_subsets(124)
    assert not np.array_equal(subsets[32], different[32])
    assert not np.array_equal(subsets[64], different[64])
    for population in range(3):
        assert set(subsets[32][population]) < set(subsets[64][population])
    genotypes = _synthetic_genotypes(40)
    counts = bridge.genotype_counts(genotypes, subsets[64])
    counts32 = bridge.genotype_counts(genotypes, subsets[32])
    assert np.all(counts.sum(axis=2) == 64)
    assert np.all(counts32 <= counts)
    for population in range(3):
        selected = genotypes[:, population, subsets[64][population]]
        for allele in range(4):
            assert np.array_equal(
                counts[:, population, allele],
                np.sum(selected == allele, axis=1),
            )
    duplicate = subsets[32].copy()
    duplicate[0, 1] = duplicate[0, 0]
    with pytest.raises(ValueError, match="duplicate"):
        bridge.genotype_counts(genotypes, duplicate)
    invalid = genotypes.copy()
    invalid[0, 0, 0] = 4
    with pytest.raises(ValueError, match="0..3"):
        bridge.genotype_counts(invalid, subsets[32])


def test_views_keep_multiallelic_sites_cap_loci_and_share_missingness_profiles():
    counts, positions, ids, sample_hash = _synthetic_counts()
    complete, complete_ids, complete_audit = bridge.apply_view(
        counts,
        positions,
        ids,
        "complete_all_observed_alleles",
        base_sample_sha256=sample_hash,
        profile_seed=11,
        sampling_seed=12,
    )
    assert len(complete) == len(counts) - 1
    assert ids[0] not in complete_ids
    assert np.any(complete[:, :, 2] > 0)
    assert complete_audit["selected_count_matrix_sha256"]

    counts64, positions, ids, sample_hash64 = _synthetic_counts(sample_copies=64)
    capped, _, capped_audit = bridge.apply_view(
        counts64,
        positions,
        ids,
        "sample_64_cap_64_maf_01",
        base_sample_sha256=sample_hash64,
        profile_seed=13,
        sampling_seed=14,
    )
    assert len(capped) == 64
    assert capped_audit["eligible_before_cap"] > 64

    _, _, missing_a = bridge.apply_view(
        counts64,
        positions,
        ids,
        "sample_64_missingness_maf_01",
        base_sample_sha256=sample_hash64,
        profile_seed=99,
        sampling_seed=100,
    )
    _, _, missing_b = bridge.apply_view(
        counts64,
        positions,
        ids,
        "sample_64_missingness_maf_01",
        base_sample_sha256=sample_hash64,
        profile_seed=99,
        sampling_seed=101,
    )
    assert missing_a["call_rates"] == missing_b["call_rates"]
    assert missing_a["sampling_seed"] != missing_b["sampling_seed"]
    assert missing_a["base_sample_sha256"] == missing_b["base_sample_sha256"]

    within, _, _ = bridge.apply_view(
        counts64,
        positions,
        ids,
        "sample_64_within_each_population",
        base_sample_sha256=sample_hash64,
        profile_seed=15,
        sampling_seed=16,
    )
    assert len(within) < len(complete)
    same_size_audits = []
    for index, view in enumerate(
        name for name, spec in bridge.VIEW_SPECS.items() if spec["sample_copies"] == 64
    ):
        _, _, audit = bridge.apply_view(
            counts64,
            positions,
            ids,
            view,
            base_sample_sha256=sample_hash64,
            profile_seed=1_000 + index,
            sampling_seed=2_000 + index,
        )
        same_size_audits.append(audit)
    assert {audit["base_sample_sha256"] for audit in same_size_audits} == {sample_hash64}
    assert len({audit["input_count_matrix_sha256"] for audit in same_size_audits}) == 1


def _fake_record(job, view, view_index, *, usable=True):
    feature = np.zeros((len(bridge.DEPTHS), len(bridge.CURVE_COLUMNS)), dtype=np.float32)
    feature[:, 0] = bridge.DEPTHS
    sample_copies = int(bridge.VIEW_SPECS[view]["sample_copies"])
    input_digest = __import__("hashlib").sha256(
        f"{job.parent_genealogy_id}|{sample_copies}".encode()
    ).hexdigest()
    return {
        **job.__dict__,
        "view": view,
        "view_index": view_index,
        "source_audit": {"polymorphic_sites": 100},
        "view_audit": {
            "view": view,
            "specification": bridge.VIEW_SPECS[view],
            "profile_seed": bridge.derived_seed(job.nuisance_profile_seed, "profile", view),
            "sampling_seed": bridge.derived_seed(job.observation_seed, "sampling", view),
            "base_sample_sha256": bridge.sample_index_sha256(
                bridge.sample_index_subsets(job.nuisance_profile_seed)[
                    sample_copies
                ]
            ),
            "fixed_sample_copies_per_population": sample_copies,
            "input_count_matrix_sha256": input_digest,
            "usable": usable,
        },
        "feature": feature if usable else None,
        "invalid_reason": None if usable else "synthetic invalid view",
    }


def _fake_parent(job):
    return [
        _fake_record(job, view, index)
        for index, view in enumerate(bridge.VIEW_SPECS)
    ]


def test_checkpoint_roundtrip_validates_contract_and_refuses_incomplete_parent(tmp_path):
    jobs, rates = bridge.make_jobs(3, 317)
    config = bridge.configuration(3, 317, jobs=jobs, rates=rates)
    digest = bridge.configuration_sha256(config)
    path = tmp_path / "checkpoint.npz"
    records = _fake_parent(jobs[0])
    bridge.save_checkpoint(path, records, digest)
    loaded = bridge.load_checkpoint(path, digest, jobs)
    assert [bridge.record_key(row) for row in loaded] == [
        bridge.record_key(row) for row in sorted(records, key=bridge.record_key)
    ]
    assert all(np.array_equal(row["feature"][:, 0], bridge.DEPTHS) for row in loaded)
    with pytest.raises(RuntimeError, match="configuration changed"):
        bridge.load_checkpoint(path, "0" * 64, jobs)
    with pytest.raises(RuntimeError, match="incomplete parent"):
        bridge.save_checkpoint(tmp_path / "incomplete.npz", records[:-1], digest)
    records[0]["view_audit"]["base_sample_sha256"] = "0" * 64
    bridge.save_checkpoint(path, records, digest)
    with pytest.raises(RuntimeError, match="view metadata changed"):
        bridge.load_checkpoint(path, digest, jobs)
    records = _fake_parent(jobs[0])
    records[0]["view_audit"]["fixed_sample_copies_per_population"] = 64
    bridge.save_checkpoint(path, records, digest)
    with pytest.raises(RuntimeError, match="view metadata changed"):
        bridge.load_checkpoint(path, digest, jobs)


def test_validity_audit_excludes_the_entire_abc_rate_family_after_one_invalid_view():
    jobs, _ = bridge.make_jobs(3, 444)
    records = [record for job in jobs[:6] for record in _fake_parent(job)]
    records[-1]["feature"] = None
    records[-1]["view_audit"]["usable"] = False
    records[-1]["invalid_reason"] = "synthetic"
    audit = bridge.validity_audit(records)
    assert audit["overall"]["usable_fraction"] == 59 / 60
    assert audit["complete_parents"] == 5
    assert audit["complete_parent_fraction"] == 5 / 6
    assert audit["complete_rate_families"] == 1
    assert audit["complete_rate_family_fraction"] == 0.5
    assert audit["complete_rate_family_ids"] == [jobs[0].rate_family_id]
    assert set(audit["complete_family_parent_ids"]) == {
        job.parent_genealogy_id for job in jobs[:3]
    }
    assert audit["by_class"]["C"]["complete_parent_fraction"] == 0.5
    assert audit["by_class"]["A"]["complete_parent_fraction"] == 1.0


def test_single_writer_lock_rejects_a_second_runner(tmp_path):
    with bridge.single_writer_lock(tmp_path) as path:
        assert path.exists()
        with pytest.raises(RuntimeError, match="already locked"):
            with bridge.single_writer_lock(tmp_path):
                pass
    assert path.exists()
    with bridge.single_writer_lock(tmp_path):
        pass


def test_analysis_preflight_rejects_too_few_complete_rate_families():
    jobs, _ = bridge.make_jobs(3, 441)
    records = [record for job in jobs[:3] for record in _fake_parent(job)]
    with pytest.raises(RuntimeError, match="too few complete A/B/C rate families"):
        bridge.analyze_records(
            records,
            seeds=(0,),
            C_grid=(0.1, 1.0),
            outer_splits=3,
            inner_splits=3,
            natural_paths=(),
        )


def test_bridge_adjudication_has_absolute_accuracy_and_attrition_floors():
    def cv(value):
        return {"per_repeat_metrics": [{
            "appreciable": {"accuracy": value},
            "appreciable_equal_rate_family": {"balanced_accuracy": value},
        }]}

    def factor(value):
        return {
            "balanced_accuracy": value,
            "appreciable": {"n": 9, "balanced_accuracy": value},
        }

    def natural(a, b):
        return {
            "coverage": {
                "result_file_bundle_balanced_descriptive": {
                    "by_bundle": {
                        "one": {"median_rms_z": a},
                        "two": {"median_rms_z": b},
                    }
                }
            }
        }

    variants = {
        "raw_all": {
            "genealogy_cv": cv(0.96),
            "genealogy_cv_by_observation_view": {
                "control": {"appreciable": {"n": 9, "accuracy": 0.96}}
            },
            "rate_family_cv": cv(0.92),
            "leave_one_observation_factor_out": {"maf": factor(0.75)},
            "natural_transfer": natural(20.0, 30.0),
        },
        "orbit_composition_mean_variance": {
            "genealogy_cv": cv(0.95),
            "genealogy_cv_by_observation_view": {
                "control": {"appreciable": {"n": 9, "accuracy": 0.95}}
            },
            "rate_family_cv": cv(0.90),
            "leave_one_observation_factor_out": {"maf": factor(0.74)},
            "natural_transfer": natural(10.0, 20.0),
        },
    }
    validity = {
        "overall": {"usable_fraction": 0.95},
        "by_class": {
            label: {"usable_fraction": 0.94, "complete_parent_fraction": 0.90}
            for label in bridge.CLASSES
        },
        "by_view": {view: {"usable_fraction": 0.90} for view in bridge.VIEW_SPECS},
        "complete_parent_fraction": 0.85,
        "complete_rate_family_fraction": 0.85,
    }
    result = bridge.adjudicate_bridge(validity, variants)
    assert result["all_exploratory_bridge_thresholds_pass"] is True
    low_family = copy.deepcopy(validity)
    low_family["complete_rate_family_fraction"] = 0.79
    assert bridge.adjudicate_bridge(low_family, variants)[
        "all_exploratory_bridge_thresholds_pass"
    ] is False
    low_class = copy.deepcopy(validity)
    low_class["by_class"]["C"]["complete_parent_fraction"] = 0.79
    assert bridge.adjudicate_bridge(low_class, variants)[
        "all_exploratory_bridge_thresholds_pass"
    ] is False
    variants["orbit_composition_mean_variance"]["genealogy_cv"] = cv(0.40)
    assert bridge.adjudicate_bridge(validity, variants)[
        "all_exploratory_bridge_thresholds_pass"
    ] is False
