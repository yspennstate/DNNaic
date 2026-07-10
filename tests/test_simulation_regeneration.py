"""Publication-contract tests for the exact DNNaic regeneration path."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import extract_padze_from_trees as extract  # noqa: E402
import simulate_demography as simulation  # noqa: E402


CANONICAL_MANIFEST_SHA256 = (
    "a61207d25ec8e59d1b190b036bcd887d73628e880e78d3f6bb833c17b4f8f28c"
)
LEGACY_COUPLED_MANIFEST_SHA256 = (
    "f69ff60941da783f94c8a569624f77ab4986455ac44a8ecc14e3c0236a63b0c0"
)


def test_canonical_manifest_counts_shared_rates_labels_and_seeds():
    jobs = simulation.build_jobs()
    config = simulation.study_config()
    document = simulation.manifest_document(jobs, config)

    assert len(jobs) == 3_200
    assert Counter(job.design for job in jobs) == {
        "fixed": 1_200,
        "continuous": 1_500,
        "control": 500,
    }
    assert Counter(job.case for job in jobs) == {
        "A": 900,
        "B": 900,
        "C": 900,
        "D": 500,
    }

    for case in "ABC":
        fixed = [job for job in jobs if job.design == "fixed" and job.case == case]
        assert Counter(job.rate for job in fixed) == {
            rate: 100 for rate in simulation.DISCRETE_RATES
        }

    # Every continuous rate index is one exact floating-point draw reused five
    # times in each of A, B, and C (not independently redrawn by class).
    continuous = defaultdict(list)
    for job in jobs:
        if job.design == "continuous":
            continuous[job.rate_index].append(job)
    assert set(continuous) == set(range(100))
    for rate_jobs in continuous.values():
        assert Counter(job.case for job in rate_jobs) == {"A": 5, "B": 5, "C": 5}
        assert len({job.rate.hex() for job in rate_jobs}) == 1

    controls = [job for job in jobs if job.design == "control"]
    assert all(job.case == "D" and job.rate == 0.0 for job in controls)

    assert simulation.FORWARD_FLOW == {
        "A": ("P1", "P2"),
        "B": ("P2", "P3"),
        "C": ("P3", "P2"),
        "D": (None, None),
    }
    assert simulation.BACKWARD_MIGRATION == {
        "A": ("P2", "P1"),
        "B": ("P3", "P2"),
        "C": ("P2", "P3"),
        "D": (None, None),
    }
    for case in "ABC":
        assert simulation.BACKWARD_MIGRATION[case] == tuple(
            reversed(simulation.FORWARD_FLOW[case])
        )

    assert [job.ancestry_seed for job in jobs] == list(range(70_001, 73_201))
    assert [job.mutation_seed for job in jobs] == list(range(170_001, 173_201))
    assert {job.ancestry_seed for job in jobs}.isdisjoint(
        {job.mutation_seed for job in jobs}
    )
    assert jobs == simulation.build_jobs()
    assert jobs[0].group == "A|cont|r001_3.164327e-04|rep00"
    assert jobs[-1].group == "C|cont|r033_2.851068e-04|rep01"
    assert document["manifest_hash"] == CANONICAL_MANIFEST_SHA256

    demography = config["demography"]
    assert demography["gene_copies_per_population"] == 200
    assert demography["ploidy"] == 1
    assert demography["haploid_coalescent_population_size"] == 10_000
    assert demography["diploid_equivalent_effective_size"] == 5_000
    assert demography["ancestry_model"] == "StandardCoalescent"
    assert demography["mutation_model"] == "JC69"
    assert demography["discrete_genome"] is True
    assert demography["experimental_populations"] == ["P1", "P2", "P3"]


def test_legacy_manifest_explicitly_records_historical_coupled_seed_policy():
    jobs = simulation.build_jobs(legacy_coupled_seeds=True)
    config = simulation.study_config(legacy_coupled_seeds=True)
    document = simulation.manifest_document(jobs, config)
    assert all(job.ancestry_seed == job.mutation_seed for job in jobs)
    assert config["seeds"]["ancestry_mutation_policy"] == "legacy_same_integer"
    assert document["manifest_hash"] == LEGACY_COUPLED_MANIFEST_SHA256


@pytest.mark.parametrize(
    ("field", "value"), (("class", "C"), ("ancestry_seed", 999_999))
)
def test_manifest_hash_detects_label_or_seed_tampering(tmp_path, field, value):
    document = simulation.manifest_document(
        simulation.build_jobs(), simulation.study_config()
    )
    document["jobs"][0][field] = value
    path = tmp_path / "simulation_manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        extract.load_and_verify_manifest(path)


def test_written_manifest_records_runtime_without_changing_design_hash(tmp_path):
    jobs = simulation.build_jobs()
    config = simulation.study_config()
    document = simulation.write_manifest(tmp_path, jobs, config)
    assert document["manifest_hash"] == CANONICAL_MANIFEST_SHA256
    assert set(("python", "platform", "msprime", "tskit", "numpy", "padze")) <= set(
        document["runtime"]
    )
    loaded = extract.load_and_verify_manifest(tmp_path / "simulation_manifest.json")
    assert loaded["runtime"] == document["runtime"]


def test_msprime_demography_uses_the_declared_backward_mapping():
    pytest.importorskip("msprime")
    rate = 2.5e-4
    for case in "ABC":
        source, destination = simulation.BACKWARD_MIGRATION[case]
        demography = simulation.build_demography(rate, source, destination)
        population_id = {
            population.name: index
            for index, population in enumerate(demography.populations)
        }
        matrix = np.asarray(demography.migration_matrix)
        assert matrix[population_id[source], population_id[destination]] == rate
        assert np.count_nonzero(matrix) == 1


def _tiny_named_tree_sequence():
    tskit = pytest.importorskip("tskit")
    tables = tskit.TableCollection(sequence_length=1.0)
    tables.populations.metadata_schema = tskit.MetadataSchema.permissive_json()
    population_ids = [
        tables.populations.add_row(metadata={"name": name})
        for name in ("P1", "P2", "P3")
    ]
    sample_nodes = []
    for population_id in population_ids:
        sample_nodes.append(
            [
                tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0,
                                     population=population_id)
                for _ in range(2)
            ]
        )
    root = tables.nodes.add_row(time=1)
    for node in np.asarray(sample_nodes).reshape(-1):
        tables.edges.add_row(0, 1, parent=root, child=int(node))
    first_site = tables.sites.add_row(position=0.25, ancestral_state="0")
    tables.mutations.add_row(
        site=first_site, node=sample_nodes[0][0], derived_state="1"
    )
    second_site = tables.sites.add_row(position=0.75, ancestral_state="0")
    tables.mutations.add_row(
        site=second_site, node=sample_nodes[1][0], derived_state="1"
    )
    tables.sort()
    return tables.tree_sequence()


def test_tree_sequence_to_padze_extractor_has_28_column_contract():
    tree_sequence = _tiny_named_tree_sequence()
    loci = extract.ts_to_loci(tree_sequence, expected_gene_copies=2)
    assert loci.populations == ["P1", "P2", "P3"]
    assert loci.metadata.n_loci_read == 2
    assert loci.metadata.n_loci_kept == 2
    np.testing.assert_array_equal(loci.sample_sizes, np.full((2, 3), 2))
    np.testing.assert_array_equal(
        loci.count_matrices[0], np.array([[1, 1], [2, 0], [2, 0]])
    )

    matrix, columns, n_loci = extract.tree_to_feature_matrix(
        tree_sequence, depths=[2], expected_gene_copies=2
    )
    assert matrix.shape == (1, 28)
    assert columns[0] == "g"
    assert columns[-1] == "pihat_23_se"
    assert matrix[0, 0] == 2
    assert n_loci == 2
