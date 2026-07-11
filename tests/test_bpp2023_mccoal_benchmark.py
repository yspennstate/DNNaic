from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from scripts import bpp2023_mccoal_benchmark as benchmark


def _small_alignment() -> bytes:
    """Two tiny BPP loci: one four-allelic site and one biallelic site."""
    return (
        "7 5\n"
        "q1^Q AAAAA\n"
        "q2^Q ACAAA\n"
        "r1^R AAAAA\n"
        "r2^R AGAAA\n"
        "d1^D AAAAA\n"
        "d2^D ATAAA\n"
        "s1^S TTTTT\n"
        "\n"
        "7 5\n"
        "q1^Q AAAAA\n"
        "q2^Q AAAAA\n"
        "r1^R AAAAA\n"
        "r2^R AAAAA\n"
        "d1^D AAAAA\n"
        "d2^D AACAA\n"
        "s1^S AAAAA\n"
    ).encode("ascii")


def _parsed_small_alignment(tmp_path: Path):
    path = tmp_path / "Seq.txt"
    path.write_bytes(_small_alignment())
    return benchmark.parse_bpp_alignments(
        path,
        locus_count=2,
        locus_length=5,
        gene_copies=2,
        outgroup_copies=1,
    )


def test_manifest_is_balanced_unique_and_starts_at_published_setting():
    jobs = benchmark.make_jobs()
    assert len(jobs) == 90
    assert len({job.job_id for job in jobs}) == 90
    assert len({job.seed for job in jobs}) == 90
    assert [(job.label, job.family_positive_phi, job.scale) for job in jobs[:3]] == [
        ("B", 0.106, 1.0),
        ("C", 0.106, 1.0),
        ("D", 0.106, 1.0),
    ]
    for family_index in range(30):
        family = [job for job in jobs if job.family_index == family_index]
        assert {job.label for job in family} == {"B", "C", "D"}
        assert len({job.family_id for job in family}) == 1
        assert len({job.seed for job in family}) == 3
    assert [benchmark.job_effective_phi(job) for job in jobs[:3]] == [
        0.106,
        0.106,
        0.0,
    ]
    assert benchmark.job_payload(jobs[2])["family_positive_phi"] == 0.106
    assert benchmark.job_payload(jobs[2])["effective_phi"] == 0.0


def test_generated_published_setting_preserves_official_asymmetric_topologies():
    # These are the official MCcoal asymmetric controls with insignificant
    # whitespace and trailing-zero formatting removed.
    expected_outflow = (
        "((((m[&phi=0.106,tau-parent=no]:0.000307,Q #0.000664)"
        "l:0.000307 #0.001568,R #0.000344)b:0.000389 #0.002429,"
        "(D #0.003314)m[&phi=0.894,tau-parent=yes]:0.000307 #0.000407)"
        "f:0.000731 #0.00093,S #0.000866)h:0.003423 #0.01101;"
    )
    expected_inflow = (
        "((((Q #0.000664)l[&phi=0.894,tau-parent=yes]:0.000307 #0.001568,"
        "R #0.000344)b:0.000389 #0.002429,"
        "(l[&phi=0.106,tau-parent=no]:0.000307,D #0.003314)"
        "m:0.000307 #0.000407)f:0.000731 #0.00093,"
        "S #0.000866)h:0.003423 #0.01101;"
    )
    assert benchmark.network_newick("B", 0.106, 1.0) == expected_outflow
    assert benchmark.network_newick("C", 0.106, 1.0) == expected_inflow
    assert benchmark.network_newick("D", 0.106, 1.0) == benchmark.network_newick(
        "C", 0.0, 1.0
    )
    assert "phi=0,tau-parent=no" in benchmark.network_newick("D", 0.106, 1.0)
    assert "phi=1,tau-parent=yes" in benchmark.network_newick("D", 0.106, 1.0)


def test_direction_mapping_and_control_contract_are_frozen():
    first = benchmark.make_jobs()[0]
    control = benchmark.control_text(first)
    assert benchmark.DNNAIC_MAPPING == {"P1": "R", "P2": "Q", "P3": "D"}
    assert "species&tree = 4 Q R D S" in control
    assert "  200 200 200 1" in control
    assert "loci&length = 500 500" in control
    assert "model = 0" in control
    assert "treefile" not in control.lower()
    assert "modelparafile" not in control.lower()
    assert control.startswith(f"seed = {first.seed}\n")
    assert benchmark.OFFICIAL_CONTROL_SHA256 == {
        "MCcoal.outflow-asym.ctl": (
            "9577804bee2467cb4ba3070a454b570c6500353ef2346822298b97fb8383b4de"
        ),
        "MCcoal.inflow-asym.ctl": (
            "c71d4a2a061eda75db6fc906cef247648059d3d1ccdba9d172b44d497c73744c"
        ),
    }
    assert all(len(value) == 64 for value in benchmark.OFFICIAL_CONTROL_SHA256.values())
    config = benchmark.configuration(benchmark.make_jobs(), {})
    assert "paper's inflow-asymmetric phi=0" in config["null_provenance"]
    assert "independently seeded" in config["stochastic_design"]


def test_small_alignment_parser_counts_only_triplet_polymorphism(tmp_path):
    counts, blocks, positions, audit = _parsed_small_alignment(tmp_path)
    assert counts.shape == (2, 3, 4)
    assert blocks.tolist() == [0, 1]
    assert positions.tolist() == [2, 3]
    assert audit["total_source_sites"] == 10
    assert audit["invariant_in_dnnaic_triplet"] == 8
    assert audit["biallelic_in_dnnaic_triplet"] == 1
    assert audit["multiallelic_in_dnnaic_triplet"] == 1
    assert audit["retained_polymorphic_sites"] == 2
    # Population order is P1=R, P2=Q, P3=D; alleles are A,C,G,T.
    assert counts[0].tolist() == [
        [1, 0, 1, 0],
        [1, 1, 0, 0],
        [1, 0, 0, 1],
    ]
    assert counts[1].tolist() == [
        [2, 0, 0, 0],
        [2, 0, 0, 0],
        [1, 1, 0, 0],
    ]
    assert audit["raw_seq_sha256"] == hashlib.sha256(_small_alignment()).hexdigest()


def test_parser_rejects_trailing_content_and_invalid_dimensions(tmp_path):
    path = tmp_path / "Seq.txt"
    path.write_bytes(_small_alignment() + b"unexpected\n")
    with pytest.raises(RuntimeError, match="after the final locus"):
        benchmark.parse_bpp_alignments(
            path,
            locus_count=2,
            locus_length=5,
            gene_copies=2,
            outgroup_copies=1,
        )
    with pytest.raises(ValueError, match="positive"):
        benchmark.parse_bpp_alignments(path, locus_count=0)


def test_imap_parser_requires_all_four_populations(tmp_path):
    path = tmp_path / "Imap.txt"
    path.write_text("Q Q\nR R\nD D\nS S\n", encoding="ascii")
    audit = benchmark.parse_imap(path)
    assert len(audit["rows"]) == 4
    path.write_text("Q Q\nR R\nD D\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="contract changed"):
        benchmark.parse_imap(path)


def test_count_checkpoint_round_trip_and_ledger_binding(tmp_path):
    counts, blocks, positions, parser_audit = _parsed_small_alignment(tmp_path)
    job = benchmark.make_jobs()[0]
    path = tmp_path / "counts" / f"{job.job_id}.npz"
    metadata = {"parser_audit": parser_audit, "imap_audit": {"rows": []}}
    benchmark.save_count_file(
        path, job, "a" * 64, counts, blocks, positions, metadata
    )
    loaded = benchmark.load_count_file(path, job, "a" * 64)
    assert np.array_equal(loaded[0], counts)
    assert np.array_equal(loaded[1], blocks)
    assert np.array_equal(loaded[2], positions)
    assert loaded[3] == metadata
    with pytest.raises(RuntimeError, match="configuration changed"):
        benchmark.load_count_file(path, job, "b" * 64)


def test_curve_checkpoint_round_trip_requires_bound_count_file(tmp_path):
    counts, blocks, positions, parser_audit = _parsed_small_alignment(tmp_path)
    job = benchmark.make_jobs()[0]
    count_path = tmp_path / "counts" / f"{job.job_id}.npz"
    benchmark.save_count_file(
        count_path,
        job,
        "a" * 64,
        counts,
        blocks,
        positions,
        {"parser_audit": parser_audit},
    )
    curve = np.zeros((198, 28), dtype=np.float32)
    curve[:, 0] = benchmark.stdbench.FULL_DEPTHS
    record = {
        **benchmark.job_payload(job),
        "count_file": count_path.relative_to(tmp_path).as_posix(),
        "count_file_bytes": count_path.stat().st_size,
        "count_file_sha256": benchmark.structured.sha256_file(count_path),
        "curve_sha256_float32": benchmark.stdbench._sha256_array(
            curve.astype("<f4", copy=False)
        ),
        "curve": curve,
    }
    checkpoint = tmp_path / "features.npz"
    benchmark.save_checkpoint(checkpoint, [record], "a" * 64)
    loaded = benchmark.load_checkpoint(
        checkpoint, "a" * 64, benchmark.make_jobs(), tmp_path
    )
    assert len(loaded) == 1
    assert np.array_equal(loaded[0]["curve"], curve)
    audit = benchmark.checkpoint_audit(checkpoint, loaded, "a" * 64)
    assert audit["records"] == 1
    assert audit["stored_curve_shape"] == [1, 198, 28]
    selection = benchmark.record_selection_audit(loaded)
    assert selection["records"] == 1
    with pytest.raises(RuntimeError, match="different records"):
        benchmark.checkpoint_audit(checkpoint, [], "a" * 64)
    with pytest.raises(RuntimeError, match="configuration changed"):
        benchmark.load_checkpoint(
            checkpoint, "b" * 64, benchmark.make_jobs(), tmp_path
        )


def test_safe_cleanup_cannot_escape_single_job_directory(tmp_path):
    root = tmp_path / "work"
    valid = root / "job-example"
    valid.mkdir(parents=True)
    benchmark._safe_remove_job_directory(valid, root)
    assert not valid.exists()
    nested = root / "job-example" / "nested"
    nested.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="unexpected"):
        benchmark._safe_remove_job_directory(nested, root)
    outside = tmp_path / "job-outside"
    outside.mkdir()
    with pytest.raises(RuntimeError, match="outside"):
        benchmark._safe_remove_job_directory(outside, root)
    assert outside.is_dir()


@pytest.mark.parametrize(
    "label,phi,scale",
    [("A", 0.1, 1.0), ("B", -0.1, 1.0), ("C", 1.1, 1.0), ("D", 0.1, 0.0)],
)
def test_network_rejects_invalid_parameters(label, phi, scale):
    with pytest.raises(ValueError):
        benchmark.network_newick(label, phi, scale)
