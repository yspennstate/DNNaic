import copy
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import structured_transfer_pilot as pilot


def _curve_table(replicates=2, depths=4):
    table = np.zeros((replicates, depths, 28), dtype=float)
    table[:, :, 0] = np.arange(2, depths + 2)
    values = table[:, :, 1:].reshape(replicates, depths, 9, 3)
    for replicate in range(replicates):
        for depth in range(depths):
            for block in range(9):
                # Alpha means have the structural one-allele baseline.
                values[replicate, depth, block, 0] = (
                    1.0 + 0.05 * (block + 1) + 0.01 * depth + 0.001 * replicate
                    if block < 3
                    else 0.02 * (block + 1) + 0.005 * depth + 0.001 * replicate
                )
                values[replicate, depth, block, 1] = (
                    0.01 * (block + 1) + 0.002 * depth + 0.0005 * replicate
                )
                values[replicate, depth, block, 2] = (
                    0.005 * (block + 1) + 0.001 * depth + 0.0002 * replicate
                )
    return table


def test_structured_representations_have_exact_dimensions_and_columns():
    table = _curve_table()
    expected = {
        "raw_all": 54,
        "raw_mean_variance": 36,
        "orbit_composition_mean_variance": 36,
    }
    for name, dimension in expected.items():
        features = pilot.representation_features(table, name)
        assert features.shape == (2, dimension)
        assert np.isfinite(features).all()
        assert len(pilot.representation_columns(name)) == dimension


def test_triple_composition_is_invariant_to_common_scale():
    original = _curve_table(replicates=1)
    scaled = original.copy()
    values = scaled[:, :, 1:].reshape(1, 4, 9, 3)
    source = original[:, :, 1:].reshape(1, 4, 9, 3)
    scales = (7.0, 0.25, 3.5)
    for orbit_index, start in enumerate((0, 3, 6)):
        scale = scales[orbit_index]
        for moment in (0, 1):
            current = source[:, :, start : start + 3, moment]
            if orbit_index == 0 and moment == 0:
                values[:, :, start : start + 3, moment] = 1.0 + scale * (current - 1.0)
            else:
                values[:, :, start : start + 3, moment] = scale * current
    structured_original = pilot.orbit_composition_summary(original, (0, 1))
    structured_scaled = pilot.orbit_composition_summary(scaled, (0, 1))
    assert structured_scaled == pytest.approx(structured_original, rel=0, abs=1e-14)
    assert not np.allclose(
        pilot.raw_summary(original, (0, 1)),
        pilot.raw_summary(scaled, (0, 1)),
    )


def test_zero_orbits_map_to_finite_zero_centered_compositions():
    table = _curve_table(replicates=1)
    values = table[:, :, 1:].reshape(1, 4, 9, 3)
    values[:, :, 3:9, :] = 0.0
    features = pilot.orbit_composition_summary(table, (0, 1))
    assert np.isfinite(features).all()
    # Private and pair-private orbits are uniform fallbacks and therefore centered zero.
    assert np.count_nonzero(np.abs(features) > 0) <= 12


def test_composition_rejects_materially_negative_coordinates_and_labels_pairs():
    table = _curve_table(replicates=1)
    values = table[:, :, 1:].reshape(1, 4, 9, 3)
    values[0, 0, 3, 0] = -1e-4
    with pytest.raises(ValueError, match="must be nonnegative"):
        pilot.orbit_composition_summary(table, (0, 1))
    columns = pilot.representation_columns("orbit_composition_mean_variance")
    assert "depth_mean__pair_private_12_mean" in columns
    assert "depth_mean__pair_private_13_mean" in columns
    assert "depth_mean__pair_private_23_mean" in columns


def test_curve_validation_rejects_shape_depth_and_nonfinite_drift():
    table = _curve_table()
    with pytest.raises(ValueError, match="28"):
        pilot.validate_curve_table(table[:, :, :-1])
    broken = table.copy()
    broken[1, 2, 0] += 1
    with pytest.raises(ValueError, match="share one depth grid"):
        pilot.validate_curve_table(broken)
    broken = table.copy()
    broken[0, 0, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pilot.validate_curve_table(broken)


def test_rate_family_ids_block_shared_exact_rates_across_directions():
    designs = np.array(["continuous"] * 6 + ["fixed"] * 3)
    rates = np.array([0.123] * 3 + [0.456] * 3 + [0.123] * 3)
    families = pilot.rate_family_ids(designs, rates)
    assert len(np.unique(families)) == 3
    assert len(set(families[:3])) == 1
    assert len(set(families[3:6])) == 1
    assert families[0] != families[6]
    assert float(0.123).hex() in families[0]


def test_grouped_folds_are_complete_balanced_and_block_disjoint():
    labels = np.tile(np.arange(3), 12)
    blocks = np.repeat([f"domain-{index}" for index in range(12)], 3)
    folds = pilot.grouped_folds(labels, blocks, n_splits=4, seed=7)
    seen = np.zeros(len(labels), dtype=int)
    for train, test in folds:
        assert not set(blocks[train]) & set(blocks[test])
        assert set(labels[train]) == {0, 1, 2}
        assert set(labels[test]) == {0, 1, 2}
        seen[test] += 1
    assert np.array_equal(seen, np.ones(len(labels), dtype=int))


def test_nested_oof_saves_auditable_disjoint_ledgers():
    rng = np.random.default_rng(9)
    labels = np.tile(np.arange(3), 12)
    blocks = np.repeat([f"rate-{index}" for index in range(12)], 3)
    groups = np.array([f"genealogy-{index}" for index in range(len(labels))])
    features = np.eye(3)[labels] + rng.normal(0, 0.05, size=(len(labels), 3))
    rates = np.repeat(np.linspace(1e-5, 3e-4, 12), 3)
    designs = np.array(["continuous"] * len(labels))
    result = pilot.nested_oof(
        features,
        labels,
        rates,
        designs,
        groups,
        blocks,
        outer_name="synthetic rate block",
        seeds=(0,),
        C_grid=(0.1, 1.0),
        outer_splits=3,
        inner_splits=3,
        evaluation_families=blocks,
    )
    assert result["mean_probability_metrics"]["overall"]["accuracy"] == 1.0
    assert len(result["oof_predictions"]) == len(labels)
    assert len(result["oof_predictions_sha256"]) == 64
    assert len(result["per_repeat_oof"]) == 1
    assert len(result["per_repeat_oof"][0]["outer_fold"]) == len(labels)
    assert len(result["per_repeat_oof"][0]["payload_sha256"]) == 64
    assert len(result["per_repeat_oof_sha256"]) == 64
    assert result["per_repeat_metrics"][0][
        "appreciable_equal_rate_family"
    ]["n_families"] > 0
    assert len(result["crossfit_support_reference"]["folds"]) == 3
    pilot.validate_nested_oof_payload(
        result,
        labels,
        rates,
        designs,
        groups,
        blocks,
        evaluation_families=blocks,
    )
    for fold in result["fold_ledger"][0]["folds"]:
        assert fold["block_overlap"] is False
        assert fold["scaler_fit_rows"] == fold["train_rows"]
        assert fold["model_fit_rows"] == fold["train_rows"]
        assert set(fold["test_class_counts"].values()) == {4}
        for inner_fold in fold["inner_selection"]["fold_ledger"]:
            assert inner_fold["block_overlap"] is False
            assert inner_fold["train_block_sha256"] != inner_fold["test_block_sha256"]

    tampered = copy.deepcopy(result)
    tampered["per_repeat_oof"][0]["probability"][0][0] += 0.01
    with pytest.raises(RuntimeError, match="payload hash changed"):
        pilot.validate_nested_oof_payload(
            tampered, labels, rates, designs, groups, blocks,
            evaluation_families=blocks,
        )
    tampered = copy.deepcopy(result)
    tampered["per_repeat_metrics"][0]["overall"]["accuracy"] = 0.0
    with pytest.raises(RuntimeError, match="metrics do not match"):
        pilot.validate_nested_oof_payload(
            tampered, labels, rates, designs, groups, blocks,
            evaluation_families=blocks,
        )
    tampered = copy.deepcopy(result)
    payload = tampered["per_repeat_oof"][0]
    payload["outer_fold"][0] = (payload["outer_fold"][0] + 1) % 3
    without_hash = {key: value for key, value in payload.items() if key != "payload_sha256"}
    payload["payload_sha256"] = pilot.hashlib.sha256(
        pilot._canonical_json(without_hash)
    ).hexdigest()
    tampered["per_repeat_oof_sha256"] = pilot.hashlib.sha256(
        pilot._canonical_json([payload["payload_sha256"]])
    ).hexdigest()
    with pytest.raises(RuntimeError, match="outer block crosses folds"):
        pilot.validate_nested_oof_payload(
            tampered, labels, rates, designs, groups, blocks,
            evaluation_families=blocks,
        )


def test_equal_rate_family_metric_does_not_weight_family_size():
    labels = np.concatenate([np.arange(3), np.tile(np.arange(3), 10)])
    prediction = np.concatenate([np.arange(3), np.tile(np.array([1, 2, 0]), 10)])
    probability = np.eye(3)[prediction] * 0.98 + 0.01
    probability /= probability.sum(axis=1, keepdims=True)
    rates = np.full(len(labels), pilot.APPRECIABLE)
    designs = np.array(["fixed"] * 3 + ["continuous"] * 30)
    families = np.array(["small"] * 3 + ["large"] * 30)
    summary = pilot._equal_rate_family_summary(
        labels, probability, rates, designs, families
    )
    assert summary["n_families"] == 2
    assert summary["balanced_accuracy"] == 0.5
    assert np.mean(prediction == labels) == pytest.approx(3 / 33)
    assert summary["by_design"]["fixed"]["balanced_accuracy"] == 1.0
    assert summary["by_design"]["continuous"]["balanced_accuracy"] == 0.0


def test_C_grid_validation_rejects_nonfinite_duplicates_and_unsorted_values():
    assert pilot.validate_C_grid([0.01, 0.1, 1.0]) == (0.01, 0.1, 1.0)
    for values in ([0.1, np.nan], [0.1, np.inf], [0.1, 0.1], [1.0, 0.1], [0.0]):
        with pytest.raises(ValueError):
            pilot.validate_C_grid(values)


def test_compute_gate_hard_aborts_on_distress(tmp_path, monkeypatch):
    monkeypatch.delenv(pilot.STOPPED_TRADING_AUTH_ENV, raising=False)
    path = tmp_path / "compute_state.json"
    path.write_text(
        json.dumps({"status": "distress", "mode": "throttle", "reasons": ["pressure"]}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="aborting before array load"):
        pilot.compute_gate(path)
    path.write_text(
        json.dumps({"status": "ok", "mode": "owner_present_polite", "reasons": []}),
        encoding="utf-8",
    )
    audit = pilot.compute_gate(path)
    assert audit["decision"] == "proceed_single_thread_below_normal"


def test_compute_gate_fails_closed_on_missing_unknown_and_stale_state(tmp_path):
    path = tmp_path / "compute_state.json"
    with pytest.raises(RuntimeError, match="unavailable"):
        pilot.compute_gate(path)
    path.write_text(json.dumps({"status": "unknown", "mode": "unknown"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="unknown"):
        pilot.compute_gate(path)
    path.write_text(json.dumps({"status": "ok", "mode": "open"}), encoding="utf-8")
    old = time.time() - 600
    os.utime(path, (old, old))
    with pytest.raises(RuntimeError, match="stale"):
        pilot.compute_gate(path, max_age_seconds=120)


def test_compute_gate_accepts_governor_utf8_bom(tmp_path):
    path = tmp_path / "compute_state.json"
    payload = json.dumps({"status": "ok", "mode": "open", "reasons": []}).encode("utf-8")
    path.write_bytes(b"\xef\xbb\xbf" + payload)
    assert pilot.compute_gate(path)["status"] == "ok"


def test_compute_state_reader_retries_a_transient_governor_lock():
    class FlakySnapshot:
        calls = 0
        payload = json.dumps({"status": "ok", "mode": "open"}).encode("utf-8")
        modified = time.time()

        def stat(self):
            return SimpleNamespace(
                st_ino=1,
                st_size=len(self.payload),
                st_mtime=self.modified,
                st_mtime_ns=int(self.modified * 1e9),
            )

        def read_bytes(self):
            self.calls += 1
            if self.calls == 1:
                raise PermissionError("governor rewrite in progress")
            return self.payload

    snapshot = FlakySnapshot()
    state, age, digest = pilot._read_compute_state_stably(
        snapshot,
        attempts=2,
        delay_seconds=0,
    )
    assert state["status"] == "ok"
    assert snapshot.calls == 2
    assert age >= 0
    assert len(digest) == 64


def test_balanced_accuracy_counts_extra_predicted_class_as_error_without_redefining_truth():
    truth = np.array([1, 1, 2, 2])
    prediction = np.array([0, 1, 2, 2])
    assert pilot._balanced_accuracy_without_spurious_class_warning(truth, prediction) == 0.75


def test_compute_gate_allows_only_pressure_safe_owner_authorized_stopped_trading(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "compute_state.json"
    state = {
        "status": "distress",
        "mode": "throttle",
        "reasons": ["azure_pressure"],
        "cpu_pct": 50.0,
        "mem_avail_mb": 16_000,
        "disk_queue": 0,
        "hung_windows": 0,
        "owner_active": False,
        "azure": {
            "status": "distress",
            "host": "trading-linux-az",
            "owner_rdp_active": False,
            "reasons": ["trading_unit_not_active:bot-acct1,thetadata"],
            "psi_cpu_some_avg60": 0.0,
            "psi_mem_some_avg60": 0.0,
            "psi_io_some_avg60": 0.0,
            "sys_slice_psi_avg60": 0.0,
            "mem_avail_mb": 24_000,
        },
    }
    path.write_text(json.dumps(state), encoding="utf-8")
    monkeypatch.delenv(pilot.STOPPED_TRADING_AUTH_ENV, raising=False)
    with pytest.raises(RuntimeError, match="distress"):
        pilot.compute_gate(path)
    monkeypatch.setenv(pilot.STOPPED_TRADING_AUTH_ENV, "1")
    monkeypatch.setenv(pilot.COMPUTE_TARGET_ENV, "local")
    audit = pilot.compute_gate(path)
    assert audit["decision"].startswith("proceed_owner_authorized")
    state["owner_active"] = True
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="distress"):
        pilot.compute_gate(path)
    state["owner_active"] = False
    state["azure"]["psi_cpu_some_avg60"] = 1.0
    path.write_text(json.dumps(state), encoding="utf-8")
    assert pilot.compute_gate(path)["pressure_evidence"]["compute_target"] == "local"
    monkeypatch.setenv(pilot.COMPUTE_TARGET_ENV, "azure")
    with pytest.raises(RuntimeError, match="distress"):
        pilot.compute_gate(path)
    state["azure"]["psi_cpu_some_avg60"] = 0.0
    path.write_text(json.dumps(state), encoding="utf-8")
    assert pilot.compute_gate(path)["pressure_evidence"]["schema"].endswith("azure")
    monkeypatch.setenv(pilot.COMPUTE_TARGET_ENV, "local")
    state["cpu_pct"] = 90.0
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(RuntimeError, match="distress"):
        pilot.compute_gate(path)


def test_process_priority_is_set_and_verified():
    audit = pilot.set_below_normal_priority()
    assert audit["verified"] is True
    if os.name == "nt":
        assert audit["priority_name"] == "BelowNormal"
        assert audit["priority_class"] == 0x00004000
    else:
        assert audit["priority_class"] >= 10


def test_direct_azure_health_requires_exact_host_idle_owner_and_finite_pressure(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "compute_health.json"
    health = {
        "status": "distress",
        "mode": "throttle",
        "host": "trading-linux-az",
        "owner_rdp_active": False,
        "reasons": ["trading_unit_not_active:bot-acct1,thetadata"],
        "psi_cpu_some_avg60": 0.0,
        "psi_mem_some_avg60": 0.0,
        "psi_io_some_avg60": 0.0,
        "sys_slice_psi_avg60": 0.0,
        "mem_avail_mb": 24_000,
    }
    path.write_text(json.dumps(health), encoding="utf-8")
    monkeypatch.setenv(pilot.STOPPED_TRADING_AUTH_ENV, "1")
    monkeypatch.setenv(pilot.COMPUTE_TARGET_ENV, "azure")
    assert pilot.compute_gate(path)["pressure_evidence"]["schema"] == "direct_azure_compute_health"
    health["psi_cpu_some_avg60"] = float("nan")
    path.write_text(json.dumps(health), encoding="utf-8")
    with pytest.raises(RuntimeError, match="distress"):
        pilot.compute_gate(path)
    health["psi_cpu_some_avg60"] = 0.0
    health["owner_rdp_active"] = True
    path.write_text(json.dumps(health), encoding="utf-8")
    with pytest.raises(RuntimeError, match="distress"):
        pilot.compute_gate(path)
    monkeypatch.setenv(pilot.AZURE_CLOSING_OWNER_AUTH_ENV, "1")
    monkeypatch.setattr(
        pilot,
        "_closing_azure_owner_session_evidence",
        lambda: (True, {"decision": "all_owner_sessions_closing", "test": True}),
    )
    audit = pilot.compute_gate(path)
    assert audit["pressure_evidence"]["closing_owner_session_override"]["test"] is True


def test_closing_owner_session_override_parses_only_closing_owner_rows(monkeypatch):
    monkeypatch.setenv(pilot.AZURE_CLOSING_OWNER_AUTH_ENV, "1")
    monkeypatch.setattr(pilot.os, "name", "posix")
    outputs = iter([
        "c2 1001 owner - - closing no -\n6673 1002 aiwork - - active no -\n",
        "c2 1001 owner - - active no -\n",
    ])

    def fake_run(*args, **kwargs):
        return SimpleNamespace(stdout=next(outputs))

    monkeypatch.setattr(pilot.subprocess, "run", fake_run)
    safe, evidence = pilot._closing_azure_owner_session_evidence()
    assert safe is True
    assert evidence["owner_session_states"] == ["closing"]
    safe, evidence = pilot._closing_azure_owner_session_evidence()
    assert safe is False
    assert evidence["owner_session_states"] == ["active"]


def test_staged_bundle_revision_attestation_is_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("DNNAIC_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setenv("DNNAIC_SOURCE_DIRTY", "1")
    audit = pilot.git_revision(repo=tmp_path, script=Path(pilot.__file__))
    assert audit["commit"] == "a" * 40
    assert audit["dirty_at_run"] is True
    assert audit["source"] == "unverified_environment_attestation_without_local_git"
    assert audit["commit_verified_locally"] is False
    assert len(audit["script_sha256"]) == 64
    with pytest.raises(RuntimeError, match="clean tracked source"):
        pilot.require_clean_tracked_revision(audit)
    monkeypatch.setenv("DNNAIC_SOURCE_COMMIT", "not-a-commit")
    with pytest.raises(RuntimeError, match="valid DNNAIC_SOURCE_COMMIT"):
        pilot.git_revision(repo=tmp_path, script=Path(pilot.__file__))


def test_clean_source_gate_allows_untracked_outputs_but_rejects_tracked_edits(tmp_path):
    script = tmp_path / "runner.py"
    test_file = tmp_path / "test_runner.py"
    script.write_text("print('clean')\n", encoding="utf-8")
    test_file.write_text("def test_clean(): pass\n", encoding="utf-8")
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "DNNaic test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "add", "runner.py", "test_runner.py"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=tmp_path, check=True, capture_output=True)
    initial = pilot.git_revision(repo=tmp_path, script=script)
    pilot.require_clean_tracked_revision(initial)
    output = tmp_path / "results" / "run" / "simulation_checkpoint.json"
    output.parent.mkdir(parents=True)
    output.write_text("{}", encoding="utf-8")
    after_output = pilot.git_revision(repo=tmp_path, script=script)
    assert after_output["dirty_at_run"] is True
    assert after_output["tracked_dirty_at_snapshot"] is False
    pilot.require_revision_unchanged(initial, after_output)
    script.write_text("print('changed')\n", encoding="utf-8")
    changed = pilot.git_revision(repo=tmp_path, script=script)
    with pytest.raises(RuntimeError, match="tracked worktree content is dirty"):
        pilot.require_clean_tracked_revision(changed)


def test_atomic_json_writer_roundtrips_and_leaves_no_partial_file(tmp_path):
    path = tmp_path / "result.json"
    audit = pilot.write_json_atomic(path, {"finite": 1.25, "rows": [1, 2, 3]})
    assert json.loads(path.read_text(encoding="utf-8"))["finite"] == 1.25
    assert audit["bytes"] == path.stat().st_size
    assert len(audit["sha256"]) == 64
    assert not path.with_suffix(".json.part").exists()


def test_structured_result_namespace_has_a_real_single_writer_lock(tmp_path):
    first = pilot.SingleWriterLease(tmp_path, ".run.lock").acquire()
    try:
        with pytest.raises(RuntimeError, match="already locked"):
            pilot.SingleWriterLease(tmp_path, ".run.lock").acquire()
    finally:
        first.close()
    second = pilot.SingleWriterLease(tmp_path, ".run.lock").acquire()
    second.close()


def test_simulation_checkpoint_rejects_payload_tampering(tmp_path):
    representation = {
        "specification": {},
        "feature_columns": ["one"],
        "feature_dimension": 1,
        "genealogy_cv": {"metric": 1},
        "rate_family_cv": {"metric": 2},
        "fixed_continuous_transfer": {"metric": 3},
    }
    digest = pilot.hashlib.sha256(pilot._canonical_json(representation)).hexdigest()
    path = tmp_path / "checkpoint.json"
    checkpoint = {
        "schema_version": pilot.SIMULATION_CHECKPOINT_SCHEMA,
        "contract_sha256": "c" * 64,
        "completed_representations": ["raw_all"],
        "representation_payload_sha256": {"raw_all": digest},
        "representations": {"raw_all": representation},
    }
    pilot.write_json_atomic(path, checkpoint)
    loaded = pilot.load_simulation_checkpoint(
        path,
        contract_sha256="c" * 64,
        representation_order=("raw_all", "structured"),
    )
    assert loaded["raw_all"]["genealogy_cv"]["metric"] == 1
    checkpoint["representations"]["raw_all"]["genealogy_cv"]["metric"] = 999
    pilot.write_json_atomic(path, checkpoint)
    with pytest.raises(RuntimeError, match="payload hash changed"):
        pilot.load_simulation_checkpoint(
            path,
            contract_sha256="c" * 64,
            representation_order=("raw_all", "structured"),
        )


def _natural_result(matrix, *, accuracy=False):
    return {
        "schema_version": "synthetic-natural-v1",
        "direction_head": {
            "dimension": 54,
            "depth_grid": list(range(2, 17)),
            "representation": (
                "mean and population SD over g=2..16 for each of 27 non-depth coordinates"
            ),
        },
        "panels": [
            {
                "panel_id": "synthetic_panel",
                "population_order": {"P1": "one", "P2": "two", "P3": "three"},
                "feature_matrix": matrix.tolist(),
                "padze": {"n_loci_kept": 100},
                "simulation_head": {"predicted_class": "A"},
                "simulation_feature_shift": {"rms_z": 99.0},
                "external_expectation": {"candidate_class": "C"},
                "adjudication": {"formal_direction_accuracy_eligible": accuracy},
            }
        ],
    }


def test_natural_loader_never_creates_an_accuracy_denominator(tmp_path):
    path = tmp_path / "bundle" / "results.json"
    path.parent.mkdir()
    matrix = _curve_table(replicates=1, depths=15)[0]
    path.write_text(json.dumps(_natural_result(matrix)), encoding="utf-8")
    panels, audit = pilot.load_natural_panels([path], max_depth=16)
    assert audit["panel_rows"] == 1
    assert audit["accuracy_eligible_rows"] == 0
    assert panels[0]["candidate_class"] == "C"
    assert panels[0]["any_true_accuracy_flag"] is False
    path.write_text(json.dumps(_natural_result(matrix, accuracy=True)), encoding="utf-8")
    with pytest.raises(AssertionError, match="accuracy-eligible"):
        pilot.load_natural_panels([path], max_depth=16)


def test_load_canonical_uses_design_and_selects_only_requested_depths(tmp_path):
    directory = tmp_path / "regen_full"
    directory.mkdir()
    groups = []
    directions = []
    magnitudes = []
    designs = []
    matrices = []
    for group_index, label in enumerate("ABCD"):
        table = _curve_table(replicates=1, depths=198)[0]
        groups.extend([f"{label}|group-{group_index}"] * 198)
        directions.extend([label] * 198)
        magnitudes.extend(([0.0] if label == "D" else [1e-4]) * 198)
        designs.extend((["control"] if label == "D" else ["continuous"]) * 198)
        matrices.append(table)
    np.save(directory / "X.npy", np.concatenate(matrices))
    np.save(directory / "direction.npy", np.array(directions))
    np.save(directory / "groups.npy", np.array(groups))
    np.save(directory / "magnitude.npy", np.array(magnitudes))
    np.save(directory / "design.npy", np.array(designs))
    loaded = pilot.load_canonical(directory, max_depth=16)
    assert loaded["table"].shape == (4, 15, 28)
    assert loaded["labels"].tolist() == list("ABCD")
    assert loaded["designs"].tolist() == ["continuous"] * 3 + ["control"]
    assert len(loaded["audit"]["array_contracts"]) == 5


def test_pilot_adjudication_excludes_natural_candidate_matches():
    def cv(accuracy, family_accuracy=None):
        return {
            "per_repeat_metrics": [{
                "appreciable": {"accuracy": accuracy},
                "appreciable_equal_rate_family": {
                    "balanced_accuracy": (
                        accuracy if family_accuracy is None else family_accuracy
                    )
                },
            }],
        }

    def coverage(first, second):
        return {
            "result_file_bundle_balanced_descriptive": {
                "rms_z_median": np.median([first, second]),
                "rms_z_p95": np.quantile([first, second], 0.95),
                "by_bundle": {
                    "bundle_a": {"median_rms_z": first},
                    "bundle_b": {"median_rms_z": second},
                },
            }
        }

    variants = {
        "raw_all": {
            "genealogy_cv": cv(0.98),
            "rate_family_cv": cv(0.20, 0.97),
            "natural_transfer": {"coverage": coverage(20.0, 30.0)},
        },
        "orbit_composition_mean_variance": {
            "genealogy_cv": cv(0.975),
            "rate_family_cv": cv(0.20, 0.96),
            "natural_transfer": {"coverage": coverage(10.0, 20.0)},
        },
    }
    result = pilot.adjudicate_pilot(variants)
    assert result["all_relative_bridge_thresholds_pass"] is True
    assert result["appreciable_accuracy"]["rate_family_cv"][
        "primary_estimand"
    ].startswith("equal-weight")
    assert "Natural candidate labels are never" in result["decision_rule"]


def test_retrained_raw_head_can_explicitly_skip_canonical_reproduction_check():
    table = _curve_table(replicates=12, depths=15)
    features = pilot.representation_features(table, "raw_all")
    labels = np.tile(np.arange(3), 4)
    rate_blocks = np.repeat([f"rate-{index}" for index in range(4)], 3)
    panel = {
        "bundle": "synthetic_bundle",
        "source_result": "synthetic",
        "panel_id": "synthetic_panel",
        "population_order": {"P1": "one", "P2": "two", "P3": "three"},
        "matrix": _curve_table(replicates=1, depths=15)[0],
        "n_loci": 100,
        "candidate_class": None,
        "source_raw_prediction": "A",
        "source_direction_rms_z": 999.0,
        "any_true_accuracy_flag": False,
    }
    reference_rows = [{
        "scaler_mean": np.zeros(features.shape[1]).tolist(),
        "scaler_scale": np.ones(features.shape[1]).tolist(),
        "source_test_rms_z_p99": 100.0,
    }]
    crossfit_reference = {
        "folds": reference_rows,
        "folds_sha256": pilot.hashlib.sha256(
            pilot._canonical_json(reference_rows)
        ).hexdigest(),
    }
    result = pilot.final_natural_score(
        features,
        labels,
        rate_blocks,
        [panel],
        representation="raw_all",
        C_grid=(1.0,),
        inner_splits=3,
        seed=1,
        crossfit_support_reference=crossfit_reference,
        verify_source_raw_head=False,
    )
    assert result["raw_head_reproduction"]["status"].startswith("not applicable")
    support = result["coverage"]["crossfit_source_test_support_descriptive"]
    assert support["result_file_bundles"] == 1
    assert "not conformal coverage" in support["guardrail"]
    invalid_reference = copy.deepcopy(crossfit_reference)
    invalid_reference["folds"][0]["scaler_scale"] = [0.0] * features.shape[1]
    invalid_reference["folds_sha256"] = pilot.hashlib.sha256(
        pilot._canonical_json(invalid_reference["folds"])
    ).hexdigest()
    with pytest.raises(ValueError, match="support reference is invalid"):
        pilot.final_natural_score(
            features,
            labels,
            rate_blocks,
            [panel],
            representation="raw_all",
            C_grid=(1.0,),
            inner_splits=3,
            seed=1,
            crossfit_support_reference=invalid_reference,
            verify_source_raw_head=False,
        )
    with pytest.raises(AssertionError, match="failed to reproduce"):
        pilot.final_natural_score(
            features,
            labels,
            rate_blocks,
            [panel],
            representation="raw_all",
            C_grid=(1.0,),
            inner_splits=3,
            seed=1,
            verify_source_raw_head=True,
        )
