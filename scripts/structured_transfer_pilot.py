#!/usr/bin/env python3
"""Evaluate nuisance-aware direction features under hierarchical grouped CV.

This is an exploratory bridge experiment, not a replacement result for the
paper.  It asks a narrow question motivated by the external-data audit: can a
population-structured, scale-reduced representation preserve the canonical
simulation signal while reducing the extreme observation-domain shift?

Three data-informed logistic heads are compared on the identical g=2..16 grid:

* ``raw_all``: the current 54-D mean+SD summary of all mean/variance/SE curves;
* ``raw_mean_variance``: the same summary with the locus-count-bearing SE
  coordinates removed; and
* ``orbit_composition_mean_variance``: alpha/private/pair-private values are
  normalized within their three population-labelled coordinate triples before
  the mean+SD summary.  Alpha means use excess richness ``alpha - 1``.  This is
  scale reduction, not an S3-equivariant model; the released A/B/C task does
  not contain all six ordered population edges needed for full S3 closure.

Every model-selection operation is nested inside the outer split, and every
scaler is fit on training data only.  Two outer estimands are reported:

* a new genealogy from the same simulation design; and
* a new exact rate family, where all A/B/C replicates sharing one rate draw are
  held out together.

The pinned standardized natural cohort is loaded only after all simulation
evaluation.  It provides unlabeled coverage/OOD diagnostics and descriptive
candidate concordance, never an accuracy denominator.  Result-file bundles
receive equal weight in the exploratory transfer diagnostic.  The script records full
fold and OOF ledgers so overlap and model selection can be audited.

The process aborts before loading arrays whenever the owner's compute governor
reports distress.  It is CPU-only, single-threaded, and BelowNormal on Windows.
"""
from __future__ import annotations

import argparse
import atexit
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Iterable, Sequence
import warnings

# Set numerical-library thread defaults before importing NumPy/sklearn.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[1]
CLASSES = np.array(["A", "B", "C"])
APPRECIABLE = 2.5e-4
DEFAULT_MAX_DEPTH = 16
SCHEMA_VERSION = "dnnaic-structured-transfer-pilot-v2"
SIMULATION_CHECKPOINT_SCHEMA = "dnnaic-structured-simulation-checkpoint-v2"
DEFAULT_RESULT_DIR = REPO / "results" / "structured_transfer_pilot_v2_2026_07_11"
DEFAULT_COMPUTE_STATE = Path.home() / ".claude" / "compute" / "compute_state.json"
STOPPED_TRADING_AUTH_ENV = "DNNAIC_OWNER_AUTHORIZED_STOPPED_TRADING_COMPUTE"
COMPUTE_TARGET_ENV = "DNNAIC_COMPUTE_TARGET"
AZURE_CLOSING_OWNER_AUTH_ENV = "DNNAIC_OWNER_AUTHORIZED_CLOSING_AZURE_SESSION"
AZURE_STOPPED_TRADING_CPU_PSI_MAX = 20.0
AZURE_STOPPED_TRADING_OTHER_PSI_MAX = 5.0
MOMENT_NAMES = ("mean", "variance", "se")
ORBIT_NAMES = ("alpha", "private", "pair_private")
COMPOSITION_NEGATIVE_TOLERANCE = 1e-10
PINNED_NATURAL_RESULTS = (
    "results/additional_external_benchmarks_2026_07_11/results.json",
    "results/dingo_weeks_2025_external_benchmark_2026_07_11/results.json",
    "results/directional_external_benchmarks_2026_07_11/results.json",
    "results/external_benchmarks_2026_07_10/results.json",
    "results/further_external_benchmarks_2026_07_11/results.json",
    "results/guppy_2020_external_benchmark_2026_07_11/results.json",
    "results/hantarcticus_2024_external_benchmark_2026_07_11/results.json",
    "results/harpagifer_external_benchmark_2026_07_11/results.json",
    "results/oyster_2017_external_benchmark_2026_07_11/results.json",
    "results/seabass_external_benchmark_2026_07_11/results.json",
    "results/tinkerbird_2024_external_benchmark_2026_07_11/results.json",
    "results/tinkerbird_external_benchmark_2026_07_11/results.json",
    "results/wrasse_external_benchmark_2026_07_11/results.json",
    "results/yellowstone_2019_external_benchmark_2026_07_11/results.json",
)
REPRESENTATIONS = {
    "raw_all": {
        "kind": "raw",
        "moments": (0, 1, 2),
        "description": "current raw mean/variance/SE curves summarized by depth mean and SD",
        "final_C_policy": "fixed canonical C=1",
    },
    "raw_mean_variance": {
        "kind": "raw",
        "moments": (0, 1),
        "description": "raw mean/variance curves; SE removed",
        "final_C_policy": "rate-family grouped CV",
    },
    "orbit_composition_mean_variance": {
        "kind": "orbit_composition",
        "moments": (0, 1),
        "description": (
            "within-triple centered compositions for alpha-1/private/pair-private "
            "mean and variance curves; scale-reduced but not S3-equivariant"
        ),
        "final_C_policy": "rate-family grouped CV",
    },
}


def set_below_normal_priority() -> dict:
    """Set and verify BelowNormal on Windows or nice >=10 on POSIX."""
    if os.name != "nt":
        try:
            before = int(os.nice(0))
            if before < 10:
                os.nice(10 - before)
            actual = int(os.nice(0))
        except OSError as exc:
            raise RuntimeError("could not set and verify Linux/POSIX nice priority") from exc
        if actual < 10:
            raise RuntimeError(f"research process nice priority {actual} is below required 10")
        return {
            "applicable": True,
            "verified": True,
            "priority_class": actual,
            "priority_name": f"nice={actual}",
            "previous_priority_class": before,
        }
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
        kernel32.GetPriorityClass.restype = wintypes.DWORD
        process = kernel32.GetCurrentProcess()
        below_normal = 0x00004000
        if not kernel32.SetPriorityClass(process, below_normal):
            raise OSError(ctypes.get_last_error(), "SetPriorityClass failed")
        actual = int(kernel32.GetPriorityClass(process))
        if actual != below_normal:
            raise RuntimeError(
                f"BelowNormal priority verification failed: expected {below_normal}, got {actual}"
            )
        return {
            "applicable": True,
            "verified": True,
            "priority_class": actual,
            "priority_name": "BelowNormal",
        }
    except Exception as exc:
        raise RuntimeError("could not set and verify BelowNormal process priority") from exc


def _safe_metric(value, *, integer: bool = False) -> float | int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -1 if integer else float("inf")
    if not math.isfinite(number) or number < 0:
        return -1 if integer else float("inf")
    return int(number) if integer else number


def _closing_azure_owner_session_evidence() -> tuple[bool, dict]:
    """Verify that a stale Azure owner-session flag refers only to closing sessions."""
    authorized = os.environ.get(AZURE_CLOSING_OWNER_AUTH_ENV) == "1"
    evidence = {
        "explicit_authorization": authorized,
        "source": "loginctl list-sessions --no-legend",
        "owner_session_states": [],
    }
    if not authorized or os.name == "nt":
        evidence["decision"] = "not_authorized_or_not_on_posix_target"
        return False, evidence
    try:
        completed = subprocess.run(
            ["loginctl", "list-sessions", "--no-legend"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        evidence["decision"] = f"loginctl_unavailable:{type(exc).__name__}"
        return False, evidence
    states = []
    rows = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) >= 6 and fields[2] == "owner":
            rows.append(fields[:6])
            states.append(fields[5])
    evidence["owner_session_states"] = states
    evidence["owner_session_row_sha256"] = hashlib.sha256(
        _canonical_json(rows)
    ).hexdigest()
    safe = bool(states) and all(state == "closing" for state in states)
    evidence["decision"] = (
        "all_owner_sessions_closing" if safe else "owner_session_not_proven_closing"
    )
    return safe, evidence


def _stopped_trading_pressure_is_safe(state: dict) -> tuple[bool, dict]:
    """Recognize stopped-trading availability alerts for one explicit target."""
    target = os.environ.get(COMPUTE_TARGET_ENV, "")
    if target == "local":
        azure = state.get("azure", {}) if isinstance(state.get("azure"), dict) else {}
        azure_reasons = list(azure.get("reasons", []))
        evidence = {
            "compute_target": target,
            "schema": "merged_local_with_nested_azure",
            "owner_active": state.get("owner_active"),
            "local_cpu_pct": _safe_metric(state.get("cpu_pct")),
            "local_mem_avail_mb": _safe_metric(state.get("mem_avail_mb"), integer=True),
            "local_disk_queue": _safe_metric(state.get("disk_queue")),
            "local_hung_windows": _safe_metric(state.get("hung_windows"), integer=True),
            "trading_health_reasons": azure_reasons,
        }
        safe = (
            state.get("status") == "distress"
            and list(state.get("reasons", [])) == ["azure_pressure"]
            and len(azure_reasons) == 1
            and str(azure_reasons[0]).startswith("trading_unit_not_active:")
            and state.get("owner_active") is False
            and evidence["local_cpu_pct"] <= 85.0
            and evidence["local_mem_avail_mb"] >= 8_192
            and evidence["local_disk_queue"] <= 1.0
            and evidence["local_hung_windows"] == 0
        )
        return bool(safe), evidence
    if target == "azure":
        health = (
            state["azure"]
            if isinstance(state.get("azure"), dict)
            else state
        )
        reasons = list(health.get("reasons", []))
        closing_owner_safe, closing_owner_evidence = (
            _closing_azure_owner_session_evidence()
            if health.get("owner_rdp_active") is True
            else (False, {"decision": "health_reports_no_owner_session"})
        )
        evidence = {
            "compute_target": target,
            "schema": (
                "merged_local_nested_azure"
                if health is not state
                else "direct_azure_compute_health"
            ),
            "host": health.get("host"),
            "owner_rdp_active": health.get("owner_rdp_active"),
            "closing_owner_session_override": closing_owner_evidence,
            "trading_health_reasons": reasons,
            "psi_cpu_some_avg60": _safe_metric(health.get("psi_cpu_some_avg60")),
            "psi_mem_some_avg60": _safe_metric(health.get("psi_mem_some_avg60")),
            "psi_io_some_avg60": _safe_metric(health.get("psi_io_some_avg60")),
            "sys_slice_psi_avg60": _safe_metric(health.get("sys_slice_psi_avg60")),
            "mem_avail_mb": _safe_metric(health.get("mem_avail_mb"), integer=True),
            "pressure_thresholds": {
                "psi_cpu_some_avg60_max": AZURE_STOPPED_TRADING_CPU_PSI_MAX,
                "psi_mem_io_system_avg60_max": AZURE_STOPPED_TRADING_OTHER_PSI_MAX,
                "basis": (
                    "one half of the canonical CPU PSI distress threshold and one third "
                    "of the memory threshold; one nice>=10 process is the minimum throttle mode"
                ),
            },
        }
        safe = (
            health.get("status") == "distress"
            and health.get("host") == "trading-linux-az"
            and (
                health.get("owner_rdp_active") is False
                or closing_owner_safe
            )
            and len(reasons) == 1
            and str(reasons[0]).startswith("trading_unit_not_active:")
            and evidence["psi_cpu_some_avg60"] <= AZURE_STOPPED_TRADING_CPU_PSI_MAX
            and evidence["psi_mem_some_avg60"] <= AZURE_STOPPED_TRADING_OTHER_PSI_MAX
            and evidence["psi_io_some_avg60"] <= AZURE_STOPPED_TRADING_OTHER_PSI_MAX
            and evidence["sys_slice_psi_avg60"] <= AZURE_STOPPED_TRADING_OTHER_PSI_MAX
            and evidence["mem_avail_mb"] >= 8_192
        )
        return bool(safe), evidence
    return False, {"compute_target": target, "schema": "invalid_target"}


def _read_compute_state_stably(
    path: Path,
    *,
    attempts: int = 8,
    delay_seconds: float = 0.05,
) -> tuple[dict, float, str]:
    """Read through the governor's brief non-atomic rewrite/lock window."""
    if attempts < 1 or delay_seconds < 0:
        raise ValueError("compute-state retry policy is invalid")
    last_error = None
    for attempt in range(attempts):
        try:
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
            identity_before = (before.st_ino, before.st_size, before.st_mtime_ns)
            identity_after = (after.st_ino, after.st_size, after.st_mtime_ns)
            if identity_before != identity_after or len(payload) != after.st_size:
                raise OSError("compute governor snapshot changed during read")
            state = json.loads(payload.decode("utf-8-sig"))
            if not isinstance(state, dict):
                raise ValueError("compute governor snapshot must be a JSON object")
            return (
                state,
                max(0.0, time.time() - after.st_mtime),
                hashlib.sha256(payload).hexdigest(),
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(delay_seconds)
    raise RuntimeError(
        f"could not read a stable compute governor snapshot after {attempts} attempts"
    ) from last_error


def compute_gate(
    path: Path = DEFAULT_COMPUTE_STATE,
    *,
    max_age_seconds: float = 120.0,
) -> dict:
    """Read the governor state and hard-abort before work on distress."""
    if not path.exists():
        raise RuntimeError(f"compute governor state is unavailable; aborting before work: {path}")
    state, age_seconds, state_sha256 = _read_compute_state_stably(path)
    if age_seconds > max_age_seconds:
        raise RuntimeError(
            "compute governor state is stale; aborting before work: "
            f"age={age_seconds:.1f}s > {max_age_seconds:.1f}s"
        )
    status = str(state.get("status", "unknown"))
    mode = str(state.get("mode", "unknown"))
    audit = {
        "state_file": str(path),
        "available": True,
        "file_age_seconds": age_seconds,
        "maximum_age_seconds": float(max_age_seconds),
        "sample_timestamp": state.get("ts"),
        "state_sha256": state_sha256,
        "status": status,
        "mode": mode,
        "reasons": list(state.get("reasons", [])),
        "decision": "proceed_single_thread_below_normal" if status == "ok" else "abort",
    }
    stopped_trading_authorized = os.environ.get(STOPPED_TRADING_AUTH_ENV) == "1"
    pressure_safe, pressure_evidence = _stopped_trading_pressure_is_safe(state)
    if status != "ok" and stopped_trading_authorized and pressure_safe:
        audit.update({
            "decision": "proceed_owner_authorized_stopped_trading_single_thread_below_normal",
            "owner_authorization_environment": STOPPED_TRADING_AUTH_ENV,
            "known_category_error": (
                "trading availability is inactive while direct local/Azure pressure telemetry "
                "remains below pinned safety thresholds"
            ),
            "pressure_evidence": pressure_evidence,
        })
        return audit
    if status != "ok":
        raise RuntimeError(
            f"compute governor reports {status!r}; aborting before array load/model fit: "
            + ",".join(audit["reasons"])
        )
    return audit


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def git_revision(
    repo: Path = REPO,
    script: Path | None = None,
) -> dict:
    """Record local Git state or a fail-closed attestation for a staged bundle."""
    repo = Path(repo).resolve()
    script = Path(__file__).resolve() if script is None else Path(script).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        tracked_dirty = subprocess.run(
            [
                "git", "-C", str(repo), "status", "--porcelain",
                "--untracked-files=no",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        try:
            relative_script = script.relative_to(repo).as_posix()
        except ValueError:
            relative_script = None
        script_in_head = False
        head_script_sha256 = None
        diff = None
        head_blob_oid = None
        worktree_blob_oid = None
        if relative_script is not None:
            tracked = subprocess.run(
                ["git", "-C", str(repo), "ls-files", "--error-unmatch", "--", relative_script],
                check=False,
                capture_output=True,
            )
            script_in_head = tracked.returncode == 0
        if script_in_head:
            head_payload = subprocess.run(
                ["git", "-C", str(repo), "show", f"HEAD:{relative_script}"],
                check=True,
                capture_output=True,
            ).stdout
            head_script_sha256 = hashlib.sha256(head_payload).hexdigest()
            diff = subprocess.run(
                ["git", "-C", str(repo), "diff", "--binary", "HEAD", "--", relative_script],
                check=True,
                capture_output=True,
            ).stdout
            head_blob_oid = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", f"HEAD:{relative_script}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            worktree_blob_oid = subprocess.run(
                [
                    "git", "-C", str(repo), "hash-object",
                    f"--path={relative_script}", relative_script,
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        source = "local_git_worktree"
        commit_verified_locally = True
        diff_bytes = None if diff is None else int(len(diff))
        diff_sha256 = None if diff is None else hashlib.sha256(diff).hexdigest()
        dirty_at_run = bool(dirty)
        tracked_dirty_at_snapshot = bool(tracked_dirty)
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = os.environ.get("DNNAIC_SOURCE_COMMIT", "").lower()
        dirty_text = os.environ.get("DNNAIC_SOURCE_DIRTY", "")
        if not re.fullmatch(r"[0-9a-f]{40,64}", commit):
            raise RuntimeError(
                "staged runner lacks Git metadata and a valid DNNAIC_SOURCE_COMMIT"
            )
        if dirty_text not in {"0", "1"}:
            raise RuntimeError(
                "staged runner requires DNNAIC_SOURCE_DIRTY=0 or 1"
            )
        source = "unverified_environment_attestation_without_local_git"
        commit_verified_locally = False
        script_in_head = None
        head_script_sha256 = None
        diff_bytes = None
        diff_sha256 = None
        dirty_at_run = dirty_text == "1"
        tracked_dirty_at_snapshot = None
        head_blob_oid = None
        worktree_blob_oid = None
    return {
        "commit": commit,
        "dirty_at_run": dirty_at_run,
        "tracked_dirty_at_snapshot": tracked_dirty_at_snapshot,
        "source": source,
        "commit_verified_locally": commit_verified_locally,
        "script_in_head": script_in_head,
        "head_script_sha256": head_script_sha256,
        "head_blob_oid": head_blob_oid,
        "worktree_blob_oid": worktree_blob_oid,
        "script": str(script),
        "script_sha256": sha256_file(script),
        "tracked_diff_bytes": diff_bytes,
        "tracked_diff_sha256": diff_sha256,
    }


def require_clean_tracked_revision(revision: dict) -> None:
    """Fail closed unless the executing source is exactly a clean local Git HEAD."""
    failures = []
    if revision.get("source") != "local_git_worktree":
        failures.append("source is not a local Git worktree")
    if revision.get("commit_verified_locally") is not True:
        failures.append("commit was not verified locally")
    if revision.get("script_in_head") is not True:
        failures.append("runner is not tracked in HEAD")
    if revision.get("tracked_dirty_at_snapshot") is not False:
        failures.append("tracked worktree content is dirty before output creation")
    if revision.get("head_blob_oid") != revision.get("worktree_blob_oid"):
        failures.append("runner canonical Git blob differs from HEAD")
    if revision.get("tracked_diff_bytes") != 0:
        failures.append("runner has a tracked diff")
    if failures:
        raise RuntimeError(
            "publishable structured pilot requires clean tracked source: "
            + "; ".join(failures)
        )


def require_revision_unchanged(initial: dict, final: dict) -> None:
    """Reject a mid-run source or HEAD transition before the final result write."""
    require_clean_tracked_revision(final)
    fields = ("commit", "head_blob_oid", "worktree_blob_oid", "script_in_head")
    if any(initial.get(field) != final.get(field) for field in fields):
        raise RuntimeError("structured pilot source changed after the initial snapshot")


def _canonical_json(value) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def validate_C_grid(values: Iterable[float]) -> tuple[float, ...]:
    """Return a finite, positive, unique, strictly increasing model grid."""
    grid = tuple(float(value) for value in values)
    if not grid:
        raise ValueError("C grid is empty")
    if any(not math.isfinite(value) or value <= 0 for value in grid):
        raise ValueError("C grid values must be finite and positive")
    if len(set(grid)) != len(grid):
        raise ValueError("C grid values must be unique")
    if tuple(sorted(grid)) != grid:
        raise ValueError("C grid values must be strictly increasing")
    return grid


class SingleWriterLease:
    """Cross-platform advisory lock released automatically on process exit."""

    def __init__(self, directory: Path, name: str):
        self.directory = Path(directory)
        self.path = self.directory / name
        self.handle = None
        self._locked = False

    def acquire(self):
        if self._locked:
            return self
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise RuntimeError(f"result namespace is already locked: {self.path}") from exc
        self.handle = handle
        self._locked = True
        metadata = _canonical_json({
            "pid": os.getpid(),
            "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        }) + b"\n"
        handle.seek(0)
        handle.truncate()
        handle.write(metadata)
        handle.flush()
        os.fsync(handle.fileno())
        atexit.register(self.close)
        return self

    def close(self):
        if not self._locked or self.handle is None:
            return
        handle = self.handle
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            self.handle = None
            self._locked = False

    def __enter__(self):
        return self.acquire()

    def __exit__(self, _exc_type, _exc, _traceback):
        self.close()


def write_json_atomic(path: Path, value: dict, *, indent: int | None = None) -> dict:
    """Durably replace one JSON artifact and return its content audit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        value,
        indent=indent,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":") if indent is None else None,
    ) + "\n"
    payload = text.encode("utf-8")
    temporary = path.with_name(
        f"{path.name}.part.{os.getpid()}.{time.time_ns()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
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
    return {
        "path": str(path.resolve()),
        "bytes": int(len(payload)),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def load_simulation_checkpoint(
    path: Path,
    *,
    contract_sha256: str,
    representation_order: Sequence[str],
) -> dict:
    checkpoint = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(checkpoint, dict):
        raise RuntimeError("structured simulation checkpoint is not a JSON object")
    if (
        checkpoint.get("schema_version") != SIMULATION_CHECKPOINT_SCHEMA
        or checkpoint.get("contract_sha256") != contract_sha256
    ):
        raise RuntimeError(
            "structured simulation checkpoint contract changed; use a fresh result directory"
        )
    completed = list(checkpoint.get("completed_representations", []))
    order = list(representation_order)
    if completed != order[: len(completed)]:
        raise RuntimeError("structured simulation checkpoint is not an ordered prefix")
    stored = checkpoint.get("representations", {})
    hashes = checkpoint.get("representation_payload_sha256", {})
    if not isinstance(stored, dict) or set(stored) != set(completed):
        raise RuntimeError("structured simulation checkpoint representation set changed")
    if not isinstance(hashes, dict) or set(hashes) != set(completed):
        raise RuntimeError("structured simulation checkpoint hash ledger changed")
    required = {
        "specification",
        "feature_columns",
        "feature_dimension",
        "genealogy_cv",
        "rate_family_cv",
        "fixed_continuous_transfer",
    }
    variants = {}
    for name in completed:
        payload = stored[name]
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise RuntimeError(f"checkpoint representation {name} is incomplete")
        observed = hashlib.sha256(_canonical_json(payload)).hexdigest()
        if observed != hashes[name]:
            raise RuntimeError(f"checkpoint representation {name} payload hash changed")
        variants[name] = payload
    return variants


def validate_curve_table(table: np.ndarray) -> np.ndarray:
    table = np.asarray(table, dtype=float)
    if table.ndim != 3 or table.shape[2] != 28:
        raise ValueError("expected (replicate, depth, 28) PADZE curve table")
    if table.shape[1] < 2:
        raise ValueError("at least two rarefaction depths are required")
    if not np.isfinite(table).all():
        raise ValueError("PADZE curve table contains a non-finite value")
    if not np.all(table[:, :, 0] == table[0, :, 0]):
        raise ValueError("replicates do not share one depth grid")
    if not np.all(np.diff(table[0, :, 0]) == 1):
        raise ValueError("PADZE depths must be consecutive")
    return table


def _summarize_curves(curves: np.ndarray) -> np.ndarray:
    curves = np.asarray(curves, dtype=float)
    if curves.ndim != 3:
        raise ValueError("expected (replicate, depth, coordinate) curves")
    return np.concatenate((curves.mean(axis=1), curves.std(axis=1)), axis=1)


def raw_summary(table: np.ndarray, moments: Sequence[int]) -> np.ndarray:
    """Summarize selected raw PADZE moments across rarefaction depth."""
    table = validate_curve_table(table)
    moments = tuple(int(value) for value in moments)
    if not moments or any(value not in (0, 1, 2) for value in moments):
        raise ValueError("moments must be a nonempty subset of 0,1,2")
    values = table[:, :, 1:].reshape(len(table), table.shape[1], 9, 3)
    curves = values[:, :, :, moments].reshape(len(table), table.shape[1], -1)
    return _summarize_curves(curves)


def orbit_composition_summary(table: np.ndarray, moments: Sequence[int]) -> np.ndarray:
    """Build scale-reduced coordinate-triple compositions, then summarize by depth.

    Every alpha/private/pair-private triple contains three population-labelled
    coordinates.  Each triple is converted to a centered composition, which is
    invariant to multiplying all three values by a common positive scale.
    Allelic richness has a structural baseline of one allele, so its mean
    triple is normalized after subtracting one.  This transform alone does not
    impose permutation equivariance on the fitted classifier.
    """
    table = validate_curve_table(table)
    moments = tuple(int(value) for value in moments)
    if not moments or any(value not in (0, 1, 2) for value in moments):
        raise ValueError("moments must be a nonempty subset of 0,1,2")
    values = table[:, :, 1:].reshape(len(table), table.shape[1], 9, 3)
    orbit_curves = []
    for orbit_index, start in enumerate((0, 3, 6)):
        for moment in moments:
            current = values[:, :, start : start + 3, moment].copy()
            if orbit_index == 0 and moment == 0:
                current = current - 1.0
            minimum = float(current.min())
            if minimum < -COMPOSITION_NEGATIVE_TOLERANCE:
                raise ValueError(
                    "compositional coordinates must be nonnegative after the "
                    f"alpha baseline correction; minimum={minimum:.17g}"
                )
            current = np.maximum(current, 0.0)
            total = current.sum(axis=2, keepdims=True)
            composition = np.divide(
                current,
                total,
                out=np.full_like(current, 1.0 / 3.0),
                where=total > 1e-15,
            )
            orbit_curves.append(composition - 1.0 / 3.0)
    curves = np.concatenate(orbit_curves, axis=2)
    return _summarize_curves(curves)


def representation_features(table: np.ndarray, name: str) -> np.ndarray:
    try:
        spec = REPRESENTATIONS[name]
    except KeyError as exc:
        raise ValueError(f"unknown representation {name!r}") from exc
    if spec["kind"] == "raw":
        return raw_summary(table, spec["moments"])
    if spec["kind"] == "orbit_composition":
        return orbit_composition_summary(table, spec["moments"])
    raise AssertionError(f"unhandled representation kind {spec['kind']!r}")


def representation_columns(name: str) -> list[str]:
    spec = REPRESENTATIONS[name]
    moments = tuple(spec["moments"])
    if spec["kind"] == "raw":
        blocks = (
            "alpha_1", "alpha_2", "alpha_3",
            "pi_1", "pi_2", "pi_3",
            "pihat_12", "pihat_13", "pihat_23",
        )
        base = [f"{block}_{MOMENT_NAMES[moment]}" for block in blocks for moment in moments]
    else:
        components = {
            "alpha": ("1", "2", "3"),
            "private": ("1", "2", "3"),
            "pair_private": ("12", "13", "23"),
        }
        base = [
            f"{orbit}_{component}_{MOMENT_NAMES[moment]}"
            for orbit in ORBIT_NAMES
            for moment in moments
            for component in components[orbit]
        ]
    return [f"depth_mean__{value}" for value in base] + [
        f"depth_sd__{value}" for value in base
    ]


def load_canonical(directory: Path, max_depth: int) -> dict:
    paths = {
        name: directory / name
        for name in ("X.npy", "direction.npy", "groups.npy", "magnitude.npy", "design.npy")
    }
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    X = np.load(paths["X.npy"], mmap_mode="r")
    direction = np.load(paths["direction.npy"], mmap_mode="r").astype("U2")
    groups = np.load(paths["groups.npy"], mmap_mode="r").astype("U80")
    magnitude = np.load(paths["magnitude.npy"], mmap_mode="r").astype(float)
    design = np.load(paths["design.npy"], mmap_mode="r").astype("U16")
    if not (len(X) == len(direction) == len(groups) == len(magnitude) == len(design)):
        raise AssertionError("canonical array row counts differ")
    unique, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    per = np.bincount(inverse)
    if per.min() != 198 or per.max() != 198:
        raise AssertionError("canonical replicates must each contain 198 depths")
    depth = np.asarray(X[:, 0], dtype=float)
    selected = np.flatnonzero(depth <= max_depth)
    selected_order = np.lexsort((depth[selected], inverse[selected]))
    rows = selected[selected_order]
    expected_per = max_depth - 1
    selected_per = np.bincount(inverse[rows], minlength=len(unique))
    if selected_per.min() != expected_per or selected_per.max() != expected_per:
        raise AssertionError(f"canonical curves do not all contain g=2..{max_depth}")
    table = np.asarray(X[rows], dtype=float).reshape(len(unique), expected_per, 28)
    expected_depths = np.arange(2, max_depth + 1, dtype=float)
    if not np.all(table[:, :, 0] == expected_depths[None, :]):
        raise AssertionError("canonical selected depth grid changed")
    labels = direction[first]
    rates = magnitude[first]
    designs = design[first]
    vocabulary = set(map(str, labels))
    if vocabulary != {"A", "B", "C", "D"}:
        raise AssertionError(
            f"canonical direction vocabulary changed: {sorted(vocabulary)}"
        )
    for values, label in ((direction, "direction"), (magnitude, "magnitude"), (design, "design")):
        if np.any(values != values[first[inverse]]):
            raise AssertionError(f"{label} varies within a canonical replicate")
    audit = {
        "directory": str(directory.resolve()),
        "replicates": int(len(unique)),
        "rows": int(len(X)),
        "rows_per_replicate": 198,
        "selected_depths": expected_depths.astype(int).tolist(),
        "selected_curve_shape": list(table.shape),
        "array_contracts": {
            name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    return {
        "table": table,
        "group_ids": unique.astype("U80"),
        "labels": labels,
        "rates": rates,
        "designs": designs,
        "audit": audit,
    }


def rate_family_ids(designs: Sequence[str], rates: Sequence[float]) -> np.ndarray:
    designs = np.asarray(designs).astype("U32")
    rates = np.asarray(rates, dtype=float)
    if len(designs) != len(rates):
        raise ValueError("design and rate arrays differ in length")
    return np.asarray(
        [f"{design}|{float(rate).hex()}" for design, rate in zip(designs, rates)],
        dtype="U96",
    )


def grouped_folds(
    labels: np.ndarray,
    blocks: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(labels)
    blocks = np.asarray(blocks).astype("U160")
    if len(labels) != len(blocks):
        raise ValueError("labels and blocks differ in length")
    if len(np.unique(blocks)) < n_splits:
        raise ValueError("fewer unique blocks than requested folds")
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    seen = np.zeros(len(labels), dtype=int)
    for train, test in splitter.split(np.zeros(len(labels)), labels, groups=blocks):
        train_blocks = set(blocks[train])
        test_blocks = set(blocks[test])
        if train_blocks & test_blocks:
            raise AssertionError("grouped fold shares a block across train and test")
        if len(np.unique(labels[train])) != len(CLASSES):
            raise AssertionError("grouped training fold lost a direction class")
        if len(np.unique(labels[test])) != len(CLASSES):
            raise AssertionError("grouped test fold lost a direction class")
        seen[test] += 1
        folds.append((train.astype(int), test.astype(int)))
    if not np.all(seen == 1):
        raise AssertionError("grouped test folds are not a partition")
    return folds


def _fit_model(features: np.ndarray, labels: np.ndarray, C: float):
    scaler = StandardScaler().fit(features)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(C=float(C), max_iter=3000, solver="lbfgs").fit(
            scaler.transform(features), labels
        )
    convergence = [warning for warning in caught if issubclass(warning.category, ConvergenceWarning)]
    if convergence or np.any(np.asarray(model.n_iter_) >= model.max_iter):
        raise RuntimeError(
            f"logistic regression did not converge for C={float(C):g}; "
            f"n_iter={np.asarray(model.n_iter_).astype(int).tolist()}"
        )
    return scaler, model


def _choose_C(
    features: np.ndarray,
    labels: np.ndarray,
    blocks: np.ndarray,
    *,
    grid: Sequence[float],
    seed: int,
    n_splits: int,
    compute_state: Path | None = None,
) -> tuple[float, dict]:
    grid = validate_C_grid(grid)
    folds = grouped_folds(labels, blocks, n_splits=n_splits, seed=seed)
    fold_ledger = []
    for fold_index, (train, test) in enumerate(folds):
        train_blocks = sorted(set(map(str, np.asarray(blocks)[train])))
        test_blocks = sorted(set(map(str, np.asarray(blocks)[test])))
        overlap = set(train_blocks) & set(test_blocks)
        if overlap:
            raise AssertionError("inner model-selection fold shares blocks")
        fold_ledger.append({
            "fold": int(fold_index),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_block_sha256": sha256_text(train_blocks),
            "test_block_sha256": sha256_text(test_blocks),
            "block_overlap": False,
        })
    scores = {}
    for C in grid:
        fold_scores = []
        for train, test in folds:
            if compute_state is not None:
                compute_gate(compute_state)
            scaler, model = _fit_model(features[train], labels[train], C)
            prediction = model.predict(scaler.transform(features[test]))
            fold_scores.append(float(balanced_accuracy_score(labels[test], prediction)))
        scores[float(C)] = fold_scores
    means = {C: float(np.mean(values)) for C, values in scores.items()}
    # Prefer stronger regularization (smaller C) on exact ties.
    selected = sorted(means, key=lambda C: (-means[C], C))[0]
    return float(selected), {
        "criterion": "mean inner-fold balanced accuracy; smaller C wins exact ties",
        "fold_ledger": fold_ledger,
        "scores": {
            f"{C:g}": {"folds": scores[C], "mean": means[C]}
            for C in sorted(scores)
        },
        "selected_C": float(selected),
    }


def _class_counts(values: np.ndarray) -> dict:
    return {str(label): int(np.sum(values == index)) for index, label in enumerate(CLASSES)}


def _balanced_accuracy_without_spurious_class_warning(
    truth: np.ndarray,
    prediction: np.ndarray,
) -> float:
    """Average recall over classes present in truth; extra predictions are errors."""
    truth = np.asarray(truth)
    prediction = np.asarray(prediction)
    present = np.unique(truth)
    if not len(present):
        raise ValueError("balanced accuracy requires at least one true row")
    return float(np.mean([
        np.mean(prediction[truth == label] == label)
        for label in present
    ]))


def _equal_rate_family_summary(
    labels: np.ndarray,
    probability: np.ndarray,
    rates: np.ndarray,
    designs: np.ndarray,
    family_ids: np.ndarray,
) -> dict:
    """Macro-average appreciable performance over exact rate families."""
    labels = np.asarray(labels)
    probability = np.asarray(probability, dtype=float)
    rates = np.asarray(rates, dtype=float)
    designs = np.asarray(designs)
    family_ids = np.asarray(family_ids)
    if not (
        len(labels) == len(probability) == len(rates) == len(designs) == len(family_ids)
    ):
        raise ValueError("equal-family metric arrays differ in length")
    appreciable = rates >= APPRECIABLE
    prediction = probability.argmax(axis=1)
    family_scores = []
    for family in sorted(np.unique(family_ids[appreciable]), key=str):
        all_family = family_ids == family
        use = all_family & appreciable
        if not np.array_equal(use, all_family):
            raise AssertionError("only part of an exact rate family is appreciable")
        family_rates = np.unique(rates[use])
        family_designs = np.unique(designs[use])
        if len(family_rates) != 1 or len(family_designs) != 1:
            raise AssertionError("exact rate family mixes rate or design metadata")
        score = _balanced_accuracy_without_spurious_class_warning(
            labels[use], prediction[use]
        )
        family_scores.append({
            "family_id": str(family),
            "design": str(family_designs[0]),
            "rate": float(family_rates[0]),
            "rate_hex": float(family_rates[0]).hex(),
            "n_rows": int(use.sum()),
            "class_counts": _class_counts(labels[use]),
            "balanced_accuracy": score,
        })
    by_design = {}
    for design in sorted({row["design"] for row in family_scores}):
        scores = [
            row["balanced_accuracy"] for row in family_scores if row["design"] == design
        ]
        by_design[design] = {
            "families": int(len(scores)),
            "balanced_accuracy": float(np.mean(scores)),
        }
    scores = [row["balanced_accuracy"] for row in family_scores]
    return {
        "metric": "equal-weight mean of within-exact-rate-family balanced accuracy",
        "n_rows": int(appreciable.sum()),
        "n_families": int(len(family_scores)),
        "balanced_accuracy": None if not scores else float(np.mean(scores)),
        "median_family_balanced_accuracy": (
            None if not scores else float(np.median(scores))
        ),
        "minimum_family_balanced_accuracy": (
            None if not scores else float(np.min(scores))
        ),
        "family_scores": family_scores,
        "by_design": by_design,
    }


def _probability_summary(
    labels: np.ndarray,
    probability: np.ndarray,
    rates: np.ndarray,
    designs: np.ndarray,
    evaluation_families: np.ndarray | None = None,
) -> dict:
    prediction = probability.argmax(axis=1)

    def metrics(mask: np.ndarray) -> dict:
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            return {"n": 0}
        truth = labels[mask]
        pred = prediction[mask]
        prob = probability[mask]
        return {
            "n": int(mask.sum()),
            "accuracy": float(np.mean(pred == truth)),
            "balanced_accuracy": _balanced_accuracy_without_spurious_class_warning(
                truth, pred
            ),
            "macro_f1": float(
                f1_score(
                    truth,
                    pred,
                    average="macro",
                    labels=np.arange(3),
                    zero_division=0,
                )
            ),
            "multiclass_log_loss": float(log_loss(truth, prob, labels=np.arange(3))),
            "multiclass_brier": float(np.mean(np.sum((prob - np.eye(3)[truth]) ** 2, axis=1))),
            "prediction_counts": _class_counts(pred),
        }

    result = {
        "overall": metrics(np.ones(len(labels), dtype=bool)),
        "appreciable": metrics(rates >= APPRECIABLE),
        "B_or_C_orientation": metrics(np.isin(labels, (1, 2))),
        "fixed_design": metrics(designs == "fixed"),
        "continuous_design": metrics(designs == "continuous"),
        "fixed_rates": {},
    }
    for rate in sorted(np.unique(rates[designs == "fixed"])):
        result["fixed_rates"][float(rate).hex()] = {
            "rate": float(rate),
            **metrics((designs == "fixed") & (rates == rate)),
        }
    if evaluation_families is not None:
        result["appreciable_equal_rate_family"] = _equal_rate_family_summary(
            labels,
            probability,
            rates,
            designs,
            np.asarray(evaluation_families),
        )
    return result


def nested_oof(
    features: np.ndarray,
    labels: np.ndarray,
    rates: np.ndarray,
    designs: np.ndarray,
    group_ids: np.ndarray,
    outer_blocks: np.ndarray,
    *,
    outer_name: str,
    seeds: Sequence[int],
    C_grid: Sequence[float],
    outer_splits: int,
    inner_splits: int,
    evaluation_families: np.ndarray | None = None,
    compute_state: Path | None = None,
) -> dict:
    if evaluation_families is not None:
        evaluation_families = np.asarray(evaluation_families)
        if len(evaluation_families) != len(labels):
            raise ValueError("evaluation-family IDs differ from OOF rows")
    repeats = []
    probability_repeats = []
    distance_repeats = []
    fold_ledgers = []
    per_repeat_oof = []
    support_references = []
    for repeat_index, seed in enumerate(seeds):
        if compute_state is not None:
            compute_gate(compute_state)
        probability = np.full((len(labels), len(CLASSES)), np.nan, dtype=float)
        distance = np.full(len(labels), np.nan, dtype=float)
        outer_fold = np.full(len(labels), -1, dtype=np.int16)
        folds = grouped_folds(labels, outer_blocks, n_splits=outer_splits, seed=int(seed))
        repeat_ledger = []
        for fold_index, (train, test) in enumerate(folds):
            if compute_state is not None:
                compute_gate(compute_state)
            selected_C, inner = _choose_C(
                features[train],
                labels[train],
                outer_blocks[train],
                grid=C_grid,
                seed=int(seed) * 1000 + fold_index + 17,
                n_splits=inner_splits,
                compute_state=compute_state,
            )
            scaler, model = _fit_model(features[train], labels[train], selected_C)
            transformed = scaler.transform(features[test])
            if np.any(outer_fold[test] != -1):
                raise AssertionError("nested OOF row assigned to multiple outer folds")
            outer_fold[test] = int(fold_index)
            probability[test] = model.predict_proba(transformed)
            distance[test] = np.sqrt(np.mean(transformed**2, axis=1))
            train_blocks = sorted(set(map(str, outer_blocks[train])))
            test_blocks = sorted(set(map(str, outer_blocks[test])))
            repeat_ledger.append({
                "fold": int(fold_index),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_blocks": int(len(train_blocks)),
                "test_blocks": int(len(test_blocks)),
                "train_block_sha256": sha256_text(train_blocks),
                "test_block_sha256": sha256_text(test_blocks),
                "block_overlap": False,
                "train_class_counts": _class_counts(labels[train]),
                "test_class_counts": _class_counts(labels[test]),
                "selected_C": selected_C,
                "inner_selection": inner,
                "scaler_fit_rows": int(len(train)),
                "model_fit_rows": int(len(train)),
                "model_n_iter": np.asarray(model.n_iter_).astype(int).tolist(),
                "convergence_warning": False,
            })
            support_references.append({
                "repeat": int(repeat_index),
                "seed": int(seed),
                "fold": int(fold_index),
                "feature_dimension": int(features.shape[1]),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_block_sha256": sha256_text(train_blocks),
                "test_block_sha256": sha256_text(test_blocks),
                "scaler_mean": np.asarray(scaler.mean_, dtype=float).tolist(),
                "scaler_scale": np.asarray(scaler.scale_, dtype=float).tolist(),
                "source_test_rms_z_p99": float(
                    np.quantile(distance[test], 0.99, method="linear")
                ),
            })
        if (
            not np.isfinite(probability).all()
            or not np.isfinite(distance).all()
            or np.any(outer_fold < 0)
        ):
            raise AssertionError("nested OOF prediction ledger is incomplete")
        repeats.append(_probability_summary(
            labels,
            probability,
            rates,
            designs,
            evaluation_families=evaluation_families,
        ))
        probability_repeats.append(probability)
        distance_repeats.append(distance)
        repeat_payload = {
            "repeat": int(repeat_index),
            "seed": int(seed),
            "outer_fold": outer_fold.astype(int).tolist(),
            "probability": probability.tolist(),
            "prediction_index": probability.argmax(axis=1).astype(int).tolist(),
            "rms_z": distance.tolist(),
        }
        repeat_payload["payload_sha256"] = hashlib.sha256(
            _canonical_json(repeat_payload)
        ).hexdigest()
        per_repeat_oof.append(repeat_payload)
        fold_ledgers.append({
            "repeat": int(repeat_index),
            "seed": int(seed),
            "folds": repeat_ledger,
        })
    probabilities = np.stack(probability_repeats)
    distances = np.stack(distance_repeats)
    mean_probability = probabilities.mean(axis=0)
    mean_distance = distances.mean(axis=0)
    selected_counts = Counter(
        f"{fold['selected_C']:g}"
        for repeat in fold_ledgers
        for fold in repeat["folds"]
    )
    selected_values = [
        float(fold["selected_C"])
        for repeat in fold_ledgers
        for fold in repeat["folds"]
    ]
    grid_max = float(max(C_grid))
    maximum_selected = int(sum(value == grid_max for value in selected_values))
    oof_rows = []
    for index in range(len(labels)):
        oof_rows.append({
            "group_id": str(group_ids[index]),
            "outer_block": str(outer_blocks[index]),
            "truth": str(CLASSES[labels[index]]),
            "rate": float(rates[index]),
            "rate_hex": float(rates[index]).hex(),
            "design": str(designs[index]),
            "mean_probability": {
                str(label): float(value)
                for label, value in zip(CLASSES, mean_probability[index])
            },
            "mean_prediction": str(CLASSES[int(np.argmax(mean_probability[index]))]),
            "mean_rms_z": float(mean_distance[index]),
        })
    oof_digest = hashlib.sha256(_canonical_json(oof_rows)).hexdigest()
    row_contract = [
        [
            str(group_ids[index]),
            str(outer_blocks[index]),
            str(CLASSES[labels[index]]),
            float(rates[index]).hex(),
            str(designs[index]),
        ]
        for index in range(len(labels))
    ]
    result = {
        "outer_estimand": outer_name,
        "outer_splits": int(outer_splits),
        "inner_splits": int(inner_splits),
        "repeat_seeds": [int(seed) for seed in seeds],
        "C_grid": [float(value) for value in C_grid],
        "per_repeat_metrics": repeats,
        "mean_probability_metrics": _probability_summary(
            labels,
            mean_probability,
            rates,
            designs,
            evaluation_families=evaluation_families,
        ),
        "selected_C_counts": dict(sorted(selected_counts.items())),
        "C_grid_diagnostic": {
            "minimum": float(min(C_grid)),
            "maximum": grid_max,
            "fold_selections": int(len(selected_values)),
            "maximum_selected": maximum_selected,
            "maximum_selected_fraction": float(maximum_selected / len(selected_values)),
            "C_grid_ceiling_reached": bool(maximum_selected > 0),
            "interpretation": (
                "Any maximum selection is a capacity-boundary diagnostic; expand the grid "
                "before interpreting a capacity-limited comparison."
            ),
        },
        "fold_ledger": fold_ledgers,
        "oof_row_contract": {
            "class_order": CLASSES.tolist(),
            "rows": int(len(labels)),
            "row_key_sha256": hashlib.sha256(_canonical_json(row_contract)).hexdigest(),
        },
        "per_repeat_oof": per_repeat_oof,
        "per_repeat_oof_sha256": hashlib.sha256(
            _canonical_json([row["payload_sha256"] for row in per_repeat_oof])
        ).hexdigest(),
        "oof_predictions": oof_rows,
        "oof_predictions_sha256": oof_digest,
        "crossfit_source_rms_z_descriptive": {
            "n": int(mean_distance.size),
            "median": float(np.median(mean_distance)),
            "p95": float(np.quantile(mean_distance, 0.95)),
            "p99": float(np.quantile(mean_distance, 0.99)),
            "maximum": float(np.max(mean_distance)),
            "values": mean_distance.tolist(),
        },
        "crossfit_support_reference": {
            "metric": "RMS standardized distance over representation features",
            "source_test_quantile": 0.99,
            "quantile_method": "linear",
            "folds": support_references,
            "folds_sha256": hashlib.sha256(
                _canonical_json(support_references)
            ).hexdigest(),
            "guardrail": (
                "Fold-local source-test p99 comparisons are descriptive OOD diagnostics, "
                "not conformal coverage, calibrated probabilities, or p-values."
            ),
        },
    }
    validate_nested_oof_payload(
        result,
        labels,
        rates,
        designs,
        group_ids,
        outer_blocks,
        evaluation_families=evaluation_families,
        features=features,
    )
    return result


def validate_nested_oof_payload(
    result: dict,
    labels: np.ndarray,
    rates: np.ndarray,
    designs: np.ndarray,
    group_ids: np.ndarray,
    outer_blocks: np.ndarray,
    *,
    evaluation_families: np.ndarray | None = None,
    features: np.ndarray | None = None,
) -> None:
    """Semantically validate the persisted repeat-level OOF and fold ledger."""
    labels = np.asarray(labels)
    rates = np.asarray(rates, dtype=float)
    designs = np.asarray(designs)
    group_ids = np.asarray(group_ids)
    outer_blocks = np.asarray(outer_blocks)
    n_rows = len(labels)
    if not all(len(array) == n_rows for array in (rates, designs, group_ids, outer_blocks)):
        raise RuntimeError("nested OOF validation arrays differ in length")
    if evaluation_families is not None:
        evaluation_families = np.asarray(evaluation_families)
        if len(evaluation_families) != n_rows:
            raise RuntimeError("nested OOF evaluation families differ in length")
    if features is not None:
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or len(features) != n_rows or not np.isfinite(features).all():
            raise RuntimeError("nested OOF validation feature matrix is invalid")
    row_contract = [
        [
            str(group_ids[index]),
            str(outer_blocks[index]),
            str(CLASSES[labels[index]]),
            float(rates[index]).hex(),
            str(designs[index]),
        ]
        for index in range(n_rows)
    ]
    expected_row_hash = hashlib.sha256(_canonical_json(row_contract)).hexdigest()
    contract = result.get("oof_row_contract", {})
    if (
        contract.get("class_order") != CLASSES.tolist()
        or contract.get("rows") != n_rows
        or contract.get("row_key_sha256") != expected_row_hash
    ):
        raise RuntimeError("nested OOF row contract changed")
    seeds = [int(seed) for seed in result.get("repeat_seeds", [])]
    payloads = result.get("per_repeat_oof", [])
    fold_ledgers = result.get("fold_ledger", [])
    saved_metrics = result.get("per_repeat_metrics", [])
    if not (len(payloads) == len(fold_ledgers) == len(saved_metrics) == len(seeds)):
        raise RuntimeError("nested OOF repeat ledgers differ in length")
    expected_combined = hashlib.sha256(
        _canonical_json([payload.get("payload_sha256") for payload in payloads])
    ).hexdigest()
    if result.get("per_repeat_oof_sha256") != expected_combined:
        raise RuntimeError("nested OOF combined repeat hash changed")

    probabilities = []
    distances = []
    outer_splits = int(result["outer_splits"])
    for repeat_index, (seed, payload, ledger) in enumerate(
        zip(seeds, payloads, fold_ledgers)
    ):
        without_hash = {
            key: value for key, value in payload.items() if key != "payload_sha256"
        }
        expected_hash = hashlib.sha256(_canonical_json(without_hash)).hexdigest()
        if payload.get("payload_sha256") != expected_hash:
            raise RuntimeError(f"nested OOF repeat {repeat_index} payload hash changed")
        if payload.get("repeat") != repeat_index or payload.get("seed") != seed:
            raise RuntimeError("nested OOF repeat identity changed")
        probability = np.asarray(payload.get("probability"), dtype=float)
        distance = np.asarray(payload.get("rms_z"), dtype=float)
        prediction = np.asarray(payload.get("prediction_index"), dtype=int)
        outer_fold = np.asarray(payload.get("outer_fold"), dtype=int)
        if (
            probability.shape != (n_rows, len(CLASSES))
            or distance.shape != (n_rows,)
            or prediction.shape != (n_rows,)
            or outer_fold.shape != (n_rows,)
        ):
            raise RuntimeError("nested OOF repeat payload shape changed")
        if (
            not np.isfinite(probability).all()
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12, rtol=0)
            or np.any(probability < 0)
            or not np.isfinite(distance).all()
            or np.any(distance < 0)
            or not np.array_equal(prediction, probability.argmax(axis=1))
            or np.any((outer_fold < 0) | (outer_fold >= outer_splits))
        ):
            raise RuntimeError("nested OOF repeat payload values are invalid")
        for block in np.unique(outer_blocks):
            if len(np.unique(outer_fold[outer_blocks == block])) != 1:
                raise RuntimeError("nested OOF outer block crosses folds")
        if (
            ledger.get("repeat") != repeat_index
            or ledger.get("seed") != seed
            or len(ledger.get("folds", [])) != outer_splits
        ):
            raise RuntimeError("nested OOF fold ledger identity changed")
        for fold_index, fold in enumerate(ledger["folds"]):
            test = np.flatnonzero(outer_fold == fold_index)
            train = np.flatnonzero(outer_fold != fold_index)
            train_blocks = sorted(set(map(str, outer_blocks[train])))
            test_blocks = sorted(set(map(str, outer_blocks[test])))
            expected_fields = {
                "fold": int(fold_index),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_blocks": int(len(train_blocks)),
                "test_blocks": int(len(test_blocks)),
                "train_block_sha256": sha256_text(train_blocks),
                "test_block_sha256": sha256_text(test_blocks),
                "train_class_counts": _class_counts(labels[train]),
                "test_class_counts": _class_counts(labels[test]),
            }
            if any(fold.get(key) != value for key, value in expected_fields.items()):
                raise RuntimeError("nested OOF fold ledger does not match row assignments")
            if set(train_blocks) & set(test_blocks) or fold.get("block_overlap") is not False:
                raise RuntimeError("nested OOF fold ledger contains block overlap")
        recomputed = _probability_summary(
            labels,
            probability,
            rates,
            designs,
            evaluation_families=evaluation_families,
        )
        if _canonical_json(recomputed) != _canonical_json(saved_metrics[repeat_index]):
            raise RuntimeError("nested OOF repeat metrics do not match saved probabilities")
        probabilities.append(probability)
        distances.append(distance)

    probability_mean = np.mean(np.stack(probabilities), axis=0)
    distance_mean = np.mean(np.stack(distances), axis=0)
    recomputed_mean = _probability_summary(
        labels,
        probability_mean,
        rates,
        designs,
        evaluation_families=evaluation_families,
    )
    if _canonical_json(recomputed_mean) != _canonical_json(result["mean_probability_metrics"]):
        raise RuntimeError("nested OOF mean metrics do not match repeat probabilities")
    expected_rows = []
    for index in range(n_rows):
        expected_rows.append({
            "group_id": str(group_ids[index]),
            "outer_block": str(outer_blocks[index]),
            "truth": str(CLASSES[labels[index]]),
            "rate": float(rates[index]),
            "rate_hex": float(rates[index]).hex(),
            "design": str(designs[index]),
            "mean_probability": {
                str(label): float(value)
                for label, value in zip(CLASSES, probability_mean[index])
            },
            "mean_prediction": str(CLASSES[int(np.argmax(probability_mean[index]))]),
            "mean_rms_z": float(distance_mean[index]),
        })
    if _canonical_json(expected_rows) != _canonical_json(result.get("oof_predictions")):
        raise RuntimeError("nested OOF mean row ledger changed")
    expected_rows_hash = hashlib.sha256(_canonical_json(expected_rows)).hexdigest()
    if result.get("oof_predictions_sha256") != expected_rows_hash:
        raise RuntimeError("nested OOF mean row hash changed")
    saved_distances = np.asarray(
        result.get("crossfit_source_rms_z_descriptive", {}).get("values"),
        dtype=float,
    )
    if not np.array_equal(saved_distances, distance_mean):
        raise RuntimeError("nested OOF mean RMS-z ledger changed")

    support = result.get("crossfit_support_reference", {})
    references = support.get("folds", [])
    if len(references) != len(seeds) * outer_splits:
        raise RuntimeError("cross-fit support reference is incomplete")
    if support.get("folds_sha256") != hashlib.sha256(
        _canonical_json(references)
    ).hexdigest():
        raise RuntimeError("cross-fit support reference hash changed")
    for reference in references:
        repeat_index = int(reference.get("repeat", -1))
        fold_index = int(reference.get("fold", -1))
        if not (0 <= repeat_index < len(seeds) and 0 <= fold_index < outer_splits):
            raise RuntimeError("cross-fit support reference identity is invalid")
        fold = fold_ledgers[repeat_index]["folds"][fold_index]
        mean = np.asarray(reference.get("scaler_mean"), dtype=float)
        scale = np.asarray(reference.get("scaler_scale"), dtype=float)
        p99 = float(reference.get("source_test_rms_z_p99", math.nan))
        if (
            mean.ndim != 1
            or scale.shape != mean.shape
            or len(mean) != int(reference.get("feature_dimension", -1))
            or not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
            or not math.isfinite(p99)
            or p99 <= 0
            or reference.get("train_rows") != fold["train_rows"]
            or reference.get("test_rows") != fold["test_rows"]
            or reference.get("train_block_sha256") != fold["train_block_sha256"]
            or reference.get("test_block_sha256") != fold["test_block_sha256"]
        ):
            raise RuntimeError("cross-fit support reference values are invalid")
        if features is not None:
            outer_fold = np.asarray(
                payloads[repeat_index]["outer_fold"], dtype=int
            )
            train = outer_fold != fold_index
            test = outer_fold == fold_index
            fitted = StandardScaler().fit(features[train])
            expected_distance = np.sqrt(
                np.mean(fitted.transform(features[test]) ** 2, axis=1)
            )
            expected_p99 = float(np.quantile(expected_distance, 0.99, method="linear"))
            if (
                not np.allclose(mean, fitted.mean_, atol=1e-12, rtol=0)
                or not np.allclose(scale, fitted.scale_, atol=1e-12, rtol=0)
                or not math.isclose(p99, expected_p99, abs_tol=1e-12, rel_tol=0)
            ):
                raise RuntimeError("cross-fit support reference does not match source features")


def design_transfer(
    features: np.ndarray,
    labels: np.ndarray,
    rates: np.ndarray,
    designs: np.ndarray,
    group_ids: np.ndarray,
    *,
    C_grid: Sequence[float],
    inner_splits: int,
    seed: int,
    compute_state: Path | None = None,
) -> dict:
    result = {}
    for train_design, test_design in (("continuous", "fixed"), ("fixed", "continuous")):
        if compute_state is not None:
            compute_gate(compute_state)
        train = np.flatnonzero(designs == train_design)
        test = np.flatnonzero(designs == test_design)
        overlap = set(map(str, group_ids[train])) & set(map(str, group_ids[test]))
        if overlap:
            raise AssertionError(
                f"fixed/continuous transfer shares {len(overlap)} genealogy groups"
            )
        selected_C, inner = _choose_C(
            features[train],
            labels[train],
            group_ids[train],
            grid=C_grid,
            seed=seed,
            n_splits=inner_splits,
            compute_state=compute_state,
        )
        scaler, model = _fit_model(features[train], labels[train], selected_C)
        transformed = scaler.transform(features[test])
        probability = model.predict_proba(transformed)
        result[f"{train_design}_to_{test_design}"] = {
            "selected_C": selected_C,
            "inner_selection": inner,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_group_sha256": sha256_text(sorted(map(str, group_ids[train]))),
            "test_group_sha256": sha256_text(sorted(map(str, group_ids[test]))),
            "group_overlap": bool(overlap),
            "model_n_iter": np.asarray(model.n_iter_).astype(int).tolist(),
            "convergence_warning": False,
            "metrics": _probability_summary(
                labels[test], probability, rates[test], designs[test]
            ),
            "test_rms_z": {
                "median": float(np.median(np.sqrt(np.mean(transformed**2, axis=1)))),
                "p95": float(np.quantile(np.sqrt(np.mean(transformed**2, axis=1)), 0.95)),
                "maximum": float(np.max(np.sqrt(np.mean(transformed**2, axis=1)))),
            },
        }
    return result


def _contains_true_accuracy_flag(value) -> bool:
    if isinstance(value, dict):
        for key, current in value.items():
            lowered = str(key).lower()
            if current is True and (
                "accuracy_eligible" in lowered or "accuracy_available" in lowered
            ):
                return True
            if _contains_true_accuracy_flag(current):
                return True
    elif isinstance(value, list):
        return any(_contains_true_accuracy_flag(current) for current in value)
    return False


def pinned_natural_paths(repo: Path = REPO) -> list[Path]:
    """Return the frozen exploratory natural cohort without filesystem globbing."""
    paths = [(repo / relative).resolve() for relative in PINNED_NATURAL_RESULTS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("pinned natural result is missing: " + ", ".join(missing))
    return paths


def load_natural_panels(paths: Sequence[Path], max_depth: int) -> tuple[list[dict], dict]:
    panels = []
    source_audit = []
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("natural result path list contains duplicates")
    seen_rows = set()
    expected_depths = np.arange(2, max_depth + 1, dtype=float)
    for path in sorted(resolved):
        raw = path.read_bytes()
        result = json.loads(raw)
        if _contains_true_accuracy_flag(result):
            raise AssertionError(f"{path}: natural bundle contains an accuracy-eligible flag")
        current = result.get("panels", [])
        head = result.get("direction_head", result.get("simulation_head_audit", {}))
        if int(head.get("dimension", -1)) != 54:
            raise AssertionError(f"{path}: source direction-head dimension is not 54")
        if list(head.get("depth_grid", [])) != expected_depths.astype(int).tolist():
            raise AssertionError(f"{path}: source direction-head depth grid changed")
        representation = str(head.get("representation", ""))
        if "27 non-depth coordinates" not in representation:
            raise AssertionError(f"{path}: source direction-head feature contract changed")
        source_audit.append({
            "path": str(path.resolve()),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "schema_version": result.get("schema_version"),
            "panel_rows": int(len(current)),
        })
        for panel in current:
            panel_id = str(panel["panel_id"])
            row_key = (path.parent.name, panel_id)
            if row_key in seen_rows:
                raise AssertionError(f"duplicate natural row {row_key}")
            seen_rows.add(row_key)
            matrix = panel.get("feature_matrix")
            if matrix is None:
                continue
            matrix = np.asarray(matrix, dtype=float)
            if matrix.shape != (max_depth - 1, 28):
                raise AssertionError(
                    f"{path}:{panel.get('panel_id')}: feature shape {matrix.shape} "
                    f"!= ({max_depth - 1}, 28)"
                )
            if not np.isfinite(matrix).all():
                raise AssertionError(f"{path}:{panel_id}: feature matrix is non-finite")
            if not np.array_equal(matrix[:, 0], expected_depths):
                raise AssertionError(f"{path}:{panel_id}: feature depth column changed")
            population_order = panel.get("population_order")
            if not isinstance(population_order, dict):
                raise AssertionError(f"{path}:{panel_id}: missing population_order mapping")
            population_values = [str(population_order.get(name, "")) for name in ("P1", "P2", "P3")]
            if any(not value for value in population_values) or len(set(population_values)) != 3:
                raise AssertionError(f"{path}:{panel_id}: invalid or duplicate population order")
            expectation = panel.get("external_expectation", {})
            adjudication = panel.get("adjudication", {})
            candidate = expectation.get("candidate_class", adjudication.get("candidate_class"))
            if candidate not in set(CLASSES):
                candidate = None
            panels.append({
                "bundle": path.parent.name,
                "source_result": str(path.resolve()),
                "panel_id": panel_id,
                "population_order": population_order,
                "matrix": matrix,
                "n_loci": int(panel.get("padze", {}).get("n_loci_kept", 0)),
                "candidate_class": candidate,
                "source_raw_prediction": panel.get("simulation_head", {}).get("predicted_class"),
                "source_direction_rms_z": panel.get("simulation_feature_shift", {}).get("rms_z"),
                "any_true_accuracy_flag": False,
            })
    if not panels:
        raise ValueError("no standardized natural panels were found")
    if any(panel["any_true_accuracy_flag"] for panel in panels):
        raise AssertionError("natural pilot unexpectedly found an accuracy-eligible row")
    audit = {
        "result_bundles": int(len(source_audit)),
        "panel_rows": int(len(panels)),
        "sources": source_audit,
        "accuracy_eligible_rows": 0,
        "guardrail": (
            "Natural rows are correlated study/filter sensitivities with no formal accuracy "
            "denominator; candidate matches are descriptive only."
        ),
    }
    return panels, audit


def final_natural_score(
    features: np.ndarray,
    labels: np.ndarray,
    rate_blocks: np.ndarray,
    panels: list[dict],
    *,
    representation: str,
    C_grid: Sequence[float],
    inner_splits: int,
    seed: int,
    crossfit_support_reference: dict | None = None,
    compute_state: Path | None = None,
    verify_source_raw_head: bool = True,
) -> dict:
    if compute_state is not None:
        compute_gate(compute_state)
    if representation == "raw_all" and verify_source_raw_head:
        selected_C = 1.0
        selection = {
            "policy": "fixed canonical C=1 for exact current-head comparison",
            "selected_C": 1.0,
        }
    else:
        selected_C, selection = _choose_C(
            features,
            labels,
            rate_blocks,
            grid=C_grid,
            seed=seed,
            n_splits=inner_splits,
            compute_state=compute_state,
        )
        selection["policy"] = "rate-family grouped simulation CV only; natural rows unseen"
    if compute_state is not None:
        compute_gate(compute_state)
    scaler, model = _fit_model(features, labels, selected_C)
    selection["final_model_n_iter"] = np.asarray(model.n_iter_).astype(int).tolist()
    selection["convergence_warning"] = False
    natural_table = np.concatenate([panel["matrix"][None, :, :] for panel in panels], axis=0)
    natural_features = representation_features(natural_table, representation)
    crossfit_rows = None
    if crossfit_support_reference is not None:
        references = crossfit_support_reference.get("folds", [])
        if not references:
            raise ValueError("cross-fit natural support reference is empty")
        if crossfit_support_reference.get("folds_sha256") != hashlib.sha256(
            _canonical_json(references)
        ).hexdigest():
            raise ValueError("cross-fit natural support reference hash changed")
        fold_distance = []
        fold_ratio = []
        for reference_row in references:
            mean = np.asarray(reference_row.get("scaler_mean"), dtype=float)
            scale = np.asarray(reference_row.get("scaler_scale"), dtype=float)
            p99 = float(reference_row.get("source_test_rms_z_p99", math.nan))
            if (
                mean.shape != (natural_features.shape[1],)
                or scale.shape != mean.shape
                or not np.isfinite(mean).all()
                or not np.isfinite(scale).all()
                or np.any(scale <= 0)
                or not math.isfinite(p99)
                or p99 <= 0
            ):
                raise ValueError("cross-fit natural support reference is invalid")
            current = np.sqrt(np.mean(((natural_features - mean) / scale) ** 2, axis=1))
            fold_distance.append(current)
            fold_ratio.append(current / p99)
        fold_distance = np.stack(fold_distance, axis=1)
        fold_ratio = np.stack(fold_ratio, axis=1)
        crossfit_rows = [
            {
                "folds": int(fold_ratio.shape[1]),
                "within_source_test_p99_folds": int(np.sum(fold_ratio[index] <= 1.0)),
                "within_source_test_p99_fraction": float(np.mean(fold_ratio[index] <= 1.0)),
                "median_rms_z": float(np.median(fold_distance[index])),
                "median_rms_z_over_source_test_p99": float(np.median(fold_ratio[index])),
                "rms_z_over_source_test_p99_by_fold": fold_ratio[index].tolist(),
            }
            for index in range(len(natural_features))
        ]
    transformed = scaler.transform(natural_features)
    probability = model.predict_proba(transformed)
    distance = np.sqrt(np.mean(transformed**2, axis=1))
    # Natural and source rows are scored by the same full-data scaler.  The
    # resulting tail fraction is explicitly descriptive: source rows helped fit
    # this scaler, so this is neither a conformal p-value nor a coverage claim.
    source_transformed = scaler.transform(features)
    reference = np.sort(np.sqrt(np.mean(source_transformed**2, axis=1)))
    if not len(reference) or not np.isfinite(reference).all():
        raise ValueError("source distance reference is empty or non-finite")
    rows = []
    for index, panel in enumerate(panels):
        prediction = str(CLASSES[int(np.argmax(probability[index]))])
        empirical_tail = float((1 + np.sum(reference >= distance[index])) / (len(reference) + 1))
        row = {
            "bundle": panel["bundle"],
            "panel_id": panel["panel_id"],
            "population_order": panel["population_order"],
            "n_loci": panel["n_loci"],
            "candidate_class": panel["candidate_class"],
            "source_raw_prediction": panel["source_raw_prediction"],
            "source_direction_rms_z": panel["source_direction_rms_z"],
            "prediction": prediction,
            "scores": {
                str(label): float(value)
                for label, value in zip(CLASSES, probability[index])
            },
            "rms_z": float(distance[index]),
            "training_reference_empirical_tail_fraction": empirical_tail,
            "within_training_reference_p99": bool(distance[index] <= np.quantile(reference, 0.99)),
            "prediction_matches_candidate": (
                None if panel["candidate_class"] is None else prediction == panel["candidate_class"]
            ),
            "formal_accuracy_eligible": False,
        }
        if crossfit_rows is not None:
            row["crossfit_source_test_support"] = crossfit_rows[index]
        rows.append(row)
    raw_reproduction = None
    if representation == "raw_all" and verify_source_raw_head:
        comparable = [
            row for row in rows
            if row["source_raw_prediction"] in set(CLASSES)
            and row["source_direction_rms_z"] is not None
        ]
        mismatches = [
            f"{row['bundle']}:{row['panel_id']}"
            for row in comparable
            if row["prediction"] != row["source_raw_prediction"]
        ]
        differences = np.array([
            abs(row["rms_z"] - float(row["source_direction_rms_z"]))
            for row in comparable
        ])
        tolerance = 1e-9
        maximum_difference = float(differences.max()) if len(differences) else None
        if mismatches or (maximum_difference is not None and maximum_difference > tolerance):
            raise AssertionError(
                "raw_all failed to reproduce stored source heads: "
                f"prediction mismatches={mismatches[:5]}, max RMS-z difference={maximum_difference}"
            )
        raw_reproduction = {
            "available_rows": int(len(comparable)),
            "prediction_mismatches": 0,
            "rms_z_absolute_tolerance": tolerance,
            "maximum_absolute_rms_z_difference": maximum_difference,
            "status": "exact predictions and RMS-z within tolerance",
        }
    elif representation == "raw_all":
        raw_reproduction = {
            "status": "not applicable: this model was retrained on a different simulation bank"
        }
    candidate_rows = [row for row in rows if row["candidate_class"] is not None]
    counts = Counter(row["prediction"] for row in rows)
    rms = np.asarray([row["rms_z"] for row in rows])
    bundle_rows = {
        bundle: [row for row in rows if row["bundle"] == bundle]
        for bundle in sorted({row["bundle"] for row in rows})
    }
    bundle_summary = {
        bundle: {
            "rows": int(len(current)),
            "median_rms_z": float(np.median([row["rms_z"] for row in current])),
            **(
                {
                    "median_crossfit_rms_z_over_source_test_p99": float(np.median([
                        row["crossfit_source_test_support"][
                            "median_rms_z_over_source_test_p99"
                        ]
                        for row in current
                    ])),
                    "median_crossfit_within_source_test_p99_fraction": float(np.median([
                        row["crossfit_source_test_support"][
                            "within_source_test_p99_fraction"
                        ]
                        for row in current
                    ])),
                }
                if crossfit_rows is not None
                else {}
            ),
        }
        for bundle, current in bundle_rows.items()
    }
    bundle_medians = np.asarray([
        bundle_summary[bundle]["median_rms_z"] for bundle in sorted(bundle_summary)
    ])
    crossfit_bundle_pass_fraction = None
    crossfit_bundle_summary = None
    if crossfit_rows is not None:
        bundle_pass = {
            bundle: bool(
                summary["median_crossfit_rms_z_over_source_test_p99"] <= 1.0
            )
            for bundle, summary in bundle_summary.items()
        }
        crossfit_bundle_pass_fraction = float(np.mean(list(bundle_pass.values())))
        crossfit_bundle_summary = {
            "fold_local_comparison": (
                "Each natural row is transformed separately by every outer-training scaler and "
                "compared only with that fold's held-out source p99."
            ),
            "result_file_bundles": int(len(bundle_pass)),
            "bundles_with_median_normalized_rms_z_at_most_one": int(sum(bundle_pass.values())),
            "fraction_of_bundles_with_median_normalized_rms_z_at_most_one": (
                crossfit_bundle_pass_fraction
            ),
            "exploratory_70_percent_bundle_threshold_pass": bool(
                crossfit_bundle_pass_fraction >= 0.70
            ),
            "by_bundle": {
                bundle: {
                    "median_normalized_rms_z": bundle_summary[bundle][
                        "median_crossfit_rms_z_over_source_test_p99"
                    ],
                    "at_most_one": bundle_pass[bundle],
                }
                for bundle in sorted(bundle_pass)
            },
            "guardrail": (
                "This is a descriptive OOD diagnostic, not conformal coverage, exchangeability, "
                "a calibrated probability, a p-value, or real-world accuracy."
            ),
        }
    return {
        "selected_model": selection,
        "feature_dimension": int(features.shape[1]),
        "training_rows": int(len(labels)),
        "training_class_counts": _class_counts(labels),
        "prediction_counts": dict(sorted(counts.items())),
        "raw_head_reproduction": raw_reproduction,
        "coverage": {
            "row_weighted_descriptive": {
                "rows": int(len(rows)),
                "within_training_reference_p99": int(
                    sum(row["within_training_reference_p99"] for row in rows)
                ),
                "rms_z_min": float(np.min(rms)),
                "rms_z_median": float(np.median(rms)),
                "rms_z_p95": float(np.quantile(rms, 0.95)),
                "rms_z_max": float(np.max(rms)),
            },
            "result_file_bundle_balanced_descriptive": {
                "bundles": int(len(bundle_summary)),
                "aggregation": (
                    "median RMS-z within each result bundle, then equal-weight summaries "
                    "over bundle medians"
                ),
                "rms_z_median": float(np.median(bundle_medians)),
                "rms_z_p95": float(np.quantile(bundle_medians, 0.95)),
                "rms_z_max": float(np.max(bundle_medians)),
                "by_bundle": bundle_summary,
            },
            "crossfit_source_test_support_descriptive": crossfit_bundle_summary,
        },
        "candidate_concordance_descriptive_only": {
            "rows_with_candidate": int(len(candidate_rows)),
            "prediction_matches": int(
                sum(row["prediction_matches_candidate"] for row in candidate_rows)
            ),
            "accuracy_denominator": None,
        },
        "distance_reference": {
            "kind": "full-fit source training RMS-z under the identical scaler used for natural rows",
            "n": int(len(reference)),
            "p99": float(np.quantile(reference, 0.99)),
            "calibration_guardrail": (
                "Empirical tail fractions are heuristic ranks against in-fit source rows, not "
                "conformal p-values, calibrated probabilities, or coverage guarantees."
            ),
            "crossfit_reference": (
                None
                if crossfit_support_reference is None
                else {
                    "outer_estimand": "new exact rate family",
                    "folds": int(len(crossfit_support_reference["folds"])),
                    "folds_sha256": crossfit_support_reference["folds_sha256"],
                    "result_file_bundle_threshold_fraction": crossfit_bundle_pass_fraction,
                }
            ),
        },
        "rows": rows,
    }


def _mean_repeat_metric(cv_result: dict, scope: str, metric: str) -> float:
    values = [repeat[scope][metric] for repeat in cv_result["per_repeat_metrics"]]
    return float(np.mean(values))


def adjudicate_pilot(variants: dict) -> dict:
    baseline = variants["raw_all"]
    structured = variants["orbit_composition_mean_variance"]
    baseline_genealogy = _mean_repeat_metric(
        baseline["genealogy_cv"], "appreciable", "accuracy"
    )
    structured_genealogy = _mean_repeat_metric(
        structured["genealogy_cv"], "appreciable", "accuracy"
    )
    baseline_rate = _mean_repeat_metric(
        baseline["rate_family_cv"],
        "appreciable_equal_rate_family",
        "balanced_accuracy",
    )
    structured_rate = _mean_repeat_metric(
        structured["rate_family_cv"],
        "appreciable_equal_rate_family",
        "balanced_accuracy",
    )
    baseline_rate_row_weighted = _mean_repeat_metric(
        baseline["rate_family_cv"], "appreciable", "accuracy"
    )
    structured_rate_row_weighted = _mean_repeat_metric(
        structured["rate_family_cv"], "appreciable", "accuracy"
    )
    genealogy_loss_points = 100.0 * (baseline_genealogy - structured_genealogy)
    rate_loss_points = 100.0 * (baseline_rate - structured_rate)
    baseline_coverage = baseline["natural_transfer"]["coverage"][
        "result_file_bundle_balanced_descriptive"
    ]
    structured_coverage = structured["natural_transfer"]["coverage"][
        "result_file_bundle_balanced_descriptive"
    ]
    baseline_bundles = baseline_coverage["by_bundle"]
    structured_bundles = structured_coverage["by_bundle"]
    if set(baseline_bundles) != set(structured_bundles):
        raise AssertionError("natural bundle sets differ between representations")
    paired_ratios = {
        bundle: (
            structured_bundles[bundle]["median_rms_z"]
            / baseline_bundles[bundle]["median_rms_z"]
        )
        for bundle in sorted(baseline_bundles)
    }
    ratio_values = np.asarray(list(paired_ratios.values()), dtype=float)
    improved_fraction = float(np.mean(ratio_values < 1.0))
    criteria = {
        "structured_genealogy_appreciable_accuracy_at_least_0_95": bool(
            structured_genealogy >= 0.95
        ),
        "structured_rate_family_macro_appreciable_balanced_accuracy_at_least_0_90": bool(
            structured_rate >= 0.90
        ),
        "genealogy_appreciable_accuracy_loss_vs_raw_at_most_one_point": bool(
            genealogy_loss_points <= 1.0 + 1e-12
        ),
        "rate_family_macro_appreciable_balanced_accuracy_loss_vs_raw_at_most_one_point": bool(
            rate_loss_points <= 1.0 + 1e-12
        ),
        "median_paired_bundle_rms_z_ratio_at_most_0_75": bool(
            np.median(ratio_values) <= 0.75
        ),
        "bundle_balanced_p95_rms_z_is_lower": bool(
            structured_coverage["rms_z_p95"] < baseline_coverage["rms_z_p95"]
        ),
        "at_least_70_percent_of_bundles_improve": bool(
            improved_fraction >= 0.70
        ),
    }
    return {
        "exploratory_relative_bridge_thresholds": criteria,
        "all_relative_bridge_thresholds_pass": bool(all(criteria.values())),
        "appreciable_accuracy": {
            "genealogy_cv": {
                "raw_all": baseline_genealogy,
                "structured": structured_genealogy,
                "structured_loss_percentage_points": genealogy_loss_points,
            },
            "rate_family_cv": {
                "raw_all": baseline_rate,
                "structured": structured_rate,
                "structured_loss_percentage_points": rate_loss_points,
                "primary_estimand": (
                    "equal-weight mean of within-exact-rate-family balanced accuracy"
                ),
                "row_weighted_secondary": {
                    "raw_all": baseline_rate_row_weighted,
                    "structured": structured_rate_row_weighted,
                },
            },
        },
        "natural_result_file_bundle_balanced_rms_z": {
            "raw_all_median": baseline_coverage["rms_z_median"],
            "structured_median": structured_coverage["rms_z_median"],
            "raw_all_p95": baseline_coverage["rms_z_p95"],
            "structured_p95": structured_coverage["rms_z_p95"],
            "paired_structured_over_raw_by_bundle": paired_ratios,
            "median_paired_ratio": float(np.median(ratio_values)),
            "fraction_of_bundles_improved": improved_fraction,
        },
        "decision_rule": (
            "Natural candidate labels are never a success criterion. If the structured view "
            "fails, proceed to observation-process simulation rather than tuning on natural labels."
        ),
    }


def runtime_audit(priority_audit: dict | None = None) -> dict:
    thread_environment = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    }
    if any(value != "1" for value in thread_environment.values()):
        raise RuntimeError(f"single-thread environment verification failed: {thread_environment}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA disablement verification failed")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "thread_environment": thread_environment,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "stopped_trading_compute_authorization": os.environ.get(STOPPED_TRADING_AUTH_ENV),
        "closing_azure_owner_session_authorization": os.environ.get(
            AZURE_CLOSING_OWNER_AUTH_ENV
        ),
        "compute_target": os.environ.get(COMPUTE_TARGET_ENV),
        "process_priority": priority_audit,
        "argv": sys.argv,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-root",
        type=Path,
        default=Path(os.environ.get("DNNAIC_DATA", "data/simulation_data")) / "regen_full",
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--C-grid", default="0.001,0.01,0.1,1,10,100,1000")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument(
        "--natural-result",
        type=Path,
        action="append",
        default=None,
        help=(
            "explicit standardized natural results.json path; repeat to override the "
            "pinned 14-bundle cohort (filesystem globs are intentionally unsupported)"
        ),
    )
    parser.add_argument("--compute-state", type=Path, default=DEFAULT_COMPUTE_STATE)
    parser.add_argument(
        "--compute-target",
        choices=("local", "azure"),
        default="local",
        help="pressure telemetry target for an explicitly authorized stopped-trading run",
    )
    parser.add_argument(
        "--allow-stopped-trading-compute",
        action="store_true",
        help=(
            "honor an explicit owner authorization only when the sole distress reason is "
            "stopped trading and all pinned local/Azure pressure checks remain safe"
        ),
    )
    parser.add_argument(
        "--allow-closing-owner-session",
        action="store_true",
        help=(
            "on Azure only, accept owner_rdp_active=true solely when a fresh loginctl check "
            "shows every owner session is in closing state"
        ),
    )
    args = parser.parse_args()
    os.environ[COMPUTE_TARGET_ENV] = args.compute_target
    if args.allow_stopped_trading_compute:
        os.environ[STOPPED_TRADING_AUTH_ENV] = "1"
    if args.allow_closing_owner_session:
        os.environ[AZURE_CLOSING_OWNER_AUTH_ENV] = "1"
    if args.max_depth < 3:
        parser.error("--max-depth must be at least 3")
    if args.outer_splits < 3 or args.inner_splits < 3:
        parser.error("outer/inner splits must be at least 3")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value != "")
    try:
        C_grid = validate_C_grid(
            float(value) for value in args.C_grid.split(",") if value != ""
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not seeds:
        parser.error("at least one repeat seed is required")

    # This is intentionally a separate pre-work gate.
    gate = compute_gate(args.compute_state)
    priority = set_below_normal_priority()
    revision = git_revision()
    require_clean_tracked_revision(revision)
    test_file_sha256 = sha256_file(
        REPO / "tests" / "test_structured_transfer_pilot.py"
    )
    # Capture and validate pristine source provenance before creating any output
    # directory or lock file inside the worktree.
    run_lock = SingleWriterLease(
        args.result_dir,
        ".structured_transfer.lock",
    ).acquire()
    canonical = load_canonical(args.canonical_root.resolve(), args.max_depth)
    positive = canonical["labels"] != "D"
    table = canonical["table"][positive]
    labels_text = canonical["labels"][positive]
    if set(map(str, labels_text)) != set(CLASSES):
        raise AssertionError(
            f"positive canonical label vocabulary changed: {sorted(set(map(str, labels_text)))}"
        )
    labels = np.searchsorted(CLASSES, labels_text)
    rates = canonical["rates"][positive]
    designs = canonical["designs"][positive]
    group_ids = canonical["group_ids"][positive]
    rate_blocks = rate_family_ids(designs, rates)

    args.result_dir.mkdir(parents=True, exist_ok=True)
    simulation_checkpoint_path = args.result_dir / "simulation_checkpoint.json"
    simulation_contract = {
        "schema_version": SIMULATION_CHECKPOINT_SCHEMA,
        "script_sha256": revision["script_sha256"],
        "test_file_sha256": test_file_sha256,
        "source_commit": revision["commit"],
        "canonical_array_contracts": canonical["audit"]["array_contracts"],
        "max_depth": int(args.max_depth),
        "seeds": list(seeds),
        "C_grid": list(C_grid),
        "outer_splits": int(args.outer_splits),
        "inner_splits": int(args.inner_splits),
        "representation_order": list(REPRESENTATIONS),
        "representation_specifications": REPRESENTATIONS,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    simulation_contract_sha256 = hashlib.sha256(
        _canonical_json(simulation_contract)
    ).hexdigest()
    variants = {}
    if simulation_checkpoint_path.exists():
        variants = load_simulation_checkpoint(
            simulation_checkpoint_path,
            contract_sha256=simulation_contract_sha256,
            representation_order=list(REPRESENTATIONS),
        )
        print(f"[checkpoint] resumed {len(variants)} representation(s)", flush=True)
    feature_bank = {}
    for name in REPRESENTATIONS:
        compute_gate(args.compute_state)
        features = representation_features(table, name)
        feature_bank[name] = features
        columns = representation_columns(name)
        if features.shape[1] != len(columns):
            raise AssertionError(f"{name}: feature-column dimension mismatch")
        if name in variants:
            validate_nested_oof_payload(
                variants[name]["genealogy_cv"],
                labels,
                rates,
                designs,
                group_ids,
                group_ids,
                features=features,
            )
            validate_nested_oof_payload(
                variants[name]["rate_family_cv"],
                labels,
                rates,
                designs,
                group_ids,
                rate_blocks,
                evaluation_families=rate_blocks,
                features=features,
            )
            print(f"[representation] {name} (checkpoint)", flush=True)
            continue
        print(f"[representation] {name}", flush=True)
        genealogy_cv = nested_oof(
            features,
            labels,
            rates,
            designs,
            group_ids,
            group_ids,
            outer_name="new genealogy within the canonical design",
            seeds=seeds,
            C_grid=C_grid,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            compute_state=args.compute_state,
        )
        rate_cv = nested_oof(
            features,
            labels,
            rates,
            designs,
            group_ids,
            rate_blocks,
            outer_name="new exact rate family; all A/B/C replicates sharing a rate are blocked",
            seeds=seeds,
            C_grid=C_grid,
            outer_splits=args.outer_splits,
            inner_splits=args.inner_splits,
            evaluation_families=rate_blocks,
            compute_state=args.compute_state,
        )
        transfer = design_transfer(
            features,
            labels,
            rates,
            designs,
            group_ids,
            C_grid=C_grid,
            inner_splits=args.inner_splits,
            seed=70_711,
            compute_state=args.compute_state,
        )
        variants[name] = {
            "specification": REPRESENTATIONS[name],
            "feature_columns": columns,
            "feature_dimension": int(features.shape[1]),
            "genealogy_cv": genealogy_cv,
            "rate_family_cv": rate_cv,
            "fixed_continuous_transfer": transfer,
        }
        write_json_atomic(
            simulation_checkpoint_path,
            {
                "schema_version": SIMULATION_CHECKPOINT_SCHEMA,
                "contract": simulation_contract,
                "contract_sha256": simulation_contract_sha256,
                "completed_representations": list(variants),
                "representation_payload_sha256": {
                    key: hashlib.sha256(_canonical_json(value)).hexdigest()
                    for key, value in variants.items()
                },
                "representations": variants,
            },
        )

    # Natural panels are deliberately unavailable during every simulation-only
    # representation/model-selection operation above.
    compute_gate(args.compute_state)
    natural_paths = (
        [path.resolve() for path in args.natural_result]
        if args.natural_result is not None
        else pinned_natural_paths()
    )
    panels, natural_audit = load_natural_panels(natural_paths, args.max_depth)
    if args.natural_result is None and (
        natural_audit["result_bundles"] != 14 or natural_audit["panel_rows"] != 102
    ):
        raise AssertionError(
            "pinned natural cohort changed: expected 14 bundles and 102 rows, got "
            f"{natural_audit['result_bundles']} bundles and {natural_audit['panel_rows']} rows"
        )
    for name in REPRESENTATIONS:
        compute_gate(args.compute_state)
        variants[name]["natural_transfer"] = final_natural_score(
            feature_bank[name],
            labels,
            rate_blocks,
            panels,
            representation=name,
            C_grid=C_grid,
            inner_splits=args.inner_splits,
            seed=80_711,
            crossfit_support_reference=variants[name]["rate_family_cv"][
                "crossfit_support_reference"
            ],
            compute_state=args.compute_state,
        )

    final_revision = git_revision()
    require_revision_unchanged(revision, final_revision)
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "exploratory_bridge_not_paper_result",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": revision,
        "final_source_recheck": final_revision,
        "compute_gate": gate,
        "runtime": runtime_audit(priority),
        "single_writer_lock": {
            "path": str(run_lock.path.resolve()),
            "mechanism": "OS advisory lock; automatically released on process exit",
        },
        "guardrail": (
            "Representations and all simulation CV completed before the natural cohort was loaded. "
            "The natural cohort informed representation design and is not a validation set. "
            "Natural rows are unlabeled correlated OOD diagnostics, not an accuracy denominator."
        ),
        "natural_cohort_informed_representation_design": True,
        "natural_cohort_is_validation_set": False,
        "real_world_accuracy_established": False,
        "canonical_source": canonical["audit"],
        "analysis_population": {
            "positive_replicates": int(len(labels)),
            "class_counts": _class_counts(labels),
            "design_counts": dict(sorted(Counter(map(str, designs)).items())),
            "unique_genealogies": int(len(np.unique(group_ids))),
            "unique_rate_families": int(len(np.unique(rate_blocks))),
            "exact_rate_family_contract": "design plus float.hex(rate)",
            "depth_grid": list(range(2, args.max_depth + 1)),
        },
        "natural_source_audit": natural_audit,
        "natural_cohort_policy": (
            "pinned 14-bundle/102-row default"
            if args.natural_result is None
            else "explicit command-line path list; no filesystem globbing"
        ),
        "simulation_checkpoint": {
            "path": str(simulation_checkpoint_path.resolve()),
            "bytes": simulation_checkpoint_path.stat().st_size,
            "sha256": sha256_file(simulation_checkpoint_path),
            "contract_sha256": simulation_contract_sha256,
            "completed_representations": list(variants),
        },
        "representations": variants,
        "pilot_adjudication": adjudicate_pilot(variants),
        "required_next_step": (
            "Regardless of relative bridge thresholds, evaluate a sealed observation-process "
            "simulation bank with held-out demography x ascertainment families; do not tune on "
            "natural candidate labels or interpret them as external accuracy."
        ),
    }
    output = args.result_dir / "results.json"
    write_json_atomic(output, result, indent=2)
    print(json.dumps({
        "output": str(output.resolve()),
        "pilot_adjudication": result["pilot_adjudication"],
        "natural_rows": natural_audit["panel_rows"],
    }, indent=2, allow_nan=False))
    run_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
