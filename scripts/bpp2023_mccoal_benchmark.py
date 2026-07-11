#!/usr/bin/env python3
"""Run a derivative Ji et al. (2023) BPP/MCcoal known-truth bank.

The official Ji et al. simulations contain exact outflow (Q -> D) and inflow
(D -> Q) truth, but only four haploid sequences per population.  That supports
PADZE depths g=2..4 and therefore cannot be scored by DNNaic's frozen g=2..16
head.  This runner preserves the published asymmetric topology and
parameterization while increasing Q/R/D sampling to 200 haploid copies and
reconstructs the paper's inflow-asymmetric phi=0 false-positive protocol as an
explicit no-introgression boundary control.  It is a larger-sample derivative
benchmark, not a byte or accuracy reproduction of the paper.

The population order is frozen before simulation: P1=R, P2=Q, P3=D.  Thus
published outflow Q->D is class B and inflow D->Q is class C.  D uses the
inflow topology with donor phi=0 and resident phi=1.  The default bank crosses
ten episodic introgression probabilities with three common theta/tau scales,
producing 30 matched B/C/D families and 90 independent jobs.

Each job has 500 independent 500-bp alignments.  Sites within an alignment
remain linked and receive a block ID.  The raw BPP alignment is parsed
streamingly, fixed A/C/G/T allele counts and hashes are checkpointed, and the
large temporary alignment is deleted only after its count checkpoint is
durable.  One B/C/D raw triad is gzip-retained for parser auditing.  PADZE uses
the full g=2..199 curve; the frozen direction head uses g=2..16.  Direction
accuracy contains B/C only.  D is evaluated only by the secondary frozen
appreciable-migration gate and never forced into an A/B/C accuracy denominator.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import gzip
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Iterable, Sequence

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import stdpopsim_neanderthal_benchmark as stdbench
from scripts import structured_transfer_pilot as structured


SCHEMA_VERSION = "dnnaic-bpp2023-mccoal-benchmark-v1"
CHECKPOINT_SCHEMA = "dnnaic-bpp2023-mccoal-checkpoint-v1"
COUNT_SCHEMA = "dnnaic-bpp2023-counts-v1"
BPP_RELEASES = {
    "4.6.1": {
        "binary_sha256": "567c18544cc8cb015ed5b206ae9976627505825b0c6aa17b6f49b80f601404b4",
        "version_token": "bpp v4.6.1",
        "status": "Ji-era pinned release used for the primary derivative bank",
    },
    "4.8.7": {
        "binary_sha256": "6c8828704e1037788e02d6943cc6cbb61d05d6aadbdd976095b71fc965e8e90e",
        "distribution_sha256": "577306b8dafa80114d09e61f460633dd567eff9c67d5f878bbc7ae9d74cf69f2",
        "version_token": "bpp v4.8.7",
        "status": "current official release sensitivity as verified 2026-07-11",
    },
}
OFFICIAL_ARCHIVE_MD5 = "c233514a93ad48fc67da3be14fa93264"
OFFICIAL_ARCHIVE_SHA256 = "492c0d8a316d7349a77b5482c46693b4e8a5a05acce4b617651b3ad1f4ef3b02"
OFFICIAL_CONTROL_SHA256 = {
    "MCcoal.outflow-asym.ctl": "9577804bee2467cb4ba3070a454b570c6500353ef2346822298b97fb8383b4de",
    "MCcoal.inflow-asym.ctl": "c71d4a2a061eda75db6fc906cef247648059d3d1ccdba9d172b44d497c73744c",
}
CONDITIONS = ("B", "C", "D")
POSITIVE_CLASSES = ("B", "C")
POPULATION_ORDER = ("Q", "R", "D", "S")
DNNAIC_MAPPING = {"P1": "R", "P2": "Q", "P3": "D"}
GENE_COPIES = 200
OUTGROUP_COPIES = 1
LOCUS_COUNT = 500
LOCUS_LENGTH = 500
# Put the published asymmetric setting first so the retained raw B/C/D triad is
# also the closest derivative of the official Ji et al. controls.  Ordering is
# otherwise scientifically immaterial, but it is part of the frozen manifest.
PHIS = (0.106, 0.005, 0.010, 0.020, 0.040, 0.060, 0.080, 0.150, 0.225, 0.300)
SCALES = (1.0, 0.5, 2.0)
REPRESENTATIONS = (
    "raw_all",
    "raw_mean_variance",
    "orbit_composition_mean_variance",
)
BASE_PARAMETERS = {
    "tau_i": 0.000307,
    "tau_QR": 0.000389,
    "tau_QRD": 0.000731,
    "tau_root": 0.003423,
    "theta_Q": 0.000664,
    "theta_R": 0.000344,
    "theta_D": 0.003314,
    "theta_l": 0.001568,
    "theta_m": 0.000407,
    "theta_QR": 0.002429,
    "theta_QRD": 0.000930,
    "theta_S": 0.000866,
    "theta_root": 0.011010,
}
DEFAULT_CACHE = (
    Path.home()
    / "Documents"
    / "Codex"
    / "2026-07-10"
    / "dnnaic-datasets2-data"
    / "bpp2023_mccoal_2026_07_11"
)
DEFAULT_RESULTS = REPO / "results" / "bpp2023_mccoal_benchmark_2026_07_11"


@dataclass(frozen=True)
class BPPJob:
    family_index: int
    family_id: str
    family_positive_phi: float
    scale: float
    label: str
    job_id: str
    seed: int


def job_effective_phi(job: BPPJob) -> float:
    return 0.0 if job.label == "D" else float(job.family_positive_phi)


def job_payload(job: BPPJob) -> dict:
    return {**asdict(job), "effective_phi": job_effective_phi(job)}


def _canonical_json(value) -> bytes:
    return json.dumps(
        stdbench._jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _format_number(value: float) -> str:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("BPP control values must be finite and nonnegative")
    return f"{value:.12g}"


def deterministic_seed(family_id: str, label: str) -> int:
    if label not in CONDITIONS:
        raise ValueError(f"unknown BPP label {label!r}")
    digest = hashlib.sha256(
        f"dnnaic-bpp-ji-v1|{family_id}|{label}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:4], "big") % 2_147_483_646 + 1


def make_jobs() -> list[BPPJob]:
    jobs = []
    family_index = 0
    for scale_index, scale in enumerate(SCALES):
        for phi_index, phi in enumerate(PHIS):
            family_id = f"ji-scale-{scale_index}-{scale:g}__phi-{phi_index}-{phi:g}"
            for label in CONDITIONS:
                jobs.append(BPPJob(
                    family_index=family_index,
                    family_id=family_id,
                    family_positive_phi=float(phi),
                    scale=float(scale),
                    label=label,
                    job_id=f"{family_id}__{label}",
                    seed=deterministic_seed(family_id, label),
                ))
            family_index += 1
    if family_index != len(PHIS) * len(SCALES) or len(jobs) != 90:
        raise AssertionError("BPP job-bank cardinality changed")
    if len({job.job_id for job in jobs}) != len(jobs):
        raise AssertionError("BPP job IDs collide")
    if len({job.seed for job in jobs}) != len(jobs):
        raise AssertionError("BPP deterministic seeds collide")
    return jobs


def network_newick(label: str, phi: float, scale: float) -> str:
    if label not in CONDITIONS:
        raise ValueError(f"unknown BPP label {label!r}")
    if not 0 <= phi <= 1 or scale <= 0 or not math.isfinite(scale):
        raise ValueError("BPP phi/scale is invalid")
    effective_phi = 0.0 if label == "D" else float(phi)
    resident_phi = 1.0 - effective_phi
    p = {name: _format_number(value * scale) for name, value in BASE_PARAMETERS.items()}
    donor = _format_number(effective_phi)
    resident = _format_number(resident_phi)
    if label == "B":
        return (
            f"((((m[&phi={donor},tau-parent=no]:{p['tau_i']},Q #{p['theta_Q']})"
            f"l:{p['tau_i']} #{p['theta_l']},R #{p['theta_R']})"
            f"b:{p['tau_QR']} #{p['theta_QR']},"
            f"(D #{p['theta_D']})m[&phi={resident},tau-parent=yes]:{p['tau_i']} #{p['theta_m']})"
            f"f:{p['tau_QRD']} #{p['theta_QRD']},S #{p['theta_S']})"
            f"h:{p['tau_root']} #{p['theta_root']};"
        )
    return (
        f"((((Q #{p['theta_Q']})l[&phi={resident},tau-parent=yes]:{p['tau_i']} #{p['theta_l']},"
        f"R #{p['theta_R']})b:{p['tau_QR']} #{p['theta_QR']},"
        f"(l[&phi={donor},tau-parent=no]:{p['tau_i']},D #{p['theta_D']})"
        f"m:{p['tau_i']} #{p['theta_m']})f:{p['tau_QRD']} #{p['theta_QRD']},"
        f"S #{p['theta_S']})h:{p['tau_root']} #{p['theta_root']};"
    )


def control_text(job: BPPJob) -> str:
    return (
        f"seed = {job.seed}\n"
        "seqfile = Seq.txt\n"
        "Imapfile = Imap.txt\n"
        "species&tree = 4 Q R D S\n"
        f"  {GENE_COPIES} {GENE_COPIES} {GENE_COPIES} {OUTGROUP_COPIES}\n"
        f"  {network_newick(job.label, job.family_positive_phi, job.scale)}\n"
        "phase = 0 0 0 0\n"
        f"loci&length = {LOCUS_COUNT} {LOCUS_LENGTH}\n"
        "model = 0\n"
    )


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_audit(
    binary: Path,
    official_controls: Path,
    official_archive: Path,
    bpp_version: str,
) -> dict:
    binary = binary.resolve()
    official_controls = official_controls.resolve()
    official_archive = official_archive.resolve()
    if not binary.is_file() or not official_controls.is_dir() or not official_archive.is_file():
        raise FileNotFoundError(
            "BPP binary, official-control directory, or official archive is unavailable"
        )
    if bpp_version not in BPP_RELEASES:
        raise ValueError(f"unsupported pinned BPP release {bpp_version!r}")
    release = BPP_RELEASES[bpp_version]
    binary_hash = structured.sha256_file(binary)
    if binary_hash != release["binary_sha256"]:
        raise RuntimeError(f"BPP binary SHA-256 changed: {binary_hash}")
    archive_md5 = _md5_file(official_archive)
    archive_sha256 = structured.sha256_file(official_archive)
    if archive_md5 != OFFICIAL_ARCHIVE_MD5 or archive_sha256 != OFFICIAL_ARCHIVE_SHA256:
        raise RuntimeError(
            "official Ji et al. archive hash changed: "
            f"MD5={archive_md5}, SHA-256={archive_sha256}"
        )
    control_names = tuple(OFFICIAL_CONTROL_SHA256)
    controls = {}
    for name in control_names:
        path = official_controls / name
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        control_sha256 = hashlib.sha256(payload).hexdigest()
        if control_sha256 != OFFICIAL_CONTROL_SHA256[name]:
            raise RuntimeError(
                f"extracted official BPP control hash changed for {name}: {control_sha256}"
            )
        controls[name] = {
            "bytes": len(payload),
            "sha256": control_sha256,
        }
    outflow = (official_controls / control_names[0]).read_text(encoding="utf-8-sig")
    inflow = (official_controls / control_names[1]).read_text(encoding="utf-8-sig")
    for required in ("phi=0.106000", "4 4 4 4", "loci&length = 500"):
        if required not in outflow or required not in inflow:
            raise RuntimeError(f"official BPP control contract changed: missing {required!r}")
    if "m[&phi=0.106000,tau-parent=no]:0.000307,Q" not in outflow.replace(" ", ""):
        raise RuntimeError("official outflow direction signature changed")
    if "l[&phi=0.106000,tau-parent=no]:0.000307,D" not in inflow.replace(" ", ""):
        raise RuntimeError("official inflow direction signature changed")
    with tempfile.TemporaryDirectory(
        prefix=".bpp-version-probe-", dir=official_archive.parent
    ) as probe_directory:
        version = subprocess.run(
            [str(binary), "--version"],
            cwd=probe_directory,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        ).stdout
        probe_side_effects = []
        for path in sorted(Path(probe_directory).iterdir(), key=lambda value: value.name):
            if not path.is_file():
                raise RuntimeError("BPP version probe created an unexpected directory")
            probe_side_effects.append({
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": structured.sha256_file(path),
            })
    if release["version_token"] not in version:
        raise RuntimeError("BPP binary version output changed")
    return {
        "bpp_version": bpp_version,
        "release_contract": release,
        "binary_bytes": binary.stat().st_size,
        "binary_sha256": binary_hash,
        "version_stdout_sha256": hashlib.sha256(version.encode("utf-8")).hexdigest(),
        "version_probe_side_effects_isolated_then_removed": probe_side_effects,
        "official_archive_bytes": official_archive.stat().st_size,
        "official_archive_md5": archive_md5,
        "official_archive_sha256": archive_sha256,
        "official_controls": controls,
        "paper_mapping": {
            "outflow": "Q->D",
            "inflow": "D->Q",
            "P1": "R",
            "P2": "Q",
            "P3": "D",
            "outflow_dnnaic": "B",
            "inflow_dnnaic": "C",
        },
    }


class _HashingReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.handle = self.path.open("rb")
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def readline(self) -> bytes:
        line = self.handle.readline()
        self.digest.update(line)
        self.bytes_read += len(line)
        return line

    def close(self):
        self.handle.close()


def _next_nonempty(reader: _HashingReader) -> bytes:
    while True:
        line = reader.readline()
        if line == b"":
            return b""
        if line.strip():
            return line


def parse_bpp_alignments(
    path: Path,
    *,
    locus_count: int = LOCUS_COUNT,
    locus_length: int = LOCUS_LENGTH,
    gene_copies: int = GENE_COPIES,
    outgroup_copies: int = OUTGROUP_COPIES,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    path = Path(path)
    if locus_count < 1 or locus_length < 1 or gene_copies < 1 or outgroup_copies < 1:
        raise ValueError("BPP parser dimensions must be positive")
    if gene_copies > np.iinfo(np.uint16).max:
        raise ValueError("BPP parser gene-copy count exceeds uint16 capacity")
    reader = _HashingReader(path)
    all_counts = []
    all_blocks = []
    all_positions = []
    invariant = 0
    biallelic = 0
    multiallelic = 0
    expected_sequences = 3 * gene_copies + outgroup_copies
    base_codes = np.frombuffer(b"ACGT", dtype=np.uint8)
    try:
        for locus_index in range(locus_count):
            header = _next_nonempty(reader)
            if not header:
                raise RuntimeError(f"Seq.txt ended before locus {locus_index}")
            fields = header.decode("ascii").split()
            if fields != [str(expected_sequences), str(locus_length)]:
                raise RuntimeError(
                    f"locus {locus_index} header is {fields}, expected "
                    f"[{expected_sequences}, {locus_length}]"
                )
            sequences = {name: [] for name in POPULATION_ORDER}
            for sequence_index in range(expected_sequences):
                line = _next_nonempty(reader)
                if not line:
                    raise RuntimeError(
                        f"Seq.txt ended within locus {locus_index}, sequence {sequence_index}"
                    )
                tokens = line.decode("ascii").split()
                if not tokens:
                    raise RuntimeError("empty BPP sequence row")
                sample_name = tokens[0]
                sequence = "".join(tokens[1:])
                while len(sequence) < locus_length:
                    continuation = reader.readline()
                    if not continuation:
                        raise RuntimeError("BPP sequence continuation ended early")
                    sequence += "".join(continuation.decode("ascii").split())
                if len(sequence) != locus_length:
                    raise RuntimeError("BPP sequence length changed")
                if "^" not in sample_name:
                    raise RuntimeError(f"BPP sample lacks an Imap suffix: {sample_name!r}")
                population = sample_name.rsplit("^", 1)[1]
                if population not in sequences:
                    raise RuntimeError(f"BPP sample has unknown population suffix {population!r}")
                encoded = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
                if not np.isin(encoded, base_codes).all():
                    raise RuntimeError("BPP JC69 alignment contains a non-ACGT base")
                sequences[population].append(encoded.copy())
            observed_counts = {name: len(values) for name, values in sequences.items()}
            expected_counts = {
                "Q": gene_copies,
                "R": gene_copies,
                "D": gene_copies,
                "S": outgroup_copies,
            }
            if observed_counts != expected_counts:
                raise RuntimeError(
                    f"BPP locus {locus_index} population counts changed: {observed_counts}"
                )
            counts = np.zeros((locus_length, 3, 4), dtype=np.uint16)
            for population_index, population in enumerate(("R", "Q", "D")):
                matrix = np.stack(sequences[population])
                for allele_index, code in enumerate(base_codes):
                    counts[:, population_index, allele_index] = np.sum(
                        matrix == code, axis=0, dtype=np.uint16
                    )
            if not np.all(counts.sum(axis=2) == gene_copies):
                raise AssertionError("BPP allele counts do not conserve sample sizes")
            present = counts.sum(axis=1) > 0
            allele_number = present.sum(axis=1)
            keep = allele_number >= 2
            invariant += int((~keep).sum())
            biallelic += int((allele_number == 2).sum())
            multiallelic += int((allele_number > 2).sum())
            if np.any(keep):
                all_counts.append(counts[keep])
                all_blocks.append(np.full(int(keep.sum()), locus_index, dtype=np.int32))
                all_positions.append(np.flatnonzero(keep).astype(np.int16) + 1)
        remainder = reader.handle.read()
        reader.digest.update(remainder)
        reader.bytes_read += len(remainder)
        if remainder.strip():
            raise RuntimeError("Seq.txt has non-whitespace content after the final locus")
    finally:
        reader.close()
    if reader.bytes_read != path.stat().st_size:
        raise AssertionError("BPP parser did not hash every raw byte")
    counts = np.concatenate(all_counts, axis=0) if all_counts else np.zeros((0, 3, 4), dtype=np.uint16)
    blocks = np.concatenate(all_blocks) if all_blocks else np.zeros(0, dtype=np.int32)
    positions = np.concatenate(all_positions) if all_positions else np.zeros(0, dtype=np.int16)
    if len(counts) < 2:
        raise RuntimeError("BPP job produced too few P1/P2/P3-polymorphic sites")
    ledger = hashlib.sha256()
    ledger.update(counts.astype("<u2", copy=False).tobytes())
    ledger.update(blocks.astype("<i4", copy=False).tobytes())
    ledger.update(positions.astype("<i2", copy=False).tobytes())
    audit = {
        "raw_seq_bytes": path.stat().st_size,
        "raw_seq_sha256": reader.digest.hexdigest(),
        "loci": locus_count,
        "sites_per_locus": locus_length,
        "total_source_sites": locus_count * locus_length,
        "gene_copies_per_ingroup_population": gene_copies,
        "outgroup_copies": outgroup_copies,
        "invariant_in_dnnaic_triplet": invariant,
        "biallelic_in_dnnaic_triplet": biallelic,
        "multiallelic_in_dnnaic_triplet": multiallelic,
        "retained_polymorphic_sites": int(len(counts)),
        "independent_locus_blocks": int(len(np.unique(blocks))),
        "count_block_position_ledger_sha256": ledger.hexdigest(),
    }
    if invariant + biallelic + multiallelic != locus_count * locus_length:
        raise AssertionError("BPP site-category ledger is not exhaustive")
    return counts, blocks, positions, audit


def parse_imap(path: Path) -> dict:
    payload = path.read_bytes()
    rows = [line.split() for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
    if any(len(row) != 2 for row in rows):
        raise RuntimeError("BPP Imap.txt row shape changed")
    expected_rows = [[population, population] for population in POPULATION_ORDER]
    if rows != expected_rows:
        raise RuntimeError(
            f"BPP Imap.txt contract changed: observed {rows}, expected {expected_rows}"
        )
    mapping = {row[0]: row[1] for row in rows}
    return {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": rows,
    }


def counts_to_curve(
    counts: np.ndarray,
    blocks: np.ndarray,
    positions: np.ndarray,
    *,
    compute_state: Path | None = None,
) -> tuple[np.ndarray, dict]:
    from padze import LociData, Metadata, compute_features

    counts = np.asarray(counts, dtype=np.uint16)
    blocks = np.asarray(blocks, dtype=np.int32)
    positions = np.asarray(positions, dtype=np.int16)
    if counts.ndim != 3 or counts.shape[1:] != (3, 4):
        raise ValueError("BPP counts must have shape (site, 3, 4)")
    if not (len(counts) == len(blocks) == len(positions)):
        raise ValueError("BPP count/block/position lengths differ")
    if not np.all(counts.sum(axis=2) == GENE_COPIES):
        raise ValueError("BPP count matrices do not sum to 200")
    count_matrices = [
        matrix[:, matrix.sum(axis=0) > 0].astype(np.int64)
        for matrix in counts
    ]
    sample_sizes = np.full((len(counts), 3), GENE_COPIES, dtype=np.int64)
    locus_ids = [f"bpp-locus-{block:04d}-site-{position:03d}" for block, position in zip(blocks, positions)]
    loci = LociData(
        populations=["P1", "P2", "P3"],
        count_matrices=count_matrices,
        sample_sizes=sample_sizes,
        locus_ids=locus_ids,
        metadata=Metadata(
            source="BPP 4.6.1 MCcoal derivative of Ji et al. 2023",
            populations=["P1", "P2", "P3"],
            sample_ids={name: [] for name in ("P1", "P2", "P3")},
            ploidy={name: 1 for name in ("P1", "P2", "P3")},
            n_loci_read=LOCUS_COUNT * LOCUS_LENGTH,
            n_loci_kept=len(counts),
            filters_applied=["polymorphic across R/Q/D; S-only variation excluded"],
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
    index = {column: position for position, column in enumerate(columns)}
    try:
        curve = matrix[:, [index[column] for column in stdbench.CURVE_COLUMNS]].astype(np.float64)
    except KeyError as exc:
        raise RuntimeError(f"PADZE feature contract changed: {exc}") from exc
    if curve.shape != (198, 28) or not np.isfinite(curve).all():
        raise AssertionError("BPP PADZE curve is invalid")
    if not np.array_equal(curve[:, 0], stdbench.FULL_DEPTHS):
        raise AssertionError("BPP PADZE depth grid changed")
    return curve, {
        "curve_sha256_float64": stdbench._sha256_array(curve.astype("<f8", copy=False)),
        "linked_site_blocks": int(len(np.unique(blocks))),
        "raw_PADZE_SE_is_not_block_robust": True,
    }


def _count_file_path(cache_dir: Path, job: BPPJob) -> Path:
    return cache_dir / "counts" / f"{job.job_id}.npz"


def save_count_file(
    path: Path,
    job: BPPJob,
    config_sha256: str,
    counts: np.ndarray,
    blocks: np.ndarray,
    positions: np.ndarray,
    metadata: dict,
) -> None:
    stdbench._atomic_npz(
        path,
        schema=np.asarray([COUNT_SCHEMA]),
        config_sha256=np.asarray([config_sha256]),
        job_json=np.asarray([_canonical_json(job_payload(job)).decode("ascii")]),
        metadata_json=np.asarray([_canonical_json(metadata).decode("ascii")]),
        counts=np.asarray(counts, dtype=np.uint16),
        blocks=np.asarray(blocks, dtype=np.int32),
        positions=np.asarray(positions, dtype=np.int16),
    )


def load_count_file(
    path: Path,
    job: BPPJob,
    config_sha256: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema", "config_sha256", "job_json", "metadata_json", "counts", "blocks", "positions"}
        if set(archive.files) != required:
            raise RuntimeError("BPP count checkpoint member set changed")
        if archive["schema"].tolist() != [COUNT_SCHEMA]:
            raise RuntimeError("BPP count checkpoint schema changed")
        if archive["config_sha256"].tolist() != [config_sha256]:
            raise RuntimeError("BPP count checkpoint configuration changed")
        if json.loads(str(archive["job_json"][0])) != job_payload(job):
            raise RuntimeError("BPP count checkpoint job changed")
        metadata = json.loads(str(archive["metadata_json"][0]))
        counts = np.asarray(archive["counts"], dtype=np.uint16)
        blocks = np.asarray(archive["blocks"], dtype=np.int32)
        positions = np.asarray(archive["positions"], dtype=np.int16)
    if counts.shape != (len(blocks), 3, 4) or len(counts) != len(positions):
        raise RuntimeError("BPP count checkpoint arrays changed shape")
    ledger = hashlib.sha256()
    ledger.update(counts.astype("<u2", copy=False).tobytes())
    ledger.update(blocks.astype("<i4", copy=False).tobytes())
    ledger.update(positions.astype("<i2", copy=False).tobytes())
    if ledger.hexdigest() != metadata["parser_audit"]["count_block_position_ledger_sha256"]:
        raise RuntimeError("BPP count checkpoint ledger hash changed")
    return counts, blocks, positions, metadata


def _gzip_atomic(source: Path, destination: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part.{os.getpid()}.{time.time_ns()}")
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as raw_output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw_output, mtime=0) as output_handle:
                shutil.copyfileobj(input_handle, output_handle, length=1024 * 1024)
            raw_output.flush()
            os.fsync(raw_output.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "bytes": destination.stat().st_size,
        "sha256": structured.sha256_file(destination),
    }


def _safe_remove_job_directory(path: Path, work_root: Path) -> None:
    resolved = path.resolve()
    root = work_root.resolve()
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("refusing to remove a BPP job directory outside the work root") from exc
    if len(relative.parts) != 1 or not relative.name.startswith("job-"):
        raise RuntimeError("refusing to remove an unexpected BPP work path")
    shutil.rmtree(resolved)


def simulate_job(
    job: BPPJob,
    *,
    binary: Path,
    cache_dir: Path,
    config_sha256: str,
    compute_state: Path | None = None,
    timeout_seconds: int = 1800,
) -> dict:
    count_path = _count_file_path(cache_dir, job)
    reused_count_file = count_path.exists()
    if reused_count_file:
        counts, blocks, positions, metadata = load_count_file(count_path, job, config_sha256)
    else:
        if compute_state is not None:
            structured.compute_gate(compute_state)
        work_root = cache_dir / "work"
        work_root.mkdir(parents=True, exist_ok=True)
        job_dir = Path(tempfile.mkdtemp(prefix=f"job-{job.job_id}-", dir=work_root))
        durable_count_checkpoint = False
        try:
            control = control_text(job)
            control_path = job_dir / "MCcoal.ctl"
            control_path.write_text(control, encoding="ascii", newline="\n")
            started = time.perf_counter()
            completed = subprocess.run(
                [str(binary.resolve()), "--simulate", "MCcoal.ctl"],
                cwd=job_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
            elapsed = time.perf_counter() - started
            if completed.returncode != 0:
                raise RuntimeError(
                    f"BPP failed for {job.job_id} with code {completed.returncode}: "
                    f"{completed.stderr[-2000:]}"
                )
            seq_path = job_dir / "Seq.txt"
            imap_path = job_dir / "Imap.txt"
            if not seq_path.is_file() or not imap_path.is_file():
                raise RuntimeError("BPP did not produce Seq.txt and Imap.txt")
            counts, blocks, positions, parser_audit = parse_bpp_alignments(seq_path)
            imap_audit = parse_imap(imap_path)
            raw_retention = None
            if job.family_index == 0:
                raw_retention = _gzip_atomic(
                    seq_path,
                    cache_dir / "raw_audit" / f"{job.job_id}.Seq.txt.gz",
                )
                (cache_dir / "raw_audit" / f"{job.job_id}.MCcoal.ctl").write_text(
                    control, encoding="ascii", newline="\n"
                )
                (cache_dir / "raw_audit" / f"{job.job_id}.Imap.txt").write_bytes(
                    imap_path.read_bytes()
                )
            metadata = {
                "parser_audit": parser_audit,
                "imap_audit": imap_audit,
                "control_sha256": hashlib.sha256(control.encode("ascii")).hexdigest(),
                "bpp_elapsed_seconds": float(elapsed),
                "bpp_returncode": int(completed.returncode),
                "stdout_bytes": len(completed.stdout.encode("utf-8")),
                "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
                "stderr_bytes": len(completed.stderr.encode("utf-8")),
                "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
                "raw_retention": raw_retention,
            }
            save_count_file(
                count_path,
                job,
                config_sha256,
                counts,
                blocks,
                positions,
                metadata,
            )
            verified_counts, verified_blocks, verified_positions, verified_metadata = (
                load_count_file(count_path, job, config_sha256)
            )
            if (
                not np.array_equal(verified_counts, counts)
                or not np.array_equal(verified_blocks, blocks)
                or not np.array_equal(verified_positions, positions)
                or verified_metadata != metadata
            ):
                raise RuntimeError("BPP count checkpoint failed its immediate read-back audit")
            durable_count_checkpoint = True
        finally:
            if durable_count_checkpoint:
                _safe_remove_job_directory(job_dir, work_root)
            else:
                print(
                    f"BPP failure evidence retained at {job_dir.resolve()}",
                    file=sys.stderr,
                    flush=True,
                )
    if compute_state is not None:
        structured.compute_gate(compute_state)
    curve, curve_audit = counts_to_curve(
        counts,
        blocks,
        positions,
        compute_state=compute_state,
    )
    curve32 = curve.astype(np.float32)
    return {
        **job_payload(job),
        "count_file": str(count_path.relative_to(cache_dir).as_posix()),
        "count_file_bytes": count_path.stat().st_size,
        "count_file_sha256": structured.sha256_file(count_path),
        "reused_count_file": bool(reused_count_file),
        "simulation_metadata": metadata,
        "curve_audit": curve_audit,
        "curve_sha256_float32": stdbench._sha256_array(curve32.astype("<f4", copy=False)),
        "curve": curve32,
    }


def record_key(record: dict) -> tuple[int, int]:
    return int(record["family_index"]), CONDITIONS.index(str(record["label"]))


def save_checkpoint(path: Path, records: Sequence[dict], config_sha256: str) -> None:
    records = sorted(records, key=record_key)
    if len({record["job_id"] for record in records}) != len(records):
        raise RuntimeError("refusing to save duplicate BPP jobs")
    curves = np.stack([np.asarray(record["curve"], dtype=np.float32) for record in records])
    if curves.shape != (len(records), 198, 28) or not np.isfinite(curves).all():
        raise RuntimeError("refusing to save invalid BPP curves")
    metadata = []
    for record, curve in zip(records, curves):
        current = {key: value for key, value in record.items() if key != "curve"}
        if stdbench._sha256_array(curve.astype("<f4", copy=False)) != current["curve_sha256_float32"]:
            raise RuntimeError("BPP curve hash changed before checkpoint save")
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
    jobs: Sequence[BPPJob],
    cache_dir: Path,
) -> list[dict]:
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as archive:
        required = {"schema", "config_sha256", "metadata_json", "curves"}
        if set(archive.files) != required:
            raise RuntimeError("BPP checkpoint member set changed")
        if archive["schema"].tolist() != [CHECKPOINT_SCHEMA]:
            raise RuntimeError("BPP checkpoint schema changed")
        if archive["config_sha256"].tolist() != [config_sha256]:
            raise RuntimeError("BPP checkpoint configuration changed")
        metadata = json.loads(str(archive["metadata_json"][0]))
        curves = np.asarray(archive["curves"], dtype=np.float32)
    manifest = {job.job_id: job for job in jobs}
    if len(metadata) != len(curves):
        raise RuntimeError("BPP checkpoint metadata/curve cardinality changed")
    records = []
    seen = set()
    for current, curve in zip(metadata, curves):
        job_id = current.get("job_id")
        if job_id not in manifest or job_id in seen:
            raise RuntimeError("BPP checkpoint has an unknown or duplicate job")
        seen.add(job_id)
        if any(
            current.get(key) != value
            for key, value in job_payload(manifest[job_id]).items()
        ):
            raise RuntimeError(f"BPP checkpoint job manifest changed for {job_id}")
        if curve.shape != (198, 28) or not np.isfinite(curve).all():
            raise RuntimeError(f"BPP checkpoint curve is invalid for {job_id}")
        if stdbench._sha256_array(curve.astype("<f4", copy=False)) != current["curve_sha256_float32"]:
            raise RuntimeError(f"BPP checkpoint curve hash changed for {job_id}")
        count_path = cache_dir / current["count_file"]
        if (
            not count_path.is_file()
            or count_path.stat().st_size != current["count_file_bytes"]
            or structured.sha256_file(count_path) != current["count_file_sha256"]
        ):
            raise RuntimeError(f"BPP count checkpoint changed for {job_id}")
        records.append({**current, "curve": curve})
    return sorted(records, key=record_key)


def record_selection_audit(records: Sequence[dict]) -> dict:
    families = {}
    for record in records:
        families.setdefault(record["family_id"], set()).add(record["label"])
    complete = sum(labels == set(CONDITIONS) for labels in families.values())
    return {
        "records": len(records),
        "complete_B_C_D_families": int(complete),
        "record_curve_hash_ledger_sha256": hashlib.sha256(_canonical_json([
            [record["job_id"], record["curve_sha256_float32"], record["count_file_sha256"]]
            for record in sorted(records, key=record_key)
        ])).hexdigest(),
    }


def checkpoint_audit(path: Path, records: Sequence[dict], config_sha256: str) -> dict:
    records = sorted(records, key=record_key)
    with np.load(path, allow_pickle=False) as archive:
        if archive["schema"].tolist() != [CHECKPOINT_SCHEMA]:
            raise RuntimeError("BPP checkpoint audit found a changed schema")
        if archive["config_sha256"].tolist() != [config_sha256]:
            raise RuntimeError("BPP checkpoint audit found a changed configuration")
        stored_metadata = json.loads(str(archive["metadata_json"][0]))
        stored_curve_shape = list(archive["curves"].shape)
    expected_metadata = [
        {key: value for key, value in record.items() if key != "curve"}
        for record in records
    ]
    if stored_metadata != expected_metadata:
        raise RuntimeError(
            "BPP checkpoint bytes and supplied full-record audit describe different records"
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


def _confusion_payload(truth: np.ndarray, prediction: np.ndarray, labels: Sequence[str]) -> dict:
    matrix = confusion_matrix(truth, prediction, labels=list(labels))
    return {
        "labels": list(labels),
        "matrix": matrix.astype(int).tolist(),
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
        families.setdefault(record["family_id"], set()).add(record["label"])
    if len(families) != 30 or any(labels != set(CONDITIONS) for labels in families.values()):
        raise RuntimeError("full BPP analysis requires 30 complete B/C/D families")
    curves = np.stack([record["curve"] for record in records]).astype(float)
    labels = np.asarray([record["label"] for record in records])
    positive = np.isin(labels, POSITIVE_CLASSES)
    null = labels == "D"
    if compute_state is not None:
        structured.compute_gate(compute_state)
    canonical = structured.load_canonical(canonical_root, max_depth=199)
    canonical_table = np.asarray(canonical["table"], dtype=float)
    canonical_labels = np.asarray(canonical["labels"])
    canonical_rates = np.asarray(canonical["rates"], dtype=float)
    canonical_positive = np.isin(canonical_labels, ["A", "B", "C"])
    primary_rows = len(stdbench.PRIMARY_DEPTHS)
    representations = {}
    representation_predictions = {}
    for name in REPRESENTATIONS:
        if compute_state is not None:
            structured.compute_gate(compute_state)
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
        representation_predictions[name] = (prediction, probability, class_index, z)
        truth = labels[positive]
        call = prediction[positive]
        family_accuracy = []
        for family_id in sorted(families):
            use = np.asarray([record["family_id"] == family_id and record["label"] in POSITIVE_CLASSES for record in records])
            if int(use.sum()) != 2:
                raise AssertionError("BPP family does not contain exactly B and C")
            family_accuracy.append(float(np.mean(prediction[use] == labels[use])))
        representations[name] = {
            "status": "target-blind fixed-C=1 representation; raw_all is primary",
            "feature_dimension": int(train.shape[1]),
            "B_C_balanced_accuracy": float(balanced_accuracy_score(truth, call)),
            "B_recall": stdbench._wilson(call[truth == "B"] == "B"),
            "C_recall": stdbench._wilson(call[truth == "C"] == "C"),
            "confusion": _confusion_payload(truth, call, ("B", "C", "A")),
            "exact_family_accuracy": stdbench._distribution_summary(family_accuracy),
            "D_forced_call_counts_diagnostic_only": {
                str(label): int(count)
                for label, count in zip(*np.unique(prediction[null], return_counts=True))
            },
            "scaler_rms_z_median": float(np.median(np.sqrt(np.mean(z**2, axis=1)))),
            "scaler_rms_z_p95": float(np.quantile(np.sqrt(np.mean(z**2, axis=1)), 0.95)),
            "scaler_max_abs_z_p95": float(np.quantile(np.max(np.abs(z), axis=1), 0.95)),
            "model": stdbench._model_payload(
                scaler,
                model,
                feature_columns=structured.representation_columns(name),
            ),
            "linked_site_SE_guardrail": (
                "raw_all includes PADZE site-level SE even though sites are linked within each "
                "500-bp locus; interpret the pre-specified raw_mean_variance sensitivity "
                "prominently because it removes this non-block-robust coordinate."
                if name == "raw_all"
                else "this representation excludes PADZE site-level SE"
            ),
            "chance_guardrail": (
                "B and C are exactly balanced; a constant B or constant C predictor has 0.5 "
                "balanced accuracy. D is excluded from this denominator."
            ),
            "grid_interval_guardrail": (
                "Wilson intervals are descriptive summaries across a fixed heterogeneous "
                "phi-by-scale grid, not population-sampling confidence intervals. Balanced "
                "accuracy is an equal-grid average, not a prevalence-weighted estimate."
            ),
        }
    gate_train, gate_contract = stdbench._gate_features(canonical_table)
    gate_external, external_contract = stdbench._gate_features(curves)
    if gate_contract != external_contract:
        raise AssertionError("BPP/canonical gate feature contracts differ")
    gate_target = (
        canonical_positive & (canonical_rates >= structured.APPRECIABLE)
    ).astype(int)
    gate_scaler, gate_model = structured._fit_model(gate_train, gate_target, C=1.0)
    gate_z = gate_scaler.transform(gate_external)
    gate_index = int(np.flatnonzero(gate_model.classes_ == 1)[0])
    gate_score = gate_model.predict_proba(gate_z)[:, gate_index]
    gate_truth = positive.astype(int)
    gate_call = gate_score >= 0.5
    primary_prediction, primary_probability, primary_class_index, primary_z = representation_predictions["raw_all"]
    prediction_ledger = []
    for index, record in enumerate(records):
        prediction_ledger.append({
            "job_id": record["job_id"],
            "family_id": record["family_id"],
            "family_index": int(record["family_index"]),
            "family_positive_phi": float(record["family_positive_phi"]),
            "effective_phi": float(record["effective_phi"]),
            "scale": float(record["scale"]),
            "truth": record["label"] if record["label"] in POSITIVE_CLASSES else None,
            "included_in_direction_accuracy": bool(positive[index]),
            "raw_all_prediction": str(primary_prediction[index]),
            "raw_all_correct": bool(primary_prediction[index] == labels[index]) if positive[index] else None,
            "raw_all_probability": {
                label: float(primary_probability[index, primary_class_index[label]])
                for label in ("A", "B", "C")
            },
            "raw_all_scaler_rms_z": float(np.sqrt(np.mean(primary_z[index] ** 2))),
            "raw_all_scaler_max_abs_z": float(np.max(np.abs(primary_z[index]))),
            "appreciable_gate_score": float(gate_score[index]),
            "appreciable_gate_call_at_0_5": bool(gate_call[index]),
        })
    by_phi = {}
    for phi in PHIS:
        use = positive & np.isclose(
            np.asarray(
                [record["family_positive_phi"] for record in records], dtype=float
            ),
            phi,
            rtol=0,
            atol=1e-15,
        )
        by_phi[f"{phi:.3f}"] = {
            "n": int(use.sum()),
            "raw_all_balanced_accuracy": float(
                balanced_accuracy_score(labels[use], primary_prediction[use])
            ),
            "guardrail": "three B and three C rows only; descriptive pilot stratum",
        }
    by_scale = {}
    for scale in SCALES:
        use = positive & np.isclose(
            np.asarray([record["scale"] for record in records], dtype=float), scale,
            rtol=0,
            atol=1e-15,
        )
        by_scale[f"{scale:g}"] = {
            "n": int(use.sum()),
            "raw_all_balanced_accuracy": float(
                balanced_accuracy_score(labels[use], primary_prediction[use])
            ),
        }
    return {
        "statistical_unit": "one independent 500-locus BPP dataset",
        "families": len(families),
        "records": len(records),
        "direction_accuracy_rows": int(positive.sum()),
        "null_rows_excluded_from_direction_accuracy": int(null.sum()),
        "representations": representations,
        "primary_by_phi": by_phi,
        "primary_by_scale": by_scale,
        "appreciable_gate": {
            "status": (
                "secondary frozen transfer diagnostic; canonical target is continuous migration "
                "rate >=2.5e-4, not episodic BPP phi"
            ),
            "contract": gate_contract,
            "positive_sensitivity_at_0_5": stdbench._wilson(gate_call[positive]),
            "D_specificity_at_0_5": stdbench._wilson(~gate_call[null]),
            "B_C_D_roc_auc": float(roc_auc_score(gate_truth, gate_score)),
            "positive_score": stdbench._distribution_summary(gate_score[positive]),
            "D_score": stdbench._distribution_summary(gate_score[null]),
            "model": stdbench._model_payload(gate_scaler, gate_model),
            "grid_interval_guardrail": (
                "Sensitivity and specificity intervals summarize the fixed heterogeneous "
                "design grid only; they are not population-sampling confidence intervals."
            ),
        },
        "simulation_record_ledger": [
            {key: value for key, value in record.items() if key != "curve"}
            for record in records
        ],
        "prediction_ledger": prediction_ledger,
        "canonical_source_audit": canonical["audit"],
        "guardrail": (
            "This is a derivative known-truth simulation bank. Ji et al. used four copies per "
            "population and BPP detection power is not a direction-accuracy comparator. Phi is "
            "episodic ancestry proportion and must not be scored as DNNaic migration magnitude. "
            "B/C/D within a family share parameters but use independent seeds, not paired common "
            "random numbers. D repeats ten independently simulated nulls per scale even though "
            "its nominal family-positive phi is ignored and its effective phi is always zero."
        ),
    }


def configuration(jobs: Sequence[BPPJob], source: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": source,
        "jobs": [job_payload(job) for job in jobs],
        "phis": list(PHIS),
        "scales": list(SCALES),
        "base_parameters": BASE_PARAMETERS,
        "mapping": DNNAIC_MAPPING,
        "truth": {"B": "Q->D", "C": "D->Q", "D": "no event"},
        "null_provenance": (
            "D reconstructs the paper's inflow-asymmetric phi=0 false-positive "
            "protocol as a donor-phi=0/resident-phi=1 boundary control; the official "
            "Zenodo archive does not supply a separate null MCcoal control file"
        ),
        "sampling": {"Q": 200, "R": 200, "D": 200, "S_unused_outgroup": 1},
        "stochastic_design": (
            "one independently seeded dataset per class and phi-by-scale cell; B/C/D are "
            "parameter-matched but are not paired common-random-number replicates"
        ),
        "alignments": {
            "independent_loci": LOCUS_COUNT,
            "sites_per_locus": LOCUS_LENGTH,
            "mutation_model": "BPP model=0 JC69",
            "linkage": "sites linked within locus; loci independent",
        },
        "padze": {
            "depths": stdbench.FULL_DEPTHS.tolist(),
            "moments": list(stdbench.MOMENTS),
            "pihat_sizes": [2],
            "bias_corrected": True,
            "primary_direction_depths": stdbench.PRIMARY_DEPTHS.tolist(),
            "raw_SE_guardrail": "site-level PADZE SE is not block-robust",
        },
        "paper_comparison_guardrail": (
            "The official paper used four copies/population and reported BPP detection power. "
            "This 200-copy derivative cannot be numerically compared as published accuracy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bpp-binary", type=Path, required=True)
    parser.add_argument(
        "--bpp-version", choices=tuple(BPP_RELEASES), default="4.6.1"
    )
    parser.add_argument("--official-control-dir", type=Path, required=True)
    parser.add_argument("--official-archive", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--limit-families", type=int, default=None)
    parser.add_argument("--simulate-only", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--compute-state", type=Path, default=structured.DEFAULT_COMPUTE_STATE)
    parser.add_argument("--compute-target", choices=("local", "azure"), default="local")
    parser.add_argument("--allow-stopped-trading-compute", action="store_true")
    parser.add_argument("--allow-closing-owner-session", action="store_true")
    args = parser.parse_args()
    if args.limit_families is not None and not 1 <= args.limit_families <= 30:
        parser.error("--limit-families must lie in [1,30]")
    if args.limit_families is not None and not args.simulate_only:
        parser.error("--limit-families requires --simulate-only")
    if args.timeout_seconds < 1:
        parser.error("--timeout-seconds must be positive")
    os.environ[structured.COMPUTE_TARGET_ENV] = args.compute_target
    if args.allow_stopped_trading_compute:
        os.environ[structured.STOPPED_TRADING_AUTH_ENV] = "1"
    if args.allow_closing_owner_session:
        os.environ[structured.AZURE_CLOSING_OWNER_AUTH_ENV] = "1"
    initial_gate = structured.compute_gate(args.compute_state)
    priority = structured.set_below_normal_priority()
    revision = structured.git_revision(script=Path(__file__))
    structured.require_clean_tracked_revision(revision)
    source = source_audit(
        args.bpp_binary,
        args.official_control_dir,
        args.official_archive,
        args.bpp_version,
    )
    jobs = make_jobs()
    config = configuration(jobs, source)
    config_sha256 = hashlib.sha256(_canonical_json(config)).hexdigest()
    requested_families = 30 if args.limit_families is None else args.limit_families
    requested_jobs = [job for job in jobs if job.family_index < requested_families]
    checkpoint = args.cache_dir / "bpp2023_mccoal_features.npz"
    with structured.SingleWriterLease(args.cache_dir, ".bpp2023_mccoal.lock"):
        records = load_checkpoint(checkpoint, config_sha256, jobs, args.cache_dir)
        completed = {record["job_id"] for record in records}
        for index, job in enumerate(requested_jobs, start=1):
            if job.job_id in completed:
                continue
            structured.compute_gate(args.compute_state)
            record = simulate_job(
                job,
                binary=args.bpp_binary,
                cache_dir=args.cache_dir,
                config_sha256=config_sha256,
                compute_state=args.compute_state,
                timeout_seconds=args.timeout_seconds,
            )
            records.append(record)
            completed.add(job.job_id)
            save_checkpoint(checkpoint, records, config_sha256)
            print(
                f"[{index}/{len(requested_jobs)}] {job.job_id}: "
                f"{record['simulation_metadata']['parser_audit']['retained_polymorphic_sites']} sites, "
                f"BPP {record['simulation_metadata']['bpp_elapsed_seconds']:.2f}s",
                flush=True,
            )
        requested_ids = {job.job_id for job in requested_jobs}
        selected = [record for record in records if record["job_id"] in requested_ids]
        if len(selected) != len(requested_jobs):
            raise RuntimeError("BPP checkpoint lacks a requested job")
    if args.simulate_only:
        final_revision = structured.git_revision(script=Path(__file__))
        structured.require_revision_unchanged(revision, final_revision)
        print(json.dumps({
            "checkpoint": checkpoint_audit(checkpoint, records, config_sha256),
            "requested_selection": record_selection_audit(selected),
            "configuration_sha256": config_sha256,
            "source_commit": final_revision["commit"],
            "requested_families": requested_families,
        }, indent=2, allow_nan=False))
        return 0
    if len(selected) != len(jobs):
        raise RuntimeError("BPP full analysis requires all 90 jobs")
    result_lock = structured.SingleWriterLease(
        args.result_dir, ".bpp2023_mccoal_result.lock"
    ).acquire()
    pre_analysis_gate = structured.compute_gate(args.compute_state)
    analysis = analyze_records(
        selected,
        args.canonical_root,
        compute_state=args.compute_state,
    )
    final_revision = structured.git_revision(script=Path(__file__))
    structured.require_revision_unchanged(revision, final_revision)
    runtime = structured.runtime_audit(priority)
    runtime["packages"]["padze"] = importlib_metadata.version("padze")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "known_truth_B_C_D_derivative_simulation_not_published_accuracy_reproduction",
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
        "primary_B_C_balanced_accuracy": analysis["representations"]["raw_all"][
            "B_C_balanced_accuracy"
        ],
        "gate_roc_auc": analysis["appreciable_gate"]["B_C_D_roc_auc"],
    }, indent=2, allow_nan=False))
    result_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
