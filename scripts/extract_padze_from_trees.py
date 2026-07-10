#!/usr/bin/env python3
"""Convert a DNNaic tree-sequence manifest into the published PADZE arrays.

This is the public tree-sequence-to-feature link that was previously missing.
It reads ``simulation_manifest.json`` from ``simulate_demography.py``, verifies
the manifest SHA-256, converts each tree sequence to per-population allele-count
matrices, and computes the 28-column DNNaic contract at depths g=2,...,199.

Usage:
    python scripts/extract_padze_from_trees.py --trees-dir data/raw/trees --out OUTPUT
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from padze import LociData, Metadata, compute_features

from simulate_demography import MANIFEST_SCHEMA, compute_manifest_hash


POPULATIONS = ("P1", "P2", "P3")
MOMENTS = ("mean", "variance", "se")
DEPTHS = np.arange(2, 200, dtype=np.int64)


def _population_name(population: Any) -> str | None:
    metadata = population.metadata
    if isinstance(metadata, Mapping):
        return metadata.get("name")
    if isinstance(metadata, bytes):
        try:
            metadata = json.loads(metadata.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        if isinstance(metadata, Mapping):
            return metadata.get("name")
    return None


def ts_to_loci(tree_sequence: Any, expected_gene_copies: int | None = 200) -> LociData:
    """Create PADZE ``LociData`` from P1/P2/P3 samples in a tree sequence.

    Biallelic sites use a vectorized path; the rare multiallelic site is counted
    allele by allele.  Monomorphic sites are dropped.  Population names are read
    from tree-sequence metadata rather than inferred from numeric order.
    """

    id_by_name: dict[str, int] = {}
    for population_id in range(tree_sequence.num_populations):
        name = _population_name(tree_sequence.population(population_id))
        if name is not None:
            id_by_name[name] = population_id
    missing = [name for name in POPULATIONS if name not in id_by_name]
    if missing:
        raise ValueError(
            "tree sequence lacks named experimental populations: " + ", ".join(missing)
        )

    samples = tree_sequence.samples()
    sample_populations = tree_sequence.tables.nodes.population[samples]
    masks = [sample_populations == id_by_name[name] for name in POPULATIONS]
    sample_sizes = np.array([int(mask.sum()) for mask in masks], dtype=np.int64)
    if expected_gene_copies is not None and not np.all(
        sample_sizes == expected_gene_copies
    ):
        raise ValueError(
            f"expected {expected_gene_copies} gene copies in each of P1/P2/P3; "
            f"observed {dict(zip(POPULATIONS, sample_sizes.tolist()))}"
        )

    genotypes = tree_sequence.genotype_matrix()
    count_matrices: list[np.ndarray] = []
    locus_ids: list[str] = []
    if genotypes.shape[0]:
        if np.any(genotypes < 0):
            raise ValueError("missing genotypes are not expected in msprime output")
        maximum_allele = genotypes.max(axis=1)
        alternate = np.stack(
            [genotypes[:, mask].sum(axis=1) for mask in masks], axis=1
        )
        reference = sample_sizes[None, :] - alternate

        for site_index in range(genotypes.shape[0]):
            max_allele = int(maximum_allele[site_index])
            if max_allele == 1:
                total_alternate = int(alternate[site_index].sum())
                if total_alternate in (0, int(sample_sizes.sum())):
                    continue
                counts = np.empty((len(POPULATIONS), 2), dtype=np.int64)
                counts[:, 0] = reference[site_index]
                counts[:, 1] = alternate[site_index]
            elif max_allele >= 2:
                row = genotypes[site_index]
                counts = np.stack(
                    [
                        np.bincount(row[mask], minlength=max_allele + 1)
                        for mask in masks
                    ]
                ).astype(np.int64, copy=False)
                if np.count_nonzero(counts.sum(axis=0)) < 2:
                    continue
            else:
                continue
            count_matrices.append(counts)
            locus_ids.append(f"s{site_index}")

    sizes = (
        np.vstack([counts.sum(axis=1) for counts in count_matrices])
        if count_matrices
        else np.zeros((0, len(POPULATIONS)), dtype=np.int64)
    )
    metadata = Metadata(
        source="msprime tree sequence",
        populations=list(POPULATIONS),
        sample_ids={name: [] for name in POPULATIONS},
        ploidy={name: 1 for name in POPULATIONS},
        n_loci_read=tree_sequence.num_sites,
        n_loci_kept=len(count_matrices),
        filters_applied=["polymorphic across P1/P2/P3"],
        missing_fraction=0.0,
    )
    return LociData(
        populations=list(POPULATIONS),
        count_matrices=count_matrices,
        sample_sizes=sizes,
        locus_ids=locus_ids,
        metadata=metadata,
    )


def tree_to_feature_matrix(
    tree_sequence: Any,
    *,
    depths: Sequence[int] = DEPTHS,
    expected_gene_copies: int | None = 200,
) -> tuple[np.ndarray, list[str], int]:
    loci = ts_to_loci(tree_sequence, expected_gene_copies=expected_gene_copies)
    if not loci.count_matrices:
        raise ValueError("tree sequence contains no polymorphic P1/P2/P3 sites")
    table = compute_features(
        loci,
        depths=np.asarray(depths, dtype=np.int64),
        pihat_sizes=(2,),
        moments=MOMENTS,
        bias_corrected=True,
    )
    matrix, columns = table.to_frame()
    return matrix.astype(float), list(columns), loci.metadata.n_loci_kept


def load_and_verify_manifest(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported manifest schema {document.get('schema')!r}; "
            f"expected {MANIFEST_SCHEMA!r}"
        )
    jobs = document.get("jobs")
    config = document.get("config")
    if not isinstance(jobs, list) or not isinstance(config, dict):
        raise ValueError("manifest must contain object 'config' and list 'jobs'")
    observed_hash = compute_manifest_hash(config, jobs)
    if observed_hash != document.get("manifest_hash"):
        raise ValueError(
            "manifest SHA-256 mismatch: "
            f"stored={document.get('manifest_hash')} computed={observed_hash}"
        )
    if document.get("job_count") != len(jobs):
        raise ValueError(
            f"manifest job_count={document.get('job_count')} but contains {len(jobs)} jobs"
        )
    return document


def save_arrays(
    out: Path,
    matrices: Sequence[np.ndarray],
    jobs: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    source_manifest_hash: str,
) -> None:
    matrix = np.vstack(matrices)
    direction: list[str] = []
    magnitude: list[float] = []
    groups: list[str] = []
    design: list[str] = []
    for features, job in zip(matrices, jobs):
        row_count = features.shape[0]
        direction.extend([str(job["class"])] * row_count)
        magnitude.extend([float(job["rate"])] * row_count)
        groups.extend([str(job["group"])] * row_count)
        design.extend([str(job["design"])] * row_count)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X.npy", matrix)
    np.save(out / "direction.npy", np.asarray(direction))
    np.save(out / "magnitude.npy", np.asarray(magnitude, dtype=float))
    np.save(out / "groups.npy", np.asarray(groups))
    np.save(out / "design.npy", np.asarray(design))
    (out / "columns.txt").write_text(
        "\n".join(columns) + "\n", encoding="utf-8"
    )
    summary = {
        "source_manifest_hash": source_manifest_hash,
        "replicates": len(jobs),
        "rows": int(matrix.shape[0]),
        "columns": int(matrix.shape[1]),
        "depth_min": int(matrix[:, 0].min()),
        "depth_max": int(matrix[:, 0].max()),
    }
    (out / "extraction_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trees-dir", default="data/raw/trees")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default="data/simulation_data/regen_full")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="extract only the first N jobs (smoke testing only)",
    )
    args = parser.parse_args()

    trees_dir = Path(args.trees_dir)
    manifest_path = (
        Path(args.manifest)
        if args.manifest is not None
        else trees_dir / "simulation_manifest.json"
    )
    document = load_and_verify_manifest(manifest_path)
    jobs = document["jobs"]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        jobs = jobs[:args.limit]

    expected_gene_copies = int(
        document["config"]["demography"]["gene_copies_per_population"]
    )
    matrices: list[np.ndarray] = []
    columns: list[str] | None = None
    for index, job in enumerate(jobs, start=1):
        tree_path = trees_dir / job["tree_file"]
        if not tree_path.is_file():
            raise FileNotFoundError(
                f"missing tree sequence for manifest job {job['ordinal']}: {tree_path}"
            )
        import tskit

        tree_sequence = tskit.load(str(tree_path))
        features, current_columns, n_loci = tree_to_feature_matrix(
            tree_sequence, expected_gene_copies=expected_gene_copies
        )
        if columns is None:
            columns = current_columns
        elif current_columns != columns:
            raise ValueError(f"feature-column drift at {tree_path}")
        matrices.append(features)
        if index % 50 == 0 or index == len(jobs):
            print(
                f"  [{index}/{len(jobs)}] {job['group']}: {n_loci} loci, "
                f"{features.shape[0]} depths",
                flush=True,
            )

    if not matrices or columns is None:
        raise ValueError("manifest selected no jobs")
    save_arrays(
        Path(args.out), matrices, jobs, columns, document["manifest_hash"]
    )
    print(
        f"[extract] wrote {len(jobs)} replicates and "
        f"{sum(matrix.shape[0] for matrix in matrices)} rows -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
