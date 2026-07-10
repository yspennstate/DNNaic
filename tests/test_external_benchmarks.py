"""Contract tests for deterministic external-VCF preparation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from external_benchmarks import prepare_vcf, subset_prepared_vcf  # noqa: E402
from dnnaic import build_matrix  # noqa: E402


def test_external_filter_is_deterministic_and_feature_compatible(tmp_path):
    manifest = tmp_path / "samples.tsv"
    manifest.write_text(
        "sample\tpopulation\n"
        "a1\tP1\n"
        "a2\tP1\n"
        "b1\tP2\n"
        "b2\tP2\n"
        "c1\tP3\n"
        "c2\tP3\n",
        encoding="utf-8",
    )
    source = tmp_path / "source.vcf"
    source.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        "a1\ta2\tb1\tb2\tc1\tc2\n"
        "1\t10\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/1\t0/0\t0/1\t1/1\t0/1\n"
        "1\t20\t.\tA\tG,T\t.\tPASS\t.\tGT\t0/0\t0/1\t0/0\t0/1\t1/1\t0/1\n"
        "1\t30\t.\tA\tG\t.\tPASS\t.\tGT\t./.\t./.\t0/0\t0/1\t1/1\t0/1\n"
        "1\t40\t.\tC\tT\t.\t.\t.\tGT\t0/1\t0/1\t0/0\t0/1\t1/1\t0/1\n"
        "1\t50\t.\tG\tA\t.\tPASS\t.\tGT\t0/0\t0/1\t0/1\t0/1\t1/1\t0/1\n"
        "1\t60\t.\tA\tC\t.\tPASS\t.\tGT\t0/0\t0/0\t0/0\t0/0\t0/0\t0/0\n",
        encoding="utf-8",
    )

    first_vcf = tmp_path / "first.vcf"
    first_map = tmp_path / "first.tsv"
    second_vcf = tmp_path / "second.vcf"
    second_map = tmp_path / "second.tsv"
    first = prepare_vcf(source, manifest, first_vcf, first_map, cap=2, seed=17, min_called_copies=2)
    second = prepare_vcf(source, manifest, second_vcf, second_map, cap=2, seed=17, min_called_copies=2)

    assert first["counts"] == {
        "source_variant_rows": 6,
        "eligible_before_cap": 3,
        "not_biallelic_snp": 1,
        "insufficient_called_copies": 1,
        "not_polymorphic_in_every_panel": 1,
        "retained_after_cap": 2,
    }
    assert first["ordered_locus_sha256"] == second["ordered_locus_sha256"]
    assert first["derived_vcf"]["sha256"] == second["derived_vcf"]["sha256"]
    assert first["derived_popmap"]["sha256"] == second["derived_popmap"]["sha256"]

    X, columns, loci = build_matrix(
        str(first_vcf), str(first_map), max_depth=2, pop_order=["P1", "P2", "P3"]
    )
    assert X.shape == (1, 28)
    assert columns[0] == "g"
    assert np.isfinite(X).all()
    assert loci.metadata.n_loci_kept == 2

    subset_vcf = tmp_path / "subset.vcf"
    subset_map = tmp_path / "subset.tsv"
    subset = subset_prepared_vcf(
        first_vcf, manifest, subset_vcf, subset_map, shared_audit=first
    )
    assert subset["ordered_locus_sha256"] == first["ordered_locus_sha256"]
    assert subset["counts"]["retained_after_cap"] == 2
    assert "same ordered locus intersection" in subset["comparison_locus_contract"]


def test_shared_filter_requires_polymorphism_in_every_declared_panel(tmp_path):
    union = tmp_path / "union.tsv"
    union.write_text(
        "sample\tpopulation\n"
        "s1\tP1\n"
        "s2\tP2\n"
        "s3\tP3\n"
        "s4\tP4\n",
        encoding="utf-8",
    )
    panel_a = tmp_path / "panel_a.tsv"
    panel_a.write_text("sample\tpopulation\ns1\tP1\ns2\tP2\ns3\tP3\n", encoding="utf-8")
    panel_b = tmp_path / "panel_b.tsv"
    panel_b.write_text("sample\tpopulation\ns1\tP1\ns2\tP2\ns4\tP4\n", encoding="utf-8")
    source = tmp_path / "union.vcf"
    source.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        "s1\ts2\ts3\ts4\n"
        "1\t10\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/0\t1/1\t0/0\n"
        "1\t20\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/0\t0/0\t1/1\n"
        "1\t30\t.\tA\tG\t.\tPASS\t.\tGT\t0/0\t0/0\t1/1\t1/1\n"
        "1\t40\t.\tA\tG\t.\tPASS\t.\tGT\t1/1\t0/0\t0/0\t0/0\n",
        encoding="utf-8",
    )
    audit = prepare_vcf(
        source,
        union,
        tmp_path / "shared.vcf",
        tmp_path / "shared.tsv",
        cap=10,
        seed=1,
        min_called_copies=2,
        require_three_populations=False,
        polymorphic_panel_manifests=(panel_a, panel_b),
    )
    assert audit["counts"] == {
        "source_variant_rows": 4,
        "not_polymorphic_in_every_panel": 2,
        "eligible_before_cap": 2,
        "retained_after_cap": 2,
    }
