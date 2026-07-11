import json
from pathlib import Path

import numpy as np

from scripts import tinkerbird_external_benchmark as tinkerbird
from scripts.external_benchmarks import read_manifest


def test_tinkerbird_source_contract_is_pinned_and_autosome_only():
    assert tinkerbird.SOURCE["bytes"] == 3_223_094
    assert tinkerbird.SOURCE["sha256"] == (
        "51144aabaddac820269af2f8ff5648393b69a20be3c0398a72ca4d9c83756a51"
    )
    assert tinkerbird.SOURCE["expected_samples"] == 85
    assert tinkerbird.SOURCE["expected_variants"] == 104_933
    assert tinkerbird.SOURCE["expected_anchored_autosome_variants"] == 57_913
    assert tinkerbird.SOURCE["expected_Z_variants"] == 1_533
    assert tinkerbird.SOURCE["expected_unplaced_variants"] == 45_487
    assert tinkerbird.SOURCE["paper_doi"] == "10.1111/mec.15691"
    assert tinkerbird.is_anchored_autosome("scaf_01_1877")
    assert tinkerbird.is_anchored_autosome("scaf_01A_8592")
    assert tinkerbird.is_anchored_autosome("scaf_04A_36906")
    assert not tinkerbird.is_anchored_autosome("scaf_Z_123")
    assert not tinkerbird.is_anchored_autosome("scaf76701")


def test_tinkerbird_manifest_is_exact_complete_and_disjoint():
    mapping = read_manifest(tinkerbird.MANIFEST)
    assert mapping == tinkerbird.validate_manifest()
    counts = {
        population: list(mapping.values()).count(population)
        for population in tinkerbird.POPULATION_ORDER
    }
    assert counts == {
        "LegacyExtoniReference": 12,
        "Admixed14": 14,
        "LegacyPusillusReference": 17,
    }
    assert len(mapping) == 43
    assert set(tinkerbird.EXTONI_REFERENCE).isdisjoint(tinkerbird.ADMIXED14)
    assert set(tinkerbird.EXTONI_REFERENCE).isdisjoint(tinkerbird.PUSILLUS_REFERENCE)
    assert set(tinkerbird.ADMIXED14).isdisjoint(tinkerbird.PUSILLUS_REFERENCE)
    assert "AR93163" in mapping
    assert set(f"AR931{number}" for number in range(76, 82)).issubset(mapping)


def test_published_sample_audit_exposes_heterogeneity_and_label_reuse():
    audit = tinkerbird.published_sample_audit()
    assert audit["prose_reported_admixed_samples"] == 14
    assert audit["table_2_named_samples"] == 13
    assert audit["fourteenth_sample"]["sample"] == "AR93163"
    assert audit["prose_reported_parental_direction_counts"] == {
        "more_extoni_mother_more_pusillus_father": 9,
        "reverse": 3,
        "equal_autosomal_and_Z_ancestry": 2,
    }
    assert audit["accuracy_eligible"] is False
    assert "same birds" in audit["direct_label_source_reuse"]


def _write_toy_panel(
    tmp_path: Path,
    rows: list[tuple[str, str, str]],
    sample_order: tuple[str, str, str] = ("s1", "s2", "s3"),
) -> tuple[Path, Path]:
    manifest = tmp_path / "panel.tsv"
    manifest.write_text(
        "sample\tpopulation\n"
        "s1\tLegacyExtoniReference\n"
        "s2\tAdmixed14\n"
        "s3\tLegacyPusillusReference\n",
        encoding="utf-8",
    )
    genotype_by_sample = {
        sample: [row[index] for row in rows]
        for index, sample in enumerate(("s1", "s2", "s3"))
    }
    vcf = tmp_path / "panel.vcf"
    lines = [
        "##fileformat=VCFv4.2",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(sample_order),
    ]
    for index in range(len(rows)):
        genotypes = [genotype_by_sample[sample][index] for sample in sample_order]
        lines.append(
            f"scaf_{index + 1:02d}_tag\t{index + 1}\trs{index + 1}\tA\tG\t.\tPASS\t.\tGT\t"
            + "\t".join(genotypes)
        )
    vcf.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return vcf, manifest


def test_frequency_projection_is_halfway_flip_invariant_and_header_invariant(tmp_path: Path):
    rows = [("0/0", "0/1", "1/1"), ("0/0", "0/1", "1/1")]
    vcf, manifest = _write_toy_panel(tmp_path, rows)
    result = tinkerbird.frequency_projection(
        vcf, manifest, bootstrap_replicates=20
    )
    assert result["n_loci"] == 2
    assert result["diagnostic_loci"] == 2
    assert np.isclose(result["P2_projection_from_P1_toward_P3_all_loci"], 0.5)
    assert np.isclose(
        result["P2_projection_from_P1_toward_P3_diagnostic_loci"], 0.5
    )
    assert np.isclose(result["f3_P2_P1_P3"], -0.25)

    flipped_dir = tmp_path / "flipped"
    flipped_dir.mkdir()
    flipped, flipped_manifest = _write_toy_panel(
        flipped_dir,
        [("1/1", "1/0", "0/0"), ("1/1", "1/0", "0/0")],
        sample_order=("s3", "s1", "s2"),
    )
    flipped_result = tinkerbird.frequency_projection(
        flipped, flipped_manifest, bootstrap_replicates=20
    )
    assert np.isclose(
        flipped_result["P2_projection_from_P1_toward_P3_all_loci"],
        result["P2_projection_from_P1_toward_P3_all_loci"],
    )
    assert np.isclose(flipped_result["f3_P2_P1_P3"], result["f3_P2_P1_P3"])


def test_frequency_projection_uses_null_not_nan_for_empty_or_zero_axis(tmp_path: Path):
    vcf, manifest = _write_toy_panel(
        tmp_path, [("0/0", "0/1", "0/1"), ("0/0", "0/1", "0/1")]
    )
    result = tinkerbird.frequency_projection(vcf, manifest, bootstrap_replicates=20)
    assert result["diagnostic_loci"] == 0
    assert result["P2_projection_from_P1_toward_P3_diagnostic_loci"] is None
    assert np.isclose(result["P2_projection_from_P1_toward_P3_all_loci"], 1.0)
    assert result["scaffold_block_bootstrap"]["projection_diagnostic_loci"] is None
    json.dumps(result, allow_nan=False)

    zero_dir = tmp_path / "zero"
    zero_dir.mkdir()
    zero_vcf, zero_manifest = _write_toy_panel(
        zero_dir, [("0/0", "0/1", "0/0"), ("0/0", "0/1", "0/0")]
    )
    zero = tinkerbird.frequency_projection(
        zero_vcf, zero_manifest, bootstrap_replicates=20
    )
    assert zero["P2_projection_from_P1_toward_P3_all_loci"] is None
    assert zero["P2_projection_from_P1_toward_P3_diagnostic_loci"] is None
    assert zero["scaffold_block_bootstrap"]["projection_all_loci"] is None
    json.dumps(zero, allow_nan=False)


def test_locus_overlap_reports_independent_filter_sets(tmp_path: Path):
    standard, _ = _write_toy_panel(
        tmp_path,
        [("0/0", "0/1", "1/1"), ("0/0", "0/1", "1/1")],
    )
    strict_dir = tmp_path / "strict"
    strict_dir.mkdir()
    strict, _ = _write_toy_panel(
        strict_dir,
        [("0/0", "0/1", "1/1")],
    )
    overlap = tinkerbird.locus_overlap(standard, strict)
    assert overlap["standard_loci"] == 2
    assert overlap["within_population_polymorphism_loci"] == 1
    assert overlap["intersection"] == 1
    assert np.isclose(overlap["jaccard"], 0.5)


def test_scaffold_thinning_skips_indel_before_first_eligible_snp(tmp_path: Path):
    source = tmp_path / "source.vcf"
    output = tmp_path / "thinned.vcf"
    source.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
        "scaf_01_a\t1\tbad\tAA\tG\t.\tPASS\t.\tGT\t0/1\n"
        "scaf_01_a\t2\tgood\tA\tG\t.\tPASS\t.\tGT\t0/1\n"
        "scaf_02_b\t3\tbad2\tA\tG,T\t.\tPASS\t.\tGT\t0/1\n",
        encoding="utf-8",
    )
    audit = tinkerbird.thin_one_per_scaffold(
        source,
        output,
        expected_source_rows=None,
        expected_source_scaffolds=None,
        expected_retained_scaffolds=None,
    )
    variants = [
        line for line in output.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#")
    ]
    assert len(variants) == 1
    assert variants[0].split("\t")[2] == "good"
    assert audit["source_scaffolds"] == 2
    assert audit["retained_scaffolds"] == 1
    assert audit["scaffolds_without_structurally_eligible_SNP"] == 1
