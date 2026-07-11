# Legacy tinkerbird asymmetric-backcrossing OOD benchmark

This bundle evaluates the released depth-matched direction and appreciable-flow
heads on the Kirschel et al. (2020) tinkerbird ddRAD release (paper DOI
[`10.1111/mec.15691`](https://doi.org/10.1111/mec.15691); data DOI
[`10.5061/dryad.jm63xsj87`](https://doi.org/10.5061/dryad.jm63xsj87), CC0).

This is not external validation and does not contribute an accuracy numerator
or denominator. The 14 P2 birds were selected as admixed from the same data,
and their direction label reuses autosomal-versus-Z ancestry in those birds.
The pooled hybrids are not a panmictic tree leaf. Candidate class C is only the
published majority cross: nine birds support relatively pusillus-enriched
fathers with extoni-enriched mothers, three support the reciprocal, and two are
equal.

## Source and sample audit

The paper's rendered Table 2 names 13 admixed birds. Its Discussion separately
identifies AR93163 as a pusillus-haplotype female with an extoni father; the
manifest therefore uses those 13 rows plus that explicitly named fourteenth
bird. It restores six source-era pusillus reference birds omitted from the
prototype, producing P1/P2/P3 sizes 12/14/17.

The P1/P3 labels are deliberately `Legacy*Reference`, not `Allopatry`.
Sebastianelli et al. (2024) later classify AR93110 and AR93178 as sympatric,
so the old geographic pools are not stable exact current allopatric sets. The
2024 study nevertheless supports strongly asymmetric mating/backcrossing
([`10.1038/s41467-024-47305-5`](https://doi.org/10.1038/s41467-024-47305-5)).
Current phylogenomics supports bidirectional introgression tails, so candidate
C is not an exclusive unidirectional migration edge
([`10.1093/sysbio/syaf033`](https://doi.org/10.1093/sysbio/syaf033)).

The pinned 3,223,094-byte source has SHA-256
`51144aabaddac820269af2f8ff5648393b69a20be3c0398a72ca4d9c83756a51`
and contains 85 birds at 104,933 variants. Exact scaffold counts are 57,913
anchored-autosomal, 1,533 Z, and 45,487 unplaced variants; the paper reports
1,532 Z variants. Across 1,533 Z rows, all 12,264 cells for the eight P2 females
are encoded with two called alleles, including 1,302 heterozygous cells.
Because female birds are ZW, every scored panel excludes Z and unplaced
scaffolds rather than consuming those calls as ordinary diploid autosomes.

## Result

| scope | loci | direction | candidate | gate | direction RMS z | gate RMS z | projection P1->P3 | f3(P2;P1,P3) |
|---|---:|---|---|---:|---:|---:|---:|---:|
| linked SNPs, standard | 15,000 | A | C majority | 1.000 | 20.85 | 26.37 | 0.37464 | -0.04115 |
| linked SNPs, within-population variation | 15,000 | A | C majority | 1.000 | 23.10 | 26.73 | 0.36729 | -0.04889 |
| first eligible SNP/scaffold, standard | 8,638 | A | C majority | 1.000 | 20.14 | 26.07 | 0.37435 | -0.04194 |
| first eligible SNP/scaffold, within-population variation | 3,985 | A | C majority | 1.000 | 21.97 | 26.51 | 0.36705 | -0.05107 |

Every scope calls A rather than candidate-majority C, every gate saturates at
1.0, and RMS feature shifts remain about 20-27 simulation-training standard
deviations. The conclusion survives removal of every within-scaffold duplicate.
It is an end-to-end severe-OOD transfer failure, not an accuracy estimate or a
calibrated probability statement.

The descriptive projection remains near 0.37 under every scope. Its
500-replicate scaffold-block bootstrap 95% intervals are 0.37009-0.37904 for
the linked standard panel and 0.37007-0.37818 for the scaffold-thinned standard
panel. Corresponding f3 intervals are -0.04256 to -0.03984 and -0.04342 to
-0.04065. These reference-invariant statistics support intermediate/admixture
geometry but cannot orient time and are not bounded ancestry estimates.

The within-population-variation scopes contain no loci with absolute P1-P3
frequency difference at least 0.95; their diagnostic projection is JSON
`null`, never non-standard `NaN`. This filter is strong ascertainment, not a
quality filter. The two independently capped linked-SNP scopes overlap at only
4,326 of 15,000 loci (Jaccard 0.1685), which is reported rather than treated as
a paired filter comparison.

## Reproduction

Run from clean code commit
`23518cfc88489da6d222c519d8e7f13159770346` with all BLAS/OpenMP thread
variables fixed to one:

```bash
python scripts/tinkerbird_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --source-vcf /path/to/revision.recode.vcf.gz \
  --cache-dir /path/to/derived \
  --result-dir results/tinkerbird_external_benchmark_2026_07_11
```

The clean run records `dirty_at_run=false`. `results.json` is 148,249 bytes
with SHA-256
`c42451640a8253c0bb830687f0f1e6a00a3098e1b8a1f5b677b5b6f0231d225d`.
An independent detached-worktree run was identical after normalizing only
repository, cache, and result paths; both canonical objects hash to
`75fcdc6209a623579f41e5a11d474ff863166c9e5f4c1fcc98f138e06d5c80c5`.

The deterministic anchored-autosome VCF hashes to
`ab4cd50d717de2c110cf18def0c624c4e4950e38162dc9e6dce08ba0f7545c23`.
Among 8,815 autosomal scaffolds, 8,744 contain a structurally eligible SNP; the
first-eligible-per-scaffold source hashes to
`f51103003cfe5fe6918d288b5396577bc8a5ea2c2643af5f83f700346dee51cb`.
