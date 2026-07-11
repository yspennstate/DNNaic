# European seabass directional transfer benchmark

This bundle evaluates the released depth-matched direction and appreciable-flow
heads on the Robinet et al. European seabass data (paper DOI
`10.1111/mec.15611`; Zenodo `10.5281/zenodo.3989825`, CC BY 4.0).

The fixed topology is `((PENI,P2),MED)`. The positive uses SINE as P2; the
source study interprets its Mediterranean ancestry excess as MED P3 -> SINE
P2, DNNaic class C. The matched contrast replaces SINE with VIGO. VIGO has
slightly less Mediterranean ancestry than the fixed PENI reference, but it is
not a claim of absolute no gene flow.

## Source benchmark

The raw author ADMIXTURE file gives:

| population | n | mean MED ancestry | sample SD | range | n >= 0.15 |
|---|---:|---:|---:|---:|---:|
| SINE positive P2 | 27 | 0.111878 | 0.092141 | 0.000010-0.402736 | 6 |
| PENI fixed P1 | 29 | 0.044890 | 0.037187 | 0.010237-0.172110 | 1 |
| VIGO contrast P2 | 30 | 0.036501 | 0.019039 | 0.000010-0.086898 | 0 |

The source summary CSV reports `VIGO min_anc_MED=0.187`, which exceeds its own
reported maximum and is incompatible with the raw Q file. `results.json`
retains that discrepancy and uses the raw ID-keyed Q values. The released PED
and Q/metadata files also substitute three non-benchmark Atlantic fish
(`DLAB_0082`, `DLAB_0135`, `DLAB_0808` versus `Dlab0076`, `Dlab0133`,
`Dlab0805`). No benchmark sample is affected, and positional joining is never
used.

## DNNaic result

| panel | loci | model-free P2 projection P1->P3 | direction | expected | gate | direction RMS z | gate RMS z |
|---|---:|---:|---|---|---:|---:|---:|
| SINE standard | 1,012 | 0.09306 | B | C | 1.000 | 20.98 | 24.17 |
| VIGO standard | 1,012 | 0.01286 | B | contrast | 1.000 | 21.10 | 24.86 |
| SINE within-population polymorphism | 927 | 0.09660 | B | C | 1.000 | 21.63 | 23.94 |
| VIGO within-population polymorphism | 927 | 0.01583 | B | contrast | 1.000 | 21.68 | 24.32 |

The model-free projection preserves the author contrast: SINE is much farther
toward the Mediterranean reference than VIGO. Nevertheless, the released
direction head assigns both panels class B, including the class-C positive, and
the gate saturates at 1.0 for both. The result is unchanged by requiring both
alleles within PENI, SINE, VIGO, and MED. RMS shifts around 21-25 training
standard deviations make this an explicit severe-OOD transfer failure, not an
accuracy estimate or calibrated probability statement.

## Reproduction

The runner converts the 837-sample PED/MAP source to a nominally oriented VCF
without PLINK. REF/ALT are lexicographic allele labels, not reference-genome or
ancestral states. The converted VCF is 3,437,599 bytes with SHA-256
`6fc291d99cac1b2d389928caa6edbd328e15e199b36776259d9a117dc8bb4bf6`.

Run from a clean checkout:

```bash
python scripts/seabass_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --ped /path/to/seabass.ped \
  --map /path/to/seabass.map \
  --metadata /path/to/metadata.csv \
  --ancestry /path/to/ancestry.Q \
  --summary /path/to/mean_ancestry.csv \
  --cache-dir /path/to/derived \
  --result-dir results/seabass_external_benchmark_2026_07_11
```

The committed run used clean code commit
`b1737b1646726810addbb37eeb00ff09f67ee13d`. An independent detached-worktree
run was byte-for-byte equal after normalizing only repository/cache paths;
both canonical JSON objects hash to
`048e0fad431849721e1c85111abeaabb5595b888b972c0f63a5cd4d7cb839c7b`.
The committed `results.json` SHA-256 is
`9caa335201f683fbb8817aaffed58a84cce0460b36ce0db92fb70efd92c2b41b`.

This panel is purpose-ascertained: Atlantic fish were SNP-chip genotyped,
Mediterranean references came from WGS, Mediterranean-private variants were
excluded, and P3 pools western and eastern Mediterranean references. The
positive documents one Mediterranean-to-Atlantic component within a
bidirectional history. These qualifications are part of the result.
