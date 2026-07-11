# Guarded 2024 H. antarcticus transfer stress test

This bundle scores two literature-frozen candidate class-A directions from
Bernal-Duran et al. (2024; DOI `10.1111/mec.17360`) against the simulation-
trained DNNaic heads. It is a qualitative out-of-distribution stress test, not
an accuracy benchmark. The ROMS matrices measure potential passive-larval
settlement during 2008--2012, not realized historical introgression. The two
comparisons come from one oceanographic study/system, share populations, and
have no historical no-flow control. Accordingly, `accuracy_estimate` is null
and the bundle contains zero independent validation panels.

## Frozen source and reconstruction

Dryad DOI `10.5061/dryad.b5mkkwhjk`, public record version ID `292109`, was
accessed through its byte-identical Zenodo mirror. The pinned files are:

- neutral VCF: 12,581,530 bytes, SHA-256
  `48d832ade62ef3ad21ced7869e6f2a9e5c418593978e6260725be0ba02f998a5`;
- particle matrices ZIP: 3,705,582 bytes, SHA-256
  `3ac56229b68ff9c77de9517015e52dfa766bc3e5590cd4b5e502e8a6aefb3456`;
- source README: 6,358 bytes, SHA-256
  `55464f867352d1f99db2fcabdd43e48efb524525667400396d99815cf8068bdc`.

The VCF contains 143 unique samples, 20,778 sorted biallelic SNPs, and 344,218
missing calls among 2,971,254 genotype cells (11.585%). Every sample is below
25% missingness, so a <=25% sample sensitivity would be identical. All loci
use artificial `CHROM=0`; physical linkage and chromosome-block uncertainty
are unavailable. The Tassel header says the unknown reference was encoded as
the global major allele, so REF is not ancestral.

Three release inconsistencies are preserved rather than silently guessed:

- the README says `HGE` for Green Reef, while the VCF uses `HGR`;
- it says `HAR=Alexander`, while matching same-project GBIF occurrences place
  those 20 samples around Adelaide/Rothera, agreeing with paper/model `AIS`;
- the VCF includes ten `HAC` Fildes samples omitted from the paper's
  133-sample Table 1 total.

The ZIP has 4,000 daily connectivity matrices: four seasons x ten release runs
x 100 days, plus four mean-trajectory matrices. The archive stores origins in
the old-ten-plus-GRE/HOS order and destination rows in reverse. Exact
current-paper Figure 3 cell matches establish the different display
permutation. The runner selects each run's chronologically last matrix,
averages ten day-100 dispersal matrices per season, and then averages seasons.
The 40-file inventory and summed day-100 matrix are pinned by semantic hashes.

Every published Figure 3 cell equals the ten-run mean day-100 count. Because
100 particles were released per source per run, these values are percentage
points of source releases. The prose reads like destination-conditional
normalization; that alternative is retained only as a semantic sensitivity.
Undefined conditional shares remain null and are never coerced to zero.

## Candidate directions

| Trio | ROMS source-release result | Run support | Role |
|---|---:|---:|---|
| DOI (P1) -> FHA (P2), HOS (P3) | 2.30% vs 0.25% reciprocal | 92 vs 10 settlers; 32/40 vs 7/40 positive runs | primary |
| AIS (P1) -> HOS (P2), DOI (P3) | 1.25% vs 0.325% reciprocal | 50 vs 13 settlers; 19/40 vs 7/40 positive runs | weaker secondary sensitivity |

Under the published source-release representation, each forward edge exceeds
its reciprocal in all four seasons. The destination-conditional sensitivity is
defined and forward-greater in 4/4 DOI/FHA seasons and 3/3 comparable AIS/HOS
seasons. In AIS/HOS `Eday_1780`, the reciprocal destination receives no
particles from any source, so its conditional share is undefined rather than
zero. The particle paths were generated independently of genotype values, but
not independently of the study design or geography; they are physical
orientation candidates, not genomic direction truth.

DOI/FHA and AIS/HOS are within the paper's central and southern regional
groups. The source genomic analyses do not establish direction for these site
pairs. ROMS therefore supplies a same-study physical orientation, not a second
pairwise genomic truth channel.

## DNNaic result

| Panel | Loci | Raw head | Gate score | Direction RMS-z | Gate RMS-z | Model-free projection |
|---|---:|---:|---:|---:|---:|---:|
| DOI/FHA/HOS, standard | 16,299 | A | 1.0 | 18.56 | 17.62 | 0.4802 |
| DOI/FHA/HOS, within-pop polymorphic | 12,074 | A | 1.0 | 21.24 | 21.91 | 0.4789 |
| AIS/HOS/DOI, standard | 16,301 | A | 1.0 | 18.34 | 17.39 | 0.3483 |
| AIS/HOS/DOI, within-pop polymorphic | 11,931 | A | 1.0 | 21.15 | 21.82 | 0.3643 |

All four raw direction heads match candidate A and every raw gate score is
1.0. They are not accepted calls: the head and gate values are uncalibrated
extrapolations on OOD data, while RMS-z is the heuristic OOD diagnostic. Every
panel crosses the prespecified severe-OOD rule (`max(direction RMS-z, gate
RMS-z) > 10`) by a wide margin, so all four adjudications are
`abstain_severe_OOD`. Gate 1.0 is classifier saturation, not confidence or
migration evidence. Indeed, within-population filtering makes the raw A scores
more extreme while the gate remains saturated at 1.0, and increases RMS-z to
about 21--22.

The fixed `|p3-p1| >= 0.95` diagnostic-locus threshold yields zero loci in all
panels (maximum observed differences 0.7153 and 0.6076), so the diagnostic
projection is unavailable. This does not mean zero differentiation or zero
gene flow, and the threshold was not lowered post hoc. All four finite-called-
copy-corrected f3-like estimates are positive, with descriptive IID-locus
bootstrap intervals above zero. Positive f3 is non-diagnostic: it neither
proves no admixture nor contradicts the ROMS orientation. Because linkage is
unknown, the IID intervals condition on these fixed samples and loci. The all-
locus projections are allele-frequency geometry, not ancestry proportions or
temporal-direction evidence.

Literature freshness was checked through 2026-07-11. This remains the latest
located peer-reviewed species-specific population-genomic plus biophysical-
connectivity study, with no indexed correction, retraction, or independent
directional reanalysis. A newer 2026 phased-assembly preprint does not revisit
these populations. Later cross-taxon WAP work describes the cited structure as
subtle/high-gene-flow context and does not validate a pairwise direction.

## Reproducibility

The tracked result was generated from clean commit
`c20b987778afcc68da24a9136164dfccc2e066a9` with `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` fixed to
`1`. Its raw `results.json` is 198,806 bytes with SHA-256
`0c086fa425450623358182c4eff190ddd0f219c1defe511355d90bfebc054151`.

A detached clean worktree at the same commit, with an empty independent cache,
produced a 198,835-byte JSON (SHA-256
`e2180d91ebe86d86cd44b709403a145cc758ac45fc236ae05a35d5f0e50f3b8d`).
After normalizing only repository, cache, and result path prefixes, the two
parsed JSON objects are exactly equal. Both runs record `dirty_at_run=false`.
