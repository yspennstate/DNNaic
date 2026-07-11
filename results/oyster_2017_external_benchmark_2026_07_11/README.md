# Guarded 2017 Sydney rock oyster crossing-sensitivity stress test

This bundle scores the two Georges River comparisons released by Thompson et
al. (2017; DOI `10.3354/meps12109`) against the simulation-trained DNNaic
direction and appreciable-introgression heads. The source study reported strong
wild-versus-B2 partitioning and no detected sustained introgression from the
selectively bred B2 line. That finding is a same-release, detection-limited
candidate null, not proof of zero migration and not a specificity label.

The workbook, source conclusion, clustering, and Q-site exclusions are not
independent. The Q B2 cohort contains only nine retained oysters: original IDs
28, 29, and 31 were omitted after the paper's DAPC on its 1,189 neutral loci
assigned them to the wild/control cluster. The public 1,200-locus workbook
derives from the same SNP-discovery data but already omits those oysters and
does not identify the exact neutral subset. Only 9--12 oysters remain per
cohort, so low gene flow, occasional hybridization, or gene flow that did not
produce sustained introgression could escape detection. Accuracy and
specificity estimates are therefore null, with zero independent validation
panels.

## Frozen source and conversion

Dryad DOI `10.5061/dryad.32q80`, sole version ID `3411`, supplies one CC0
workbook. The canonical inner file is 729,706 bytes, MD5
`572a079597af8530b15aaffd07325b55`, and SHA-256
`e0f6983f1a15c9d7a1aeb4a76e220f24b1d4c766600502413b2cb5c4fdde8029`.
Dryad generates the version ZIP; acquisition route and wrapper presence, bytes,
timestamps, size, and digest are deliberately excluded from result identity.

The `SNP_data_M12109` worksheet is exactly `A1:CNJ93`: 90 oysters, 1,200
paired-code diploid loci, and 108,000 genotypes. It contains 3,647 missing
pairs (overall call rate 96.623%), no partial missing pair, and no reversed
`2/1` heterozygote. Original numeric IDs 28, 29, and 31 are absent. Nine
individuals and 387 loci fall below the paper's stated 95% call-rate threshold;
all individuals nevertheless remain above 94%.

The workbook exposes only generic `Locus1`--`Locus1200` labels. It does not say
which 11 were consensus selection outliers, so the paper's 1,189-neutral-locus
analysis cannot be reconstructed. Nor does it supply bases, chromosomes,
physical positions, or a linkage map. The deterministic conversion maps
GenAlEx code 1 to synthetic allele A, code 2 to C, and `0/0` to missing; it uses
synthetic `CHROM=0` and workbook-order `POS`. The derived source VCF is 470,039
bytes with SHA-256
`7d978cb745008e880a023f4c6347c54d50abd9c19cfb5daeba1f964fc829d756`.
No allele or coordinate is interpreted as ancestral, derived, or physical.

## Panels and shared filters

The primary W panel is WWC (P1), WOC rack overcatch (P2), and WB2 selected
stock (P3), with 12 oysters each. The secondary, outcome-conditioned Q panel is
QWC (P1), QOC (P2), and QB2 (P3), with counts 12/12/9. Class C (B2 P3 to
overcatch P2) is only the exposure orientation if an event exists; no positive
direction truth exists. The operational `((WC,OC),B2)` order is not a rooted
species tree. B2 is a sixth-generation selected/founder stock, so pedigree,
selection, drift, and heterozygosity differences are material confounders.

Both sites use the exact same ordered locus intersection within a filter:

| Filter | Retained | Called-copy exclusions | Polymorphism exclusions | Ordered-locus SHA-256 |
|---|---:|---:|---:|---|
| Standard, polymorphic across each complete site panel | 1,101 | 82 | 17 | `edafecad96e334e40dfff485bb99ba6e1354a0d7841af7257851673c286a75ed` |
| Polymorphic within all six cohorts | 589 | 82 | 529 | `df6659235f9030e4ecec84c36be610239e0912d8986daf2f053144892d71ec26` |

The shared minimum of 16 called copies is unequal in missing-genotype
tolerance: QBB2 (n=9) can miss at most one diploid genotype per locus, while an
n=12 cohort can miss four. Because the intersection is shared, the primary W
panel is also conditioned on Q missingness. Standard and strict scopes are
ascertainment sensitivities, not reliability tiers or additional trials.

## DNNaic result

| Panel | Loci | Raw head | Raw gate | Direction RMS-z | Gate RMS-z | All-locus projection |
|---|---:|---:|---:|---:|---:|---:|
| W, standard | 1,101 | A | 1.0 | 21.50 | 24.78 | 0.1873 |
| Q, standard | 1,101 | A | 1.0 | 21.10 | 24.41 | 0.1971 |
| W, within-pop polymorphic | 589 | A | 1.0 | 24.24 | 25.09 | 0.2164 |
| Q, within-pop polymorphic | 589 | B | 1.0 | 25.66 | 25.39 | 0.2688 |

All four raw gate crossings are qualitatively in tension with the source
failure to detect sustained introgression, and none of the forced direction
heads returns counterfactual class C. These are not accepted biological calls.
The raw values are classifier outputs on severe-OOD natural inputs, not
probabilities, migration estimates, confidence values, or OOD-detector scores.
Every panel greatly exceeds the prespecified severe-OOD rule
`max(direction RMS-z, gate RMS-z) > 10`; all four therefore adjudicate to
`abstain_severe_OOD`. Calling the gate crossings false positives, or the
direction outputs incorrect, would require truth labels this source does not
provide.

The fixed `|p3-p1| >= 0.95` diagnostic threshold yields zero loci in every
panel (maximum observed differences 0.6111--0.75), so diagnostic projection is
null without lowering the threshold after seeing the data. All-locus
projections range from 0.1873 to 0.2688; they are frequency geometry, not
ancestry proportions, migration rates, or temporal direction.

Finite-called-copy-corrected f3-like values are positive (0.00126--0.00316).
The descriptive IID-locus intervals cross zero for both W views and remain
positive for both Q views. The subtraction assumes independent binomial
called-copy sampling and is not generally unbiased. Positive f3 does not prove
a candidate null; negative f3 would not supply direction. With no linkage map,
IID resampling is only fixed-sample conditional sensitivity, never chromosome-
block uncertainty.

## Literature freshness and scope

No later independent reanalysis of the exact 90-by-1,200 release was located
through 2026-07-11. O'Hare et al. (2021; DOI
`10.1007/s10592-021-01343-4`) analyze a different 363-wild-oyster,
3,400-neutral-SNP question with no B2 or overcatch cohort, and share the lead
researcher and coauthors with the 2017 study. Their broad connectivity, high
effective population size, and no-recent-bottleneck result argues against
generalizing the Georges River inference to compromised species-wide wild-
population resilience; it does not re-test or overturn the local wild-versus-
B2 contrast. Bishop et al. (2023; DOI `10.3389/fmars.2023.1162487`) is a review
that cites Thompson among studies reporting little evidence of aquaculture-line
introgression, without genotype reanalysis or an independent truth label.

## Reproducibility

The tracked result was generated from clean commit
`cd9097b62c7d1d3b38b850250d7ed351d06af6d6`, with `OMP_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` fixed to
`1`. Its raw `results.json` is 180,323 bytes with SHA-256
`ae5e68fb9541f2239e96edd37af288760235334b8b7b0f68171663453e0927ff`.

A detached clean worktree at the same commit, using a separate empty derived
cache, produced a 180,379-byte JSON with SHA-256
`6540f188f1c6bd3e13b56b2c5d6ee9014c1fbc37e8741dc913ec3759bdb4be60`.
After normalizing only repository, cache, and result path prefixes (including
the main run's relative result argument), the parsed JSON objects are exactly
equal. Their normalized compact representation plus a final LF is 121,105 bytes with SHA-256
`57a3e27b44f8768d1872b852f31660c2f97447ec5d151a47cc0d34ebcc6374ae`.
Both runs record `dirty_at_run=false`.
