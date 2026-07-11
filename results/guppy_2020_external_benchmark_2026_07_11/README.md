# Fitzpatrick 2020 guppy genetic-rescue transfer stress test

This result was generated from six hash- and Git-blob-pinned files at clean
code commit `48254193bb0ffbde21a5b1f72e9fc388542d7c23`. The two ecological
recipient units are Caigual (NCA pre-flow P1, PCA post-flow P2) and Taylor (NTY
pre-flow P1, PTY post-flow P2), with the same SGS mainstem source proxy P3. The
2009 translocations and waterfall barriers establish downstream gene flow
independently of these SNPs, making class C (`P3 -> P2`) the candidate
orientation. SGS is a proxy and P1/P2 are serial samples, so neither drainage
is an exclusive simulation-tree truth or a formal accuracy datum.

| correlated sensitivity | loci | raw head | direction RMS-z | gate | gate RMS-z | P2 projection toward P3 (95% chromosome-block interval) |
|---|---:|---:|---:|---:|---:|---:|
| Caigual, standard | 6,877 | C | 30.95 | 1.0 | 36.84 | 0.875 (0.854--0.892) |
| Caigual, within-population polymorphic | 30 | B | 151.40 | 1.0 | 156.03 | 0.937 (0.845--0.991) |
| Taylor, standard | 6,696 | C | 30.86 | 1.0 | 35.93 | 0.765 (0.743--0.785) |
| Taylor, within-population polymorphic | 22 | B | 159.71 | 1.0 | 164.26 | 0.706 (0.513--1.184) |

All four rows exceed the prespecified RMS-z > 10 severe-OOD diagnostic and
therefore abstain. The standard filters produce raw candidate-C concordance,
whereas the 30- and 22-locus strict sensitivities flip to B under extreme
feature shift. These are two recipient units and four correlated analytic
views, not a 2/2 or 2/4 accuracy calculation. The gate saturates at 1.0 in all
four rows and is not an OOD detector.

The projection and plug-in f3 summaries are descriptive sample-frequency
geometry. Their chromosome-block bootstrap resamples 23 chromosomes for the
standard panels and only 16/15 represented chromosomes for the strict panels;
it does not resample fish or establish temporal direction. The strict Taylor
projection interval exceeding 1 illustrates why projection is not a bounded
ancestry estimate.

The author release and both benchmark filters use post-flow populations during
locus selection. Experimental direction is SNP-independent, but locus
inclusion is not prospective held-out. The source audit also records that the
paper reports 12,407 SNPs while each of the five runtime-verified VCFs contains
11,417; the released formatting script wrongly comments that NCA is post-flow;
and the data repository has no explicit license. Source bytes are therefore
runtime-only and are not vendored.

A detached clean CRLF checkout reproduced every scientific value. The only
raw-JSON differences were the result-directory argument, repository path, and
recorded source-manifest line endings. After normalizing those environmental
fields, both runs had canonical semantic SHA-256
`e2e14f5cd85a6d74c3067c2ca4e61db46a7314232307d762f274bf3eb1063b0d`.
