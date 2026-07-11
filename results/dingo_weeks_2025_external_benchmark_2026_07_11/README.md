# Weeks 2025 dingo--dog transfer stress test

The result was generated from hash-pinned Figshare release 1 bytes at clean
code commit `ae90b6d955a81a3cdb5785a6c23c3be7a8549117`. The panel is Alpine dingo
P1 (248), eight known captive `(dingo x dog) F1 x dingo` backcrosses P2, and
domestic dog P3 (39). The pedigree independently anchors a dog-introgressing
component whose candidate orientation is class C (`P3 -> P2`), but it is not
an exclusive population-level one-edge history or an accuracy datum.

| released/filter scope | loci | raw head | direction RMS-z | gate | gate RMS-z | P2 projection toward P3 (95% chromosome-block interval) |
|---|---:|---:|---:|---:|---:|---:|
| all released rows, standard | 2,193 | A | 18.64 | 1.0 | 21.63 | 0.249 (0.217--0.281) |
| all released rows, within-population polymorphic | 1,594 | A | 22.09 | 1.0 | 24.25 | 0.270 (0.236--0.306) |
| VCF PASS only, standard | 1,992 | A | 19.14 | 1.0 | 21.98 | 0.245 (0.213--0.278) |
| VCF PASS only, within-population polymorphic | 1,551 | A | 22.31 | 1.0 | 24.40 | 0.268 (0.235--0.306) |

All four correlated sensitivities exceed the prespecified RMS-z > 10 severe-OOD
diagnostic and therefore abstain. Class A is a raw extrapolative output, not an
accepted call. The model-free frequency projection is numerically close to the
nominal 0.25 dog component under every filter, but is descriptive geometry—not a
bounded ancestry estimate, migration rate, or temporal direction estimate.

The source audit also records that the analysis metadata swap the paper's
Desert/Mallee counts, the geolocation file has a third conflicting split and
three unmatched IDs, and 233 of the paper's 2,466 released VCF rows retain
non-PASS tranche tags. These groups are excluded; both defensible VCF row scopes
are reported. Because P2 has exactly eight diploids, `g=16` additionally makes
every analyzed locus complete-case in P2.

A detached clean checkout with CRLF working-tree conversion produced the same
canonical semantic result SHA-256
`cefd4cab47ec78d8fc0eedecae713c8f26ce71f3248bab1e5c2ba6fc10741f57` as the
main LF checkout after removing absolute paths, runtime argv, and the recorded
working-tree line-ending representation. This also exercises the canonical-LF
source-manifest contract.
