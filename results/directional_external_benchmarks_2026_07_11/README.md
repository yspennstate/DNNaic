# Direction-labelled external benchmark (2026-07-11)

This bundle evaluates the Ciona contact-zone system. The source study supports
recent introduced *C. robusta* (P3) -> Southampton *C. intestinalis* (P2),
which is DNNaic class C. Jersey is the fixed P1 reference. Poole replaces
Southampton in a matched site-control panel and is absent from both source
positive-call methods.

This is a mechanistic out-of-distribution stress test, not unbiased validation.
The released VCF is purpose-ascertained, reads were mapped to a *C. robusta*
reference, linked ddRAD SNPs share tags, and the chromosome-5 windows were
selected because introgression had already been detected.

## Learned-head results

| Scope | Loci | Southampton direction / gate | Poole direction / gate |
|---|---:|---|---|
| Whole VCF, standard shared contract | 15,000 of 27,418 | A / 1.0000 | A / 1.0000 |
| Whole VCF, within-pop polymorphic | 475 | A / 1.0000 | A / 1.0000 |
| Figure-3 chr5 0.5--2.0 Mb | 544 | A / 7.53e-5 | A / 2.86e-17 |
| Figure-3 window, one SNP per RAD tag | 79 | C / 8.47e-125 | C / 1.92e-114 |
| Exact Southampton Table-S4 interval | 262 | A / 1.65e-26 | C / 1.56e-55 |
| Exact interval, one SNP per RAD tag | 40 | A / 6.56e-298 | C / 1.61e-256 |

No scope produces a successful detect-then-orient result. Genome-wide, the gate
fires on both the class-C positive and the source-negative control while the
direction call is A. In the published windows the gate abstains on the
positive. One-per-tag thinning makes the broader-window direction call C, but
the gate still abstains by more than 100 orders of magnitude; the exact-interval
direction remains wrong.

All learned scores are severely outside simulation support. Direction-head RMS
z ranges from 21.5 to 169.1 and gate-head RMS z from 27.7 to 175.0. Softmax and
gate outputs are therefore uncalibrated scores, not probabilities or
posteriors.

## The biological contrast is present

A reference-invariant frequency contrast was recomputed on exactly the loci
given to each learned head. In the exact Southampton interval,
`mean[(pJer-pSth)*(pCioAB-pPoo)]` is -0.01282, and squared frequency distance to
`CioAB` is 0.21785 for Southampton versus 0.24356 for Jersey and 0.24467 for
Poole. After one SNP per RAD tag the ordering persists: 0.19235 versus 0.20899
and 0.21490. The whole-genome capped contrast is near zero (+0.00018), as
expected for a localized event.

Thus the failure is not absence of the labelled local sharing signal. The
simulation-trained representation/gate does not transfer reliably to this
purpose-ascertained linked-RAD domain. Patterson's D is not reported because
the released VCF has neither a defensible outgroup nor ancestral-allele labels.

## Reproduction

The source file, population manifests, filter counts, ordered-locus hashes, and
published interval bounds are runtime assertions. The result was generated
from clean code commit `fdcd012254a34f4be2cbf09edb32054642cd919f`:

```text
python scripts/directional_external_benchmarks.py \
  --data-root /path/to/simulation_data \
  --ciona-vcf /path/to/Ciona_data3_introgression_mac2.vcf.gz
```

The full suite passed 49 tests before the run. `results.json` is 454,777 bytes
with SHA-256
`4ccd2504f0ab065bdbe04331c41b4141489a1e2794c47364f93e16ab92b75573`.
An independent detached-worktree run at the same commit used a separate cache
and output directory. After normalizing only the absolute repository and cache
prefixes, the complete JSON objects were identical (canonical normalized
SHA-256 `7fdea1d2024782fb1f3aa94b0a51435febb83f180c564064508bfd9bfe80b749`).
