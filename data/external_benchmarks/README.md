# External benchmark manifests

These fixed manifests support `scripts/external_benchmarks.py`. They add one small targeted
dataset and one larger genome-wide dataset as **exploratory out-of-distribution diagnostics**,
not as independent validation of a classifier trained only on the repository simulations.

## Andean ducks (small)

The Dryad release for Graham et al. contains 10 low-elevation yellow-billed pintails (P1),
10 high-elevation pintails (P2), and 10 high-elevation speckled teal (P3). The beta-globin
panel is a published positive (D=0.78, fD=0.68), with ST-high -> YBP-high inferred in a
separate IMa2 analysis; alpha-globin is a published negative control (D=-0.04, P=0.477).
The actual quartet is `((YBP-high,YBP-low),(ST-high,ST-low))`, so this is explicitly also a
topology/domain-shift test rather than a clean match to the pectinate training tree.

## Scarlet runner bean (large)

The OSF VCF contains 237 samples. Three author-metadata-derived manifests reproduce the
published Table S4 population definitions while following the canonical relatedness-filtered
sample set: Cult-SUR-CH / Cult-TMVB / Wild-TMVB-CDMX is the primary class-C benchmark
(D=0.109339, f4-ratio=0.421888, z=8.37171), and replacing the donor with Wild-TMVB-Tepoz is a
published replicate (D=0.107835, f4-ratio=0.156145, z=6.95236). A third manifest uses the
published null Cult-SMOCC / Cult-TMVB-Spain / Cult-TMVB contrast (D=0.0008093, P=0.96497).
The direction head has no no-event class, so the null is evaluated as a discordance/abstention
diagnostic, not by demanding a null label.

All three bean panels are selected from one six-population callable-site intersection that is
also polymorphic in every positive/null trio before the deterministic 15,000-locus cap. Their
locus identities and order are therefore identical; population or donor changes cannot silently
change the tested sites.

The released provenance has a one-sample inconsistency that matters for exact comparison:
the canonical 183-sample metadata includes `FrijCol_11` and excludes `FrijCol_10`, whereas
several Dsuite scenario maps use `FrijCol_10` while still labelling the group n=19. This
manifest follows the canonical included set (`FrijCol_11`). It must not be described as a
verbatim replay of those Table S4 scenario files.

Exact URLs, byte sizes, hashes, citations, and the author-metadata commit are in `sources.json`.
Large VCFs and derived filtered VCFs remain ignored under `data/real/`.

Example:

```bash
python scripts/external_benchmarks.py \
  --data-root /path/to/simulation_data \
  --duck-root /path/to/extracted/dryad \
  --runner-vcf /path/to/coccineus.recode.vcf
```

## Kenyan giraffe (additional positive)

The Zenodo bundle accompanying Coimbra et al. contains a 126-sample VCF, the authors'
population map and OrientAGraph tree, and exact Dsuite outputs. The committed manifest uses
the sister reticulated groups as P1/P2 and Nubian giraffe as P3. The supplied tree therefore
matches `((P1,P2),P3)`, and the exact author result is D=0.211001, Z=25.4689,
f4-ratio=0.199169. OrientAGraph and the paper's asymmetric ancestry analysis support
Nubian P3 -> Laikipia-reticulated P2, so the expected DNNaic class is C. D itself is not
treated as directional. This is an ancient natural event and remains out of distribution;
the paper did not find significant contemporary migration.

Under the standard external-data contract, 72,759 loci have at least 16 called copies per
population and are polymorphic across the complete trio; a fixed-seed reservoir retains
15,000. A deliberately stricter robustness panel additionally requires both alleles within
each population, leaving 10,588 loci. The stricter count is not the primary contract: it
removes fixed between-population differences that are valid rarefaction input. Both are
reported so the transfer conclusion cannot hinge on this filtering choice.

## Rhode Island brook trout (additional null controls)

Michaelides et al. release 444 fish at 16,336 SNPs, including two hatchery strains. The
study found no significant captive-to-wild introgression overall (with one UTA individual
showing hatchery ancestry). The two committed panels compare non-stocked AFP (P1), stocked
BAK (P2), and either LFA or LFR hatchery fish (P3). They are specificity stress tests: the
depth-matched gate should abstain, while the always-three-way direction score is reported
but has no gold-standard direction. They are not presented as site-specific D=0 benchmarks.

Run both additional datasets with:

```bash
python scripts/additional_external_benchmarks.py \
  --data-root /path/to/simulation_data \
  --giraffe-vcf /path/to/dsuite_introgression/snps.sampled.vcf \
  --brook-vcf /path/to/BT_ALL.vcf
```

## Scrub jay (exact sampled-trio null)

The pinned author VCF supplies 18 interior/western Woodhouse's scrub-jays (P1),
15 northern-Mexico Woodhouse's scrub-jays (P2), and 10 *A. sumichrasti* (P3).
The author Dsuite row for exactly `((iw,mw),s)` is D=0.00619049, Z=0.205384,
P=0.418636, so this is the cleanest site-specific gate control in the bundle.
It is not a claim of species-wide isolation; a narrow phenotypic contact zone is
known near Mexico City. Both the normal across-trio polymorphism filter and a
stricter within-each-population filter are reported.

## Lake Malawi cichlids (matched positive and negative)

One four-group callable-site intersection is used for both panels. P1 is 20
*A. calliptera*, P2 is eight mbuna, and the alternative P3 groups are nine
pelagic fishes (positive excess sharing) or ten deep-benthic fishes (negative
control). The standard shared contract preserves between-population fixed
differences; a stricter all-populations-polymorphic panel is also reported.
Frequency-based D with `Nbrichardi` as outgroup is recomputed on each exact
filter. The standard panel is positive for pelagics (D about 0.0855, Z about
4.20) and null for deep benthics (D about -0.0121, Z about -0.59). Under the
stricter filter the positive weakens to Z about 1.66, which is reported rather
than hidden. The source evidence does not orient donor and recipient, so no
Lake Malawi A/B/C score is counted as directional accuracy.

Run these further panels with:

```bash
python scripts/further_external_benchmarks.py \
  --data-root /path/to/simulation_data \
  --scrub-vcf /path/to/unzipped.filtered.vcf.gz \
  --malawi-vcf /path/to/Malinsky_et_al_2018_LakeMalawiCichlids_scaffold_0.vcf.gz
```

## Ciona (direction-labelled contact-zone stress test)

Le Moan et al. support recent introduced *C. robusta* -> native
*C. intestinalis* introgression in Southampton. The committed positive maps
Jersey to P1, Southampton to P2, and Atlantic *C. robusta* to P3, so the gold
orientation is class C. Poole replaces Southampton in the site-control panel;
Jersey and Poole are absent from both source positive-call methods. Pairwise
FST is closely matched (Jer-Sth=0.005, Jer-Poo=0.005, Sth-Poo=0.002), although
Jersey was sampled in 2014 and the other two in 2012.

The runner reports the purpose-ascertained whole VCF, the published Table-S4
Southampton interval (`chromosome5:661065-1174846`), and the broader Figure-3
window (`chromosome5:500000-2000000`). Positive and control panels share the
same ordered loci in every scope. Because linked ddRAD SNPs can overweight one
tag, both hotspot scopes also report a one-SNP-per-RAD-tag sensitivity (40 tags
in the exact interval and 79 in the broad window). A reference-invariant
four-population frequency contrast is recomputed; Patterson's D is not used
because the VCF contains no defensible outgroup or ancestral allele.

This is not unbiased external validation. The authors removed *C. robusta*-private
variants, mapped reads to a *C. robusta* reference, and selected the hotspot
because introgression was already detected. Run it with:

```bash
python scripts/directional_external_benchmarks.py \
  --data-root /path/to/simulation_data \
  --ciona-vcf /path/to/Ciona_data3_introgression_mac2.vcf.gz
```
