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

## European seabass (directional positive and reversed contrast)

Robinet et al. report a localized Mediterranean-ancestry excess in Atlantic
seabass from SINE. The fixed design uses Atlantic PENI as P1, either SINE
(positive) or VIGO (reversed contrast) as P2, and ten Mediterranean references
as P3. The source interpretation supports MED P3 -> SINE P2, DNNaic class C.
Raw author ancestry estimates are 0.111878 for SINE, 0.044890 for PENI, and
0.036501 for VIGO; six SINE fish but no VIGO fish exceed 0.15 Mediterranean
ancestry. VIGO is therefore a useful same-source contrast, not a pristine
no-flow null.

The source is a 1,012-marker SNP-chip/WGS merge. The committed converter pins a
pure-Python PED/MAP-to-VCF contract and labels its lexicographic REF/ALT coding
as nominal rather than reference-genome orientation. Both panels use one shared
PENI/SINE/VIGO/MED callable-site intersection. A robustness pass additionally
requires both alleles inside every one of the four populations. The authors
excluded Mediterranean-private variants and typed Atlantic and Mediterranean
fish on different platforms, so this remains an ascertainment-heavy OOD stress
test rather than an unbiased accuracy estimate.

The released PED and ancestry/metadata bundles also differ by three Atlantic
fish outside the benchmark groups. The runner records the discrepancy and
joins ancestry by explicit sample ID; positional alignment is forbidden. Run:

```bash
python scripts/seabass_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --ped /path/to/seabass.ped \
  --map /path/to/seabass.map \
  --metadata /path/to/metadata.csv \
  --ancestry /path/to/ancestry.Q \
  --summary /path/to/mean_ancestry.csv
```

## Tinkerbird (majority-direction, admixture-enriched stress test)

Kirschel et al. infer asymmetric backcrossing between yellow-fronted
*P. c. extoni* and red-fronted *P. p. pusillus* from autosomal,
Z-chromosome, and mitochondrial ancestry. With legacy source-era extoni and
pusillus reference pools as P1/P3 and their 14 admixed offspring as P2, the
majority cross maps to candidate class C: P3 ancestry into P2. This is
a heterogeneous majority label (9 candidate C, 3 reciprocal, 2 equal), not
exact per-panel truth. The temporal interpretation comes from the published
autosome/Z/mtDNA cross analysis, not from the frequency comparator.

The rendered Table 2 names only 13 of the 14 birds. The Discussion separately
identifies AR93163 as a pusillus-haplotype female with an extoni father, so the
manifest records the 13 table rows plus that explicitly named fourteenth bird.
It also includes all 17 source-era pusillus reference samples present in the
VCF; six valid references omitted by the prototype are restored. Current
metadata do not preserve exact allopatric labels for every legacy bird:
Sebastianelli et al. (2024) classify AR93110 and AR93178 as sympatric. The
legacy pools are therefore not presented as exact current allopatric sets.

P2 was selected for intermediate ancestry from this same ddRAD dataset, and
its direction label reuses ancestry from the same birds. P2 is also a hybrid
pool, not a panmictic tree leaf. The result is therefore explicitly circular
and excluded from accuracy. The source VCF encodes female Z genotypes as
diploid, including heterozygotes, so the runner scores anchored autosomes only.
It reports both the ordinary linked-SNP panel and a first structurally eligible
source SNP per scaffold sensitivity, with scaffold-blocked projection and f3
uncertainty.

Run:

```bash
python scripts/tinkerbird_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --source-vcf /path/to/revision.recode.vcf.gz
```

## Tinkerbird 2024 (sample-disjoint holdout and controls)

Sebastianelli et al. (2024) expand the system to a 452-bird Stacks ddRAD VCF
and infer strongly asymmetric mating/backcrossing in 95 sympatric females. The
main text reports z=6.949 while Supplementary Table S14 reports estimate 0.582,
SE 0.086, z=6.714, P<0.001; both source values are retained. Current evidence
supports asymmetry, not a strictly unidirectional edge: Rancilhac et al. (2025)
find bidirectional introgression tails.

The primary P2 contains nine Mpofu chrysoconus males selected by sex, locality,
and author taxon label and is paired with exact author `ref_extoni` (n=23) and
`ref_pusillus` (n=8) sets. All 40 P1/P2/P3 holdout samples are disjoint from
the 95 females used to infer the direction statistic. Candidate C remains a
weak system-level majority label, so even this holdout is excluded from
accuracy. A direct 14-daughter panel reproduces the author label but is
explicitly circular. Two gate-only contrasts use geographically separated
extoni references and daughters whose inferred parents are both near-pure
pusillus; neither is a proven historical zero-flow null.

Only `SUPER_1` through `SUPER_44` are scored. Z, W, S76/non-numbered, and
unplaced sequence is excluded. One source-ordered SNP per Stacks RAD locus is
the primary scope; it removes within-tag pseudoreplication but not
chromosome-scale linkage. The all-SNP scope is explicitly a linked sensitivity
analysis. Cross-family gate contrasts use separately ascertained loci and are
qualitative only. Run:

```bash
python scripts/tinkerbird_2024_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --source-vcf /path/to/southern_africa_biallelic_snps_minDP4_MaxMiss20_MAF5.vcf.gz \
  --female-metadata /path/to/MS_SouthernAfrica_ddRADS_95SympF_14Mar23.xlsx \
  --master-metadata /path/to/MS_Tinkerbird138_Master_06Oct23.xlsx \
  --supplement /path/to/41467_2024_47305_MOESM1_ESM.pdf
```

## Corkwing wrasse (candidate-direction sensitivity and comparators)

Faust et al. (2018) document transport of southern corkwing wrasse into
Flatanger and report escapees plus hybrid descendants. Three balanced panels
use Austevoll (P1), all 40 Flatanger fish (P2), and separate Kristiansand,
Stromstad, or Kungsbacka donor references (P3), giving candidate class C. The
same donors are repeated on identical loci in descriptive role-changed
contrasts with Stavanger (P1) and Austevoll (P2). They are not matched controls
or no-flow populations; Stavanger itself contained reported hybrids.

The primary source removes both 15 author HWE exclusions and all 200 loci used
to generate the original NewHybrids labels before any benchmark filtering.
The all-released-locus scope retains them only as circular sensitivity. All
240 exact VCF IDs are reconstructed mechanically from the pinned 240-row
metadata, including 48 documented technical-replicate suffixes. Every panel
uses all 40 fish per source population, avoiding hybrid-only cherry-picking.

Each `CHROM` value is a unique 2bRAD tag and `POS` is only the within-tag
position. There is no physical chromosome/linkage map, so uncertainty is a
naive IID-locus sensitivity, never chromosome-block uncertainty. The recent
anthropogenic southern-to-Flatanger event coexists with ongoing bidirectional
contact whose older/background component is asymmetric west-to-south
(Mattingsdal et al. 2020). A larger 2021 survey found six high-probability
southern-origin Flatanger fish and 70 potential hybrids there, concentrated at
the northern edge and reaching about 20% locally. A 2026 mesocosm study found
strong selective winter mortality in hybrid offspring and suggested potential
assortative mating. Raw class A can therefore reflect dominant western or
range-expansion background in a mixed-history cohort; the one-edge head cannot
isolate superimposed class C. No panel is accuracy-eligible. Run:

```bash
python scripts/wrasse_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --source-vcf /path/to/west.filt.maf0.01.recode.vcf \
  --metadata /path/to/Sampleinfo_metadata.txt \
  --hwe-genepop /path/to/west_genepop4357ID.txt \
  --newhybrid /path/to/newhybrid200SNPs.dat
```

## Harpagifer (literature-dominant same-SNP sensitivity)

Segovia et al. (2022) report a dominant BayesAss estimate from a northern
Patagonia SNP cluster into a southern Patagonia cluster (18.8%, versus 2.03%
reciprocal). With those clusters as P1/P2 and the Falklands/Malvinas M1 cluster
as P3, this maps to candidate class A. It is not ground truth: the groups and
label reuse the same 2,993 putatively neutral GBS SNPs, other migration edges
were reported, and BayesAss used only 10,000 iterations with 10% burn-in. The
paper also treats the nominal taxa as one evolutionary unit, so P3 is a
population cluster rather than a species outgroup.

No author individual crosswalk accompanies the VCF. The committed block record
reconstructs only the VCF-positive site blocks from released column order and
published contiguous site order: TEM--IC3 become P1 (50), PY--PW P2 (43), and
Hookers Point P3 (25). Supplementary Table S1 totals 117 and reports TEM n=2,
whereas the VCF has 118 columns and three leading TEM-like samples; main-text
Table 3 totals 128. The block record is not a complete collection-site roster.

Four runs are two sample scopes (all 118 and a prespecified <=25% individual
missingness sensitivity) crossed with standard and within-population
polymorphism filters. They are four analytic sensitivities of one comparison,
not four validations. The missingness scope strongly reweights P1 (50 to 40)
while retaining 42/43 P2 and all 25 P3. Standard and strict scopes retain 2,993
and 2,977 exact shared loci, respectively.

All source `CHROM` fields are the artificial value `0`; physical linkage is
unknown. Uncertainty is therefore a naive IID-locus sensitivity only. The
prespecified diagnostic-frequency threshold yields zero loci and remains null,
without post-hoc threshold tuning. The Dryad ZIP wrapper is also regenerated
with variable timestamps, so the runner downloads and pins the stable inner
VCF endpoint rather than a ZIP digest. Run:

```bash
python scripts/harpagifer_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --source-vcf /path/to/Hbi_Hpal_118_2993_6May19_GEO.vcf
```

## 2024 H. antarcticus (independent biophysical direction stress test)

Bernal-Duran et al. (2024; DOI 10.1111/mec.17360) released 143 fish, 20,778
source-defined neutral SNPs, and raw ROMS particle matrices. Two topology-
compatible directions are frozen before DNNaic scoring: DOI (P1) to FHA (P2),
with HOS as P3, and the weaker AIS (P1) to HOS (P2), with DOI as P3. The
released archive yields four-season means of 2.30 versus 0.25 percentage points
of source releases for DOI/FHA and 1.25 versus 0.325 for AIS/HOS. Each forward
edge exceeds its reciprocal in all four simulated seasons.

As of 2026-07-11 this remains the latest peer-reviewed species-specific study
combining population genomics with biophysical connectivity. No correction,
retraction, or independent directional reanalysis was found. A newer 2026
phased-assembly preprint is a genomic resource, but does not reanalyse these
populations or their connectivity. A later cross-taxon WAP study describes the
cited structure as subtle/high-gene-flow context; it likewise does not validate
any pairwise direction.

The ROMS matrices are numerically independent of the SNPs, but they represent
potential passive-larval settlement in 2008--2012, not realized historical
gene flow or introgression. The two comparisons share one oceanographic system
and only test candidate class A. They are correlated qualitative OOD stress
tests, not accuracy trials; there is no historical zero-flow null.

The runner audits several source defects explicitly. The archive stores the
old ten-site order with GRE/HOS appended, whereas the paper figure displays a
different permutation. The README says HGE although the VCF uses HGR for Green
Reef, and says HAR=Alexander although same-project GBIF occurrences place the
HAR samples around Adelaide/Rothera, matching the paper/model AIS group. The
VCF also contains ten HAC Fildes samples omitted from the paper's 133-sample
Table 1 totals.

Every current-paper Figure 3 cell equals the ten-run mean day-100 dispersal
count. With 100 particles released per origin/run, those values are percentage
points of source releases. The prose reads like destination-conditional
normalization, so the runner records that alternative only as a semantic
sensitivity. It never substitutes the different 2023 thesis values for the
current 2024 result.

All VCF loci have artificial `CHROM=0`, and the Tassel header states that the
unknown reference was encoded as the global major allele. Physical linkage is
unavailable, REF is not ancestral, and uncertainty is a naive IID-locus
sensitivity only. Both standard and within-population-polymorphic scopes are
run; all fish are already below 25% missingness, so a sample-missingness view
would be identical. Run:

```bash
python scripts/hantarcticus_2024_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --source-vcf /path/to/Hantarcticus_20778_143_neutralSNPs.vcf \
  --matrices-zip /path/to/biophysical_matrices_Hantarcticus.zip \
  --source-readme /path/to/README.md
```

## 2017 Sydney rock oyster (same-release candidate-null crossing-sensitivity stress test)

Thompson et al. (2017; DOI 10.3354/meps12109) reported strong partitioning
between wild and selectively bred B2 Sydney rock oysters and no detectable
sustained introgression at two Georges River sites. This is not a gold zero-
flow label. The conclusion, clusters, and Q-site exclusions reuse the released
SNPs; only 9--12 individuals remain per cohort; and low gene flow, occasional
hybridization, or gene flow not producing sustained introgression could have
escaped detection.

The primary W comparison is WWC (P1), WOC rack overcatch (P2), and WB2 selected
stock (P3). QWC/QOC/QB2 is secondary because three B2-labelled Q oysters
(original IDs 28, 29, and 31) were removed after same-SNP DAPC classified them
as wild-like. Class C (B2 P3 to overcatch P2) is only the exposure orientation
if an event exists. Neither direction nor gate truth is available.

Both sites use one exact shared locus intersection. The standard view retains
1,101 loci; the within-population-polymorphic sensitivity retains 589. These
are two correlated sites crossed with two ascertainment filters, not four
validation trials. W/Q share sampling date, B2 stock/pedigree, DArT discovery,
anonymous loci, and source inference. The 21 Port Stephens hatchery-reference
oysters are audited but excluded because two cohorts do not supply a defensible
three-population topology.

The shared minimum of 16 called copies is deliberately explicit but unequal in
missing-genotype tolerance: QBB2 (n=9) can miss at most one diploid genotype per
locus, while each n=12 cohort can miss four. Because both sites use the same
intersection, even the primary W panel is conditioned on Q missingness.

The Dryad workbook contains 90 oysters and 1,200 generic paired-code loci. It
does not identify the paper's 1,189 neutral versus 11 consensus selection-
outlier loci, nor provide bases, chromosomes, coordinates, or a linkage map.
The runner therefore uses explicit synthetic A/C coding on CHROM 0 and treats
IID-locus resampling only as a fixed-sample sensitivity. It also records that
the final workbook has 9 individuals and 387 loci below the paper's stated 95%
call-rate threshold.

A raw gate below 0.5 is qualitatively concordant with the study's failure to
detect sustained introgression; a crossing is qualitatively in tension.
Neither is a true negative or false positive because the source is same-data,
power-limited, and does not establish migration rate zero. Severe-OOD panels
abstain regardless of the raw class or gate score. Run:

```bash
python scripts/oyster_2017_external_benchmark.py \
  --data-root /path/to/simulation_data \
  --source-xlsx /path/to/SNP_data_M12109.xlsx
```
