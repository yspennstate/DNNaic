# Data sources

Every dataset the paper uses, with the exact public location. Large files are not committed;
the scripts fetch what they need (the 1000 Genomes and archaic scripts slice only the required
windows over HTTP range requests, so no full-genome download is needed).

## Simulated coalescent data (primary)

The 3,200-replicate rarefaction feature arrays with ground-truth replicate, direction, and rate
labels — and the two independent held-out batches used for the generalization test — are archived
at Zenodo:

- **DOI:** [10.5281/zenodo.21233067](https://doi.org/10.5281/zenodo.21233067)
- Contents: `regen_full/` (3,200 replicates), `regen_extra_round1/`, `regen_extra_round2/`
  (3,200 each, independent seeds and rate draws). Each directory holds `X.npy` (per-row 28-D
  features), `direction.npy`, `magnitude.npy`, and `groups.npy` (the replicate split unit that
  makes evaluation leakage-free).
- Regenerate from scratch instead: `python scripts/simulate_demography.py` (msprime), then
  extract the rarefaction features with PADZE (see [README.md](README.md)).

## Feature engine: PADZE

- Repository: <https://github.com/Andres42611/PADZE> — `pip install padze`.
- Computes allelic richness, private allelic richness, and pairwise-private richness by
  rarefaction directly from VCF. Validated against the reference ADZE implementation of Szpiech
  et al. (2008) on the HGDP microsatellite panel.

## Real data

### 1000 Genomes Project (phase 3, GRCh37)

- VCFs: `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/ALL.chr{N}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz`
- Sample panel: `https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel`
- Used for the feature-faithfulness check, the injected-signal validation, and as the modern
  populations (YRI, CEU, CHB, LWK) in the Neanderthal analysis.

### Heliconius (labelled adaptive introgression)

- Whole-genome `.geno` data from Simon Martin's ABBA-BABA tutorial:
  <https://github.com/simonhmartin/tutorials> (`ABBA_BABA_whole_genome`), from Martin et al. (2013),
  *Genome Research*.
- Species: *H. melpomene*, *H. timareta*, *H. cydno*, with *H. numata* as the outgroup.

### Neanderthal (labelled archaic introgression, GRCh37)

The two high-coverage Neanderthal genomes Altai and Vindija 33.19, pooled into one archaic
population, from the Max Planck EVA archive (contig naming `22` etc., matching 1000 Genomes
phase 3):

- Altai: `https://ftp.eva.mpg.de/neandertal/Vindija/VCF/Altai/chr{N}_mq25_mapab100.vcf.gz`
- Vindija 33.19: `https://ftp.eva.mpg.de/neandertal/Vindija/VCF/Vindija33.19/chr{N}_mq25_mapab100.vcf.gz`

Source publications: Prüfer et al. (2014, 2017).

### Mouse (labelled adaptive introgression, mm10)

A second labelled system, in a mammal. The warfarin-resistance haplotype at *Vkorc1* (chromosome 7)
introgressed from *M. spretus* into *M. m. domesticus* (Song et al. 2011). The trio maps to the
caterpillar tree as P1 = *M. m. musculus*, P2 = *M. m. domesticus* (recipient), P3 = *M. spretus*
(the divergent donor), so the documented spretus → domesticus flow is class C. Wild-mouse joint
genotypes from Harr et al. (2016), read by tabix over HTTP range requests:

- VCF: `https://wwwuser.gwdguser.de/~evolbio/evolgen/wildmouse/vcf/AllMouse.vcf_90_recalibrated_snps_raw_indels_reheader_PopSorted.PASS.vcf.gz`
- Populations by sample-name prefix: `Mmm_` (musculus), `Mmd_` (domesticus), `Ms_` (spretus).

Source publication: Harr et al. (2016), *Scientific Data*.

### Additional external OOD benchmarks (not paper validation)

Additional independently published systems are available through the external-benchmark scripts.
They are reported as exploratory simulation-to-data transfer diagnostics with positive and
negative/control expectations, source hashes, fixed sample manifests, feature-shift metrics,
and uncalibrated direction scores:

- Andean duck beta-globin positive and alpha-globin negative control: Dryad
  [10.5061/dryad.bnzs7h4b4](https://doi.org/10.5061/dryad.bnzs7h4b4), Graham et al.
  [10.1038/s41437-021-00437-6](https://doi.org/10.1038/s41437-021-00437-6).
- Scarlet runner bean genome-wide GBS VCF: [OSF h7sa5](https://osf.io/h7sa5/),
  Guerra-Garcia et al. [10.1002/evl3.285](https://doi.org/10.1002/evl3.285).
- Scrub-jay exact near-zero D control: Dryad
  [10.5061/dryad.8sf7m0cph](https://doi.org/10.5061/dryad.8sf7m0cph),
  DeRaad et al. [10.1093/sysbio/syac034](https://doi.org/10.1093/sysbio/syac034).
- Lake Malawi cichlid matched positive/negative panels: Zenodo
  [10.5281/zenodo.4134522](https://doi.org/10.5281/zenodo.4134522),
  Malinsky et al. [10.1038/s41559-018-0717-x](https://doi.org/10.1038/s41559-018-0717-x).
- Ciona direction-labelled contact-zone and site-control panels: Zenodo
  [10.5281/zenodo.5346932](https://doi.org/10.5281/zenodo.5346932),
  Le Moan et al. [10.1111/mec.16189](https://doi.org/10.1111/mec.16189).
- European seabass direction-labelled positive and reversed contrast: Zenodo
  [10.5281/zenodo.3989825](https://doi.org/10.5281/zenodo.3989825),
  Robinet et al. [10.1111/mec.15611](https://doi.org/10.1111/mec.15611).
- Tinkerbird heterogeneous majority-direction, admixture-enriched contact-zone stress test: Dryad
  [10.5061/dryad.jm63xsj87](https://doi.org/10.5061/dryad.jm63xsj87),
  Kirschel et al. [10.1111/mec.15691](https://doi.org/10.1111/mec.15691).

See `data/external_benchmarks/` for exact file hashes and author-derived sample manifests.
