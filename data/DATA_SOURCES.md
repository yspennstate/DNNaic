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
