# Brook-trout microsatellite transfer diagnostic

This bundle applies the frozen canonical DNNaic heads to two deposited
multiallelic microsatellite studies: White et al. (2018) in Pennsylvania and
Lehnert et al. (2020) in Nova Scotia. The primary representation calls the
management-candidate class C (`P3 -> P2`) in 9/11 descriptive primary views:
3/3 Pennsylvania targets and 6/8 recorded-stocking Nova Scotia river systems.

**That 9/11 is not accuracy.** Every call is unaccepted and every formal
direction/gate accuracy field is false. Published ancestry labels were inferred
from the same markers scored here; stocking history establishes management
exposure, not realized ancestry or the exclusive three-population topology.
All 28 primary and sensitivity records also trip the frozen severe-OOD heuristic.

## Data and panel accounting

- Pennsylvania: 2,048 deposited fish, 12 loci, and three outcome-selected wild
  targets (DOUB, LICK, CONK), each compared with a deterministic eight-site wild
  reference and a balanced five-strain hatchery proxy. The raw workbook has
  1,748 wild rows; the paper's final 1,742 cannot be reconstructed because the
  six FLAG/POLE exclusions are not identified. The methods name locus SfoC-79,
  while the deposited workbook contains C115 and no C79.
- The Pennsylvania source is internally inconsistent for DOUB: Table 1 marks
  stocking at the sample location and within 2 km, while the Discussion says
  every site above 10% assigned introgression was not directly stocked. LICK is
  marked only within 2 km. CONK has no stocking record for more than 50 years.
- Nova Scotia: 1,729 fish across 39 raw populations and 100 loci. Eight
  recorded-stocking river systems form the descriptive primary views. The
  shared primary list has 90 loci; the correlated within-population-polymorphic
  sensitivity has 62. P1/P2 were fixed as lower/higher same-marker STRUCTURE-Q
  rows within each river, and raw hatchery populations are proxies rather than
  the paper's centered simulated sources.
- Three zero-record contrasts create six filter views but represent only two
  named biological systems. Together with reference and locus-filter reruns,
  the bundle has 28 records: 11 primary and 17 correlated sensitivities.

All numeric microsatellite alleles are retained directly in PADZE `LociData`;
the biallelic VCF bridge is not used.

## Transfer result and guardrails

The primary `raw_all` view has descriptive candidate concordance 9/11 (0.818):

- Pennsylvania: 3/3 class C.
- Nova Scotia: 6/8 class C; Margaree and Musquodoboit call class A.
- Correlated sensitivities: 12/17 class C.

This pattern is representation-dependent: `raw_mean_variance` calls C in 6/11
primary views and `orbit_composition_mean_variance` in only 1/11. The raw-all C
score has Spearman correlation -0.546 with the Nova Scotia same-locus published
Q contrast, so the classifier score does not reproduce the published ancestry
magnitude ordering.

The external feature shift is extreme: raw-all median RMS-z 528.15,
95th-percentile RMS-z 1,537.57, and 95th-percentile maximum absolute z 6,812.27.
Several softmax scores numerically saturate and the depth-matched gate mostly
saturates near zero. These scores are not posterior probabilities or calibrated
biological effects. No gate threshold decision is accepted.

## Frozen provenance and reproduction

- Source commit: `0a6cfd35ee062e94caa1e97feca62b9e3f3a4f8e`
- Runner SHA-256: `a37a1d87a1fb18fb67d002fc3840c8fda276e7fff6aa5e24deab3aa89e345045`
- Configuration SHA-256: `39ef95df10f024432c3df81d0bed58592d87fcb701bd6bde81d652827c482d4c`
- `results.json`: 985,008 bytes; SHA-256
  `d69ee22f2447526f2f0d5c7577b14e219e756dd60966dd75838deab9ffed6615`
- Runtime: Python 3.12.3, NumPy 2.5.1, scikit-learn 1.9.0, PADZE 0.1.0;
  Azure CPU at nice 15, one numerical thread, GPU disabled.
- Source data: Dryad DOIs `10.5061/dryad.mb37t1q` and
  `10.5061/dryad.rv15dv44w` (CC0). Exact file URLs, byte counts, and hashes are
  embedded in `results.json`; the raw source files are not committed.

With the four downloaded inputs, canonical training arrays, and an approved
Azure compute state available, run:

```bash
python scripts/brook_trout_microsatellite_benchmark.py \
  --pa-workbook /path/to/White_et_al_Brook_Trout_Introgression.xlsx \
  --ns-genepop /path/to/Lehnert_Brooktrout_100Microsatellites.txt \
  --ns-population-names /path/to/Pop_Names_Lehnert_Brooktrout_100micros.csv \
  --ns-introgression /path/to/Introgression_AnthroEnviro_data.csv \
  --canonical-root /path/to/canonical_regen_full \
  --result-dir results/brook_trout_microsatellite_benchmark_2026_07_11 \
  --compute-state /var/local/compute_health.json \
  --compute-target azure
```

The recorded run used explicit owner-authorized stopped-trading and closing-
session overrides. Those flags should only be supplied under the corresponding
owner authorization.
