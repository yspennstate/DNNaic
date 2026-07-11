# stdpopsim Neanderthal-pulse transfer benchmark (negative result)

This is a known-model simulation transfer test, not natural-data accuracy and not a
three-class benchmark. It asks whether the frozen DNNaic head recognizes one official
ancient pulse whose direction is known from the demographic model.

## Frozen run

- Catalog: `stdpopsim==0.3.0`, HomSap
  `OutOfAfricaExtendedNeandertalAdmixturePulse_3I21`.
- Frozen population order: P1=YRI, P2=CEU, P3=NEA. The model's backward-time
  CEU-to-NEA lineage movement corresponds to forward-time NEA-to-CEU ancestry, so the
  true DNNaic class is C.
- Model contract: 1,298 positive backward-time events, one at every generation
  856--2153, with integrated hazard 0.03 and forward single-lineage probability
  0.02955446645. The control zeros exactly those pulse rates and preserves all other
  demographic events.
- Bank: 30 pulse and 30 control simulations; each uses an independent 1 Mb contig,
  100 diploid samples per population (200 gene copies), and the full PADZE `g=2..199`
  curve.
- Simulation/checkpoint generator: `86d32f0a830f47204ab5d74f8b26e9b831d4d9a1`.
- Final analyzer: `3522aec7cfbec9d0c917976c1692521f5cc3bfbd`.
- Configuration SHA-256:
  `ff1015398d4faff85b2f6a7ebd65c69fd01cabe0a494c49830fb97ab42a719ad`.
- External checkpoint SHA-256:
  `029da37a712acc3017bd99ac53adede3c93abaaa05c180461981b574eea38d27`
  (1,220,526 bytes; 60 x 198 x 28 float32 curves).
- `results.json` SHA-256:
  `a7ec73b04c6cf5eadea306da525d8e37082493e8fa316dc7ab218d49dcc6269d`.
- Canonical compact-JSON `analysis` SHA-256:
  `212231ef71e4410ae61b5aa056949fe0b0153dfe589e739649d9a71329ecfbf0`.

The generator-to-analyzer diff adds provenance and reporting guardrails only; it does
not change the simulation, configuration, or checkpoint logic. Both commits are listed
because the immutable checkpoint was generated before the reporting hardening.

## Primary result

The frozen raw-all direction head predicts C for only 1 of 30 pulse simulations:

- class-C recall: 0.0333 (1/30; Wilson 95% interval 0.00591--0.16670);
- predicted classes: A=25, B=4, C=1;
- mean C probability: 0.01953; median C probability: 0.000165;
- mean C-versus-next-best margin: -0.8488;
- scaler RMS-z: median 7.4225, p95 7.6624; p95 maximum absolute z: 18.5272.

An always-C rule has 100% recall here because every pulse has the same true direction.
This design can strongly falsify class-C transfer, but even perfect recall could not
establish A/B/C skill.

The secondary appreciable-migration gate calls 0/30 pulses at threshold 0.5 and 0/30
controls, with pulse-versus-control ROC AUC 0.7389. Its canonical target is sustained
migration rate at least 2.5e-4, whereas this model contains an episodic ancient pulse;
the gate is therefore only a transfer diagnostic.

## Independent audit and guardrails

A separate read-only audit reproduced the model event catalog, direction mapping,
sample contract, all 60 curve hashes, canonical-array hashes, model coefficients,
predictions, probabilities, summaries, Wilson intervals, and OOD metrics exactly.

Two target-blind fixed-`C=1` representation sensitivities were calculated only after
the primary result and remain secondary:

- raw mean/variance predicts C for 30/30 pulses but also for 22/30 controls; median
  RMS-z is 5.93;
- orbit-composition mean/variance predicts C for 26/30 pulses but also for 12/30
  controls; median RMS-z is 2.33.

Those sensitivities do not rescue directional evidence: a single true direction,
forced-C control behavior, and severe residual OOD prevent an accuracy claim. Raw tree
sequences and count matrices were not retained, so the saved curves and contracts are
fully auditable but reconstructing pre-cast count ledgers requires deterministic
resimulation.

The machine-readable result contains full simulation, prediction, model, source,
configuration, and runtime ledgers. The correct interpretation is a severe transfer
failure under an official ancient-admixture model.
