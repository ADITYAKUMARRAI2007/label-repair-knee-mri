# Phase 4 — Completed supervision investigation

**N02 and all existing checkpoints are unchanged. No competition submission was made.**

## Original DICOM verification

All 144 native frames across each study’s five selected series have identical ordered pixels and geometry. Both cached volumes match fresh original-DICOM reconstruction. File bytes and deidentified patient identifiers differ. Both image headers say right knee; one report says left knee and the other right knee.

This confirms duplicated imaging content with inconsistent report correspondence. It does not verify patient identity or establish which report is correct. Both studies remain in one validation group and will be excluded from supervised training for the new model. No report was chosen as truth.

## Frozen blinded report review

The sample contained 48 unique weakly labeled studies: two extractor disagreements and two random controls per target. Seed: 20261016. Reviewer inputs concealed study identifiers, label probabilities and sample roles until judgments were saved.

The assistant’s generic finding judgments were 21 positive, 6 negative and 21 uncertain. These are report interpretations, not radiologist image labels. Checking the [host’s clinical definitions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) revealed that generic finding presence was too broad for the severity-defined targets. The initial apparent degeneration false negatives must not be called competition labeling errors.

## Exactly one correction experiment

A conservative, predefined rule mapped explicit isolated grade-1 MCL sprains to a soft negative of 0.05. It excluded Gold58, known report-linked cases, the duplicate pair, and reports with competing MCL tear or high-grade descriptions. It changed 139 weak targets, including 137 previously positive targets. Original confidence weights, frozen features, ridge regularization and all 15 grouped splits were retained.

The reference was the original 58 image-reviewed expert targets. Those targets were independent of this correction mechanism, but had already been reused for model selection. **Gold58 is not an independent clinical evaluation cohort.**

| Reused Gold58 diagnostic | Original | Corrected | Difference |
|---|---:|---:|---:|
| Macro AUC | 0.751264 | 0.747484 | −0.003779 |
| MCL AUC | 0.633409 | 0.588057 | −0.045351 |

The other 11 class AUCs were unchanged. Paired group bootstrap, 2,000 valid draws:

- Macro difference 95% CI: [−0.010636, +0.001714].
- MCL difference 95% CI: [−0.127637, +0.020563].
- Macro differences were negative in all three repeated split comparisons.

The correction fit and analysis used 191.1 CPU seconds and no GPU time. The intervals include no effect; this is not proof that the rule is intrinsically harmful.

**Decision: reject this correction for current training. Phase 4 is closed.** The next model uses the original targets, with the two verified ambiguous duplicate studies excluded. No further correction searches will be conducted in this track.

## Limitations and recorded failures

Report interpretation cannot establish image truth. Severity thresholds, uncertain wording, report correspondence and the small reused expert cohort limit conclusions. The frozen-feature ridge result does not establish how a corrected end-to-end MRI model would behave.

The initial CPU notebook launch was rejected with a title/slug mismatch; the matching-title retry completed. Correction membership serialization failed before any fits; the integer cast was fixed without changing the frozen rule or membership. Both failures were preserved in the experiment registry.
