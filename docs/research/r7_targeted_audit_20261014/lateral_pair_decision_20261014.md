# Lateral OA reader experiment — 14 October 2026

## Decision

The fixed 50/50 blend of the prior ROI head and the new adjacent-window pair head passed the exploratory grouped weak-report proxy gate. A private Kaggle native-DICOM pilot and full R7 inference notebook passed operational checks. One conservative code submission was sent for official scoring: **57184339**, currently pending. The confirmed R7 best remains **0.956** until Kaggle reports a score.

## What changed

The new head uses the same frozen ImageNet ConvNeXt-Tiny feature vectors and lateral-compartment crops as the prior ROI reader. It scores adjacent windows within sagittal and coronal MRI acquisitions and fuses their context. It changes the sequence head, not the encoder or pixel pipeline. Five patient/duplicate/report-group folds were reused, with Gold58-linked groups excluded. The clear-label-only training variant was tested separately and rejected.

## Results on the report-label proxy

| Reader | Lateral OA clear-subset AUC |
| --- | ---: |
| Prior ROI | 0.7605 |
| New pair ROI | 0.7693 |
| Fixed 50/50 rank blend | **0.7826** |

The pair head alone improved the pooled AUC by +0.0088, but the paired group-bootstrap 95% interval was -0.0046 to +0.0235, and it improved only three of five folds. It failed its prespecified standalone gate. The fixed 50/50 blend improved the pooled AUC by +0.0221 against prior ROI, with gains in all five folds and a paired interval of +0.0143 to +0.0299. The old and new ROI predictions had Spearman correlation 0.724 on the eligible out-of-fold cohort. This blend was specified after inspecting the standalone result, so its interval is descriptive and does not account for model selection.

These are **weak report-label** measurements on 4,301 eligible studies, with 1,052 clear positive and 2,051 clear negative endpoints. They are not expert-image-label or R7 ensemble AUCs. Gold58 is repeatedly reused and was not used to select the blend. Exact R7 fold-exclusive OOF predictions are unavailable, so no independent R7 incremental gain is established.

## Execution and submitted candidate

All five pair checkpoints reproduced their saved held-out predictions with zero numerical error; the packaged model class matched the training class. The private Kaggle pilot decoded the three visible test studies, matched cached training features with maximum error 0.0, produced finite predictions for all three, took 42.35 seconds including QA, and allocated 402,402,816 peak GPU bytes.

The full R7 notebook completed in about 7.5 minutes. Its lateral pair reader took 13.1 seconds on the three visible test studies. A local audit verified all IDs, twelve output columns, finite probabilities, exact formula, zero pilot prediction drift, unchanged values in the other eleven R7 columns, and matching output hashes.

The submitted formula is fixed:

```text
Lateral OA = 0.90 rank(R7 Lateral OA)
           + 0.10 rank(0.50 rank(prior ROI) + 0.50 rank(pair ROI))
```

All other classes retain the exact scored R7 values. Submission **57184339** is pending. The earlier two-class OA ROI submission **57179067** completed with displayed public score **0.956**, equal to R7. Kaggle's score for 57184339, not the three visible studies, will determine whether the lateral-pair candidate improves the public score.

## Score expectation and next decision

The reported +0.0221 applies to one weak-label class proxy. With a 10% adjustment to only one of twelve macro-AUC classes, the practical public-score expectation is near the existing 0.956, perhaps around 0.956–0.957. This is a planning estimate, not a measured leaderboard score. It does not support expecting 0.960 from this candidate alone.

If the official score improves, preserve the candidate as a small lateral OA correction. If it does not, return to R7 and focus on more reliable lesion-level supervision or a broader multi-class improvement; do not tune weights against this public result.

## Evidence files

- `lateral_continuity_protocol.json`, `lateral_continuity_validation/results.json`
- `lateral_pair_blend_protocol.json`, `lateral_pair_blend_validation/results.json`
- `lateral_pair_serving_dataset/manifest_pair.json`
- `lateral_pair_pilot_output/lateral_pair_test.csv.receipt.json`
- `r7_pair_output/local_output_audit.json`
- `r7_pair_official_score.json`
