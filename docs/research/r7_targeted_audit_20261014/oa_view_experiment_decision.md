# R7 targeted OA view experiment — decision, 14 October 2026

## Direct decision

**The fixed compartment view contains some complementary signal, especially for Lateral OA, but it has not passed the prespecified advancement gate and is not a submission candidate.** Keep R7 (official public 0.956) unchanged. There was no new competition submission from this experiment.

## What ran

- 24-study MRI image-view QA with physical lateral femorotibial and patellofemoral boxes. The lateral sagittal design was revised to sample three consecutive lateral-end windows after contact-sheet review, before full feature scoring.
- A private Kaggle GPU feature run on all 4,407 cached training studies. Frozen ImageNet ConvNeXt-Tiny, identical six three-slice windows per target and arm, whole 140 mm view versus fixed compartment view. The receipt reports 566.61 seconds of GPU work and 402,402,816 peak allocated GPU bytes. No labels were read or model trained in the Kaggle feature job. Complete target tokens: 4,363/4,407 Lateral OA and 4,407/4,407 PF OA. Laterality was unresolved for 44 studies; these were masked for Lateral OA.
- Five grouped folds of the **same** small 24-epoch adjacent-window head for both view arms. Every Gold58-linked group was excluded; 4,345 studies were eligible before missing-view masking. The primary endpoint used clear report-derived weak labels (target >=0.60 or <=0.15), not image-adjudicated OA labels. Head training and bootstrap ran locally; no Kaggle submission was generated.

The frozen experiment protocol was saved before the full-feature outcome was known in [oa_feature_protocol.json](oa_feature_protocol.json). Full receipts and exact results are in [receipt.json](oa_feature_output/oa_frozen_features/receipt.json) and [results.json](oa_feature_validation/results.json).

| Target | Whole AUC | Compartment AUC | Difference | Positive folds | Paired group 95% interval for difference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lateral OA | 0.7409 | 0.7605 | +0.0197 | 5/5 | −0.0009 to +0.0403 |
| PF OA | 0.7393 | 0.7478 | +0.0085 | 3/5 | −0.0102 to +0.0270 |

The prespecified gate required at least a +0.01 target gain, a lower paired group-bootstrap bound above zero, and improvement on at least four folds. Lateral OA met the point estimate and fold tests but missed the uncertainty test narrowly. PF OA missed the point estimate, uncertainty, and fold tests. The overall result therefore **fails the advance gate**.

## Exploratory complement check

After seeing the matched-view result, I made **one fixed 10% rank-blend diagnostic** with the existing Phase 5 regional reader's grouped out-of-fold predictions. This was selected after the primary result and is exploratory; its intervals do not account for that selection. It does not evaluate R7, whose fold-exclusive predictions are unavailable.

| Clear weak-report target | Phase 5 regional alone | +10% new whole-view reader | +10% new compartment reader | Compartment minus Phase 5 |
| --- | ---: | ---: | ---: | ---: |
| Lateral OA | 0.8607 | 0.8619 | 0.8636 | +0.0029 |
| PF OA | 0.8502 | 0.8516 | 0.8529 | +0.0027 |

The paired group intervals for compartment additions were +0.0015 to +0.0044 for Lateral OA and +0.0013 to +0.0040 for PF OA. Exact results are in [phase5_roi_complement.json](oa_feature_validation/phase5_roi_complement.json). This suggests weak-label ranking information that the Phase 5 reader does not fully capture. The improvement is small, on a report-derived proxy, and cannot be extrapolated to R7's public 0.956. Even if the two class gains transferred unchanged, their contribution to a 12-class macro AUC would be only about +0.0005.

## Practical recommendation

Stop here for this variant. Do not spend Kaggle submission or another GPU training run on this fixed frozen-feature crop. If new independent image-adjudicated cases become available, the worthwhile next test is a **single frozen Lateral OA compartment-reader hypothesis** against fold-exclusive R7 predictions, with the same group and uncertainty rules. The current results do not justify a claim of improved competition score.

## Later user-authorized exploratory submission

The user subsequently directed a competition test despite the conservative recommendation above. The ten exact grouped-fold ROI heads were reproduced with zero prediction drift from their saved OOF outputs. A native-DICOM Kaggle pilot returned finite Lateral OA and PF OA predictions for all three visible studies in 19.46 seconds. A private combined notebook completed in about 8 minutes and changed only those two classes using the frozen formula `0.90 × rank(R7) + 0.10 × rank(ROI)`; the other ten R7 columns were byte-equivalent as parsed numeric values.

Kaggle accepted code submission **57179067** on 14 October 2026. Its official public score is still pending. The [submission receipt](r7_oa_official_submission.json), [combined notebook receipt](r7_oa_output/r7_oa_blend_receipt.json), and [score status](r7_oa_official_score.json) preserve the distinction between a working notebook and a scored competition result. No improvement over 0.956 is claimed until Kaggle posts that result.
