# Phase 6: one route from the confirmed 0.956 R7 baseline

Status: **locked protocol; no Phase 6 submission yet**. Existing OA submissions are scored or pending; they are not Phase 6 tuning data.

## Objective and frozen assets

Target: score strictly above 0.960 public macro ROC-AUC while preserving the scored R7 notebook and all checkpoints. R7's confirmed displayed score is 0.956 (submission 57146375). A four-thousandths macro gain requires roughly +0.048 summed AUC across the twelve classes. There is no defensible numerical forecast that this route will achieve it.

Use the **already trained five-fold Phase 5 regional ConvNeXt reader**. Its grouped weak-report OOF result exceeded a separately trained matched global reader by +0.00568 macro AUC (paired group bootstrap CI +0.00354 to +0.00780). This is evidence for regional input within that training experiment, **not** evidence of incremental gain over R7 or clinical validation. The four prespecified target classes are Medial Meniscus, Lateral Meniscus, Contusion, and Synovitis; their regional-minus-global OOF differences were +0.0234, +0.0162, +0.0116, and +0.0073 respectively. Do not promote the reused Gold58 to independent validation.

## Sequential gates

1. **Track prior experiments.** Record the final score of pending OA submission 57184339 when it arrives; this does not delay the Phase 6 engineering work. Keep R7 as the comparison baseline; if that pending submission scores higher, record it separately. Do not change the Phase 6 formula in response.
2. **Make Phase 5 inference operational.** Optimize DICOM loading only, preserving ordered slice selection, physical crop, pixel scaling, all five checkpoint hashes, and per-fold predictions. The existing 16-study pilot averaged 10.37 seconds in DICOM preparation and 0.59 seconds in the five models; decode is the bottleneck. Test the optimized reader on the same 16 frozen studies plus the 3 visible test studies. Require exact reconstructed volume hashes where expected, all 12 finite predictions, no defaults, and maximum absolute prediction drift <=0.001 versus the existing pilot. Measure end-to-end combined R7 + Phase 5 runtime, setup included. Require a conservative projection under the 9-hour notebook limit with a stated safety margin, based on an expanded label-blind native-DICOM sample. If this gate fails, stop this route; do not submit an inference package that may time out.
3. **Freeze one exploratory blend before leaderboard feedback.** On four target classes only, use `0.80 * percentile_rank(R7) + 0.20 * percentile_rank(Phase5 regional)`, with average tie ranks over the complete inference cohort. Keep the other eight R7 columns exactly as scored. This 20% is a fixed conservative experiment, not an optimized weight. Verify UID alignment, unique IDs, probability bounds, class order, and that all five Phase 5 folds contribute. Do not test a grid of weights or use visible test cases as labels.
4. **One official evaluation if gates 2–3 pass.** Save the offline Kaggle notebook, per-component predictions, runtime and parity receipts, candidate CSV, and formula hash. Submit this one frozen candidate using the user's standing authorization to test promising candidates. Record the exact submission ID and official displayed score. A result above 0.960 meets the stated score goal; a lower result is evidence against this specific blend, not proof the regional reader is intrinsically bad. Do not relabel a 0.956 tie as improvement.
5. **Decision.** Preserve R7 regardless of outcome. If this candidate does not exceed R7, close Phase 6 and perform a new error audit before proposing another model or hyperparameter change. If it improves but remains below 0.960, report the gain and the remaining gap before selecting another experiment. No new backbone training, OA-only correction, arbitrary class weights, or multiple parallel GPU jobs in this cycle.

## Current evidence and limitations

- Phase 5 runtime receipt: `../competitive_phase45_20261011/inference_output/phase5_inference/runtime_report.json`. The 2×p95 × 1,300 estimate is 12.48 hours, beyond the documented 9-hour cap. Its hard-case sample is small, so this is a failed conservative projection, not proof actual hidden inference would time out.
- Source bottleneck: `../competitive_phase45_20261011/dependencies/e03_dicom_reader.py` reads every DICOM header in each chosen series, then decodes selected slices. Any shortcut must prove identical image tensors, because slice ordering and normalization affect predictions.
- Phase 5 has grouped weak-report OOF predictions. R7 does not have verified fold-exclusive OOF predictions for the same subjects, so an independent paired estimate of the proposed R7 blend is unavailable. The one official score is an exploratory evaluation.

## Stopping rule

Do not spend GPU time on new training until the regional candidate has either completed this protocol or been rejected at the runtime/parity gate. A failure closes this route; it does not trigger an immediate switch to another architecture.
