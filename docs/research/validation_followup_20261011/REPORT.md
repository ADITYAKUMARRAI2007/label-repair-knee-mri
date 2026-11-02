# RSNA Knee — validation follow-up, 11 October 2026

N02 version1 and its 0.950 submission remain unchanged. No competition submission or blend-weight fitting was performed.

## 1. Report overlap — investigated

One gold and four weak studies have identical report content after whitespace normalization; there are four raw strings, and one weak string exactly matches the gold string. The report describes a medial meniscus tear with otherwise normal named structures. This is report reuse, not by itself proof of the same patient or examination.

Selected canonical MRI volumes matched across 0 of 10 linked-study pairs; there were 30 distinct volume hashes and 0 shared series hashes. The tested inputs support report-template reuse without detected duplicate selected volumes. Patient identity and all native images remain unverified.

Conservatively keep all five in one report cluster. When the gold case is held out, exclude all four peers from training. E04 already assigns all five to report_group425/fold0. Dropping only the gold case changes the diagnostic cohort and cannot quantify/remove any learned contamination.

## 2. Baseline OOF — not reconstructable from available assets

CoAtNet has one all-data SWA checkpoint, no matching held-out checkpoint collection or training manifests. The reader release has only folds0/1/2, no prediction CSVs or original labels_v0/fold table; oof.py is collection code. Replaying the published split source now places the four linked weak cases in fold1, whereas an older reconstruction says fold4. The discrepancy's cause is unverified. Under the documented recipe the weak group would enter at least two deployed reader checkpoints, but exact checkpoint membership is not confirmed.

An independently trained comparator must start with generic public pretraining, frozen report/patient groups, held-out group exclusion, and a predeclared final checkpoint rule. Existing E04 has 58 held-out predictions for a different ConvNeXt-Small model; it is a reusable training control, not N02 OOF or fresh selection evidence. Do not repeat its unchanged training merely to collect the same result.

## 3. Acquisition fallback — completed controlled inference

Both arms used the same decoded pixels, three hashed reader checkpoints, k16, resolution and current GPU/runtime. Only opposite-fat-suppression slot duplication changed. **66/82 studies were affected**, including **47/58 gold cases**. Predictions on all **16 unaffected studies matched exactly**. The original 78-study reader CSV was preserved byte-for-byte.

| Target | Studies with changed predictions | Mean absolute probability change | Maximum change |
|---|---:|---:|---:|
| ACL | 66 | 0.023249 | 0.218111 |
| MCL | 66 | 0.022897 | 0.222444 |
| Medial Meniscus | 66 | 0.016266 | 0.102969 |
| Lateral Meniscus | 66 | 0.023974 | 0.156812 |
| Medial OA | 66 | 0.031235 | 0.190559 |
| Lateral OA | 66 | 0.026419 | 0.164384 |
| PF OA | 66 | 0.027779 | 0.265211 |
| Effusion | 66 | 0.046551 | 0.222807 |
| Synovitis | 66 | 0.057425 | 0.230710 |
| Baker's | 66 | 0.054689 | 0.268000 |
| Contusion | 66 | 0.022865 | 0.145779 |
| Fracture | 66 | 0.017665 | 0.075690 |


Measured runtime: 339.8 seconds; peak allocated GPU memory: 1.61 GiB. Version1 failed before any study inference because my wrapper omitted its source directory from the import path. That failure and the single import-only repair to version2 are recorded. **Prediction changes do not establish better accuracy.**

## 4. Lateral OA — independent persistence remains unproven

Removing the report-linked case leaves reader AUC0.853755 versus CoAtNet0.776680 on 57 reused gold studies. Thus that case does not explain the diagnostic difference. This remains reused evidence, not an independent test. No eligible paired CoAtNet/reader predictions on a selection-independent expert imaging cohort were recovered; the previously examined extra16 “Gold74” cases are report-derived, training-eligible cases. Reader blend weight remains0.08.

## 5. Next three GPU experiments

| Rank | Experiment | Information / cost | Leakage risk and generalization gate |
|---|---|---|---|
| 1 | Generic CoAtNet/Tiny feature probe with grouped heads | High information; moderate feature-pass cost | Exclude complete groups; check pretraining/prompt provenance; Gold remains a rejection diagnostic |
| 2 | One blinded-review-supported target correction versus E04 control | High if a specific labeling fault exists; high one-fold training cost | No held-out-label remapping; advance only after matched fold evidence |
| 3 | Physical-position pooling versus relative-index pooling | Medium information; low/moderate head cost after cached features | Check acquisition/site confounding and consistency across folds |

All three are **proposed, not launched**. No performance estimates are assigned. The next executable research direction is rank1 after freezing its eligible cohort, generic checkpoint identities and grouped-head protocol. An untouched, permitted expert imaging holdout is still required to claim independent generalization; new labels on images used to train a comparator do not supply that holdout.

## Rules and artifacts

The run was private and offline, using competition training images and the existing publicly released reader weights/decoder. No new controlled-access patient data was obtained. Official [competition code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/) specify offline notebook execution and permit freely available external data/models; source permissions must still be respected. This is an inference diagnostic, not a submission.

Evidence: `overlap_findings.json`, `overlap_imaging_followup.json`, `oof_feasibility.json`, `lateral_oa_evidence.json`, `paired_execution_verification.json`, `experiment_registry.json`, and `attempt2_output/fallback_pair/`. The exact runnable package is `fallback_pair.ipynb`; the GPU experiment is [private notebook version2](https://www.kaggle.com/code/adityakumarrai0401/rsna-knee-side-fallback-pair-20261011).
