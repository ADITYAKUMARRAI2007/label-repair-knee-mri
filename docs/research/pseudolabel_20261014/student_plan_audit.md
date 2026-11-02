# Audit of proposed Gemini + R7 teacher / ResNet34 student plan

Date: 2026-10-14. Decision: **do not launch the full plan as written**. The idea of report-derived soft supervision plus an image student is testable, but two premises of the frozen recipe do not hold in the available artifacts.

## Recomputed 58-study diagnostic

Inputs: `data/train.csv` expert labels; `vision_llm_out/gemini-3.8-flash__report.jsonl` (58 studies); `../r7_refinement_20261013/gold58_fixed_comparison/R7.csv`. All twelve classes and all 58 UIDs were aligned. Ties receive average ranks.

| Prediction on Gold58 | Macro ROC-AUC |
| --- | ---: |
| R7 | 0.935289 |
| Gemini 3.8 Flash, report only | 0.906596 |
| 50/50 percentile-rank blend | 0.939146 |
| 50/50 raw probability blend, Gemini scores divided by 100 | 0.937013 |

In the original report-only formulation, the quoted 0.940 refers to the **rank blend**, not the proposed probability-valued training label. The rank result exceeds R7 by about 0.0039; a 2,000-replicate paired study bootstrap for that rank difference gave approximately [-0.0103, +0.0158]. Gold58 has been reused for model selection, so even this interval is only a diagnostic, not a fresh estimate of leaderboard gain. The standalone Gemini report result is below R7 on this cohort, though it may contain useful class-specific information.

The subsequently revised proposal changes two classes to Gemini 4 Argon image+report. On the same 58 studies, using Gemini 3.8 report outputs for ten classes and Gemini 4 image+report for Synovitis and Effusion gives: **0.941208** for the raw probability mix with full R7, but **0.936924** for the raw probability mix with the available fold-model family (R7 ConvNeXt-only). Their rank-blend diagnostics are 0.942123 and 0.940694 respectively. The distinction matters because the ConvNeXt branch is the only R7 component with potential fold-exclusive checkpoint routing. None of these numbers are out-of-fold student results or leaderboard forecasts.

## Missing teacher dependency

`../validation_followup_20261011/oof_feasibility.json` finds that **exact full R7 out-of-fold predictions cannot be reconstructed from the available assets**. The scored ensemble contains an all-data CoAtNet checkpoint, and the older reader's training membership / complete matching fold set is unavailable. The newer R7 ConvNeXt branch has four fold checkpoints in its manifest, but that is only one component of R7; its fold mapping and exclusion history require verification before any OOF claim. Running the scored R7 ensemble on the 4,349 training studies would yield in-sample teacher predictions, not the proposed OOF labels. A genuine full-R7 cross-fit would require retraining the missing components on fixed folds or obtaining the original eligible fold checkpoints and manifests.

The R7 ConvNeXt trainer does expose a specific fold rule: `int(sha1(uid).hexdigest()[:8], 16) % 4`. The proposed shorthand `sha1(id) % 4`, if interpreted as the whole digest modulo four, disagrees on **3,241 / 4,349 (74.5%)** nonexpert studies. This would route most studies through a model that may have trained on them. The four fold checkpoints can potentially form a **ConvNeXt-only** OOF teacher after checkpoint/training-membership verification. That teacher is not the scored full R7 ensemble. On Gold58 the saved ConvNeXt-only output scored 0.931239; a raw 50/50 probability blend with Gemini 3.8 report labels scored 0.933512, while its percentile-rank blend scored 0.939890. These are still reused-Gold diagnostics, not OOF performance on 4,349 studies.

## Architecture and time evidence

The E02 ResNet34 reader scored 0.890 publicly and 0.895 on Gold58 (`../e02_reader_comparison.md`). Its 3-slice windows sometimes crossed MRI series boundaries, and training/inference used different slice spans. The corrected E03 sequence-aware ResNet34 reader ran successfully but scored 0.881 on Gold58 (`../e03_series_reader.md`). Thus code/resource feasibility is established; a strong new student is not. A proposed attention-pooling variant needs a frozen source and training/inference parity, rather than being treated as the already-scored E02 model.

Holding out only the 58 expert UIDs is insufficient for the local Gold screen: the earlier frozen report-group split excluded four additional weak-label studies whose reports overlap expert studies. Reuse that group exclusion and verify patient/image duplicate links where possible.

The 58 Gemini 3.8 Flash report calls averaged 16.81 seconds and 1,095 prompt / 1,799 total tokens each. At that sequential rate, 4,349 calls take about 20.3 hours. A two-hour run requires sustained concurrency of at least 11 calls plus allowance for long tails and retries. Cloud quota, VM transfer of the competition data, disk sizing, and combined R7+fivefold-student inference runtime are not yet measured. The competition requires an offline notebook under nine hours and has a 2026-10-22 23:59 UTC final deadline.

## Bounded correction

1. Resolve the exact teacher provenance first. If no full R7 OOF exists, **change the label recipe explicitly** and rerun its diagnostic; do not call in-sample R7 predictions OOF. Do not launch the 4,349-study label job against the current specification.
2. Pilot Gemini 3.8 Flash report+MRI for Synovitis and Effusion on the existing 58 cases. Previous image gains were observed with other Gemini models, not yet with 3.8 Flash. Treat these 58 as a reused diagnostic.
3. If the label recipe is defensible, freeze one corrected, sequence-aware student architecture; run one fold/resource pilot before five folds. Compare the final student and fixed 80/20 R7 blend on Gold58 as a rejection screen only, and benchmark the complete Kaggle inference notebook before submission.
4. Drop the optional ConvNeXt-Small student until the first student produces an actual complementary improvement. It overlaps with an existing R7 model family and would consume time without a tested hypothesis.

Do not infer a 0.960 score from the 0.939 label diagnostic. Report-derived targets may contain information not recoverable from the MRI, and a student trained on those targets can still underperform its label source.
