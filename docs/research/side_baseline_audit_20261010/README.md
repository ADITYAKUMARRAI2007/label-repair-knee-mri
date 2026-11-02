# RSNA Knee: audit of the verified 0.950 baseline

## Status and scope

This audit was performed in a separate side conversation on 10 October 2026. It reads the exact accepted implementation and archived outputs, and writes CPU analysis artifacts only in this directory. It does not launch training, change the working baseline, submit predictions, or control the main thread's jobs.

**Highest verified notebook:** [N02 Fixed Reader Blend V3](https://www.kaggle.com/code/adityakumarrai0401/rsna-knee-n02-fixed-reader-blend-v3), version 1, submission **56967269**, public AUC **0.950**. The saved official receipt records COMPLETE and the score. Rank 415 is user supplied; this audit does not verify current rank or private score. The milestones 0.955/0.960/0.970/0.975 remain research objectives without score forecasts.

The first task is complete: source audit, recovered component diagnostics, three ranked hypotheses, and a validation plan. Independent validation and a stronger final candidate are **not established**.

## Reproduce this CPU audit

From the repository root:

```sh
.venv/bin/python research/side_baseline_audit_20261010/analyze.py
```

The script reads only archived CSVs, Python source literals, JSON receipts, and metadata. It never imports or executes the inference/training code. It checks source/module identities, exact ID alignment, finite predictions, the blend arithmetic, the E04 OOF artifact identity, and input hashes again at the end. Bootstrap seed is 20261010 with 2,000 shared report-group draws. Macro intervals use all 12 classes in every retained draw. All 2,000 draws were valid in this run.

Outputs:

- `source_identity.json`: exact baseline code identity, checkpoint records, official score, archived execution audit.
- `diagnostic_summary.json` and `per_class_diagnostic.csv`: component AUCs, paired deltas, confidence intervals, correlations, and pairwise disagreement.
- `ranking_failure_cases.csv`: five studies per target with the most misordered opposite-label comparisons; this is a review index, not an anatomical diagnosis.
- `label_disagreement_diagnostic.csv`: comparisons of three available report-label sources, with missing cells explicitly excluded.
- `metadata_coverage.csv` and `coverage_summary.json`: metadata slot presence across all 4,407 studies; this does not test DICOM readability.
- `validation_inventory.json`: recovered OOF evidence and limitations.
- `experiment_registry.json`: ranked experiments, actual findings, and promotion gates.
- `input_manifest.json`: hashes of inputs and the analysis script.

## 1. Exact baseline specification

### CoAtNet component

| Item | Verified implementation or archived checkpoint metadata |
|---|---|
| Architecture | `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k` with Raptor study classifier |
| Checkpoint | `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt` |
| Identity | SHA-256 `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef` |
| Lineage | Nartaa accuracy release; Raptor architecture attributed to dreaddevelopment; timm hybrid backbone |
| SWA | Archived checkpoint lists epochs 14, 11, 12; this is not a record of the entire training schedule |
| Study head | Backbone features → LayerNorm → 12 attention distributions over windows → 12 class-specific weighted feature sums → learned class weights/biases |
| Input slots | Sagittal fluid-preferred 26; second sagittal non-fluid-preferred 22; coronal fluid-preferred 18; second coronal non-fluid-preferred 12; axial 18 |
| Slice count | 96 sampled frames in five concatenated blocks; 94 overlapping three-frame windows when all blocks are present |
| Series selection | First metadata row matching plane and preferred fluid flag; fallback to first unused series in that plane; one series cannot fill two slots |
| Coverage | Rounded uniform picks from approximately 2–98% of each selected series; repeated picks possible in short series |
| Crop | Central physical crop up to 140 mm, resized to 384×384, then central 320×320 model input |
| Effective FOV | Approximately 116.7 mm when the initial crop reaches 140 mm; smaller when acquisition FOV caps the crop |
| Ordering/orientation | Sort by projected image position along IOP-derived normal; InstanceNumber fallback. No canonical LPS transpose/flip step |
| Intensity | Modality LUT, MONOCHROME1 inversion, pooled selected-series 2nd–98th percentiles, uint8 quantization, ImageNet channel normalization |
| Active decoder | Embedded `study_lut` fast reader with pydicom fallbacks, histogram/LUT normalization, 32 header/16 pixel threads per prep process |
| TTA | Sagittal triplets reverse slice channels; coronal/axial triplets flip image width; equal plain/mirrored sigmoid-probability average |
| GPU execution | One model copy per GPU, per-study batch 1; backbone processes at most 16 windows per chunk; FP16 autocast |
| Spatial information | No explicit plane, sequence, or physical-position embedding in the Raptor head; token aggregation is order invariant |

**Executable code takes precedence over stale comments.** The inherited introductory text describes older 64-slice/0.924 recipes and the ARMS comment mentions folds, but the accepted executable loads **one SWA checkpoint**, uses 96 frames, and performs anatomical TTA. The 96 value is part of that checkpoint's fixed corpus/inference contract; it was not selected through this audit.

The per-slice mask bounds the first and last window centers. It does **not** exclude internal missing-slot windows or prevent triplets crossing series boundaries. With all five slots present, eight triplets cross a block boundary. This is a representation concern, not evidence that deleting those windows improves predictions.

### ConvNeXt component

The archived checkpoint inspection establishes **ConvNeXt-Tiny**, not Small or ResNet-50. All three checkpoints record `convnext_tiny.fb_in22k_ft_in1k`, model input resolution 256, training `k=8`, training evaluation `k_eval=12`, 14 configured epochs, and EMA 0.998. Final inference uses `k=16`.

| Checkpoint | SHA-256 |
|---|---|
| `cnxt_v0_fold0.pt` | `0b64baa9d1c29c8d5c4dffe06d9ab2c941119b37738e5a2346e882ab694e7b80` |
| `cnxt_v0_fold1.pt` | `61696b6e67474695242fc8049e96e2ea13987db6e20d04410dc2961a689585cb` |
| `cnxt_v0_fold2.pt` | `f5cf19cf591600c1c1a04f1aa5ff467751ef01513426a4ea3baa789443656cb6` |

Each archived file is 123,035,271 bytes. This audit verifies the recorded identities and source bindings; it does not download and rehash model binaries.

- Six acquisition slots: sagittal/coronal/axial with fat suppression, then the same planes without fat suppression. This uses `Fat_Suppression`, whereas CoAtNet preferences use `Fluid_Sensitive`.
- Candidate series are ordered by file count capped at 64; the first candidate is loaded. If that candidate fails, the published dataset skips that slot rather than trying every alternative candidate.
- A missing fat-suppression slot can copy the opposite slot in the same plane. The same acquisition then receives two different slot embeddings. This fallback is present in the accepted inference code and absent in the archived training `knee.py`.
- DICOM images are transposed/flipped into canonical LPS plane orientation, sorted by canonical slice direction, resampled at 0.4 mm/pixel into 384×384 (153.6 mm FOV), and padded if needed.
- More than 64 decoded frames are uniformly reduced to 64. Sixteen adjacent three-frame windows are selected per slot over approximately 4–96% of the resulting volume.
- Series intensities use 1st–99.5th percentiles of a spatially subsampled resampled volume, MONOCHROME1 handling and linear rescale, and rounded uint8 values. Model views crop a centered 0.92 FOV fraction and resize to 256, approximately 141.3 mm when fully populated.
- The backbone feeds a 384-dimensional projection, six slot embeddings, relative slice-position MLP, two 8-head transformer layers, and per-target attention pooling. Position is relative frame index, not physical distance.
- Three checkpoint probabilities are averaged equally; no anatomical TTA is applied to the reader. FP16 inference, channels-last models, batch 4, four data workers, one GPU for this stage.

**96 reader windows are not 96 raw MRI slices:** six slots × sixteen overlapping triplets creates at most 96 tokens; duplicate slots can represent the same source acquisition.

### Final blend and safeguards

For each target: obtain average percentile ranks for the CoAtNet output and the averaged reader probability over the complete serving cohort; compute `(1-w) * anchor_rank + w * reader_rank`; rank that mixture again. Reader weights:

| Target | Reader weight | Target | Reader weight |
|---|---:|---|---:|
| Baker's | 0.35 | Contusion | 0.30 |
| Medial OA | 0.25 | ACL | 0.25 |
| Medial Meniscus | 0.20 | MCL | 0.20 |
| Lateral Meniscus | 0.08 | Fracture | 0.08 |
| Lateral OA | 0.08 | PF OA | 0.08 |
| Effusion | 0.04 | Synovitis | 0.03 |

These weights are a frozen published recipe, not parameters independently selected in this audit. CoAtNet internally exports ordinal ranks using `argsort().argsort()`; equal probabilities can have their ties broken before the outer tie-aware fusion. Raw CoAtNet probabilities are not in the recovered component CSV.

The accepted validator checks **exact ordered study IDs**, uniqueness, all 12 target names/order, finite values, and range. There is no row-alignment bug in the final wrapper. It rejects stale submission files and logged study-level FAILED/FALLBACK events, rejects a reader study with no decoded acquisition, installs/probes the offline decoder, and writes the final CSV only after both components validate.

**Remaining recovery gap:** CoAtNet's fast per-slice decoder can turn failures into absent/zero slices without a per-slice warning. The mask is not a full head attention mask. A zero-volume study without a logged preparation exception is not explicitly rejected by the anchor. Reader acquisition recovery is logged. Finite output alone cannot establish usable imaging input. An isolated candidate should eventually add measured decode counters and reject unexplained failure; keep the accepted baseline intact.

The saved 78-study blend replays exactly when the original arithmetic order and round-trip float parsing are retained. Algebraically regrouping rank division/multiplication changed tie outcomes during an initial CPU check; the audit was corrected to match the accepted implementation. The working notebook was not modified. This is a reproducibility issue to guard against in future ports.

### Dependencies and measured resource evidence

- Accepted offline GPU script: `public_kernels/fixed_reader_blend_v3/rsna-knee-n02-fixed-reader-blend-v3.py`; SHA-256 `ca4dbebfc379e45064d6cd7441c281d4f05114d54f426d68c0a4a34e653f2032`. It matches the saved submission-source download, and all 12 bound module hashes match.
- Kaggle inputs: competition data; `nartaa/rsna-knee-publication-swa-weights-20261007`; `goodpjw2008/rsna-knee-2-5d-convnext-reader`; `adityakumarrai0401/rsna-knee-decoder-cp313`.
- Pinned Docker digest is recorded in notebook metadata; Torch 2.11.0+cu128, timm 1.0.29, Python cp313, **exactly two GPUs** are asserted. Decoder wheels are individually hashed: pylibjpeg 2.1.0, libjpeg 2.4.0, openjpeg 2.6.0.
- Saved visible execution: **177.69 seconds**, including decoder/model setup, 20-study reader qualification, and three visible test studies. Anchor stage 31.16 seconds; reader stage 104.93 seconds including qualification.
- Reader qualification: five real batches of four studies, 62.06 seconds, approximately **5.01 GiB peak allocated GPU memory**. Synthetic B4/all-six-slot inference was also checked.
- One combined wall-time budget: 600 seconds for visible data, 7,200 seconds for the hidden branch. These are wrapper limits, not a freshly verified statement of official competition rules.
- The official score proves the accepted version completed competition evaluation. Local artifacts do **not** provide exact hidden-study count, hidden-run inference duration, peak CPU RAM, decoder-event distribution, or safety margin for a larger candidate. The visible runtime cannot supply those numbers.

### Training provenance: known versus unknown

CoAtNet publisher documentation names a blend of public report-derived labels (including JEV, Steven and Pilkwang) plus its own targets and masked OAI supervision from 2,399 baseline knees. This is documented provenance, not an independently reconstructed target table. Exact per-study targets/weights, external split identities, and full training logs are unavailable here. The package contains learned weights and acknowledgements rather than raw OAI records.

Reader checkpoint args specify `labels_v0`. Archived `make_labels.py` defines v0 as the mean of Steven v4, Pilkwang v2, Lixin/GPT56 and Qwen tables, clipped to 0.01–0.99. Archived training source excludes Gold58 from image training, groups weak studies by normalized report, and trains folds 0/1/2 with final EMA checkpoints. The exact released `labels_v0.csv`, Qwen source table and complete source training-membership manifests were not recovered. The three available tables therefore cannot reproduce the exact v0 training targets.

The normalized reports contain **one Gold group also present in four weak studies**. This is a report-level contamination risk in pipelines that exclude Gold rows but retain report duplicates; it is not proof of shared patient identity. The existing E04 grouped contract assigns the entire group to one fold. Both published and local selection have repeatedly used Gold58, so exclusion from image training does not make those studies independent model-selection evidence.

## 2. Recovered measurements — diagnostic only

The recovered predictions came from the archived Colab A100 port using the same checkpoints/preprocessing/recipe, with a different CUDA build from the scored two-T4 version. Their rank reference cohort has 78 studies; labels exist for 58. No new GPU inference was run for this audit.

| Component | Gold58 macro-AUC | Conditional 95% bootstrap interval |
|---|---:|---|
| CoAtNet | 0.922943 | 0.893174–0.947857 |
| Three-checkpoint ConvNeXt reader | 0.915286 | 0.885561–0.940011 |
| Frozen N02 blend | 0.925662 | 0.895355–0.949703 |

Blend minus CoAtNet: **+0.002720**, paired interval **−0.000137 to +0.005622**. Blend minus reader: +0.010377, paired interval −0.005170 to +0.026519. These are conditional sampling intervals on reused data, not confidence intervals for leaderboard AUC or future training performance.

| Target | Positive / negative | CoAtNet | Reader | N02 |
|---|---:|---:|---:|---:|
| ACL | 24 / 34 | 0.9645 | 0.9767 | 0.9712 |
| MCL | 9 / 49 | 0.9433 | 0.9456 | 0.9524 |
| Medial Meniscus | 26 / 32 | 0.9820 | 0.9663 | 0.9808 |
| Lateral Meniscus | 23 / 35 | 0.8957 | 0.8696 | 0.8957 |
| Medial OA | 15 / 43 | 0.9814 | 0.9814 | 0.9845 |
| Lateral OA | 11 / 47 | 0.7756 | 0.8549 | 0.7795 |
| PF OA | 21 / 37 | 0.8597 | 0.8340 | 0.8597 |
| Effusion | 35 / 23 | 0.9727 | 0.8820 | 0.9714 |
| Synovitis | 27 / 31 | 0.8292 | 0.8005 | 0.8280 |
| Baker's | 12 / 46 | 0.9764 | 0.9837 | 0.9764 |
| Contusion | 19 / 39 | 0.9366 | 0.9568 | 0.9501 |
| Fracture | 18 / 40 | 0.9583 | 0.9319 | 0.9583 |

Per-target intervals, component deltas, Spearman correlations and pair counts are in the CSV. Mean per-target component Spearman correlation is **0.8902**. They both strictly misorder 73 positive-negative pairs for Lateral OA, 87 for PF OA, and 98 for Synovitis. Changing weights cannot repair a pair that both components order incorrectly, except that tie handling can change equality.

Metadata checks: 42/58 Gold studies have all five anchor slots; 14 lack the second coronal slot and two lack the second sagittal slot. The latter slots prefer non-fluid sequences but can fallback, so absence is specifically absence of a second unused series. Under metadata assumptions, 14/24 of the 94 anchor windows respectively touch missing blocks. Potential opposite-fat-suppression slot duplication affects **47/58** Gold studies. These counts describe metadata; actual decoded volumes need separate checks.

Three-source weak-target disagreement is greatest by range>0.5 frequency in medial meniscus (13.18%), contusion (12.60%) and lateral meniscus (9.93%). Synovitis has a substantial mean probability range (0.2993), although few differences exceed 0.5. Report-source versus expert-label AUC is roughly 0.68–0.79 for Synovitis and 0.79–0.83 for Lateral OA, with one source covering only 57 Gold studies. This suggests a supervision question; correlated source errors and prior prompt tuning prevent calling any source independent ground truth.

Excluding the one Gold study whose normalized report also appears in weak training gives 57-study baseline AUC **0.924759**. This changes the cohort, not the model; it neither proves independence nor represents a score improvement.

## 3. Three most credible plateau explanations and ranked cheap experiments

### Rank 1 — Report supervision may not match image-defined targets

**Evidence:** Weak report-label agreement is limited for Synovitis/Lateral OA, which are also baseline weaknesses. Exact source targets and selection lineage are incomplete. Three labelers can agree on the same report omission or ambiguous severity.

**Low-cost experiment:** Freeze a balanced report-review sample before changing labels: disagreements and random controls across Lateral OA, PF OA, Synovitis and meniscal findings. Audit omission, negation, compartment and severity against verified competition annotation definitions, with a reviewer blind to model predictions. The initial CPU source-disagreement analysis is already complete; it guides sampling without declaring a new target correct. No image-model training is needed for this diagnostic.

**Confirm/reject:** Support a specific supervision change only if blinded review repeatedly identifies the same extraction/ontology error relative to controls. If the reviewed targets are consistent and errors remain image-related, deprioritize label extraction for that class. Report review alone cannot establish improved image AUC; a subsequent fold-controlled training comparison is mandatory.

**Critic:** Removing all confidence weights was already tried and did not establish a gain. Agreement between LLMs is not a calibrated probability of correctness. Do not repeat the uniform-weight experiment.

### Rank 2 — Shared errors and a fixed blend may obscure useful class information

**Evidence:** Mean correlation 0.8902; numerous shared ranking errors. Reader Lateral OA AUC exceeds the anchor but receives 8% weight. The diagnostic blend gain over the anchor has an interval crossing zero. A tri-reader variant improved Gold diagnostics yet tied 0.950 publicly.

**Low-cost experiment:** The saved-prediction audit already separates shared errors, disagreements, component strengths and blend gains per target. Next, collect genuine fold-specific component predictions with manifests, and compare the frozen baseline blend with one preregistered regularized cross-fitted blend. Fit inside development folds and evaluate only untouched outer folds. Use raw probabilities for probability averaging; rank CSVs cannot substitute for them.

**Confirm/reject:** Useful complementarity must survive outer folds and cohort-rank sensitivity, rather than appearing only on one reused sample. Predominantly shared errors or a confidence interval consistent with no gain argues for new supervision/representation. Do not tune the 8% weight using public submissions or the observed 58-study result.

**Critic:** Existing reader fold0/1/2 averages are not baseline OOF. Folds named in a checkpoint do not make every study out of fold. Cross-fitting only the blend cannot undo a base model's training or selection contamination.

### Rank 3 — Acquisition handling may discard anatomy or introduce sequence mismatch

**Evidence:** Different physical FOVs; anchor has series-boundary triplets and internal missing-block windows; reader duplicates acquisition content into alternate sequence embeddings although that fallback is absent in archived training source. Metadata counts identify affected studies.

**Low-cost experiment:** One frozen-checkpoint reader ablation: disable opposite-fat-suppression slot duplication while retaining k=16, resolution, decoder recovery, series choice and blend weights. Run paired predictions on the identical preregistered cohort, and analyze affected versus unaffected acquisitions. First verify with CPU metadata that the intended duplication is the only changed input. This requires inference only, no retraining, and remains queued for the main execution workflow.

**Confirm/reject:** A stable affected-stratum benefit across held-out folds supports a sequence-tag mismatch; no benefit or regression rejects this particular fallback change. Unaffected studies should retain identical reader inputs. Gold58 can reject a harmful change but cannot independently promote a positive one.

**Critic:** The already tested series-coherent window filter did not improve the baseline. Removing a seemingly invalid window can break the distribution learned by the checkpoint. This experiment is distinct from that filter and must not combine cropping, sampling and fallback changes.

## 4. Independent validation plan and current blocker

Recovered E04: **58 held-out expert predictions** from a different, locally trained model, macro-AUC **0.883299**, matched to the audited artifact hash. Its fixed 10% blend with N02 was previously **−0.000787**, interval −0.003546 to +0.001822. It supplies a useful rejection comparator, not genuine OOF for the published CoAtNet/reader ensemble and not a fresh independent selection set.

The official CSV lacks patient/site IDs. Normalized-report grouping is feasible and necessary, but does not establish patient-level separation. Use permitted DICOM identifiers or permitted trusted identity metadata if available; never equate study UID with patient UID.

Mandatory next validation steps:

1. Inventory training membership, target-source provenance and every use of Gold58 for all candidate checkpoints. Mark each study/model pair as trained, held-out, or unknown. Unknown is ineligible for an independent estimate.
2. Recover real weak-fold reader outputs and exact labels/folds from the publisher if available. CoAtNet's all-data SWA cannot become OOF by rerunning inference or averaging released models.
3. For new training, freeze report/patient groups, preprocessing, target rules and checkpoint selection before training. Exclude the complete held-out group from all parameter updates, adaptation and annotation-driven label rule tuning. Keep the Kaggle test set out of fitting and diagnostics.
4. For stronger evidence, obtain permitted expert image labels on a locked, adequately sized held-out imaging cohort. New annotations on images already used to train a published component are **not** a fresh imaging holdout for that component. Retrain compared models with the locked groups excluded, or use genuinely unseen permitted studies with justified domain relevance.
5. Report per-class counts/AUC, paired group bootstrap, fold variability, calibration where raw probabilities exist, shared/worsened/repaired pairs, and acquisition-stratified failures. Abstain on unsupported strata/classes rather than changing the macro target set silently.
6. Use Gold58 only as a documented, reused rejection diagnostic. Labels from report extractors are supervision estimates; ROC-AUC against their thresholded labels is not independent image-ground-truth validation.

No experiment in this audit establishes independent clinical ground truth or a private-leaderboard improvement. This blocks confident promotion, not further diagnostics.

## 5. Existing experiments constrain the next queue

| Previously recorded experiment | Measured outcome | Decision supported |
|---|---|---|
| Published tri-reader | Public 0.950, same as N02 | No verified leaderboard gain; retain fallback |
| Series-coherent triplet filtering | N02 Gold delta −0.000305; interval −0.001608 to +0.000860 | Reject promotion of this exact filter; anatomical concerns remain open |
| Equal ViT/CoAtNet/ResNet-50 blend | Gold 0.910772 vs N02 0.925662 | Reject this equal blend, not all model diversity |
| 10% blend of that three-family ensemble into N02 | Delta +0.000023; interval −0.002308 to +0.002048 | Inconclusive, no promotion |
| E04 held-out model fixed 10% blend | Delta −0.000787; interval −0.003546 to +0.001822 | Reject this recipe |
| Uniform supervision weights, paired fold4 | 0.815112 vs control 0.877612; 11 expert studies | This fold regressed; not a completed five-fold candidate result |

These archived experiments share small reused selection cohorts, noisy report supervision or inference-only architectural changes. The plateau protocol should distinguish **supervision mismatch**, **lost acquisition/anatomical information**, and **selection noise/insufficient complementary signal**. A failed blend rejects that blend; it does not falsify every architecture or supervision strategy.

## 6. Independent role reviews

- **Research architect:** Train a new family only after an error stratum and input/supervision hypothesis are identified. Frozen feature probes can screen candidates cheaply before backbone training, but must respect encoder training membership.
- **Medical imaging specialist:** Do not label a ranking error as a missed ligament/meniscal lesion without image review. FOV, sequence tags, obliquity and physical slice spacing remain acquisition questions.
- **Deep learning engineer:** Exact code/checkpoint bindings are recoverable. Restore source identity and arithmetic parity before porting. Measure full-cohort decode, CPU RAM and runtime rather than extrapolating visible three-study speed.
- **Validation scientist:** No independent baseline OOF is available; Gold reuse and report overlap are material. Fold training exclusion and model-selection independence are separate requirements.
- **Independent critic:** Higher Lateral OA reader AUC on 11 positives does not authorize increasing its weight. Better labels or more resolution are hypotheses, not results. The original negative experiments constrain repeat spending.
- **Experiment coordinator:** Complete V00 and CPU diagnostics first; inference ablations second; one controlled pilot training comparison third. No expensive jobs launched here. Cost entries in the registry are scope estimates, not promised execution times.

These reviews are separate reasoning checks performed by one assistant; no sub-agents were spawned in this side conversation.

## 7. Submission readiness and checklist

**Fallback:** retain the original scored notebook/version, checkpoint identities and CSV. A new experiment writes a separate candidate and never overwrites the fallback. The highest verified submission remains 0.950.

Before any new submission: reconcile training/inference changes, verify all model and wheel hashes, require every expected study and target, record decoded input availability and recovery events, check finite outputs and no unexplained defaults, measure the complete permitted test workload and worst-case acquisition load, check offline dependencies and current competition constraints, review validation provenance, and execute the final notebook reproducibly. A runtime cap alone is not a runtime measurement.

- [x] Baseline documented and archived reproduction verified: accepted code, checkpoints, score receipt and measured visible runtime. No fresh GPU reproduction performed here.
- [ ] Independent validation established.
- [ ] A performance bottleneck causally confirmed by a controlled predictive experiment.
- [ ] New model or supervision produces validated predictive information.
- [ ] Ensemble improvement validated on independent evidence.
- [ ] New final submission verified against the full hidden-workload constraints.

## Evidence index

Paths are relative to the repository root:

- `runs/fixed_reader_blend_v3/submission.json`
- `runs/fixed_reader_blend_v3/independent_audit.json`
- `runs/fixed_reader_blend_v3/contract.json`
- `runs/fixed_reader_blend_v3/outputs/fixed_blend_source/{anchor/published_inference.py,reader/knee.py,reader/preprocess.py,reader/infer.py,reader_recovery.py,nartaa_reproduction_guard.py}`
- `models/{convnext_public_reader/inspection.json,nartaa_publication_20261007/checkpoint_inspection.json}`
- `research/new_public/goodpjw_weights/source/{make_labels.py,train.py,knee.py}`
- `runs/colab_n03/{build.json,output/diagnostic_receipt.json}`
- `runs/published_tri_reader/official_result_evaluation.json`
- `runs/window_integrity_20261010/pair_evaluation.json`
- `runs/three_family_ensemble_20261010/result.json`
- `runs/e05_confidence_ablation/{compute_review_control_audit.json,complete_control_blend_review.json,partial_compute_review.json}`
- `runs/e05_kaggle_pair4/training_evaluation.json`

The local archived code/documentation supplies this report's source evidence. Architecture and training provenance attributed to publishers are distinguished above from measurements reproduced by this audit.
