# Phase 5B — Published 0.954 Dual SOTA notebook audit

Date: 2026-10-11  
Status: source and run-metadata audit complete; exact V3 reproduction and eligible accuracy comparison blocked.

## Executive finding

The Kaggle page identifies `jenillangaliya/rsna-knee-lb-0-954-dual-sota-ensemble`, kernel id `138049049`, with V3 as the version associated with its displayed best public score of 0.954. Kaggle reports V3 completed on a T4 with runtime 203.72 seconds. This is the author's reported/public result, not a reproduction or score from our account.

V4 completed in 10.96 seconds but has no output files, no accelerator metadata, and its only saved log records notebook-to-HTML conversion. V4 is a rendered source save, not a full inference run. Its short runtime cannot validate hidden-test throughput.

The V4 source reuses N02's exact CoAtNet checkpoint (SHA-256 below). Its genuinely new image classifier is the Compact v1 four-fold ConvNeXt-Small reader. The listed OrthoFoundation dataset provides code and a hashed anatomy anchor used to place physical/anatomical crops; the manifest selects a `convnext_small` classifier, not an OrthoFoundation-L image encoder. The optional OFL-L branch is conditional on `backbone == 'ofl_l'` and is inactive for this bundle.

Thus the notebook is a plausible source of complementary preprocessing and classifier signal, but no eligible data here establishes that it corrects N02 errors or improves macro-AUC. Test prediction differences alone would not establish that.

## Source identity and execution evidence

| Item | Verified evidence |
|---|---|
| Public kernel | `jenillangaliya/rsna-knee-lb-0-954-dual-sota-ensemble` (`138049049`) |
| Current retrieved version | V4 source; SHA-256 `d7ea572189f1fb8f32429cd1797bdc178d80b7ba0ec27dd030002f3828df6bc0` |
| V3 run | Run id `357198699`, COMPLETE, T4, 203.7237 s, page metadata records best public score 0.954 |
| V4 run | Run id `357241354`, COMPLETE, 10.9616 s, output file list empty |
| V4 saved log | Only `nbconvert` converting `__notebook__.ipynb` to HTML and writing `__results__.html` |
| V3 historical source/output | Kaggle's history endpoint exposed run metadata and artifact names, but the source revision and output downloads were not retrievable in this session. Artifact-fetch approval timed out twice. |

The retrieved V4 notebook has 8 cells with null execution counts and no cell outputs. Its exact source is saved at [the retrieved notebook](published_0954_latest/rsna-knee-lb-0-954-dual-sota-ensemble.ipynb); Kaggle metadata is at [kernel-metadata.json](published_0954_latest/kernel-metadata.json). The V4 output log is at [published_0954_v4_outputs](published_0954_v4_outputs/rsna-knee-lb-0-954-dual-sota-ensemble.log).

The V3 run metadata shows that a GPU run completed and that prediction artifacts were created. It does not show that V3 ran the complete hidden workload. The 203.7-second runtime therefore is not accepted as proof of full-test runtime.

## Actual V4 ensemble and reader use

### Stage 1: Raptor CoAtNet-384

- Checkpoint: `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`
- SHA-256: `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`
- Architecture in source: `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, with the published Raptor attention/MIL head.
- Inputs: five plane/sequence slots totaling 96 selected slices; 3 adjacent slices per 2.5D window; 384px images; approximately 140 mm crop; 42 windows per study at inference, with mirrored-anatomy TTA.
- Rank-transforms each class's predictions before final blending.
- The code actually loads one selected CoAtNet checkpoint. It does not load both the 0.949 and 0.945 SWA weights as separate ensemble members.

### Stage 2: Compact S6 ConvNeXt-Small

- Bundle id: `209a32c8b50a6b1c`; arm `E_lrprior`.
- Manifest: `convnext_small`, 384px input, 352px crop, four folds, 16 training epochs, three adjacent slices, physical spacing 3.5 mm, report-derived targets, 4,344 pooled studies, and 58 Gold studies excluded from this training run.
- It uses four fold checkpoints `E_lrprior_s0_f0.pt` through `f3.pt`; their hashes are recorded in [the downloaded manifest](published_asset_preflight/asset_0/manifest.json).
- A side classifier is also loaded. The source uses the `anatomy.pt` anchor weights and anatomy-aware physical views. Because the bundle backbone is `convnext_small`, the conditional branch that rebuilds an OFL-L encoder from `rsna-knee-ofl-ssl-weights` is not taken.
- The published fixed blend is 0.70 CoAtNet rank-percentile + 0.30 Compact-reader rank-percentile, per class. Failure handling falls back to the surviving component's rank; both-fail rows receive 0.5. The V4 source also replaces any final nonfinite value with 0.5, a silent fallback that should be hardened before adopting it.

### Checkpoint lineage and overlap with N02

The CoAtNet filename and hash exactly match the N02 anchor checkpoint. N02's source preservation receipt confirms the same hash. Therefore this branch is fully redundant with N02's CoAtNet model weights and cannot be counted as a new model.

N02 also has three ConvNeXt reader checkpoints from a separate reader family. Those checkpoints differ from this four-fold Compact S6 ConvNeXt-Small bundle. S6's different architecture scale, report targets, physical 3.5 mm spacing, side handling, and anchored/local view set may supply distinct signal. However, the exact N02 ConvNeXt training membership and independently held-out paired predictions are unavailable, so redundancy and complementarity cannot currently be measured on a defensible label set.

The compact manifest says training used report-derived targets and that official labels were not read by its trainer. Its public README says the dataset was restored but provides no clear explicit dataset/model license statement in the retrieved files. The SWA package explicitly allows research/competition use of the learned weights while retaining OAI's controlled-access data restrictions. The Compact asset's permissions still need confirmation from its Kaggle license metadata before reuse in a submitted candidate.

## Reproducibility and candidate readiness

| Requirement | Result |
|---|---|
| Exact V4 source/hash | Retrieved and saved |
| Exact V3 source | Not retrieved; V4 is not a substitute for historical V3 source |
| Checkpoint identity | CoAtNet identity/hash and Compact manifest/fold hashes verified from public manifests/source |
| Complete local run | Not run: local environment has no Torch/GPU and does not hold the required full checkpoints/DICOM test inputs |
| Full hidden-test runtime | Not established; V3's 203.7 s and V4's render-only 10.96 s do not establish it |
| Submission format and finite-output checks | V4 source contains checks; V3 output CSV was not retrieved, so no independent output-format receipt was verified |
| Submission candidate | None created; no Kaggle submission made |
| N02 preservation | Preserved; source SHA-256 remains `ca4dbebfc379e45064d6cd7441c281d4f05114d54f426d68c0a4a34e653f2032` |

An exact unchanged reproduction is not claimed. A follow-up should first retrieve the historical V3 source and output receipts, confirm the Compact asset's terms, and establish enough inference budget for an actual full-workload preflight before spending GPU time.

## Complementarity and controlled candidate comparison

The available V3 artifact names include raw CoAtNet predictions, Compact-reader predictions, final submission, and blend receipts, but the downloads were blocked by the Kaggle access-review timeout. Consequently, no test prediction correlation or rank-disagreement values are reported. Even if downloaded, hidden-test prediction disagreement would measure diversity only, not accuracy gain.

No shared eligible OOF predictions exist for N02 and the published S6 reader. N02 checkpoint training membership is not verifiable, and the 58 expert-labeled studies have been reused for selection. The recently completed Phase 5 regional ConvNeXt has grouped OOF results on weak report proxies, not labels independent of report-derived supervision; its Gold58 diagnostic is reused. It cannot serve as an independent comparison to the S6 reader.

Current candidate table:

| Candidate | Evidence | Decision |
|---|---|---|
| A. N02 unchanged | User-reported public 0.950 / rank 415; safe existing baseline | Retain fallback |
| B. Published 0.954 blend unchanged | Author's best public score; V3 run complete but exact source/output not independently reproduced | Do not promote yet |
| C. N02 + Compact S6 only | S6 is the only new learned classifier in the public blend; no eligible paired accuracy comparison | Research candidate only |
| D. N02 + complete published blend | Would count N02's CoAtNet twice unless it is removed/reweighted; no OOF evidence | Reject as currently specified |
| E. N02 + minimal S6 reader | Most informative later comparison if paired OOF is established and asset terms are confirmed | Preferred next test, conditional |

No blend-weight tuning was performed. No candidate has demonstrated improved generalizable macro-AUC, and no Kaggle submission was made.

## Phase 5 work status

The grouped five-fold Phase 5 regional/global ConvNeXt experiments completed. On weak report-proxy targets, regional was 0.82675 macro-AUC and a matched global model was 0.82107 (paired difference +0.00568, 95% group-bootstrap CI +0.00354 to +0.00780); a fixed equal probability blend was 0.82849. These are proxy-label results, not clinical truth or competition scores. On reused Gold58, the fixed blend was 0.85628 versus 0.85926 regional-only (paired difference −0.00298, CI −0.01200 to +0.00473). N02 lacks verifiable OOF membership for a valid paired comparison.

The Phase 5 inference notebook is technically checked on the three public test studies, produces finite predictions without defaults, and agrees with the public replay exactly. But the measured p95 of 17.28 seconds per study gives a conservative 12.48-hour projection at an assumed 1,300 studies, beyond the 9-hour competition cap; full hidden-test execution was not run. This Phase 5 model is not submission-ready and does not establish a win over N02.

## Decision and next executable step

**Decision:** Treat the public 0.954 as a promising but unverified candidate. The published pipeline duplicates N02's CoAtNet and adds a differently trained, anatomically anchored ConvNeXt-Small four-fold reader. It may help N02, but the necessary evidence is missing to claim that it does. The OrthoFoundation asset does not act as the classification encoder in this selected configuration.

**Next step, before GPU work:** retrieve V3 historical source and public output receipts (or ask the notebook author for the pinned V3 source/assets if Kaggle's run files remain inaccessible); confirm Compact v1 reuse terms; then produce an eligible grouped OOF set for the S6 reader and a matched N02 comparison. If that evidence supports complementary class-level ranking, run only the minimal S6 + N02 candidate and preflight full hidden-test runtime. Preserve N02 unchanged and submit only after review.

## Recorded blockers and non-results

- Kaggle artifact downloads timed out in automatic network approval twice; no downloaded V3 test CSV, receipt, or historical notebook source is available locally.
- No independent label set supports an accuracy claim for the S6 reader or any proposed blend.
- No full hidden-test runtime or GPU-memory run was completed for the published notebook.
- The Compact v1 asset's explicit license terms remain unverified from the local package.
- No model-training job or competition submission was launched for Phase 5B.
