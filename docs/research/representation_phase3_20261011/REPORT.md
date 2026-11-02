# RSNA Knee Phase 3 — measured representation complementarity

**N02 and existing checkpoints preserved. No competition submission.** This experiment compares generic ImageNet CoAtNet-0 and ConvNeXt-Tiny features, not the MRI-adapted N02 components or a reproduced leaderboard score.

## Protocol and provenance

All 4,407 studies were assigned to 4,072 conservative report/image groups. Three independently shuffled five-fold assignments were frozen before extraction; all A/B/C variants used identical held-out groups. Every study received one held-out prediction per repeat. Validation groups were excluded from head fitting, feature centering/scaling and parameter selection. There were zero supervised encoder updates, and frozen encoder buffers were checked unchanged.

The existing five-study gold/weak report cluster stayed together. A further pair of weak studies had identical full cached-volume hashes across different report groups; both complete groups were merged before any features or scores were observed. Original and amended manifests are preserved. The pair has different reports and distinct selected series UIDs, with conflicting hard MCL/Effusion targets; native pixels are needed to distinguish image duplication from a cache mapping error. Patient identity and near-duplicate imaging remain unverified.

The checkpoints are publisher-documented ImageNet-1k weights: [CoAtNet](https://huggingface.co/timm/coatnet_0_rw_224.sw_in1k) and [ConvNeXt](https://huggingface.co/timm/convnext_tiny.fb_in1k), with pinned revisions and verified SHA-256 hashes. No published RSNA-trained checkpoint was used. Full ImageNet membership manifests are not available.

Both encoders used the same owned, hash-verified wide-depth MRI cache, 140 mm crop, 224 pixel resized input, five acquisition slots and three deterministic adjacent-slice windows per slot. Each used its published intensity normalization. Slot embeddings were mean pooled with missing-slot masks. A/B used separate fixed alpha=1 weighted ridge heads; C concatenated training-standardized feature blocks with equal total feature scaling and the same regularization. Scores are continuous ridge outputs, not calibrated probabilities. No hyperparameter search was conducted.

## Macro-AUC and paired uncertainty

Reported point estimates average the three repeated OOF AUCs. The 2,000-draw paired bootstrap resamples complete report/image groups, applies the same draw to A/B/C and all repeats, and recomputes max(A,B) in every draw. Repeated splits share cases and are not independent cohorts. Intervals are conditional on fitted heads and frozen labels.

| Evaluation cohort | A: CoAtNet | B: ConvNeXt | C: Fusion | C − max(A,B), 95% CI |
|---|---:|---:|---:|---|
| 4,349 weak studies — report agreement proxy | 0.763592 | 0.767077 | 0.774120 | +0.007042 [+0.004719, +0.009004] |
| 58 reused expert studies — diagnostic only | 0.755421 | 0.727298 | 0.751264 | -0.004157 [-0.021860, +0.011478] |

Weak-cohort deltas by repeat: +0.006856, +0.007150, +0.007121.
Gold diagnostic deltas by repeat: -0.001611, -0.005443, -0.005418.

Weak targets are four report-extractor medians thresholded at 0.5 for AUC; their source agreement weights are not calibrated label certainty. Public label-prompt development may have reused Gold. Gold58 has repeatedly informed prior research. **Neither cohort supports an independent clinical or private-leaderboard improvement claim.**

## All 12 abnormalities

| Target | Weak A | Weak B | Weak C | Gold A | Gold B | Gold C |
|---|---:|---:|---:|---:|---:|---:|
| ACL | 0.75216 | 0.77317 | 0.77620 | 0.73366 | 0.72141 | 0.74183 |
| MCL | 0.69375 | 0.68992 | 0.70034 | 0.67423 | 0.52834 | 0.63341 |
| Medial Meniscus | 0.75917 | 0.76733 | 0.77183 | 0.73037 | 0.75361 | 0.72837 |
| Lateral Meniscus | 0.69218 | 0.69430 | 0.69921 | 0.73623 | 0.69731 | 0.71925 |
| Medial OA | 0.82682 | 0.83109 | 0.83612 | 0.82894 | 0.89561 | 0.88217 |
| Lateral OA | 0.78023 | 0.79044 | 0.79376 | 0.89426 | 0.76273 | 0.83559 |
| PF OA | 0.77429 | 0.78153 | 0.78922 | 0.75547 | 0.78893 | 0.78764 |
| Effusion | 0.80111 | 0.79583 | 0.80527 | 0.86460 | 0.83644 | 0.87122 |
| Synovitis | 0.84770 | 0.84021 | 0.84931 | 0.70251 | 0.59578 | 0.64118 |
| Baker's | 0.73866 | 0.73702 | 0.74874 | 0.78744 | 0.83937 | 0.83213 |
| Contusion | 0.73505 | 0.73273 | 0.74033 | 0.70040 | 0.67656 | 0.69006 |
| Fracture | 0.76199 | 0.77139 | 0.77910 | 0.65694 | 0.63148 | 0.65231 |

Per-class positive/negative counts, 95% intervals, model correlations and top-20% priority disagreements are in analysis/per_class_comparison.csv. Fold-level undefined class AUCs are explicitly missing; no incomplete-class score is called twelve-class macro-AUC.

Within-fold check on weak targets: fusion exceeds the stronger single model in 15/15 folds with all12 class AUCs defined. Folds share training data and are not independent experiments.

## Decision and next hypothesis

**Accept fusion for further representation research on report agreement; do not promote it to a competition candidate.** The fixed fusion passed the predeclared weak-proxy gate: positive delta in every repeat and a positive lower paired interval bound.

The images, windows and series selection were identical, so the experiment supports complementary frozen features under this protocol. It does not isolate anatomy: architecture, pretraining recipe and normalization differ.

**Next controlled experiment, proposed and not launched:** remove both coronal-slot feature blocks from all A/B/C variants, refit the same fixed heads on the same grouped splits, and measure the change in class-specific fusion advantage. Reuse cached features; do not re-extract images or search blend weights. This tests whether coronal information contributes to complementarity, while checking acquisition/site confounding. An untouched permitted expert imaging holdout remains required for clinical confirmation.

**Priority before that representation ablation:** investigate supervision/proxy mismatch. Fusion did not outperform the stronger single model on Gold in any of the three repeats; its paired Gold interval crosses zero, so neither benefit nor harm is established. First verify native images for the conflicting duplicate-cache pair and review the linked reports blinded to predictions. This distinguishes a cache mapping defect from inconsistent supervision without changing ensemble weights. The pair alone cannot explain the macro-AUC gap.

Then preregister one class-specific report extraction correction only if the audit supports it; compare corrected versus existing targets using cached features, identical fixed heads and the same group exclusions. Do not use held-out Gold to choose the correction. Obtain untouched expert imaging truth before promoting a candidate.

Lateral OA must be judged from this new paired protocol, not from the previous 0.8549/0.7756 Gold comparison. Its measured A/B/C results are above; differences between generic probes and N02 readers cannot validate the published-checkpoint complementarity claim.

## Measured resources

| Operation | GPU seconds | Seconds/study | Peak allocated GiB |
|---|---:|---:|---:|
| coatnet features | 147.7 | 0.0335 | 0.53 |
| convnext features | 133.9 | 0.0304 | 0.70 |
| All 45 heads | 4.2 | — | — |

Measured job time: 808.7 seconds (0.2246 allocated GPU-hours) on Tesla T4; Torch 2.11.0+cu128, timm 1.0.29. These costs exclude platform setup and native DICOM decoding. Full hidden-test inference runtime is unverified.

## Execution record and blockers

The acquisition-fallback experiment was not repeated. Setup/download/build/monitoring failures and any GPU runtime failures are retained in the experiment registry. The initial oversized launch request created no run (confirmed HTTP404); the manifest was losslessly compressed for the accepted package without changing its hash.

A strict bit-parity diagnostic on the duplicate-cache pair failed: relative feature differences were 0.0554% for CoAtNet and 0.0305% for ConvNeXt, with maximum held-out ridge-score difference 0.000389. Slot masks match exactly. These small differences are compatible with FP16 batch-dependent numerical variation, but the cause was not independently established. Batch size4, pinned software and cached feature files are preserved; full execution replay was not performed.

Remaining blockers: untouched expert imaging truth, trusted patient linkage, independently verified report-label development, and competition-scale raw-DICOM runtime. Published model scores are not reproduced results. No leaderboard score is predicted.

Artifacts: manifest.json, provenance_audit.json, phase3.ipynb, output/phase3_results/{A,B,C}_oof_repeat{0,1,2}.csv, analysis/per_class_comparison.csv, analysis/fold_comparison.csv, analysis/failure_cases.csv, analysis/results.json, analysis/gpu_usage.json and experiment_registry.json.
