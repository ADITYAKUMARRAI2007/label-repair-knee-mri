# E03 series reader: resource pilot

Prepared and launched 6 October 2026 after official standalone results rejected
E02 (0.890), C12 (0.911) and C13 (0.563). B03 remains pending. This is a new
private training experiment, not another competition submission.

## Hypothesis and limits

E02 has a proven representation defect: adjacent indices in its concatenated
44-slice cache can belong to different acquisitions. Its official score is
close to its independently recomputed Gold58 score. This suggests a useful
training baseline, but does not measure the score effect of correcting the
representation. E03 changes several components together and cannot establish
a causal ablation of the boundary fix.

E03 retains the pinned ImageNet ResNet34 initialization and the fixed 75/25
teacher/JEV logit supervision. It changes the reader to:

- three-slice windows contained in one of the five Raptor acquisition slots;
- four sampled windows per present slot in training, eight fixed windows in evaluation;
- explicit acquisition-slot and relative-position embeddings;
- two transformer layers and separate attention pooling for each finding;
- full encoder adaptation and EMA;
- spatial and intensity augmentation shared within an acquisition;
- one model process on GPU 0, with bounded mmap page release and no DataParallel.

The allocation may expose two T4 devices; the pilot deliberately uses only
one. Allocation costs must be considered before full training. It is not a
copy of the ConvNeXt reader already in B03, although its study-head structure
uses ideas from that published reader. Complementary errors and sufficient
standalone strength remain unproven.

## Pilot contract

Exact accepted job: `adityakumarrai0401/rsna-knee-e03-series-resource-pilot`,
version 1. Kaggle returned this slug from the title; the local builder metadata
was corrected to match it without pushing another version.

The pilot performs 300 optimizer steps at 288 pixels, two studies per batch,
on non-Gold studies only. It writes no submission and computes no Gold58 score.
The inherited I/O helpers and all three input label/initialization hashes are
pinned in the source-build receipt. The checkpoint is marked `pilot_only` and
must not be used as a trained competition model or silently resumed as the full
training experiment.

It checks finite logits/loss, bounded AMP overflow, host RSS and peak GPU
allocation every 25 steps, real evaluation token count, invariance to pixels
inside a masked missing acquisition, and restoration from the written EMA
checkpoint. Proposed operational gates are:

- host RSS below 20 GiB and peak allocated GPU memory below 13 GiB;
- RSS growth below 2 GiB between steps 50 and 300;
- estimated full training epoch below 30 minutes;
- exact checkpoint replay and absent-acquisition invariance within 1e-6.

Epoch extrapolation is a resource estimate, not a guarantee of full-run stability
or accuracy. The local window checks cover 6,600 cases across 1,100 presence and
partial-decode masks, including 122,460 selected windows. They establish only
the selector's boundary/mask behavior. GPU runtime and model quality require
the job artifacts.

## Before a full run

1. Collect the existing pilot's terminal status and independently inspect its
   manifest, time/memory samples and checkpoint receipt.
2. Freeze report-duplicate groups and an independent weak-label validation split;
   exclude every Gold58 study and its report group from training. Report grouping
   does not establish patient or scanner independence.
3. Freeze the epoch count, learning-rate schedule, final EMA checkpoint rule and
   augmentation/input contract before training. Save full resumable state after
   each completed epoch, and record exact train/validation identities.
4. Independently reproduce the final Gold58 AUC and compare paired predictions
   with E02, including uncertainty and per-target behavior. Do not fit blend
   weights to the 58 cases.
5. Apply the medal plan's validation, official scoring, complementarity and
   runtime gates. A completed pilot alone qualifies for none of them.

The expected competition score is unknown. A 0.95–0.96 or medal claim would be
unsupported at this stage. The remaining competition submission slot stays
reserved while B03 is evaluated.

## Observed pilot result and full training launch

The pilot completed 300 steps in 244.56 script seconds with zero AMP-skipped
updates. Host RSS peaked at 2.107 GiB in its recorded samples, GPU allocation at
1.953 GiB, and RSS growth from step 50 to 300 was 0.0308 GiB. Both the masked
acquisition perturbation and checkpoint restoration had exactly zero maximum
output difference. The 361,661,296-byte downloaded state matches the recorded
SHA-256 `3528f31bfa3753e7acee7c8886b198f0e545e331ddcca711d4dfe4d1a7d1ed90`.
The resource audit is `runs/E03_pilot/resource_audit.json`; these measurements
do not establish accuracy or longer-run stability.

`scripts/freeze_e03_split.py` independently constructed and checked the split:
3,939 training, 406 weak-validation, 58 expert-validation, and four weak studies
excluded because their report group contains an expert study. The split has
4,073 report groups; its largest group has 59 studies. Reversing input order
leaves it unchanged. No Gold target was used to balance the groups. The frozen
split SHA-256 is
`3e8a12dfa81a764b1af7488d009b65014b45c82991322fe7599fc789f36a8811`.

Full trainer `adityakumarrai0401/rsna-knee-e03-resnet-series-train` v1 was accepted
and reports RUNNING. Its startup log confirmed the exact split hash and counts.
It starts from clean pinned ImageNet weights. The fixed recipe is 14 epochs,
one epoch of linear learning-rate warmup, cosine decay to 1%, full-encoder AdamW
with 1e-4 encoder / 5e-4 head rates, and final EMA only. Every completed epoch
writes a full resumable state; weak validation is a training-health measurement,
and Gold58 is evaluated only after the fixed final epoch. A pilot state cannot
be resumed as a full training state.

The authenticated quota endpoint reported 0.23 hours used and 44.77 hours
remaining out of 45 hours before this full launch. The end-to-end pilot rate
extrapolates to 26.16 minutes per training epoch for 3,939 studies, or 6.10 hours
for gradient updates alone. The provisional full-run estimate is 6–8 hours
including validation/checkpoint overhead; startup/queue variability and longer
memory behavior remain uncertain. Competition scoring would be additional.

The final independent auditor is `scripts/audit_e03_training.py`. It checks
artifact hashes, the exact frozen split, all 14 completed epochs, exact Gold58
and weak-validation identities, finite probabilities, tie-aware expert AUCs,
and a paired E02 comparison. It does not submit, fit blend weights, claim a
public score or automatically qualify the model for a blend.

The first 100 full-training batches completed in 90.81 seconds, with host RSS
2.182 GiB and peak GPU allocation 2.046 GiB. Throughput needs a completed epoch
before a tighter finish-time estimate is justified; roughly 7–9 hours is a
conservative planning range using this early sample and validation overhead.

## Upstream supervision qualification

An independent diagnostic on the teacher soft labels yields 0.996299 macro AUC
on Gold58. This is the quality of supplied labels, not the new image model's
score. The teacher dataset's authenticated metadata has an empty description.
The author's accessible public `rsna-knee-public941-control` inference notebook
does not establish how these labels were generated or whether its upstream
teacher excluded Gold58 during training. The metadata and source-search receipts
are saved under `research/new_labels/teacher` and `runs/E03_contract`.

E03 excludes Gold58 and its report groups from its own gradient updates. That
does not prove independence of upstream supervision from Gold58. A near-perfect
teacher-label diagnostic alone proves neither leakage nor an independently
validated image model. Treat E03's eventual Gold58 score as a coarse screen,
record this upstream limitation, and require the official standalone and
complementarity gates before any blend. This qualification does not change the
frozen training recipe or an already running notebook.

## Depth span and pixel reconstruction verification

The authenticated descriptions for both attached corpus parts specify
15–85% of the native series depth, 44 cached slices with widths 12/10/8/6/8,
140 mm center crops, 336-pixel uint8 images, and fluid-sensitive acquisition
preferences. Both description snapshots are in `research/e03_corpus`.
The extension's original `extra_meta.json` reports 1,207 built studies, zero
failed studies and 13 included expert studies; it does not contain a generator.

E02's old inference decoder used 6–94%, so its comment claiming that the span
matched these corpora was wrong. E03 training reads the actual cache and needs
no change; its future inference must use the documented 15–85% geometry. The
train/inference score effect of the old mismatch remains unmeasured.

Staged `inference/e03_dicom_reader.py` preserves the acquisition selection,
modality LUT, MONOCHROME1 inversion, per-selected-series 2/98-percentile scaling,
millimeter crop and area resize inherited from the public reader, with a
15–85% span. These steps beyond the dataset description are still hypotheses
until compared with actual cached pixels. The decoder records header failures,
missing metadata fallbacks, selected-slice attempts, decode failures and constant
slices, plus missing/empty/unreadable acquisitions. A study without any nonzero
decoded slice fails visibly. A future window selector additionally rejects
studies without a series-contained three-slice window.

Private CPU job `adityakumarrai0401/rsna-knee-e03-cache-pixel-parity` v1 was
accepted. It reconstructs all 58 expert studies and one example per otherwise
unrepresented acquisition-presence pattern, covering both corpus parts. It
compares both documented 15–85% and old 6–94% spans against stored pixels and
masks, without computing predictions, fitting to targets or spending a
competition submission. Exact matches are the initial parity gate; any
differences require diagnosis rather than a passing claim from metadata alone.
Its source receipt is `runs/e03_cache_parity_source.json`.

The CPU job subsequently completed in 231.39 script seconds. The independent
auditor `scripts/audit_e03_cache_parity.py` verified **59/59 exact pixel and mask
matches** at 15–85%: all Gold58 plus one nonexpert presence-pattern example,
four acquisition-presence patterns, and both corpus parts. All 2,476 selected
slice attempts decoded with zero selected-slice or header failures. SHA-256s
of each reconstructed volume match its cached counterpart. The old 6–94% span
has mean absolute error 25.227 on the 0–255 pixel scale across these studies;
the minimum study error is 14.242. This measures pixel distribution mismatch,
not an AUC effect. The passing receipt is
`runs/E03_cache_parity/pixel_parity_audit.json`.

At the next training observation E03 had completed ten epochs and reached
epoch 11, batch 1,400. The first epoch took 1,678 seconds; subsequent epochs
including weak validation took approximately 297 seconds after cache warming.
This invalidates the earlier cold-cache extrapolation of 7–9 hours. Training
remains bounded near 2.2 GiB host RSS and 2.56 GiB peak GPU allocation, and
epoch-10 weak-label threshold AUC is 0.916229. This is a training-health metric
against supplied soft labels, not expert Gold58 or public leaderboard AUC.
The fixed epoch-14 EMA rule remains unchanged. Recent timings imply roughly
15–20 further minutes from the epoch-11 observation if throughput holds, plus
final artifact saving/download and independent evaluation.

## Final result: rejected before competition scoring

E03 v1 completed all 14 fixed epochs in 5,613.63 script seconds (93.56 minutes).
The first cold-cache epoch took roughly 28 minutes and later epochs about five
minutes, so the early cold-cache estimate overstated its full runtime.

`scripts/audit_e03_training.py` independently verified the complete artifact
hashes, expert/weak identities, frozen report-group partition, all 14 completed
epochs and tie-aware Gold58 AUCs. Final expert macro AUC is **0.881107253950169**.
The paired point difference from E02 on the same expert studies is **−0.0140985**;
E03 is lower on 10 of 12 targets. Lateral OA (0.7911) and Synovitis (0.7766) are
its weakest expert targets. Final weak-label threshold AUC is 0.920066, which
does not establish clinical or hidden-test strength. The checkpoint SHA-256 is
`cbc5bd00fe1ce4ee088cec842535c3fecba5e271a6b67b4655e82403d67525de`.

There were four recoverable AMP-skipped updates during training, within the
frozen ten-overflow limit. Host/GPU memory stayed bounded. A transient local
DNS failure interrupted output retrieval; the same completed run was retrieved
with streamed atomic downloads and all manifest checksums verified. The
download failure did not stop or restart training.

This result fails the preregistered minimum expert AUC of 0.90. **No competition
submission or blend will use E03.** The independent audit is
`runs/E03_training/e03_training_audit.json`; output-download evidence is
`download_receipt.json` in the same directory. This is a model-strength rejection,
not an operational failure or a hidden-test score. The changed architecture,
sampling, training set and augmentation were not a controlled ablation, so
their individual causal effects are unresolved. Do not choose another epoch
or target-specific blend after observing these expert labels.

The useful reusable results are the measured 15–85% pixel contract, instrumented
DICOM reader, exact series-contained selector, bounded single-process training,
and completed-epoch resume design. A stronger pretrained encoder and proper
expert-training folds can be researched next; neither is a demonstrated score
gain. B03 still needs official evaluation before another competition submission.
