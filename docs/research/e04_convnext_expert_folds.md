# E04: owned ConvNeXt reader with expert folds

Prepared 7 October 2026. This experiment has no measured image-validation or
competition score yet. It follows E03's independently verified 0.8811 rejection.

## Frozen inputs and validation

`scripts/freeze_e04_contract.py` independently reconstructs the earlier T03
report-group split using CSV and NumPy. Its exact fold-map SHA-256 is
`62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357`.
The 4,407 rows divide into 882/882/881/881/881 studies and
12/12/12/11/11 expert cases. All 4,073 report groups stay within one fold.
This grouping does not prove patient or site separation.

E04 weak targets are a fixed cellwise median of Pilkwang V2, Lixin soft labels,
Steven V4, and Dread soft labels, clipped to 0.01–0.99. All weak cells have four
finite source values. Confidence weights are `clip(1 - range, 0.25, 1)`;
official expert truth receives weight 2 inside training folds only. No image
teacher, JEV per-target swaps, Rayan calibration, or Yunus expert overwrites
enter E04's training targets or weights. Historical Rayan/Yunus labels are
used only to reproduce the previously frozen fold allocation.

JEV's shipped README explicitly describes selecting per-target winners on
Gold58, member weights on Gold67, and choosing the global blend using those
scores. It is not an independent label source for a pristine Gold58 claim.
Rayan's report calls Gold diagnostic, but its full calibration procedure is
not independently available here. E04 avoids that calibration rather than
asserting it is fold-safe. Public prompt development for the four retained
families is still not proved independent of expert cases. Gold has also been
used in previous experiments. Complete expert OOF remains a coarse rejection
gate, not an unbiased estimate of a medal score or proof of no overfitting.

`runs/E04_contract/e04_contract.json` records the exact input hashes, family
recipe, target order, groups and folds. Its SHA-256 is
`88a7e801945c4cc5145cd3ece2dced031d662decd9588bcb0cf3a816e088272f`.
Runtime training reconstructs these arrays from already available public
Kaggle datasets and requires identical target, weight and fold hashes.

## Verified generic pretrained weights

The [official model card](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k)
specifies ConvNeXt-Small, ImageNet-22k pretraining followed by ImageNet-1k fine
tuning, 768-dimensional embeddings and Apache-2.0 licensing. This is generic
pretraining, not the failed C12 knee checkpoint. The model file was downloaded
through the Hugging Face CLI at revision
`fe6b358aa409414cdb522ae3d43a2c9922f7cc1b` and verified against the Hub LFS SHA:

`2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9`

The file is 200,928,946 bytes with 344 safetensors entries. Configuration and
model-card receipts are in `research/E04_pretrained/convnext_small/`.
File identity is verified; strict architecture loading is a separate GPU gate.
The [official timm implementation](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py)
provides classifier reset and gradient checkpointing used by the staged reader.

## Owned wide cache: launched, no accuracy claim

Two private CPU jobs were accepted:

- `adityakumarrai0401/rsna-knee-e04-wide-cache-base` v1: 3,200 studies,
  expected volume file 15,895,756,928 bytes.
- `adityakumarrai0401/rsna-knee-e04-wide-cache-extension` v1: COMPLETE, 1,207
  studies, volume file 5,995,680,896 bytes. Small-output audit PASS: all IDs,
  masks and study receipts match their checksums; 50,978 selected slices have
  zero header/slice failures or physical-position/spacing fallbacks; all 13
  expert wide-pixel references present in this part match. This does not audit
  the still-running base part or independently download/hash the 6-GB volume.
  Full mounted-volume verification remains required before training.

The jobs preserve the published corpus study order and five 12/10/8/6/8
acquisition slots. The unchanged verified DICOM reader reconstructs the wider
6–94% span at 336 pixels and a 140-mm physical crop. Future full training and
inference must both use this wide contract; E03 trained on 15–85% instead.

The builder embeds independently recorded wide-span pixel SHA references for
all 58 expert studies plus the additional acquisition-presence pattern case.
Workers reject decode/header failures, absent valid three-slice windows and
reference mismatches. Outputs require complete identity, ordered UID hash,
every-study decode receipts, file checksums and all expected reference matches.
No score is inferred from wider sampling or pixel agreement.

Four spawned CPU workers and at most eight outstanding results bound retention.
The writer flushes and releases its memory-map pages every 32 studies. Initial
logs reached 256 base and 384 extension rows with parent peak RSS about
0.213 GiB; this is parent-process evidence, not total worker memory or a
completed-cache audit. Complete artifacts are required before full training.

## GPU resource pilot: complete; memory pass, runtime redesign required

`scripts/build_e04_pilot.py` compiled a private resource-only GPU script. Version
1 of `adityakumarrai0401/rsna-knee-e04-convnext-resource-pilot` completed all
300 updates in 427.38 seconds inside the script. It uses
the existing published cache, whose tensor shapes and dtypes match the new
wide cache. It excludes all expert cases from its updates and excludes the
entire fold-0 report groups. It never scores accuracy, submits predictions, or
provides weights for the full trainers.

The planned probe is 300 steps, one model on one T4 GPU, batch size two,
288-pixel inputs, four windows per acquisition in training and eight in
evaluation, full ConvNeXt encoder adaptation with gradient checkpointing,
study transformer, finding attention and EMA. Backbone/head learning rates
are 3e-5/3e-4 with AdamW weight decay 0.025. It verifies strict public pretrained
loading, finite gradients/outputs, bounded RSS, missing-acquisition invariance,
checkpoint replay and estimated total training time. A projected five-fold
14-epoch workload above 17 hours requires a resource redesign before launch.
Version 1 starts its per-step timer after DataLoader returns the batch, so its
numeric projection excludes batch-loading time as well as validation. Treat
that field as compute/optimizer time, not a full training ETA. The heartbeat's
elapsed seconds include batch loading. Before approving the remaining folds,
measure complete epoch wall times and distinguish first-epoch cold I/O from
later epochs. A reported pilot resource PASS alone cannot prove the full
five-fold workload fits the remaining allocation.

The independent resource audit verifies all 300 steps, zero AMP skips, GPU
peak allocation 2.158 GiB, peak sampled RSS 2.586 GiB, and only 0.018 GiB RSS
growth after step 50. Missing-acquisition invariance and saved-checkpoint replay
both have exact error zero. Frozen inputs and strict generic pretrained loading
passed at runtime. The large pilot checkpoint is not downloaded or reused;
the source-controlled execution verified its save/reload, but the local audit
does not independently claim its 814-MB byte identity.

The original compute projection is 35.21 hours for five sequential 14-epoch
models, so the pilot correctly requests a resource redesign. Median steady
heartbeat wall time after step 50 is 0.8545 seconds per step. Extrapolating
that rate gives 29.29 hours for five models before final validation and
checkpoint I/O. Three paired T4 jobs (two folds, two folds, one fold) would
take about 17.58 hours only if each model retained that rate without CPU/I/O
contention. This is a hypothesis requiring a paired resource test, not a
full-training launch authorization or a committed ETA.

The before/after quota snapshots show 453.806 seconds consumed by this kernel,
with roughly 19.07 allocated GPU hours remaining. Its script/runtime log spans
about 453 seconds. This observation uses one active model; verify actual quota
accounting and available devices with two active models before relying on
paired-job arithmetic. Preserve complete checkpoint rules and all five expert
folds when changing execution; do not reduce the objective to fit this probe.

Current quota evidence is saved in `runs/e04_accelerator_quota_after_pilot.json`;
roughly 19.07 GPU hours remain from the allocated 21-hour pool. The account's existing
pay-to-scale setting is enabled. Do not purchase extra compute or knowingly
launch work beyond the allocated budget without explicit user authorization.

After the pilot and cache audits pass, freeze the full trainer contract and
train five fresh, separate backbones. Each expert case must be predicted only
by its held-out model. Use fixed final-epoch EMA, not expert-AUC checkpoint
selection. Require all 58 expert OOF predictions, independent metric audits,
stable target/fold behavior and the existing 0.90 Gate B before hidden inference.
Public standalone strength and measured complementarity still govern blending.

## Upload approval outcome

Automatic approval review rejected an upload of the combined weights, labels
and fold package because competition-derived records were included. That
package was not uploaded. Do not retry that payload or move it through another
upload channel without explicit authorization.

The safer upload contains only the three unchanged public Hugging Face files
(`model.safetensors`, `config.json`, `README.md`) plus private-dataset metadata.
The exact public-only file list and checksums were verified before upload to
`adityakumarrai0401/rsna-knee-e04-convnext-imagenet-weights`. Its original upload
process completed successfully and Kaggle dataset status reported `ready`.
The creation receipt explicitly confirms it is private. No local competition
labels or fold map are part of this upload.

## Submission policy

No E04 competition submission has been made. B03 completed at 0.944 on submission
56881506, becoming the new best scored anchor. The latest official snapshot places
the team at rank 545 among 5,338 teams. All four 6 October submissions are terminal;
the user's sequential evaluation instruction still applies to any next submission.
B01/B02 at 0.943 remain fallbacks. E04 does not yet support a
numeric expected public score. Targets remain 0.946 for top 300 with public
margin, 0.948 for top 200 with margin, and the 0.95–0.96 stretch ambition.

## Paired resource test independently audited — 7 October

The original private paired probe completed. Two independent model processes each
ran 300 steps on a separate T4, with no expert updates or shared adapted backbone.
The independent small-output audit passed frozen-source, fold, device, memory,
numerics and checkpoint replay checks. Peak aggregate RSS was 7.134 GiB; each GPU
allocated about 2.158 GiB. Steady end-to-end steps were 0.8044 and 0.7797 seconds.
This is operational evidence, not an accuracy result. Pilot checkpoints are not reused.

Observed quota consumption was 477.247 seconds for a 460.095-second parent run,
including startup/finalization, rather than twice the kernel wall time. Remaining
allocated quota is 18.9409 hours. Three jobs for five fixed 14-epoch folds project
16.545 hours before validation/checkpoint I/O; with 15% margin, 19.027 hours. The
complete workload is therefore not proven to fit the remaining allocation. Start
with a bounded resumable first pair after all cache gates pass, measure complete
epochs, then budget each subsequent job. Preserve the fixed recipe and resume
after quota refresh if needed. Do not knowingly use paid overflow.

Evidence: `runs/E04_paired_pilot/resource_audit.json`,
`runs/e04_quota_after_pair.json`. Base cache is still RUNNING; full training has
not started. No new competition submission has been made.

## Full trainer and complete cache receipt audit — staged

Both cache producers generated complete manifests. All 4,407 study IDs, masks,
receipts, source hashes and 59 independent wide-reference hashes passed the
complete local audit. There were 185,974 selected-slice attempts and zero slice
or header failures. Base volume is 15,895,756,928 bytes, SHA-256
`6de60f4452158aa40219e786c86984e69a3711e18eced0a80f7b7d9830b38419`.
The producer's Kaggle status still reports RUNNING while finalizing output.
The launch guard correctly stopped before any push; this is not a training failure.

The first-pair script is assembled for folds 0/1 with an eight-hour job cap,
a five-minute finalization reserve, completed-epoch atomic states and strict
recipe/runtime checks on resume. Each fold uses a fresh ImageNet backbone,
expert labels only in its four training folds, fixed final epoch-14 EMA,
one final validation pass, and a saved-checkpoint prediction replay check.
No Gold checkpoint selection or blend weight search is performed.
The parent will hash both complete mounted volumes before creating model workers.

Compilation and assembled-definition checks passed. Local pandas/timm imports
were stubbed only for the definition check; model execution is not claimed from
that check. A CPU optimizer/scheduler/EMA completed-epoch resume test had exact
error zero; this does not prove bit-identical CUDA training.

The independent fold-output auditor verifies every held-out ID/order, exact
training/report-group separation, fourteen completed epochs, source/recipe
identity, prediction hashes, finite probabilities and recomputed expert AUCs.
It requires all five folds and all 58 experts before exposing complete OOF
Gate B evidence. Missing local checkpoint bytes remain an explicit integrity
gate; small-output validation is not a claim to have verified those bytes.

Files: `scripts/build_e04_fold_job.py`, `scripts/launch_e04_fold_job.py`,
`training/e04_fold_supervisor.py`, `training/e04_convnext_fold_train.py`,
`scripts/audit_e04_fold_training.py`. Receipts: `runs/e04_folds_01_source.json`,
`runs/E04_contract/cache_audit.json`, `resume_mechanism_check.json`,
`assembled_definitions_check.json`. No full training job has been accepted yet.

## First full fold pair accepted — 7 October, 02:20 IST

The original base cache is now remotely COMPLETE. The guarded upload of private
`adityakumarrai0401/rsna-knee-e04-convnext-folds-01` was accepted as version 1
at 2026-10-06 20:50:41 UTC, after confirming both cache inputs were COMPLETE
and freshly reading 18.9409 allocated free GPU hours. Its hard job limit is
eight hours. The launcher writes a receipt before a push and refuses duplicate
attempts if that receipt exists. Observe this original version; do not restart it
on polling timeouts. No other full fold pair was launched.

The first absence guard originally queried a new slug with the status endpoint;
Kaggle returned 403 for that nonexistent slug. This occurred before any push.
The guard now uses a complete authenticated owned-kernel listing and confirms
the known private cache is visible. The subsequent upload succeeded.

Training updates still require the parent to hash both complete mounted volume
files, then verify runtime labels/folds and strict generic checkpoint loading.
Small-output audits alone are not claimed to have completed that runtime gate.
All five held-out model folds remain required for Gold58 OOF validation. Neither
private training nor kernel upload is a competition submission.

Launch receipt: `runs/e04_folds_01_launch.json`; quota receipt:
`runs/e04_quota_before_folds_01.json`.

Latest first-pair observation: 2026-10-06T20:57:54.010146+00:00, status RUNNING.
The live-log endpoint returns an empty 400 response; persisted output/logs are
not available yet. This observation does not prove a runtime cache-hash PASS or
first model update, and does not establish a terminal training failure. Continue
observing the original accepted version without restarting it.

## Runtime cache gate and first full updates verified

Live logs are now available for the original version. The parent verified all
ten mounted cache files, including both complete volume hashes, before spawning
the models. Runtime verification receipt SHA-256:
`fdb3a1d732b13e0b00bc541c2ad5af1437ec5642f5f700ad802cad11f025b975`.
The receipt bytes are not locally downloaded yet; this is source-controlled
runtime execution evidence, not a claim to have locally streamed all volume bytes.

Both workers reproduced the frozen inputs, loaded the pinned pretrained encoder
strictly, and passed their startup contracts. Each trains 3,525 studies, including
46 experts, and holds out 882 studies including 12 experts. Runtime is Torch
2.11.0+cu128, timm1.0.29, NumPy2.1.3, CUDA12.8, T4 capability7.5.

Both have completed 200 batches of epoch1 with finite losses. Sampled aggregate
RSS peaked at 6.063 GiB; individual RSS was about2.6 GiB; GPU allocations about
2.157 GiB. No complete epoch has been observed, so further full fold jobs remain
unlaunched pending epoch wall-time budgeting. No expert metrics or public score
have been measured. Audit: `runs/E04_contract/first_pair_startup_audit.json`.

## Inference preparation while training runs

The new deterministic input packer exactly matches the executed training Dataset
on 75 readable missing-acquisition/slice patterns; both reject the one completely
unreadable case. Shapes, byte tensors, binary slot masks and float32 positions
match. `runs/E04_contract/eval_input_packing_check.json` records the source hashes
and limits; this check does not establish DICOM or GPU prediction parity.

Staged CPU decoding explicitly uses the training depth span0.06–0.94, rather
than the inherited reader's default0.15–0.85. Model prediction helpers use the
verbatim executed architecture/transforms, only pinned final epoch14 EMA files,
strict recipe/runtime checks, GPU FP16 inference and fixed equal probability
averaging across all five independent folds. No target-specific weights or
checkpoint selection is introduced. These helpers are staged code, not executed
GPU inference or submission artifacts.

Before standalone scoring, replay held-out expert predictions directly from
raw DICOM using the same validation batch partners and held-out checkpoint.
Do not validate Gold58 using the five-model average: four models saw each expert
case during training. Full hidden-inference schema, byte integrity and runtime
audits still apply. Before any blend, official standalone strength, measured
complementarity and the complete B03-plus-E04 nine-hour runtime remain gates.

Files: `inference/e04_series_inputs.py`, `e04_dicom_decode_worker.py`,
`e04_fold_predict.py`, `scripts/check_e04_eval_inputs.py`. No inference kernel
or new competition submission has been launched.

## Complete first epochs and second pair accepted — 7 October

Both original workers completed exactly1,763 batches and saved their epoch1
checkpoints. Fold0 wall time1,421.19s, checkpoint1.14s; fold1 wall time1,473.76s,
checkpoint1.05s. Median steady batch times after100 were0.71195s and0.74294s.
The complete-epoch timing audit passed and archived its exact source log.
A conservative fourteen-times-cold-epoch projection including checkpoint I/O,
measured setup, fifteen minutes of final validation and five minutes of parent
finalization is6.126 hours; with15% margin7.045 hours. This is a time forecast,
not an accuracy prediction. It supports the existing eight-hour bounded job.

Private `adityakumarrai0401/rsna-knee-e04-convnext-folds-23` was accepted as
version1 at21:31:40 UTC (03:01:40 IST). It uses the byte-identical worker
SHA`082cd85baed7bf293983e86b572e7f42347a008f350756264c34f69346212a80`.
No first-pair source, hyperparameter, epoch count or checkpoint rule changed.
Both remaining paired folds start from fresh generic pretrained weights.

At upload18.25984 allocated hours were free. The launch budget reserved7.81683
hours for the original live job, plus8.5 hours for the proposed job; these include
a half-hour per-job setup/finalization margin. The sum16.31683 hours fits the
allocation. Kaggle reported zero explicit reserved hours, so the launcher now
counts our own still-live training commitments before a new push. A third full
job is not funded yet; fold4 remains unlaunched until commitments are released.

New checks passed active-budget arithmetic and completed-epoch metadata
selection. Resume jobs use separate attempt IDs and accept only incomplete
folds from an explicit, downloaded owned-job manifest; mounted resume state
bytes must match its SHA before model workers start. Fully trained folds are
not rerun. The fourteen-epoch optimization rule and strict worker/runtime
resume checks remain unchanged. No resume was launched; tests of metadata
selection are synthetic and do not prove CUDA checkpoint resumption.

Files: `scripts/e04_job_budget.py`, `e04_resume_contract.py`,
`audit_e04_epoch_budget.py`, updated build/launch scripts. Evidence:
`runs/E04_contract/first_pair_epoch_budget_audit.json`, its archived log,
`concurrent_budget_check.json`, `resume_metadata_check.json`,
`runs/e04_folds_23_launch.json`, `e04_quota_before_folds_23.json`.

All five independently held-out expert folds are still required before GateB.
No expert metric, new public score, inference kernel or competition submission
has been produced by E04. B03 remains the scored anchor0.944; the top200–300
and0.95–0.96 objectives remain incomplete.

## Raw-DICOM replay and scoring preparation, 7 October

Both private fold jobs were confirmed RUNNING at 21:46 UTC on 6 October.
The first pair's log records fold0 completing epoch2 in 1,270.11 seconds,
with its completed-epoch checkpoint saved; fold1 was at batch1700/1763 in
epoch2. Training losses are not validation accuracy. The second pair subsequently
passed the same ten-file mounted cache hash gate in 196.40 seconds, with receipt
SHA `fdb3a1d732b13e0b00bc541c2ad5af1437ec5642f5f700ad802cad11f025b975`.
This was runtime execution evidence; the cache receipt bytes were not downloaded
or independently streamed locally a second time.

Both fold2/3 workers subsequently reported ready with the expected 3,526
training and 881 validation rows. Fold2 holds out12 experts, fold3 holds out11;
their training sets include46 and47 experts respectively. Independent AST
reconstruction of each executed training contract matches its runtime SHA,
and both use Torch2.11.0+cu128, timm1.0.29, NumPy2.1.3 and T4 capability7.5.
`runs/E04_contract/second_pair_startup_audit.json` and its archived log prove
cache and ready contracts only; they do not establish completed optimization
updates or model accuracy.

`scripts/stage_e04_raw_inference.py` assembles the exact executed SeriesMIL,
spatial transforms, generic-weight identity, series selector and owned reader
into importable modules. The staged replay selects original validation batches
containing expert studies, including each original batch partner. Each expert
uses only its independently held-out fold. For every replayed study it requires
raw-DICOM pixels to match the owned wide cache's pixel SHA, then probabilities
to match the saved final OOF predictions within 1e-6. Checkpoint, split, OOF and
pixel-reference files must match actual pinned byte counts and SHA values.
No invented checkpoint receipt, pilot checkpoint or favorable earlier epoch
can enter this route.

`scripts/check_e04_raw_replay_plan.py` passed the real frozen five-fold roster:
58 experts require 54 original batches and 108 study replays including batch
partners. Synthetic decoder checks verified ordered delivery, at most eight
pending decodes, and exception propagation without substituted predictions.
These are local planning checks, not DICOM or GPU replay results. The real
replay still requires all five completed final models and the OOF acceptance
and stability audit. Evidence: `runs/E04_contract/raw_replay_plan_check.json`
and `runs/e04_raw_inference_source.json`.

The completed-fold auditor now exports the exact Gold58 held-out OOF predictions
only after verifying all five complete fold artifacts. That CSV is intended for
the existing fixed-seed study bootstrap and target/fold/error-diversity review.
No such prediction file or uncertainty result exists yet. The small expert set,
repeated experiment use and unavailable patient/site identities limit these
checks; they cannot guarantee absence of overfitting or predict a public score.

The hidden prediction driver discovers inputs from test.csv, test_series.csv,
test_series and sample_submission.csv, without requiring training series. It
streams CPU decoding through four workers, uses fixed batch2 and the equal
probability mean of all five epoch14 EMA models, preserves raw per-fold and
mean predictions, and writes final submission.csv only after complete study,
target, numeric and decoding checks. It also requires a pinned successful raw
replay for the same checkpoints and inference source hashes. It has compiled
but has not executed on a GPU or produced a submission. There is no API
submission call in either driver.

Fold4 is staged locally with the identical frozen worker and a 7.25-hour
bounded job budget. `runs/e04_folds_4_source.json` records its build and existing
epoch timing gate. The quota snapshot used for this local build is historical;
the launcher must freshly verify quota and every running job's remaining
commitment before any push. Fold4 has not been launched. Training recipes,
original pair builds and official competition submissions remain unchanged.

## First updates, reserved quota and final validation gates

The second pair's log now proves finite optimization losses through batch600
of epoch1 for both folds. Aggregate RSS peaked at6.139GiB, with GPU peaks
2.1564/2.1567GiB. This is startup and first-update evidence, not complete-epoch
timing or accuracy. It is independently recorded in
`runs/E04_contract/second_pair_first_updates_audit.json` and its archived log.
The first pair has progressed to epoch3 for both folds.

The fresh 21:52UTC quota observation reported12,342.81 seconds used and
41,964.39 seconds reserved out of the existing75,600-second pool. This leaves
5.9147 unreserved hours. Fold4 requires7.25 hours plus0.5 hour margin, so it
cannot launch from this snapshot. `runs/e04_fifth_fold_funding.json` records
both live training commitments and the failed funding decision. No third job,
extra allocation or competition submission was initiated. Reservations are
not measured completed compute; observe their release and actual time used
before scheduling further work. The additional local commitment checks are
conservative when server reservations are also present.

Final validation review is now implemented in `scripts/review_e04_oof.py`.
It requires the independently audited complete Gold58 held-out OOF export,
recomputes pooled and per-fold AUCs against official truth, and reports a fixed
5,000-draw study bootstrap with seed20261007. The existing minimum pooled
macro AUC0.90 remains the rejection gate. Before any E04 expert results,
the following broad stability diagnostics were frozen: pooled target AUC below
0.70, any fold macro below0.80, or fold macro range above0.15 triggers further
review. These thresholds are diagnostics, not weights fitted to expert labels.
Undefined bootstrap draws and the small/reused expert set's limitations are
explicit. Freeze receipt: `runs/E04_contract/oof_review_rule_freeze.json`,
recorded at22:00:51UTC while the complete E04 expert prediction file was absent.

`scripts/build_e04_raw_replay_job.py` requires all five selected final artifacts,
the complete OOF audit, matching uncertainty/stability review, actual EMA
hashes, unchanged executed recipe and checksummed split/OOF/pixel-reference
files. It then assembles importable worker files and a private offline replay
notebook with a45-minute subprocess cap. The launcher
`scripts/launch_e04_raw_replay_job.py` checks completed input jobs, unchanged
source/gate bytes, slug absence and fresh funded quota before a single push.
An ambiguous push must be observed, never retried blindly. This path has not
built or pushed a notebook because the real completed-fold inputs are absent.

Five negative metadata controls passed: missing fifth fold, partial OOF with a
favorable score, macro below0.90, unresolved review, and outstanding stability
flags despite a favorable status all stop before notebook creation. Evidence:
`runs/E04_contract/raw_replay_build_gate_check.json`. This proves those rejection
paths only; compilation and synthetic metadata do not prove a passing final
model, GPU replay, hidden runtime or leaderboard result.

## Both second-pair first epochs complete

Fold2 completed epoch1 and saved its checkpoint in1,485.70+1.16 seconds;
fold3 completed it in1,432.51+1.18 seconds. Both covered all1,763 batches and
had finite losses. The independently archived first-epoch receipt audit is
`runs/E04_contract/second_pair_epoch_budget_audit.json`. Initial ready events,
rather than repeated cached ready messages during warmup, define setup time.
The conservative14-cold-epoch projection plus1,200 seconds for validation and
finalization and15% timing margin remains below the eight-hour job bound and
the locally staged7.25-hour fifth-fold budget. Actual quota still has to fund
the fifth job; a runtime projection does not release server reservations or
prove enough allocated hours. The first pair is in epoch4. No E04 expert metric
or official score exists yet.

The fresh public-source audit in `research/new_public/candidate_audit_20261007.json`
does not supply another eligible scored arm. Fleet's published run explicitly
fell back to its base stack. Creaky's three ConvNeXt anatomical MoE models did
execute, but their weights and validation artifacts were not found in public
metadata/search; retain it as a future candidate if those become accessible.
The new notebook titled0.946 changes per-target routing of existing families
without establishing an independently measured gain here. These public searches
caused no extra GPU launch or competition submission.

## Final prediction build and visible execution preparation

`scripts/build_e04_final_predict_job.py` now prepares the standalone five-fold
prediction notebook from the actual gated replay build and completed replay
outputs. It requires download integrity, the same five checkpoint identities,
all58 unique expert studies, each study's frozen held-out fold, exact inference
source hashes, zero decoding/substitution failures, and maximum probability
drift1e-6. It also checks replay detail probabilities, study identities, depth
span and selected-slice coverage. It does not create a notebook before those
real artifacts exist; the current state is source preparation only.

The generated wrapper materializes importable modules, uses the frozen bounded
file finder instead of walking DICOM directories, and starts a separate process
group for inference. Its timeout covers the whole inference process and CPU
workers. Cohorts of at most three visible studies use a30-minute cap; hidden
cohorts use8.5 hours. The unchanged numerical driver still averages probabilities
equally across all five final epoch14 EMA models, streams four CPU workers with
at most eight pending studies, preserves raw fold predictions, and publishes a
final CSV only after complete checks. No public-score-derived or Gold-fitted
weights enter this path.

`scripts/launch_e04_final_predict_job.py` checks immutable build/gate bytes,
completed model/replay inputs, a previously unused owned slug, and current
allocated quota before one private push. Before resource commitment it downloads
the current visible test.csv from Kaggle and requires1–3 unique studies. The
visible runtime cap plus startup/finalization margin is one hour; changed visible
cohort sizes require a fresh resource estimate. Download errors are sanitized
so temporary signed URLs are not printed. No quota purchase or competition
submission is performed by this launcher. An ambiguous push is observed using
its persisted receipt, not repeated.

`scripts/check_e04_final_predict_gates.py` passed eight negative replay controls:
wrong origin, changed manifest bytes, partial expert coverage, excessive or
nonfinite drift, all-model averaging on experts, changed checkpoint generation,
and changed inference source. The generated parent wrapper also compiled using
the real frozen file-finder source. No notebook or model was created by these
checks, and they do not prove a passing GPU or hidden-data run. Evidence:
`runs/E04_contract/final_prediction_gate_check.json` and
`runs/e04_final_prediction_preparation.json`.

Both original training jobs were confirmed RUNNING at22:29:53UTC on6 October.
Their logs showed fold0 in epoch5, fold1 near the end of epoch4, and folds2/3
in epoch2. All training recipes and original source builds remain unchanged.
Fold4 remains staged, not launched. Complete five-fold OOF, actual raw replay,
visible inference auditing and one official standalone score still precede any
new blend or final submission choice. B03 remains0.944; E04 has no accuracy or
competition result yet.


## Resumed review, 7 October 2026 12:18 UTC

Both original pair jobs are COMPLETE. The independent four-fold receipt audit
passes: all fourteen fixed epochs, exact splits, prediction hashes and recorded
checkpoint replay. Checkpoint bytes were not downloaded for local verification.
Fold expert macro AUCs are 0.883185, 0.888222, 0.887442 and 0.910284.
Pooled held-out predictions on the 47 available experts give 0.880350.
On exactly these studies E02 gives 0.897984 and E03 gives 0.883498.
This is partial OOF, not complete Gold58 validation or a public score.

Shelve E04 before fold4: the observed evidence does not justify consuming most
of the remaining quota. Fresh allocated free GPU time is 8.491683h, reservations
zero; funding alone is not a reason to train. No fifth model, replay, hidden
inference or official submission was launched. Retain B03's verified 0.944.
Do not select different epochs, fit target weights or present this as a complete
Gold58 gate failure. Preserve all existing remote checkpoints for later analysis.
