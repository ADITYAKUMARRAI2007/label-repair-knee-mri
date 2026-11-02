# Research findings — updated 7 October 2026

## Focused resumed review, 7 October 2026

E04's four completed folds pass the small-artifact audit but pooled expert OOF
on 47 studies is 0.880350 versus E02's 0.897984 on exactly the same studies.
E04 is shelved before spending roughly seven hours on the remaining fold. This
is not a complete Gold58 rejection gate or a Kaggle score. B03 remains 0.944.

A single recent public candidate, [RSNA Knee D4](https://www.kaggle.com/code/matterhorn3838/rsna-knee-d4), adds
[renta0426's meniscus specialist](https://www.kaggle.com/datasets/renta0426/rsna-knee-public0033-meniscus-bag-v1).
However, its actual published log shows the specialist failed to capture the
required 6x12x336 cache and both the model and overlay fell back to the parent.
No improved score or independent validation for this added arm was established.
Four small provenance files and the runtime were inspected; no weights, GPU
jobs or submissions were launched. The source says all 58 official-label cases
were excluded, but no additional report-connected exclusions were made.
Artifact details are in research/new_public/d4_resume/audit.json.

## New public notebook audit, 7 October

An authenticated search reviewed100 recent and100 score-sorted public notebooks.
Three newly updated routes were downloaded and inspected without execution.
Their authoritative public kernel statuses were COMPLETE at22:06UTC on
6 October. This establishes published execution, not a competition score.

- [Fleet Release Public](https://www.kaggle.com/code/juzer1/rsna-knee-fleet-release-public)
  shares the existing stack's14 attached datasets. Its optional student branch
  expects eff6/ens14 assets absent from public metadata and public dataset
  search. The published run explicitly reports missing assets, no fleet CSV,
  and a base-stack-only final output. It supplies no usable new arm as published.
- [Creaky ensemble](https://www.kaggle.com/code/nswitzer/creaky-0-94-ensemble)
  actually ran three ConvNeXt-Small anatomical MoE models, then blended their
  predictions at15%. Its log pins cache_v1 preprocessing for fold0 and raw_v1
  for folds1/2. The public metadata contains three blank dataset references;
  no named checkpoint datasets were found, and the author's public notebook
  search returned only this inference notebook. No model files are published
  as its outputs. Retain it as a candidate if weights, training provenance and
  independent validation become accessible. Its three visible predictions and
  title do not establish standalone strength or hidden-test performance.
- [Pure CoAtNet 0946](https://www.kaggle.com/code/ctogaurav/rsna-knee-pure-coatnet-0946)
  retains the existing DINO/A5/RadImageNet/Raptor/CoAt families and the same14
  attached datasets. Its material change is per-target outer routing: default
  CoAt weight0.60, with selected columns at0.75/0.80/1.00. No new independent
  checkpoint or reproduced validation justifies this routing here, and the
  title's0.946 is unverified. It does not establish a stronger reproducible arm.

Evidence: `research/new_public/candidate_audit_20261007.json`, the downloaded
sources/metadata, public output provenance, checkpoint-loading messages, and
public search receipts. Absence from public search does not prove that private
assets do not exist. No third-party private-access bypass, additional GPU run
or competition submission was attempted. E04's own five-fold training remains
the active independent-arm experiment; B03 is the verified0.944 anchor.

## Competition and data

- 4,407 training studies and roughly 1,300 hidden test studies.
- 570 GB and about 820,000 DICOM files, so full local development is impractical on the current machine.
- Only 58 training studies have all 12 expert labels. The other 4,349 have multilingual radiology reports and require weak labels.
- Each study has 3–14 series, median 5. There are 24,371 training series across sagittal, coronal and axial planes.
- Metric: unweighted mean of the 12 per-target ROC AUCs. Rank averaging is the natural ensemble operation.
- Hidden test prevalence may differ from training and public test prevalence.
- Kaggle's current competition metadata lists the deadline as 22 October 2026 at 23:59 UTC, with 4,586 teams and a USD 77,000 prize pool. There is enough calendar time for the planned independent-arm scoring and evidence-backed blending sequence.
- The historical 29 September leaderboard export contained 4,586 teams. The newer official export at 6 October 13:33 UTC contains 5,308 teams; our score remains 0.943 and rank is 829. A score of 0.944 now spans ranks 323–546, so it no longer reaches the requested top-300 band. A score of 0.946 spans 208–259 and 0.948 spans 148–175. The current working targets are 0.946 for top 300 and 0.948 for top 200 with margin; 0.95–0.96 remains the user's stretch goal. Full evidence is saved in `research/leaderboard_20261006/`.
- Kaggle's [official progression rules](https://www.kaggle.com/progression/competitions), rechecked on 6 October, give 1,000-plus-team cutoffs of top 10 plus one gold place per 500 teams, top 5% silver and top 10% bronze, with percentage calculations rounded down. Applying them to this 5,308-team public snapshot gives illustrative cut positions of rank 20 at 0.958 for gold, rank 265 at 0.945 for silver, and rank 530 at 0.944 for bronze. The 0.945 score band extends below the silver cut, so 0.946 provides useful public margin. These are provisional public comparisons; final medals depend on eligibility, final team count and private standings. The calculation is saved in `provisional_medal_cutoffs.json` alongside the leaderboard export.

Official references: [competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview), [data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data).

## Weak-label audit

Report-derived labels are the primary bottleneck. On the 58 expert-labelled cases:

| Public label source | Macro AUC |
|---|---:|
| Steven v4 soft blend | 0.893 |
| Pilkwang LLM labels | 0.866 |
| GPT-5.6-Sol hard labels | 0.835 |

The calibrated public target package reports 0.899 macro AUC before copying expert targets into its final training table. Its largest weak targets are synovitis, lateral OA, contusion and fracture. Files that overwrite the 58 expert rows score 1.0 by construction and cannot be used to estimate label quality.

Different report label sources remain highly correlated on the 4,349 weak studies (mean per-target Spearman correlations mostly 0.74–0.94). Label ensembling can improve stability, but it does not create an independent image model.

A new local audit downloaded all six selected public label packages and enforced source-level provenance. Dread supplies exactly the 4,349 weak studies and no Gold58 rows; Pilkwang covers 57/58 Gold studies; Yunus V5 copies expert truth into all 58 Gold rows and is therefore excluded from validation even though it remains usable on the weak corpus. Among the three nonleaking sources with near-complete Gold coverage, Steven V4 scores 0.8927 macro AUC and an equal rank mean scores 0.8934, an observed gain of only 0.0007 with a bootstrap 5th–95th percentile delta of -0.0050 to +0.0057 and 61.2% probability of improvement. The calibrated Rayan package reports 0.8991 before overwriting its exported Gold rows. M01 therefore uses Rayan as a 60% anchor and the median of the other five families as a 40% robustness component, while T01/E01 preserve their already-running six-family median recipe. The reproducible audit is saved in `runs/label_consensus_audit.json` and `runs/label_consensus_target_auc.csv`.

The first trainer draft's generic CSV discovery exposed a target-construction bug before compute started: it included the calibrated package's confidence-weight and hard-target CSVs as label predictions, and it medianed older Steven revisions with V4. The corrected selector uses one intended soft table per provider, choosing calibrated soft targets, Steven V4, and the latest available release from each remaining family.

The corrected live audit selected exactly six families: Pilkwang V2, GPT-5.6-Sol soft labels, Steven V4, Rayan calibrated soft targets, Dread soft labels and Yunus four-source V5. The calibrated package's `train_targets_weights.csv` is consumed separately as loss confidence and is no longer mistaken for target probabilities.

The final trainer's live preflight passed: 58 expert-labelled studies, pseudo-label coverage for all 4,407 studies, six selected families, calibrated confidence weights mounted from the intended Rayan package, two T4 GPUs and the pinned offline ConvNeXt Tiny ImageNet-22K checkpoint. It then entered MRI precaching without an exception.

## Public model landscape

The strongest widely shared stack combines:

- 20 DINOv2 slot-attention members;
- five A5 attention-pooling folds;
- several RadImageNet heads;
- four Raptor views;
- residual, D4 depth-zone, Global96 and Repair-v1 CoAtNet readers.

The verified public anchor is about 0.943. Recent public notebooks improve it to about 0.945 by increasing the default outer CoAt/Raptor routing from 0.60 to 0.70 while keeping specialist weights for ACL, both menisci, lateral OA and fracture.

The exact B01 output was downloaded after its visible three-study execution and passed a fail-closed artifact audit. Its study order and schema match `test.csv`, every probability is finite and within range, all component receipts report valid completion, and all component fallback counts are zero. The submitted CSV SHA-256 is `8f0c5e7b2b561538dae55439a4d2e9173141096da3defd9bc0a4bac351eef37a`. Submission `56680909` scored 0.943 public AUC. This validates the expected public anchor but leaves it in the large 0.943 tie below the desired top-300 safety band, so an independently strong diversity arm is still required.

The run also saved 0.60, 0.65, 0.70 and 0.75 outer-routing candidates. The 0.60 and 0.65 files were downloaded and passed schema audits, but Kaggle rejected a direct 0.60 code submission because this competition requires the selected kernel output to be named exactly `submission.csv`. A different candidate therefore needs a new executable kernel version whose primary output has that name; auxiliary output files cannot be selected directly. No submission slot was consumed by the rejected request.

An audit of newer notebook titles claiming 0.946–0.957 found no reproducible improvement over this anchor. Two contain the common 0.942–0.943 stack despite stronger titles. Another advertised ensemble depends on unavailable private EfficientNet assets; its only listed public dataset is an unrelated 2019 knee MRI collection. These are not useful submission candidates without a verifiable output or accessible weights.

The public `rsna-knee-fast-parent-0-957-v2` notebook was inspected separately because its title would imply a gold-level result. Its own fixed configuration says `PRESET="parent"`, flat 0.60 outer CoAtNet routing, and explicitly documents that this parent scored 0.939 public. The only 0.957385 number in the source is called a *local diagnostic*; no output cell or submission receipt connects it to the competition leaderboard. Its attached stack is the same DINO/A5/RadImageNet/Raptor/CoAtNet lineage already covered by B01, while B01 uses the stronger 0.70 outer routing. The title therefore supplies no new 0.957 evidence or independent diversity arm.

Two other score-like titles were also rejected on source evidence. `rsna-knee-speedy947` states in its own opening cell that its public score is 0.942. `rsna-knee-1-9455a3` is a CPU EDA notebook with only three report-label datasets attached; it has no image model or checkpoint and writes only a prevalence-constant baseline file. Neither title documents a 0.945–0.947 competition result.

The newer public `rsna-knee-tony-rank-ensemble` was also reproduced and audited. Its visible run completed all 20 DINOv2 members, five DINOv3 folds, both Raptor readers and all four CoAtNet family readers with zero reported fallback, and its `submission.csv` passed the exact schema, order, finiteness and range checks (SHA-256 `24dcf60e34741e17aa49c1e44985e7ba7add230b2059fa35c227b9f1793dd137`). The attached DINOv3 asset metadata explicitly documents a verified source score of 0.941 and a 0.942 target, while the current notebook supplies no independent leaderboard result for the repackaged ensemble. It is operationally reproducible and may provide diversity, but there is no evidence that it improves on B01's stronger public anchor; it will not displace the owned T01, E01 or M01 experiments merely on its title or component count.

The 29 September notebook `rsna-knee-d4-blend-0946-ours10` was audited after it appeared in the recent-run list. Its published metadata contains two blank dataset sources, while its final 10% leg explicitly requires private checkpoints named `npz_convnext_tiny_1ltfrywm_epoch10_fold0.pt` and `npz_convnext_tiny_source_reliability_best_fold0.pt` plus an unavailable prediction package. The claimed “public 0.946” anchor is the same DINO/A5/Rad/Raptor/four-CoAt lineage: the source notebook's own banner says 0.943, uses a 0.60 default outer CoAt weight, and records `score_recovery_claimed=False`. The newer title therefore supplies no reproducible 0.946 evidence. Its useful idea—an independent ConvNeXt leg at 10% in rank space—is already covered by T01 with owned checkpoints and held-out predictions.

The separate public notebook titled `rsna-knee-d4-public0946` was inspected as well. Despite its filename, its own banner and benchmark table identify the implemented recipe as 0.943, its outer finding weights still default to 0.60, and its final receipt explicitly records `score_recovery_claimed=False`. Its attached sources are the same public DINO/A5/Rad/Raptor/four-CoAt family already represented in B01. It therefore provides useful execution hardening but no measured 0.946 candidate.

The newly published `rsna-knee-super-ensemble-0920` also does not expose the advertised 0.946 route. Its executable begins with an unconditional `PARENT CELLS MISSING` exception and is only a skeleton that instructs the reader to merge a now-forbidden/private `pjmathematician/rsna-knee-d4-blend` parent. Its claimed extra `rsna-knee-eff6-assets` and `rsna-knee-ens14-assets` datasets are absent from the attached metadata and return no public Kaggle dataset search results. The notebook therefore cannot produce a submission as published and supplies no verifiable leaderboard receipt or accessible checkpoints.

The two public S75 notebooks that report about 0.945 explain the remaining gain more clearly: their public branch uses the same 0.70 outer routing now present in B01, then blends an independently trained five-fold family supplied through `aastikrajan15/knee-ckpt`. The notebook metadata contains an empty dataset source, and the named checkpoint dataset returns HTTP 403 to this account. Its embedded inference source confirms a mixture of 2D attention models and a 3D arm over multi-plane volumes, but the weights and training files are unavailable. B01 therefore transfers the reproducible 0.70 routing signal while T01 and E01 independently recreate the missing diversity principle with auditable training.

Two newer independent-looking public routes were checked after E01 scored. `ziyiming/rsna-exp013-infer-axial-weights` completed and publishes a CC0 85 MB axial ResNet34 checkpoint (SHA-256 `19b1805d261c1620b2540fe83c0fbed0fc8cc7802fc79ec4ae3c56c7ffc80380`), but its metadata documents only one fold, three epochs and no validation AUC or leaderboard receipt. Its visible inference spent about 31 minutes replacing the Kaggle PyTorch stack and reports only the three visible predictions. `anhtunnguynchu/05a-a-rsna-knee-multi-plane-dense-slice-mil-train` has a sounder multi-plane dense-slice design, but its authoritative status is `ERROR`, its default configuration runs only two development folds and skips final training, and its required pseudo-label and manifest dataset sources are blank. Neither currently supplies evidence strong enough to consume a submission.

Recent competition discussion provides a useful calibration for owned models. Multiple participants report 0.942–0.943 from one model family with TTA at 288 px, 0.947 after moving to five folds, and 0.943 from another five-fold 224 px model. A separate discussion reports that retaining 32 rather than 24 slices per series produced a larger gain than label changes, while several EfficientNet-B0 teams reportedly reached 0.94+ with five folds and TTA. These are participant reports rather than reproducible receipts, but they consistently point toward fold diversity, slice coverage, augmentation and regularization instead of larger input resolution. T01 already implements the relevant efficient form: one adapted ConvNeXt encoder, dense multi-series token extraction, and five independently trained attention heads. The next owned run should only duplicate this direction with a different backbone after T01's independent score is known. Sources: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304 and https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/743148.

Because many teams use this same lineage, it is likely to have correlated private-leaderboard errors. Public discussion reports that a decorrelated external arm with prediction correlation around 0.83 added roughly 0.004 AUC to one ensemble. Strong small CNNs at 224–288 px also reach 0.936–0.943, suggesting label handling, regularization and slice sampling matter more than raw resolution.

A recent three-submission diagnostic on one 0.940 CoAtNet split the targets into four-column groups. ACL, MCL and both menisci averaged 0.962; synovitis, Baker's cyst, medial OA and fracture averaged 0.941; effusion, contusion, PF OA and lateral OA averaged 0.914. This profile is model-specific, but it gives a useful hypothesis: a complementary arm that clearly improves the last group may be worth more than a uniform blend. The diversity analysis now reports these groups separately while retaining all per-target results. Reference: [per-finding public readout](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/744294).

The latest single-model discussion strengthens that conclusion: reported five-fold models reach 0.942–0.943 at only 224–288 px, while one competitor's move from DINOv2 Small to Base gained just 0.0011 against a measured 0.002 noise floor. The highest-value experiments are therefore label quality, study geometry and genuinely different model errors rather than a larger backbone. References: [single-model discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304), [encoder scaling result](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735154).

Participants also note that the adjudicated competition rubric is stricter than routine report wording, so reports tend to over-call some findings. This supports calibrated soft targets and confidence weighting instead of hard report labels. Reference: [training-label clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733836).

Experienced participants report that small 224-pixel CNNs with five-fold ensembling can reach roughly 0.94 public AUC. They also report that report-label cross-validation around 0.87–0.90 correlates with roughly 0.94 public AUC, while the 58 expert studies are too small to resolve changes of 0.002–0.003. This reinforces the use of Gold58 as a coarse rejection gate and prediction-diversity check, rather than a source for fine blend-weight optimization.

Further ensemble discussion supports the current experiment design: experienced competitors list label quality, preprocessing, representation, architecture, backbone, augmentation and schedule as distinct axes, and specifically emphasize model disagreement. A reported 57-study experiment found fixed 50/50 report/E11 blending more reliable than nested fine weight fitting, reinforcing coarse blend gates. Reference: [improving a 0.940 ensemble](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/742926).

## ConvNeXt diversity run

The selected independent arm uses ConvNeXt Tiny pretrained on ImageNet-22K, three epochs of MRI adaptation on non-gold weak labels, study-level multi-series features, and five label-aware attention heads.

The upstream public trainer did not complete. Its feature-extraction stage wrapped the encoder in `nn.DataParallel`; resident memory then grew from 7.9 GB at study 100 to 25.3 GB at study 400, and Kaggle killed the worker.

The first patch removed `DataParallel`, but the upstream throughput projected beyond the runtime cap. The revised run:

- decodes four bounded 20-study shards concurrently and exits each worker to return native DICOM/OpenCV memory;
- keeps CUDA exclusively in the stable parent process and uses fixed microbatches of 16;
- retains five depth positions across at most six selected series, or at most 30 study tokens;
- writes small float16 feature arrays without zlib compression;
- logs RSS every 100 studies and stops at 18 GB before an opaque worker death.

This preserves the model and training labels while bounding native allocator growth and reducing projected extraction time through CPU parallelism.

The first V2 run completed all 4,349 adaptation caches with zero fallback triplets and trained all three adaptation epochs, reducing mean loss from 0.5605 to 0.5141. It then exposed a narrow Python naming bug after its first 80-study feature wave: the decoder loop reused `proc` for a `Popen` object, overwriting the `psutil.Process` captured by `rss_gb()`. Kaggle retained the three adaptation checkpoints, the exported backbone and the first 80 feature files. The backbone ZIP passed integrity checks with 180 tensor members and SHA-256 `bec04468edeaaa0c75191a221d4e3b80bada5cfd263eeec36cb093548de93147`; all 80 feature files exactly match the first 80 train IDs and contain finite arrays with the expected 768-feature and six-metadata dimensions. The collision is repaired. Kaggle does not accept an errored notebook version as a kernel source, so a source-only resume attempt was stripped of that dependency and failed closed. Uploading the recovered weights as a separate Kaggle dataset was not authorized by the automatic approval review, so the corrected full run was launched from scratch.

The corrected run completed all 4,349 precaches in 72.6 minutes with zero fallbacks and 100% valid triplets, reproduced all three adaptation losses exactly (0.5605, 0.5348, 0.5141), held RSS near 4.09 GB, and then extracted features for all 4,407 studies in 59.9 minutes with zero fallbacks and RSS near 1.55 GB. Its five fold heads produced a complete Gold58 OOF macro AUC of 0.78695. The independent artifact audit reproduced that value, exact per-target AUCs, the manifest and OOF hashes, and all six deployment weight hashes. The strongest targets were medial OA at 0.9225 and effusion at 0.9068, while the four ligament/meniscus targets averaged only 0.7069. The predefined diversity diagnostic also showed E01 outperforming T01 overall and across that strong public target group. T01 therefore fails the written 0.88 validation gate and is rejected without consuming a leaderboard submission.

The failure clarifies the next design. T01 excluded every expert-labelled study from training and learned only a lightweight head after a shared weak-label adaptation. The strongest accessible competitive recipe instead uses fold-wise expert supervision, full or partial backbone fine tuning, 24–32 slices, and 320–336 pixel inputs. The fallback experiment will put four fifths of the 58 expert studies into each training fold with higher loss weight while keeping the remaining fifth strictly out of fold, and will fine tune a multi-plane 2.5D ConvNeXt rather than freeze a single shared representation.

T02 implements the first controlled correction using the exact 11.13 GiB public pixel cache: six MRI slots, three adjacent-slice groups, 224 pixels and a 130 mm crop. It fully fine tunes ConvNeXt Tiny for six epochs in each of five folds, averages the last three epoch weights, and initializes from T01's audited MRI-adapted backbone. The 4,349 report-labelled studies use a 60% calibrated-Rayan plus 40% median-of-five target; four fifths of Gold58 enter each fold at 4x loss weight and the held-out fifth supplies true OOF evaluation. The uploaded kernel reconstructs all labels inside Kaggle from six public datasets with seven pinned SHA-256 hashes. Its source audit passed balanced 882/882/881/881/881 folds, balanced 12/12/12/11/11 Gold partitions with both classes for all targets, full compilation, exact sources, and absence of embedded study identifiers or target arrays. The unsplit runs made no epoch progress after more than 20 minutes of actual GPU time, projecting too close to or past the 12-hour limit for all five folds. The exact work is therefore split without changing model logic: version 3 trains folds 0–2 (source SHA-256 `56eb1edf1a12aa16567e0712ee706e980db83329324be211d7073affa684569e`), and a companion trains folds 3–4 (source SHA-256 `c4cd5ef2648d42ee2be909bab114e4cf0bd9133266858e785658665edbfcc41d`). Both static audits pass. The companion launch is waiting for B02's hidden scoring run to release Kaggle's second batch-GPU session.

B02 separately reproduces `yamadan96/rsna-knee-d4-public0946` byte for byte (notebook SHA-256 `c7326deb3a03a18212f64cfe8558409920ad0d443bfd60425bcb911b4672b446`). It preserves the public route that gives the CoAtNet/Raptor family a default outer weight of 0.60, with ACL 0.75, medial meniscus 0.80, lateral meniscus 1.00, lateral OA 0.75 and fracture 0.75. This is materially different from B01's unverified transfer to a default 0.70, which scored 0.943. The owned B02 kernel completed in 284.9 seconds. Its artifact audit verified 20/20 DINO members, five A5 folds, four Raptor views, four CoAt readers, zero fallbacks, all component receipts, exact visible study order and final receipt SHA-256 `8f0c5e7b2b561538dae55439a4d2e9173141096da3defd9bc0a4bac351eef37a`. Submission `56690090` is pending its hidden rerun and score.

T02's live stream confirmed the exact cache shape `(4407, 6, 9, 224, 224)`, slot coverage `[0.9346, 0.9553, 0.9855, 0.6263, 0.7722, 0.9653]`, and a successful Tesla T4 launch for folds 0–2. A split-aware independent auditor now requires two manifests, five six-epoch histories, five checkpoint hashes/ZIPs and exact merged 4,407-study OOF coverage before computing Gold58 and derived-label AUC. The hidden inference kernel is also rebuilt and source-audited against both training sources (SHA-256 `74e111f42b2152128892f64cda42f6145536c5c0d50c73385776c5ff503cb0f2`); its launch remains gated on the merged training audit and Gold58 OOF strength.

The downstream hidden-test blend uses the same memory boundary. It writes a minimal CPU-only decoder into the notebook working directory and launches four fresh subprocesses per bounded wave with CUDA hidden. This avoids forking the large, CUDA-initialized public ensemble process and prevents inherited GPU state or copy-on-write host memory from destabilizing the final code submission.

## EfficientNetV2 Raptor run

A second independent CNN experiment uses the published two-part Raptor corpus because its training and hidden-test geometry are both available in public source. This avoids the preprocessing uncertainty found in several large pixel caches whose manifests are public but whose exact hidden-test decoder is not. The run uses EfficientNetV2-S with mounted ImageNet-21K weights, 336-pixel three-slice windows, target-specific attention pooling, and all 58 expert-labelled studies held out of training.

Its live label audit selected the same six intended families as T01 and produced complete targets for all 4,407 studies. The upstream corpus finder also revealed a startup performance defect: it recursively walked the 820,000-file competition DICOM tree for each small corpus artifact. The local reproducible builder now prunes image directories; the first live version was already running when Kaggle's two-session limit prevented immediate replacement, so the corrected source is staged for its next version.

The first training epoch completed in 1,745 seconds with loss 0.947 and Gold58 macro AUC 0.6725. The second completed in 2,454 seconds with loss 0.925 and AUC 0.7480. Epoch three completed in 3,168 seconds with loss 0.891 and AUC 0.7942. Epoch four completed in 3,891 seconds with loss 0.855 and AUC 0.8096. Epoch five completed in 4,611 seconds with loss 0.830 and AUC 0.8188. Epoch six completed in 5,334 seconds with loss 0.808 and AUC 0.8306, replacing the best checkpoint again. Epoch seven lowered training loss to 0.785 at 6,062 seconds, but Gold58 AUC declined to 0.8207. Epoch eight then recovered to 0.8423 with loss 0.764 at 6,783 seconds, epoch nine set 0.8463 with loss 0.751 at 7,512 seconds, epoch ten reached 0.8464 with loss 0.738 at 8,240 seconds, epoch eleven reached 0.8492 with loss 0.732 at 8,960 seconds, and the final epoch set the best 0.85357 with loss 0.729 at 9,681 seconds. A top-three weight average over epochs 12, 11 and 10 scored 0.85150, so the best epoch was selected. The local artifact audit independently verified the exact 4,349/58 split, all 12 epochs, every expert ID and truth, per-target AUCs, checkpoint ZIP integrity and embedded architecture contract. Standalone inference then completed the visible test in 14.57 seconds with zero fallbacks, passed exact submission audit with SHA-256 `1ba258ec5eace18c0bc4e09e9fb66ddefc3604765674bae468232c6f308f60f3`, and submission `56686670` scored 0.877 public AUC. That large validation-to-leaderboard gap makes E01 unsuitable for a medal-targeted blend; E01-B remains unsubmitted and is rejected.

The downloaded corpus contracts were checked independently: the two ID arrays contain 4,407 unique studies and exactly match the competition training-study set, while both mask arrays have depth 44. Their coverage boundaries identify the original `12 + 10 + 8 + 6 + 8` five-slot layout. The E01 inference arm therefore uses this 44-slice contract rather than the later public checkpoint's 64-slice layout. It evaluates all 42 possible three-slice windows because the Raptor author reported roughly +0.003 public AUC from increasing window coverage with identical weights.

A pre-deployment inference audit found that the first E01 reader converted `Fluid_Sensitive` metadata with `int(...)`, which can fail on hidden studies whose protocol flag is unknown or `NaN`. The staged E01 and M01 readers now use the public Raptor parser's explicit tri-state conversion and fall back only at the plane-selection level. Both generated kernels compile after the repair, and isolated picker tests pass for `NaN`, string-boolean and ordinary binary metadata.

## MedicalNet 3D candidate

A third candidate is prepared for the next free GPU slot. It uses Tencent MedicalNet ResNet-10, a 3D encoder pretrained across 23 medical imaging datasets. The official architecture and license were downloaded from `Tencent/MedicalNet`; the public Kaggle model exposes the matching `resnet_10_23dataset.pth` checkpoint under MIT. Its local audit found a 55 MB checkpoint with the expected `module.conv1`, batch-normalization and four residual-stage tensors and SHA-256 `afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605`.

The current official rules allow external data and models when they are publicly and equally accessible at minimal cost, and require open-source code that does not restrict commercial use. MedicalNet is public, free and MIT licensed, so it satisfies that rule. The Kinetics VideoMAE alternative was rejected because the original checkpoint is CC-BY-NC even though a third-party Kaggle mirror labels it Apache 2.0; MedicalNet avoids that provenance conflict and is more closely matched to volumetric medical images.

The trainer does not concatenate unrelated modalities into one artificial 44-slice volume. It preserves the corpus's five fixed sequence slots (`12 + 10 + 8 + 6 + 8`), resamples each valid slot to a 16-slice 112-pixel volume, encodes the slots independently, and learns target-specific attention over them. The first three 3D stages remain frozen; the fourth stage and classification head adapt to knee MRI for five epochs. Its weak targets combine the reported strongest calibrated family at 60% with the median of the other five families at 40%; the calibrated confidence CSV weights the element-level loss. Gold58 remains entirely outside training. Training completed all five epochs. Epoch four was best at 0.67356 macro Gold58 AUC, while the top-three average scored 0.67282. The independent local audit verified the exact split and truths, all five epochs, the pinned MedicalNet source, label-fusion contract, checkpoint ZIP integrity and 79-tensor state payloads. Only patellofemoral OA, effusion and synovitis exceeded the conservative 0.70 per-target blend gate. Standalone inference completed in 571.15 seconds with zero fallbacks, passed the exact submission audit with SHA-256 `1ea69d659b5bfd3b8bb97fe17a9ebcb795c578a75229ed290bb1d38c3d858116`, and submission `56687587` scored 0.687 public AUC. Log timing showed 576 seconds were spent locating checkpoints recursively before roughly 11 seconds of actual visible inference. Version 2 now resolves the trainer output directly and uses a depth-bounded, competition-pruned fallback; it completed in 10.11 seconds, produced identical study IDs and ranked predictions, and reproduced the submitted CSV byte for byte. Raw probabilities differed by at most 0.000129 from ordinary GPU numerical nondeterminism without changing any ranks. The low official score rejects M01-B despite the three apparently promising Gold58 targets; the 58-study per-target gate did not transfer reliably enough to justify spending a blend submission.

The public mask arrays confirm that this five-slot design is well supported: all 4,407 studies contain the first sagittal, first coronal and axial slots; 4,288 contain the second sagittal slot; and 3,283 contain the second coronal slot. Exactly 3,166 studies have all five, 1,239 have four, two have three, and none has fewer than three. Each present slot is fully populated at its declared width. The audit is saved in `runs/raptor_slot_coverage.json`. Linear depth interpolation converts the 6–12 acquired slot planes to 16 frames while target attention masks missing slots.

## Next decisions

The user's 6 October correction requires an official result and evaluation before another submission. The pending queue must therefore be evaluated sequentially before spending the fifth slot.

1. Treat B03 as the primary candidate for a measured improvement over 0.943. It exactly reproduces the newly released independent ConvNeXt reader route that repeatedly scored 0.943–0.944. The new leaderboard shows that 0.944 alone no longer meets the requested top-300 band, so B03 is an intermediate candidate rather than proof that the medal objective is achieved.
2. E02 officially scored 0.890, C12 scored 0.911 and C13 scored 0.563; all are rejected from blends. Gold58 is a rejection screen, not sufficient evidence for a blend weight.
3. Both new standalone arms failed the score gate. Do not submit either blend. Future checkpoint-only candidates require independently reproduced validation and an authoritative input/forward contract before competition scoring; follow the blend gates and complete runtime budget in `MEDAL_PLAN.md`.
4. Retain the strongest eligible scored anchor and do not spend the last submission on a speculative blend. A B03 result of 0.943–0.944 remains below the current rank target and does not complete the medal objective.
5. Keep E01-B, M01-B, J01-B and T01 rejected from all blends based on their completed standalone or Gold58 evidence.

Kaggle currently allows five submissions per day. Four submissions have been accepted in the 6 October quota window: E02 `56881026` is COMPLETE at 0.890 and C12 `56881514` COMPLETE at 0.911; C13 `56881626` is COMPLETE at 0.563; B03 `56881506` remains pending as of 20:49 IST. One slot remains and is reserved for a blend supported by an official standalone score. The failed direct-file C13 upload returned HTTP 400 and consumed no slot; the accepted code submission selected notebook version 2 explicitly.

Timing correction: Kaggle CLI submission timestamps are UTC. E02 was submitted at 12:47 UTC / 18:17 IST, B03 and C12 at 13:11 UTC / 18:41 IST, and C13 at 13:16 UTC / 18:46 IST. The earlier message treated the UTC times as IST and overstated elapsed time by 5.5 hours. Pending status by itself does not establish that GPU inference has started. B03's source author reports 6.5–8 hours for hidden scoring; that estimate implies roughly 01:11–02:41 IST on 7 October if processing begins promptly. Other standalone runtime estimates remain unverified extrapolations from three visible studies.

At 18:50 IST the authenticated Kaggle API reported all four submissions pending, with no score and no error description. This observation is saved in `runs/submission_status_20261006.json`. Independent visible artifact audits for all four arms are saved as `visible_artifact_audit.json` in their run directories; they verify exact test identity, target order, finite values, variation and receipt hashes. B03 additionally verifies the final 70/30 percentile-rank formula against the downloaded `_public_stack.csv` and `_own.csv` with zero discrepancy, three checkpoints blended and no logged reader failure. Its visible final CSV hash equals B01's hash because the three-study rankings remain identical after blending. This is not evidence that hidden predictions or leaderboard scores will match. No additional submission was made during this evaluation turn.

The follow-up provenance audit queried both official dataset file lists and read checkpoint metadata with a restricted tensor-metadata unpickler. It is saved in `research/new_public/checkpoint_provenance_audit.json`. Both public datasets expose only two `.pt` files, with no training code, split manifest or validation predictions. Consequently C12's 0.9022 and C13's 0.9133 are author-reported embedded values, not independently recomputed AUCs. C12 specifies slots but not its full pixel transform. C13's slots are inferred from the named Raptor MaxSpan-v5 source; the checkpoint does not store them. C13 also shares its source architecture and checkpoint lineage with the existing stack, so its diversity is unproven. The inherited standalone readers catch individual slice errors silently; `fallback_studies: 0` verifies absence of study-level substitute predictions, not complete DICOM decoding. These limits must be resolved or explicitly accounted for before using either arm in a final blend. No additional kernel was pushed or submitted during the follow-up audit.

After E02 officially scored 0.890, the public ConvNeXt reader's complete training, model, label and preprocessing sources were downloaded and hashed. Comparing them revealed a concrete E02 representation weakness: its three-slice picker crosses MRI sequence boundaries when adjacent slots are populated. Six of 32 deterministic evaluation windows cross boundaries in fully populated studies; weighting the known training-corpus slot patterns gives 17.07% across all studies. The published reader keeps windows inside each series and explicitly encodes plane/fat-suppression slot and slice position before a study transformer and per-finding attention pooling. This identifies a plausible contributor, not a measured causal effect or promised score gain. The investigation and future decisions are saved in `research/e02_reader_comparison.md`, with the calculation in `runs/E02_inference/sequence_boundary_audit.json`. B03 already includes this reader, so another clone would not by itself provide a new complementary family. The pending-results monitor is running under exec session `97172` and stops on the next official completion or error; it never submits or restarts Kaggle jobs.

At 19:08 IST the live monitor revalidated B03, C12 and C13 as pending with no reported score or error. A fresh public-kernel search found no public competition training notebook for the C12/C13 authors. The updated `romantamrazov/rsna-knee-dinosaur-v5` now attaches the same goodpjw reader used by B03 and explicitly runs three public checkpoints with a 70/30 percentile-rank blend; its changes parallelize decoding and inference and add a runtime guard. The newly checked `matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed` also uses the same public reader and 30% weight. Neither introduces a new complementary model or an independently verified 0.946-plus recipe. Source identities and decisions are saved in `research/new_public/updated_stack_audit_20261006.json`. No new competition submission was made.

E02's strict expert validation uncertainty was measured with 5,000 fixed-seed study-bootstrap draws using the saved 58-study predictions. The recomputed macro AUC exactly matches the independent training audit at 0.8952057; its percentile 95% interval is 0.8630249–0.9229970. AUC was defined for every target in 4,998 draws; the two undefined macro draws were reported and excluded. Individual target class counts and intervals are saved in `runs/E02_training/gold58_bootstrap.json`. The reusable stdlib-only implementation is `scripts/bootstrap_gold58.py`; it handles probability ties, replicated-study weights and one-class bootstrap draws. This diagnostic supports caution about small Gold58 differences and target-specific weight fitting. It does not estimate private-leaderboard AUC, capture checkpoint-selection or training uncertainty, or establish that C13's higher embedded score is significantly better; paired predictions would be required for that comparison. At 19:23 IST the monitor still reported B03, C12 and C13 pending, with no error descriptions. No new model or blend was submitted.

At 20:43 IST on 6 October, the authenticated Kaggle API reported C12 submission `56881514` COMPLETE at public AUC 0.911 with no error. The preserved receipt is `runs/submission_monitor_remaining_20261006/result.json`. Observed submission-to-result elapsed time was 122.2 minutes, including queueing; it is not a measured GPU runtime or a C13 ETA. C12 is 0.032 below the 0.943 anchor and is rejected from final blends. No paired validation evidence establishes complementary errors sufficient to justify overriding the score gate. Its preprocessing and split provenance limits remain unresolved, so the score alone cannot identify the cause of the weakness. The previous observer exited normally after saving this result. A new read-only observer for the remaining B03 and C13 refs runs under exec session `34489`, with output in `runs/submission_monitor_b03_c13_20261006`; no Kaggle jobs were restarted or newly submitted.

At 20:49 IST, the authenticated API confirmed C13 `56881626` COMPLETE at 0.563 with no reported execution error, matching the user's screenshot. The receipt is `runs/submission_monitor_b03_c13_20261006/result.json`; observed elapsed time was 122.4 minutes including queueing. The 0.3503 gap from embedded author-reported Gold58 AUC is severe. Strict state loading, matching labels and visible finite raw probabilities do not establish forward/preprocessing equivalence or hidden accuracy. No source code or saved Gold predictions from the checkpoint author are available locally. C13 is rejected from all blends; do not fit weights or retry a guessed transform on the leaderboard. The remaining B03 observer is exec session `19307`, output `runs/submission_monitor_b03_20261006`. No competition submission or remote job restart was made.

## Risks

- The 58 gold studies are too small for stable model selection, especially for rare findings.
- Public leaderboard tuning can overfit because many teams are probing similar blend weights.
- The public baseline is highly shared, so a silver-level public score may not hold privately.
- DICOM decoding and series selection must fail visibly; silent zero or 0.5 fallbacks can produce valid-looking submissions with poor scores.


## 6 October: E03 resource pilot passed; full training underway

E03 is a new owned ResNet34 series-transformer experiment, preserving the pinned 75/25 teacher/JEV supervision while using acquisition-contained windows, explicit slot/position information, full encoder adaptation and final EMA. It is not a competition submission. The 300-step private pilot completed in 244.56 script seconds, with zero AMP skips, sampled host peak 2.107 GiB, GPU peak 1.953 GiB and only 0.0308 GiB RSS growth after step 50. Missing-acquisition perturbation and checkpoint replay both had zero error; the 361,661,296-byte downloaded state matches its recorded checksum. Local window checks covered 6,600 cases and 122,460 windows. Full training starts from clean pinned ImageNet weights rather than this pilot state.

The frozen report-group split contains 3,939 training, 406 weak-validation, 58 Gold58, and four additional excluded weak studies in Gold report groups. Full trainer `adityakumarrai0401/rsna-knee-e03-resnet-series-train` v1 is RUNNING and its startup log confirmed the exact split contract. Its first 100 batches completed with host RSS 2.182 GiB and GPU peak 2.046 GiB. Epoch count is fixed at 14 and the final EMA is used; Gold58 is evaluated only after the final epoch. Full resumable state is saved after each completed epoch. The provisional planning range is 7–9 hours including validation, pending an actual completed epoch.

The teacher soft labels independently score 0.996299 on Gold58, but the public dataset has no description and its author's available public inference notebook does not establish upstream teacher training/exclusion provenance. E03's own gradient updates exclude Gold58; independence of upstream supervision remains unverified. This limits validation interpretation and does not prove either leakage or medal-level image performance. Require the existing official scoring and complementary-error gates before blending. No new competition submission was made; B03 `56881506` remains pending, and B01/B02 remain the verified 0.943 anchor. Full details and receipts are in `research/e03_series_reader.md`, `runs/E03_pilot/resource_audit.json`, `runs/E03_contract`, and `runs/E03_training/launch_receipt.json`.


## 7 October: E03 audited rejection; exact cache reconstruction established

The corpus author metadata confirms a 15–85% native-depth span for both public 44-slice cache parts. E02's submitted decoder used 6–94%, so its comment claiming input agreement was incorrect. E03's new instrumented 15–85% decoder was verified on all 58 expert studies plus one additional missing-acquisition-pattern example: 59/59 reconstructed volumes and masks match the actual cache exactly, covering four presence patterns and both parts. All 2,476 selected slices decoded with zero failures. The old span's mean pixel MAE is 25.227/255; no causal AUC effect has been measured. Evidence: `runs/E03_cache_parity/pixel_parity_audit.json`.

E03 full training completed all 14 frozen epochs in 5,613.63 script seconds. Once the first cold-cache epoch finished, epochs took roughly five minutes rather than the pilot's cold-cache extrapolation. The complete checkpoint, validation CSVs, split and resume state have been downloaded and their checksums verified. Independent `scripts/audit_e03_training.py` recomputes final Gold58 AUC 0.88110725395, versus E02's 0.89520574647, with a paired point change of −0.0140985 and 10/12 targets lower. Weak-label threshold AUC 0.920066 is not an expert or public score. Four AMP updates were skipped and recovered; no persistent resource/numerical error was observed. A transient DNS download error was resolved by reading the same completed output with atomic streaming; no job restart occurred.

E03 fails the frozen 0.90 expert rejection gate and will not be submitted or blended. It has no official public score. Retain the verified input/resource implementations, but do not select a different epoch or target weights after seeing the expert outcomes. The multi-change experiment does not identify a unique reason for the lower clinical metric. A stronger ImageNet-22k pretrained ConvNeXt encoder with separately trained expert-validation folds is a next research hypothesis, not a demonstrated gain. B03 `56881506` remains the sole pending official submission; B01/B02 at 0.943 remain the verified anchor.


## 7 October: B03 new anchor and paired resource test

B03 submission 56881506 completed at 0.944 (+0.001 over B01/B02). The official
20:36 UTC snapshot places the team at rank 545 of 5,338. This is below the
requested top 200–300; the medal objective remains incomplete. All four 6 October
results have now been evaluated. No further competition submission has been made.

E04 paired resource test completed and independently passed operational auditing
(two separate models/GPUs, 300 steps each, no Gold updates, zero AMP skips, exact
mask/replay checks). Full accuracy training has not started. Cache base remains
running; mounted full-volume hashes and bounded resumable training are next.
Remaining allocated quota is 18.9409 hours. See the E04 research plan and audit
receipts for timing limits.


## Past-challenge and noisy-label research informing the next plan

The new section in MEDAL_PLAN.md uses primary references: Theo Viel's RSNA2023
second-place repository, Siwoo Yong's RSNA2024 fifth-place repository, MRNet,
CheXpert, Confident Learning and DivideMix. Transferable hypotheses are
anatomically relevant slice aggregation and explicit handling of uncertain
report-derived targets. Original rankings, metrics and benchmark improvements
are not estimates for the current twelve-finding challenge.

The first planned comparison holds encoder, inputs and training recipe fixed
while changing one supervision policy. Reuse existing consensus analysis and
perform report evidence review before any feature extraction or training.
The observed UNK rates in one source are descriptive, not verified disease
status or evidence that masking will improve AUC. Primary-source citations,
resource caps, stop rules, required serving checks and optional expert support
are in the plan. No GPU job or competition submission was launched for this
research/planning step.
