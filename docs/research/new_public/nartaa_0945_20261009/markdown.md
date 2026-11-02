# RSNA Knee: efficiency inference (0.945 public AUC)

**Runnable inference notebook.** The exact checkpoint is attached and a T4 GPU is enabled. This version runs the scored pipeline and writes `submission.csv`.

224-pixel training and inference; 307-pixel center crop from the native 384-pixel field, bilinearly resized to 224. K94, FP16 autocast, one view. One seed-42 training run, one averaged checkpoint at inference; averaged epochs [15, 13, 12]. The accuracy model is the original 384-trained checkpoint: the later native-320-crop-trained alternative scored 0.946 and is not substituted here.

The score is the displayed ordinary full-12 public leaderboard macro-AUC for submission `56707773`. The observed 7.2 minutes is submit-to-scored turnaround including queueing, orchestration and polling, not official worker runtime or an efficiency rank. Gold58 was reused for selection and development; it is not independent CV. This release does not establish significance or private-test performance.

The scored source's opening docstring and some comments describe older Raptor versions (0.924, 64 slices, 42 windows and earlier ensembles). They are retained to preserve the complete scored source bytes. This introduction, the live constants and the manifest describe the selected recipe: 96 slices, 94 windows, and one checkpoint.

## Reproduction requirements

Accept the competition rules and attach `rsna-knee-abnormality-detection`. Attach the exact checkpoint `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt` from the [selected SWA weights](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007), version 1. This public dependency contains only learned checkpoints and aggregate metadata. OAI images, OAI caches, report text and training label tables are not inference inputs and are not distributed.

Use the original Kaggle GPU environment: two Tesla T4 GPUs were available in the scored machine; the unchanged source uses its original CUDA worker policy. Runtime dependencies are Python, PyTorch with CUDA, timm, NumPy, pandas, pydicom and OpenCV. No downloads or pip installs are added. The scored Kaggle container is pinned to `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`. The earlier local precision qualification used torch 2.8.0+cu128; that is not a claim about the scored Kaggle image.

Source SHA-256: `e85d994fa93caf1bbc2df04f396649207a69d2463924c41fa3b05ba8801cd38c`. Checkpoint SHA-256: `3394fdd7e668286956e85af2554bef0386df5cd79d1c1af223b095c828b815d6`. The original private scored source is `nartaa/rsna-knee-coat224c-oai-ac-infer`, version 1. Its private dependency is `nartaa/rsna-knee-raptor-alldata-d96-blendjev-oai-c08r224`, version 1; it remains private. This publication uses a separate notebook slug and leaves the private scored notebook unchanged.

The metadata enables T4 GPUs with internet disabled. `RUN_INFERENCE = True` runs the two code cells in order after verifying the attached checkpoint. The final source writes `submission.csv`; competition submission is managed separately.

## Credits and required acknowledgement

Built on dreaddevelopment's [Raptor CoAtNet/MIL inference recipe and public baseline weights](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense). These are our retrained checkpoints, rather than those baseline weights. Weak-label sources include [stevenleehans](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels), [pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels), and [riadmohamed42's JEV labels and verification](https://www.kaggle.com/code/riadmohamed42/jev-knee-labels-verification), combined with our report-derived labels. Source reports and study-level label tables are not included here. The CoAtNet backbone comes from timm. Competition data is provided under the [RSNA Knee competition rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).

Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development. Dataset identifier(s): 10.15154/0hcg-f676.

[Shared NDA Study](https://nda.nih.gov/study.html?id=3407) · [DOI: 10.15154/0hcg-f676](https://doi.org/10.15154/0hcg-f676). Official acknowledgement source: [NDA manuscript preparation](https://nda.nih.gov/nda/manuscript-preparation).
