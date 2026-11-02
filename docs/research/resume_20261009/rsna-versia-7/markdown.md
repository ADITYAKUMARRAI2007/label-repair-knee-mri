# 🩺 RSNA Knee Abnormality Detection: A Diagnostic Quest

Welcome to the clinic of the future!  
You are a radiologist — but not an ordinary one. You are armed with neural networks.  
Your task is to make **12 key diagnoses** from knee MRI studies, using both images and, during training, the accompanying radiology reports.

---

## 🧩 The Core Challenge

Each study is a **puzzle** made of several MRI series — slices acquired in different planes and with different contrast weightings.  
You must assemble the full picture and answer 12 questions: Is there a ligament tear? Meniscal damage? Osteoarthritis? Inflammation? A cyst? A fracture?

**Metric:** macro-averaged ROC AUC across all 12 labels.  
**Key twist:** during training, you have access not only to the images but also to the **radiologists’ free-text reports**. At inference time, reports are absent — so you must learn to squeeze every bit of signal from them while training.

---

## 🧠 Analogy: “Training a Virtual Radiology Resident”

Imagine you are teaching a new resident:

- **DICOM series** — different examination tools: sagittal, coronal, and axial views.
- **Fluid-sensitive sequences** (T2, PD, STIR) — a flashlight that highlights fluid, edema, and tears.
- **train.csv** — patient charts with partially completed diagnoses.
- **train_series.csv** — a description of each scan: plane, fat suppression, fluid sensitivity.
- **Radiology reports** — notes from an experienced mentor who sometimes writes “ACL tear” and sometimes “intact menisci.”

Your resident (the model) must learn to make diagnoses even when the mentor is not there (at test time).

---

## 🏆 Difficulty Tiers of the Diagnoses

| Tier | Diagnoses | Why They Are Hard | What Helps |
|------|-----------|-------------------|------------|
| 🟢 **Easier** | Baker's cyst, Effusion | Clearly visible on fluid-sensitive sequences | A single sagittal series |
| 🟡 **Moderate** | MCL, Medial/Lateral Meniscus, PF OA | Need assessment across multiple planes | 2–3 series + attention |
| 🔴 **Hard** | ACL, Synovitis, Contusion, Fracture | May be subtle; require broader context | Multi-plane aggregation, 2.5D/3D inputs |
| 🟣 **Very Hard** | Medial OA, Lateral OA | Cartilage thickness assessment; need thin slices | Specialized models + post-processing |

---

## 💡 The Main Takeaway

**This competition is won by whoever does three things best:**

1. **Select the right series** — not all 500 GB of data is useful.
2. **Exploit the reports** to expand weak labels — they are free supervision.
3. **Aggregate carefully** across planes and slices.

No magic — only solid data engineering and thoughtful modeling.

---

**Let’s go!** 🚀  
We start with a simple diagnostic tool and gradually turn it into a radiology expert.

# 🦖 RSNA Knee Abnormality Detection — Evolution of Public Solutions (0.937 → 0.951)

### 📊 Seven Public Solutions | From Ensemble Engineering to Independent MRI Models

This notebook documents the evolution of seven public Kaggle solutions and the architectural decisions behind their reported leaderboard improvements.

Our goal is to understand how MRI preprocessing, model diversity, anatomical representations, and ensemble strategies evolved — while preserving full credit to the original authors.

**Current reproduced pipeline: Version 20 — Heliosli's 0.951 public leaderboard blend.**

**Important:** Public leaderboard improvements do not guarantee stronger private leaderboard performance. The scores below describe publicly reported results, not independently verified validation improvements.

---

## 🥉 Work 1 — 0.937
### Head and Shoulders, Knees and Toes

**Author:** prvsiyan  
**Notebook:** https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes

- Expands the CoAtNet family to **four arms**: v5, v10, v5-reverse, and v8.
- Introduces reverse-flip inference to increase prediction diversity without additional training.
- Uses a global **0.60 CoAtNet blending weight**.

**Key idea:** Improve ensemble diversity by combining complementary inference variants.

---

## 🥈 Work 2 — 0.940
### RSNA Knee Hybrid Raptor — Lateral Meniscus R100

**Author:** renta0426  
**Notebook:** https://www.kaggle.com/code/renta0426/rsna-knee-hybrid-raptor-lateral-meniscus-r100

- Reproduces its 0.939 parent solution.
- Modifies the ensemble routing for **Lateral Meniscus only**.
- Assigns this finding entirely to the strengthened Raptor branch: **Raptor 1.00 / Transformer-Rad 0.00**.
- Preserves the original predictions for the other 11 findings.

**Key idea:** A single problematic target may benefit from specialized routing rather than global ensemble changes.

---

## 🥇 Work 3 — 0.941
### Bend the Knee to the Dinosaurs (ALL PUBLIC)

**Author:** Mattia Angeli  
**Notebook:** https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs

- Retains the four-arm Raptor family.
- Introduces **residual-gated CoAtNet** models.
- Combines the existing Raptor branch with additional CoAtNet predictions.
- Incorporates a CoAtNet MRI reader using adjacent-slice triplets and finding-specific aggregation.
- Refines target-specific routing, including increased Raptor contributions for ACL, meniscus injuries, lateral osteoarthritis, and fracture.

**Key idea:** Strengthen the ensemble with alternative visual representations and finding-specific prediction routing.

---

## 🚀 Work 4 — 0.942
### rsna-knee-0942-restructured

**Author:** maverick_ss_26  
**Notebook:** https://www.kaggle.com/code/maverickss26/rsna-knee-0942-restructured

### Main changes

| Component | Previous | Updated |
|---|---|---|
| `a5_w` | 0.45 | **0.52** |
| `rad_alpha` | 0.50 | **0.55** |
| `rad_second_alpha` | 0.15 | **0.20** |
| Raptor evaluation windows | 62 / 42 | **94** |

- Increases MRI sampling density without introducing a new model architecture.
- Adds alternative RadImageNet heads.
- Parallelizes selected inference branches to reduce runtime.
- Introduces diagnostic tables and execution presets.

**Key idea:** Better sampling coverage, computational efficiency, and blending can improve an existing architecture.

---

## ⚡ Work 5 — 0.943
### rsna-knee-fast-2xt4-inference

**Author:** Jiwei Liu  
**Notebook:** https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference

- Expands the CoAtNet family from two to **four complementary readers**.
- Adds **Global96** and **Repair-v1** checkpoint families.
- Changes ensemble reduction from averaging individual model ranks to **ranking averaged probabilities**.
- Executes model pairs in parallel.
- Reuses DICOM header information to accelerate preprocessing.

**Key idea:** Model diversity and probability-space aggregation can improve the ensemble without relying exclusively on more complex blending weights.

---

## 🦖 Work 6 — 0.950
### RSNA Knee V2 — Velciraptor Dinosaur Speed

**Author:** Matterhorn3838  
**Notebook:** https://www.kaggle.com/code/matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed/notebook

### Major architectural simplification

Replaces the large multi-family ensemble with two complementary MRI readers:

**1. CoAtNet Raptor**
- One SWA checkpoint.
- 96-slice anatomical representation at 384 px.
- 94 overlapping 2.5D evaluation windows.
- Anatomical mirror test-time augmentation.
- Accelerated DICOM preprocessing.

**2. ConvNeXt 2.5D MIL**
- Three pretrained checkpoints.
- Multi-series MRI representation.
- Adjacent-slice triplets.
- Finding-specific attention and study-level aggregation.

**Final ensemble:** Target-specific rank fusion between the CoAtNet and ConvNeXt branches.

**Key idea:** Two complementary architectures can outperform a much larger collection of partially redundant models.

---

## 🏆 Work 7 — 0.951
### RSNA Knee Blend Gold

**Author:** Heliosli  
**Notebook:** https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold/notebook

### What changed compared with Work 6?

The public CoAtNet Raptor branch is retained, but the second branch is fundamentally different.

Instead of reusing the previous public ConvNeXt MIL ensemble, Heliosli introduces **a separately trained compact MRI classification system**, developed with its own preprocessing, fold checkpoints, and report-derived learning targets.

| Component | Work 6 — 0.950 | Work 7 — 0.951 |
|---|---|---|
| First model | CoAtNet Raptor SWA | **CoAtNet Raptor SWA retained** |
| Second model | Public ConvNeXt 2.5D MIL | **Heliosli's custom fine-tuned MRI model** |
| Image preprocessing | Raptor + ConvNeXt pipelines | **Raptor + native DICOM/geometry pipeline** |
| Slice representation | Multiple anatomical series, 2.5D windows | **Physical slice ordering, local views, 2.5D windows** |
| Second-model training | Existing public checkpoints | **Separately trained fold checkpoints** |
| Label strategy | Existing model training targets | **Report-derived weak targets and `E_lrprior` objective** |
| Anatomical correction | Existing reader conventions | **Optional knee anchors and image-based laterality check** |
| Final blending | Target-specific rank weights | **Fixed 70% Raptor / 30% custom model** |

### Custom model architecture

- **Native DICOM preprocessing:** reconstructs knee MRI series while preserving anatomical geometry and physical slice information.
- **Anatomical views:** supports full-field and localized crops, including patellofemoral and posterior regions.
- **2.5D representation:** combines neighboring MRI slices into multi-channel image windows.
- **Image encoder:** supports compact CNN backbones and the OrthoFoundation-L pretrained encoder.
- **Finding-specific attention:** aggregates MRI windows independently for all 12 target abnormalities.
- **Report-based supervision:** supports soft targets, likelihood-ratio priors, and auxiliary report-state objectives.
- **Fold inference:** combines separately trained fold models by averaging logits before applying the sigmoid.
- **Laterality verification:** optionally checks whether knee orientation is consistent with anatomical evidence.
- **Reproducibility:** verifies source files, training-code modules, and checkpoint assets using SHA-256.

### Final fusion

For each of the 12 findings:

`Final prediction = 0.70 × Rank(Raptor) + 0.30 × Rank(Custom Model)`

- Uses percentile ranks rather than directly mixing probabilities.
- Keeps the weights fixed rather than fitting them to the small official labeled subset.
- Checks study identifiers, target columns, model agreement, and missing predictions.
- Falls back to the available model when one inference branch fails.
- Saves a verified competition submission and diagnostic receipts.

### Why this work matters

Heliosli's solution marks a meaningful change in the public-solution trajectory.

The previous improvements primarily involved **recombining existing checkpoints, modifying ensemble weights, and accelerating inference**.

This work contributes a different type of advancement: **a custom image-learning pipeline with its own preprocessing, supervision strategy, architecture, and trained checkpoints**.

Its reported public improvement from **0.950 to 0.951** is modest, but the technical contribution is considerably broader than a simple ensemble-weight adjustment.

**Key idea:** Build a complementary MRI representation and learn from weak radiology-report supervision, rather than relying entirely on existing public model families.

---

## 🔑 Key Patterns Across the Seven Solutions

### Phase 1 — Ensemble expansion (0.937 → 0.943)

The early public improvements emphasize:

1. Additional CoAtNet arms and inference diversity.
2. Finding-specific routing and ensemble weights.
3. Denser slice sampling.
4. Probability-space aggregation.
5. GPU parallelism and DICOM preprocessing efficiency.

### Phase 2 — Architectural simplification (0.943 → 0.950)

Work 6 replaces the large ensemble with **two complementary readers**: CoAtNet Raptor and ConvNeXt 2.5D MIL.

The main lesson is that complementary architectures may be more useful than simply increasing the number of checkpoints.

### Phase 3 — Independent model development (0.950 → 0.951)

Work 7 retains the strong Raptor baseline and replaces the second branch with **a separately developed and trained MRI model**.

This introduces new research directions:

- Better anatomical alignment and localized MRI views.
- Improved supervision from radiology reports.
- Explicit handling of uncertainty and label quality.
- Robust prediction across MRI acquisition protocols.
- More reliable validation of complementary image representations.

---

## 🔬 Research Direction — Beyond Public Leaderboard Replication

Our current notebook reproduces the Work 7 inference approach as a baseline.

The next objective is **not merely to tune its 70/30 ensemble weights**.

Instead, we aim to understand the mechanisms behind the existing solutions and develop an independently validated model capable of generalizing to the private leaderboard.

### Research priorities

**1. MRI understanding:** study structure, acquisition planes, physical slice geometry, scanner differences, and image intensity normalization.

**2. Label quality:** report-derived targets, uncertain diagnoses, systematic annotation errors, and differences between the 12 findings.

**3. Validation strategy:** patient/study grouping, fold construction, potential leakage, scanner/site effects, and public/private distribution shift.

**4. Independent modeling:** hypothesis-driven preprocessing, anatomical representations, training objectives, and complementary model architectures.

**5. Error analysis:** identify which findings improve, why predictions disagree, and whether observed gains are robust rather than leaderboard-specific.

**Long-term objective:** develop a strong private-leaderboard solution through independent research, not just replication of public checkpoints.

---

## 📚 Credits & Acknowledgements

All seven public solutions are credited to their respective authors. Their contributions include independently developed models, preprocessing pipelines, trained checkpoints, and ensemble designs.

Our reproduction does not imply authorship of the original public models.

- **Work 1 — 0.937:** [prvsiyan](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes) — four-arm CoAtNet ensemble and reverse-flip diversity.
- **Work 2 — 0.940:** [renta0426](https://www.kaggle.com/code/renta0426/rsna-knee-hybrid-raptor-lateral-meniscus-r100) — specialized Lateral Meniscus routing.
- **Work 3 — 0.941:** [Mattia Angeli](https://www.kaggle.com/code/mattiaangeli/bend-the-knee-to-the-dinosaurs) — residual-gated CoAtNet and finding-specific ensemble refinement.
- **Work 4 — 0.942:** [maverick_ss_26](https://www.kaggle.com/code/maverickss26/rsna-knee-0942-restructured) — improved sampling, blending, and inference scheduling.
- **Work 5 — 0.943:** [Jiwei Liu](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference) — expanded CoAtNet family and probability-space aggregation.
- **Work 6 — 0.950:** [Matterhorn3838](https://www.kaggle.com/code/matterhorn3838/rsna-knee-v2-velciraptor-dinosaur-speed/notebook) — streamlined CoAtNet + ConvNeXt MIL ensemble.
- **Work 7 — 0.951:** [Heliosli](https://www.kaggle.com/code/heliosli/rsna-knee-blend-gold/notebook) — independently trained compact MRI branch, native preprocessing, report-derived supervision, anatomical corrections, and Raptor/custom-model rank fusion.

### Additional contributors

Special thanks to contributors whose publicly shared work underpins the broader RSNA Knee modeling ecosystem, including **dreaddevelopment / Roman Tamrazov, Pilkwang, Sofia Anjenje, Antoine G., Marwan Mahmoud, Anvith Pothula**, and the creators of publicly released pretrained checkpoints and weak-label resources.

Credit also belongs to the original medical imaging dataset providers, competition organizers, and researchers whose annotations and radiology reports made this challenge possible.

---

*Version 20 — Public Solution Evolution and Heliosli 0.951 Reproduction.*

*Competition objective: robust generalization to the private leaderboard, supported by independent validation and reproducible experiments.*

## 1. 🦖 CoAtNet Raptor — Verified Inference Pipeline

Creates the standalone CoAtNet Raptor inference script with checkpoint verification, accelerated DICOM preprocessing, and anatomical mirror TTA.

**Key components:**
- `RUN_INFERENCE = True` — enables CUDA inference.
- **SHA-256** — verifies the exact SWA checkpoint `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`.
- **CoAtNet RMLP-2** — backbone with finding-specific attention for all 12 abnormalities.
- `IMG = 384`, `INPUT_RES = 320` — MRI preprocessing resolution and model input crop.
- `MAXS = 96`, `K_EVAL = 94` — 96-slice anatomical representation and 94 evaluation windows.
- **Mirror TTA** — averages original and anatomically mirrored predictions.
- `_FASTREAD_B64` — embedded accelerated DICOM reader using the `study_lut` strategy.
- **Parallel preprocessing** — prepares MRI studies and distributes inference across available GPUs.

**Output:** Writes `/tmp/raptor_run.py`, including the Raptor inference pipeline and generation of raw and ranked predictions for the next stage.

## 2. 🚀 CoAtNet Raptor — GPU Inference Execution

Runs the verified CoAtNet Raptor inference script in a separate Python process, releasing its GPU memory before the second model starts.

**Key components:**
- `subprocess.run()` — executes `/tmp/raptor_run.py` with unbuffered output.
- `timeout = 2 * 3600` — limits Raptor execution to 2 hours.
- **Exit code verification** — stops the notebook if Raptor inference fails.
- `raptor_sub.csv` — ranked predictions for all 12 abnormalities.
- `raptor_raw.csv` — raw model probabilities for the final fusion stage.
- `RAPTOR_SECONDS` — records the total execution time.
- `nvidia-smi` — displays GPU memory usage after completion.

**Output:** Confirms successful execution with `RAPTOR_STAGE_DONE` and verifies that both prediction files exist.

## 3. 🧠 Custom Compact Model — Source Code & Verified Assets

Prepares the author's independently trained MRI model, restores its preprocessing pipeline, and verifies all required source files and checkpoints.

**Key components:**
- `ARM = 'E_lrprior'` — selects the custom model's prediction arm.
- `BUNDLE_ID` — verifies the expected checkpoint bundle.
- `DEADLINE_HOURS = 8.3` — defines the overall notebook execution budget.
- `SOURCES` — embeds the training, preprocessing, inference, GPU controller, anatomical anchor, and laterality-check modules.
- `CODE_SHA` / `SOURCE_SHA` — verifies that the restored Python modules match their expected SHA-256 hashes.
- **Offline dependencies** — installs the supplied Python wheels without internet access.
- `rsna-knee-compact-v1` — provides the author's trained fold checkpoints and model manifest.
- **Native MRI preprocessing** — preserves the image geometry and intensity normalization used during training.
- **Optional anatomy and laterality models** — loads verified auxiliary weights when required by the manifest.
- **OrthoFoundation-L support** — reconstructs the frozen pretrained encoder from verified weight parts when selected.

**Output:** Prepares the verified model bundle and inference configuration, then prints `ASSETS_READY`.

## 4. 🔬 Custom Compact Model — Multi-GPU Inference

Executes the author's pretrained compact MRI model across available GPUs, using the verified source code, checkpoints, and inference configuration from Block 3.

**Key components:**
- `probe_dual.py` — distributes test studies across GPUs and merges worker predictions.
- `compact_infer.py` — performs MRI preprocessing and inference using the trained fold checkpoints.
- `ARM = 'E_lrprior'` — selects the custom model's prediction arm.
- `DEADLINE_HOURS = 8.3` — enforces the overall runtime limit, including the preceding Raptor stage.
- **Automatic mixed precision** — selects the appropriate GPU precision with `--amp auto`.
- `--loader-workers 2` — configures two DICOM loading workers per inference process.
- **Failure checks** — stops execution if the controller fails.
- **Inference diagnostics** — preserves GPU worker logs and records processed studies, fallback counts, and runtime.

**Output:** Saves `/kaggle/working/ours_sub.csv`, worker logs, and `submission_receipt.json`. Prints `S5_STAGE_COMPLETE` and previews the model predictions.

## 5. 🏆 Final Rank Fusion — Raptor 70% + Custom Model 30%

Combines the public CoAtNet Raptor and the author's custom compact MRI model using fixed, finding-wise percentile-rank blending.

**Key components:**
- `W_RAPTOR = 0.7`, `W_OURS = 0.3` — fixed ensemble weights, without fitting on competition labels.
- `raptor_raw.csv` — raw CoAtNet probabilities used for ranking.
- `raptor_sub.csv` — original Raptor submission used for consistency checks.
- `ours_sub.csv` — predictions from the custom compact model.
- **Study alignment** — verifies all study identifiers and all 12 target columns against the competition test set.
- `rank(method='average', pct=True)` — computes percentile ranks with proper tie handling.
- **Failure-aware blending** — uses the available model's rank when the other model fails; assigns 0.5 if both fail.
- **3% Raptor failure limit** — stops execution if too many studies fail.
- **Spearman diagnostics** — verifies Raptor ranking consistency and measures agreement between the two models.
- **Submission validation** — checks prediction ordering, finite values, and the [0, 1] range.

**Output:** Saves `/kaggle/working/submission.csv` and `blend_receipt.json`, prints `NOTEBOOK_COMPLETE`, and previews the final predictions.