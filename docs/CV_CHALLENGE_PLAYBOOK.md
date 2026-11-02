# CV / Medical-Imaging Challenge Playbook

*Compiled 2026-10-21 from published Kaggle winner write-ups, grandmaster guidance, the "Bag of Tricks" paper, and our own measured results in the RSNA Knee Abnormality Detection competition. "Measured" marks evidence we produced ourselves. Everything else is published practice from others.*

---

## 0. How to use this file

1. Read §1 (principles) before writing any code.
2. Follow §2 phase by phase.
3. Pick techniques from §3 in the stated **order of value**. Don't skip ahead to ensembling or tuning before data, labels and validation are right.
4. Use §6 to choose final submissions.
5. §7 records what worked and what failed in RSNA Knee, so we don't relearn it.

---

## 1. Core principles (what every top solver repeats)

| Principle | What it means in practice |
|---|---|
| **Trust your CV, not the public LB** | The public LB is often a small slice of the test set (sometimes 5–20 %). Chris Deotte picked his best-CV submission while it ranked ~120th on the public LB; it finished **6th** on private. Select models by validation that mirrors the hidden test. |
| **Validation must mirror the test split** | Group by patient, site, scanner or time if the test is grouped. Never use random K-fold on grouped data. Freeze folds in a file and reuse them for every experiment. |
| **Data and labels beat architecture** | In RSNA Knee, the 0.949 team's biggest gain was **more training data (+0.013)**, while switching backbones lost 0.02+ (measured by them; see §7). |
| **Localise, then classify** | Every recent RSNA 1st place (lumbar 2024, abdominal 2023, cervical 2022) found the anatomy first, cropped it, then classified the crop with a standard backbone. |
| **Diversity is what makes ensembles work** | Averaging near-identical models adds little or hurts. In RSNA Knee, averaging a retrained CoAtNet with the original **lost 0.002–0.004** because they were 97 % correlated (measured). |
| **Know why a submission works** | Only choose submissions with good CV whose gain you can explain. Unexplained LB jumps usually vanish on private. |

---

## 2. Phase-by-phase playbook

### Phase A: Triage (first hours)
- Read the full description, data page, rules and evaluation metric. Save them to `NOTES.md`.
- **Write the metric locally first** and unit-test it on edge cases. A wrong metric silently ruins everything.
- Record runtime limits (GPU type, hours, internet off), external-data rules and submission format.
- Check the leaderboard shape: how many teams, how tight the top is, what the public notebooks score. If public notebooks sit near the top, the race is about labels, data and small edges.

### Phase B: Data audit
- Shapes, label distribution per class, missing values, duplicates between train and test, ID formats.
- **Find the hidden split structure**: patients, sites, scanners, vendors, time. What will make test differ from train?
- **Name the nuisance factor** (scanner, pixel spacing, slice thickness, which series exist, left/right side) and **canonicalise it**: resample to fixed mm, mirror to one side, fixed series-selection rules.
- For medical imaging, check: series types per study (plane, fluid-sensitive or not), slice counts, pixel spacing, missing series, orientation.
- Look at images of the **worst-predicted cases** early. Errors usually cluster by acquisition type or by label ambiguity.

### Phase C: Validation design
- GroupKFold by patient (or study). 5 folds, saved to `folds.csv`.
- If an **expert-labelled subset** exists (like Gold58), treat it as the gold-standard check, but know its noise: with ~58 cases, differences under ~0.008 AUC are not detectable (measured: bootstrap SD of blend differences ~0.004).
- **Estimate the CV-to-LB ratio** from early submissions. In RSNA Knee, expert-set gains shrank to roughly **⅕ on the LB** (measured).
- Simulate public/private splits by resampling your validation set to see how much the ranking of your candidates could flip.

### Phase D: Strong baseline → first entry
- A complete end-to-end pipeline first: preprocessing → model → inference → submission file, dry-run on a fresh copy of the data.
- Use a proven backbone at moderate resolution and a standard recipe (§3.4). Don't innovate before the baseline works.
- Submit early to calibrate the CV-to-LB relation.

### Phase E: Climb (order of value)
1. **Labels and data**: better labels, more data, cleaning (§3.1).
2. **Input representation**: resolution, slice density, crops of the right region, right planes (§3.2–3.3).
3. **Training recipe**: schedule, augmentation, EMA/SWA, loss (§3.4).
4. **Ensembling for diversity**: seeds, folds, then *different* strong models (§3.6).
5. **Per-target blending**: decide the blend per class using reliable validation evidence (§3.7).
6. Hyperparameter tuning **last**.

---

## 3. Technique catalogue

### 3.1 Labels and data (the biggest lever)
| Technique | When it helps | Notes |
|---|---|---|
| **Better labels for training** | Labels are weak, derived from reports or noisy | Measured in RSNA Knee: same model, only labels changed → **+0.025 AUC** (2 seeds, P 0.99). Choose the label source **per class** by agreement with expert labels. |
| **Soft labels / probability targets** | Labels are uncertain | Train with BCE on soft targets (e.g. 0.8 for "suspected"). |
| **Knowledge distillation (teacher → student)** | A strong teacher exists; labels are noisy | Train the student on the teacher's soft predictions mixed with labels. Self-distillation can learn correct labels despite noise (published research). |
| **Pseudo-labelling** | Unlabelled or extra data available | Train a strong model, predict on unlabelled data, add high-confidence predictions as training data. Do it **fold-safely**, so no fold's pseudo-labels come from a model that saw that fold. Used in RANZCR CLiP winners. |
| **Label cleaning (cleanlab, co-teaching)** | Many wrong labels | Co-teaching trains two networks that each pick likely-clean samples for the other. cleanlab flags likely label errors. Use cautiously: on rare classes it can remove true positives. |
| **More data** | Always, if legal | The 0.949 team: 3,155 → 4,349 studies gave **+0.013**. Check external-data rules carefully (the OAI legality in RSNA Knee was never resolved). |

### 3.2 Localisation and crops (the RSNA-winner pattern)
- **Lumbar 2024 (1st):** predict slice + coordinates for each spine level, then classify multiview crops (sagittal + axial 2.5D stacks) with ConvNeXt-S / EffNetV2-S + attention MIL + bi-LSTM. Attention MIL alone improved the LB 0.37 → 0.35.
- **Abdominal 2023 (1st):** 3D segmentation → crop each organ's bounding box → 2.5D CNN (CoaT / EffNetV2-S) + GRU per organ, with auxiliary segmentation loss.
- **Cervical spine 2022 (1st):** segment C1–C7 → crop each vertebra → per-vertebra classifier (14,000 samples from ~2,000 studies).
- **Pattern:** a common backbone + **crop to the anatomy** + **sequence/attention pooling across slices**.
- **Caveat (measured):** our knee compartment-crop model, trained on low-detail 44-slice images, scored 0.876 on its findings vs 0.934 for the blend. A crop model only helps if the crops are accurate and the images are full resolution.

### 3.3 Medical-imaging input handling
- **2.5D windows**: stack 3 adjacent slices as RGB channels so ImageNet-pretrained 2D backbones see local 3D context. Pool across windows with attention (MIL).
- **Slice density matters for small findings.** Measured: the same CoAtNet scored Fracture **0.925 on 44-slice images vs 0.958 on 96-slice images**. Small, local lesions (fractures, bruises, tears) need dense slices. Spread-out findings (synovitis, effusion, OA) tolerate sparse ones.
- **Train and test on exactly the same preprocessing.** Measured: fine-tunes on one image format improved Fracture by +0.02 on that format, but lost it on the submission format. Rebuild the training cache with the **exact test-time builder**, and verify pixel parity.
- Fixed physical field of view (e.g. a 140 mm crop), percentile windowing per series (2–98 %), INTER_AREA resizing, consistent series-selection rules (plane + fluid-sensitive flag).
- **Mirror to a canonical side** (left/right knees) or use anatomical mirror TTA. Generic flips and rotations are *not* valid symmetries in fixed-orientation anatomy (the 0.949 team: generic flip TTA 0.940 → 0.939).

### 3.4 Training recipe (Bag of Tricks + winners)
| Trick | Detail |
|---|---|
| **Cosine / OneCycle LR** with warm-up | Cosine decay to ~0; 5–15 % warm-up. |
| **Separate learning rates** | Backbone ~10× lower than head when fine-tuning. |
| **AdamW**, weight decay ~0.01–0.06 | The Adam family dominates winning solutions. |
| **bf16 / fp16 mixed precision** | Check hardware: T4 and P100 have **no bf16**; use fp16 there (measured: bf16 crashed on P100). |
| **EMA or SWA** | Average weights over the last epochs or keep an EMA; usually a free +0.002–0.005. |
| **Label smoothing** | ε ≈ 0.05–0.1 for hard labels; skip if already using soft labels. |
| **Class imbalance** | pos_weight in BCE (clip to ~1–10) or focal loss. |
| **Augmentation** | Mild for medical images: small affine, ±10 % intensity, gamma. Finishing the last epochs without augmentation often helps. |
| **Gradient accumulation** | Effective batch ~16 when only 1 study fits in memory. |
| **Progressive resizing** | Train at lower resolution first, finish at full resolution. |

### 3.5 Test-time augmentation
- Use only **symmetries that are true for the data** (anatomical left/right mirror for knees). Average probabilities.
- Multi-crop or multi-window TTA only if validation shows a gain.

### 3.6 Ensembling
| Method | Evidence |
|---|---|
| **Multi-seed averaging of the same improved model** | Reduces run-to-run luck. Measured spread between two identical fine-tune runs: ~0.004 AUC. |
| **K-fold models averaged** | Standard; each fold model sees different data. |
| **Diverse architectures** | Gains only if each member is strong *and* different. The 0.949 team: CoAtNet + Swin + EffNetV2 looked like +0.004 on the expert set but **+0.001 on LB**. |
| **Rank averaging** | For AUC, average ranks rather than probabilities; it makes models with different calibration comparable. |
| **Hill climbing / stacking** | Greedy weight search or a level-2 model on out-of-fold predictions (Deotte's GPU hill climbing). Needs good OOF predictions; risky with tiny validation sets. |
| **Don't average a better model with its worse parent** | Measured: retrain + original averaged lost 0.002 vs per-finding picking. |

### 3.7 Per-target decisions (multi-label tasks)
- Decide **per class** which model or weight to use, but **only on reliable evidence**: bootstrap the validation set and require the better component to win in ≥90 % of resamples.
- Make moderate shifts (e.g. weight 0.5 → 0.7), not the validation-optimal extreme.
- Measured wins in RSNA Knee: PF OA (R6 beats Raptor, P 0.98) and Fracture (Raptor beats R6, P 0.99) each gave about +0.001 on the expert set.
- **AUC is threshold-free**, so threshold tuning and calibration change nothing. Spend effort on ranking quality.

---

## 4. Compute and workflow tricks
- **Build heavy data caches once** (e.g. parallel CPU jobs on Kaggle), store compressed shards, and download them to the training machine with signed URLs and parallel downloaders (aria2).
- **Save checkpoints off the training VM** (Drive, Kaggle dataset) before releasing it. We lost trained weights twice by releasing first.
- Keep a **blocking watcher cell** so Colab doesn't recycle an idle VM mid-training.
- **Guard every inference stage**: back up the submission, restore it on any error or late clock, and assert ≥90 % of studies were scored.
- Before submitting, run a **check job on validation cases through the exact submission pipeline**. Measured: it caught 2 bugs (multiprocessing module registration; bf16 on P100) that would have silently disabled a component.
- If one GPU queue (T4) is stuck, try another (P100) for check jobs.

---

## 5. Things that did NOT work in RSNA Knee (measured, don't repeat)
- **Add-on models weaker than the blend** (0.89–0.92 alone) blended at 10–20 %: ~0 gain each.
- **External OAI/MOAKS specialist models** (several designs): ≤ +0.002 on the expert set; **−0.001 on LB**. The public weights already contained OAI.
- **Fine-tuning a strong model on a different image format** than the test format: gains on small findings lost.
- **Fine-tuning while holding strong classes fixed (self-distillation)**: this also froze the backbone, so weak classes couldn't improve.
- **Averaging highly correlated models**: lost 0.002–0.004.
- **Generic geometric TTA** on anatomy: slight loss.
- **Threshold or calibration tweaks** for AUC: no effect.
- **DINOv2 / generic ViTs** on this task: 0.63–0.70 alone.

---

## 6. Final-submission selection and shake-up survival
1. **Pick by validation, not public LB.** Use public LB only to confirm the CV-to-LB direction.
2. **Two picks = best + diverse hedge.** One pick is your strongest validated version; the other is a genuinely different model family. Kaggle scores the better of the two on private.
3. **Avoid public-LB-tuned picks.** Teams that tune on a small public LB suffer the biggest shake-downs.
4. **Estimate shake-up**: resample validation to see how often your top candidates swap order. If they swap often, prefer the more robust one (multi-seed, fewer post-hoc choices).
5. **Know the medal maths.** For 1,000+ teams: bronze top 10 %, silver top 5 %, gold top 10 + 0.2 %.

---

## 7. RSNA Knee: evidence summary and current state (2026-10-21)

| Step | Expert-set (Gold58) | Public LB / rank |
|---|---|---|
| Public 0.958 recipe (B) | 0.9385 | 0.958, rank 555 |
| + CoAtNet fine-tuned on better labels (Lateral OA, Synovitis) | 0.9407 | 0.958, rank 414 |
| + PF OA trusts R6 (P 0.98) | 0.9418 | 0.958 |
| + PF OA specialist (axial + mid-sagittal) | 0.9426 | 0.958, rank 420 |
| + Fracture trusts Raptor (P 0.99) | 0.9437 | pending (57485439) |
| + **full d96 retrain on per-class best labels** (ACL, Medial OA, Effusion, Synovitis, Contusion) | **0.9450** | pending (57491833) |

**Diagnosis per finding (measured):**
- **Label-limited** (model already better than its labels): Synovitis, Lateral Meniscus, Fracture, Effusion, Contusion, Baker's, Medial OA → better labels help.
- **Image-limited** (labels 0.94–0.96, models 0.83–0.92): PF OA → needs better views, not labels.
- **Noise-dominated**: Lateral OA (only 11 positives in Gold58).

**What would help next (ranked by evidence):**
1. Retrain the 224 CoAtNet on the d96 cache with the same per-class labels (same lever that just passed).
2. A second seed of the d96 384 retrain, averaged with the first (variance reduction; helps private robustness).
3. Only with more time: a strong, *different* model family trained on the same d96 cache and labels (the only route to bigger ensemble gains), or a full-resolution compartment-crop model.

---

## 8. Checklists

**Before training**
- [ ] Metric implemented and unit-tested.
- [ ] Folds frozen (grouped); expert/gold subset excluded from training.
- [ ] Training preprocessing **identical** to test-time preprocessing (pixel-parity check).
- [ ] Label source chosen per class by agreement with expert labels.

**Before submitting**
- [ ] Check job on validation cases through the exact submission pipeline.
- [ ] Guard: backup + restore on error; assert row count, IDs, no NaN.
- [ ] Hardware precision correct (fp16 on T4/P100).
- [ ] Runtime fits the limit with margin (stages have clock guards).
- [ ] Decision rules fixed **before** looking at results; evidence logged.

**Final picks**
- [ ] Best validated version selected.
- [ ] Diverse hedge selected.
- [ ] Both selected in the Submissions tab before the deadline.

---

## Sources
- [The Kaggle Grandmasters Playbook (NVIDIA)](https://developer.nvidia.com/blog/the-kaggle-grandmasters-playbook-7-battle-tested-modeling-techniques-for-tabular-data/)
- [Grandmaster Pro Tip: stacking with cuML (NVIDIA)](https://developer.nvidia.com/blog/grandmaster-pro-tip-winning-first-place-in-a-kaggle-competition-with-stacking-using-cuml/)
- [Kaggle Handbook: fundamentals to survive a shake-up](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-fundamentals-to-survive-a-kaggle-shake-up-3dec0c085bc8)
- [Kaggle Handbook: tips & tricks to survive a shake-up](https://medium.com/global-maksimum-data-information-technologies/kaggle-handbook-tips-tricks-to-survive-a-kaggle-shake-up-23675beed05e)
- [Chris Deotte, 52nd silver write-up (public LB vs private)](https://www.kaggle.com/competitions/jigsaw-toxic-severity-rating/writeups/chris-deotte-52nd-silver-medal-rapids-forest-infer)
- [Chris Deotte, 1st place GPU hill climbing](https://www.kaggle.com/competitions/playground-series-s5e5/writeups/chris-deotte-1st-place-gpu-hill-climbing)
- [Bag of Tricks for Image Classification (He et al., CVPR 2019)](https://arxiv.org/pdf/1812.01187)
- [The Kaggle Book: pseudo-labelling](https://www.packtpub.com/en-us/product/the-kaggle-book-9781801817479/chapter/modeling-for-tabular-competitions-9/section/pseudo-labeling-ch09lvl1sec52)
- [A Guide to Pseudolabelling (PyData keynote)](https://github.com/stanleyjzheng/PyData-Pseudolabelling-Keynote)
- [Image segmentation tips from 39 Kaggle competitions](https://dev.to/jakubczakon/image-segmentation-tips-and-tricks-from-39-kaggle-competitions-l97)
- [Secret ingredients found in Kaggle competitions 2021–2023](https://medium.com/@sblaizer/secret-ingredients-found-in-kaggle-competitions-from-2021-to-2023-47c6294a42e4)
- [Co-teaching: robust training with extremely noisy labels](https://arxiv.org/html/1804.06872v3)
- [Collection of Kaggle solutions (faridrashidi)](https://github.com/faridrashidi/kaggle-solutions)
- [RSNA 2024 Lumbar Spine discussion](https://www.kaggle.com/c/rsna-2024-lumbar-spine-degenerative-classification/discussion/540091)
- [RSNA 2023 Abdominal Trauma 1st place (GitHub)](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)
- [RSNA 2022 Cervical Spine 1st place write-up](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/writeups/qishen-ha-1st-place-solution)
- [Performance of the winning algorithms, RSNA 2022 Cervical Spine](https://pubs.rsna.org/doi/full/10.1148/ryai.230256)
