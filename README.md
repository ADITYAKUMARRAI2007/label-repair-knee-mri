# RSNA Knee MRI Abnormality Detection (Kaggle 2026)

Multi-label detection of **12 knee-MRI findings** (ACL tear, meniscal tears, medial/lateral/patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, contusion, fracture, …) from multi-sequence DICOM studies, scored by **macro ROC-AUC**, with **offline inference under 9 hours on 2×T4**.

**Best submission: public LB 0.958 → private LB 0.944** · [final notebook on Kaggle](https://www.kaggle.com/code/adityakumarrai0401/rsna-knee-label-ft-coatnet-specialists-blend) · 6,000+ teams

## Approach
A per-finding **rank blend** of strong public models, plus our own **label-centred** contributions:

| Component | What it adds |
|---|---|
| Public CoAtNet (384/224 px, mirrored-knee view), compact R6 with flip TTA, 2.5D ConvNeXt reader | A strong, diverse base (credited in the notebook) |
| **CoAtNet fine-tuned on relabelled findings** | Labels rebuilt from images plus radiology reports for lateral OA and synovitis, where the original labels were noisiest |
| **Retrained CoAtNet + 384 weight soup** | Per-finding best labels; seed soup for stability |
| **Patellofemoral-OA specialist** | Axial + mid-sagittal ConvNeXt with pixel-exact preprocessing, 25% on the PF-OA column |
| **Gold-set gating** | Every change had to win on a hand-checked 58-study set with bootstrap confidence before submission |

## Results
| Version | Public | Private |
|---|---|---|
| Public recipe, unchanged | 0.958 | 0.942 |
| + label-fine-tuned CoAtNet | 0.958 | 0.943 |
| + PF-OA specialist, fracture weighting | 0.958 | 0.943 |
| + retrained CoAtNet, per-finding labels | 0.958 | 0.944 |
| **+ 384 weight soup (final)** | **0.958** | **0.944** |

The public leaderboard was flat at 0.958, but **every label-focused change held or improved the private score**.

## What we learned
* **Labels were the bottleneck, not architectures.** New standalone arms scored well below the blend (EfficientNetV2-S 0.877, ResNet34 student 0.890, MedicalNet-3D 0.687 public AUC) and never earned blend weight; label work did.
* **Small gold sets mislead below about 0.003 AUC.** A 58-study gate caught large regressions but could not rank tiny changes.
* **Pixel-exact preprocessing parity** between training and Kaggle inference had to be verified explicitly; small mismatches silently cost AUC.
* Full reasoning is in [`docs/research`](docs/research), with a reusable checklist in [`docs/CV_CHALLENGE_PLAYBOOK.md`](docs/CV_CHALLENGE_PLAYBOOK.md) and every run logged in [`docs/experiments.csv`](docs/experiments.csv).

## Repository layout
```
src/scripts     data audits, label building, Kaggle packaging and evaluation tools
src/training    training code for the arms we tried (CoAtNet, ConvNeXt specialists, readers, …)
src/inference   offline inference and blending
notebooks/      the exact final submission notebook (private 0.944)
docs/           research reports, playbook, experiment log
```

**Not included:** competition data, derived labels and model weights. The competition rules don't allow redistributing the data; get it from the [competition page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

## Credits
The base blend uses public work by **goodpjw2008**, **heliosli**, **nartaa**, **rostigor** and **varshithakolupuri**; see the notebook for links. Our code is released under the Apache-2.0 licence.
