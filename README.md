# Labels over architectures: 12-finding knee MRI classification

**Multi-sequence knee MRI · 12 findings · macro ROC-AUC · offline, 9 hours on 2 × T4** · held-out private score **0.944** ([final notebook](https://www.kaggle.com/code/adityakumarrai0401/rsna-knee-label-ft-coatnet-specialists-blend))

<p align="center"><img src="assets/system.svg" width="100%"></p>

**Question.** The training labels for several findings (lateral and patellofemoral osteoarthritis, synovitis) disagree with what is visible in the images. With a strong public baseline already in place, is it better to add new architectures or to repair the labels?

**What we did.**
* Rebuilt the noisiest labels from **images plus radiology reports**, then fine-tuned a CoAtNet on them.
* Retrained a CoAtNet on the best label source per finding (with a seed **weight soup**).
* Added a **patellofemoral-OA specialist** (axial + mid-sagittal views, pixel-exact preprocessing).
* Blended everything **per finding**.
* Every change had to win on a **hand-checked 58-study gold set** with a bootstrap test before submission.

## Findings

<table><tr>
<td width="50%"><img src="assets/architectures_vs_labels.svg"></td>
<td width="50%"><img src="assets/public_vs_private.svg"></td>
</tr><tr>
<td><sub><b>New architectures alone did not compete.</b> A 3D MedicalNet, EfficientNetV2-S and a distilled ResNet34 scored 0.69–0.89 standalone and never earned blend weight.</sub></td>
<td><sub><b>Label work generalised.</b> The public leaderboard stayed flat at 0.958 while every label-focused change held or improved the held-out private score (0.942 → 0.944).</sub></td>
</tr></table>

* **Small gold sets have a resolution limit.** 58 studies caught regressions but could not rank changes below about 0.003 AUC; one change that looked +0.002 on the gold set scored lower on the leaderboard.
* **Preprocessing parity is a silent failure mode.** Training and inference preprocessing had to be verified pixel-exact; mismatches cost AUC without any error.

Full reports are in [`docs/research`](docs/research), a reusable checklist in [`docs/CV_CHALLENGE_PLAYBOOK.md`](docs/CV_CHALLENGE_PLAYBOOK.md), and every run in [`docs/experiments.csv`](docs/experiments.csv).

## Code
```
src/scripts     audits, label building, packaging and evaluation
src/training    training code for every arm we tried
src/inference   offline inference and blending
notebooks/      the exact final notebook (private 0.944)
docs/           research reports, playbook, experiment log
```
Not included: competition data, derived labels and weights, which the competition rules don't allow redistributing ([data here](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)).

## Credits
The base blend uses public models by goodpjw2008, heliosli, nartaa, rostigor and varshithakolupuri (linked in the notebook). Our code: Apache-2.0.
