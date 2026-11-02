# Public RSNA Baseline: consolidated reproduction

This package reproduces the model recipe in `evgendvorkin/rsna-baseline` version 15,
downloaded on 2026-09-12. Only models used by that public notebook are included.
It updates the existing dataset `tonylica/rsna-knee-bend-dinov3-0917-repro-assets`
and notebook `tonylica/rsna-knee-dino-radimagenet-rank-ensemble`.

Verified source LB: **0.941** (Tony's completed submission `56173926`, 2026-09-11).
The repackaged version has not yet been competition-scored. **0.942 is a target**.
There is no promise of a leaderboard improvement from packaging changes.

## Inventory

41 unique checkpoint members: 20 DINOv2-small, five DINOv3-small, five reference
RadImageNet heads, five E13 RadImageNet heads, three public Raptor CoAtNet
checkpoints and three residual-gated CoAtNet epoch snapshots. The Rad heads share
one ResNet50 encoder. The DINOv2 initialization weights are bundled too.
One frozen 12-output linear calibration table completes the recipe.

The E13 heads run on their primary layout and on the E11 layout with original and
horizontal-flip pixels. Raptor v5 runs in normal and reverse-channel views. These
are additional views of existing checkpoints, not additional learned models.
The source's five alternative Rad heads have zero influence on the final output
and are omitted. DINOv2-base, E11 heads, labels, figures, training fallbacks and
discarded native/DINO predictions are not required inference assets.

## Exact workflow

`pipeline/transformer_rad.py`: DICOMs → 20 DINOv2 percentile-rank votes → five
DINOv3 fold-rank votes → 55/45 blend → reference/E13 Rad rank blend → second E13
layout/flip vote → frozen calibration → `branches/transformer_rad.csv`.

`pipeline/public_raptor.py`: v5/v10/v5-reverse/v8 probabilities at 60/10/10/20 →
original per-target ordinal ranks → `_raptor.csv`.

`pipeline/residual_coat.py`: use the verified OpenCV 4.12 wheel and public artifact
contract to run epochs 4/6/8 → equal mean of checkpoint percentile ranks →
`branches/residual_coat.csv`. This is an epoch ensemble, not three folds.

`pipeline/blend.py`: H = rank(0.60 rank(public Raptor) + 0.40 rank(residual CoAt)).
Final finding = rank((1-w) rank(transformer/Rad) + w H), where w is .75 for ACL,
Lateral OA and Fracture; .80 for Medial Meniscus; 1.00 for Lateral Meniscus;
and .60 for the remaining seven findings. Every row, column and value is checked
before `submission.csv` is atomically written. See `recipe.json` for exact weights.

All original window pooling, image geometry, checkpoint loading and rank order
are retained. The complete 20-member DINOv2 frontier is mandatory; time-dependent
partial voting and neutral-score fallback are disallowed. The unused jitter
branch is removed because the selected public frontier uses original windows.
Stage isolation releases GPU memory between model families.

DINOv2 architectures are constructed from the bundled configuration, then each
complete competition state dictionary is loaded strictly. This avoids redundant
initialization-weight loading and a stall observed with the threaded pretrained
loader in Kaggle's serving environment. Fingerprint checks remain mandatory. The
original backbone weights are retained in the bundle for provenance and comparison.

Long runs do not use a timed asynchronous traceback dump. A 49-study Kaggle
diagnostic reproduced a process segmentation fault at the first 180-second timer
tick in the prior release. Fatal-signal reporting remains enabled; all model,
asset and submission integrity checks are unchanged.

## Reproduce inference on Kaggle

1. Open the canonical notebook version associated with this dataset version.
2. Attach the competition and this one asset dataset. No other model, dataset or
   notebook-output attachment is needed. Keep Internet off and select T4 × 2.
3. Run all cells. The pinned manifest and every payload file must pass SHA-256
   checks before any model loads. Never replace a mismatched file to bypass a gate.
4. Confirm `run_receipt.json` says `VALID_SUBMISSION` and inspect `submission.csv`.
5. To score it, select that notebook version and `submission.csv` in Kaggle's
   Submit to Competition flow. No submission is made automatically.

The image is pinned in `kernel-metadata.json`; the residual renderer additionally
installs its exact bundled OpenCV wheel offline. All runtime code is readable in
`pipeline/`. The upstream notebook snapshot and calibration coefficients are
included. Notebook generation and asset staging live in `build.py` in the project
source; edits belong in that builder or its templates, followed by regeneration.

The public dataset also contains all generator inputs under `reproduce/`. Copy
that folder to a writable location and run:

```bash
python reproduce/build.py --asset-root /path/to/published-dataset \
  --stage /path/to/rebuilt-dataset --dataset-version 5
```

This regenerates the runtime sources, manifest and notebook without GitHub access.
The command only writes local files; it neither publishes nor submits.
Use the actual matching dataset version number when rebuilding a later release.
The explicit version pin prevents Kaggle from reusing an older attached input;
the embedded manifest hash provides an independent verification.

## Scope of reproduction

This is a complete checkpoint **inference** reproduction. The public residual
CoAtNet package also includes its trainer snapshots, history and validation
contracts, retained unchanged because its loader verifies them. Other models'
upstream training sources and original data contracts remain with their authors;
full independent retraining of every checkpoint is not claimed. No training,
private experimental checkpoint or competition submission is part of this run.

Visible-test success validates execution and schema; it does not establish the
new version's hidden-test leaderboard score. Small visible-test ranks can change
with GPU arithmetic, so stage hashes and formula checks are retained as evidence.
