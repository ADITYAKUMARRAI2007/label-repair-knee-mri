# C13 public-score failure diagnostic

Observed 6 October 2026, 20:49 IST. Root cause remains unresolved.

## Reproduction and scope

- Submitted kernel: `adityakumarrai0401/rsna-knee-c13-distilled-coatnet`, version 2, submission `56881626`.
- Official outcome: COMPLETE, public AUC **0.563**, no reported execution error. This agrees with the user's screenshot.
- Receipt: `runs/submission_monitor_b03_c13_20261006/result.json`.
- Checkpoint: `raptor_ft_coatnet_v5_distilled_swa.pt`, SHA-256 `00f5cf10d69afcf3ef4a65656fe84652c109c3f0718eae10167b5cc82eb0de23`.
- Embedded author-reported Gold58 AUC: 0.9132610080477783. The public result is 0.350261 lower; the embedded score has not been independently reproduced.
- Observed submission-to-result elapsed time: 122.4 minutes, including queueing. This is not a measured GPU runtime.

## What the evidence establishes

The visible execution loaded all 486 checkpoint tensors with `strict=True`, checked the architecture and target labels, and used the expected checkpoint hash. Its three raw probability vectors were finite and distinct within every target; their range was approximately 0.206–0.628. The saved CSV rankings agree exactly with those probabilities. These facts rule out a visibly missing classifier, a visible label-order mismatch, and visible NaN contamination. They do not establish hidden prediction accuracy, full DICOM decoding, or equivalence to the author's forward and preprocessing functions.

The classifier follows the public Raptor forward: encode windows, apply LayerNorm, softmax target attention over windows, pool features, apply `clsW` and `clsb`. Bounded backbone microbatches are used in evaluation mode. Version 1 produced nonfinite fp16 predictions before competition submission; version 2 switched to fp32 and passed the visible numeric checks. Fixing numerical execution did not establish correctness of the complete scoring pipeline.

A restricted tensor reader inspected the saved classifier and normalization tensors and all batch-normalization running means and variances. The inspected values were finite, and every inspected running variance was positive. Several variances are small (minimum approximately 1.34e-6), but this alone does not establish corruption, an epsilon mismatch or an input-scale bug. The scope and statistics are saved in `runs/C13_standalone/checkpoint_numeric_diagnostic.json`; this is not a complete inference-equivalence audit.

## Unresolved hypotheses

1. **Input or forward contract mismatch.** The package provides no training/inference source or validation predictions. The 64-slice layout, 140 mm crop, ImageNet normalization, depth span and 62-window selection were inferred from a related MaxSpan Raptor pipeline. C13's checkpoint does not embed its slots. The comment in the submitted script describing those slots as checkpoint-embedded is inaccurate; the prior provenance audit already documented that inference. Strict state loading cannot validate these assumptions or runtime normalization configuration.
2. **Checkpoint/export or batch-normalization state mismatch.** The SWA metadata names epochs `[0, 3, 2]`, but its averaging and normalization-statistics procedure cannot be reproduced from the package. The finite buffer inspection neither proves nor disproves this hypothesis.
3. **Hidden decode or prediction issues.** The reader silently catches individual slice failures and uses zero planes. The visible `fallback_studies: 0` means no study-level substitute predictions, not that every selected DICOM decoded. Hidden raw probabilities and decode counts are unavailable from the current receipt.
4. **Validation provenance or transfer failure.** The embedded AUC has no split manifest or saved prediction artifact. Its claimed evaluation conditions and independence are unverified. A 0.563 public score alone cannot distinguish this from implementation mismatch.

None of these hypotheses is a demonstrated root cause. C12's 0.911 result from a related reader does not prove that C13's input contract is correct.

## Decision and required repair evidence

- Reject C13 from every blend. Keep the 0.943 scored anchor and await B03.
- Preserve the submitted version and all receipts; no speculative correction or additional competition submission.
- Require an authoritative training/inference contract, then reproduce held-out validation with exact study identity, target order, raw logits/probabilities, decode counts and checkpoint hash before reconsidering C13.
- Investigate normalization and batch-normalization configuration against that contract, rather than replacing buffers or trying transform/weight sweeps against the leaderboard.
- Require a reproduced validation improvement and a qualifying official standalone result before any later blend. The author's embedded AUC and a three-study smoke run are insufficient.

The remaining read-only observer monitors B03 `56881506` under exec session `19307`, output directory `runs/submission_monitor_b03_20261006`. No Kaggle job was restarted and no additional competition submission was made during this diagnosis.
