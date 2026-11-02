# RSNA R7 versus scored Candidate C — local audit

Source inspected: the public notebook `rsna-knee-0-956-raptor-convnext-r7-ensemble.ipynb`.
The notebook title claims public AUC 0.956. Its saved execution output covers **three visible test studies** and reports 403.6 seconds total, zero R7 compact fallbacks, and zero CoAtNet fallbacks. The displayed notebook output does not contain a Kaggle score receipt.

## Architecture comparison

- R7 reuses the exact CoAtNet source and checkpoint already used in Candidate C. The CoAtNet source SHA256 is `663e33fcf6022c7470fc0fa3d23e8d57292dc147e2c07609ad5e960588a3162a`; the recorded checkpoint SHA256 is `7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`.
- R7 uses a different compact bundle ID, `8c7b666e27dc0e8f`, versus the old compact bundle `209a32c8b50a6b1c`. Its `compact_train.py` and `compact_infer.py` source hashes also differ. The public [R7 manifest](r7_manifest.json) confirms four new ConvNeXt-Small checkpoint hashes, a fluid stacker, and a state stacker. The full checkpoints are not stored locally; their bytes and the original training run have not been independently replayed here.
- None of the four R7 checkpoint hashes match any of the four older compact checkpoint hashes. The R7 manifest reports four folds, 16 epochs, spatial attention, synovitis decomposition, pair ranking and OA extent training. Those are manifest claims, not an independently reproduced training run.
- The downloaded notebook's final formula is 50% percentile rank of CoAtNet predictions plus 50% percentile rank of R7 compact predictions for all 12 classes, with explicit handling for failed studies. Candidate C is 70% N02 plus 30% older compact, where N02 uses CoAtNet and an original three-checkpoint ConvNeXt reader.

## Completed checks

- Replayed the downloaded R7 50/50 formula from its displayed compact output and our saved CoAtNet visible output. Maximum difference from the author's displayed final output: `3.34e-7`, consistent with display rounding.
- On the three visible studies, the R7 compact and old compact percentile ranks differ in 4 of 36 study/target cells. This establishes different outputs on that tiny cohort; it says nothing about accuracy.
- Created a frozen exploratory blend, `0.80 × R7 final + 0.20 × Candidate C final`, in [blend_r7_c.py](blend_r7_c.py). It requires aligned, complete prediction CSVs and refuses duplicate IDs, missing studies, nonfinite values, or values outside `[0,1]`.
- The blend's visible [CSV](visible_audit/c20_r7_80_visible.csv) contains three rows, valid 12-column predictions, and differs numerically from Candidate C in 28 of 36 cells. The standalone CSV mode reproduced this output exactly.
- A private [exact R7 reproduction notebook](../../public_kernels/r7_exact_reproduction/inference.ipynb) preserves all five original code-cell sources and clears stale outputs. It has not been pushed or run on our Kaggle account.

## Decision

**Exploratory candidate only.** There are no R7 predictions for an eligible labeled validation cohort or the full hidden test in this workspace. The notebook's displayed three studies have no labels. Therefore, neither the published 0.956 claim nor the proposed blend's ability to exceed 0.956 has been confirmed on our account. No competition submission was made.

The next informative comparison is to establish the unmodified R7 notebook's official score and preserve its full component outputs, then compare a single frozen C/R7 blend. Avoid fitting weights to the reused Gold58 or treating low correlation alone as evidence of improvement.
