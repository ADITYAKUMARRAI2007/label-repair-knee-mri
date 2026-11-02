# E02 result and comparison with the published ConvNeXt reader

Observed 6 October 2026. E02 submission `56881026` scored **0.890** public AUC; its strict Gold58 checkpoint validation AUC was **0.8952057**. B01/B02 remain the best owned, officially scored submissions at **0.943**. E02 is excluded from final blends.

## Concrete input weakness

E02 stores five MRI sequence slots in one 44-slice array with widths `12, 10, 8, 6, 8`. Its `StudyWindows` picker checks that three adjacent array indices are populated, but does not check that they belong to the same acquired MRI sequence. Boundaries are at indices `12, 22, 30, 36`.

For a study with all five slots present, 8 of 42 candidate windows cross a sequence boundary. The deterministic 32-window evaluation selects 6 of those windows: centres `12, 21, 22, 29, 30, 35`. For example, centre 22 uses the last image of the second sagittal sequence together with the first two coronal images. These are not three adjacent anatomical planes from one series.

The existing slot-coverage audit reports 3,166 studies with all slots, 117 missing only the second sagittal slot, 1,122 missing only the second coronal slot, and two missing both. Applying the implemented evaluation rule to these presence patterns gives approximately **17.07%** windows crossing sequence boundaries across the training corpus. The calculation and limits are saved in [the boundary audit](../runs/E02_inference/sequence_boundary_audit.json).

This proves an input representation weakness. It does not prove how much AUC would be recovered by fixing it. Both E02 training and inference used this representation, so changing only inference would create a different input distribution for the existing checkpoint. A correction needs a retrained and independently evaluated model.

## Corpus depth-span mismatch confirmed by author metadata

The corpus author's authenticated dataset descriptions, retrieved later on
6 October, explicitly state a **15–85%** slice span for both
`dreaddevelopment/knee-raptor-corpus` and `knee-raptor-corpus-ext`. E02 trained
on these cached pixels, while its submitted `build_study` reconstructs **6–94%**
with `int(n * 0.06)` and `int(n * 0.94) - 1`. The inference comment saying that
this wide span matches the corpus is incorrect for the datasets actually used.
The author distinguishes this earlier cache from later wider-span work.

This is a separate train/inference mismatch from the sequence-boundary bug.
Its causal score effect has not been measured, and the official 0.890 result
does not resolve it. Existing scored submissions are preserved. E03 continues
training on the actual 15–85% cache; its future decoder must match that span.
The private CPU job `adityakumarrai0401/rsna-knee-e03-cache-pixel-parity` v1
compares reconstructed pixels directly with the cache before inference is
considered. Descriptions and build metadata are saved in `research/e03_corpus`.

## Differences supported by source code

The published reader sources were downloaded from `goodpjw2008/rsna-knee-2-5d-convnext-reader`; source hashes are recorded in `new_public/goodpjw_weights/source/source_manifest.json`. These are implementation facts, not an ablation proving which change creates the score gap.

| Feature | E02 | Published ConvNeXt reader |
|---|---|---|
| Slice windows | 16 train / 32 evaluation across concatenated slots | 8 train / 12 evaluation per present series; three slices always belong to the same series |
| Sequence information | No explicit slot or slice-position token | Six plane/fat-suppression slots and relative slice-position embeddings |
| Study model | Mean/max features plus top-quartile slice logits | Two transformer layers followed by separate attention pooling for each finding |
| Backbone adaptation | ResNet34 with stem and first residual stage frozen | ConvNeXt Tiny with all encoder parameters in the optimizer |
| Spatial preprocessing | 140 mm centre crop without canonical in-plane reorientation | Canonical orientation, 0.4 mm physical pixel spacing, 153.6 mm cache field of view |
| Augmentation | Intensity jitter, noise and horizontal flip | Shared per-series rotation, scale, translation and intensity transforms |
| Weight averaging | Fixed average of epochs 15–18 | EMA throughout training, final EMA checkpoint |
| Labels | 75% teacher / 25% JEV blend in logit space | Default source code averages four report-label tables in probability space |
| Ensemble | One trained model | Public package exposes three independently trained folds; author also reports five-fold experiments |

Both exclude all 58 Gold studies from gradient updates. The reader's public notebook reports about 0.93 standalone and repeated 0.943–0.944 stack blends. Those results remain source-author evidence until our B03 submission is scored. The architecture, labels and geometry change together, so the comparison cannot isolate the benefit of any one component.

## Decisions

1. Reject E02 from the final blend based on its official 0.890 score. The small 0.0052 validation-to-public gap does not suggest a large generalization collapse, although the two evaluation sets differ.
2. Wait for B03, C12 and C13 before the next submission. B03 already includes the published reader, so copying its architecture again would add little new diversity by itself.
3. If a new owned model becomes necessary, preserve sequence boundaries, add explicit sequence/position information, use grouped folds, and validate a complete retrained model. A controlled boundary-correction comparison would identify whether that specific E02 weakness materially contributes to the gap; it is not a reason to submit the old checkpoint with altered inference.
4. Do not infer 0.95 capability from an architectural fix or a checkpoint's embedded validation score. Further submissions require measured standalone strength and evidence that their errors complement the current anchor.

The original boundary investigation launched no job. The subsequent E03
training and CPU pixel-parity checks are documented in `e03_series_reader.md`;
they do not constitute new competition submissions.
