# Path from 0.956 toward 0.960 — evidence and experiment order

## Score arithmetic

The competition score averages AUC across twelve classes. Moving a confirmed 0.956 to 0.960 requires about **+0.048 summed class-AUC gain**. A single-class specialist would need approximately +0.048 in that class if every other class stayed fixed. A 10% correction to one class, such as submission 57184339, is unlikely to supply that alone. The actual public class AUCs are unavailable, so this is a gain budget, not a score forecast.

## What existing experiments say

1. R7 already uses 384-pixel ConvNeXt-Small, four folds, three-slice input, spatial attention, physical spacing, OA extent loss, pair ranking, and class-specific state mechanisms. Small changes to learning rate, weight decay, epochs, EMA, or blend coefficients have no demonstrated route to +0.004 macro score.
2. Five grouped Phase 5 folds gave a regional reader +0.00568 macro AUC over a separately trained global control on **weak report labels**, paired interval +0.00354 to +0.00780. The largest class contrasts were medial meniscus +0.0234, lateral meniscus +0.0162, contusion +0.0116, and synovitis +0.0073. These sum to +0.0585 against that Phase 5 control, but **cannot be transferred to R7 or the public leaderboard**. Gold58 diagnostic AUC for the Phase 5 reader was generally below R7, so replacement is not supported.
3. Lateral OA crop versus broad view gained +0.0197 on the clear weak-report proxy, with an interval crossing zero. A clear-label-only training change gained +0.0038 versus the prior crop head, also uncertain. The new local-pair head alone gained +0.0088, uncertain; its fixed blend with the prior crop head gained +0.0221, all five folds positive. This suggests anatomical coverage and complementary representation matter more than arbitrary optimizer tuning, but the latter blend was designed after observing the standalone result and uses repeatedly explored folds.
4. R7's reused Gold58 errors concentrate in lateral OA, PF OA, synovitis, and lateral meniscus. Several leading OA/synovitis cases have report descriptions that do not straightforwardly express the image-defined target. Thus weak report-label optimization can reward the wrong changes. Gold58 cannot be used again as a weight-tuning set.

## Priority 1: Multi-class regional signal, using existing work first

The strongest route to +0.004 macro is to improve several classes, not one. Audit the existing Phase 5 regional reader's checkpoints and full inference cost, then design **one fixed low-weight combination** focused on medial/lateral meniscus, contusion, and possibly synovitis. Do not run a weight grid. The four Phase 5 class gains are promising against its matched global control, while complementarity to R7 remains unproven because exact R7 fold-exclusive OOF predictions are unavailable. Wait for the two pending OA submissions before another public-score experiment; then use at most one frozen regional candidate if operational and label checks pass.

## Priority 2: One image-sampling hyperparameter experiment

The first useful hyperparameters to change are **physical crop coverage and adjacent-window density** in the meniscal and OA compartments. They alter whether a small lesion enters the model. Hold backbone, labels, optimizer, folds, epochs, and inference aggregation fixed while comparing the new sampling to the existing regional view. Use a 20-study native-DICOM crop QA first, including laterality, pixel spacing, meniscal horns, cartilage surfaces, and missing series. If coverage is valid, run one fold as a resource pilot and then the same five grouped folds for the frozen hypothesis. A region-localized meniscus detector has clinical precedent; this does not establish a score gain in this competition.

## Priority 3: Label quality, especially synovitis

Synovitis has only 564 high and 125 low weak targets in the current 4,407-study manifest; most are ambiguous. Noncontrast MRI makes synovial lining and effusion difficult to separate. Verify sequence types and obtain blinded image adjudication or a trustworthy independent image-labeled set before a synovitis specialist. A different threshold or loss weight alone is unlikely to solve this label mismatch.

## Low-priority knobs and stop rule

Do not run an unfocused grid over learning rate, weight decay, epochs, dropout, backbone name, test-time augmentation, or twelve blend coefficients. Repeated selection on five folds, Gold58, or the public board can overfit the selection process. A useful experiment changes one causal factor, has a matched control and group-held-out predictions, and freezes the rule before evaluation. If it fails the paired gain or acquisition checks, stop rather than enlarge the search.

The two-class ROI correction, submission 57179067, completed at displayed public score 0.956. The one-class lateral-pair correction, submission 57184339, is pending. Confirmed best: 0.956.

## Primary research references

- *Radiology* 2018 cartilage-lesion study: localized articular-surface MRI patches with radiologist labels, [PubMed](https://pubmed.ncbi.nlm.nih.gov/30063195/).
- *Diagnostic and Interventional Imaging* 2019 meniscus study: meniscal horn localization followed by tear detection, [journal article](https://www.sciencedirect.com/science/article/pii/S2211568419300555).
- Meniscal localization and lesion classification using sagittal and coronal MRI, [journal article](https://www.sciencedirect.com/science/article/pii/S1120179721000983).
- Cawley and Talbot, *JMLR* 2010: model-selection overfitting from noisy validation criteria, [paper](https://www.jmlr.org/papers/v11/cawley10a.html).
- Knee synovitis assessment on noncontrast qDESS versus contrast MRI, [research article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7881494/).
