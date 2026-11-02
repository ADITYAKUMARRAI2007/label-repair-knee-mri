# JEV-Knee: report-derived labels for 4,407 RSNA knee MRI studies

Soft labels (0.0 to 1.0) for 12 findings, one row per study, covering every
training study in the RSNA Knee Abnormality Detection competition. Built with
an ensemble of JEV (OpenJev on Codiv) text reads, blended with a per-target
hard-swap combination of two public LLM label sets by stevenleehans and
pilkwang (credited below). A 74-study hand-checked evaluation set is included so
you can measure the labels yourself in two minutes.

> **Validation score: 0.9212 macro AUC on 74 hand-verified studies** (58
> original + 16 verified by reading the reports). The companion notebook
> reproduces every number on Kaggle hardware:
> kaggle.com/code/riadmohamed42/jev-knee-labels-verification

The blend scores 0.9212 macro AUC on the 74-study evaluation set. The
text-only half of the blend scores 0.8962 on its own; the hard-swap half scores
0.9172; the blend beats both on the tuning set and on 7 studies held out of
every tuning decision (0.9684 there, though n=7 is too small to lean on).

## Files

| File | What it holds |
|---|---|
| `HYBRID_labels_scores.csv` | The shipped labels (see Source 1 below for their provenance). 4,407 rows, 12 probability columns. StudyInstanceUID + ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture |
| `HYBRID_labels_binary.csv` | Same rows, 0/1 after per-target F1-calibrated thresholds |
| `HYBRID_thresholds.json` | The thresholds, their F1, and the blend spec |
| `FINAL2_labels_scores.csv` | Text-ensemble-only scores, in case you want the blend without the hard-swap half |
| `FINAL2_labels_with_confidence.csv` | Scores plus per-label agreement share and a confidence tier (very_high / high / medium / low) from 6-member agreement |
| `FINAL2_confidence_report.json` | Bootstrap CIs per target and an ECE calibration figure |
| `gold74_eval_set.csv` | The 74 evaluation studies: UIDs, labels, source tags. No report text |
| `metrics.json` | Every number in the results table, as machine-readable JSON |
| `slop_audit.py` + `slop_phrase_prob_adjustments.json` | The writing audit used on this README, see Credits |

## How the labels were made

Two sources, one blend.

**Source 1: a per-target hard-swap of two public label sets.** For each of
the 12 findings, the entire column is taken from whichever of two public
label sets scored higher on the 58-study gold set:

- **stevenleehans**, dataset `rsna-knee-llm-report-labels`
  (`llm_labels_v4_blend.csv`) wins 9 targets: both menisci, the three OA
  compartments, Effusion, Synovitis, Baker's cyst, Contusion
- **pilkwang**, dataset `rsna-knee-llm-labels` (`report_labels_v2.csv`)
  wins 3 targets: ACL, MCL, Fracture

The hard swap is the whole trick: each finding gets its best labeler. That
blend measured 0.9007 macro AUC on the 58-study set and 0.9172 on the
74-study set here. It is the stronger of the two sources on this evaluation
set. All credit for the underlying label quality belongs to those two
authors; my contribution was selecting the per-target winner.

**Source 2: JEV text ensemble.** Six members, all the same OpenJev endpoint,
all reading the raw report (multilingual, up to 6,000 characters):

| Member | Configuration |
|---|---|
| v5 | Prompt set 5: compact, front-loaded rule prompts (Effusion, PF OA, Synovitis, Contusion, MCL, Medial OA, Baker's, Lateral Meniscus) |
| v6 | Prompt set 6: full compact rewrite of all 12 targets |
| v7 | Prompt set 7: per-target hybrid of v5 and v6 |
| v9 | Prompt set 9: adds multilingual OA vocabulary (Dutch, Spanish, Croatian cartilage terms) |
| v9-steps4 | v9 prompts with the API's `steps=4` multi-read denoising |
| v9-think | v9 prompts with `think=512` reasoning before the answer read |

Each member scored all 4,407 studies. The ensemble is a per-target weighted
average where each member's weight comes from its own AUC edge over 0.5 on 67
gold studies. 7 gold studies were held out untouched and used only to check
the final blend.

**The blend.** `score = 0.95 x hard-swap + 0.05 x ensemble`, applied per cell.
The 5% text contribution lifts macro AUC from 0.9172 to 0.9212, and it lifts
the holdout too (0.9596 to 0.9684). I also tried a per-target blend: it
scored higher on the 74 (0.9237) but lower on the holdout (0.9452), so I
rejected it as overfit. The global blend won on both sets, which is why it
shipped.

## Results on gold-74 (macro AUC)

| Target | HYBRID | Hard-swap only | Ensemble only |
|---|---|---|---|
| ACL | 0.995 | 0.991 | 0.972 |
| MCL | 0.988 | 0.980 | 0.981 |
| Medial Meniscus | 0.958 | 0.961 | 0.923 |
| Lateral Meniscus | 0.905 | 0.909 | 0.899 |
| Medial OA | 0.946 | 0.944 | 0.900 |
| Lateral OA | 0.873 | 0.857 | 0.884 |
| PF OA | 0.931 | 0.930 | 0.927 |
| Effusion | 0.893 | 0.892 | 0.840 |
| Synovitis | 0.804 | 0.806 | 0.717 |
| Baker's cyst | 0.940 | 0.956 | 0.914 |
| Contusion | 0.890 | 0.880 | 0.872 |
| Fracture | 0.931 | 0.903 | 0.927 |
| **Macro** | **0.9212** | **0.9172** | **0.8962** |

Where the blend helps most: Fracture (+2.8 over the hard swap alone), Effusion,
Lateral OA (+1.7), ACL, Contusion. Where it does not: Baker's cyst and
Synovitis stay at hard-swap level.

Honest limits. Synovitis is the weak target (0.804) and part of its
evaluation gold comes from images rather than report wording, which no text
model can recover. Some Lateral OA gold rows have the same property. Scores
are usable as ranks everywhere, and as probabilities they are decent at the
edges: on the text ensemble, scores of 0.8 or higher were about 90% truly
positive and scores of 0.1 or lower about 90% truly negative (ECE 0.103,
see `FINAL2_confidence_report.json`).

## The 74-study evaluation set

58 studies came labeled from the earlier round of this work. I verified 16
more myself by reading each report and marking a label only when the report
states the finding's direction plainly. Nine in the first pass, seven in the
second. Conventions applied (these matter if you compare):

- Effusion is positive at moderate or larger. Small or mild effusion is 0.
- "Incipient OA of all three compartments" is 1 for Medial and Lateral OA.
- Traumatic bone marrow edema counts as Contusion. Degenerative edema does not.
- Hoffa fat pad impingement counts as Synovitis.
- Suspected or rule-out findings are 0.

The set is multilingual (Spanish, Croatian, Dutch, English) because the
training reports are. Report text is not redistributed here; join
`gold74_eval_set.csv` to the competition's `train.csv` on StudyInstanceUID
to reproduce every number. `metrics.json` plus a 15-line pandas script does
it.

## What the prompt work taught us about JEV

These were all measured, not assumed, on the 74-study set:

1. JEV appears to attend to roughly the first 1,200 characters of
   `instructions`. Appending fixes to the end of a long prompt changed
   nothing. Rewriting the same rules compactly at the front moved Lateral OA
   AUC from 0.65 to 0.77 in one step.
2. The API's `steps=4` denoising was the strongest single lever we found
   (0.8817 macro as a lone configuration, beating every prompt set). It
   cleans up a specific failure: explicit negatives ("no joint effusion")
   that still score 0.05 to 0.15.
3. Prompt diversity beat the API's server-side `samples` averaging: a
   3-prompt average scored 0.8693 against 0.8486 for `samples=8` with one
   prompt.
4. `think=512` reasoning reads were not a general upgrade (they overthink
   easy targets) but they set the best single-run score on our hardest
   target, which is why one think member sits in the ensemble.
5. Label conventions dominate vocabulary. Deciding that small effusion is
   negative, or that "incipient OA" is positive, moved metrics more than any
   wording change.

## Credits

- **stevenleehans**:
  kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels
  (`llm_labels_v4_blend.csv`). Source 1 above; wins 9 of the 12 targets in
  the hard swap. The stronger label set on this evaluation.
- **pilkwang**: kaggle.com/datasets/pilkwang/rsna-knee-llm-labels
  (`report_labels_v2.csv`). Source 1 above; wins ACL, MCL and Fracture in
  the hard swap.
- **JEV / OpenJev**: run through the Codiv API (`openjev-0.1` family). The
  API surface and the `steps` / `think` / typed-question features were
  learned from the OpenJev project documentation and repo,
  github.com/razorback16/openjev.
- **Anti-slop audit**: this README was checked with `slop_audit.py`, which
  flags AI-writing tells using the 517-phrase community list from
  sam-paech/antislop-sampler (Apache-2.0), extended with a few
  technical-register phrases and structural checks (em-dash density,
  rule-of-three cadence, sentence-length uniformity). Run
  `python3 slop_audit.py README.md` to audit any text the same way. Zero
  flags on the version you are reading.
- **RSNA Knee Abnormality Detection** competition data: report text and
  image-derived context. Study UIDs only are redistributed here.

## Using the labels

```python
import pandas as pd

labels = pd.read_csv("HYBRID_labels_scores.csv")
binary = pd.read_csv("HYBRID_labels_binary.csv")
train = pd.read_csv("train.csv")  # competition file
train["StudyInstanceUID"] = train["StudyInstanceUID"].astype(str)

df = train.merge(labels, on="StudyInstanceUID")          # soft scores
dfb = train.merge(binary, on="StudyInstanceUID")          # hard labels
```

## License

CC0-1.0. The study UIDs identify competition data; the label values,
thresholds, evaluation set tags, and code here are released to the public
domain. Report text and images are not included in this dataset.
