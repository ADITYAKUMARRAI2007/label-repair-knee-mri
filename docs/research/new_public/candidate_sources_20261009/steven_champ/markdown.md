# RSNA knee — our baseline

Twelve probabilities per study: ACL and MCL injury, medial and lateral meniscal tear,
osteoarthritis in each of the three compartments, effusion, synovitis, Baker's cyst,
bone contusion, fracture. Scored as the unweighted mean of twelve per-label ROC AUCs.

Built on top of two community notebooks, `rsna-knee-eda-to-2-5d` and
`rsna-knee-baseline-v1`. Everything inherited is marked in the source comments. The
short version of what is different here:

| | source notebooks | this one |
|---|---|---|
| validation | single 1-of-5 holdout | grouped 5-fold, OOF over every study, folds ensembled |
| annotated reference | scored over **all** gold studies, including trained ones, and that number picks the checkpoint | gold held out with its own fold; only held-out gold scores |
| report labels | rules only, *or* TF-IDF fitted to the 58 annotated studies | rules **distilled** into a text model fitted to all 4,407, gated per target |
| augmentation | vertical flip | small in-plane affine — a vertical flip swaps femur and tibia |
| backbone | `pretrained=False`, silently | resolves real weights, and shouts if it cannot |
| fusion | mean of probabilities | rank mean, with weak components gated out on OOF evidence |

Three bugs in the inherited report extractor are fixed in place, each of which silently
inverted a label: a bare `Fractures :` heading scoring positive when the next line reads
`Aucune.`, `disruption` failing to match "the ACL is disrupted", and German compounds
(`Innenmeniskusriss`, `Kreuzbandruptur`) failing to match the tear vocabulary at all.

## 1. What the metric decides

$$\text{Score} = \frac{1}{12}\sum_{i=0}^{11} \mathrm{AUC}_i$$

Two consequences, and both remove a design choice rather than adding one.

**Only order matters.** $\mathrm{AUC}_i$ is invariant under any strictly increasing map
of column $i$, so calibration is worth nothing and a threshold is worth nothing. It also
settles how to combine models: averaging probabilities lets whichever model is most
confident dominate, averaging ranks combines the only information the metric reads.
Every combination below is a rank mean.

**Every label costs the same.** A label left at chance forfeits $(M - 0.5)/12$ of the
score no matter how well the other eleven do — about $0.029$ at $M = 0.85$. Rare
findings therefore deserve *more* attention than common ones, because a rare finding is
where a model most easily ends up at chance.
## 0. Run settings

Kaggle has no way to pass environment variables into a notebook, so the knobs are set
here instead — this is the only cell meant to be edited.

**`RSNA_FOLDS` ships at `1` on purpose.** A full five-fold run is roughly 6 hours of
GPU, and a run that dies at hour five on some data quirk costs a large slice of the
weekly quota for nothing. Prove it works on one fold first, then raise it.

**`RSNA_TRAIN_SEED` is deliberately separate from `CFG.seed`.** It drives only head
init, batch order, group choice and augmentation. Folds come from a hash of the report
and the derived targets come from `CFG.seed`, so both stay byte-identical when this
one moves. Run the same config twice with two values and the gap between the two OOF
scores is the noise floor — the margin any later change has to beat before its result
means anything.
## 2. Where the targets come from

`train.csv` has a `Report` column and `test.csv` does not. Text is available when
fitting and absent when predicting, which rules out any fusion model with a text branch
and leaves the reports usable only as a source of training targets and of sample
weights.

Two facts shape the extractor:

**Reports are graded, annotations are thresholded.** The reporting radiologist and the
annotator do not share a cutoff — a report saying *small joint effusion* can sit against
a negative annotation. So a rule of the form *term present ⇒ positive* is wrong by
construction; grading the mention (trace / unqualified / marked) is right, and costs
nothing, because §1 established that only order is read.

**Silence is not a negative.** A finding the report never mentions gets a low score and
a low *confidence*, and the confidence becomes the sample weight. A study whose report
says nothing about synovitis pulls on the synovitis head far less than one that names it.
### The lexicon hole, and what to do about it

The dangerous failure of a rule extractor is silent: a rule that never fires does not
raise, it emits a negative. A lexicon thick in English and thin in Greek does not look
broken — it looks like a corpus where Greek patients have fewer findings. And because
language tracks the reporting institution, which tracks the scanner and the population,
that gap is a bias aligned with a site rather than noise that averages out.

The fix here is to fit a character n-gram model **to the rule scores**, over all 4,407
reports, out of fold. `char_wb` 3–5 needs no tokeniser, stemmer or stopword list, so
Greek and Turkish are handled on the same footing as English; the model picks up the
phrasings that co-occur with rule-positive reports and scores them without anyone having
written them into the lexicon.

The precise claim matters, because the stronger one is false. The model cannot find
findings in a language the lexicon misses *entirely* — its only supervision is the rule
output, so with no variation there is nothing to learn. What it repairs is **partial**
coverage, which is the real situation. `tests/test_distillation.py` measures exactly
this: on a corpus where 64% of positive reports use phrasings outside the lexicon, the
rules sit at chance (0.500) on those reports and the blend reaches 0.940.

Each target is gated on how well the text model recovered the rules out of fold. Below
a correlation of 0.15 it contributes nothing rather than noise — the negative control in
that test file confirms the gate fires when coverage is zero.

The other trap is on the way back. Combining as ranks is right, but reading the combined
rank back through the *sorted* rule scores re-quantises everything into the rules' four
atoms, and since most mass sits in one enormous "silent" atom, that throws away precisely
the ordering the text model was added to supply. Interpolating between the distinct
levels instead keeps the prevalence and the meaning of 0.5 while preserving the ordering.
## 3. Reading the acquisition

`train_series.csv` gives each series a plane and two flags, `Fluid_Sensitive` and
`Fat_Suppression`. The EDA notebook found those two columns are **byte-identical across
all 24,371 rows**.

They name two physically independent things. *Fluid sensitivity* is a property of the
contrast weighting, set by $T_R$ and $T_E$:

$$\text{weighting} = \begin{cases}
T_1 & T_R \lesssim 800\,\text{ms}\\
T_2 & T_R \gtrsim 800\,\text{ms},\ T_E \gtrsim 60\,\text{ms}\\
\text{PD} & T_R \gtrsim 800\,\text{ms},\ T_E \lesssim 60\,\text{ms}
\end{cases}$$

*Fat suppression* is a preparation applied on top of any weighting — a chemically
selective pulse, STIR, or water excitation — and it is what makes marrow oedema
conspicuous. Two identical columns carry one bit between them, not two, so recovering
both axes from the DICOM header is the difference between six slots and three.

Two string-matching traps, both of which silently invert the answer:

- underscore is a word character, so a token test for `we` (water excitation) never
  fires inside `t2_de3d_we_tra`. Separators must be normalised first.
- `ScanOptions` must be matched as **exact tokens**: one vendor writes `SAT_GEMS` for
  *spatial* saturation, so a substring test for `SAT` marks non-suppressed series as
  suppressed.

| slot | plane | weighting | fat sat | what it carries |
|---|---|---|---|---|
| `SAG_FLUID_FS` | sagittal | PD / T2 | yes | meniscal tears, marrow oedema, effusion |
| `COR_FLUID_FS` | coronal | PD / T2 | yes | collateral ligaments, meniscal body |
| `AX_FLUID_FS` | axial | PD / T2 | yes | patellofemoral joint, synovium |
| `SAG_FLUID_NOFS` | sagittal | PD / T2 | no | meniscal morphology at high CNR |
| `COR_T1` | coronal | T1 | no | marrow architecture, bone outline |
| `SAG_T1` | sagittal | T1 | no | anatomy, chronic change |
## 4. Sampling at a fixed physical scale

A slice of $N\times N$ pixels at spacing $s$ mm/pixel covers $Ns$ millimetres. Both vary
widely here, so a fixed-pixel resize hands the encoder images whose physical scale
differs by about $3\times$, and a meniscus occupies a different number of pixels in
different studies for no anatomical reason.

Cropping to a constant physical extent $L$ first and resampling to $P\times P$ leaves

$$s_\text{eff} = L/P \ \text{mm/pixel}$$

independent of the acquisition — $160\,\text{mm} / 224 = 0.71$ mm/pixel here.

Intensity needs the same treatment: MR has no absolute scale, so each series is
normalised to its own 1st–99th percentile over the whole volume (not per slice, so
slices keep their relative contrast), which removes an offset that would otherwise track
the site.

### Laterality

Four targets are medial/lateral pairs, and medial is defined against the body midline —
so which side of the *image* it falls on depends on which knee was scanned. The
correction differs by plane: coronal and axial mirror under a horizontal flip, but a
sagittal stack does not, because there the medial–lateral direction is the *slice* axis
and what differs is the order the stack traverses the joint. Where `Laterality` is
absent the volume is left alone — a wrong flip is worse than no flip, and the presence
mask lets the head learn how much to trust each slot.
## 5. Aggregating slots into twelve decisions

A study arrives as up to six slot embeddings $x_s$ with a presence mask $m_s$. Pooling
them identically would discard the reason the protocol has three planes at all. Give
every diagnosis $o$ its own query and let it attend over the slots, absent ones masked
out of the softmax:

$$h_s = \phi(x_s) + e_s,\qquad
\alpha_{o,s} = \frac{\exp(\langle h_s, q_o\rangle/\sqrt{H})\, m_s}
{\sum_{s'} \exp(\langle h_{s'}, q_o\rangle/\sqrt{H})\, m_{s'}}$$

The masked softmax renormalises over whatever the study actually contains, so a missing
axial series shifts a diagnosis's attention onto the sequences that are present rather
than feeding it a zero vector.

**The head is deliberately this small.** The label is attached to the *study*, so
nothing in the supervision says which part of it carries the finding. Extra attention
parameters have no signal to learn that from and spend their capacity on noise. Where
the supervision is coarse, the aggregation should be too.

**Why the encoder is trained rather than frozen.** A frozen self-supervised encoder is
the cheap option, and the way it stops paying is informative: resolution, encoder size,
slice coverage and slot aggregation each move the score by less than validation noise.
That pattern is diagnostic — those four axes change how much the model looks and how it
summarises, but none changes the *vocabulary* it looks with. So the last blocks are
adapted, at a learning rate two orders of magnitude below the head's, because the head
is random at initialisation and the encoder starts from a good solution.
## 6. Validating without fooling yourself

**Shared reports.** Some reports are byte-identical across studies — a template read for
an unremarkable knee — and every study in such a group gets the same derived target.
Split the group across folds and the model is scored on a target whose source it trained
on. Folds are assigned by a hash of the report text, which keeps duplicates whole.

**Two references, two meanings.** Performance is reported against the derived targets,
which cover every study and measure whether the imaging model learned what the text
says; and against the annotated studies, which are far fewer and measure agreement with
a reading of the *images*. The second resembles the test set. The first has enough
positives per label to tell a real difference from noise.

**The annotated reference has to be held out.** This is where the source notebook leaks:
it scores the annotated AUC over *every* gold study, including the ones in the training
split, and then uses that number to choose the checkpoint. Here gold studies take their
fold assignment like everything else and only held-out gold is scored.

**Choosing an epoch.** The obvious rule — best epoch on the larger reference — is wrong,
because the two measure different things and an epoch that gains on one while losing on
the other has been shown to be *different*, not better. Ranking by

$$\text{score}(e) = \min\big(\mathrm{AUC}^\text{derived}_e,\ \mathrm{AUC}^\text{annot}_e\big)$$

refuses that trade, and makes training length a measured quantity rather than a guess.
## 7. Run

A valid `submission.csv` is written before anything else and overwritten only once real
predictions exist: a submission that never writes scores nothing at all, which is
strictly worse than scoring badly, and a kill for memory is a SIGKILL that no
`try`/`except` will catch.

Knobs worth touching, all via environment variables so nothing below needs editing:
`RSNA_FOLDS` (default 5 — drop to 2 if the run is time-bound), `RSNA_TIME_BUDGET`
(seconds, default 8 h), `RSNA_BACKBONE` (a timm name, overrides the DINOv2 search),
`SLOT_SCHEME=public` (ablate the header recovery back to the provided flags).