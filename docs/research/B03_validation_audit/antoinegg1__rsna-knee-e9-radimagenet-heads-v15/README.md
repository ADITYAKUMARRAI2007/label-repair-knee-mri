# RSNA Knee E9 RadImageNet heads

These five query heads were trained by private Kaggle kernel version 15 on the public
RSNA Knee training images. The frozen encoder is the official RadImageNet ResNet-50
checkpoint, SHA-256 `08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734`,
released under CC-BY-NC-SA-4.0. Training uses three independently published CC0 report-label
tables; complete official image labels override weak labels for the 58 available studies.

`radimagenet_arm_v15.py` is the exact attributed training/inference source embedded in that
kernel. `rad_e9_audit.json` is the remote gate receipt. `v52_oof.csv` is the grouped five-fold
OOF table and `v52_radimagenet_heads.pt` contains the five head state dictionaries. The primary
visible output remained byte-identical to E2 because Fracture regressed under an all-target
blend.

E9b does not retune the 20% public blend weight. It applies that fixed vote only to the ten
targets whose AUC improved in both the independently obtained public OOF run and this account's
fresh remote OOF run. Baker's and Fracture remain exactly on E2. The exact contract and artifact
hashes are in `e9b_contract.json`.

The generated heads and accompanying artifacts inherit CC-BY-NC-SA-4.0. Competition images are
not redistributed.

## E10: the weight ladder E9 could not see

E9 searched the blend weight on a grid that stopped at `0.25`, and four of its five outer
folds selected that ceiling. The deployed `0.20` was therefore a censored search, not an
optimum. `e10_contract.json` re-runs the same nested, grouped-fold, dual-source protocol on
an extended ladder (`0.20/0.35/0.50/0.60/0.70`) and records the per-target gain of every rung
in both independent OOF runs, a 3000-draw bootstrap against `0.20`, and each deployable
weight map. `alpha_ladder_diagnostic.py` is the exact source that produced it.

At `0.20` the contract reproduces E9b's published `0.850343` (public) and `0.851948`
(this account's v15) exactly, which anchors the new ladder to the already validated parent.
`0.35` is the largest uniform weight at which all ten E9b targets still improve in both runs.
Baker's and Fracture receive no vote at any rung. Every number is a 58-study out-of-fold
diagnostic on official training labels, not a Kaggle score.

### Did E10 tune on the 58 gold rows?

Yes, and the contract reports the honest held-out form of that same choice.
`uniform_alpha_nested_selection` picks the rung on four grouped folds and scores the fifth.
All five outer folds choose `0.60` on the public run and `0.70` on this account's v15 run,
with nested macro `0.862110` and `0.873005` against a base of `0.840009`. No fold ever picks
`0.20`, and none picks anything at or below `0.35`. Every rung at or below `0.35` is therefore
more conservative than what unbiased selection would choose, and the deployed rung is
re-derived and re-gated inside the kernel before it is allowed to replace the parent.
