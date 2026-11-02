# RSNA Knee E11 Diverse RadImageNet Heads V20

These five query heads were trained by private Kaggle kernel version 20 on the public
RSNA Knee training images. The frozen encoder is the official RadImageNet ResNet-50
checkpoint, SHA-256 `08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734`,
released under CC-BY-NC-SA-4.0. Three public report-label teachers provide weak
supervision; complete official image labels override them for the 58 available studies.

E11 is deliberately different from the parent RadImageNet arm. It reads sagittal,
coronal, and axial non-fat-suppressed slots plus one sagittal fat-suppressed anchor,
uses a 130 mm physical crop, and retains eight slices per slot at 224 pixels. Version 20
filled 72.0% of all slots, trained five report-group-safe folds, and completed on a Tesla
T4 in 2,065 seconds after the E2 parent inference. Its primary output remained byte-for-byte
equal to E2 because the run was training-only.

`radimagenet_arm_v20.py` is the exact final cell pulled from kernel version 20.
`rad_e11_audit.json` and `kernel.log` are its remote receipts. `v52_e11_oof.csv` is the
five-fold OOF table and `v52_e11_heads.pt` contains the five head state dictionaries.
`e11_contract.json` records the three-arm rank grid, all twelve target AUCs, artifact
hashes, held-out selection, and deployable weight maps.

The generated heads and accompanying artifacts inherit CC-BY-NC-SA-4.0. Competition
images are not redistributed.
