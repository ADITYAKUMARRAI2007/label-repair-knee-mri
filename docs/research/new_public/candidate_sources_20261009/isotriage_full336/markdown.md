# RSNA Knee Abnormality Detection - training notebook

Recipe taken from the current top of the public leaderboard and discussion tab:

- **Labels**: the twelve findings are weak labels extracted from the free-text
  report (multilingual rule extractor, per-finding confidence) with the 58 gold
  studies weighted `GOLD_WEIGHT` above every derived row. A report that never
  mentions a finding is an unknown at low confidence, not a negative.
- **Images**: 130 mm physical centre crop, canonical laterality from geometry,
  `CACHE_IMG` uint8 cache, three consecutive slices in the 0.20-0.80 band of
  each plane stacked as the three encoder channels, six slots
  (Sagittal/Coronal/Axial x fluid-sensitive / structural).
- **Model**: DINOv2 ViT-S/14, last `UNFREEZE_LAST` blocks open, slot-attention
  head with one query per finding and an anatomical slot prior.
- **Optimisation**: AdamW, head 1e-3 / backbone 8e-6, weight decay 0.02,
  confidence-weighted BCE plus a pairwise ranking term, `EPOCHS` per fold,
  macro-AUC on the held-out fold, best state saved.
- **Budget**: one notebook run stays inside `TIME_BUDGET_S` on a single GPU,
  internet off, competition data plus the attached DINOv2 weights only.