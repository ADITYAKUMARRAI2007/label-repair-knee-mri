# 🦴 RSNA Knee 2026: SOTA Triple-Reader Apex Stack (LB 0.951+ / Rank <150)
## **Dual CoAtNet Spatial Pyramid (384px + 224-Crop) + ConvNeXt 2.5D MIL Reader**

**Author:** `KRISH RABARI (@rabari9999)` | *RSNA Knee Abnormality Detection AI Challenge 2026*

---

## 🌟 Executive Summary: Breaking the 0.950 Leaderboard Barrier

While standard 2-model ensembles (CoAtNet 384 + ConvNeXt 2.5D) hit an unrounded asymptote at `~0.9502`, this **Triple-Reader Architecture** breaks into **Rank <150 (LB 0.951+)** through three complementary pillars:

1. **Dual-Scale CoAtNet Spatial Pyramid**:
   - **CoAtNet 384px SWA (Native Crop, 94 Windows)**: Captures global joint-space alignment, cartilage erosion, and subtle cortical fractures.
   - **CoAtNet 224-Crop SWA (Localized High-Density Crop)**: Specializes in localized cruciate ligaments (ACL/MCL) and meniscal tear roots.
   - **Spatial Fusion**: CoAtNet 384 (`70%`) + CoAtNet 224 (`30%`) provides an unprecedented high-resolution anchor.

2. **Orthogonal Volumetric 2.5D MIL Reader**:
   - 3-Fold ConvNeXt 2.5D MIL Reader captures through-plane signal continuity across consecutive DICOM slices.
   - Provides massive empirical lifts on volumetric & fluid-filled pathologies:
     - **Baker's Cyst**: ConvNeXt is dominant (`0.35` weight).
     - **Bone Contusion**: 3D trabecular marrow edema patterns (`0.30` weight).
     - **Medial OA & ACL Tear**: (`0.25` weight each).

3. **Calibrated Per-Target Rank Fusion**:
   - Non-parametric percentile rank fusion weighted strictly by clinical finding strengths.
   - 100% deterministic, offline Kaggle GPU compatible, 0% fallback rate.

---

## 📊 Model Weights & Routing Configuration

| Pathology Target | CoAtNet Pyramid Weight | ConvNeXt 2.5D Weight | Dominant Clinical Rationale |
| :--- | :---: | :---: | :--- |
| **Baker's Cyst** | `0.65` | **`0.35`** | High fluid volume; ConvNeXt slice context dominates |
| **Bone Contusion** | `0.70` | **`0.30`** | 3D trabecular bone marrow edema patterns |
| **Medial OA** | `0.75` | **`0.25`** | Medial tibial-femoral joint-space wear |
| **ACL Tear** | `0.75` | **`0.25`** | Sagittal oblique ligament fiber continuity |
| **Medial Meniscus** | `0.80` | **`0.20`** | Sagittal bow-tie slice evaluation |
| **MCL Tear** | `0.80` | **`0.20`** | Coronal collateral ligament coverage |
| **Lateral Meniscus** | **`0.92`** | `0.08` | CoAtNet 384px fine spatial resolution |
| **Fracture** | **`0.92`** | `0.08` | Subtle cortical bone discontinuities |
| **Lateral OA** | **`0.92`** | `0.08` | Lateral joint compartment preservation |
| **PF OA** | **`0.92`** | `0.08` | Patellofemoral axial sequence anchor |
| **Effusion** | **`0.96`** | `0.04` | CoAtNet is near-perfect ($>0.986$ AUC) |
| **Synovitis** | **`0.97`** | `0.03` | CoAtNet text-supervised report pretraining |

---

> ### 🤝 Acknowledgments & Lineage
> This notebook builds upon foundational open-source contributions from the RSNA Knee community:
> - **CoAtNet 384px & 224-Crop SWA Checkpoints:** Learned weights by `nartaa` & architecture by `dreaddevelopment`.
> - **2.5D ConvNeXt MIL Reader:** Checkpoints by `goodpjw2008` & baseline by `sujanmajhisuzan`.
> - **Apex Variant B Formulation:** Formulated by `ranjeet258`.
