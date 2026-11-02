# RSNA Knee: selected accuracy and efficiency weights

Two original learned checkpoints support the accompanying inference notebooks. The accuracy model is the original 384-trained seed42 SWA (epochs 14,11,12), scored at 0.949 with native320/K94/anatomical mirroring. The efficiency model is the 224-crop seed42 SWA (epochs 15,13,12), scored at 0.945 with K94/autocast/one view. The later crop-trained 0.946 model is not substituted. See checkpoint-manifest.json for source and checkpoint SHA-256 values.

Both models used the same 2,399 baseline OAI knees for masked external supervision. This package contains only these two learned checkpoints and aggregate checkpoint metadata. No MRI images, participant identifiers, records, reports, subject-level label tables, predictions or cached volumes are distributed. Gold58 was reused for selection and is not independent cross-validation. Public scores do not establish clinical validity or statistical significance.

## Model-use terms

These learned weights are provided for research and educational use, including the RSNA Knee competition. This model-use permission grants no rights to obtain, publish, redistribute or re-identify OAI participant data, source MRI images, reports or other controlled-access records. Access to OAI data remains governed by NDA's terms. Third-party source code and pretrained components retain their respective licences. No CC0 or other open-data licence is applied to OAI data or raw corpora by this model release.

## Source attribution and acknowledgement

The model architecture and inference lineage build on [dreaddevelopment's Raptor CoAtNet/MIL recipe](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense) and timm CoAtNet backbones. Weak-label contributors include [stevenleehans](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels), [pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels), and [riadmohamed42's JEV labels](https://www.kaggle.com/code/riadmohamed42/jev-knee-labels-verification), combined with our report-derived targets. These source tables and reports are not included here.

Data and/or research tools used in the preparation of this manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific investigation and OA drug development. Dataset identifier(s): 10.15154/0hcg-f676.

[Shared NDA Study](https://nda.nih.gov/study.html?id=3407) · [DOI: 10.15154/0hcg-f676](https://doi.org/10.15154/0hcg-f676). Official acknowledgement source: [NDA manuscript preparation](https://nda.nih.gov/nda/manuscript-preparation).
