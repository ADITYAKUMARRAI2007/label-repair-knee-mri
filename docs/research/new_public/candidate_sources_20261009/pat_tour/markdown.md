# RSNA 67knee receipts and weights tour

Shows how to read the two public artifacts of our RSNA Knee Abnormality Detection work:

- dataset **rsna-67knee-run-receipts**: JSON receipts our inference notebooks write next to `submission.csv`;
- model **rsna-67knee-coatnet2** (variations `t4`, `t5`): CoAtNet-2 fold-0 state_dicts plus `run_config.json`.

No competition data is attached; everything runs on CPU in under a minute.
## Run receipts

`fusion_config.json` holds the blend weights and window stride of one fusion run; `f1_receipt.json` says whether our leg was blended into the public stack and with what weight.
## CoAtNet-2 weights

Each variation ships a state_dict and the preprocessing settings it was trained with.