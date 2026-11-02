# RSNA Knee Template

**Dataset-driven submission template for RSNA Knee Abnormality Detection**

This Notebook is intentionally kept thin.

The submission logic does **not** live in the Notebook itself. Instead, the Notebook loads and executes a versioned Kaggle Dataset containing the inference pipeline.

That means the normal workflow is:

```text
develop / modify logic locally
        ↓
build dataset.zip
        ↓
create a new version of the logic Dataset
        ↓
attach / select that Dataset version in this Notebook
        ↓
Save Version / Run All
        ↓
submission.csv
```

To try a different model, ensemble, preprocessing rule, inference optimization, or post-processing method, you normally update the **Dataset contents**, not this Notebook.

This follows the same idea as a reusable submission launcher: keep the Kaggle Note stable and make the attached Dataset the replaceable implementation.

## Why use this pattern?

- The public Notebook stays small and easy to audit.
- Large inference logic does not need to be pasted into Notebook cells.
- Local development can use normal `.py` files and version control.
- Changing a submission candidate only requires a new Dataset version.
- The same Notebook can execute many different candidate implementations as long as they follow the Dataset contract below.

## Kaggle execution contract

This competition uses Notebook submissions. The pipeline executed by this template must ultimately create:

```text
/kaggle/working/submission.csv
```

The attached implementation must also be compatible with the competition's Internet-disabled Notebook environment and runtime limits.

## Logic Dataset contract

The Notebook expects the attached logic Dataset to expose
the following files after Kaggle extracts the uploaded archive:

- `rsna_knee_manifest.json`
- `rsna_knee_pipeline.py`

The manifest must declare:

- `bundle_id`: `rsna-knee-local-logic`
- `runner`: `rsna_knee_pipeline.py`

The Notebook executes the runner once. The runner is responsible
for generating `/kaggle/working/submission.csv`.

The Notebook does not inspect or extract ZIP archives and does
not perform SHA256 or other checksum-based file identity checks.

The uploaded ZIP filename is not part of the execution contract.
Only the extracted files and their interface matter.

## Origin and credits

This template was created from the inference workflow used in **Bend the Knee to Speedy Raptors**, which itself builds on **Bend the Knee to the Dinosaurs** and other public RSNA Knee work.

Credits include:

[Jiwei Liu](https://www.kaggle.com/code/jiweiliu/rsna-knee-fast-2xt4-inference) ·
[Pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-weights) ·
Sofia Anjenje · Antoine G. · prvsiyan · Marwan Mahmoud ·
dreaddevelopment / Roman Tamrazov ·
[renta.k](https://www.kaggle.com/code/renta0426/rsna-knee-hybrid-raptor-lateral-meniscus-r100) ·
[Anvith Pothula](https://www.kaggle.com/code/anvithpothula/rsna-base)


## Logic Dataset contract

The recommended setup is one dedicated Kaggle Dataset for the submission logic.

The **Dataset name/slug may be anything**. What this Notebook identifies is the manifest inside it.

### Recommended Dataset contents

A convenient way to upload the implementation is as a ZIP archive containing these two files:

```text
dataset.zip
├── rsna_knee_pipeline.py
└── rsna_knee_manifest.json
```

After Kaggle extracts the archive, the files are exposed to the Notebook as:

```text
<your Kaggle Dataset>
├── rsna_knee_pipeline.py
└── rsna_knee_manifest.json
```

Alternatively, the same two files may be uploaded directly to the Dataset root.

The launcher reads **only the extracted or directly available files**. It does not open, inspect, or extract ZIP archives.

### `rsna_knee_pipeline.py`

This is the executable submission logic.

It may contain preprocessing, model loading, inference, ensembling, post-processing, and submission generation.

When executed by this Notebook, it runs in the Notebook's global Python namespace. Its final responsibility is to create:

```text
/kaggle/working/submission.csv
```

The current development bundle generates this file from multiple local source files, but the Kaggle-side contract remains a single executable `rsna_knee_pipeline.py`.

### `rsna_knee_manifest.json`

The manifest identifies the logic bundle and tells the launcher which runner file to execute.

Minimal example:

```json
{
  "bundle_id": "rsna-knee-local-logic",
  "schema_version": 1,
  "runner": "rsna_knee_pipeline.py"
}
```

Required by this launcher:

- `bundle_id` must be `rsna-knee-local-logic`.
- `runner` must be `rsna_knee_pipeline.py`.

Additional descriptive metadata, such as build version, commit identity, or model description, may be included in the manifest. Such metadata is not used for file identity or checksum validation.

## How to build `dataset.zip`

A minimal local build can be done with Python's standard library:

```python
from pathlib import Path
import json
import zipfile

runner = Path("rsna_knee_pipeline.py")
runner_bytes = runner.read_bytes()

manifest = {
    "bundle_id": "rsna-knee-local-logic",
    "schema_version": 1,
    "runner": "rsna_knee_pipeline.py",
}

with zipfile.ZipFile(
    "dataset.zip",
    "w",
    compression=zipfile.ZIP_DEFLATED,
) as z:
    z.writestr("rsna_knee_pipeline.py", runner_bytes)
    z.writestr(
        "rsna_knee_manifest.json",
        json.dumps(manifest, indent=2).encode("utf-8"),
    )
```

Then:

```text
1. Develop and test the logic locally.
2. Build a new dataset.zip.
3. Upload it as a new version of the logic Dataset.
4. Attach/select that Dataset version in this Notebook.
5. Keep the Notebook code unchanged.
6. Save Version / Run All.
7. Confirm that submission.csv is produced.
```

Public model weights and other reusable Kaggle inputs do **not** need to be copied into this logic Dataset. They can remain separate Notebook inputs. The logic Dataset is intended to carry only the code/configuration that you want to swap between submissions.

---

## Launcher

The cell below:

1. searches only the shallow Kaggle input directories;
2. finds exactly one compatible logic bundle;
3. reads and validates its manifest;
4. loads the fixed runner `rsna_knee_pipeline.py`, as declared by the manifest;
5. executes `rsna_knee_pipeline.py`.

If zero or multiple compatible bundles are found, execution stops rather than guessing which one to use.
