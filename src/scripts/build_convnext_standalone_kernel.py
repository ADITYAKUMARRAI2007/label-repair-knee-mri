#!/usr/bin/env python3
"""Build a standalone Kaggle inference script for the T01 ConvNeXt arm."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "inference/convnext_diversity_cell.py"
WORKER = ROOT / "inference/convnext_decode_worker.py"
OUTPUT_DIR = ROOT / "public_kernels/convnext_diversity_infer"
OUTPUT = OUTPUT_DIR / "rsna-knee-convnext-diversity-infer.py"
METADATA = OUTPUT_DIR / "kernel-metadata.json"


PRELUDE = r'''#!/usr/bin/env python3
"""Standalone hidden-test inference for the independently trained T01 ConvNeXt arm."""
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ["RSNA_CONVNEXT_STANDALONE"] = "1"
T0 = time.time()


def _standalone_comp_root():
    for candidate in (
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (candidate / "test.csv").is_file():
            return candidate
    raise FileNotFoundError("RSNA Knee competition input is not mounted")


_standalone_root = _standalone_comp_root()
_RSNA_TEST_IDS = pd.read_csv(
    _standalone_root / "test.csv", dtype={"StudyInstanceUID": str}
)["StudyInstanceUID"].astype(str).tolist()


def rsna_json(path, payload):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def rsna_save_predictions(name, study_ids, values, labels):
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(study_ids), len(labels)) or not np.isfinite(values).all():
        raise RuntimeError(f"invalid {name} prediction artifact: {values.shape}")
    destination = Path("/kaggle/working/diagnostics") / f"{name}_predictions.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        study_uids=np.asarray(study_ids),
        predictions=values,
        labels=np.asarray(labels),
    )

'''


def build() -> None:
    main = MAIN.read_text(encoding="utf-8")
    worker = WORKER.read_text(encoding="utf-8")
    writer = (
        "_worker_path = Path('/kaggle/working/_convnext_decode_worker.py')\n"
        f"_worker_path.write_text({worker!r}, encoding='utf-8')\n"
        "print('wrote isolated ConvNeXt CPU decoder', _worker_path, flush=True)\n\n"
    )
    script = PRELUDE + writer + main + "\n"
    required = (
        'os.environ["RSNA_CONVNEXT_STANDALONE"] = "1"',
        "dinov5_convnext_standalone_inference_v1",
        'Path("/kaggle/working/submission.csv")',
        "fallback_studies",
        "submission_sha256",
    )
    for token in required:
        if token not in script:
            raise RuntimeError(f"standalone ConvNeXt source is missing {token}")

    metadata = {
        "id": "adityakumarrai0401/rsna-knee-convnext-diversity-infer-v1",
        "title": "RSNA Knee ConvNeXt Diversity Infer V1",
        "code_file": OUTPUT.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [],
        "kernel_sources": [
            "adityakumarrai0401/rsna-knee-convnext-diversity-train-v2"
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(script, encoding="utf-8")
    METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"wrote {METADATA}")


if __name__ == "__main__":
    build()
