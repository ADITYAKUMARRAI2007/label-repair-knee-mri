#!/usr/bin/env python3
"""Build and audit the E02 standalone inference kernel."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "inference" / "e02_resnet34_infer.py"
DEST_DIR = ROOT / "public_kernels" / "e02_resnet34_infer"
DEST = DEST_DIR / "rsna-knee-e02-resnet34-infer.py"
AUDIT = ROOT / "runs" / "e02_inference_source_audit.json"


def main():
    ast.parse(SOURCE.read_text())
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, DEST)
    text = DEST.read_text()
    required = [
        '"e02_resnet34_distill_swa.pt"', "class SimpleMIL",
        "K_EVAL = 32", "NETWORK_IMG = 288", "fallback_studies\": 0",
        "gold_rows_used_for_training", "submission.csv",
    ]
    missing = [x for x in required if x not in text]
    if missing:
        raise RuntimeError(f"E02 inference contract missing: {missing}")
    meta = {
        "id": "adityakumarrai0401/rsna-knee-e02-resnet34-infer",
        "title": "RSNA Knee E02 ResNet34 Infer",
        "code_file": DEST.name,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["gpu"], "dataset_sources": [],
        "kernel_sources": ["adityakumarrai0401/rsna-knee-e02-resnet34-distill"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    (DEST_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    result = {
        "schema": "e02_inference_source_audit_v1", "status": "PASS",
        "source": str(DEST), "bytes": DEST.stat().st_size,
        "sha256": hashlib.sha256(DEST.read_bytes()).hexdigest(),
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
