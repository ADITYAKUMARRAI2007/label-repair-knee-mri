#!/usr/bin/env python3
"""Build and statically audit the E02 Kaggle trainer source."""

import ast
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "training" / "e02_resnet34_distill.py"
DEST = ROOT / "public_kernels" / "e02_resnet34_distill" / "rsna-knee-e02-resnet34-distill.py"
AUDIT = ROOT / "runs" / "e02_source_audit.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ast.parse(SOURCE.read_text())
    DEST.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE, DEST)
    text = DEST.read_text()
    required = [
        "gold_rows_used_for_training\": 0", "mmap.MADV_DONTNEED",
        "TEACHER_LOGIT_WEIGHT = 0.75", "JEV_LOGIT_WEIGHT = 0.25",
        "SWA_EPOCHS = (15, 16, 17, 18)", "nn.DataParallel",
    ]
    missing = [token for token in required if token not in text]
    if missing:
        raise RuntimeError(f"E02 source contract missing: {missing}")
    result = {
        "schema": "e02_source_audit_v1", "status": "PASS",
        "source": str(DEST), "bytes": DEST.stat().st_size, "sha256": sha(DEST),
        "strict_gold_holdout": True, "architecture": "resnet34_simple_mil",
    }
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
