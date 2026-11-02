#!/usr/bin/env python3
"""Build the J01 BiomedCLIP masked-latent adaptation Kaggle trainer."""

from __future__ import annotations

import hashlib
import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "training/j01_biomed_jepa.py"
CONTRACT = ROOT / "scripts/t03_contract.py"
OUT_DIR = ROOT / "public_kernels/j01_biomed_jepa_train"
OUTPUT = OUT_DIR / "rsna-knee-j01-biomed-jepa-train.py"

DATASETS = [
    "dreaddevelopment/knee-raptor-corpus",
    "dreaddevelopment/knee-raptor-corpus-ext",
    "stevenleehans/rsna-knee-llm-report-labels",
    "lixin73/rsna-knee-llm-report-labels-sol56",
    "pilkwang/rsna-knee-llm-labels",
    "rayanbabur/rsna-knee-2026-calibrated-soft-targets",
    "dreaddevelopment/rsna-knee-labels",
    "yunusgmsoy/rsna-knee-llm-labels-4-source-merged",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    template = TEMPLATE.read_text()
    contract = CONTRACT.read_text()
    start = contract.index("TARGETS = [")
    stop = contract.index('\nif __name__ == "__main__":')
    source = template.replace("# __CONTRACT__", contract[start:stop].rstrip())
    if "# __CONTRACT__" in source:
        raise RuntimeError("contract insertion failed")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(source)
    py_compile.compile(str(OUTPUT), doraise=True)
    metadata = {
        "id": "adityakumarrai0401/rsna-knee-j01-biomed-jepa-train",
        "title": "RSNA Knee J01 Biomed JEPA Train",
        "code_file": OUTPUT.name, "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": True, "keywords": ["gpu"],
        "dataset_sources": DATASETS,
        "kernel_sources": [],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    receipt = {"schema": "j01_source_build_v1", "source": str(OUTPUT),
               "sha256": digest(OUTPUT), "bytes": OUTPUT.stat().st_size,
               "fold_sha256": "62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357"}
    (ROOT / "runs/j01_source_build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
