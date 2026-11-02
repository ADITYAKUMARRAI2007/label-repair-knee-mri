#!/usr/bin/env python3
"""Generate five one-fold Kaggle trainers for the preregistered T03 experiment."""

from __future__ import annotations

import hashlib
import json
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "training/t03_effb0_fold.py"
CONTRACT = ROOT / "scripts/t03_contract.py"

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


def source_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    template = TEMPLATE.read_text()
    contract = CONTRACT.read_text()
    start = contract.index("TARGETS = [")
    stop = contract.index('\nif __name__ == "__main__":')
    embedded = contract[start:stop].rstrip()
    if template.count("# __CONTRACT__") != 1 or template.count("__RUN_FOLD__") != 1:
        raise RuntimeError("T03 template markers drifted")
    common = template.replace("# __CONTRACT__", embedded)
    outputs = []
    for fold in range(5):
        directory = ROOT / f"public_kernels/t03_effb0_f{fold}"
        directory.mkdir(parents=True, exist_ok=True)
        code_name = f"rsna-knee-t03-effb0-f{fold}.py"
        code = common.replace("__RUN_FOLD__", str(fold))
        path = directory / code_name
        path.write_text(code)
        py_compile.compile(str(path), doraise=True)
        metadata = {
            "id": f"adityakumarrai0401/rsna-knee-t03-effb0-fold-{fold}",
            "title": f"RSNA Knee T03 EffB0 Fold {fold}",
            "code_file": code_name,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "keywords": ["gpu"],
            "dataset_sources": DATASETS,
            "kernel_sources": [],
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "model_sources": ["timm/tf-efficientnet/PyTorch/tf-efficientnet-b0/1"],
            "machine_shape": "NvidiaTeslaT4",
        }
        (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        outputs.append({"fold": fold, "source": str(path), "sha256": source_sha(path),
                        "bytes": path.stat().st_size})
    receipt = {
        "schema": "t03_effb0_source_build_v1",
        "fold_contract_sha256": "62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357",
        "pretrained_sha256": "827b6e33c2a10291671e2426b902c2fe87a77e88ab6e3ae769cda9c85b9be02e",
        "outputs": outputs,
    }
    receipt_path = ROOT / "runs/t03_source_build.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
