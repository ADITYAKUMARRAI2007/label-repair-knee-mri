#!/usr/bin/env python3
"""Fail-closed static and local-data audit for all five generated T03 trainers."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from t03_contract import construct_targets, local_label_paths, make_folds


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FOLD_SHA = "62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357"
EXPECTED_WEIGHT_SHA = "827b6e33c2a10291671e2426b902c2fe87a77e88ab6e3ae769cda9c85b9be02e"


def main():
    train = pd.read_csv(ROOT / "data/train.csv", dtype={"StudyInstanceUID": str})
    targets, weights, gold = construct_targets(train, local_label_paths(ROOT / "labels"))
    folds, fold_audit = make_folds(train, targets, gold)
    assert fold_audit["fold_sha256"] == EXPECTED_FOLD_SHA, fold_audit
    assert weights.max() == 2.0 and weights.min() >= 0.25
    normalized = []
    sources = []
    for fold in range(5):
        directory = ROOT / f"public_kernels/t03_effb0_f{fold}"
        metadata = json.loads((directory / "kernel-metadata.json").read_text())
        source = directory / metadata["code_file"]
        text = source.read_text()
        ast.parse(text)
        assert f"RUN_FOLD = {fold}\n" in text
        assert text.count("RUN_FOLD = ") == 1
        assert EXPECTED_FOLD_SHA in text and EXPECTED_WEIGHT_SHA in text
        assert "EPOCHS = 10" in text and "SWA_EPOCHS = (8, 9, 10)" in text
        assert "num_workers=0" in text and "batch_size=4" in text
        assert "weights[gold] = 2.0" in text
        assert "best" not in text[text.index("def main():"):], "score-selected checkpoint leaked in"
        assert not re.search(r"1\.2\.826\.0\.1\.3680043", text)
        assert metadata["is_private"] and not metadata["enable_internet"]
        assert metadata["model_sources"] == ["timm/tf-efficientnet/PyTorch/tf-efficientnet-b0/1"]
        normalized.append(text.replace(f"RUN_FOLD = {fold}\n", "RUN_FOLD = FOLD\n"))
        sources.append({"fold": fold, "bytes": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    assert len(set(normalized)) == 1, "fold sources differ outside RUN_FOLD"
    result = {"status": "PASS", "schema": "t03_source_audit_v1",
              "fold_audit": fold_audit, "sources": sources,
              "targets": list(targets.shape), "gold": int(gold.sum()),
              "weight_range": [float(weights.min()), float(weights.max())]}
    path = ROOT / "runs/t03_source_audit.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
