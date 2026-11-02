#!/usr/bin/env python3
"""Fail-closed audit for downloaded J01 training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
FOLD_SHA = "62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--receipt", type=Path, default=Path("runs/j01_training_audit.json"))
    args = parser.parse_args()
    root = args.output_dir
    manifest_path = root / "j01_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    errors = []
    if manifest.get("schema") != "rsna_knee_j01_biomed_jepa_v1":
        errors.append("manifest schema drift")
    if manifest.get("status") != "COMPLETE":
        errors.append("manifest is not COMPLETE")
    if manifest.get("gold_excluded_from_adaptation") is not True:
        errors.append("Gold exclusion flag is absent")
    if int(manifest.get("non_gold_adaptation_studies", -1)) != 4349:
        errors.append("non-Gold adaptation count drift")
    if manifest.get("fold_audit", {}).get("fold_sha256") != FOLD_SHA:
        errors.append("fold hash drift")
    required = {"j01_visual.ts", "j01_heads.pt", "j01_gold58_oof.csv"}
    if set(manifest.get("artifacts", {})) != required:
        errors.append("artifact set drift")
    artifact_audit = {}
    for name in required:
        path = root / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing/empty {name}")
            continue
        digest = sha256(path)
        expected = manifest.get("artifacts", {}).get(name, {})
        if digest != expected.get("sha256") or path.stat().st_size != expected.get("bytes"):
            errors.append(f"hash/size mismatch for {name}")
        artifact_audit[name] = {"bytes": path.stat().st_size, "sha256": digest}

    target_auc = {}
    oof_path = root / "j01_gold58_oof.csv"
    if oof_path.is_file():
        oof = pd.read_csv(oof_path)
        expected_columns = TARGETS + [f"truth_{x}" for x in TARGETS]
        if oof.columns.tolist() != expected_columns or len(oof) != 58:
            errors.append(f"OOF shape/columns drift: {oof.shape}")
        else:
            pred = oof[TARGETS].to_numpy(np.float64)
            truth = oof[[f"truth_{x}" for x in TARGETS]].to_numpy(np.uint8)
            if not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
                errors.append("OOF prediction range/finite failure")
            if not np.isin(truth, [0, 1]).all():
                errors.append("OOF truth is not binary")
            for j, target in enumerate(TARGETS):
                target_auc[target] = float(roc_auc_score(truth[:, j], pred[:, j]))
                recorded = float(manifest.get("per_target_auc", {}).get(target, np.nan))
                if not np.isclose(target_auc[target], recorded, atol=1e-12, rtol=0):
                    errors.append(f"AUC mismatch for {target}")
            macro = float(np.mean(list(target_auc.values())))
            if not np.isclose(macro, float(manifest.get("gold58_auc", np.nan)), atol=1e-12, rtol=0):
                errors.append("macro Gold58 AUC mismatch")
    receipt = {"schema": "j01_training_artifact_audit_v1",
               "status": "PASS" if not errors else "FAIL", "errors": errors,
               "manifest_sha256": sha256(manifest_path),
               "artifacts": artifact_audit, "per_target_auc": target_auc}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
