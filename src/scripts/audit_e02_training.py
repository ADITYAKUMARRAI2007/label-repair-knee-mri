#!/usr/bin/env python3
"""Independently audit downloaded E02 artifacts and Gold58 predictions."""

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


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact_dir", nargs="?", default="runs/E02_training")
    args = ap.parse_args()
    root = Path(args.artifact_dir)
    manifest = json.loads((root / "e02_manifest.json").read_text())
    if manifest["status"] != "COMPLETE" or not manifest["strict_gold_holdout"]:
        raise RuntimeError("E02 completion or holdout contract failed")
    for name, recorded in manifest["artifacts"].items():
        path = root / name
        if path.stat().st_size != recorded["bytes"] or sha(path) != recorded["sha256"]:
            raise RuntimeError(f"artifact mismatch: {name}")
    train = pd.read_csv("data/train.csv", dtype={"StudyInstanceUID": str})
    gold = train.loc[train[TARGETS].notna().all(axis=1), ["StudyInstanceUID", *TARGETS]]
    pred = pd.read_csv(root / "e02_gold_predictions.csv", dtype={"StudyInstanceUID": str})
    if set(pred.StudyInstanceUID) != set(gold.StudyInstanceUID) or pred.StudyInstanceUID.duplicated().any():
        raise RuntimeError("Gold prediction identity mismatch")
    joined = gold.merge(pred, on="StudyInstanceUID", suffixes=("_truth", "_pred"), validate="one_to_one")
    per_target = {
        t: float(roc_auc_score(joined[f"{t}_truth"], joined[f"{t}_pred"])) for t in TARGETS
    }
    score = float(np.mean(list(per_target.values())))
    recorded = float(manifest["validation"]["gold_auc"])
    if abs(score - recorded) > 1e-9:
        raise RuntimeError(f"Gold AUC mismatch: recomputed={score} recorded={recorded}")
    result = {
        "schema": "e02_training_audit_v1", "status": "PASS",
        "gold_auc": score, "per_target_auc": per_target,
        "checkpoint_sha256": sha(root / "e02_resnet34_distill_swa.pt"),
        "strict_gold_holdout": True,
    }
    (root / "e02_training_audit.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
