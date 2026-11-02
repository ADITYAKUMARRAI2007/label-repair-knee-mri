#!/usr/bin/env python3
"""Fail-closed audit for T01 ConvNeXt manifest, Gold58 OOF, and weights."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, label: str, tolerance: float = 1e-6) -> None:
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > tolerance:
        raise RuntimeError(f"{label} mismatch: recomputed={actual} recorded={expected}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/T01_training_audit.json"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    if manifest.get("schema") != "dinov5_convnext_diversity_v1":
        raise RuntimeError(f"unexpected T01 schema {manifest.get('schema')!r}")
    if manifest.get("targets") != TARGETS:
        raise RuntimeError("T01 target order mismatch")
    expected_input = {
        "crop_mm": 150.0,
        "image_size": 224,
        "max_series": 6,
        "clip_quantiles": [0.15, 0.325, 0.5, 0.675, 0.85],
        "order": "physical ImagePositionPatient projection",
    }
    if manifest.get("input") != expected_input:
        raise RuntimeError(f"T01 input contract mismatch: {manifest.get('input')}")
    validation = manifest.get("validation", {})
    if int(validation.get("gold_n", -1)) != 58:
        raise RuntimeError(f"T01 manifest has Gold n={validation.get('gold_n')}")
    folds = validation.get("fold_metrics", {})
    if set(folds) != {"0", "1", "2", "3", "4"}:
        raise RuntimeError(f"T01 fold manifest mismatch: {sorted(folds)}")
    if sum(int(folds[str(i)]["n_gold"]) for i in range(5)) != 58:
        raise RuntimeError("T01 fold validation counts do not sum to 58")

    frame = pd.read_csv(args.oof, dtype={"StudyInstanceUID": str})
    if frame.columns.tolist() != ["StudyInstanceUID", *TARGETS]:
        raise RuntimeError(f"T01 OOF columns drifted: {frame.columns.tolist()}")
    if len(frame) != 58 or frame.StudyInstanceUID.duplicated().any():
        raise RuntimeError(f"T01 OOF identity contract failed: {len(frame)} rows")
    values = frame[TARGETS].to_numpy(np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise RuntimeError("T01 OOF predictions are not finite probabilities")
    if (np.ptp(values, axis=0) <= 1e-8).any():
        flat = [TARGETS[i] for i in np.flatnonzero(np.ptp(values, axis=0) <= 1e-8)]
        raise RuntimeError(f"T01 OOF has constant targets: {flat}")

    train = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    gold = train.loc[train[TARGETS].notna().all(axis=1), ["StudyInstanceUID", *TARGETS]]
    if len(gold) != 58 or gold.StudyInstanceUID.duplicated().any():
        raise RuntimeError(f"competition expert-label contract changed: {len(gold)} rows")
    joined = frame.merge(gold, on="StudyInstanceUID", suffixes=("_pred", "_truth"),
                         validate="one_to_one")
    if len(joined) != 58 or set(frame.StudyInstanceUID) != set(gold.StudyInstanceUID):
        raise RuntimeError("T01 OOF does not cover the exact Gold58 set")
    predictions = joined[[f"{name}_pred" for name in TARGETS]].to_numpy(np.float64)
    truth = joined[[f"{name}_truth" for name in TARGETS]].to_numpy(np.float64)

    per_target = {}
    for index, target in enumerate(TARGETS):
        if np.unique(truth[:, index]).size != 2:
            raise RuntimeError(f"Gold58 lacks both classes for {target}")
        per_target[target] = float(roc_auc_score(truth[:, index], predictions[:, index]))
    macro = float(np.mean(list(per_target.values())))
    close(macro, float(validation["oof_macro_auc"]), "T01 OOF macro AUC")
    recorded_target = validation.get("per_target_auc", {})
    if set(recorded_target) != set(TARGETS):
        raise RuntimeError("T01 manifest per-target AUC keys drifted")
    for target in TARGETS:
        close(per_target[target], float(recorded_target[target]), f"T01 {target} AUC")

    beta = manifest.get("recommended_main_beta", {})
    if set(beta) != set(TARGETS):
        raise RuntimeError("T01 recommended beta keys drifted")
    for target in TARGETS:
        auc = per_target[target]
        expected = 0.0 if auc < 0.66 else float(np.clip(0.04 + 0.20 * (auc - 0.66), 0.04, 0.12))
        close(float(beta[target]), expected, f"T01 {target} recommended beta", tolerance=1e-10)

    weights = manifest.get("weights", {})
    expected_heads = [f"dinov5_convnext_head_fold{i}.pt" for i in range(5)]
    if weights.get("backbone") != "dinov5_convnext_backbone.pt" or weights.get("heads") != expected_heads:
        raise RuntimeError(f"T01 weight manifest drifted: {weights}")
    artifacts = [weights["backbone"], *weights["heads"]]
    artifact_report = {}
    for name in artifacts:
        path = args.artifact_dir / name
        if not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError(f"missing or empty T01 weight artifact: {path}")
        artifact_report[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    report = {
        "schema": "convnext_training_artifact_audit_v1",
        "status": "PASS",
        "gold_studies": 58,
        "oof_macro_auc": macro,
        "per_target_auc": per_target,
        "recommended_main_beta": beta,
        "manifest_sha256": sha256(args.manifest),
        "oof_sha256": sha256(args.oof),
        "weights": artifact_report,
        "warning": "Gold58 is a coarse model and blend gate; recommended weights remain capped at 0.12.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
