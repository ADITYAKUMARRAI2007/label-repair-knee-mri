#!/usr/bin/env python3
"""Fail-closed audit for M01 MedicalNet3D training artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
MEDICALNET_SHA256 = "afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual: float, expected: float, label: str, tolerance: float = 1e-6) -> None:
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > tolerance:
        raise RuntimeError(f"{label} mismatch: recomputed={actual} recorded={expected}")


def score_npz(path: Path, gold: pd.DataFrame) -> dict:
    with np.load(path, allow_pickle=True) as payload:
        required = {"pred", "truth", "ids", "lab"}
        if not required.issubset(payload.files):
            raise RuntimeError(f"{path} is missing {sorted(required - set(payload.files))}")
        predictions = np.asarray(payload["pred"], np.float64)
        truth = np.asarray(payload["truth"], np.float64)
        ids = payload["ids"].astype(str)
        labels = payload["lab"].astype(str).tolist()
    if labels != TARGETS:
        raise RuntimeError(f"{path} label order mismatch: {labels}")
    if predictions.shape != (58, 12) or truth.shape != (58, 12) or ids.shape != (58,):
        raise RuntimeError(f"{path} shape mismatch: {predictions.shape}, {truth.shape}, {ids.shape}")
    if len(set(ids.tolist())) != 58 or set(ids.tolist()) != set(gold.StudyInstanceUID):
        raise RuntimeError(f"{path} does not contain the exact Gold58 identities")
    expected = gold.set_index("StudyInstanceUID").loc[ids.tolist(), TARGETS].to_numpy(np.float64)
    if not np.array_equal(truth, expected):
        raise RuntimeError(
            f"{path} truth mismatch; max_abs_diff={float(np.max(np.abs(truth - expected)))}"
        )
    if not np.isfinite(predictions).all() or (predictions < 0).any() or (predictions > 1).any():
        raise RuntimeError(f"{path} predictions are not finite probabilities")
    if (np.ptp(predictions, axis=0) <= 1e-8).any():
        flat = [TARGETS[i] for i in np.flatnonzero(np.ptp(predictions, axis=0) <= 1e-8)]
        raise RuntimeError(f"{path} has constant targets: {flat}")
    per_target = {}
    for index, target in enumerate(TARGETS):
        if np.unique(truth[:, index]).size != 2:
            raise RuntimeError(f"Gold58 lacks both classes for {target}")
        per_target[target] = float(roc_auc_score(truth[:, index], predictions[:, index]))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "macro_auc": float(np.mean(list(per_target.values()))),
        "per_target_auc": per_target,
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
    }


def audit_checkpoint(path: Path) -> dict:
    size = path.stat().st_size
    if size < 10_000_000:
        raise RuntimeError(f"{path} checkpoint is implausibly small: {size} bytes")
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"{path} is not a modern torch.save ZIP checkpoint")
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt is not None:
            raise RuntimeError(f"{path} has a corrupt ZIP member: {corrupt}")
        names = archive.namelist()
        pickle_names = [name for name in names if name.endswith("/data.pkl") or name == "data.pkl"]
        tensor_members = [name for name in names if "/data/" in name]
        if len(pickle_names) != 1 or len(tensor_members) < 50:
            raise RuntimeError(
                f"{path} has an invalid torch archive layout: "
                f"data.pkl={pickle_names}, tensor_members={len(tensor_members)}"
            )
        metadata = archive.read(pickle_names[0])
    required_tokens = [
        b"medicalnet3d_checkpoint_v1", b"model", b"gold_auc", b"gold_target_auc",
        b"lab", b"depth", b"res", b"medicalnet_sha256",
        MEDICALNET_SHA256.encode(), *[target.encode() for target in TARGETS],
    ]
    missing = [token.decode() for token in required_tokens if token not in metadata]
    if missing:
        raise RuntimeError(f"{path} checkpoint metadata is missing {missing}")
    return {
        "bytes": size,
        "sha256": sha256(path),
        "archive_members": len(names),
        "tensor_members": len(tensor_members),
        "embedded_contract": "medicalnet3d_v1_12_targets",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--best-gold", type=Path, required=True)
    parser.add_argument("--swa-gold", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/M01_training_audit.json"))
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    if metrics.get("schema") != "medicalnet3d_training_v1" or metrics.get("status") != "COMPLETE":
        raise RuntimeError(f"M01 training is not complete: {metrics.get('schema')}, {metrics.get('status')}")
    if int(metrics.get("n_train", -1)) != 4349 or int(metrics.get("n_gold", -1)) != 58:
        raise RuntimeError(f"M01 split mismatch: {metrics.get('n_train')}, {metrics.get('n_gold')}")
    history = metrics.get("history", [])
    if len(history) != 5 or [int(row.get("epoch", -1)) for row in history] != list(range(5)):
        raise RuntimeError(f"M01 epoch history mismatch: {[row.get('epoch') for row in history]}")
    if metrics.get("medicalnet_sha256") != MEDICALNET_SHA256:
        raise RuntimeError(f"M01 MedicalNet source hash mismatch: {metrics.get('medicalnet_sha256')}")
    if len(metrics.get("label_families", [])) != 6:
        raise RuntimeError(f"M01 label family count mismatch: {metrics.get('label_families')}")
    fusion = metrics.get("label_fusion", {})
    if (
        float(fusion.get("calibrated_weight", -1)) != 0.60
        or int(fusion.get("other_family_count", -1)) != 5
        or float(fusion.get("other_family_median_weight", -1)) != 0.40
        or float(fusion.get("post_fusion_temperature", -1)) != 1.0
        or "calibrated-soft-targets" not in str(fusion.get("calibrated_family", "")).lower()
    ):
        raise RuntimeError(f"M01 label fusion contract mismatch: {fusion}")
    confidence = metrics.get("confidence_contract", {})
    if (
        float(confidence.get("weak_min", -1)) < 0.0
        or float(confidence.get("weak_max", 2)) > 1.0
        or float(confidence.get("expert_marker", -1)) != 5.0
        or float(confidence.get("expert_runtime_weight", -1)) != 1.0
    ):
        raise RuntimeError(f"M01 confidence contract mismatch: {confidence}")

    train = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    gold = train.loc[train[TARGETS].notna().all(axis=1), ["StudyInstanceUID", *TARGETS]]
    if len(gold) != 58 or gold.StudyInstanceUID.duplicated().any():
        raise RuntimeError(f"competition expert-label contract changed: {len(gold)} rows")

    best = score_npz(args.best_gold, gold)
    swa = score_npz(args.swa_gold, gold)
    close(best["macro_auc"], float(metrics["best_gold_auc"]), "M01 best macro AUC")
    close(swa["macro_auc"], float(metrics["swa_gold_auc"]), "M01 SWA macro AUC")
    best_epoch = int(metrics["best_epoch"])
    if best_epoch not in range(5):
        raise RuntimeError(f"M01 best epoch is invalid: {best_epoch}")
    best_history = history[best_epoch]
    close(best["macro_auc"], float(best_history["gold_macro_auc"]), "M01 best history AUC")
    for target in TARGETS:
        close(
            best["per_target_auc"][target],
            float(best_history["gold_target_auc"][target]),
            f"M01 best {target} AUC",
        )
        close(
            swa["per_target_auc"][target],
            float(metrics["swa_target_auc"][target]),
            f"M01 SWA {target} AUC",
        )

    checkpoint_report = {}
    for name in ("medicalnet3d_best.pt", "medicalnet3d_swa.pt"):
        path = args.artifact_dir / name
        if not path.is_file():
            raise RuntimeError(f"missing M01 checkpoint: {path}")
        checkpoint_report[name] = audit_checkpoint(path)
    selected = "swa" if swa["macro_auc"] > best["macro_auc"] else "best"
    report = {
        "schema": "medicalnet3d_training_artifact_audit_v1",
        "status": "PASS",
        "train_studies": 4349,
        "gold_studies": 58,
        "medicalnet_sha256": MEDICALNET_SHA256,
        "label_fusion": fusion,
        "confidence_contract": confidence,
        "candidates": {"best": best, "swa": swa},
        "gold_selected_candidate": selected,
        "metrics_sha256": sha256(args.metrics),
        "checkpoints": checkpoint_report,
        "warning": "Gold58 is a coarse checkpoint gate, not a precise blend-weight tuner.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
