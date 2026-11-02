#!/usr/bin/env python3
"""Fail-closed audit for the E01 EfficientNetV2 training artifacts."""

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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_gold(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=True) as payload:
        required = {"pred", "truth", "ids"}
        if not required.issubset(payload.files):
            raise RuntimeError(f"{path} is missing {sorted(required - set(payload.files))}")
        predictions = np.asarray(payload["pred"], dtype=np.float64)
        truth = np.asarray(payload["truth"], dtype=np.float64)
        ids = payload["ids"].astype(str)
    return predictions, truth, ids


def score_artifact(
    path: Path,
    expected_ids: list[str],
    expected_truth: np.ndarray,
) -> dict:
    predictions, truth, ids = load_gold(path)
    expected_shape = (len(expected_ids), len(TARGETS))
    if predictions.shape != expected_shape or truth.shape != expected_shape:
        raise RuntimeError(
            f"{path} shape mismatch: pred={predictions.shape} truth={truth.shape} "
            f"expected={expected_shape}"
        )
    if ids.tolist() != expected_ids:
        raise RuntimeError(f"{path} Gold58 identity or order mismatch")
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError(f"{path} contains duplicate study IDs")
    if not np.array_equal(truth, expected_truth):
        difference = float(np.max(np.abs(truth - expected_truth)))
        raise RuntimeError(f"{path} expert truth mismatch; max_abs_diff={difference}")
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{path} contains non-finite predictions")
    if (predictions < 0).any() or (predictions > 1).any():
        raise RuntimeError(f"{path} predictions fall outside [0,1]")
    if (np.ptp(predictions, axis=0) <= 1e-8).any():
        flat = [TARGETS[i] for i in np.flatnonzero(np.ptp(predictions, axis=0) <= 1e-8)]
        raise RuntimeError(f"{path} has constant target predictions: {flat}")

    per_target = {}
    for index, target in enumerate(TARGETS):
        labels = truth[:, index]
        if np.unique(labels).size != 2:
            raise RuntimeError(f"Gold58 lacks both classes for {target}")
        per_target[target] = float(roc_auc_score(labels, predictions[:, index]))
    return {
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(ids),
        "macro_auc": float(np.mean(list(per_target.values()))),
        "per_target_auc": per_target,
        "prediction_min": float(predictions.min()),
        "prediction_max": float(predictions.max()),
    }


def close(actual: float, expected: float, label: str, tolerance: float = 1e-6) -> None:
    if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual - expected) > tolerance:
        raise RuntimeError(f"{label} mismatch: recomputed={actual} recorded={expected}")


def audit_checkpoint(path: Path, *, swa: bool) -> dict:
    # The local audit environment intentionally has no PyTorch dependency. Validate
    # the torch.save ZIP container and its embedded contract strings here; the
    # offline inference kernel performs the authoritative torch.load/state check.
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
        if len(pickle_names) != 1 or len(tensor_members) < 100:
            raise RuntimeError(
                f"{path} has an invalid torch archive layout: "
                f"data.pkl={pickle_names}, tensor_members={len(tensor_members)}"
            )
        metadata = archive.read(pickle_names[0])
    required_tokens = [
        b"model", b"gold_auc", b"aucs", b"src", b"res", b"arch", b"lab", b"epoch",
        b"tf_efficientnetv2_s.in21k", *[target.encode() for target in TARGETS],
    ]
    if swa:
        required_tokens.append(b"swa_over")
    missing = [token.decode() for token in required_tokens if token not in metadata]
    if missing:
        raise RuntimeError(f"{path} checkpoint metadata is missing {missing}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "bytes": size,
        "archive_members": len(names),
        "tensor_members": len(tensor_members),
        "embedded_contract": "effv2s_336_12_targets_swa" if swa else "effv2s_336_12_targets",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--swa-gold", type=Path)
    parser.add_argument("--swa-checkpoint", type=Path)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/E01_training_audit.json"))
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    required = {
        "tag", "best_gold_auc", "aucs", "hist", "epochs", "epochs_done",
        "n_train", "n_gold", "partial",
    }
    missing = required - set(metrics)
    if missing:
        raise RuntimeError(f"metrics JSON is missing {sorted(missing)}")
    if metrics["tag"] != "effv2s_336_k10_consensus_v1":
        raise RuntimeError(f"unexpected E01 tag {metrics['tag']!r}")
    if bool(metrics["partial"]):
        raise RuntimeError("E01 metrics still report a partial training run")
    if int(metrics["epochs"]) != 12 or int(metrics["epochs_done"]) != 12:
        raise RuntimeError(
            f"E01 epoch contract mismatch: {metrics['epochs_done']}/{metrics['epochs']}"
        )
    if int(metrics["n_train"]) != 4349 or int(metrics["n_gold"]) != 58:
        raise RuntimeError(
            f"E01 split mismatch: train={metrics['n_train']} gold={metrics['n_gold']}"
        )
    if len(metrics["hist"]) != 12:
        raise RuntimeError(f"E01 history has {len(metrics['hist'])} epochs instead of 12")

    train = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    gold = train.loc[train[TARGETS].notna().all(axis=1), ["StudyInstanceUID", *TARGETS]]
    if len(gold) != 58 or gold.StudyInstanceUID.duplicated().any():
        raise RuntimeError(f"competition expert-label contract changed: {len(gold)} rows")
    expected_ids = gold.StudyInstanceUID.astype(str).tolist()
    expected_truth = gold[TARGETS].to_numpy(np.float64)

    best = score_artifact(args.gold, expected_ids, expected_truth)
    close(best["macro_auc"], float(metrics["best_gold_auc"]), "best macro AUC")
    if set(metrics["aucs"]) != set(TARGETS):
        raise RuntimeError("metrics per-target AUC keys do not match the 12 targets")
    for target in TARGETS:
        close(
            best["per_target_auc"][target],
            float(metrics["aucs"][target]),
            f"best {target} AUC",
        )

    best["checkpoint"] = audit_checkpoint(args.checkpoint, swa=False)
    candidates = {"best": best}
    if (args.swa_gold is None) != (args.swa_checkpoint is None):
        raise RuntimeError("--swa-gold and --swa-checkpoint must be supplied together")
    if args.swa_gold is not None:
        swa = score_artifact(args.swa_gold, expected_ids, expected_truth)
        swa["checkpoint"] = audit_checkpoint(args.swa_checkpoint, swa=True)
        candidates["swa"] = swa
    selected = max(candidates, key=lambda name: candidates[name]["macro_auc"])
    report = {
        "schema": "effv2_training_artifact_audit_v1",
        "status": "PASS",
        "metrics": str(args.metrics),
        "metrics_sha256": sha256(args.metrics),
        "epochs": 12,
        "train_studies": 4349,
        "gold_studies": 58,
        "candidates": candidates,
        "gold_selected_candidate": selected,
        "warning": "Gold58 is a coarse checkpoint gate, not a precise blend-weight tuner.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
