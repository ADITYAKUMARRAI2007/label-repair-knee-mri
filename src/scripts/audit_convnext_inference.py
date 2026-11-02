#!/usr/bin/env python3
"""Fail-closed audit for T01 standalone ConvNeXt inference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


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


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return np.full_like(values, 0.5)
    return (
        pd.DataFrame(values).rank(method="average").to_numpy(dtype=np.float64) - 1
    ) / (len(values) - 1)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--training-audit", type=Path, required=True)
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/T01_inference_audit.json")
    )
    args = parser.parse_args()

    for path in (
        args.submission, args.arm, args.predictions, args.receipt,
        args.training_audit, args.test,
    ):
        require(path.is_file() and path.stat().st_size > 0, f"missing artifact: {path}")

    receipt = json.loads(args.receipt.read_text())
    training = json.loads(args.training_audit.read_text())
    require(
        receipt.get("schema") == "dinov5_convnext_standalone_inference_v1",
        f"unexpected inference schema: {receipt.get('schema')!r}",
    )
    require(receipt.get("status") == "COMPLETE", "inference receipt is incomplete")
    require(
        training.get("schema") == "convnext_training_artifact_audit_v1"
        and training.get("status") == "PASS",
        "training artifact audit is not a PASS",
    )

    test_ids = pd.read_csv(
        args.test, dtype={"StudyInstanceUID": str}
    ).StudyInstanceUID.astype(str).tolist()
    require(test_ids and len(test_ids) == len(set(test_ids)), "invalid test identity list")
    require(int(receipt.get("studies", -1)) == len(test_ids), "receipt study count mismatch")
    require(int(receipt.get("fallback_studies", -1)) == 0, "inference used fallbacks")
    require(int(receipt.get("fallbacks", -1)) == 0, "inference used fallbacks")
    require(int(receipt.get("token_count_min", 0)) > 0, "one or more studies have no tokens")
    require(
        int(receipt.get("token_count_max", 31)) <= 30,
        "token count exceeds the six-series by five-window contract",
    )
    require(
        set(receipt.get("per_target_gold58_auc", {})) == set(TARGETS),
        "receipt per-target AUC keys mismatch",
    )
    beta = receipt.get("per_target_beta", {})
    require(set(beta) == set(TARGETS), "receipt beta keys mismatch")
    require(
        all(abs(float(beta[target])) <= 1e-12 for target in TARGETS),
        "standalone inference unexpectedly applied blend weights",
    )

    expected_columns = ["StudyInstanceUID", *TARGETS]
    submission = pd.read_csv(args.submission, dtype={"StudyInstanceUID": str})
    arm = pd.read_csv(args.arm, dtype={"StudyInstanceUID": str})
    for name, frame in (("submission", submission), ("arm", arm)):
        require(frame.columns.tolist() == expected_columns, f"{name} columns mismatch")
        require(
            frame.StudyInstanceUID.astype(str).tolist() == test_ids,
            f"{name} study identity/order mismatch",
        )

    with np.load(args.predictions, allow_pickle=False) as payload:
        require(
            set(payload.files) == {"study_uids", "predictions", "labels"},
            f"prediction archive keys mismatch: {payload.files}",
        )
        prediction_ids = payload["study_uids"].astype(str).tolist()
        labels = payload["labels"].astype(str).tolist()
        predictions = np.asarray(payload["predictions"], dtype=np.float64)
    require(prediction_ids == test_ids, "prediction archive identity/order mismatch")
    require(labels == TARGETS, "prediction archive target order mismatch")
    require(predictions.shape == (len(test_ids), len(TARGETS)), "prediction shape mismatch")
    require(np.isfinite(predictions).all(), "predictions contain non-finite values")
    require(((predictions >= 0) & (predictions <= 1)).all(), "predictions outside [0,1]")
    require(
        np.allclose(predictions, rank01(predictions), atol=6e-8, rtol=0),
        "prediction archive is not the expected per-target percentile rank",
    )

    submission_values = submission[TARGETS].to_numpy(dtype=np.float64)
    arm_values = arm[TARGETS].to_numpy(dtype=np.float64)
    require(
        np.allclose(submission_values, arm_values, atol=1e-12, rtol=0),
        "standalone submission differs from the saved ConvNeXt arm",
    )
    require(
        np.allclose(submission_values, predictions, atol=6e-8, rtol=0),
        "submission differs from the saved prediction archive",
    )

    digests = {
        "submission": sha256(args.submission),
        "arm": sha256(args.arm),
        "predictions": sha256(args.predictions),
    }
    for name, key in (
        ("submission", "submission_sha256"),
        ("arm", "arm_sha256"),
        ("predictions", "predictions_sha256"),
    ):
        require(receipt.get(key) == digests[name], f"receipt {key} mismatch")
    require(
        receipt.get("manifest_sha256") == training.get("manifest_sha256"),
        "inference and training manifest hashes differ",
    )
    receipt_weights = receipt.get("weight_sha256", {})
    training_weights = {
        name: item["sha256"] for name, item in training.get("weights", {}).items()
    }
    require(receipt_weights == training_weights, "inference and audited weight hashes differ")

    report = {
        "schema": "convnext_inference_artifact_audit_v1",
        "status": "PASS",
        "studies": len(test_ids),
        "token_count_min": int(receipt["token_count_min"]),
        "token_count_median": float(receipt["token_count_median"]),
        "token_count_max": int(receipt["token_count_max"]),
        "gold58_oof_macro_auc": float(receipt["gold58_oof_macro_auc"]),
        "manifest_sha256": receipt["manifest_sha256"],
        "weight_sha256": receipt_weights,
        "artifact_sha256": digests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
