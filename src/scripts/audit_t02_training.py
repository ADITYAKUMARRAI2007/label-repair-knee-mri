#!/usr/bin/env python3
"""Independent fail-closed audit for completed T02 training artifacts."""

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
PUBLIC_LABEL_SHA256 = {
    "train_targets_soft.csv": "6dac0e2d9ba79065ec794fee743c0286978c2323fcb9625bb74ef2a73d4d1475",
    "train_targets_weights.csv": "3fedf0a0f792d5644f0e9fd537a3add2e0f81aa25e9540db74ec523bdd3a3c18",
    "report_labels_v2.csv": "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920",
    "labels_llm_gpt56sol.csv": "a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae",
    "llm_labels_v4_blend.csv": "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4",
    "labels_llm_soft.csv": "8cec477fb555e88bbb4196197fd3f806cbafed8cc2f6b2d48d26a8d49fbd86a1",
    "report_labels_v5.csv": "16cf44a89e027c80ff9b1a69e1ac6713d1f6da1765b3748c01f6ae91827ef167",
}
INIT_SHA256 = "bec04468edeaaa0c75191a221d4e3b80bada5cfd263eeec36cb093548de93147"


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("runs/T02_training_audit.json"))
    args = parser.parse_args()

    manifest_path = args.artifact_dir / "t02_manifest.json"
    oof_path = args.artifact_dir / "t02_gold58_oof.csv"
    need(manifest_path.is_file(), "missing t02_manifest.json")
    need(oof_path.is_file(), "missing t02_gold58_oof.csv")
    manifest = json.loads(manifest_path.read_text())
    need(manifest.get("schema") == "rsna_knee_goldfold_convnext_v1", "schema mismatch")
    need(manifest.get("targets") == TARGETS, "target order mismatch")

    training = manifest.get("training", {})
    need(training.get("folds") == 5 and training.get("epochs") == 6,
         "fold or epoch count mismatch")
    need(training.get("swa_last") == 3 and training.get("gold_weight") == 4.0,
         "SWA or Gold weight mismatch")
    need(training.get("lr_backbone") == 2e-5 and training.get("lr_head") == 5e-4,
         "optimizer learning rates mismatch")

    expected_input = {
        "img": 224, "crop_mm": 130.0,
        "slots": ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
                  "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"],
        "group": 3, "n_group": 3, "window": "0.35,0.65",
    }
    need(manifest.get("input") == expected_input, "input contract mismatch")
    provenance = manifest.get("provenance", {})
    need(provenance.get("init_sha256") == INIT_SHA256, "initialization hash mismatch")
    need(provenance.get("public_label_sha256") == PUBLIC_LABEL_SHA256,
         "public label provenance mismatch")
    cache = provenance.get("cache_contract", {})
    for key, value in expected_input.items():
        if key != "slots":
            need(cache.get(key) == value, f"cache contract mismatch for {key}")
    need(cache.get("slots") == expected_input["slots"], "cache slot order mismatch")

    histories = manifest.get("histories", {})
    need(set(histories) == {str(i) for i in range(5)}, "history fold set mismatch")
    for fold in range(5):
        history = histories[str(fold)]
        need(len(history) == 6, f"fold {fold} did not complete six epochs")
        need([row.get("epoch") for row in history] == list(range(1, 7)),
             f"fold {fold} epoch sequence mismatch")
        for row in history:
            need(np.isfinite(float(row["loss"])), f"fold {fold} non-finite loss")
            need(np.isfinite(float(row["derived_auc"])), f"fold {fold} non-finite derived AUC")
            need(np.isfinite(float(row["gold_auc"])), f"fold {fold} non-finite Gold AUC")

    weights = manifest.get("weights", {})
    expected_names = {f"t02_goldfold_convnext_fold{i}.pt" for i in range(5)}
    need(set(weights) == expected_names, "checkpoint set mismatch")
    checkpoint_report = {}
    for name in sorted(expected_names):
        path = args.artifact_dir / name
        need(path.is_file(), f"missing checkpoint {name}")
        recorded = weights[name]
        need(path.stat().st_size == int(recorded["bytes"]), f"size mismatch for {name}")
        digest = sha256(path)
        need(digest == recorded["sha256"], f"SHA-256 mismatch for {name}")
        need(zipfile.is_zipfile(path), f"checkpoint is not a valid torch ZIP: {name}")
        with zipfile.ZipFile(path) as archive:
            bad = archive.testzip()
            need(bad is None, f"checkpoint ZIP corruption in {name}: {bad}")
            member_count = len(archive.infolist())
        need(member_count >= 150, f"checkpoint {name} has too few tensor members")
        checkpoint_report[name] = {
            "bytes": path.stat().st_size, "sha256": digest, "zip_members": member_count,
        }

    train = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    gold = train[TARGETS].notna().all(axis=1)
    gold_frame = train.loc[gold, ["StudyInstanceUID", *TARGETS]].reset_index(drop=True)
    oof = pd.read_csv(oof_path, dtype={"StudyInstanceUID": str})
    need(oof.columns.tolist() == ["StudyInstanceUID", *TARGETS], "OOF schema mismatch")
    need(len(oof) == 58 and oof.StudyInstanceUID.is_unique, "OOF identity count mismatch")
    need(oof.StudyInstanceUID.tolist() == gold_frame.StudyInstanceUID.tolist(),
         "OOF expert study order mismatch")
    prediction = oof[TARGETS].to_numpy(np.float64)
    truth = gold_frame[TARGETS].to_numpy(np.int8)
    need(np.isfinite(prediction).all(), "OOF contains non-finite predictions")
    need(((prediction >= 0) & (prediction <= 1)).all(), "OOF predictions outside [0,1]")
    per_target = {
        target: float(roc_auc_score(truth[:, j], prediction[:, j]))
        for j, target in enumerate(TARGETS)
    }
    macro = float(np.mean(list(per_target.values())))
    validation = manifest.get("validation", {})
    need(validation.get("gold_n") == 58, "manifest Gold count mismatch")
    need(validation.get("fold_counts") == [882, 882, 881, 881, 881],
         "manifest fold counts mismatch")
    need(validation.get("gold_fold_counts") == [12, 12, 12, 11, 11],
         "manifest Gold fold counts mismatch")
    need(abs(macro - float(validation["gold_oof_macro_auc"])) < 1e-12,
         "Gold macro AUC mismatch")
    for target in TARGETS:
        need(abs(per_target[target] - float(validation["per_target_auc"][target])) < 1e-12,
             f"Gold AUC mismatch for {target}")
    need(0 <= float(validation["derived_oof_macro_auc"]) <= 1,
         "derived OOF AUC outside [0,1]")

    report = {
        "schema": "t02_training_audit_v1", "status": "PASS",
        "manifest_sha256": sha256(manifest_path), "oof_sha256": sha256(oof_path),
        "gold58_oof_macro_auc": macro, "per_target_auc": per_target,
        "derived_oof_macro_auc": float(validation["derived_oof_macro_auc"]),
        "checkpoints": checkpoint_report,
        "fold_counts": validation["fold_counts"],
        "gold_fold_counts": validation["gold_fold_counts"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
