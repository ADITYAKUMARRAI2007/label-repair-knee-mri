#!/usr/bin/env python3
"""Independent fail-closed audit for the two completed T02 split trainers."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
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
EXPECTED_INPUT = {
    "img": 224, "crop_mm": 130.0,
    "slots": ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
              "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"],
    "group": 3, "n_group": 3, "window": "0.35,0.65",
}


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def macro_auc(truth: np.ndarray, prediction: np.ndarray) -> tuple[float, dict[str, float]]:
    scores = {
        target: float(roc_auc_score(truth[:, j], prediction[:, j]))
        for j, target in enumerate(TARGETS)
    }
    return float(np.mean(list(scores.values()))), scores


def load_design(path: Path):
    spec = importlib.util.spec_from_file_location("t02_split_audit_builder", path)
    need(spec is not None and spec.loader is not None, "could not load T02 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoded, _ = module.build_payload()
    return np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dirs", type=Path, nargs=2)
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--builder", type=Path,
                        default=Path("scripts/build_t02_goldfold_trainer.py"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/T02_split_training_audit.json"))
    args = parser.parse_args()

    manifests: dict[str, tuple[Path, dict, Path, pd.DataFrame, Path]] = {}
    for directory in args.artifact_dirs:
        manifest_files = list(directory.glob("t02_manifest_*.json"))
        oof_files = list(directory.glob("t02_partial_oof_*.csv"))
        need(len(manifest_files) == len(oof_files) == 1,
             f"split artifacts missing or ambiguous in {directory}")
        manifest_path, oof_path = manifest_files[0], oof_files[0]
        manifest = json.loads(manifest_path.read_text())
        tag = manifest.get("split_tag")
        need(tag in {"f012", "f34"} and tag not in manifests,
             f"invalid or duplicate split tag {tag}")
        oof = pd.read_csv(oof_path, dtype={"StudyInstanceUID": str})
        manifests[tag] = (manifest_path, manifest, oof_path, oof, directory)
    need(set(manifests) == {"f012", "f34"}, "both T02 split manifests are required")
    need(manifests["f012"][1].get("run_folds") == [0, 1, 2], "f012 fold contract mismatch")
    need(manifests["f34"][1].get("run_folds") == [3, 4], "f34 fold contract mismatch")

    combined_histories = {}
    combined_weights = {}
    checkpoint_report = {}
    manifest_report = {}
    oof_frames = []
    for tag in ("f012", "f34"):
        manifest_path, manifest, oof_path, oof, directory = manifests[tag]
        need(manifest.get("schema") == "rsna_knee_goldfold_convnext_v1", "schema mismatch")
        need(manifest.get("split_schema") == "rsna_knee_goldfold_convnext_split_v1",
             "split schema mismatch")
        need(manifest.get("targets") == TARGETS, "target order mismatch")
        training = manifest.get("training", {})
        need(training.get("total_folds") == 5 and training.get("epochs") == 6,
             "fold or epoch count mismatch")
        need(training.get("swa_last") == 3 and training.get("gold_weight") == 4.0,
             "SWA or Gold weight mismatch")
        need(training.get("lr_backbone") == 2e-5 and training.get("lr_head") == 5e-4,
             "optimizer learning rates mismatch")
        need(manifest.get("input") == EXPECTED_INPUT, "input contract mismatch")
        provenance = manifest.get("provenance", {})
        need(provenance.get("init_sha256") == INIT_SHA256, "initialization hash mismatch")
        need(provenance.get("public_label_sha256") == PUBLIC_LABEL_SHA256,
             "public label provenance mismatch")
        validation = manifest.get("validation", {})
        need(validation.get("fold_counts") == [882, 882, 881, 881, 881],
             "manifest fold counts mismatch")
        need(validation.get("gold_fold_counts") == [12, 12, 12, 11, 11],
             "manifest Gold fold counts mismatch")
        histories = manifest.get("histories", {})
        need(set(histories) == {str(fold) for fold in manifest["run_folds"]},
             f"{tag} history fold set mismatch")
        for fold_text, history in histories.items():
            need(len(history) == 6, f"fold {fold_text} did not complete six epochs")
            need([row.get("epoch") for row in history] == list(range(1, 7)),
                 f"fold {fold_text} epoch sequence mismatch")
            for row in history:
                need(all(np.isfinite(float(row[key]))
                         for key in ("loss", "derived_auc", "gold_auc")),
                     f"fold {fold_text} has a non-finite history metric")
        overlap = set(combined_histories) & set(histories)
        need(not overlap, f"duplicate histories {sorted(overlap)}")
        combined_histories.update(histories)

        weights = manifest.get("weights", {})
        expected_names = {f"t02_goldfold_convnext_fold{fold}.pt"
                          for fold in manifest["run_folds"]}
        need(set(weights) == expected_names, f"{tag} checkpoint set mismatch")
        need(not (set(combined_weights) & set(weights)), "duplicate split checkpoint")
        combined_weights.update(weights)
        for name, record in weights.items():
            checkpoint = directory / name
            need(checkpoint.is_file(), f"missing checkpoint {name}")
            need(checkpoint.stat().st_size == int(record["bytes"]), f"size mismatch for {name}")
            digest = sha256(checkpoint)
            need(digest == record["sha256"], f"SHA-256 mismatch for {name}")
            need(zipfile.is_zipfile(checkpoint), f"invalid torch ZIP {name}")
            with zipfile.ZipFile(checkpoint) as archive:
                need(archive.testzip() is None, f"checkpoint ZIP corruption in {name}")
                members = len(archive.infolist())
            need(members >= 150, f"checkpoint {name} has too few tensor members")
            checkpoint_report[name] = {"bytes": checkpoint.stat().st_size,
                                       "sha256": digest, "zip_members": members}
        need(oof.columns.tolist() == ["StudyInstanceUID", "Fold", *TARGETS],
             f"{tag} OOF schema mismatch")
        need(len(oof) == int(validation["partial_n"]), f"{tag} OOF row count mismatch")
        need(oof.StudyInstanceUID.is_unique, f"{tag} duplicate OOF study")
        need(sha256(oof_path) == manifest["oof"]["sha256"], f"{tag} OOF hash mismatch")
        oof_frames.append(oof)
        manifest_report[tag] = {"manifest_sha256": sha256(manifest_path),
                                "oof_sha256": sha256(oof_path),
                                "elapsed_seconds": manifest.get("elapsed_seconds")}

    need(set(combined_histories) == {str(i) for i in range(5)}, "five histories not complete")
    expected_checkpoints = {f"t02_goldfold_convnext_fold{i}.pt" for i in range(5)}
    need(set(combined_weights) == expected_checkpoints, "five checkpoints not complete")

    payload = load_design(args.builder)
    uids = payload["uids"].astype(str)
    targets = payload["targets"].astype(np.float64)
    folds = payload["folds"].astype(np.int64)
    gold = payload["gold"].astype(bool)
    train = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    need(uids.tolist() == train.StudyInstanceUID.tolist(), "design/train study order mismatch")
    combined = pd.concat(oof_frames, ignore_index=True).set_index("StudyInstanceUID")
    need(len(combined) == 4407 and combined.index.is_unique, "combined OOF coverage mismatch")
    need(set(combined.index) == set(uids), "combined OOF study identity mismatch")
    combined = combined.reindex(uids)
    need(np.array_equal(combined["Fold"].to_numpy(np.int64), folds), "combined OOF fold mismatch")
    prediction = combined[TARGETS].to_numpy(np.float64)
    need(np.isfinite(prediction).all(), "combined OOF has non-finite predictions")
    need(((prediction >= 0) & (prediction <= 1)).all(), "combined OOF outside [0,1]")

    gold_truth = train.loc[gold, TARGETS].to_numpy(np.int8)
    gold_macro, per_target = macro_auc(gold_truth, prediction[gold])
    derived_macro, _ = macro_auc((targets >= 0.5).astype(np.int8), prediction)
    for tag in ("f012", "f34"):
        manifest = manifests[tag][1]
        selection = np.isin(folds, manifest["run_folds"])
        partial_gold = selection & gold
        partial_macro, partial_targets = macro_auc(
            train.loc[partial_gold, TARGETS].to_numpy(np.int8), prediction[partial_gold]
        )
        validation = manifest["validation"]
        need(int(partial_gold.sum()) == int(validation["partial_gold_n"]),
             f"{tag} partial Gold count mismatch")
        need(abs(partial_macro - float(validation["partial_gold_oof_macro_auc"])) < 1e-12,
             f"{tag} partial Gold macro mismatch")
        for target in TARGETS:
            need(abs(partial_targets[target] - float(validation["partial_per_target_auc"][target])) < 1e-12,
                 f"{tag} partial target mismatch for {target}")

    report = {
        "schema": "t02_split_training_audit_v1", "status": "PASS",
        "gold58_oof_macro_auc": gold_macro, "per_target_auc": per_target,
        "derived_oof_macro_auc": derived_macro,
        "fold_counts": np.bincount(folds, minlength=5).tolist(),
        "gold_fold_counts": np.bincount(folds[gold], minlength=5).tolist(),
        "manifests": manifest_report, "checkpoints": checkpoint_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
