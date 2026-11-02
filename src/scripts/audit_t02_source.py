#!/usr/bin/env python3
"""Fail-closed static audit for the generated T02 Kaggle trainer."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


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
DATASET_SOURCES = {
    "stevenleehans/rsna-knee-llm-report-labels",
    "lixin73/rsna-knee-llm-report-labels-sol56",
    "pilkwang/rsna-knee-llm-labels",
    "rayanbabur/rsna-knee-2026-calibrated-soft-targets",
    "dreaddevelopment/rsna-knee-labels",
    "yunusgmsoy/rsna-knee-llm-labels-4-source-merged",
}


def need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_builder(path: Path):
    spec = importlib.util.spec_from_file_location("t02_builder_audit", path)
    need(spec is not None and spec.loader is not None, "could not load T02 builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--builder", type=Path,
        default=Path("scripts/build_t02_goldfold_trainer.py"),
    )
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--run-folds", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("runs/T02_source_audit.json"))
    args = parser.parse_args()

    text = args.script.read_text()
    metadata = json.loads(args.metadata.read_text())

    # Uploaded code derives labels from mounted public datasets. No study identifiers
    # or locally constructed target arrays may be embedded in the script.
    need("T02_PAYLOAD" not in text, "generated source contains obsolete embedded payload")
    need("base64.b64decode" not in text, "generated source contains a base64 decoder")
    need(not re.search(r'1\.2\.840\.113619\.', text), "generated source contains a DICOM UID")
    need("_construct_targets(train)" in text, "runtime target construction missing")
    for name, digest in PUBLIC_LABEL_SHA256.items():
        need(f'"{name}": "{digest}"' in text, f"pinned public hash missing for {name}")

    # Recreate the design locally to verify target bounds, expert override, and folds.
    # This local artifact is used only by this audit and is never uploaded.
    builder = load_builder(args.builder)
    encoded, payload_info = builder.build_payload()
    raw = base64.b64decode(encoded)
    payload = np.load(io.BytesIO(raw), allow_pickle=False)
    uids = payload["uids"].astype(str)
    targets = payload["targets"].astype(np.float64)
    weights = payload["weights"].astype(np.float64)
    folds = payload["folds"].astype(np.int64)
    gold = payload["gold"].astype(bool)

    train = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    gold_truth = train[TARGETS].notna().all(axis=1).to_numpy()
    need(uids.tolist() == train.StudyInstanceUID.tolist(), "local UID order mismatch")
    need(targets.shape == weights.shape == (4407, 12), "target/weight shape mismatch")
    need(np.isfinite(targets).all() and np.isfinite(weights).all(), "non-finite targets")
    need(((targets >= 0) & (targets <= 1)).all(), "targets outside [0,1]")
    need(np.array_equal(gold, gold_truth) and int(gold.sum()) == 58, "Gold58 mismatch")
    need(np.array_equal(targets[gold], train.loc[gold, TARGETS].to_numpy(np.float64)),
         "Gold targets are not exact expert truth")
    need(np.all(weights[gold] == 4.0), "Gold weights are not 4.0")
    need(set(folds.tolist()) == set(range(5)), "fold IDs mismatch")
    need(np.bincount(folds, minlength=5).tolist() == [882, 882, 881, 881, 881],
         "fold sizes are not balanced")
    need(np.bincount(folds[gold], minlength=5).tolist() == [12, 12, 12, 11, 11],
         "Gold fold sizes are not balanced")
    truth = targets[gold].astype(np.int8)
    for fold in range(5):
        y = truth[folds[gold] == fold]
        need(all(np.unique(y[:, j]).size == 2 for j in range(12)),
             f"Gold fold {fold} lacks both classes for a target")

    required_fragments = [
        'T02_SCHEMA = "rsna_knee_goldfold_convnext_v1"',
        f'CFG.epochs = {args.epochs}', 'CFG.swa_epochs = 3',
        'CFG.batch_studies = 16', 'CFG.eval_batch = 32',
        'nn.DataParallel(base', 'runtime_folds != fold', 'runtime_folds == fold',
        'T02_INIT_SHA256 = "bec04468edeaaa0c75191a221d4e3b80bada5cfd263eeec36cb093548de93147"',
        'if not torch.cuda.is_available() or torch.cuda.device_count() < 2:',
        'torch.backends.cudnn.benchmark = True',
        '_load_cache(uids)', 'cache_meta.json',
        'targets = np.clip(0.60 * rayan + 0.40 * np.nanmedian(np.stack(others), axis=0)',
        'weights[gold] = 4.0',
    ]
    missing = [fragment for fragment in required_fragments if fragment not in text]
    need(not missing, f"generated source contract missing: {missing}")
    if args.run_folds is None:
        need('t02_manifest.json' in text, "full-run manifest contract missing")
    else:
        expected_folds = sorted(args.run_folds)
        need(expected_folds in ([0, 1, 2], [3, 4]), "unexpected split fold set")
        need(f"T02_RUN_FOLDS = {expected_folds!r}" in text,
             "split source fold set mismatch")
        need('T02_SPLIT_SCHEMA = "rsna_knee_goldfold_convnext_split_v1"' in text,
             "split schema missing")
        need('t02_partial_oof_' in text and 't02_manifest_' in text,
             "split output contract missing")
        need('completed = np.flatnonzero(np.isfinite(oof).all(axis=1))' in text,
             "split OOF completeness check missing")
    need(metadata.get("machine_shape") == "NvidiaTeslaT4", "dual T4 metadata missing")
    need(metadata.get("enable_internet") is False, "internet must be disabled")
    need(metadata.get("competition_sources") == ["rsna-knee-abnormality-detection"],
         "competition source mismatch")
    need(set(metadata.get("dataset_sources", [])) == DATASET_SOURCES,
         "public label dataset source mismatch")
    need(set(metadata.get("kernel_sources", [])) == {
        "stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache",
        "adityakumarrai0401/rsna-knee-convnext-diversity-train-v2",
    }, "kernel source mismatch")

    report = {
        "schema": "t02_source_audit_v2", "status": "PASS",
        "script_sha256": hashlib.sha256(args.script.read_bytes()).hexdigest(),
        "local_design_sha256": hashlib.sha256(raw).hexdigest(),
        "public_label_sha256": PUBLIC_LABEL_SHA256,
        "studies": len(uids), "gold": int(gold.sum()),
        "fold_counts": np.bincount(folds, minlength=5).tolist(),
        "gold_fold_counts": np.bincount(folds[gold], minlength=5).tolist(),
        "gold_positive_counts_by_fold": [
            truth[folds[gold] == fold].sum(axis=0).astype(int).tolist()
            for fold in range(5)
        ],
        "payload_embedded": False,
        "runtime_public_reconstruction": True,
        "run_folds": args.run_folds if args.run_folds is not None else list(range(5)),
        "payload_info": payload_info,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
