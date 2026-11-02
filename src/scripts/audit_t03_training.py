#!/usr/bin/env python3
"""Merge and independently audit all five completed T03 fold artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from t03_contract import TARGETS, construct_targets, local_label_paths, make_folds


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "rsna_knee_t03_effb0_slotmil_v1"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def aucs(truth, pred):
    values = []
    for column in range(len(TARGETS)):
        if len(np.unique(truth[:, column])) != 2:
            values.append(float("nan"))
        else:
            values.append(float(roc_auc_score(truth[:, column], pred[:, column])))
    return np.asarray(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "runs/T03")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    train = pd.read_csv(ROOT / "data/train.csv", dtype={"StudyInstanceUID": str})
    targets, _, gold = construct_targets(train, local_label_paths(ROOT / "labels"))
    folds, fold_audit = make_folds(train, targets, gold)
    expected = dict(zip(train.StudyInstanceUID.astype(str), folds))
    frames, manifests, checkpoint_info = [], [], {}
    for fold in range(5):
        directory = args.root / f"fold{fold}"
        manifest_path = directory / f"t03_manifest_fold{fold}.json"
        checkpoint = directory / f"t03_effb0_fold{fold}.pt"
        oof_path = directory / f"t03_oof_fold{fold}.csv"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema"] == SCHEMA and manifest["status"] == "COMPLETE"
        assert manifest["fold"] == fold and manifest["fold_audit"] == fold_audit
        for artifact in (checkpoint, oof_path):
            recorded = manifest["artifacts"][artifact.name]
            assert artifact.stat().st_size == recorded["bytes"]
            assert sha256(artifact) == recorded["sha256"]
        with zipfile.ZipFile(checkpoint) as archive:
            bad = archive.testzip()
            assert bad is None and len(archive.namelist()) > 300
        frame = pd.read_csv(oof_path, dtype={"StudyInstanceUID": str})
        assert frame.columns.tolist() == ["StudyInstanceUID", "Fold", *TARGETS]
        assert len(frame) == int((folds == fold).sum()) and frame.Fold.eq(fold).all()
        assert frame.StudyInstanceUID.is_unique
        assert np.isfinite(frame[TARGETS].to_numpy()).all()
        assert ((frame[TARGETS].to_numpy() >= 0) & (frame[TARGETS].to_numpy() <= 1)).all()
        assert all(expected[uid] == fold for uid in frame.StudyInstanceUID)
        frames.append(frame)
        manifests.append(manifest)
        checkpoint_info[checkpoint.name] = manifest["artifacts"][checkpoint.name]

    merged = pd.concat(frames, ignore_index=True)
    assert len(merged) == len(train) and merged.StudyInstanceUID.is_unique
    assert set(merged.StudyInstanceUID) == set(train.StudyInstanceUID.astype(str))
    merged = train[["StudyInstanceUID"]].merge(merged, on="StudyInstanceUID", validate="one_to_one")
    pred = merged[TARGETS].to_numpy(np.float64)
    hard = (targets >= 0.5).astype(np.int8)
    gold_mask = gold > 0
    derived_target = aucs(hard, pred)
    gold_target = aucs(train.loc[gold_mask, TARGETS].to_numpy(np.int8), pred[gold_mask])
    fold_derived = [float(np.nanmean(aucs(hard[folds == f], pred[folds == f]))) for f in range(5)]
    fold_gold = [float(np.nanmean(aucs(train.loc[(folds == f) & gold_mask, TARGETS].to_numpy(np.int8),
                                        pred[(folds == f) & gold_mask]))) for f in range(5)]

    rng = np.random.default_rng(20260930)
    gold_truth = train.loc[gold_mask, TARGETS].to_numpy(np.int8)
    gold_pred = pred[gold_mask]
    samples = []
    attempts = 0
    while len(samples) < args.bootstrap and attempts < args.bootstrap * 20:
        attempts += 1
        take = rng.integers(0, len(gold_truth), len(gold_truth))
        score = aucs(gold_truth[take], gold_pred[take])
        if np.isfinite(score).all():
            samples.append(float(score.mean()))
    if len(samples) < args.bootstrap:
        raise RuntimeError("insufficient valid bootstrap samples")
    interval = np.quantile(samples, [0.05, 0.50, 0.95]).tolist()

    merged_path = args.root / "t03_oof_merged.csv"
    merged.to_csv(merged_path, index=False)
    result = {
        "status": "PASS", "schema": "t03_training_audit_v1",
        "fold_audit": fold_audit,
        "derived_macro_auc": float(np.nanmean(derived_target)),
        "derived_target_auc": dict(zip(TARGETS, derived_target.tolist())),
        "gold_macro_auc": float(np.nanmean(gold_target)),
        "gold_target_auc": dict(zip(TARGETS, gold_target.tolist())),
        "fold_derived_auc": fold_derived, "fold_gold_auc": fold_gold,
        "gold_bootstrap_5_50_95": interval,
        "checkpoints": checkpoint_info,
        "merged_oof": {"file": str(merged_path), "sha256": sha256(merged_path)},
    }
    output = args.root / "t03_training_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
