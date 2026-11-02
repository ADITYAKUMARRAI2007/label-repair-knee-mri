#!/usr/bin/env python3
"""Independently verify E03 completed artifacts and expert validation using stdlib.

This does not submit, choose blend weights, or infer public leaderboard strength.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

from bootstrap_gold58 import TARGETS, load, auc, grouped_scores

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", nargs="?", type=Path, default=ROOT / "runs/E03_training")
    args = parser.parse_args()
    directory = args.artifact_dir
    manifest = json.loads((directory / "e03_manifest.json").read_text())
    if manifest["status"] != "COMPLETE":
        raise RuntimeError("E03 is not complete")
    contract = manifest["contract"]
    if (contract["epochs"] != 14 or contract["checkpoint_rule"] != "final_epoch_ema_only"
            or contract["gold_rows_used_for_training"] != 0 or contract["gold_checkpoint_selection"]):
        raise RuntimeError("training contract drift")
    if not manifest["strict_gold_holdout"] or not manifest["gold_report_groups_excluded_from_training"]:
        raise RuntimeError("holdout contract failed")
    contract_digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    if contract_digest != manifest["contract_sha256"]:
        raise RuntimeError("training contract hash mismatch")
    if [r["epoch"] for r in manifest["history"]] != list(range(1, 15)):
        raise RuntimeError("missing or duplicated completed epoch")
    for name, expected in manifest["artifacts"].items():
        path = directory / name
        if path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise RuntimeError(f"artifact integrity failed: {name}")
    spec = importlib.util.spec_from_file_location("e03_contract", ROOT / "training/e03_split_contract.py")
    split_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(split_module)
    truth_records = load(ROOT / "data/train.csv")
    independent_split = split_module.freeze_split(truth_records, TARGETS)
    supplied_split = json.loads((directory / "e03_split.json").read_text())
    if independent_split != supplied_split:
        raise RuntimeError("split does not match independently frozen report groups")
    if split_module.split_sha256(independent_split) != contract["split_sha256"]:
        raise RuntimeError("split hash mismatch")
    local_split_hash = json.loads((ROOT / "runs/E03_contract/split_audit.json").read_text())["split_sha256"]
    if contract["split_sha256"] != local_split_hash:
        raise RuntimeError("split differs from preregistration")
    truth = {r["StudyInstanceUID"]: r for r in truth_records}
    predictions = load(directory / "e03_gold_predictions.csv")
    ids = [r["StudyInstanceUID"] for r in predictions]
    if len(set(ids)) != len(ids) or set(ids) != set(independent_split["gold"]):
        raise RuntimeError("Gold58 predictions have incorrect identities")
    per_target = {}
    for t in TARGETS:
        values = [float(r[t]) for r in predictions]
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
            raise RuntimeError(f"invalid expert probabilities: {t}")
        labels = [int(float(truth[uid][t])) for uid in ids]
        score = auc(labels, grouped_scores(values), [1] * len(ids))
        if score is None:
            raise RuntimeError(f"undefined expert AUC: {t}")
        if abs(score - manifest["validation"]["gold_per_target_auc"][t]) > 1e-9:
            raise RuntimeError(f"per-target AUC mismatch: {t}")
        per_target[t] = score
    macro = sum(per_target.values()) / len(TARGETS)
    if abs(macro - manifest["validation"]["gold_auc"]) > 1e-9:
        raise RuntimeError("macro expert AUC mismatch")
    weak = load(directory / "e03_weak_predictions.csv")
    weak_ids = [r["StudyInstanceUID"] for r in weak]
    if len(set(weak_ids)) != len(weak_ids) or set(weak_ids) != set(independent_split["weak_validation"]):
        raise RuntimeError("weak validation identities do not match frozen split")
    if any(not math.isfinite(float(r[t])) or not 0 <= float(r[t]) <= 1 for r in weak for t in TARGETS):
        raise RuntimeError("invalid weak validation probabilities")
    baseline = {r["StudyInstanceUID"]: r for r in load(ROOT / "runs/E02_training/e02_gold_predictions.csv")}
    if set(baseline) != set(ids):
        raise RuntimeError("E02 paired comparison does not cover the same expert set")
    paired = {}
    for t in TARGETS:
        labels = [int(float(truth[uid][t])) for uid in ids]
        values = [float(baseline[uid][t]) for uid in ids]
        paired[t] = per_target[t] - auc(labels, grouped_scores(values), [1] * len(ids))
    report = {"schema": "e03_training_audit_v1", "status": "PASS",
              "gold_auc": macro, "gold_per_target_auc": per_target,
              "paired_E02_auc_delta": sum(paired.values()) / len(paired),
              "paired_E02_per_target_delta": paired,
              "minimum_Gate_B_auc_threshold_met": macro >= 0.90,
              "public_score": None, "blend_eligibility": "unproven",
              "split_sha256": contract["split_sha256"], "contract_sha256": contract_digest,
              "checkpoint_sha256": digest(directory / "e03_resnet34_series_ema.pt"),
              "limits": ["Gold58 remains a coarse rejection gate, not a precise public score estimate.",
                         "Report grouping does not prove patient/scanner independence.",
                         "Gold58 is excluded from this reader's updates; upstream teacher training/exclusion provenance remains unverified.",
                         "Fold/target stability, paired uncertainty and official scoring still require review.",
                         "Weak validation AUC is author-run training health against soft-label thresholds; it is not independently reproduced here."]}
    (directory / "e03_training_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
