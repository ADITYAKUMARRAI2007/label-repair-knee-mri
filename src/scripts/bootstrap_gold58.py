#!/usr/bin/env python3
"""Quantify study-bootstrap uncertainty without fitting models or blend weights."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from pathlib import Path

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def grouped_scores(predictions: list[float]) -> list[list[int]]:
    groups = []
    for i in sorted(range(len(predictions)), key=predictions.__getitem__):
        if not groups or predictions[i] != predictions[groups[-1][0]]:
            groups.append([])
        groups[-1].append(i)
    return groups


def auc(truth: list[int], groups: list[list[int]], counts: list[int]) -> float | None:
    positives = sum(c * y for c, y in zip(counts, truth))
    negatives = sum(counts) - positives
    if not positives or not negatives:
        return None
    preceding_negatives, wins = 0, 0.0
    for group in groups:
        positive_weight = sum(counts[i] for i in group if truth[i])
        negative_weight = sum(counts[i] for i in group if not truth[i])
        wins += positive_weight * (preceding_negatives + 0.5 * negative_weight)
        preceding_negatives += negative_weight
    return wins / (positives * negatives)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--truth", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.draws < 1000:
        parser.error("use at least 1000 bootstrap draws")
    gold = [r for r in load(args.truth) if all(r[t].strip() for t in TARGETS)]
    if len(gold) != 58:
        raise RuntimeError(f"expected 58 expert studies, found {len(gold)}")
    pred = load(args.predictions)
    by_id = {r["StudyInstanceUID"]: r for r in pred}
    ids = [r["StudyInstanceUID"] for r in gold]
    if len(by_id) != len(pred) or set(by_id) != set(ids):
        raise RuntimeError("prediction IDs do not uniquely match the complete Gold58 set")
    truth = [[int(float(r[t])) for r in gold] for t in TARGETS]
    if any(float(r[t]) not in (0.0, 1.0) for r in gold for t in TARGETS):
        raise RuntimeError("expert truths must be binary")
    probabilities = [[float(by_id[uid][t]) for uid in ids] for t in TARGETS]
    if any(not math.isfinite(x) for col in probabilities for x in col):
        raise RuntimeError("prediction contains nonfinite values")
    groups = [grouped_scores(col) for col in probabilities]
    base = [auc(y, g, [1] * len(gold)) for y, g in zip(truth, groups)]
    if any(x is None for x in base):
        raise RuntimeError("original expert set has an undefined target AUC")
    rng = random.Random(args.seed)
    per_target = [[] for _ in TARGETS]
    macros = []
    undefined = [0] * len(TARGETS)
    for _ in range(args.draws):
        counts = [0] * len(gold)
        for _ in gold:
            counts[rng.randrange(len(gold))] += 1
        values = [auc(y, g, counts) for y, g in zip(truth, groups)]
        for j, value in enumerate(values):
            if value is None:
                undefined[j] += 1
            else:
                per_target[j].append(value)
        if all(x is not None for x in values):
            macros.append(sum(values) / len(values))
    if len(macros) < args.draws // 2:
        raise RuntimeError("too few draws have all 12 target AUCs defined")
    report = {
        "schema": "rsna_gold58_study_bootstrap_v1",
        "prediction_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "truth_sha256": hashlib.sha256(args.truth.read_bytes()).hexdigest(),
        "studies": len(gold), "seed": args.seed, "draws": args.draws,
        "macro_auc": sum(base) / len(base),
        "macro_auc_percentile_95_interval": [quantile(macros, .025), quantile(macros, .975)],
        "draws_with_all_targets_defined": len(macros),
        "per_target": {t: {
            "auc": base[j], "positive_studies": sum(truth[j]),
            "negative_studies": len(gold) - sum(truth[j]),
            "percentile_95_interval": [quantile(per_target[j], .025), quantile(per_target[j], .975)],
            "undefined_draws": undefined[j],
        } for j, t in enumerate(TARGETS)},
        "limitations": [
            "Studies, rather than identified patients or scanners, are resampled.",
            "Intervals condition on target AUC being defined; undefined draws are reported and excluded.",
            "This interval does not capture training randomness, checkpoint selection or train-test distribution shift.",
            "This is an uncertainty diagnostic; it does not fit blend weights or establish leaderboard performance.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in (
        "per_target", "prediction_sha256", "truth_sha256"
    )}, indent=2))


if __name__ == "__main__":
    main()
