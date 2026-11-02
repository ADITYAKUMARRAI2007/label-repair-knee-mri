#!/usr/bin/env python3
"""Pure-data contracts shared by the T03 builder, auditor and Kaggle trainers."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def construct_targets(train: pd.DataFrame, paths: dict[str, str | Path]):
    """Build the preregistered weak-label fusion and fold-safe expert weights."""
    uids = train["StudyInstanceUID"].astype(str).tolist()

    def read(name: str) -> np.ndarray:
        path = Path(paths[name])
        actual = sha256_file(path)
        if actual != PUBLIC_LABEL_SHA256[name]:
            raise RuntimeError(f"{name} SHA-256 drift: {actual}")
        frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
        if frame["StudyInstanceUID"].duplicated().any():
            raise RuntimeError(f"duplicate study in {name}")
        values = frame.set_index("StudyInstanceUID").reindex(uids)[TARGETS]
        return values.to_numpy(np.float32)

    rayan = read("train_targets_soft.csv")
    others = np.stack([
        read("report_labels_v2.csv"),
        read("labels_llm_gpt56sol.csv"),
        read("llm_labels_v4_blend.csv"),
        read("labels_llm_soft.csv"),
        read("report_labels_v5.csv"),
    ])
    targets = np.clip(0.60 * rayan + 0.40 * np.nanmedian(others, axis=0),
                      0.01, 0.99).astype(np.float32)
    weights = np.clip(np.nan_to_num(read("train_targets_weights.csv"), nan=0.35),
                      0.25, 1.0).astype(np.float32)
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    targets[gold] = train.loc[gold, TARGETS].to_numpy(np.float32)
    weights[gold] = 2.0
    if targets.shape != (4407, 12) or not np.isfinite(targets).all():
        raise RuntimeError(f"target contract failed: {targets.shape}")
    if int(gold.sum()) != 58:
        raise RuntimeError(f"Gold count drift: {int(gold.sum())}")
    return targets, weights, gold.astype(np.uint8)


def _normalise_report(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"\b\d+(?:[.,]\d+)?\b", " <n> ", text)
    text = re.sub(r"[^a-z<>]+", " ", text)
    return " ".join(text.split())


def _simhash(text: str) -> int:
    words = text.split()
    grams = [" ".join(words[i:i + 3]) for i in range(max(len(words) - 2, 1))]
    if len(words) < 3:
        grams = [text]
    votes = np.zeros(64, np.int32)
    for gram in grams:
        value = int.from_bytes(hashlib.sha1(gram.encode()).digest()[:8], "big")
        for bit in range(64):
            votes[bit] += 1 if value & (1 << bit) else -1
    out = 0
    for bit in range(64):
        if votes[bit] >= 0:
            out |= 1 << bit
    return out


def report_groups(reports) -> np.ndarray:
    """Join exact and extremely close report templates without stochastic NLP."""
    normal = [_normalise_report(v) for v in reports]
    n = len(normal)
    parent = np.arange(n, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    exact: dict[str, int] = {}
    for i, text in enumerate(normal):
        if text in exact:
            union(i, exact[text])
        else:
            exact[text] = i

    hashes = [_simhash(text) for text in normal]
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, value in enumerate(hashes):
        for band in range(4):
            buckets.setdefault((band, (value >> (16 * band)) & 0xFFFF), []).append(i)
    checked = set()
    for members in buckets.values():
        if len(members) > 80:  # generic language buckets are not evidence of duplication
            continue
        for p, a in enumerate(members):
            for b in members[p + 1:]:
                pair = (min(a, b), max(a, b))
                if pair in checked:
                    continue
                checked.add(pair)
                la, lb = len(normal[a]), len(normal[b])
                if min(la, lb) < 40 or min(la, lb) / max(la, lb, 1) < 0.90:
                    continue
                if (hashes[a] ^ hashes[b]).bit_count() <= 3:
                    union(a, b)

    roots = [find(i) for i in range(n)]
    labels = {root: j for j, root in enumerate(sorted(set(roots)))}
    return np.asarray([labels[root] for root in roots], np.int32)


def make_folds(train: pd.DataFrame, targets: np.ndarray, gold: np.ndarray,
               seed: int = 20260930) -> tuple[np.ndarray, dict]:
    """Five grouped, multilabel-balanced folds with Gold balanced independently."""
    groups = report_groups(train["Report"].fillna(""))
    unique = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in unique}
    gold_groups = [g for g in unique if gold[members[g]].any()]
    if any(int(gold[members[g]].sum()) != 1 for g in gold_groups):
        raise RuntimeError("near-duplicate grouping joined expert-labelled studies")
    rng = np.random.default_rng(seed)
    gold_idx = np.flatnonzero(gold > 0)
    gold_y = targets[gold_idx].astype(np.int8)
    capacities = np.asarray([12, 12, 12, 11, 11], np.int64)
    total_gold_pos = gold_y.sum(0)
    best = None
    for _ in range(50000):
        permutation = rng.permutation(len(gold_idx))
        gold_folds = np.empty(len(gold_idx), np.int8)
        start = 0
        for fold, count in enumerate(capacities):
            gold_folds[permutation[start:start + count]] = fold
            start += count
        fold_pos = np.stack([gold_y[gold_folds == fold].sum(0) for fold in range(5)])
        if np.any(fold_pos == 0) or np.any(fold_pos == capacities[:, None]):
            continue
        expected = capacities[:, None] * total_gold_pos[None, :] / len(gold_idx)
        denom = np.maximum(expected * (1 - total_gold_pos[None, :] / len(gold_idx)), 0.5)
        objective = float(np.sum((fold_pos - expected) ** 2 / denom))
        if best is None or objective < best[0]:
            assignment = {int(groups[idx]): int(fold)
                          for idx, fold in zip(gold_idx, gold_folds)}
            best = (objective, assignment, capacities.astype(float), fold_pos.astype(float))
    if best is None:
        raise RuntimeError("could not construct valid grouped Gold folds")

    assignment = best[1]
    hard = (targets >= 0.5).astype(np.float64)
    desired_n = len(train) / 5
    desired_pos = hard.sum(0) / 5
    fold_n = np.zeros(5, np.float64)
    fold_pos = np.zeros((5, len(TARGETS)), np.float64)
    for g, fold in assignment.items():
        idx = members[g]
        fold_n[fold] += len(idx)
        fold_pos[fold] += hard[idx].sum(0)

    remaining = [int(g) for g in unique if int(g) not in assignment]
    rarity = 1 / np.maximum(hard.sum(0), 1)
    remaining.sort(key=lambda g: (
        -float((hard[members[g]].sum(0) * rarity).sum()),
        -len(members[g]), g,
    ))
    for g in remaining:
        idx = members[g]
        n = len(idx)
        pos = hard[idx].sum(0)
        costs = []
        for fold in range(5):
            size_before = ((fold_n[fold] - desired_n) / desired_n) ** 2
            size_after = ((fold_n[fold] + n - desired_n) / desired_n) ** 2
            scale = np.maximum(desired_pos, 1)
            pos_before = np.mean(((fold_pos[fold] - desired_pos) / scale) ** 2)
            pos_after = np.mean(((fold_pos[fold] + pos - desired_pos) / scale) ** 2)
            costs.append((size_after - size_before) + 0.35 * (pos_after - pos_before))
        chosen = min(range(5), key=lambda f: (costs[f], fold_n[f], f))
        assignment[g] = chosen
        fold_n[chosen] += n
        fold_pos[chosen] += pos

    folds = np.asarray([assignment[int(g)] for g in groups], np.int8)
    gold_counts = np.bincount(folds[gold > 0], minlength=5)
    gold_pos = np.stack([targets[(folds == f) & (gold > 0)].sum(0) for f in range(5)])
    if np.any(gold_counts < 9) or np.any(gold_counts > 14):
        raise RuntimeError(f"Gold fold counts invalid: {gold_counts.tolist()}")
    if np.any(gold_pos <= 0) or np.any(gold_pos >= gold_counts[:, None]):
        raise RuntimeError("a Gold fold lacks a class for at least one target")
    if any(len(np.unique(folds[idx])) != 1 for idx in members.values()):
        raise RuntimeError("report group crossed fold boundary")

    mapping = "".join(f"{uid},{int(fold)}\n" for uid, fold in
                      sorted(zip(train["StudyInstanceUID"].astype(str), folds)))
    audit = {
        "schema": "t03_grouped_folds_v1",
        "seed": seed,
        "fold_sha256": hashlib.sha256(mapping.encode()).hexdigest(),
        "fold_counts": np.bincount(folds, minlength=5).tolist(),
        "gold_fold_counts": gold_counts.tolist(),
        "report_groups": int(len(unique)),
        "duplicate_groups": int(sum(len(v) > 1 for v in members.values())),
        "largest_group": int(max(map(len, members.values()))),
    }
    return folds, audit


def local_label_paths(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    locations = {
        "train_targets_soft.csv": "rayanbabur_rsna-knee-2026-calibrated-soft-targets",
        "train_targets_weights.csv": "rayanbabur_rsna-knee-2026-calibrated-soft-targets",
        "report_labels_v2.csv": "pilkwang_rsna-knee-llm-labels",
        "labels_llm_gpt56sol.csv": "lixin73_rsna-knee-llm-report-labels-sol56",
        "llm_labels_v4_blend.csv": "stevenleehans_rsna-knee-llm-report-labels",
        "labels_llm_soft.csv": "dreaddevelopment_rsna-knee-labels",
        "report_labels_v5.csv": "yunusgmsoy_rsna-knee-llm-labels-4-source-merged",
    }
    return {name: root / directory / name for name, directory in locations.items()}


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    train = pd.read_csv(repo / "data/train.csv", dtype={"StudyInstanceUID": str})
    targets, weights, gold = construct_targets(train, local_label_paths(repo / "labels"))
    folds, audit = make_folds(train, targets, gold)
    audit.update({"target_shape": list(targets.shape), "weight_range": [float(weights.min()),
                  float(weights.max())]})
    print(json.dumps(audit, indent=2, sort_keys=True))
