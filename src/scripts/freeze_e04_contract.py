#!/usr/bin/env python3
"""Freeze E04 labels and reproduce the previously registered report-group folds.

Requires NumPy, not pandas. The historical fusion is used only to reproduce the
existing split. E04 training uses a fixed median of four report-label families;
it does not use the image teacher, JEV's Gold-tuned swaps, or calibrated Rayan
targets/weights. Public prompt development is not proved independent of Gold.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_FOLDS = "62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357"
FAMILIES = ("report_labels_v2.csv", "labels_llm_gpt56sol.csv",
            "llm_labels_v4_blend.csv", "labels_llm_soft.csv")


class Column(list):
    """Only the two column operations needed by the unchanged split algorithm."""
    def fillna(self, default):
        return Column(default if value is None else value for value in self)

    def astype(self, kind):
        return Column(kind(value) for value in self)


class Table(dict):
    def __len__(self):
        return len(self["StudyInstanceUID"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source = ROOT / "scripts/t03_contract.py"
    text = source.read_text()
    wanted = {"TARGETS", "PUBLIC_LABEL_SHA256", "_normalise_report", "_simhash",
              "report_groups", "make_folds", "local_label_paths"}
    nodes = []
    for node in ast.parse(text).body:
        names = {getattr(node, "name", "")}
        if isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        if wanted & names:
            nodes.append(ast.get_source_segment(text, node))
    scope = {"np": np, "hashlib": hashlib, "re": re,
             "unicodedata": unicodedata, "Path": Path}
    exec("from __future__ import annotations\n" + "\n\n".join(nodes), scope)
    targets_order = scope["TARGETS"]
    with (ROOT / "data/train.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    uids = [r["StudyInstanceUID"] for r in rows]
    if len(uids) != 4407 or len(set(uids)) != len(uids):
        raise RuntimeError("official study identity drift")
    gold = np.asarray([all(r[t] != "" for t in targets_order) for r in rows], np.uint8)
    if int(gold.sum()) != 58:
        raise RuntimeError("expert-label coverage drift")
    truth = np.asarray([[float(r[t]) if r[t] else np.nan for t in targets_order]
                        for r in rows], np.float32)
    if not np.isin(truth[gold > 0], (0, 1)).all():
        raise RuntimeError("expert truth is not binary")
    paths = scope["local_label_paths"](ROOT / "labels")
    arrays, receipts = {}, {}
    for name, path in paths.items():
        actual = digest(path)
        if actual != scope["PUBLIC_LABEL_SHA256"][name]:
            raise RuntimeError(f"source drift: {name}")
        with path.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        index = {r["StudyInstanceUID"]: r for r in records}
        if len(index) != len(records):
            raise RuntimeError(f"duplicate UID: {name}")
        arrays[name] = np.asarray([
            [float(index.get(uid, {}).get(t) or "nan") for t in targets_order]
            for uid in uids], np.float32)
        receipts[name] = {"path": str(path.relative_to(ROOT)), "sha256": actual}

    historical = np.clip(.60 * arrays["train_targets_soft.csv"] + .40 * np.nanmedian(
        np.stack([arrays[n] for n in (*FAMILIES, "report_labels_v5.csv")]), axis=0),
        .01, .99).astype(np.float32)
    historical[gold > 0] = truth[gold > 0]
    frame = Table(StudyInstanceUID=Column(uids), Report=Column(r["Report"] for r in rows))
    folds, audit = scope["make_folds"](frame, historical, gold)
    if audit["fold_sha256"] != EXPECTED_FOLDS:
        raise RuntimeError(f"historical split failed reproduction: {audit}")
    groups = scope["report_groups"](frame["Report"])
    weak = gold == 0
    family_values = np.stack([arrays[n] for n in FAMILIES])
    coverage = np.isfinite(family_values).sum(0)
    if np.any(coverage[weak] < 2):
        raise RuntimeError("weak-label cell has fewer than two report sources")
    probabilities = np.clip(np.nanmedian(family_values[:, weak], axis=0), .01, .99)
    spread = np.nanmax(family_values[:, weak], axis=0) - np.nanmin(family_values[:, weak], axis=0)
    y = truth.copy()
    y[weak] = probabilities
    weights = np.full_like(y, 2.0)
    weights[weak] = np.clip(1.0 - spread, .25, 1.0)
    if not np.isfinite(y).all() or not np.isfinite(weights).all():
        raise RuntimeError("nonfinite frozen labels/weights")
    out = ROOT / "runs/E04_contract"
    out.mkdir(parents=True, exist_ok=True)
    for name, array in (("e04_targets.npy", y), ("e04_weights.npy", weights)):
        path = out / name
        if path.exists() and not np.array_equal(np.load(path), array):
            raise RuntimeError(f"refusing to replace a different frozen artifact: {name}")
        if not path.exists():
            np.save(path, array)
    contract = {
        "schema": "e04_frozen_report_contract_v1", "status": "FROZEN",
        "target_order": targets_order, "official_train_sha256": digest(ROOT / "data/train.csv"),
        "split_source_sha256": digest(source), "split_audit": audit,
        "fusion": {"families": list(FAMILIES), "rule": "cellwise median, clipped 0.01--0.99",
                   "weak_weight": "clip(1 - source probability range, 0.25, 1.0)",
                   "expert_weight": 2.0, "expert_labels": "official binary truth in training folds only",
                   "minimum_weak_cell_sources": int(coverage[weak].min()),
                   "source_receipts": {n: receipts[n] for n in FAMILIES}},
        "fold_reconstruction_only_sources": receipts,
        "arrays": {n: digest(out / n) for n in ("e04_targets.npy", "e04_weights.npy")},
        "validation_rules": [
            "five independently trained backbones; no shared expert-adapted backbone",
            "exclude every held-out report group from model updates",
            "fixed checkpoint rule before training; do not select on expert-fold scores",
            "complete all 58 expert OOF predictions before the coarse validation gate"],
        "limitations": ["report grouping does not establish patient/site separation",
                        "public report-label prompt tuning independence from Gold is unverified",
                        "previous experiments have already used Gold; OOF is a coarse rejection gate"],
        "rows": [{"StudyInstanceUID": uid, "fold": int(f), "report_group": int(g),
                  "expert": bool(ex)} for uid, f, g, ex in zip(uids, folds, groups, gold)]}
    path = out / "e04_contract.json"
    encoded = json.dumps(contract, indent=2) + "\n"
    if path.exists() and path.read_text() != encoded:
        raise RuntimeError("refusing to overwrite a different frozen contract")
    path.write_text(encoded)
    print(json.dumps({"status": "PASS", "fold_audit": audit,
                      "contract_sha256": digest(path), "arrays": contract["arrays"],
                      "minimum_weak_cell_sources": int(coverage[weak].min())}, indent=2))


if __name__ == "__main__":
    main()
