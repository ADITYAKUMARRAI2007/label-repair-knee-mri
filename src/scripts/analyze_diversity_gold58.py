#!/usr/bin/env python3
"""Compare T01 and E01 predictions on the held-out 58 expert studies.

This is a diagnostic gate, not a hyperparameter tuner.  It reports uncertainty,
target-level rank correlation, and a small blend grid so weak or redundant arms
can be rejected before spending a leaderboard submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]

# Public three-submission readout from topic 744294. These groups are a useful
# diagnostic view only; their ordering came from one 0.940 CoAtNet and is not
# assumed to transfer exactly to our models.
TARGET_GROUPS = {
    "reported_strong": ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus"],
    "reported_middle": ["Synovitis", "Baker's", "Medial OA", "Fracture"],
    "reported_weak": ["Effusion", "Contusion", "PF OA", "Lateral OA"],
}


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(method="average", pct=True).to_numpy(np.float64)


def auc(y: np.ndarray, p: np.ndarray) -> float | None:
    valid = np.isfinite(y) & np.isfinite(p)
    if valid.sum() < 3 or np.unique(y[valid]).size < 2:
        return None
    return float(roc_auc_score(y[valid], p[valid]))


def macro(y: np.ndarray, p: np.ndarray) -> float:
    values = [auc(y[:, j], p[:, j]) for j in range(y.shape[1])]
    finite = [v for v in values if v is not None]
    return float(np.mean(finite)) if finite else float("nan")


def group_macro(y: np.ndarray, p: np.ndarray, names: list[str]) -> float:
    positions = [TARGETS.index(name) for name in names]
    return macro(y[:, positions], p[:, positions])


def percentile(values: list[float], q: float) -> float | None:
    finite = np.asarray([v for v in values if np.isfinite(v)], np.float64)
    return float(np.percentile(finite, q)) if len(finite) else None


def load_convnext(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    missing = [t for t in TARGETS if t not in frame]
    if missing:
        raise ValueError(f"ConvNeXt OOF is missing targets: {missing}")
    if frame.StudyInstanceUID.duplicated().any():
        raise ValueError("ConvNeXt OOF contains duplicate studies")
    return frame[["StudyInstanceUID", *TARGETS]]


def load_effnet(path: Path) -> pd.DataFrame:
    with np.load(path, allow_pickle=True) as payload:
        pred = np.asarray(payload["pred"], np.float64)
        ids = payload["ids"].astype(str)
    if pred.shape != (len(ids), len(TARGETS)):
        raise ValueError(f"unexpected EfficientNet prediction shape: {pred.shape}")
    return pd.DataFrame(pred, columns=TARGETS).assign(StudyInstanceUID=ids)[
        ["StudyInstanceUID", *TARGETS]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--convnext", type=Path, required=True)
    parser.add_argument("--effnet", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/diversity_gold58.json"))
    parser.add_argument("--bootstrap", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()

    truth = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    truth = truth[truth[TARGETS].notna().all(axis=1)][["StudyInstanceUID", *TARGETS]]
    cx = load_convnext(args.convnext)
    ev = load_effnet(args.effnet)
    joined = truth.merge(cx, on="StudyInstanceUID", suffixes=("_truth", "_cx"), validate="one_to_one")
    joined = joined.merge(ev, on="StudyInstanceUID", validate="one_to_one")
    if len(joined) != len(truth):
        raise RuntimeError(f"prediction coverage is {len(joined)}/{len(truth)} expert studies")

    y = joined[[f"{t}_truth" for t in TARGETS]].to_numpy(np.float64)
    pcx = joined[[f"{t}_cx" for t in TARGETS]].to_numpy(np.float64)
    pev = joined[TARGETS].to_numpy(np.float64)
    rcx, rev = rank01(pcx), rank01(pev)

    rng = np.random.default_rng(args.seed)
    boot = {"convnext": [], "effnet": []}
    group_boot = {
        model: {group: [] for group in TARGET_GROUPS}
        for model in ("convnext", "effnet")
    }
    target_delta_boot = {target: [] for target in TARGETS}
    grid = np.linspace(0, 0.5, 11)
    blend_boot = {f"{a:.2f}": [] for a in grid}
    for _ in range(args.bootstrap):
        take = rng.integers(0, len(y), len(y))
        yt, cxt, evt = y[take], rcx[take], rev[take]
        cx_score, ev_score = macro(yt, cxt), macro(yt, evt)
        boot["convnext"].append(cx_score)
        boot["effnet"].append(ev_score)
        for group, names in TARGET_GROUPS.items():
            group_boot["convnext"][group].append(group_macro(yt, cxt, names))
            group_boot["effnet"][group].append(group_macro(yt, evt, names))
        for j, target in enumerate(TARGETS):
            cx_auc = auc(yt[:, j], cxt[:, j])
            ev_auc = auc(yt[:, j], evt[:, j])
            if cx_auc is not None and ev_auc is not None:
                target_delta_boot[target].append(ev_auc - cx_auc)
        for alpha in grid:
            blend_boot[f"{alpha:.2f}"].append(
                macro(yt, (1 - alpha) * cxt + alpha * evt)
            )

    per_target = {}
    for j, target in enumerate(TARGETS):
        corr = float(pd.Series(rcx[:, j]).corr(pd.Series(rev[:, j]), method="spearman"))
        per_target[target] = {
            "positives": int(y[:, j].sum()),
            "convnext_auc": auc(y[:, j], rcx[:, j]),
            "effnet_auc": auc(y[:, j], rev[:, j]),
            "prediction_spearman": corr,
            "effnet_minus_convnext_auc": (
                None
                if auc(y[:, j], rcx[:, j]) is None or auc(y[:, j], rev[:, j]) is None
                else auc(y[:, j], rev[:, j]) - auc(y[:, j], rcx[:, j])
            ),
            "effnet_minus_convnext_bootstrap_p05": percentile(target_delta_boot[target], 5),
            "effnet_minus_convnext_bootstrap_p95": percentile(target_delta_boot[target], 95),
            "bootstrap_probability_effnet_better": (
                float(np.mean(np.asarray(target_delta_boot[target]) > 0))
                if target_delta_boot[target] else None
            ),
        }

    blend_grid = {}
    for alpha in grid:
        key = f"{alpha:.2f}"
        score = macro(y, (1 - alpha) * rcx + alpha * rev)
        samples = blend_boot[key]
        blend_grid[key] = {
            "macro_auc": score,
            "bootstrap_p05": percentile(samples, 5),
            "bootstrap_p50": percentile(samples, 50),
            "bootstrap_p95": percentile(samples, 95),
        }

    group_report = {}
    for group, names in TARGET_GROUPS.items():
        group_report[group] = {
            "targets": names,
            "convnext_macro_auc": group_macro(y, rcx, names),
            "convnext_bootstrap_p05": percentile(group_boot["convnext"][group], 5),
            "convnext_bootstrap_p95": percentile(group_boot["convnext"][group], 95),
            "effnet_macro_auc": group_macro(y, rev, names),
            "effnet_bootstrap_p05": percentile(group_boot["effnet"][group], 5),
            "effnet_bootstrap_p95": percentile(group_boot["effnet"][group], 95),
        }

    report = {
        "schema": "rsna_knee_diversity_gold58_v2",
        "warning": "Gold58 is a noisy gate; do not select fine-grained weights from this table alone.",
        "studies": len(y),
        "models": {
            "convnext": {
                "macro_auc": macro(y, rcx),
                "bootstrap_p05": percentile(boot["convnext"], 5),
                "bootstrap_p95": percentile(boot["convnext"], 95),
            },
            "effnet": {
                "macro_auc": macro(y, rev),
                "bootstrap_p05": percentile(boot["effnet"], 5),
                "bootstrap_p95": percentile(boot["effnet"], 95),
            },
        },
        "per_target": per_target,
        "reported_public_difficulty_groups": group_report,
        "blend_grid_effnet_fraction": blend_grid,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
