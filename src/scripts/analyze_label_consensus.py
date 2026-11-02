#!/usr/bin/env python3
"""Audit independent report-label families and conservative consensus choices.

The 58 expert studies are used as a coarse diagnostic only. The script reports
equal-family ensembles and uncertainty; it deliberately does not fit per-target
weights or thresholds on this small set.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
SOURCES = {
    "steven_v4": "steven/llm_labels_v4_blend.csv",
    "pilkwang_v2": "pilkwang/report_labels_v2.csv",
    "gpt56": "gpt56/labels_llm_gpt56sol.csv",
    "dread_soft": "dread/labels_llm_soft.csv",
    "yunus_v5": "yunus/report_labels_v5.csv",
}


def rank01(values: np.ndarray) -> np.ndarray:
    return pd.DataFrame(values).rank(method="average", pct=True).to_numpy(np.float64)


def aucs(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    result = {}
    for index, target in enumerate(TARGETS):
        labels = truth[:, index]
        valid = np.isfinite(labels) & np.isfinite(prediction[:, index])
        if valid.sum() < 3 or np.unique(labels[valid]).size != 2:
            result[target] = float("nan")
        else:
            result[target] = float(roc_auc_score(labels[valid], prediction[valid, index]))
    return result


def macro(score: dict[str, float]) -> float:
    values = np.asarray(list(score.values()), np.float64)
    return float(np.nanmean(values))


def percentile(values: list[float], q: float) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], np.float64)
    return float(np.percentile(finite, q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--label-root", type=Path, default=Path("data/labels"))
    parser.add_argument("--output", type=Path, default=Path("runs/label_consensus_audit.json"))
    parser.add_argument("--table", type=Path, default=Path("runs/label_consensus_target_auc.csv"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260930)
    args = parser.parse_args()

    train = pd.read_csv(args.train, dtype={"StudyInstanceUID": str})
    if len(train) != 4407 or train.StudyInstanceUID.duplicated().any():
        raise RuntimeError(f"training identity contract changed: {len(train)} rows")
    gold_mask = train[TARGETS].notna().all(axis=1).to_numpy()
    if int(gold_mask.sum()) != 58:
        raise RuntimeError(f"expert-label contract changed: {gold_mask.sum()} rows")
    ids = train.StudyInstanceUID.astype(str).tolist()
    truth = train.loc[gold_mask, TARGETS].to_numpy(np.float64)

    values = {}
    coverage = {}
    for name, relative in SOURCES.items():
        path = args.label_root / relative
        frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
        missing = [column for column in ["StudyInstanceUID", *TARGETS] if column not in frame]
        if missing:
            raise RuntimeError(f"{path} is missing {missing}")
        frame = frame[["StudyInstanceUID", *TARGETS]].copy()
        source_ids = set(frame.StudyInstanceUID)
        if frame.StudyInstanceUID.duplicated().any() or not source_ids.issubset(set(ids)):
            raise RuntimeError(f"{path} contains duplicate or noncompetition identities")
        if len(source_ids) < 4349:
            raise RuntimeError(f"{path} has insufficient study coverage: {len(source_ids)}")
        frame = frame.set_index("StudyInstanceUID").reindex(ids)
        matrix = frame[TARGETS].apply(pd.to_numeric, errors="coerce").to_numpy(np.float64)
        finite = np.isfinite(matrix)
        if (matrix[finite] < 0).any() or (matrix[finite] > 1).any():
            raise RuntimeError(f"{path} contains invalid probabilities")
        gold_rows = matrix[gold_mask]
        gold_finite_rows = np.isfinite(gold_rows).all(axis=1)
        gold_truth_overwrite = bool(
            gold_finite_rows.any()
            and np.array_equal(gold_rows[gold_finite_rows], truth[gold_finite_rows])
        )
        values[name] = matrix
        coverage[name] = {
            "studies": int(np.isfinite(matrix).all(axis=1).sum()),
            "gold_studies": int(gold_finite_rows.sum()),
            "non_gold_studies": int(np.isfinite(matrix[~gold_mask]).all(axis=1).sum()),
            "gold_truth_overwrite": gold_truth_overwrite,
        }

    ranked = {name: rank01(matrix) for name, matrix in values.items()}
    evaluation_sources = [
        name for name in SOURCES
        if coverage[name]["gold_studies"] >= 57 and not coverage[name]["gold_truth_overwrite"]
    ]
    if len(evaluation_sources) < 3:
        raise RuntimeError(f"too few nonleaking Gold58 label sources: {evaluation_sources}")
    candidates = {name: ranked[name] for name in evaluation_sources}
    candidates[f"equal_rank_mean_{len(evaluation_sources)}"] = np.nanmean(
        [ranked[name] for name in evaluation_sources], axis=0
    )
    candidates[f"raw_probability_median_{len(evaluation_sources)}"] = np.nanmedian(
        [values[name] for name in evaluation_sources], axis=0
    )
    for omitted in evaluation_sources:
        candidates[f"rank_mean_without_{omitted}"] = np.nanmean(
            [ranked[name] for name in evaluation_sources if name != omitted], axis=0
        )

    scores = {name: aucs(truth, matrix[gold_mask]) for name, matrix in candidates.items()}
    macros = {name: macro(score) for name, score in scores.items()}
    best_source = max(evaluation_sources, key=lambda name: macros[name])
    reference = f"equal_rank_mean_{len(evaluation_sources)}"

    rng = np.random.default_rng(args.seed)
    boot = {name: [] for name in candidates}
    delta = []
    for _ in range(args.bootstrap):
        take = rng.integers(0, len(truth), len(truth))
        y = truth[take]
        for name, matrix in candidates.items():
            boot[name].append(macro(aucs(y, matrix[gold_mask][take])))
        delta.append(boot[reference][-1] - boot[best_source][-1])

    non_gold = ~gold_mask
    correlation = {}
    for left, right in itertools.combinations(SOURCES, 2):
        per_target = {}
        for index, target in enumerate(TARGETS):
            per_target[target] = float(pd.Series(values[left][non_gold, index]).corr(
                pd.Series(values[right][non_gold, index]), method="spearman"
            ))
        correlation[f"{left}__{right}"] = {
            "mean_target_spearman": float(np.mean(list(per_target.values()))),
            "per_target_spearman": per_target,
        }

    rayan_audit_path = args.label_root / "rayan/target_quality_audit.json"
    rayan = json.loads(rayan_audit_path.read_text())
    rayan_reported = {
        "macro_auc": float(rayan["gold_macro_auc_soft"]),
        "per_target_auc": {
            target: float(rayan["per_target"][target]["gold_auc_soft"])
            for target in TARGETS
        },
        "provenance": str(rayan_audit_path),
        "note": "Provider-reported pre-overwrite Gold58 audit; train_targets_soft.csv itself contains gold truth.",
    }

    candidate_report = {}
    for name in candidates:
        candidate_report[name] = {
            "macro_auc": macros[name],
            "per_target_auc": scores[name],
            "bootstrap_p05": percentile(boot[name], 5),
            "bootstrap_p50": percentile(boot[name], 50),
            "bootstrap_p95": percentile(boot[name], 95),
        }
    report = {
        "schema": "rsna_knee_label_consensus_audit_v1",
        "status": "PASS",
        "studies": len(train),
        "gold_studies": int(gold_mask.sum()),
        "non_gold_studies": int(non_gold.sum()),
        "sources": {
            name: {"path": str(args.label_root / path), **coverage[name]}
            for name, path in SOURCES.items()
        },
        "gold_evaluation_sources": evaluation_sources,
        "best_individual_source_on_gold58": best_source,
        "candidates": candidate_report,
        "equal_rank_mean_minus_best_source": {
            "observed": macros[reference] - macros[best_source],
            "bootstrap_p05": percentile(delta, 5),
            "bootstrap_p50": percentile(delta, 50),
            "bootstrap_p95": percentile(delta, 95),
            "bootstrap_probability_positive": float(np.mean(np.asarray(delta) > 0)),
        },
        "rayan_calibrated_soft_reported": rayan_reported,
        "non_gold_pairwise_correlations": correlation,
        "decision_rule": (
            "Use family-balanced soft consensus and confidence weighting. Gold58 is too small "
            "to justify fitted per-target source weights; treat the reported intervals as a coarse gate."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    rows = []
    for target in TARGETS:
        row = {"target": target}
        for name in candidates:
            row[name] = scores[name][target]
        row["rayan_calibrated_soft_reported"] = rayan_reported["per_target_auc"][target]
        rows.append(row)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.table, index=False)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
