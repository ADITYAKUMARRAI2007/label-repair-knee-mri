#!/usr/bin/env python3
"""Review completed held-out OOF uncertainty and broad target/fold failures.

No checkpoint, label, threshold or ensemble weight is fitted by this review.
Passing authorizes raw replay, not competition submission or a medal claim.
"""
import argparse
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

from bootstrap_gold58 import TARGETS, auc, grouped_scores, quantile

ROOT = Path(__file__).resolve().parents[1]
SEED, DRAWS = 20261007, 5000
# Broad failures require inspection. Small Gold folds cannot support precise
# target-level selection; these diagnostics are frozen before any E04 OOF exists.
DIAGNOSTICS = {"minimum_pooled_target_auc": .70, "minimum_fold_macro_auc": .80,
               "maximum_fold_macro_range": .15}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-audit", type=Path,
                        default=ROOT / "runs/E04_contract/training_audit.json")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/E04_contract/oof_review.json")
    args = parser.parse_args()
    audit = json.loads(args.training_audit.read_text())
    require(audit["status"] == "PASS_COMPLETE_OOF_SMALL_OUTPUTS" and audit["complete_expert_oof"]
            and sorted(r["fold"] for r in audit["folds"]) == list(range(5)), "complete OOF audit required")
    artifact = audit["expert_oof_artifact"]
    require(artifact["prediction_rule"] == "each study uses its independently heldout fold only"
            and artifact["studies"] == 58, "expert OOF scope changed")
    prediction = args.training_audit.parent / artifact["filename"]
    require(prediction.stat().st_size == artifact["bytes"] and sha(prediction) == artifact["sha256"],
            "audited expert prediction bytes changed")
    train = ROOT / "data/train.csv"
    require(sha(train) == "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33",
            "official expert truth table changed")
    frozen_path = ROOT / "runs/E04_contract/e04_contract.json"
    frozen = json.loads(frozen_path.read_text())
    by_uid = {r["StudyInstanceUID"]: r for r in frozen["rows"]}
    with train.open(newline="") as handle:
        expert_rows = [r for r in csv.DictReader(handle) if all(r[t].strip() for t in TARGETS)]
    require(len(expert_rows) == 58, "expert coverage changed")
    with prediction.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ["StudyInstanceUID", *TARGETS], "OOF target order changed")
        predicted = list(reader)
    ids = [r["StudyInstanceUID"] for r in expert_rows]
    require(all(by_uid[uid]["expert"] for uid in ids), "frozen expert identities changed")
    require([r["StudyInstanceUID"] for r in predicted] == ids, "expert OOF order changed")
    truth = [[int(float(r[t])) for r in expert_rows] for t in TARGETS]
    require(all(float(r[t]) in (0, 1) for r in expert_rows for t in TARGETS), "nonbinary expert truths")
    scores = [[float(r[t]) for r in predicted] for t in TARGETS]
    require(all(0 <= x <= 1 for col in scores for x in col), "invalid expert probabilities")
    groups = [grouped_scores(col) for col in scores]
    pooled = [auc(y, g, [1] * 58) for y, g in zip(truth, groups)]
    require(all(a is not None for a in pooled), "undefined pooled target AUC")
    require(all(abs(a - audit["expert_per_target_auc"][t]) < 1e-9 for t, a in zip(TARGETS, pooled)),
            "independent pooled AUC differs from training audit")
    folds = []
    for entry in sorted(audit["folds"], key=lambda r: r["fold"]):
        selected = [i for i, uid in enumerate(ids) if by_uid[uid]["fold"] == entry["fold"]]
        per_target = {t: auc([y[i] for i in selected], grouped_scores([s[i] for i in selected]),
                             [1] * len(selected)) for t, y, s in zip(TARGETS, truth, scores)}
        require(all(a is not None and abs(a - entry["expert_per_target_auc"][t]) < 1e-9
                    for t, a in per_target.items()), "fold-specific expert AUC differs")
        macro = statistics.mean(per_target.values())
        require(abs(macro - entry["expert_macro_auc"]) < 1e-9, "fold macro differs")
        folds.append({"fold": entry["fold"], "expert_rows": len(selected),
                      "macro_auc": macro, "per_target_auc": per_target})
    rng = random.Random(SEED)
    macros, target_draws, undefined = [], [[] for _ in TARGETS], [0] * len(TARGETS)
    for _ in range(DRAWS):
        counts = [0] * 58
        for _ in range(58):
            counts[rng.randrange(58)] += 1
        values = [auc(y, g, counts) for y, g in zip(truth, groups)]
        for j, value in enumerate(values):
            if value is None:
                undefined[j] += 1
            else:
                target_draws[j].append(value)
        if all(a is not None for a in values):
            macros.append(statistics.mean(values))
    require(len(macros) >= DRAWS // 2, "too few defined uncertainty draws")
    macro = statistics.mean(pooled)
    require(abs(macro - audit["expert_macro_auc"]) < 1e-9, "pooled macro differs")
    spread = max(r["macro_auc"] for r in folds) - min(r["macro_auc"] for r in folds)
    flags = [f"pooled_target_auc_below_0.70:{t}" for t, a in zip(TARGETS, pooled) if a < .70]
    flags += [f"fold_macro_auc_below_0.80:{r['fold']}" for r in folds if r["macro_auc"] < .80]
    if spread > .15:
        flags.append("fold_macro_range_above_0.15")
    status = ("REJECTED_VALIDATION" if macro < .90 else
              "REQUIRES_STABILITY_REVIEW" if flags else "PASS_FOR_RAW_REPLAY")
    result = {"schema": "e04_complete_oof_review_v1", "status": status,
              "training_audit_sha256": sha(args.training_audit), "prediction_sha256": sha(prediction),
              "truth_sha256": sha(train), "frozen_contract_sha256": sha(frozen_path),
              "review_source_sha256": sha(Path(__file__).resolve()), "diagnostic_rules": DIAGNOSTICS,
              "bootstrap_helper_sha256": sha(ROOT / "scripts/bootstrap_gold58.py"),
              "expert_rows": 58, "expert_macro_auc": macro, "folds": folds,
              "fold_macro_range": spread, "review_flags": flags,
              "bootstrap": {"unit": "study", "seed": SEED, "draws": DRAWS,
                            "draws_with_all_targets_defined": len(macros),
                            "macro_percentile_95_interval": [quantile(macros, .025), quantile(macros, .975)],
                            "per_target": {t: {"auc": a, "positive_studies": sum(y),
                                               "negative_studies": 58 - sum(y),
                                               "percentile_95_interval": [quantile(draws, .025), quantile(draws, .975)],
                                               "undefined_draws": undefined[j]}
                                           for j, (t, a, y, draws) in enumerate(zip(TARGETS, pooled, truth, target_draws))}},
              "selection_or_weight_fitting_performed": False, "competition_submission_enabled": False,
              "limitations": ["Gold58 has been reused across experiments and is a coarse rejection screen.",
                              "Unavailable patient/site identities prevent patient or site bootstrap claims.",
                              "Intervals condition on defined AUC and exclude undefined target draws.",
                              "This does not capture training randomness, report-label prompt tuning or hidden-test shift.",
                              "Passing does not establish leaderboard strength or guarantee absence of overfitting."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("bootstrap", "folds")}, indent=2))


if __name__ == "__main__":
    main()
