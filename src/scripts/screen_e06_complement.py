"""Evaluate the precommitted E06 mixtures on the same reused expert fold.

Runs the independent fold auditor first. Produces a descriptive diagnostic only;
never selects weights, launches training, or submits predictions.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/e06_spatial"
PROTOCOL_SHA256 = "675050a413c032c5b4aee7c6e522f2c3a8cb5469dc7ee9b5f1629fd4f43fa202"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def select_predictions(path, ids, labels):
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    require(frame.columns.tolist() == ["StudyInstanceUID", *labels], "target order mismatch")
    require(frame.StudyInstanceUID.is_unique, "duplicate prediction identity")
    require(set(ids) <= set(frame.StudyInstanceUID), "missing held-out prediction")
    values = frame.set_index("StudyInstanceUID").loc[ids, labels].to_numpy(float)
    require(np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "invalid probabilities")
    return values


def summarize(truth, baseline, candidate, labels, weight):
    b = pd.DataFrame(baseline).rank(method="average", pct=True).to_numpy()
    c = pd.DataFrame(candidate).rank(method="average", pct=True).to_numpy()
    mix = (1 - weight) * b + weight * c
    ab = np.atleast_1d(roc_auc_score(truth, b, average=None))
    ac = np.atleast_1d(roc_auc_score(truth, c, average=None))
    am = np.atleast_1d(roc_auc_score(truth, mix, average=None))
    details = {}
    for i, target in enumerate(labels):
        corr = None
        if b[:, i].std() > 0 and c[:, i].std() > 0:
            corr = float(np.corrcoef(b[:, i], c[:, i])[0, 1])
        details[target] = {
            "baseline_auc": float(ab[i]), "candidate_auc": float(ac[i]),
            "fixed_mix_auc": float(am[i]), "mix_delta": float(am[i] - ab[i]),
            "positives": int(truth[:, i].sum()),
            "negatives": int(len(truth) - truth[:, i].sum()), "rank_correlation": corr,
        }
    return {"baseline_auc": float(ab.mean()), "candidate_auc": float(ac.mean()),
            "fixed_mix_auc": float(am.mean()), "mix_delta": float((am - ab).mean()),
            "per_target": details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUN / "best_blend_screen.json")
    args = parser.parse_args()
    protocol_path = RUN / "best_blend_screen_protocol.json"
    require(sha(protocol_path) == PROTOCOL_SHA256, "frozen blend protocol changed")
    protocol = json.loads(protocol_path.read_text())
    require(sha(ROOT / "experiments/kaggle_e06_spatial/protocol.json") ==
            protocol["e06_protocol_sha256"], "training protocol changed")
    baseline_path = ROOT / protocol["baseline_path"]
    require(sha(baseline_path) == protocol["baseline_sha256"], "baseline predictions changed")
    frozen = json.loads((ROOT / "runs/E04_contract/e04_contract.json").read_text())
    ids, labels = protocol["cohort_ids"], protocol["target_order"]
    require(ids == [r["StudyInstanceUID"] for r in frozen["rows"] if r["expert"] and r["fold"] == 2]
            and len(ids) == 12 and labels == frozen["target_order"], "held-out expert cohort changed")
    arms = protocol["candidate_arms"]
    require(arms == ["control", "spatial"] and protocol["candidate_weight"] == .10,
            "fixed diagnostic changed")
    result_path = args.job_output / "e06_result.json"
    require(result_path.is_file(), "E06 result is not available; wait for the existing job")
    job = json.loads(result_path.read_text())
    require(job["status"] == "COMPLETE", "incomplete E06 job cannot be scored")
    audits, preflights, splits, weights, predictions, manifests = {}, {}, {}, {}, {}, {}
    for arm in arms:
        directory = args.job_output / arm
        prefix = f"e06_{arm}_fold2"
        audit_path = args.output.parent / f"e06_{arm}_audit.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/audit_e04_fold_training.py"),
                        "--experiment", f"e06_{arm}", str(directory), "--output", str(audit_path)],
                       check=True, stdout=subprocess.DEVNULL)
        audits[arm] = json.loads(audit_path.read_text())
        require(len(audits[arm]["folds"]) == 1 and audits[arm]["folds"][0]["fold"] == 2,
                "unexpected audited folds")
        manifests[arm] = json.loads((directory / f"{prefix}_manifest.json").read_text())
        preflights[arm] = json.loads((directory / "model_preflight.json").read_text())
        splits[arm] = json.loads((directory / f"{prefix}_split.json").read_text())
        weights[arm] = json.loads((directory / "effective_weights.json").read_text())
        predictions[arm] = select_predictions(directory / f"{prefix}_oof_predictions.csv", ids, labels)
    require(splits["control"] == splits["spatial"] and weights["control"] == weights["spatial"],
            "paired split or supervision mismatch")
    require(preflights["control"]["shared_initial_state_sha256"] ==
            preflights["spatial"]["shared_initial_state_sha256"], "paired initialization mismatch")
    baseline = select_predictions(baseline_path, ids, labels)
    official = pd.read_csv(ROOT / "data/train.csv", dtype={"StudyInstanceUID": str})
    truth = official.set_index("StudyInstanceUID").loc[ids, labels].to_numpy()
    require(np.isin(truth, [0, 1]).all(), "invalid expert labels")
    summaries = {arm: summarize(truth, baseline, predictions[arm], labels, .10) for arm in arms}
    for arm in arms:
        actual = summaries[arm]["candidate_auc"]
        require(abs(actual - audits[arm]["folds"][0]["expert_macro_auc"]) < 1e-9 and
                abs(actual - job["comparison"][f"{arm}_macro_auc"]) < 1e-9,
                "independent comparison disagrees with training result")
    result = {"status": "DESCRIPTIVE_SINGLE_FOLD_COMPARISON", "rows": len(ids),
              "arms": summaries, "protocol_sha256": PROTOCOL_SHA256,
              "prediction_sha256": {arm: manifests[arm]["artifacts"][f"e06_{arm}_fold2_oof_predictions.csv"]["sha256"] for arm in arms},
              "weight_search_performed": False, "competition_submission_enabled": False,
              "public_score_prediction": None, "decision": "REVIEW_WITH_FULLER_HELD_OUT_EVIDENCE",
              "checkpoint_integrity_still_required_if_not_separately_verified": True,
              "limitations": protocol["limitations"] + [protocol["ranking"]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "arms": {
        arm: {k: v for k, v in summaries[arm].items() if k != "per_target"} for arm in arms}}, indent=2))


if __name__ == "__main__":
    main()
