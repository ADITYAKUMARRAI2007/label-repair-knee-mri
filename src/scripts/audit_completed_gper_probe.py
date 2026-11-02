"""Independently replay completed predictions; never train or select blend weights."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/resume_20261009"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    rows = {r["StudyInstanceUID"]: r for r in csv.DictReader((ROOT / "data/train.csv").open())}
    split = json.loads((ROOT / "runs/E03_contract/split.json").read_text())
    features = ROOT / "runs/grade_supervision_probe_outputs_v2/grade_probe"
    all_ids = np.load(features / "study_ids.npy").astype(str)
    audits = []
    for version in [2, 3]:
        run = ROOT / f"runs/gper_pseudolabel_probe_v{version}"
        contract = json.loads((run / "contract.json").read_text())
        result = json.loads((run / "result.json").read_text())
        missing_inputs = []
        for relative, expected in contract["inputs"].items():
            if not (ROOT / relative).is_file():
                missing_inputs.append(relative)
                continue
            assert sha(ROOT / relative) == expected, relative
        assert all(p.endswith('/features.npy') for p in missing_inputs)
        ids = np.load(run / "expert_ids.npy").astype(str)
        train = np.load(run / "train_indices.npy")
        assert len(ids) == len(set(ids)) == 58 and set(ids) == set(split["gold"])
        assert len(train) == len(set(train)) == 4343
        assert not set(all_ids[train]) & (set(ids) | set(split["excluded_gold_group_weak"]))
        pseudo_path = next(ROOT / p for p in contract["inputs"] if p.endswith("pseudolabeled-train.csv"))
        pseudo = {r["StudyInstanceUID"]: r for r in csv.DictReader(pseudo_path.open())}
        assert all(pseudo[uid]["label_source"] == "pseudo" for uid in all_ids[train])
        truth = np.asarray([[float(rows[uid][t]) for t in TARGETS] for uid in ids])
        scores, hashes = {}, {}
        for arm in ["control", "pseudo"]:
            scores[arm] = []
            for seed in contract["seeds"]:
                path = run / f"{arm}_{seed}_expert.npy"
                pred = np.load(path)
                assert pred.shape == truth.shape and np.isfinite(pred).all()
                assert ((pred >= 0) & (pred <= 1)).all()
                scores[arm].append(roc_auc_score(truth, pred, average=None).tolist())
                hashes[path.name] = sha(path)
            np.testing.assert_allclose(scores[arm], result["scores"][arm], rtol=0, atol=1e-14)
        control, candidate = (np.asarray(scores[k]) for k in ["control", "pseudo"])
        audits.append({"run": str(run.relative_to(ROOT)), "available_inputs_match": True,
                       "missing_inputs_not_reverified": missing_inputs,
                       "train_rows": len(train), "evaluation_rows": len(ids),
                       "control_macro": float(control.mean()), "candidate_macro": float(candidate.mean()),
                       "mean_delta": float((candidate-control).mean()),
                       "seed_deltas": (candidate-control).mean(1).tolist(),
                       "prediction_hashes": hashes, "reported_result_replayed": True})
    assert audits[0]["prediction_hashes"] == audits[1]["prediction_hashes"]
    report = {"status": "COMPLETED_REJECT_FOR_ADVANCEMENT", "audits": audits,
              "duplicate_runs_byte_identical": True,
              "decision": "Reject this label replacement on the tested frozen-feature head; no blend or submission.",
              "limitations": "Gold58 has been repeatedly inspected. This is diagnostic evidence, not an independent estimate of hidden performance. Saved predictions and contracts do not prove every training operation; no training checkpoints were saved.",
              "correction": "Prior interruption report was false: both 30-epoch six-fit jobs produced final results. Mere presence of expert labels in a table is not proof they entered training. The recorded and replayed training indices exclude all 58 expert and four report-linked studies.",
              "previous_goal_turn_classification": "progress: completed experiment artifacts provide new evidence; its subsequent status report was inaccurate",
              "competition_submission_enabled": False}
    (OUT / "gper_completed_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "audits"}, indent=2))
    print(json.dumps({k: v for k, v in audits[1].items() if k != "prediction_hashes"}, indent=2))


if __name__ == "__main__":
    main()
