"""Fixed CPU rejection probe for Gold-redacted Gperdrizet pseudolabels.

The public table's labelled Gold58 rows are never loaded into the training
arrays.  Evaluation happens only after every predeclared fit completes.
"""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "training"))
from spatial_pooling_head import SpatialHead

OUT = ROOT / "runs/gper_pseudolabel_probe_v3"
FEATURES = ROOT / "runs/grade_supervision_probe_outputs_v2/grade_probe"
PSEUDO = ROOT / "research/new_public/label_candidates_20261009/gperdrizet__rsna-knee-abnormality-pseudolabeled-training-data/data/pseudolabeled-train.csv"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SEEDS = [20261008, 20261009, 20261010]


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def table(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["StudyInstanceUID"]: row for row in rows}
    assert len(rows) == len(result)
    return result


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def auc_binary(truth: np.ndarray, score: np.ndarray) -> float:
    """Tie-aware binary AUC without scikit-learn or scipy."""
    positive = score[truth == 1]
    negative = score[truth == 0]
    if not len(positive) or not len(negative):
        raise ValueError("AUC requires both classes")
    return float(((positive[:, None] > negative[None, :]).mean()) + .5 * (positive[:, None] == negative[None, :]).mean())


def main() -> None:
    if OUT.exists():
        raise RuntimeError("preserve frozen prior probe")
    OUT.mkdir(parents=True)
    started = time.monotonic()
    contract = {
        "schema": "gper_pseudolabel_probe_v1",
        "seeds": SEEDS,
        "epochs": 30,
        "batch_size": 64,
        "device": "cpu",
        "comparison": "same cohort: existing four-source control targets versus Gper pseudo-only targets",
        "gper_exclusion": "all rows with label_source=gold and all report-linked weak rows are excluded",
        "checkpoint_rule": "final epoch only",
        "gates": {"candidate_macro_min": 0.90, "gain_min": 0.003, "positive_seeds_min": 2},
        "competition_submission_enabled": False,
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in [PSEUDO, ROOT / "data/train.csv", ROOT / "runs/E03_contract/split.json", ROOT / "runs/grade_supervision_trial_v1/control_targets.csv", ROOT / "training/spatial_pooling_head.py", FEATURES / "features.npy", FEATURES / "token_masks.npy", FEATURES / "study_ids.npy"]},
    }
    write("contract.json", contract)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    features = np.load(FEATURES / "features.npy", mmap_mode="r")
    masks = np.load(FEATURES / "token_masks.npy")
    ids = np.load(FEATURES / "study_ids.npy").astype(str)
    assert features.shape == (4407, 20, 2048) and masks.shape == (4407, 20)
    split = json.loads((ROOT / "runs/E03_contract/split.json").read_text())
    gold = set(split["gold"])
    excluded = gold | set(split["excluded_gold_group_weak"])
    pseudo = table(PSEUDO)
    assert sum(row.get("label_source") == "gold" for row in pseudo.values()) == 58
    pseudo_train_ids = {uid for uid, row in pseudo.items() if row.get("label_source") == "pseudo"}
    train = np.asarray([i for i, uid in enumerate(ids) if uid in pseudo_train_ids and uid not in excluded], dtype=np.int64)
    expert = np.asarray([i for i, uid in enumerate(ids) if uid in gold], dtype=np.int64)
    assert not (set(ids[train]) & gold) and len(expert) == 58 and len(train) > 4000
    control_rows = table(ROOT / "runs/grade_supervision_trial_v1/control_targets.csv")
    y_control = np.zeros((len(ids), len(TARGETS)), dtype=np.float32)
    y_pseudo = np.zeros_like(y_control)
    for i in train:
        y_control[i] = [float(control_rows[ids[i]][t]) for t in TARGETS]
        row = pseudo[ids[i]]
        y_pseudo[i] = [float(row[t]) for t in TARGETS]
    assert np.isfinite(y_pseudo[train]).all() and np.isin(y_pseudo[train], [0, 1]).all()
    np.save(OUT / "train_indices.npy", train)
    np.save(OUT / "expert_ids.npy", ids[expert])

    def predict(model: SpatialHead) -> np.ndarray:
        model.eval(); blocks = []
        with torch.inference_mode():
            for start in range(0, len(expert), 16):
                b = expert[start:start + 16]
                blocks.append(model(torch.from_numpy(np.asarray(features[b])), torch.from_numpy(masks[b])).sigmoid().numpy())
        return np.concatenate(blocks)

    predictions: dict[tuple[int, str], np.ndarray] = {}
    for seed in SEEDS:
        initial = []
        for arm, labels in [("control", y_control), ("pseudo", y_pseudo)]:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = SpatialHead()
            initial.append(hashlib.sha256(b"".join(v.numpy().tobytes() for v in model.state_dict().values())).hexdigest())
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
            for epoch in range(30):
                order = np.random.default_rng(seed + epoch).permutation(train)
                model.train()
                for start in range(0, len(order), 64):
                    if time.monotonic() - started > 1200:
                        raise TimeoutError("fixed CPU cap exceeded")
                    b = order[start:start + 64]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(torch.from_numpy(np.asarray(features[b])), torch.from_numpy(masks[b]))
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.from_numpy(labels[b]))
                    assert torch.isfinite(loss)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 3); optimizer.step()
                if (epoch + 1) % 5 == 0:
                    print(json.dumps({"seed": seed, "arm": arm, "epoch": epoch + 1, "elapsed_seconds": time.monotonic() - started}), flush=True)
            prediction = predict(model); assert np.isfinite(prediction).all()
            predictions[seed, arm] = prediction
            np.save(OUT / f"{arm}_{seed}_expert.npy", prediction)
        assert len(set(initial)) == 1
    # Access the expert truth only after all six final-epoch fits are complete.
    official = table(ROOT / "data/train.csv")
    truth = np.asarray([[float(official[uid][t]) for t in TARGETS] for uid in ids[expert]])
    score = {arm: np.asarray([[auc_binary(truth[:, j], predictions[seed, arm][:, j]) for j in range(12)] for seed in SEEDS]) for arm in ["control", "pseudo"]}
    delta = score["pseudo"] - score["control"]
    result = {
        "status": "PASS_REQUIRES_INDEPENDENT_AUDIT" if float(score["pseudo"].mean()) >= .90 and float(delta.mean()) >= .003 and int((delta.mean(1) > 0).sum()) >= 2 else "REJECT_FOR_ADVANCEMENT",
        "train_rows": int(len(train)), "expert_rows": 58, "gold_rows_loaded_for_training": 0,
        "control_macro": float(score["control"].mean()), "pseudo_macro": float(score["pseudo"].mean()), "mean_delta": float(delta.mean()), "seed_deltas": delta.mean(1).tolist(),
        "per_target_delta": dict(zip(TARGETS, delta.mean(0).tolist())), "scores": {k: v.tolist() for k, v in score.items()},
        "elapsed_seconds": time.monotonic() - started, "competition_submission_enabled": False, "public_score": None,
    }
    write("result.json", result)
    print(json.dumps(result), flush=True)


def fast_rejection(seed: int, arm: str) -> None:
    """One five-epoch arm for a bounded rejection-only screen."""
    if arm not in {"control", "pseudo"}:
        raise ValueError("arm")
    out = ROOT / "runs/gper_pseudolabel_fast_v1"
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    features = np.load(FEATURES / "features.npy", mmap_mode="r")
    masks = np.load(FEATURES / "token_masks.npy")
    ids = np.load(FEATURES / "study_ids.npy").astype(str)
    split = json.loads((ROOT / "runs/E03_contract/split.json").read_text())
    gold = set(split["gold"]); excluded = gold | set(split["excluded_gold_group_weak"])
    pseudo = table(PSEUDO)
    pseudo_ids = {u for u,r in pseudo.items() if r.get("label_source") == "pseudo"}
    train = np.asarray([i for i,u in enumerate(ids) if u in pseudo_ids and u not in excluded], dtype=np.int64)
    expert = np.asarray([i for i,u in enumerate(ids) if u in gold], dtype=np.int64)
    control = table(ROOT / "runs/grade_supervision_trial_v1/control_targets.csv")
    labels = np.zeros((len(ids),12),np.float32)
    for i in train:
        row = pseudo[ids[i]] if arm == "pseudo" else control[ids[i]]
        labels[i] = [float(row[t]) for t in TARGETS]
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    model=SpatialHead();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    for epoch in range(5):
        order=np.random.default_rng(seed+epoch).permutation(train);model.train()
        for start in range(0,len(order),64):
            b=order[start:start+64];opt.zero_grad(set_to_none=True)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(model(torch.from_numpy(np.asarray(features[b])),torch.from_numpy(masks[b])),torch.from_numpy(labels[b]))
            if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),3);opt.step()
    model.eval();blocks=[]
    with torch.inference_mode():
        for start in range(0,len(expert),16):
            b=expert[start:start+16];blocks.append(model(torch.from_numpy(np.asarray(features[b])),torch.from_numpy(masks[b])).sigmoid().numpy())
    pred=np.concatenate(blocks);assert np.isfinite(pred).all()
    np.save(out/f"{arm}_{seed}_expert.npy",pred)
    receipt={"arm":arm,"seed":seed,"epochs":5,"train_rows":int(len(train)),"gold_rows_loaded_for_training":0,"elapsed_seconds":time.monotonic()-started,"prediction_sha256":sha(out/f"{arm}_{seed}_expert.npy"),"rejection_only":True,"competition_submission_enabled":False}
    write_path=out/f"{arm}_{seed}_receipt.json";write_path.write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--fast-rejection",action="store_true")
    parser.add_argument("--seed",type=int)
    parser.add_argument("--arm",choices=["control","pseudo"])
    args=parser.parse_args()
    if args.fast_rejection:
        if args.seed is None or args.arm is None: parser.error("--fast-rejection requires --seed and --arm")
        fast_rejection(args.seed,args.arm)
    else:
        main()
