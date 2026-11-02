#!/usr/bin/env python3
"""Audit completed fold receipts and predictions; require all five for Gate B."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--experiment", choices=["e04", "e05_uniform", "e06_control", "e06_spatial"], default="e04")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-e06-control-fold2", action="store_true",
                        help="Include explicitly qualified E06 control with original provenance")
    args = parser.parse_args()
    prefix = args.experiment
    require(not args.reuse_e06_control_fold2 or prefix == "e05_uniform", "Reuse is only valid for E05 uniform")
    is_e06 = prefix.startswith("e06_")
    if args.output is None:
        args.output = ROOT / ("runs/E04_contract/training_audit.json" if prefix == "e04"
                              else f"runs/e06_spatial/{prefix}_audit.json" if is_e06
                              else "runs/e05_confidence_ablation/training_audit.json")
    schema = "rsna_knee_e05_confidence_fold_v1" if prefix == "e05_uniform" else "rsna_knee_e04_convnext_expert_fold_v1"
    if is_e06:
        schema = "rsna_knee_e06_global_spatial_fold_v1"
        e06_root = ROOT / "experiments/kaggle_e06_spatial"
        e06_build = json.loads((e06_root / "build.json").read_text())
        e06_protocol = json.loads((e06_root / "protocol.json").read_text())
        require(sha(e06_root / "protocol.json") == e06_build["protocol_sha256"], "E06 protocol source changed")
        for name in ("worker.py", "model.py", "supervisor.py", "notebook/experiment.py"):
            require(sha(e06_root / name) == e06_build["files"][name], "E06 built source changed")
        launch = json.loads((e06_root / "launch.json").read_text())
        require(launch["code_sha256"] == e06_build["files"]["notebook/experiment.py"], "E06 launch source changed")
    frozen = json.loads((ROOT / "runs/E04_contract/e04_contract.json").read_text())
    by_uid = {r["StudyInstanceUID"]: r for r in frozen["rows"]}
    targets = frozen["target_order"]
    with (ROOT / "data/train.csv").open(newline="") as f:
        official = {r["StudyInstanceUID"]: r for r in csv.DictReader(f)}
    cache_ids = np.concatenate([np.load(ROOT / f"runs/E04_wide_cache_{p}/e04_wide_{p}_ids.npy", allow_pickle=False)
                                for p in ("base", "extension")]).astype(str).tolist()
    expected_components = {str(p.relative_to(ROOT)): sha(p) for p in (
        ROOT / "public_kernels/e04_convnext_resource_pilot/rsna-knee-e04-convnext-resource-pilot.py",
        ROOT / "training/e04_wide_corpus.py", ROOT / "training/e04_convnext_fold_train.py")}
    seen, expert_predictions, results = set(), {}, []
    all_local_checkpoints = True
    for directory in args.directories:
        paths = sorted(directory.rglob(f"{prefix}_fold[0-4]_manifest.json"))
        require(bool(paths), f"no fold manifests in {directory}")
        for path in paths:
            manifest = json.loads(path.read_text())
            fold = manifest["fold"]
            require(fold in range(5) and fold not in seen, "missing/duplicate fold identity")
            seen.add(fold)
            require(manifest["schema"] == schema, "unexpected schema")
            require(manifest["status"] == "COMPLETE" and manifest["completed_epoch"] == 14,
                    "incomplete fold requires a completed-epoch resume, not scoring")
            contract = manifest["contract"]
            if prefix == "e05_uniform" or is_e06:
                if is_e06:
                    require(contract["arm"] == prefix.removeprefix("e06_")
                            and contract["protocol_sha256"] == e06_build["protocol_sha256"]
                            and fold == e06_protocol["fold"]
                            and contract["architecture"] == "convnext_small_global_spatial_residual"
                            and contract["residual_scale"] == e06_protocol["spatial_residual_scale"], "E06 arm/protocol changed")
                    effective = json.loads((path.parent / "effective_weights.json").read_text())
                    preflight = json.loads((path.parent / "model_preflight.json").read_text())
                    require(preflight["status"] == "PASS_CUDA_PARITY_MASK_AND_GRADIENT"
                            and preflight["arm"] == contract["arm"] and preflight["fold"] == fold
                            and preflight["optimizer_steps"] == 0
                            and preflight["global_path_max_abs_error"] <= 1e-5
                            and preflight["zero_residual_max_abs_error"] == 0
                            and preflight["missing_acquisition_max_abs_error"] <= 1e-6,
                            "E06 CUDA preflight failed")
                else:
                    protocol = ROOT / "runs/e05_confidence_ablation/protocol.json"
                    require(contract["arm"] == "uniform" and contract["protocol_sha256"] == sha(protocol), "E05 protocol changed")
                    effective = json.loads((path.parent / f"{prefix}_fold{fold}_effective_weights.json").read_text())
                require(contract["weak_confidence"] == "uniform_training_fold_mean", "weak weighting rule changed")
                import sys
                sys.path.insert(0, str(ROOT / "training"))
                from e05_confidence import uniform_weak_weights
                ordered = {r["StudyInstanceUID"]: i for i, r in enumerate(frozen["rows"])}
                indices = [ordered[u] for u in cache_ids]
                original = np.load(ROOT / "runs/E04_contract/e04_weights.npy")[indices]
                is_expert = np.array([by_uid[u]["expert"] for u in cache_ids])
                training_indices = np.array([i for i,u in enumerate(cache_ids) if by_uid[u]["fold"] != fold])
                expected_weights, expected_means = uniform_weak_weights(original, is_expert, training_indices)
                expected_hash = hashlib.sha256(expected_weights.tobytes()).hexdigest()
                if is_e06:
                    require(not effective["heldout_rows_used"]
                            and effective["weight_array_sha256"] == expected_hash
                            and effective["training_weak_means"] == expected_means.tolist(),
                            "E06 independently reconstructed effective weights differ")
                else:
                    require(effective["arm"] == "uniform" and effective["fold"] == fold
                            and not effective["heldout_rows_used_to_set_weights"]
                            and effective["effective_weight_array_sha256"] == expected_hash,
                            "E05 independently reconstructed effective weights differ")
            require(contract["worker_component_sha256"] == expected_components, "worker components changed")
            require(contract["fold"] == fold and contract["epochs"] == 14
                    and contract["checkpoint_rule"] == "final_epoch_14_ema_only"
                    and not contract["expert_checkpoint_selection"], "checkpoint selection rule changed")
            require(contract["fold_sha256"] == frozen["split_audit"]["fold_sha256"]
                    and contract["target_sha256"] == frozen["arrays"]["e04_targets.npy"]
                    and contract["weight_sha256"] == frozen["arrays"]["e04_weights.npy"], "frozen inputs changed")
            require(hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
                    == manifest["contract_sha256"], "training contract identity changed")
            require(not manifest["competition_submission_enabled"] and not manifest["checkpoint_selection_on_expert_labels"],
                    "fold output scope drift")
            rows = [u for u in cache_ids if by_uid[u]["fold"] == fold]
            training = [u for u in cache_ids if by_uid[u]["fold"] != fold]
            expert = [u for u in rows if by_uid[u]["expert"]]
            require([h["epoch"] for h in manifest["history"]] == list(range(1, 15)), "epoch coverage drift")
            batches = math.ceil(len(training) / 2)
            require(all(h["batches"] == batches and h["step"] == h["epoch"] * batches
                        and math.isfinite(h["loss"]) and math.isfinite(h["epoch_wall_seconds"])
                        and h["epoch_wall_seconds"] > 0 for h in manifest["history"]), "epoch numerical/step drift")
            require(manifest["amp_skipped_updates"] <= 10 and manifest["checkpoint_step"] == 14 * batches,
                    "checkpoint updates failed")
            require(set(manifest["artifacts"]) == {f"{prefix}_fold{fold}_{suffix}" for suffix in
                    ("last.pt", "split.json", "ema.pt", "oof_predictions.csv")}, "required fold artifacts missing")
            for name, receipt in manifest["artifacts"].items():
                p = path.parent / name
                if name.endswith(".pt") and not p.exists():
                    all_local_checkpoints = False
                    continue
                require(p.exists() and p.stat().st_size == receipt["bytes"] and sha(p) == receipt["sha256"],
                        f"artifact hash/size mismatch: {name}")
            split = json.loads((path.parent / f"{prefix}_fold{fold}_split.json").read_text())
            require(split["training_ids"] == training and split["validation_ids"] == rows
                    and split["expert_validation_ids"] == expert
                    and split["expert_training_ids"] == [u for u in training if by_uid[u]["expert"]]
                    and split["fold_audit"] == frozen["split_audit"], "independent split mismatch")
            require(not ({by_uid[u]["report_group"] for u in training}
                         & {by_uid[u]["report_group"] for u in rows}), "report group leakage")
            with (path.parent / f"{prefix}_fold{fold}_oof_predictions.csv").open(newline="") as f:
                reader = csv.DictReader(f)
                require(reader.fieldnames == ["StudyInstanceUID", *targets], "prediction target order changed")
                predictions = list(reader)
            require([r["StudyInstanceUID"] for r in predictions] == rows, "prediction study order changed")
            for r in predictions:
                require(all(math.isfinite(float(r[t])) and 0 <= float(r[t]) <= 1 for t in targets), "invalid probabilities")
                if by_uid[r["StudyInstanceUID"]]["expert"]:
                    expert_predictions[r["StudyInstanceUID"]] = r
            fold_aucs = {}
            for target in targets:
                fold_aucs[target] = float(roc_auc_score([int(float(official[u][target])) for u in expert],
                                                      [float(expert_predictions[u][target]) for u in expert]))
                require(abs(fold_aucs[target] - manifest["validation"]["expert_per_target_auc"][target]) < 1e-9,
                        "independent expert target AUC mismatch")
            macro = float(np.mean(list(fold_aucs.values())))
            require(abs(macro - manifest["validation"]["expert_macro_auc"]) < 1e-9
                    and not manifest["validation"]["selection_used"]
                    and manifest["validation"]["final_checkpoint_replay_max_abs_error"] <= 1e-6,
                    "validation/replay contract failed")
            results.append({"fold": fold, "expert_rows": len(expert), "expert_macro_auc": macro,
                            "expert_per_target_auc": fold_aucs, "manifest_sha256": sha(path),
                            "median_epoch_wall_seconds": float(np.median([h["epoch_wall_seconds"] for h in manifest["history"][1:]]))})
    reuse_provenance = None
    if args.reuse_e06_control_fold2:
        import subprocess
        import sys
        require(2 not in seen, "Fold2 already supplied; cannot duplicate reused fold")
        amendment_path = ROOT / "runs/e05_confidence_ablation/e06_control_reuse_amendment.json"
        amendment = json.loads(amendment_path.read_text())
        require(amendment["status"] == "QUALIFIED_REUSE_E06_CONTROL_AS_E05_UNIFORM_FOLD2", "Reuse not qualified")
        for name, expected in amendment["evidence_sha256"].items():
            require(sha(ROOT / name) == expected, "Reuse evidence changed")
        directory = ROOT / amendment["artifact_directory"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        separate_audit = args.output.parent / "e06_control_reuse_current_audit.json"
        subprocess.run([sys.executable, str(Path(__file__).resolve()), str(directory),
                        "--experiment", "e06_control", "--output", str(separate_audit)],
                       check=True, stdout=subprocess.DEVNULL, timeout=120)
        audited = json.loads(separate_audit.read_text())
        require(len(audited["folds"]) == 1 and audited["folds"][0]["fold"] == 2, "Reuse fold identity differs")
        with (directory / "e06_control_fold2_oof_predictions.csv").open(newline="") as handle:
            for row in csv.DictReader(handle):
                uid = row["StudyInstanceUID"]
                if by_uid[uid]["expert"]:
                    require(uid not in expert_predictions and by_uid[uid]["fold"] == 2, "Reused prediction overlap")
                    expert_predictions[uid] = row
        seen.add(2)
        results.append({**audited["folds"][0], "source_experiment": "E06 control", "qualified_reuse": True})
        all_local_checkpoints &= audited["checkpoint_bytes_independently_verified_locally"]
        reuse_provenance = {"amendment_sha256": sha(amendment_path),
                            "original_experiment": "E06 control", "fold": 2,
                            "final_acceptance_after_accuracy_known": True,
                            "limitations": amendment["limitations"]}
    complete_oof = seen == set(range(5))
    aucs, macro, expert_oof_artifact = None, None, None
    if complete_oof:
        experts = [u for u in official if by_uid[u]["expert"]]
        require(len(experts) == 58 and set(experts) == set(expert_predictions), "Gold58 OOF identity mismatch")
        aucs = {t: float(roc_auc_score([int(float(official[u][t])) for u in experts],
                                     [float(expert_predictions[u][t]) for u in experts])) for t in targets}
        macro = float(np.mean(list(aucs.values())))
        # Preserve the independently audited held-out predictions for uncertainty
        # and error-diversity analysis; never average expert-trained folds here.
        args.output.parent.mkdir(parents=True, exist_ok=True)
        expert_path = args.output.parent / f"{prefix}_gold58_heldout_oof_predictions.csv"
        with expert_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["StudyInstanceUID", *targets])
            writer.writeheader()
            writer.writerows(expert_predictions[u] for u in experts)
        expert_oof_artifact = {"filename": expert_path.name, "bytes": expert_path.stat().st_size,
                               "sha256": sha(expert_path), "studies": len(experts),
                               "prediction_rule": "each study uses its independently heldout fold only"}
    result = {"schema": f"{prefix}_independent_fold_training_audit_v1",
              "status": "PASS_COMPLETE_OOF_SMALL_OUTPUTS" if complete_oof else "PASS_PARTIAL_FOLDS_SMALL_OUTPUTS",
              "folds": sorted(results, key=lambda r: r["fold"]), "complete_expert_oof": complete_oof,
              "expert_oof_rows": len(expert_predictions), "expert_macro_auc": macro, "expert_per_target_auc": aucs,
              "expert_oof_artifact": expert_oof_artifact,
              "minimum_Gate_B_threshold_met": complete_oof and macro >= .90,
              "checkpoint_bytes_independently_verified_locally": all_local_checkpoints,
              "competition_submission_enabled": False, "public_score": None,
              "remaining_gates": ["checkpoint-byte integrity if not locally verified", "target/fold stability and uncertainty",
                                  "complete inference input parity and runtime", "official standalone strength before blending"],
              "limitations": frozen["limitations"]}
    if reuse_provenance:
        result["schema"] = "e05_uniform_qualified_reuse_fold_training_audit_v1"
        result["reuse_provenance"] = reuse_provenance
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
