#!/usr/bin/env python3
"""Build one private replay job from real completed, independently audited folds.

No push or competition submission occurs here. An incomplete fold, changed
receipt or failed validation review stops assembly before a notebook is written.
"""
import argparse
import ast
import hashlib
import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLUG = "adityakumarrai0401/rsna-knee-e04-final-raw-replay"
OUT = ROOT / "public_kernels/e04_final_raw_replay"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def verified_local(path, receipt):
    require(path.is_file() and path.stat().st_size == receipt["bytes"] and sha(path) == receipt["sha256"],
            f"changed local artifact: {path.name}")


def read_fold(output, expected, audit):
    # Format: fold=owned-kernel=directory downloaded with fetch_kernel_outputs.py.
    number, origin, directory = output.split("=", 2)
    fold, directory = int(number), Path(directory).resolve()
    require(fold in range(5), "invalid fold number")
    require(ROOT in directory.parents, "fold artifacts must be in the current workspace")
    require(origin.startswith("adityakumarrai0401/rsna-knee-e04-convnext-folds-"), "unowned fold input")
    download_path = directory / "download_receipt.json"
    download = json.loads(download_path.read_text())
    require(download["status"] == "COMPLETE" and download["kernel"] == origin
            and not download["job_mutations"], "download does not identify the owned original job")
    downloaded = {r["file"]: r for r in download["files"]}
    hits = list(directory.rglob(f"e04_fold{fold}_manifest.json"))
    require(len(hits) == 1, "expected exactly one selected final fold manifest")
    path = hits[0]
    require(str(path.relative_to(directory)) in downloaded, "fold manifest not in download receipt")
    verified_local(path, downloaded[str(path.relative_to(directory))])
    require(sha(path) == audit["manifest_sha256"], "fold manifest differs from independently audited input")
    manifest = json.loads(path.read_text())
    contract = manifest["contract"]
    require(manifest["status"] == "COMPLETE" and manifest["completed_epoch"] == 14
            and manifest["schema"] == "rsna_knee_e04_convnext_expert_fold_v1"
            and manifest["fold"] == fold and contract["fold"] == fold
            and contract["worker_component_sha256"] == expected
            and contract["checkpoint_rule"] == "final_epoch_14_ema_only"
            and not contract["expert_checkpoint_selection"], "final-fold recipe or coverage drift")
    require(hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            == manifest["contract_sha256"], "final-fold contract hash changed")
    checkpoint = f"e04_fold{fold}_ema.pt"
    state = manifest["artifacts"][checkpoint]
    require(state["bytes"] > 0 and len(state["sha256"]) == 64, "missing real final EMA byte receipt")
    result = {"fold": fold, "filename": checkpoint, **state, "contract": contract,
              "contract_sha256": manifest["contract_sha256"]}
    gates = [download_path, path]
    for kind, suffix in (("split", "split.json"), ("oof", "oof_predictions.csv")):
        name = f"e04_fold{fold}_{suffix}"
        local = path.parent / name
        recorded = manifest["artifacts"][name]
        relative = str(local.relative_to(directory))
        require(relative in downloaded and downloaded[relative]["sha256"] == recorded["sha256"]
                and downloaded[relative]["bytes"] == recorded["bytes"], "small artifact download identity differs")
        verified_local(local, recorded)
        result[kind] = {"filename": name, **recorded}
        gates.append(local)
    return result, origin, gates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-output", action="append", required=True,
                        help="fold=owned-kernel=downloaded-output-directory; give all five folds")
    parser.add_argument("--training-audit", type=Path, default=ROOT / "runs/E04_contract/training_audit.json")
    parser.add_argument("--oof-review", type=Path, default=ROOT / "runs/E04_contract/oof_review.json")
    args = parser.parse_args()
    require(not (ROOT / "runs/e04_final_raw_replay_launch.json").exists(),
            "a replay launch receipt exists; preserve and observe that original job")
    require(len(args.fold_output) == 5, "all five selected fold outputs are required")
    audit = json.loads(args.training_audit.read_text())
    require(audit["status"] == "PASS_COMPLETE_OOF_SMALL_OUTPUTS" and audit["complete_expert_oof"]
            and audit["minimum_Gate_B_threshold_met"] and audit["expert_macro_auc"] >= .90,
            "complete independently measured OOF acceptance is required")
    by_fold = {r["fold"]: r for r in audit["folds"]}
    require(len(audit["folds"]) == 5 and set(by_fold) == set(range(5)), "OOF fold identities changed")
    review = json.loads(args.oof_review.read_text())
    require(review["schema"] == "e04_complete_oof_review_v1" and review["status"] == "PASS_FOR_RAW_REPLAY"
            and not review["review_flags"] and not review["selection_or_weight_fitting_performed"]
            and review["training_audit_sha256"] == sha(args.training_audit)
            and review["prediction_sha256"] == audit["expert_oof_artifact"]["sha256"]
            and review["review_source_sha256"] == sha(ROOT / "scripts/review_e04_oof.py")
            and review["bootstrap_helper_sha256"] == sha(ROOT / "scripts/bootstrap_gold58.py"),
            "matching complete uncertainty and fold/target review is required")
    staged_path = ROOT / "runs/e04_raw_inference_source.json"
    staged = json.loads(staged_path.read_text())
    for relative, expected in staged["sources"].items():
        require(sha(ROOT / relative) == expected, "staged inference source changed")
    prepared = ROOT / "runs/E04_staged_raw_inference"
    modules = {}
    for name, expected in staged["assembled_files"].items():
        path = prepared / name
        require(sha(path) == expected, "assembled inference module changed")
        ast.parse(path.read_text())
        modules[name] = path.read_text()
    require(staged["frozen_worker_sha256"] == "082cd85baed7bf293983e86b572e7f42347a008f350756264c34f69346212a80",
            "different training worker requires a separate reviewed recipe")
    original = json.loads((ROOT / "runs/e04_folds_01_source.json").read_text())
    components = {name: digest for name, digest in original["sources"].items()
                  if name in ("public_kernels/e04_convnext_resource_pilot/rsna-knee-e04-convnext-resource-pilot.py",
                              "training/e04_wide_corpus.py", "training/e04_convnext_fold_train.py")}
    require(len(components) == 3, "missing executed training components")
    receipts, origins, gates = [], set(), [args.training_audit, args.oof_review, staged_path]
    for output in args.fold_output:
        fold = int(output.split("=", 1)[0])
        require(fold in by_fold, "invalid selected fold")
        receipt, origin, local_gates = read_fold(output, components, by_fold[fold])
        receipts.append(receipt)
        origins.add(origin)
        gates.extend(local_gates)
    require(sorted(r["fold"] for r in receipts) == list(range(5)), "duplicate or missing selected final fold")
    pixels = []
    for part in ("base", "extension"):
        directory = ROOT / f"runs/E04_wide_cache_{part}"
        manifest_path = directory / f"e04_wide_{part}_manifest.json"
        require(all(sha(manifest_path) == r["contract"]["wide_cache"][part]["manifest_sha256"]
                    for r in receipts), "pixel reference manifest differs from the executed training caches")
        manifest = json.loads(manifest_path.read_text())
        name = f"e04_wide_{part}_study_receipts.jsonl"
        recorded = manifest["artifacts"][name]
        verified_local(directory / name, recorded)
        pixels.append({"filename": name, **recorded})
        gates.extend((manifest_path, directory / name))
    receipts.sort(key=lambda r: r["fold"])
    entry = ("import json\nfrom pathlib import Path\n\n"
             "def main():\n"
             "    from e04_raw_replay import replay_raw_experts\n"
             f"    folds = json.loads({json.dumps(receipts)!r})\n"
             f"    pixels = json.loads({json.dumps(pixels)!r})\n"
             "    replay_raw_experts(folds, pixels, Path('/kaggle/working'))\n\n"
             "if __name__ == '__main__':\n    main()\n")
    modules["entry.py"] = entry
    checksums = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in modules.items()}
    wrapper = ("import hashlib\nimport json\nimport subprocess\nimport sys\nfrom pathlib import Path\n\n"
               f"MODULES = {modules!r}\nCHECKSUMS = {checksums!r}\n\n"
               "def main():\n"
               "    directory = Path('/kaggle/working/_e04_runtime')\n"
               "    directory.mkdir(parents=True, exist_ok=True)\n"
               "    for name, code in MODULES.items():\n"
               "        if hashlib.sha256(code.encode()).hexdigest() != CHECKSUMS[name]:\n"
               "            raise RuntimeError('embedded replay source identity changed')\n"
               "        (directory / name).write_text(code)\n"
               "    subprocess.run([sys.executable, str(directory / 'entry.py')], check=True, timeout=2700)\n\n"
               "if __name__ == '__main__':\n    main()\n")
    ast.parse(wrapper)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rsna-knee-e04-final-raw-replay.py"
    path.write_text(wrapper)
    py_compile.compile(str(path), doraise=True)
    metadata = {"id": SLUG, "title": "RSNA Knee E04 Final Raw Replay", "code_file": path.name,
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": ["adityakumarrai0401/rsna-knee-e04-convnext-imagenet-weights"],
                "competition_sources": ["rsna-knee-abnormality-detection"],
                "kernel_sources": sorted(origins) + ["adityakumarrai0401/rsna-knee-e04-wide-cache-base",
                                                     "adityakumarrai0401/rsna-knee-e04-wide-cache-extension"],
                "model_sources": []}
    meta_path = OUT / "kernel-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    result = {"schema": "e04_final_raw_replay_job_build_v1", "status": "BUILT_NOT_PUSHED",
              "kernel": SLUG, "syntax": "PASS", "fold_receipts": receipts,
              "runtime_cap_hours": .75, "startup_finalization_margin_hours": .5,
              "fresh_quota_check_required_before_push": True, "competition_submission_enabled": False,
              "sources": {str(p.relative_to(ROOT)): sha(p) for p in (path, meta_path, Path(__file__).resolve())},
              "module_sha256": checksums,
              "gate_receipts": {str(p.resolve().relative_to(ROOT)): sha(p) for p in gates}}
    (ROOT / "runs/e04_final_raw_replay_source.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "fold_receipts"}, indent=2))


if __name__ == "__main__":
    main()
