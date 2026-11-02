#!/usr/bin/env python3
"""Build the five-fold prediction notebook only after the exact raw replay passes.

Uses real fold receipts from the gated replay build. No notebook push or
competition submission is performed by this command.
"""
import argparse
import ast
import hashlib
import json
import py_compile
from pathlib import Path

from build_e04_raw_replay_job import require, sha, verified_local

ROOT = Path(__file__).resolve().parents[1]
SLUG = "adityakumarrai0401/rsna-knee-e04-final-predict"
OUT = ROOT / "public_kernels/e04_final_predict"


def verified_replay(directory, build, staged):
    require(ROOT in directory.parents, "replay outputs must be in the current workspace")
    download_path = directory / "download_receipt.json"
    download = json.loads(download_path.read_text())
    require(download["status"] == "COMPLETE" and not download["job_mutations"]
            and download["kernel"] == build["kernel"], "original owned replay output receipt required")
    known = {r["file"]: r for r in download["files"]}
    hits = list(directory.rglob("e04_raw_replay_manifest.json"))
    require(len(hits) == 1, "exactly one completed raw replay manifest is required")
    path = hits[0]
    relative = str(path.relative_to(directory))
    require(relative in known, "replay manifest omitted from download receipt")
    verified_local(path, known[relative])
    replay = json.loads(path.read_text())
    identities = [{k: r[k] for k in ("fold", "filename", "sha256", "bytes", "contract_sha256")}
                  for r in sorted(build["fold_receipts"], key=lambda r: r["fold"])]
    require(replay["schema"] == "e04_raw_dicom_expert_replay_v1" and replay["status"] == "PASS"
            and replay["expert_rows"] == 58 and 0 <= replay["max_abs_probability_error"] <= 1e-6
            and replay["pixel_sha256_matches_owned_training_cache"]
            and replay["expert_prediction_rule"] == "each study uses its heldout fold only"
            and not replay["expert_checkpoint_selection"] and not replay["competition_submission_enabled"]
            and replay["fold_checkpoints"] == identities and replay["sources"] == staged["sources"],
            "raw replay scope, checkpoint, input or inference source identity changed")
    detail = path.parent / replay["detail"]["filename"]
    relative_detail = str(detail.relative_to(directory))
    require(relative_detail in known and known[relative_detail]["sha256"] == replay["detail"]["sha256"]
            and known[relative_detail]["bytes"] == replay["detail"]["bytes"], "replay detail download differs")
    verified_local(detail, replay["detail"])
    rows = json.loads(detail.read_text())
    require(len(rows) == replay["replayed_studies_including_batch_partners"], "replay detail coverage changed")
    expert_rows = [r for r in rows if r["expert"]]
    require(len(expert_rows) == len({r["StudyInstanceUID"] for r in expert_rows}) == 58,
            "replay detail does not cover Gold58 uniquely")
    frozen_path = ROOT / "runs/E04_contract/e04_contract.json"
    require(sha(frozen_path) == "88a7e801945c4cc5145cd3ece2dced031d662decd9588bcb0cf3a816e088272f",
            "frozen expert/fold identity map changed")
    by_uid = {r["StudyInstanceUID"]: r for r in json.loads(frozen_path.read_text())["rows"]}
    require(all(r["StudyInstanceUID"] in by_uid
                and r["fold"] == by_uid[r["StudyInstanceUID"]]["fold"]
                and r["expert"] == by_uid[r["StudyInstanceUID"]]["expert"] for r in rows),
            "raw replay used a study's trained fold or changed expert identities")
    expected_folds = {r["fold"] for r in identities}
    require(expected_folds == set(range(5)) and {r["fold"] for r in rows if r["expert"]} == expected_folds,
            "replay detail omits one or more held-out folds")
    require(all(r["fold"] in expected_folds and 0 <= r["max_abs_error"] <= 1e-6
                and not r["decode"]["prediction_fallback_used"]
                and r["decode"]["slice_failures"] == r["decode"]["header_failures"] == 0
                and r["decode"]["StudyInstanceUID"] == r["StudyInstanceUID"]
                and r["decode"]["span"] == r["decode"]["inference_input_span"] == [.06, .94]
                and len(r["probabilities"]) == 12 and all(0 <= p <= 1 for p in r["probabilities"])
                and r["decode"]["selected_slice_attempts"] == sum(s["decoded_selected_slices"]
                                                                   for s in r["decode"]["slots"]) for r in rows),
            "replay detail contains decoding failures, substitutes or prediction drift")
    receipt = {"filename": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
    return receipt, [download_path, path, detail, frozen_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-replay-output", type=Path, required=True)
    args = parser.parse_args()
    require(not (ROOT / "runs/e04_final_predict_launch.json").exists(),
            "preserve the original prediction build and observe its existing launch")
    replay_build_path = ROOT / "runs/e04_final_raw_replay_source.json"
    build = json.loads(replay_build_path.read_text())
    require(build["schema"] == "e04_final_raw_replay_job_build_v1"
            and build["kernel"] == "adityakumarrai0401/rsna-knee-e04-final-raw-replay"
            and not build["competition_submission_enabled"], "gated owned replay build required")
    for relative, digest in {**build["sources"], **build["gate_receipts"]}.items():
        require(sha(ROOT / relative) == digest, f"reviewed replay source or gate changed: {relative}")
    staged_path = ROOT / "runs/e04_raw_inference_source.json"
    staged = json.loads(staged_path.read_text())
    for relative, digest in staged["sources"].items():
        require(sha(ROOT / relative) == digest, "validated inference source changed")
    receipts = sorted(build["fold_receipts"], key=lambda r: r["fold"])
    require([r["fold"] for r in receipts] == list(range(5)), "five independent final models required")
    replay_receipt, extra_gates = verified_replay(args.raw_replay_output.resolve(), build, staged)
    prepared = ROOT / "runs/E04_staged_raw_inference"
    modules = {}
    for name, digest in staged["assembled_files"].items():
        path = prepared / name
        require(sha(path) == digest, "validated assembled module changed")
        ast.parse(path.read_text())
        modules[name] = path.read_text()
    entry = ("import json\nfrom pathlib import Path\n\n"
             "def main():\n"
             "    from e04_hidden_predict import predict_hidden_studies\n"
             f"    folds = json.loads({json.dumps(receipts)!r})\n"
             f"    replay = json.loads({json.dumps(replay_receipt)!r})\n"
             "    predict_hidden_studies(folds, replay, Path('/kaggle/working'))\n\n"
             "if __name__ == '__main__':\n    main()\n")
    ast.parse(entry)
    modules["entry.py"] = entry
    checksums = {name: hashlib.sha256(code.encode()).hexdigest() for name, code in modules.items()}
    definitions = ast.parse(modules["e04_inference_definitions.py"])
    finder = next(n for n in definitions.body if isinstance(n, ast.FunctionDef) and n.name == "bounded_files")
    bounded_finder = ast.get_source_segment(modules["e04_inference_definitions.py"], finder)
    # Only the three visible studies can use the short execution branch. Hidden
    # cohorts use the unchanged full driver with the competition runtime bound.
    wrapper = ("import csv\nimport hashlib\nimport os\nimport signal\nimport subprocess\nimport sys\nfrom pathlib import Path\n\n"
               f"MODULES = {modules!r}\nCHECKSUMS = {checksums!r}\n\n"
               + bounded_finder + "\n\n"
               "def main():\n"
               "    directory = Path('/kaggle/working/_e04_runtime')\n"
               "    directory.mkdir(parents=True, exist_ok=True)\n"
               "    for name, code in MODULES.items():\n"
               "        if hashlib.sha256(code.encode()).hexdigest() != CHECKSUMS[name]:\n"
               "            raise RuntimeError('embedded prediction source identity changed')\n"
               "        (directory / name).write_text(code)\n"
               "    # Discover the test table without importing Torch in the parent.\n"
               "    roots = []\n"
               "    for p in bounded_files('test.csv'):\n"
               "        if (p.parent / 'test_series.csv').is_file() and (p.parent / 'test_series').is_dir() and (p.parent / 'sample_submission.csv').is_file():\n"
               "            roots.append(p)\n"
               "    if len(roots) != 1:\n"
               "        raise RuntimeError('test table discovery failed')\n"
               "    with roots[0].open(newline='') as handle:\n"
               "        count = sum(1 for _ in csv.DictReader(handle))\n"
               "    if count < 1:\n"
               "        raise RuntimeError('empty test cohort')\n"
               "    timeout = 1800 if count <= 3 else 8.5 * 3600\n"
               "    child = subprocess.Popen([sys.executable, str(directory / 'entry.py')], start_new_session=True)\n"
               "    try:\n"
               "        result = child.wait(timeout=timeout)\n"
               "    except BaseException:\n"
               "        try:\n"
               "            os.killpg(child.pid, signal.SIGTERM)\n"
               "        except ProcessLookupError:\n"
               "            pass\n"
               "        try:\n"
               "            child.wait(timeout=10)\n"
               "        except subprocess.TimeoutExpired:\n"
               "            try:\n"
               "                os.killpg(child.pid, signal.SIGKILL)\n"
               "            except ProcessLookupError:\n"
               "                pass\n"
               "            child.wait()\n"
               "        raise\n"
               "    if result:\n"
               "        raise RuntimeError(f'final prediction process exited with code {result}')\n\n"
               "if __name__ == '__main__':\n    main()\n")
    ast.parse(wrapper)
    replay_meta_path = ROOT / "public_kernels/e04_final_raw_replay/kernel-metadata.json"
    replay_meta = json.loads(replay_meta_path.read_text())
    origins = [s for s in replay_meta["kernel_sources"]
               if s.startswith("adityakumarrai0401/rsna-knee-e04-convnext-folds-")]
    require(bool(origins) and len(set(origins)) == len(origins), "model input kernels missing or duplicated")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "rsna-knee-e04-final-predict.py"
    path.write_text(wrapper)
    py_compile.compile(str(path), doraise=True)
    metadata = {"id": SLUG, "title": "RSNA Knee E04 Final Predict", "code_file": path.name,
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": ["adityakumarrai0401/rsna-knee-e04-convnext-imagenet-weights"],
                "competition_sources": ["rsna-knee-abnormality-detection"],
                "kernel_sources": [*origins, build["kernel"]], "model_sources": []}
    meta = OUT / "kernel-metadata.json"
    meta.write_text(json.dumps(metadata, indent=2) + "\n")
    result = {"schema": "e04_final_predict_job_build_v1", "status": "BUILT_NOT_PUSHED",
              "kernel": SLUG, "syntax": "PASS", "replay_receipt": replay_receipt,
              "fold_receipts": receipts, "module_sha256": checksums,
              "private_visible_runtime_cap_hours": .5, "startup_finalization_margin_hours": .5,
              "hidden_runtime_cap_hours": 8.5,
              "fresh_quota_and_visible_cohort_check_required_before_private_push": True,
              "competition_submission_enabled": False,
              "sources": {str(p.relative_to(ROOT)): sha(p) for p in (path, meta, Path(__file__).resolve())},
              "gate_receipts": {str(p.resolve().relative_to(ROOT)): sha(p)
                                for p in [replay_build_path, staged_path, *extra_gates]},
              "fixed_ensemble_rule": "equal_probability_mean_of_all_five_final_epoch14_EMA_models",
              "scope": "builder preparation; visible execution and official standalone result are separate gates"}
    (ROOT / "runs/e04_final_predict_source.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "fold_receipts"}, indent=2))


if __name__ == "__main__":
    main()
