#!/usr/bin/env python3
"""Ensure incomplete/weak/unreviewed E04 artifacts cannot create a replay notebook."""
import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "scripts/build_e04_raw_replay_job.py"
    spec = importlib.util.spec_from_file_location("replay_build_under_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_argv = sys.argv[:]
    passed = []
    try:
        with tempfile.TemporaryDirectory(prefix="e04_replay_build_negative_") as directory:
            fixture = Path(directory)
            module.ROOT, module.OUT = fixture, fixture / "must_not_be_written"
            (fixture / "runs").mkdir()
            audit_path, review_path = fixture / "audit.json", fixture / "review.json"
            outputs = [f"{i}=adityakumarrai0401/rsna-knee-e04-convnext-folds-fixture={fixture}"
                       for i in range(5)]

            def reject(label, audit, review, count, expected):
                audit_path.write_text(json.dumps(audit))
                review_path.write_text(json.dumps(review))
                sys.argv = [str(path), "--training-audit", str(audit_path), "--oof-review", str(review_path)]
                for output in outputs[:count]:
                    sys.argv.extend(("--fold-output", output))
                try:
                    module.main()
                except RuntimeError as error:
                    assert expected in str(error), str(error)
                else:
                    raise AssertionError("unsafe replay build was accepted")
                assert not module.OUT.exists()
                assert not (fixture / "runs/e04_final_raw_replay_source.json").exists()
                passed.append(label)

            reject("missing fifth fold", {}, {}, 4, "all five")
            reject("partial OOF despite favorable score", {"status": "PASS_PARTIAL_FOLDS_SMALL_OUTPUTS",
                   "expert_macro_auc": .99}, {}, 5, "complete independently measured")
            complete = {"status": "PASS_COMPLETE_OOF_SMALL_OUTPUTS", "complete_expert_oof": True,
                        "minimum_Gate_B_threshold_met": True, "expert_macro_auc": .899,
                        "folds": [{"fold": i} for i in range(5)]}
            reject("below fixed 0.90 validation gate", complete, {}, 5, "complete independently measured")
            complete["expert_macro_auc"] = .93
            reject("stability review unresolved", complete, {"schema": "e04_complete_oof_review_v1",
                   "status": "REQUIRES_STABILITY_REVIEW"}, 5, "matching complete uncertainty")
            reject("favorable status with outstanding stability flags", complete,
                   {"schema": "e04_complete_oof_review_v1", "status": "PASS_FOR_RAW_REPLAY",
                    "review_flags": ["fold_macro_range_above_0.15"]}, 5, "matching complete uncertainty")
    finally:
        sys.argv = original_argv
    result = {"schema": "e04_raw_replay_build_gate_check_v1", "status": "PASS_NEGATIVE_METADATA_CONTROLS",
              "rejected_cases": passed, "notebook_created_by_any_rejected_case": False,
              "builder_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "scope": "synthetic invalid metadata rejection; no passing final-model, DICOM or GPU execution claim",
              "competition_submission_enabled": False}
    (ROOT / "runs/E04_contract/raw_replay_build_gate_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
