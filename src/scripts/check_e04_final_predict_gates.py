#!/usr/bin/env python3
"""Check raw-replay rejection controls and prediction wrapper syntax locally."""
import ast
import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    builder = ROOT / "scripts/build_e04_final_predict_job.py"
    spec = importlib.util.spec_from_file_location("prediction_builder_under_check", builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    passed = []
    with tempfile.TemporaryDirectory(prefix="e04_prediction_gate_negative_") as directory:
        fixture = Path(directory)
        module.ROOT = fixture
        output = fixture / "downloaded"
        output.mkdir()
        manifest = output / "e04_raw_replay_manifest.json"
        download = output / "download_receipt.json"
        build = {"kernel": "adityakumarrai0401/rsna-knee-e04-final-raw-replay",
                 "fold_receipts": [{"fold": i, "filename": f"e04_fold{i}_ema.pt", "sha256": "a" * 64,
                                    "bytes": 10, "contract_sha256": "b" * 64} for i in range(5)]}
        staged = {"sources": {"test_only_source": "c" * 64}}
        base = {"schema": "e04_raw_dicom_expert_replay_v1", "status": "PASS", "expert_rows": 58,
                "max_abs_probability_error": 0.0, "pixel_sha256_matches_owned_training_cache": True,
                "expert_prediction_rule": "each study uses its heldout fold only",
                "expert_checkpoint_selection": False, "competition_submission_enabled": False,
                "fold_checkpoints": build["fold_receipts"], "sources": staged["sources"]}

        def reject(label, changed, expected, corrupt_hash=False, wrong_origin=False):
            data = json.loads(json.dumps(base))
            data.update(changed)
            manifest.write_text(json.dumps(data))
            record = {"file": manifest.name, "bytes": manifest.stat().st_size,
                      "sha256": "0" * 64 if corrupt_hash else hashlib.sha256(manifest.read_bytes()).hexdigest()}
            download.write_text(json.dumps({"status": "COMPLETE", "job_mutations": False,
                                           "kernel": "someone/other-replay" if wrong_origin else build["kernel"],
                                           "files": [record]}))
            try:
                module.verified_replay(output, build, staged)
            except RuntimeError as error:
                assert expected in str(error), str(error)
            else:
                raise AssertionError("unsafe replay accepted")
            passed.append(label)

        reject("wrong owned replay origin", {}, "owned replay", wrong_origin=True)
        reject("changed downloaded manifest bytes", {}, "changed local artifact", corrupt_hash=True)
        reject("partial expert coverage", {"expert_rows": 57}, "raw replay scope")
        reject("excessive probability drift", {"max_abs_probability_error": 1e-3}, "raw replay scope")
        reject("nonfinite probability drift", {"max_abs_probability_error": float("nan")}, "raw replay scope")
        reject("trained-model averaging on experts", {"expert_prediction_rule": "average all five"}, "raw replay scope")
        reject("different checkpoint generation", {"fold_checkpoints": []}, "raw replay scope")
        reject("changed inference source", {"sources": {"different": "d" * 64}}, "raw replay scope")
    # Evaluate only our wrapper string expression; no model definition is imported
    # or instantiated and no synthetic kernel/checkpoint is put in public_kernels.
    tree = ast.parse(builder.read_text())
    wrapper_expr = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "wrapper" for t in n.targets))
    definitions = (ROOT / "runs/E04_staged_raw_inference/e04_inference_definitions.py").read_text()
    finder = next(n for n in ast.parse(definitions).body
                  if isinstance(n, ast.FunctionDef) and n.name == "bounded_files")
    sample_modules = {"entry.py": "raise RuntimeError('source-only local check; no checkpoints')\n"}
    values = {"modules": sample_modules,
              "checksums": {n: hashlib.sha256(c.encode()).hexdigest() for n, c in sample_modules.items()},
              "bounded_finder": ast.get_source_segment(definitions, finder)}
    code = eval(compile(ast.Expression(wrapper_expr), str(builder), "eval"), {"__builtins__": {}}, values)
    wrapper_tree = ast.parse(code)
    compile(wrapper_tree, "<prediction-wrapper-check>", "exec")
    assert "torch" not in [a.name for n in wrapper_tree.body if isinstance(n, ast.Import) for a in n.names]
    assert "train_series" in code and "test_series" in code and "start_new_session=True" in code
    assert "timeout = 1800 if count <= 3 else 8.5 * 3600" in code
    result = {"schema": "e04_final_prediction_gate_check_v1", "status": "PASS_NEGATIVE_METADATA_AND_WRAPPER_SYNTAX",
              "rejected_cases": passed, "wrapper_syntax": "PASS", "notebook_or_model_created": False,
              "builder_sha256": hashlib.sha256(builder.read_bytes()).hexdigest(),
              "scope": "invalid replay metadata rejection and generated wrapper compilation; no passing final model, DICOM or GPU execution claim",
              "competition_submission_enabled": False}
    (ROOT / "runs/E04_contract/final_prediction_gate_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
