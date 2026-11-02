#!/usr/bin/env python3
"""Assemble source only, without a kernel push, checkpoints or submission."""
import ast
import hashlib
import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble():
    worker = ROOT / "public_kernels/e04_convnext_folds_01/e04_full_fold_worker_preview.py"
    receipt = json.loads((ROOT / "runs/e04_folds_01_source.json").read_text())
    if sha(worker) != receipt["worker_source_sha256"]:
        raise RuntimeError("executed training worker changed")
    source = worker.read_text()
    required = {"TARGETS", "PRETRAINED_SHA256", "IMAGE_SIZE", "sha256_file", "bounded_files",
                "find_unique", "find_competition_root", "make_views", "SeriesMIL"}
    snippets, found = [], set()
    for node in ast.parse(source).body:
        names = {getattr(node, "name", "")}
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
        matched = names & required
        if matched:
            if matched & found:
                raise RuntimeError("ambiguous frozen inference definition")
            found.update(matched)
            snippets.append(ast.get_source_segment(source, node))
    if found != required:
        raise RuntimeError("missing frozen inference definitions")
    paths = [ROOT / f"inference/{name}.py" for name in (
        "e03_dicom_reader", "e04_series_inputs", "e04_dicom_decode_worker", "e04_fold_predict",
        "e04_replay_batches", "e04_decode_stream", "e04_raw_replay", "e04_hidden_predict")]
    paths += [ROOT / "training/e03_window_contract.py", worker, Path(__file__).resolve()]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    # The selector in the current standalone file must be the exact executed one.
    frozen_selector = next(n for n in ast.parse(source).body
                           if isinstance(n, ast.FunctionDef) and n.name == "select_slot_windows")
    standalone = (ROOT / "training/e03_window_contract.py").read_text()
    other = next(n for n in ast.parse(standalone).body
                 if isinstance(n, ast.FunctionDef) and n.name == "select_slot_windows")
    if ast.dump(frozen_selector, include_attributes=False) != ast.dump(other, include_attributes=False):
        raise RuntimeError("standalone selector differs from the executed worker")
    out = ROOT / "runs/E04_staged_raw_inference"
    out.mkdir(parents=True, exist_ok=True)
    decoder = "\n\n".join((ROOT / p).read_text() for p in (
        "inference/e03_dicom_reader.py", "training/e03_window_contract.py",
        "inference/e04_dicom_decode_worker.py"))
    files = {"e04_decode_worker.py": decoder}
    imports = ("import hashlib, json, math, os\nfrom pathlib import Path\n"
               "import numpy as np\nimport timm\nimport torch\nimport torch.nn as nn\n"
               "import torch.nn.functional as F\nfrom safetensors.torch import load_file\n")
    definitions = imports + "\n\n".join(snippets)
    definitions += "\n\n" + standalone + "\n\n" + (ROOT / "inference/e04_series_inputs.py").read_text()
    definitions += "\n\n" + (ROOT / "inference/e04_fold_predict.py").read_text()
    definitions += f"\n\nINFERENCE_SOURCE_SHA256 = {hashes!r}\n"
    files["e04_inference_definitions.py"] = definitions
    for name in ("e04_replay_batches", "e04_decode_stream"):
        files[f"{name}.py"] = (ROOT / f"inference/{name}.py").read_text()
    files["e04_raw_replay.py"] = ("from e04_inference_definitions import *\n\n"
                                   + (ROOT / "inference/e04_raw_replay.py").read_text())
    files["e04_hidden_predict.py"] = ("from e04_inference_definitions import *\n"
                                     "from e04_raw_replay import verified_artifact, packed_batch\n\n"
                                     + (ROOT / "inference/e04_hidden_predict.py").read_text())
    for name, code in files.items():
        path = out / name
        ast.parse(code)
        path.write_text(code)
        py_compile.compile(str(path), doraise=True)
    result = {"schema": "e04_raw_inference_source_assembly_v1", "status": "STAGED_NOT_GPU_EXECUTED",
              "frozen_worker_sha256": sha(worker), "sources": hashes,
              "assembled_files": {name: sha(out / name) for name in files},
              "competition_submission_enabled": False, "kernel_push_performed": False,
              "missing_runtime_inputs": ["all five complete final EMA/OOF artifact receipts",
                                         "complete OOF acceptance and stability audit"],
              "remaining_execution_gates": ["58 held-out expert raw-DICOM pixel and prediction replay",
                                            "full hidden inference timing and decoding audit"]}
    (ROOT / "runs/e04_raw_inference_source.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    assemble()
