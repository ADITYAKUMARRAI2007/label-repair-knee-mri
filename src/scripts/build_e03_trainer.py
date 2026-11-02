#!/usr/bin/env python3
"""Assemble the E03 trainer from the exact tested pilot and frozen split code."""
import ast
import hashlib
import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    pilot = ROOT / "public_kernels/e03_resnet_series_pilot/rsna-knee-e03-series-pilot.py"
    source = pilot.read_text()
    tree = ast.parse(source)
    pieces = []
    for node in tree.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.If) or getattr(node, "name", None) == "main":
            continue
        pieces.append(ast.get_source_segment(source, node))
    core = "\n\n".join(pieces)
    split_file = ROOT / "training/e03_split_contract.py"
    split_audit = json.loads((ROOT / "runs/E03_contract/split_audit.json").read_text())
    template_file = ROOT / "training/e03_resnet_series_train.py"
    code = template_file.read_text().replace("# __CORE__", core)
    code = code.replace("# __SPLIT__", split_file.read_text())
    code = code.replace("__EXPECTED_SPLIT_SHA256__", split_audit["split_sha256"])
    ast.parse(code)
    directory = ROOT / "public_kernels/e03_resnet_series_train"
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / "rsna-knee-e03-resnet-series-train.py"
    dest.write_text(code)
    py_compile.compile(str(dest), doraise=True)
    metadata = json.loads((ROOT / "public_kernels/e03_resnet_series_pilot/kernel-metadata.json").read_text())
    metadata.update(id="adityakumarrai0401/rsna-knee-e03-resnet-series-train",
                    title="RSNA Knee E03 ResNet Series Train", code_file=dest.name)
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    paths = [pilot, split_file, template_file, dest]
    receipt = {"schema": "e03_trainer_build_v1", "syntax": "PASS", "runtime_verified": False,
               "split_sha256": split_audit["split_sha256"],
               "submission_enabled": False, "checkpoint_selection": "fixed_final_ema_epoch14",
               "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ROOT / "runs/e03_trainer_source_build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
