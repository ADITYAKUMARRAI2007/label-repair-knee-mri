#!/usr/bin/env python3
"""Build a self-contained E03 pilot and record the exact source composition."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    io_path = ROOT / "training/e02_resnet34_distill.py"
    template_path = ROOT / "training/e03_resnet_series_pilot.py"
    contract_path = ROOT / "training/e03_window_contract.py"
    original = io_path.read_text()
    keep = {"TARGETS", "log", "sha256_file", "seed_all", "bounded_files",
            "find_unique", "find_competition_root", "TwoPartCorpus", "open_corpus",
            "read_labels", "build_targets"}
    pieces = []
    for node in ast.parse(original).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in keep:
            pieces.append(ast.get_source_segment(original, node))
            keep.remove(name)
    if keep:
        raise RuntimeError(f"missing I/O helpers: {keep}")
    template = template_path.read_text()
    code = template.replace("# __E02_IO__", "\n\n".join(pieces))
    code = code.replace("# __WINDOW_CONTRACT__", contract_path.read_text())
    ast.parse(code)
    directory = ROOT / "public_kernels/e03_resnet_series_pilot"
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / "rsna-knee-e03-series-pilot.py"
    dest.write_text(code)
    metadata = json.loads((ROOT / "public_kernels/e02_resnet34_distill/kernel-metadata.json").read_text())
    metadata.update(id="adityakumarrai0401/rsna-knee-e03-series-resource-pilot",
                    title="RSNA Knee E03 Series Resource Pilot", code_file=dest.name)
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    paths = [io_path, template_path, contract_path, dest]
    receipt = {"schema": "e03_pilot_build_v1", "syntax": "PASS",
               "runtime_verified": False, "submission_enabled": False,
               "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in paths}}
    (ROOT / "runs/e03_pilot_source_build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
