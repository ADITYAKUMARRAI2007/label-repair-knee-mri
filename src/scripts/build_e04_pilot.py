#!/usr/bin/env python3
"""Assemble a private GPU pilot from audited owned sampler and frozen hashes."""
import ast
import hashlib
import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def extract(path, needed):
    source = path.read_text()
    pieces = []
    for node in ast.parse(source).body:
        name = getattr(node, "name", "")
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
        if name in needed:
            pieces.append(ast.get_source_segment(source, node))
    if len(pieces) != len(needed):
        raise RuntimeError(f"missing source symbol in {path}")
    return "\n\n".join(pieces)


def main():
    contract_path = ROOT / "runs/E04_contract/e04_contract.json"
    contract = json.loads(contract_path.read_text())
    e02 = ROOT / "training/e02_resnet34_distill.py"
    e03 = ROOT / "training/e03_resnet_series_pilot.py"
    old_split = ROOT / "scripts/t03_contract.py"
    selector = ROOT / "training/e03_window_contract.py"
    inputs = ROOT / "training/e04_input_contract.py"
    model = ROOT / "training/e04_convnext_series_core.py"
    template = ROOT / "training/e04_convnext_resource_pilot.py"
    io = extract(e02, {"TARGETS", "log", "sha256_file", "seed_all", "bounded_files",
                      "find_unique", "find_competition_root", "TwoPartCorpus", "open_corpus"})
    transforms = extract(e03, {"SeriesWindows", "make_views", "current_rss_gib", "atomic_save"})
    algorithm = extract(old_split, {"PUBLIC_LABEL_SHA256", "_normalise_report", "_simhash",
                                    "report_groups", "make_folds"})
    # The split algorithm's regular-expression and Unicode imports are explicit.
    algorithm = "import re\nimport unicodedata\n" + algorithm
    constants = "\n".join([
        f"E04_TRAIN_SHA256 = {contract['official_train_sha256']!r}",
        f"E04_FOLD_SHA256 = {contract['split_audit']['fold_sha256']!r}",
        f"E04_TARGET_SHA256 = {contract['arrays']['e04_targets.npy']!r}",
        f"E04_WEIGHT_SHA256 = {contract['arrays']['e04_weights.npy']!r}"])
    input_code = inputs.read_text().replace("# __FROZEN_HASHES__", constants)
    input_code = input_code.replace("# __SPLIT_ALGORITHM__", algorithm)
    code = template.read_text().replace("# __IO__", io).replace("# __WINDOW_CONTRACT__", selector.read_text())
    code = code.replace("# __DATASET_AND_TRANSFORMS__", transforms).replace("# __INPUT_CONTRACT__", input_code)
    code = code.replace("# __MODEL__", model.read_text())
    directory = ROOT / "public_kernels/e04_convnext_resource_pilot"
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / "rsna-knee-e04-convnext-resource-pilot.py"
    dest.write_text(code)
    py_compile.compile(str(dest), doraise=True)
    sources = ["dreaddevelopment/knee-raptor-corpus", "dreaddevelopment/knee-raptor-corpus-ext",
               "adityakumarrai0401/rsna-knee-e04-convnext-imagenet-weights",
               "rayanbabur/rsna-knee-2026-calibrated-soft-targets",
               "pilkwang/rsna-knee-llm-labels", "lixin73/rsna-knee-llm-report-labels-sol56",
               "stevenleehans/rsna-knee-llm-report-labels", "dreaddevelopment/rsna-knee-labels",
               "yunusgmsoy/rsna-knee-llm-labels-4-source-merged"]
    metadata = {"id": "adityakumarrai0401/rsna-knee-e04-convnext-resource-pilot",
                "title": "RSNA Knee E04 ConvNeXt Resource Pilot", "code_file": dest.name,
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
                "machine_shape": "NvidiaTeslaT4", "dataset_sources": sources,
                "competition_sources": ["rsna-knee-abnormality-detection"],
                "kernel_sources": [], "model_sources": []}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    receipt = {"schema": "e04_pilot_source_build_v1", "syntax": "PASS", "runtime_verified": False,
               "accuracy_measured": False, "submission_enabled": False,
               "competition_records_in_pretrained_upload": False,
               "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (e02, e03, old_split, selector, inputs, model, template, contract_path, dest)}}
    (ROOT / "runs/e04_pilot_source_build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
