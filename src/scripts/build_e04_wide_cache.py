#!/usr/bin/env python3
"""Build two private CPU scripts, each below Kaggle's 20-GB output limit."""
import ast
import hashlib
import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    io = ROOT / "training/e02_resnet34_distill.py"
    reader = ROOT / "inference/e03_dicom_reader.py"
    selector = ROOT / "training/e03_window_contract.py"
    template = ROOT / "training/e04_wide_cache.py"
    parity = ROOT / "runs/E03_cache_parity/e03_cache_parity.json"
    audited = ROOT / "runs/E03_cache_parity/pixel_parity_audit.json"
    document = json.loads(parity.read_text())
    if document["pixel_parity_gate"] != "PASS" or document["studies"] != 59:
        raise RuntimeError("verified parity reference unavailable")
    reference = {r["StudyInstanceUID"]: r["span_results"]["0.06-0.94"]["reconstructed_sha256"]
                 for r in document["rows"]}
    needed = {"bounded_files", "find_unique", "find_competition_root", "sha256_file"}
    source = io.read_text()
    helpers = [ast.get_source_segment(source, n) for n in ast.parse(source).body
               if getattr(n, "name", None) in needed]
    if len(helpers) != len(needed):
        raise RuntimeError("missing reader I/O helper")
    contract = {str(p.relative_to(ROOT)): sha(p) for p in (reader, selector, template, parity, audited)}
    builds = []
    for part in ("base", "extension"):
        name = f"rsna-knee-e04-wide-cache-{part}"
        directory = ROOT / "public_kernels" / name.replace("rsna-knee-", "").replace("-", "_")
        directory.mkdir(parents=True, exist_ok=True)
        config = f"PART = {part!r}\nWIDE_REFERENCE = {reference!r}\nSOURCE_CONTRACT = {contract!r}"
        code = template.read_text().replace("# __IO__", "\n\n".join(helpers))
        code = code.replace("# __READER__", reader.read_text()).replace("# __WINDOW_CONTRACT__", selector.read_text())
        code = code.replace("# __CONFIG__", config)
        destination = directory / f"{name}.py"
        destination.write_text(code)
        py_compile.compile(str(destination), doraise=True)
        metadata = {"id": f"adityakumarrai0401/{name}",
                    "title": f"RSNA Knee E04 Wide Cache {part.title()}", "code_file": destination.name,
                    "language": "python", "kernel_type": "script", "is_private": True,
                    "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                    "dataset_sources": ["dreaddevelopment/knee-raptor-corpus" if part == "base"
                                        else "dreaddevelopment/knee-raptor-corpus-ext"],
                    "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"],
                    "model_sources": []}
        (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        builds.append({"part": part, "kernel": metadata["id"], "code_sha256": sha(destination),
                       "shape": [3200 if part == "base" else 1207, 44, 336, 336],
                       "expected_volume_bytes": (3200 if part == "base" else 1207) * 44 * 336 * 336 + 128})
    receipt = {"schema": "e04_wide_cache_build_v1", "status": "SYNTAX_PASS",
               "runtime_verified": False, "submission_enabled": False, "sources": contract, "builds": builds}
    (ROOT / "runs/e04_wide_cache_source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
