#!/usr/bin/env python3
"""Build the CPU-only parity job from pinned training I/O and the staged reader."""
import ast
import hashlib
import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    io = ROOT / "training/e02_resnet34_distill.py"
    text = io.read_text()
    names = {"bounded_files", "find_unique", "find_competition_root", "sha256_file", "TwoPartCorpus", "open_corpus"}
    helpers = [ast.get_source_segment(text, node) for node in ast.parse(text).body if getattr(node, "name", None) in names]
    if len(helpers) != len(names):
        raise RuntimeError("missing corpus I/O helper")
    reader = ROOT / "inference/e03_dicom_reader.py"
    template = ROOT / "training/e03_cache_parity.py"
    code = template.read_text().replace("# __IO__", "\n\n".join(helpers))
    code = code.replace("# __READER__", reader.read_text())
    directory = ROOT / "public_kernels/e03_cache_pixel_parity"
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / "rsna-knee-e03-cache-pixel-parity.py"
    dest.write_text(code)
    py_compile.compile(str(dest), doraise=True)
    metadata = {"id": "adityakumarrai0401/rsna-knee-e03-cache-pixel-parity",
                "title": "RSNA Knee E03 Cache Pixel Parity", "code_file": dest.name,
                "language": "python", "kernel_type": "script", "is_private": True,
                "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
                "dataset_sources": ["dreaddevelopment/knee-raptor-corpus", "dreaddevelopment/knee-raptor-corpus-ext"],
                "kernel_sources": [], "competition_sources": ["rsna-knee-abnormality-detection"], "model_sources": []}
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    receipt = {"schema": "e03_cache_parity_source_v1", "syntax": "PASS", "gpu": False,
               "submission_enabled": False,
               "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [io, reader, template, dest]}}
    (ROOT / "runs/e03_cache_parity_source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
