#!/usr/bin/env python3
"""Static contract audit for the staged T03 inference kernel."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "public_kernels/t03_effb0_infer/rsna-knee-t03-effb0-infer.py"
METADATA = SOURCE.parent / "kernel-metadata.json"


def main():
    text = SOURCE.read_text()
    ast.parse(text)
    metadata = json.loads(METADATA.read_text())
    expected_sources = [f"adityakumarrai0401/rsna-knee-t03-effb0-fold-{fold}"
                        for fold in range(5)]
    required = [
        'SCHEMA = "rsna_knee_t03_effb0_slotmil_v1"',
        'NETWORK_SIZE = 224', 'EVAL_PER_SLOT = 5',
        'checkpoint["epoch"] != [8, 9, 10]',
        'components = np.full((5, len(test_ids), len(LAB))',
        'raw = components.mean(0)', 'fallback_studies": 0',
        'raise RuntimeError(f"T03 failed on study',
    ]
    missing = [value for value in required if value not in text]
    if missing:
        raise RuntimeError(f"inference contract missing: {missing}")
    if metadata["kernel_sources"] != expected_sources:
        raise RuntimeError("training kernel source order drift")
    if not metadata["is_private"] or metadata["enable_internet"]:
        raise RuntimeError("unsafe kernel visibility/network metadata")
    if text.count("def build_study(") != 1 or text.count("def SlotMIL(") != 0:
        raise RuntimeError("decoder/model definition drift")
    result = {
        "status": "PASS", "schema": "t03_inference_source_audit_v1",
        "source": str(SOURCE), "bytes": SOURCE.stat().st_size,
        "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "kernel_sources": expected_sources,
    }
    output = ROOT / "runs/t03_inference_source_audit.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
