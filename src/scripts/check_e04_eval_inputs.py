#!/usr/bin/env python3
"""Compare new inference packing directly with the frozen training Dataset."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]


def main():
    worker = ROOT / "public_kernels/e04_convnext_folds_01/e04_full_fold_worker_preview.py"
    source = worker.read_text()
    build = json.loads((ROOT / "runs/e04_folds_01_source.json").read_text())
    if hashlib.sha256(source.encode()).hexdigest() != build["worker_source_sha256"]:
        raise RuntimeError("executed fold worker identity changed")
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "SeriesWindows")
    namespace = {"np": np, "torch": torch, "Dataset": Dataset}
    selector = ROOT / "training/e03_window_contract.py"
    inputs = ROOT / "inference/e04_series_inputs.py"
    exec(compile(selector.read_text(), str(selector), "exec"), namespace)
    exec(compile(ast.get_source_segment(source, node), str(worker), "exec"), namespace)
    exec(compile(inputs.read_text(), str(inputs), "exec"), namespace)
    frozen_set, pack = namespace["SeriesWindows"], namespace["make_eval_arrays"]
    ranges = ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))
    patterns = [np.asarray([int(bool(p & (1 << s))) for s, (a, b) in enumerate(ranges)
                           for _ in range(a, b)], np.uint8) for p in range(32)]
    patterns.extend([np.asarray([int(i != missing) for i in range(44)], np.uint8) for missing in range(44)])
    matched, rejected = 0, 0
    for mask in patterns:
        # Each plane identifies its original slice, exposing channel/axis/slot mistakes.
        volume = np.broadcast_to(np.arange(1, 45, dtype=np.uint8)[:, None, None], (44, 336, 336)).copy()
        volume[mask == 0] = 0
        dataset = frozen_set([0], np.zeros((1, 12), np.float32), mask[None], 8, False)
        dataset.corpus = [volume]
        try:
            expected = dataset[0][:3]
        except ValueError:
            try:
                pack(volume, mask)
            except ValueError:
                rejected += 1
                continue
            raise RuntimeError("inference accepted a study rejected by training")
        actual = pack(volume, mask)
        if not all(np.array_equal(a, b.numpy()) and a.dtype == b.numpy().dtype
                   for a, b in zip(actual, expected)):
            raise RuntimeError("inference packing differs from frozen training")
        matched += 1
    result = {"schema": "e04_eval_input_packing_check_v1", "status": "PASS",
              "patterns": len(patterns), "matched_readable_patterns": matched,
              "both_rejected_unreadable_patterns": rejected,
              "worker_sha256": hashlib.sha256(worker.read_bytes()).hexdigest(),
              "input_helper_sha256": hashlib.sha256(inputs.read_bytes()).hexdigest(),
              "selector_sha256": hashlib.sha256(selector.read_bytes()).hexdigest(),
              "scope": "exact array/dtype/axis packing against executed training Dataset; no DICOM reconstruction or GPU prediction claim",
              "remaining_gate": "held-out expert checkpoint predictions replayed from raw DICOM with matching validation batches"}
    (ROOT / "runs/E04_contract/eval_input_packing_check.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
