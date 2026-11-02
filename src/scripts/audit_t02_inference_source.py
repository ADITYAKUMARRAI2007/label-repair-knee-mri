#!/usr/bin/env python3
"""Static fail-closed audit for the staged T02 inference kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/T02_inference_source_audit.json"))
    args = parser.parse_args()

    source = args.script.read_text()
    metadata = json.loads(args.metadata.read_text())
    required = [
        'T02_INFER_SCHEMA = "rsna_knee_goldfold_convnext_infer_v1"',
        'T02_SCHEMA = "rsna_knee_goldfold_convnext_v1"',
        'T02_INIT_SHA256 = "bec04468edeaaa0c75191a221d4e3b80bada5cfd263eeec36cb093548de93147"',
        'for tag in ("f012", "f34"):',
        'if [m.get("run_folds") for m in manifests] != [[0, 1, 2], [3, 4]]:',
        'expected_names = [f"t02_goldfold_convnext_fold{i}.pt" for i in range(5)]',
        'model.load_state_dict(payload["state"], strict=True)',
        'if not torch.cuda.is_available() or torch.cuda.device_count() < 2:',
        'studies, cache, mask = build_cache(slot_map, plane_map, lat_map, "test", 3)',
        'cache.shape != (len(expected), 6, 9, 224, 224)',
        'if sorted(fold_ids) != list(range(5)):',
        'mean_probability = np.mean(np.stack(fold_predictions), axis=0)',
        'method="average", pct=True',
        '"fallback_studies": 0',
        '"submission_sha256": _sha256(output)',
    ]
    missing = [fragment for fragment in required if fragment not in source]
    need(not missing, f"inference source contract missing: {missing}")
    need("T02_PAYLOAD" not in source, "obsolete embedded target payload present")
    need("_construct_targets" not in source, "training label construction leaked into inference")
    need(not re.search(r'1\.2\.840\.113619\.', source), "embedded DICOM UID present")

    need(metadata.get("id") ==
         "adityakumarrai0401/rsna-knee-goldfold-convnext-infer-v1",
         "kernel ID mismatch")
    need(metadata.get("is_private") is True, "inference kernel must remain private")
    need(metadata.get("enable_internet") is False, "internet must be disabled")
    need(metadata.get("machine_shape") == "NvidiaTeslaT4", "dual T4 metadata missing")
    need(metadata.get("dataset_sources") == [], "unexpected dataset source")
    need(metadata.get("kernel_sources") == [
        "adityakumarrai0401/rsna-knee-goldfold-convnext-train-v1",
        "adityakumarrai0401/rsna-knee-goldfold-convnext-f34",
    ], "training kernel source mismatch")
    need(metadata.get("competition_sources") == ["rsna-knee-abnormality-detection"],
         "competition source mismatch")

    report = {
        "schema": "t02_inference_source_audit_v1", "status": "PASS",
        "script_sha256": hashlib.sha256(args.script.read_bytes()).hexdigest(),
        "five_fold_probability_mean": True,
        "hidden_test_decode": {"slots": 6, "groups": 3, "image": 224,
                               "crop_mm": 130.0},
        "fail_closed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
