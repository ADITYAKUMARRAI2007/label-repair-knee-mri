#!/usr/bin/env python3
"""Verify B02 is the exact public 0.60 routing notebook under owned metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_NOTEBOOK_SHA256 = "c7326deb3a03a18212f64cfe8558409920ad0d443bfd60425bcb911b4672b446"
EXPECTED_DATASETS = {
    "dreaddevelopment/raptor-knee-maxspan",
    "dreaddevelopment/raptor-knee-native384",
    "dreaddevelopment/raptor-knee-native384dense",
    "mattiaangeli/knee-mri-fold-weights",
    "mattiaangeli/opencv-python-headless-4120088-x86",
    "marwanmath/resnet-50-radimagenet-marwan",
    "mattiaangeli/rsna-knee-coat-resgated-ep10-top3",
    "mattiaangeli/rsna-knee-coatnet-d4-depthzone-swa3-b2",
    "mattiaangeli/rsna-knee-coatnet-global96-top3",
    "antoinegg1/rsna-knee-e11-diverse-heads-v20",
    "antoinegg1/rsna-knee-e9-radimagenet-heads-v15",
    "pilkwang/rsna-knee-llm-labels",
    "prvsiyan/rsna-knee-v52-radimagenet-heads-20260812",
    "pilkwang/rsna-knee-weights",
}


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/B02_source_audit.json"))
    args = parser.parse_args()

    raw = args.notebook.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    need(digest == EXPECTED_NOTEBOOK_SHA256, "B02 notebook differs from pinned public source")
    notebook = json.loads(raw)
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"] if cell.get("cell_type") == "code"
    )
    need(source.count("_coatnet_weight = {label: 0.60") == 1,
         "exact 0.60 outer route is missing")
    for fragment in [
        "'ACL': 0.75", "'Medial Meniscus': 0.80", "'Lateral Meniscus': 1.00",
        "'Lateral OA': 0.75", "'Fracture': 0.75", "private_alpha = 0.4",
        "_DINOV2_MATCHED_MEMBERS!=20", "V18_CALIBRATOR_APPLIED",
    ]:
        need(fragment in source, f"public route contract missing: {fragment}")

    metadata = json.loads(args.metadata.read_text())
    need(metadata["id"] == "adityakumarrai0401/rsna-knee-public-0946-route-v1",
         "owned kernel ID mismatch")
    need(metadata.get("is_private") is True, "B02 must remain private")
    need(metadata.get("enable_internet") is False, "internet must be disabled")
    need(metadata.get("machine_shape") == "NvidiaTeslaT4", "dual T4 metadata missing")
    need(set(metadata.get("dataset_sources", [])) == EXPECTED_DATASETS,
         "public dataset source mismatch")
    need(metadata.get("competition_sources") == ["rsna-knee-abnormality-detection"],
         "competition source mismatch")
    need(set(metadata.get("kernel_sources", [])) == {
        "sofiaanjenje/rsna-knee-e11-train", "sofiaanjenje/rsna-knee-e13-train",
    }, "kernel sources mismatch")
    need(set(metadata.get("model_sources", [])) == {
        "metaresearch/dinov2/PyTorch/small/1",
    }, "model source mismatch")

    report = {
        "schema": "b02_source_audit_v1", "status": "PASS",
        "notebook_sha256": digest,
        "code_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "route": {"default": 0.60, "ACL": 0.75, "Medial Meniscus": 0.80,
                  "Lateral Meniscus": 1.00, "Lateral OA": 0.75, "Fracture": 0.75},
        "source": "yamadan96/rsna-knee-d4-public0946",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
