#!/usr/bin/env python3
"""Verify the actual CPU parity receipt and its study/byte identities."""
import ast
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "runs/E03_cache_parity"
    path = directory / "e03_cache_parity.json"
    result = json.loads(path.read_text())
    if result["status"] != "COMPLETE" or result["matched_span"] != [0.15, 0.85]:
        raise RuntimeError("pixel-parity job did not finish the intended comparison")
    tree = ast.parse((ROOT / "training/e02_resnet34_distill.py").read_text())
    targets = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "TARGETS")
    with (ROOT / "data/train.csv").open(newline="") as handle:
        train = list(csv.DictReader(handle))
    all_ids = {r["StudyInstanceUID"] for r in train}
    expert_ids = {r["StudyInstanceUID"] for r in train if all(r[t] for t in targets)}
    ids = [r["StudyInstanceUID"] for r in result["rows"]]
    if (len(ids) != result["studies"] or len(set(ids)) != len(ids)
            or not expert_ids <= set(ids) or not set(ids) <= all_ids or len(expert_ids) != 58):
        raise RuntimeError("parity receipt does not uniquely cover Gold58")
    patterns, parts, wide_errors = set(), set(), []
    slice_attempts, slice_failures, header_failures = 0, 0, 0
    for row in result["rows"]:
        parts.add(row["part"])
        matched = row["span_results"]["0.15-0.85"]
        if (not matched["exact_pixel_match"] or not matched["mask_matches"]
                or matched["pixel_mean_abs_error"] != 0 or matched["pixel_max_abs_error"] != 0
                or matched["fraction_pixels_different"] != 0
                or matched["cached_sha256"] != matched["reconstructed_sha256"]):
            raise RuntimeError(f"pixel or mask mismatch: {row['StudyInstanceUID']}")
        receipt = matched["decode_receipt"]
        if receipt["StudyInstanceUID"] != row["StudyInstanceUID"] or receipt["span"] != [0.15, 0.85]:
            raise RuntimeError("decode receipt identity/geometry drift")
        if [s["slot"] for s in receipt["slots"]] != list(range(5)):
            raise RuntimeError("missing acquisition diagnostics")
        patterns.add(tuple(s.get("nonzero_cached_slices", 0) > 0 for s in receipt["slots"]))
        for slot in receipt["slots"]:
            if slot["selected_slice_attempts"] != slot["decoded_selected_slices"] + slot["slice_failures"]:
                raise RuntimeError("unaccounted selected DICOM slices")
            slice_attempts += slot["selected_slice_attempts"]
            slice_failures += slot["slice_failures"]
            header_failures += slot["header_failures"]
        wide = row["span_results"]["0.06-0.94"]
        if wide["cached_sha256"] != matched["cached_sha256"]:
            raise RuntimeError("the span comparisons used different cached targets")
        wide_errors.append(wide["pixel_mean_abs_error"])
    if parts != {"base", "extension"} or len(patterns) != result["presence_patterns"]:
        raise RuntimeError("corpus part or acquisition-presence coverage mismatch")
    expected_source = json.loads((ROOT / "runs/e03_cache_parity_source.json").read_text())
    for name, sha in expected_source["sources"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != sha:
            raise RuntimeError(f"parity source changed since launch: {name}")
    audit = {"schema": "e03_cache_parity_audit_v1", "status": "PASS",
             "pixel_parity_gate": "PASS", "studies": len(ids), "expert_studies": len(expert_ids),
             "additional_nonexpert_studies": len(set(ids) - expert_ids),
             "presence_patterns": len(patterns), "corpus_parts": sorted(parts),
             "matching_span": [0.15, 0.85], "all_pixels_exact": True, "all_masks_exact": True,
             "selected_slice_attempts": slice_attempts, "selected_slice_failures": slice_failures,
             "header_failures": header_failures,
             "wide_span_mean_pixel_mae": statistics.mean(wide_errors),
             "wide_span_median_pixel_mae": statistics.median(wide_errors),
             "wide_span_min_pixel_mae": min(wide_errors),
             "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "elapsed_job_seconds": result["elapsed_seconds"],
             "scope": "exact cached pixel/mask equality on Gold58 plus presence-pattern examples; no prediction/AUC or hidden-test guarantee"}
    (directory / "pixel_parity_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
