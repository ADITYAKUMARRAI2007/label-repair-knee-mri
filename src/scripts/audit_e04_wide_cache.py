#!/usr/bin/env python3
"""Audit small cache outputs locally; full volume hashes remain a runtime gate.

Download manifests, study receipts, IDs and masks, not the 22-GB volumes. This
auditor does not pretend the manifests alone independently verify every pixel.
Full trainers must verify the complete mounted volume-file hashes before use.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "runs/E04_wide_cache_base")
    parser.add_argument("--extension", type=Path, default=ROOT / "runs/E04_wide_cache_extension")
    parser.add_argument("--part", choices=("all", "base", "extension"), default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/E04_contract/cache_audit.json")
    args = parser.parse_args()
    parity_path = ROOT / "runs/E03_cache_parity/e03_cache_parity.json"
    reference = json.loads(parity_path.read_text())
    expected = {r["StudyInstanceUID"]: r["span_results"]["0.06-0.94"]["reconstructed_sha256"]
                for r in reference["rows"]}
    with (ROOT / "data/train.csv").open(newline="") as handle:
        official = {r["StudyInstanceUID"] for r in csv.DictReader(handle)}
    all_ids, checked, summaries = [], set(), []
    parts = [(p, d, n) for p, d, n in (("base", args.base, 3200), ("extension", args.extension, 1207))
             if args.part == "all" or p == args.part]
    for part, directory, count in parts:
        prefix = f"e04_wide_{part}"
        manifest_path = directory / f"{prefix}_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        require(manifest["schema"] == "e04_owned_wide_cache_v1" and manifest["status"] == "COMPLETE",
                "cache manifest is not complete")
        require(manifest["part"] == part and manifest["shape"] == [count, 44, 336, 336], "part/shape drift")
        require(manifest["span"] == [.06, .94] and manifest["crop_mm"] == 140.0 and manifest["dtype"] == "uint8",
                "wide pixel contract drift")
        for path in (ROOT / "inference/e03_dicom_reader.py", ROOT / "training/e03_window_contract.py", parity_path):
            require(manifest["source_contract"][str(path.relative_to(ROOT))] == sha(path), "source hash drift")
        small = [directory / f"{prefix}_{suffix}" for suffix in ("ids.npy", "masks.npy", "study_receipts.jsonl")]
        for path in small:
            artifact = manifest["artifacts"][path.name]
            require(path.stat().st_size == artifact["bytes"] and sha(path) == artifact["sha256"],
                    f"small output checksum mismatch: {path.name}")
        ids = np.load(small[0], allow_pickle=False).astype(str)
        masks = np.load(small[1], allow_pickle=False)
        require(len(ids) == count and len(set(ids)) == count, "UID coverage drift")
        require(masks.shape == (count, 44) and np.isin(masks, (0, 1)).all(), "invalid masks")
        uid_sha = hashlib.sha256("".join(uid + "\n" for uid in ids).encode()).hexdigest()
        require(uid_sha == manifest["ordered_uid_sha256"], "ordered UID digest mismatch")
        with small[2].open() as handle:
            receipts = [json.loads(line) for line in handle if line.strip()]
        require(len(receipts) == count and {r["row"] for r in receipts} == set(range(count)),
                "study receipt row coverage drift")
        totals = {key: 0 for key in manifest["decode_totals"]}
        for item in receipts:
            row, uid = item["row"], item["StudyInstanceUID"]
            require(ids[row] == uid, "receipt UID is not its cache row")
            require(item["span"] == [.06, .94] and len(item["slots"]) == 5, "receipt pixel contract drift")
            require(not item["slice_failures"] and not item["header_failures"], "failed decoding accepted")
            require(int(masks[row].sum()) == item["nonzero_cached_slices"], "mask count mismatch")
            valid_triple = False
            for start, stop in ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44)):
                valid_triple |= any(masks[row, c - 1:c + 2].all() for c in range(start + 1, stop - 1))
            require(valid_triple, "no readable series-contained triple")
            slots = item["slots"]
            require(sum(s["nonzero_cached_slices"] for s in slots if "nonzero_cached_slices" in s)
                    == item["nonzero_cached_slices"], "slot/study mask count mismatch")
            require(sum(s["selected_slice_attempts"] for s in slots) == item["selected_slice_attempts"],
                    "slice-attempt count mismatch")
            for key in ("selected_slice_attempts", "slice_failures", "header_failures"):
                totals[key] += item[key]
            totals["parity_reference_checks"] += int(item["parity_reference_checked"])
            totals["missing_acquisitions"] += sum(s["state"] == "missing_acquisition" for s in slots)
            for key in ("position_fallbacks", "spacing_fallbacks"):
                totals[key] += sum(s[key] for s in slots)
            require(bool(item["parity_reference_checked"]) == (uid in expected), "reference coverage flag drift")
            if uid in expected:
                require(item["volume_sha256"] == expected[uid], "independent wide reference mismatch")
                checked.add(uid)
        require(totals == manifest["decode_totals"], "aggregate decode receipts disagree")
        volume = manifest["artifacts"][f"{prefix}_vols.npy"]
        require(volume["bytes"] == count * 44 * 336 * 336 + 128, "volume file size drift")
        require(len(volume["sha256"]) == 64, "invalid volume digest")
        all_ids.extend(ids.tolist())
        summaries.append({"part": part, "studies": count, "manifest_sha256": sha(manifest_path),
                          "decode_totals": totals, "volume_hash_to_check_on_mounted_input": volume})
    require(len(all_ids) == len(set(all_ids)) == sum(n for _, _, n in parts)
            and set(all_ids).issubset(official), "cache part identity mismatch")
    if args.part == "all":
        require(set(all_ids) == official, "complete cache study identity mismatch")
    require(checked == set(expected) & set(all_ids), "incomplete expert/presence-pattern wide reference coverage")
    result = {"schema": "e04_independent_cache_receipt_audit_v1",
              "status": "PASS_SMALL_OUTPUTS" if args.part == "all" else "PART_PASS_SMALL_OUTPUTS",
              "scope": args.part, "studies": len(all_ids),
              "independent_wide_reference_studies": len(checked), "parts": summaries,
              "complete_volume_bytes_independently_verified_locally": False,
              "remaining_gate": "full trainers must verify mounted volume-file SHA-256 before training",
              "accuracy_claim": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
