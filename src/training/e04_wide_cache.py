#!/usr/bin/env python3
"""Build one owned wide-depth cache part with bounded CPU workers and receipts."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import csv
import hashlib
import json
import mmap
import multiprocessing
import resource
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

# __IO__
# __READER__
# __WINDOW_CONTRACT__
# __CONFIG__

WIDE_SPAN = (0.06, 0.94)
WORKERS = 4
MAX_IN_FLIGHT = 8
_RECORDS = None
_SERIES_ROOT = None
_READER = None


def initialise_worker(records, series_root):
    global _RECORDS, _SERIES_ROOT, _READER
    import cv2
    cv2.setNumThreads(1)
    _RECORDS, _SERIES_ROOT = records, series_root
    _READER = make_reader()


def reconstruct(item):
    row, uid = item
    volume, mask, receipt = build_study(uid, _RECORDS, _SERIES_ROOT, _READER, span=WIDE_SPAN)
    if receipt["slice_failures"] or receipt["header_failures"]:
        raise RuntimeError(f"decode failure in cache study {uid}")
    # A nonzero study alone is insufficient: the reader needs a valid triple.
    select_slot_windows(mask, 1, training=False)
    sha = hashlib.sha256(volume.tobytes()).hexdigest()
    reference = WIDE_REFERENCE.get(uid)
    if reference and sha != reference:
        raise RuntimeError(f"wide parity reference mismatch: {uid}")
    receipt.update(row=row, volume_sha256=sha, parity_reference_checked=bool(reference))
    return row, volume, mask, receipt


def flush_and_release(array):
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and hasattr(mapping, "madvise"):
        mapping.madvise(mmap.MADV_DONTNEED)


def main():
    started = time.time()
    root = find_competition_root()
    id_file = find_unique("all_ids.npy" if PART == "base" else "extra_ids.npy")
    ids = np.load(id_file, allow_pickle=True).astype(str)
    count = 3200 if PART == "base" else 1207
    if len(ids) != count or len(set(ids)) != count:
        raise RuntimeError("source part study identity drift")
    with (root / "train.csv").open(newline="") as handle:
        official_ids = {r["StudyInstanceUID"] for r in csv.DictReader(handle)}
    if not set(ids).issubset(official_ids):
        raise RuntimeError("cache includes non-training study")
    with (root / "train_series.csv").open(newline="") as handle:
        records = {}
        for record in csv.DictReader(handle):
            records.setdefault(record["StudyInstanceUID"], []).append(record)
    out = Path("/kaggle/working")
    prefix = f"e04_wide_{PART}"
    volumes_path = out / f"{prefix}_vols.npy"
    if volumes_path.exists():
        raise RuntimeError("refusing to overwrite an existing cache; partial output is not resumable")
    volumes = np.lib.format.open_memmap(volumes_path, mode="w+", dtype=np.uint8,
                                       shape=(count, MAXS, IMG, IMG))
    masks = np.zeros((count, MAXS), dtype=np.uint8)
    done = np.zeros(count, dtype=bool)
    totals = {"selected_slice_attempts": 0, "slice_failures": 0, "header_failures": 0,
              "parity_reference_checks": 0, "missing_acquisitions": 0,
              "position_fallbacks": 0, "spacing_fallbacks": 0}
    work = iter(enumerate(ids.tolist()))
    # spawn avoids inheriting the parent's 16-GB writable mapping into workers.
    context = multiprocessing.get_context("spawn")
    receipts = out / f"{prefix}_study_receipts.jsonl"
    with receipts.open("w") as handle, ProcessPoolExecutor(
            max_workers=WORKERS, mp_context=context, initializer=initialise_worker,
            initargs=(records, str(root / "train_series"))) as executor:
        pending = set()
        for _ in range(min(MAX_IN_FLIGHT, count)):
            pending.add(executor.submit(reconstruct, next(work)))
        completed = 0
        while pending:
            ready, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in ready:
                row, volume, mask, receipt = future.result()
                if done[row]:
                    raise RuntimeError("duplicate cache output row")
                volumes[row] = volume
                masks[row] = mask
                done[row] = True
                completed += 1
                for key in ("selected_slice_attempts", "slice_failures", "header_failures"):
                    totals[key] += receipt[key]
                totals["parity_reference_checks"] += int(receipt["parity_reference_checked"])
                totals["missing_acquisitions"] += sum(s["state"] == "missing_acquisition" for s in receipt["slots"])
                for key in ("position_fallbacks", "spacing_fallbacks"):
                    totals[key] += sum(s[key] for s in receipt["slots"])
                handle.write(json.dumps(receipt) + "\n")
                del volume, mask, receipt
                if completed % 32 == 0:
                    flush_and_release(volumes)
                    handle.flush()
                    print(json.dumps({"part": PART, "done": completed, "total": count,
                                      "elapsed_seconds": round(time.time() - started, 2),
                                      "parent_peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20}), flush=True)
                try:
                    pending.add(executor.submit(reconstruct, next(work)))
                except StopIteration:
                    pass
            del ready
    if not done.all() or totals["parity_reference_checks"] != len(set(ids) & set(WIDE_REFERENCE)):
        raise RuntimeError("cache coverage or independent reference coverage incomplete")
    flush_and_release(volumes)
    del volumes
    ids_path, masks_path = out / f"{prefix}_ids.npy", out / f"{prefix}_masks.npy"
    np.save(ids_path, ids, allow_pickle=False)
    np.save(masks_path, masks, allow_pickle=False)
    paths = [volumes_path, ids_path, masks_path, receipts]
    manifest = {"schema": "e04_owned_wide_cache_v1", "status": "COMPLETE", "part": PART,
                "shape": [count, MAXS, IMG, IMG], "dtype": "uint8", "span": list(WIDE_SPAN),
                "crop_mm": CROP_MM, "slots": SLOTS, "workers": WORKERS,
                "max_in_flight": MAX_IN_FLIGHT, "decode_totals": totals,
                "ordered_uid_sha256": hashlib.sha256("".join(u + "\n" for u in ids).encode()).hexdigest(),
                "source_ids_sha256": sha256_file(id_file), "source_contract": SOURCE_CONTRACT,
                "elapsed_seconds": time.time() - started, "submission_enabled": False,
                "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in paths}}
    (out / f"{prefix}_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
