#!/usr/bin/env python3
"""Reconstruct held-out pixels from DICOMs and compare with the actual cache.

CPU only. It does not read prediction scores or fit preprocessing to labels.
"""
import hashlib
import json
import mmap
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

# __IO__
# __READER__

T0 = time.time()
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main():
    root = find_competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    series = pd.read_csv(root / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    records = {uid: frame.to_dict("records") for uid, frame in series.groupby("StudyInstanceUID", sort=False)}
    corpus, masks, ids = open_corpus()
    row_by_uid = {uid: i for i, uid in enumerate(ids)}
    selected = train.loc[train[TARGETS].notna().all(1), "StudyInstanceUID"].astype(str).tolist()
    if len(selected) != 58:
        raise RuntimeError("Gold58 identity drift")
    # Add one study for each slot-presence pattern missing from the expert set.
    patterns = {tuple(bool(masks[row_by_uid[u], a:b].any()) for a, b in
                      ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))) for u in selected}
    for row, uid in enumerate(ids):
        pattern = tuple(bool(masks[row, a:b].any()) for a, b in
                        ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44)))
        if pattern not in patterns:
            selected.append(uid)
            patterns.add(pattern)
    results = []
    reader = make_reader()
    for uid in selected:
        row = row_by_uid[uid]
        expected = np.array(corpus[row], copy=True)
        item = {"StudyInstanceUID": uid, "part": "base" if row < corpus.n_a else "extension", "span_results": {}}
        for span in ((0.15, 0.85), (0.06, 0.94)):
            volume, mask, receipt = build_study(uid, records, str(root / "train_series"), reader, span=span)
            error = np.abs(volume.astype(np.int16) - expected.astype(np.int16))
            item["span_results"][f"{span[0]:.2f}-{span[1]:.2f}"] = {
                "exact_pixel_match": bool(np.array_equal(volume, expected)),
                "pixel_mean_abs_error": float(error.mean()), "pixel_max_abs_error": int(error.max()),
                "fraction_pixels_different": float(np.count_nonzero(error) / error.size),
                "mask_matches": bool(np.array_equal(mask, masks[row])),
                "decode_receipt": receipt,
                "reconstructed_sha256": hashlib.sha256(volume.tobytes()).hexdigest(),
                "cached_sha256": hashlib.sha256(expected.tobytes()).hexdigest()}
            del volume, error
        results.append(item)
        corpus.release_pages()
        print(json.dumps({"done": len(results), "total": len(selected), "part": item["part"],
                          "matching_span_mae": item["span_results"]["0.15-0.85"]["pixel_mean_abs_error"],
                          "wide_span_mae": item["span_results"]["0.06-0.94"]["pixel_mean_abs_error"]}), flush=True)
        del expected
    aligned = [r["span_results"]["0.15-0.85"] for r in results]
    exact = all(r["exact_pixel_match"] and r["mask_matches"] for r in aligned)
    result = {"schema": "e03_cache_pixel_parity_v1", "status": "COMPLETE",
              "pixel_parity_gate": "PASS" if exact else "REQUIRES_DIAGNOSIS",
              "studies": len(selected), "gold_studies": 58, "presence_patterns": len(patterns),
              "matched_span": [0.15, 0.85], "elapsed_seconds": time.time() - T0,
              "exact_matching_span_studies": sum(r["exact_pixel_match"] for r in aligned),
              "rows": results,
              "scope": "all Gold58 and one per otherwise-unrepresented acquisition-presence pattern; not hidden-test decoding"}
    output = Path("/kaggle/working/e03_cache_parity.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
