"""CPU worker definitions for E04 raw-DICOM replay and hidden inference.

The builder embeds the frozen e03_dicom_reader and window selector verbatim in
the same importable worker module. No default depth span is relied on here.
"""
import csv
import hashlib
from pathlib import Path

_E04_RECORDS = None
_E04_SERIES_ROOT = None
_E04_READER = None


def initialize_decode_worker(series_csv, series_root):
    global _E04_RECORDS, _E04_SERIES_ROOT, _E04_READER
    import cv2
    cv2.setNumThreads(1)
    records = {}
    with Path(series_csv).open(newline="") as handle:
        for row in csv.DictReader(handle):
            records.setdefault(row["StudyInstanceUID"], []).append(row)
    _E04_RECORDS, _E04_SERIES_ROOT, _E04_READER = records, str(series_root), make_reader()


def decode_study_for_e04(item):
    row, uid = item
    volume, mask, receipt = build_study(uid, _E04_RECORDS, _E04_SERIES_ROOT,
                                      _E04_READER, span=(0.06, 0.94))
    if receipt["slice_failures"] or receipt["header_failures"]:
        raise RuntimeError(f"E04 DICOM decoding failed for {uid}; no substitute prediction is allowed")
    select_slot_windows(mask, 8, training=False)
    receipt.update(row=row, volume_sha256=hashlib.sha256(volume.tobytes()).hexdigest(),
                   inference_input_span=[.06, .94], prediction_fallback_used=False)
    return row, volume, mask, receipt
