"""CPU-only spawned worker using the shared contiguous-volume reader."""
import csv
import time
from pathlib import Path
from orthodiffusion_volume import build_study

RECORDS=None
SERIES_ROOT=None

def initialize(series_csv,series_root):
    global RECORDS,SERIES_ROOT
    import cv2
    cv2.setNumThreads(1)
    RECORDS={}
    with Path(series_csv).open(newline='') as f:
        for row in csv.DictReader(f):RECORDS.setdefault(row['StudyInstanceUID'],[]).append(row)
    SERIES_ROOT=series_root

def decode(item):
    row,uid=item;begin=time.monotonic()
    try:
        volume,mask,receipt=build_study(uid,RECORDS[uid],SERIES_ROOT)
        return dict(row=row,uid=uid,volume=volume,mask=mask,receipt=receipt,
                    seconds=time.monotonic()-begin,error=None)
    except Exception as error:
        return dict(row=row,uid=uid,error=type(error).__name__,message=str(error),
                    seconds=time.monotonic()-begin)
