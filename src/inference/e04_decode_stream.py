"""Bounded CPU decoding with ordered results, shared by replay and scoring."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from e04_decode_worker import initialize_decode_worker, decode_study_for_e04


def decoded_studies(items, series_csv, series_root):
    # Submit at most eight volumes. The yielded volume leaves the pending queue
    # before another task is added. An exception propagates without substitution.
    source = iter(items)
    pool = ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn"),
                               initializer=initialize_decode_worker,
                               initargs=(str(series_csv), str(series_root)))
    pending = deque()
    try:
        for _ in range(8):
            item = next(source, None)
            if item is None:
                break
            pending.append((item, pool.submit(decode_study_for_e04, item)))
        while pending:
            item, future = pending.popleft()
            result = future.result()
            if result[0] != item[0] or result[3]["StudyInstanceUID"] != item[1]:
                raise RuntimeError("CPU decode result identity changed")
            yield result
            item = next(source, None)
            if item is not None:
                pending.append((item, pool.submit(decode_study_for_e04, item)))
    finally:
        for _, future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
