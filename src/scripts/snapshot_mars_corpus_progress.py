"""Collect one bounded progress snapshot without launching or changing a job."""
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import signal
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

import fetch_kernel_live_log
from observe_coherent_anchor_pair import remote_status_name

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/foundation_refresh_20261010/MARS'


def main():
    launch = json.loads((BASE / 'corpus_launch.json').read_text())
    kernel = launch['kernel']
    assert kernel == 'adityakumarrai0401/rsna-knee-mars-full-frozen-corpus'
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(45)
    api = KaggleApi(); api.authenticate()
    status = remote_status_name(api.kernels_status(kernel))
    observed = datetime.now(timezone.utc).isoformat()
    capture = io.StringIO()
    old_argv = sys.argv
    try:
        sys.argv = ['fetch_kernel_live_log.py', kernel, '--seconds', '5']
        with contextlib.redirect_stdout(capture):
            fetch_kernel_live_log.main()
    finally:
        sys.argv = old_argv
        signal.alarm(0)
    raw = capture.getvalue()
    (BASE / 'corpus_live_log_latest.txt').write_text(raw)
    workers, errors = {}, []
    for line in raw.splitlines():
        if 'Traceback' in line or 'Error' in line:
            errors.append(line[:180])
        start = line.find('{')
        if start < 0:
            continue
        try:
            item = json.loads(line[start:])
        except ValueError:
            continue
        if item.get('stage') == 'full_frozen_features':
            workers[str(item['shard'])] = item
    for shard, item in workers.items():
        assert shard in {'0', '1'}
        assert item['required'] == (2204 if shard == '0' else 2203)
        assert 0 <= item['completed'] <= item['required']
    terminal = {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED', 'CANCELED'}
    classification = ('VERIFIED_WAIT_EXISTING_ACCEPTED_CORPUS_RUN' if status in {'RUNNING', 'QUEUED'}
        else 'TERMINAL_STAGE_REQUIRES_EXECUTION_AUDIT' if status in terminal
        else 'UNRECOGNIZED_STAGE_STATUS_REQUIRES_OBSERVATION')
    record = {'observed_utc': datetime.now(timezone.utc).isoformat(),
        'classification': classification,
        'kernel': kernel, 'remote_status': status, 'remote_status_observed_utc': observed,
        'progress_lower_bound': sum(item['completed'] for item in workers.values()) if len(workers) == 2 else None,
        'required_studies': 4407, 'worker_snapshot': workers,
        'errors_in_log_snapshot': errors, 'classifier_started': False,
        'new_gpu_jobs': 0, 'new_competition_submission': False,
        'complete_coverage_requires_native_receipt_audit': True}
    (BASE / 'corpus_progress_latest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))


if __name__ == '__main__':
    main()
