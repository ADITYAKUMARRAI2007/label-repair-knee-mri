"""Observe the accepted E05 fold4 pair and audit final outputs without launching."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/e05_kaggle_pair4'
PATTERN = (r'^(e05_paired_fold_job_manifest\.json|'
           r'(e04_fold4|e05_uniform_fold4)/(worker\.log|effective_weights\.json|'
           r'(e04_fold4|e05_uniform_fold4)_(manifest|split|history|effective_weights)\.json|'
           r'(e04_fold4|e05_uniform_fold4)_oof_predictions\.csv))$')
TERMINAL = {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED', 'CANCELED'}


def now():
    return datetime.now(timezone.utc).isoformat()


def bounded(*_):
    raise TimeoutError('observation timeout')


def main():
    lock = (RUN / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = RUN / 'watcher.json'
    if state_path.exists():
        assert json.loads(state_path.read_text())['status'] != 'FINISHED', 'Inspect completed results'
    launch = json.loads((RUN / 'launch.json').read_text())
    assert launch['status'] == 'PUSH_ACCEPTED'
    api = KaggleApi(); api.authenticate()
    record = {'status': 'OBSERVING', 'pid': os.getpid(), 'started_utc': now(),
              'kernel': launch['kernel'], 'competition_submission_enabled': False,
              'poll_interval_seconds': 300}
    signal.signal(signal.SIGALRM, bounded)

    def persist():
        record['updated_utc'] = now()
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(record, indent=2) + '\n'); temp.replace(state_path)

    persist()
    started = time.monotonic()
    while time.monotonic() - started < 9 * 3600:
        try:
            signal.alarm(60)
            status = api.kernels_status(launch['kernel']).to_dict()['status']
            signal.alarm(0)
            record.update(remote_status=status, remote_observed_utc=now())
            record.pop('observation_error_type', None)
            persist()
            print(json.dumps({'status': status, 'utc': now()}), flush=True)
            if status in TERMINAL:
                signal.alarm(120)
                path = Path(api.kernels_pull(launch['kernel'], str(RUN / 'remote_source'),
                                            metadata=False, quiet=True))
                assert hashlib.sha256(path.read_bytes()).hexdigest() == launch['code_sha256']
                files, _ = api.kernels_output(launch['kernel'], str(RUN / 'output'),
                                              file_pattern=PATTERN, force=True, quiet=True)
                signal.alarm(0)
                record['downloaded_small_files'] = len(files)
                manifest = RUN / 'output/e05_paired_fold_job_manifest.json'
                if status != 'COMPLETE':
                    record.update(status='FINISHED', result='REMOTE_JOB_FAILED_OR_CANCELLED')
                    persist(); break
                if manifest.exists():
                    job = json.loads(manifest.read_text())
                    assert job['status'] == 'COMPLETE' and job['training_complete']
                    assert {(r['arm'], r['fold'], r['completed_epoch']) for r in job['results']} == {
                        ('control', 4, 14), ('uniform', 4, 14)}
                    with (RUN / 'evaluation.log').open('a') as log:
                        for experiment, directory in [('e04', 'e04_fold4'),
                                                       ('e05_uniform', 'e05_uniform_fold4')]:
                            subprocess.run([sys.executable, str(ROOT / 'scripts/audit_e04_fold_training.py'),
                                            str(RUN / 'output' / directory), '--experiment', experiment,
                                            '--output', str(RUN / f'{experiment}_audit.json')],
                                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                           check=True, timeout=120)
                    record.update(status='FINISHED', result='BOTH_FOLD4_SMALL_OUTPUT_AUDITS_PASSED',
                                  checkpoint_bytes_independently_verified=False)
                    persist(); break
                record['output_state'] = 'WAITING_FOR_RESULT_FILE'
        except (AssertionError, subprocess.CalledProcessError) as error:
            signal.alarm(0)
            record.update(status='FINISHED', result='REVIEW_REQUIRED',
                          observation_error_type=type(error).__name__)
            persist(); break
        except Exception as error:
            signal.alarm(0)
            record['observation_error_type'] = type(error).__name__
            print(json.dumps({'observation_error_type': type(error).__name__}), flush=True)
        persist()
        time.sleep(300)
    else:
        record.update(status='OBSERVATION_WINDOW_ENDED', result='RECHECK_SAME_REMOTE_HANDLE')
        persist()
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
