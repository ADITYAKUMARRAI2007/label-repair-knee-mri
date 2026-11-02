"""Read-only watcher for one accepted diagnostic reproduction."""
from datetime import datetime, timezone
import fcntl
import json
import os
import subprocess
import sys
import time

from audit_mars_cuda_fix import ROOT, RUN, sha


def main():
    lock = (RUN / 'watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launch = json.loads((RUN / 'launch.json').read_text())
    assert launch['status'] == 'PUSH_ACCEPTED'
    review = json.loads((RUN / 'reproduction_review.json').read_text())
    collector = ROOT / 'scripts/observe_mars_cuda_fix.py'
    expected = review['source_sha256']['scripts/observe_mars_cuda_fix.py']
    state = {'status': 'OBSERVING_ACCEPTED_DIAGNOSTIC', 'pid': os.getpid(),
             'kernel': launch['kernel'], 'version': launch['version'],
             'poll_interval_seconds': 120, 'gpu_run_requested': False,
             'competition_submission_enabled': False}
    started = time.monotonic()
    while time.monotonic() - started < 2 * 3600:
        assert sha(collector) == expected
        with (RUN / 'watcher.log').open('a') as log:
            try:
                result = subprocess.run([sys.executable, str(collector)], cwd=ROOT,
                                        stdout=log, stderr=log, timeout=300)
            except subprocess.TimeoutExpired:
                result = None
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        if result is None or result.returncode:
            state['observation_error_type'] = 'RECHECK_SAME_HANDLE_AFTER_LOCAL_COLLECTION_ERROR'
        else:
            observation = json.loads((RUN / 'observation.json').read_text())
            state.update(remote_status=observation.get('remote_status'),
                         remote_observed_utc=observation['observed_utc'])
            if observation.get('execution_audit_status'):
                state.update(status='FINISHED_RESOURCE_REVIEW_REQUIRED',
                             audit_status=observation['execution_audit_status'])
            elif observation.get('remote_status') in {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED', 'CANCELED'}:
                state.update(status='TERMINAL_DIAGNOSIS_REQUIRED',
                             observation_error_type=observation.get('observation_error_type'))
        (RUN / 'watcher.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)
        if state['status'] != 'OBSERVING_ACCEPTED_DIAGNOSTIC': return
        time.sleep(120)
    state['status'] = 'OBSERVATION_ENDED_RECHECK_SAME_HANDLE'
    (RUN / 'watcher.json').write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
