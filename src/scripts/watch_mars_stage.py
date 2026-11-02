"""Read-only monitoring; collection failures retry the same terminal handle."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
import subprocess
import sys
import time

from launch_mars_stage import ROOT, BASE, STAGES, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=STAGES, required=True)
    args = parser.parse_args(); stage = args.stage
    launch = read(BASE / f'{stage}_launch.json')
    assert launch['status'] == 'PUSH_ACCEPTED'
    lock = (BASE / f'{stage}_watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = {'status': 'OBSERVING_ACCEPTED_MARS_STAGE', 'stage': stage,
             'pid': os.getpid(), 'kernel': launch['kernel'], 'version': launch['version'],
             'poll_interval_seconds': 120, 'gpu_launched': False,
             'competition_submission_enabled': False}
    started = time.monotonic()
    while time.monotonic() - started < 6 * 3600:
        with (BASE / f'{stage}_watcher.log').open('a') as log:
            try:
                completed = subprocess.run([
                    sys.executable, str(ROOT / 'scripts/observe_mars_stage.py'), '--stage', stage],
                    cwd=ROOT, stdout=log, stderr=log, timeout=300)
                successful_poll = completed.returncode == 0
            except subprocess.TimeoutExpired:
                successful_poll = False
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        if successful_poll:
            observation = read(BASE / f'{stage}_observation.json')
            state.update(remote_status=observation.get('remote_status'),
                         remote_observed_utc=observation['observed_utc'])
            error = observation.get('observation_error_type')
            if error:
                state['collection_error_type'] = error
            else:
                state.pop('collection_error_type', None)
                if observation.get('execution_audit_status'):
                    state.update(status='FINISHED_STAGE_REVIEW_REQUIRED',
                                 audit_status=observation['execution_audit_status'])
                    if observation.get('evaluation_decision'):
                        state['evaluation_decision'] = observation['evaluation_decision']
                elif observation.get('remote_status') in {
                        'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED', 'CANCELED'}:
                    state['status'] = 'TERMINAL_STAGE_FAILURE_REVIEW_REQUIRED'
        else:
            state['collection_error_type'] = 'RECHECK_SAME_HANDLE_AFTER_LOCAL_OBSERVATION_ERROR'
        (BASE / f'{stage}_watcher.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)
        if state['status'] != 'OBSERVING_ACCEPTED_MARS_STAGE':
            return
        time.sleep(120)
    state['status'] = 'OBSERVATION_LIMIT_RECHECK_SAME_REMOTE_HANDLE'
    (BASE / f'{stage}_watcher.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
