"""Monitor exactly one accepted resource pilot; never launch, fit or submit."""
from datetime import datetime, timezone
import fcntl
import json
import os
import subprocess
import sys
import time

from audit_mars_resource_pilot import ROOT, RUN, sha


def main():
    lock = (RUN / 'resource_watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launch = json.loads((RUN / 'resource_launch.json').read_text())
    assert launch['status'] == 'PUSH_ACCEPTED'
    prepared = json.loads((RUN / 'resource_control_preparation.json').read_text())
    collector = ROOT / 'scripts/observe_mars_resource_pilot.py'
    expected = prepared['sources_sha256']['scripts/observe_mars_resource_pilot.py']
    assert sha(collector) == expected
    state = {'status': 'OBSERVING_ACCEPTED_RESOURCE_PILOT', 'pid': os.getpid(),
             'kernel': launch['kernel'], 'version': launch['version'],
             'collector_sha256': expected, 'poll_interval_seconds': 120,
             'gpu_launched': False, 'classification_fitting_enabled': False,
             'competition_submission_enabled': False}
    started = time.monotonic()
    while time.monotonic() - started < 2 * 3600:
        assert sha(collector) == expected
        with (RUN / 'resource_watcher.log').open('a') as log:
            try:
                result = subprocess.run([sys.executable, str(collector)], cwd=ROOT,
                                        stdout=log, stderr=log, timeout=600)
            except subprocess.TimeoutExpired:
                result = None
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        if result is None:
            state['observation_error_type'] = 'LOCAL_COLLECTION_TIMEOUT_REPOLL_SAME_HANDLE'
        elif result.returncode:
            state['status'] = 'COLLECTOR_REVIEW_REQUIRED_NO_RELAUNCH'
        else:
            observation = json.loads((RUN / 'resource_observation.json').read_text())
            assert observation['kernel'] == launch['kernel'] and observation['accepted_version'] == launch['version']
            state.update(remote_status=observation.get('remote_status'),
                         remote_observed_utc=observation['observed_utc'])
            if observation.get('execution_audit_status') == 'PASS_REAL_DICOM_RESOURCE_OUTPUT_IDENTITY_AND_FEATURE_ARRAYS':
                state.update(status='FINISHED_RESOURCE_COST_REVIEW_REQUIRED',
                             audit_sha256=sha(RUN / 'resource_execution_audit.json'))
            elif observation.get('remote_status') in {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED', 'CANCELED'}:
                state['status'] = 'TERMINAL_RESOURCE_RUN_REVIEW_REQUIRED_NO_RELAUNCH'
                state['observation_error_type'] = observation.get('observation_error_type')
        (RUN / 'resource_watcher.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)
        if state['status'] != 'OBSERVING_ACCEPTED_RESOURCE_PILOT':
            return
        time.sleep(120)
    state['status'] = 'OBSERVATION_WINDOW_ENDED_RECHECK_SAME_HANDLE'
    (RUN / 'resource_watcher.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
