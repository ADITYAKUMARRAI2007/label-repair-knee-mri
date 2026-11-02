"""Observe one accepted paired experiment until its accuracy audit; never launch."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from evaluate_coherent_anchor_pair import ROOT, RUN, sha


def main():
    lock = (RUN / 'pair_watcher.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    launch = json.loads((RUN / 'pair_launch.json').read_text())
    assert launch['status'] == 'PUSH_ACCEPTED'
    check = json.loads((RUN / 'pair_collector_check.json').read_text())
    collector = ROOT / 'scripts/observe_coherent_anchor_pair.py'
    assert sha(collector) == check['collector_sha256']
    primary_path = RUN / 'pair_decision_protocol.json'
    primary_sha = sha(primary_path)
    primary = json.loads(primary_path.read_text())
    assert primary['primary_comparison'] == 'n02'
    state = {'status': 'OBSERVING_ACCEPTED_PAIR', 'pid': os.getpid(),
             'kernel': launch['kernel'], 'version': launch['version'],
             'collector_sha256': sha(collector), 'primary_protocol_sha256': primary_sha,
             'poll_interval_seconds': 120, 'gpu_launched': False,
             'competition_submission_enabled': False}
    started = time.monotonic()
    while time.monotonic() - started < 2 * 3600:
        assert sha(collector) == state['collector_sha256'] and sha(primary_path) == primary_sha
        with (RUN / 'pair_watcher.log').open('a') as log:
            result = subprocess.run([sys.executable, str(collector)], cwd=ROOT,
                                    stdout=log, stderr=log, timeout=300)
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        if result.returncode:
            state['status'] = 'COLLECTOR_REVIEW_REQUIRED_NO_RELAUNCH'
        else:
            observation = json.loads((RUN / 'pair_observation.json').read_text())
            assert observation['kernel'] == launch['kernel']
            state['remote_status'] = observation.get('remote_status')
            state['remote_observed_utc'] = observation['observed_utc']
            if observation.get('evaluation_status') == 'PAIRED_INPUT_ACCURACY_DIAGNOSTIC_COMPLETE':
                evaluation_path = RUN / 'pair_evaluation.json'
                evaluation = json.loads(evaluation_path.read_text())
                state.update(status='FINISHED_PAIRED_ACCURACY_REVIEW_REQUIRED',
                             evaluation_sha256=sha(evaluation_path),
                             primary_comparison=evaluation['comparisons']['n02'])
            elif observation.get('result') == 'INSPECT_FAILED_EXISTING_RUN_NO_RELAUNCH':
                state['status'] = 'REMOTE_PAIR_FAILED_NO_RELAUNCH'
            elif observation.get('remote_status') == 'COMPLETE' and observation.get('observation_error_type'):
                state.update(status='TERMINAL_OUTPUT_REVIEW_REQUIRED_NO_RELAUNCH',
                             observation_error_type=observation['observation_error_type'])
        (RUN / 'pair_watcher.json').write_text(json.dumps(state, indent=2) + '\n')
        print(json.dumps(state), flush=True)
        if state['status'] != 'OBSERVING_ACCEPTED_PAIR':
            return
        time.sleep(120)
    state['status'] = 'OBSERVATION_WINDOW_ENDED_RECHECK_SAME_HANDLE'
    (RUN / 'pair_watcher.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
