#!/usr/bin/env python3
"""Observe one existing job, then fetch and audit its terminal outputs."""
import json
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, choices=[1,2], default=1)
    attempt = parser.parse_args().attempt
    suffix = '' if attempt == 1 else '_v2'
    launch = json.loads((ROOT / f'runs/grade_supervision_probe_launch{suffix}.json').read_text())
    ref = launch['kernel']
    if launch['status'] != 'PUSH_ACCEPTED':
        raise RuntimeError('only observe the accepted original job')
    api = KaggleApi()
    api.authenticate()
    end = time.monotonic() + 70 * 60
    last, next_poll = None, 0
    observation_path = ROOT / f'runs/grade_supervision_probe_observation{suffix}.json'
    while time.monotonic() < end:
        if time.monotonic() < next_poll:
            time.sleep(min(60, next_poll - time.monotonic()))
            continue
        next_poll = time.monotonic() + 180
        try:
            state = api.kernels_status(ref).to_dict()
        except Exception as error:
            print(json.dumps({'observation_error_type': type(error).__name__,
                              'job_not_restarted': True}), flush=True)
            continue
        observation = {'observed_utc': datetime.now(timezone.utc).isoformat(),
                       'kernel': ref, 'state': state}
        observation_path.write_text(json.dumps(observation, indent=2) + '\n')
        if state != last:
            print(json.dumps(observation), flush=True)
            last = state
        if state['status'] not in {'COMPLETE', 'ERROR', 'CANCELLED'}:
            continue
        output = ROOT / f'runs/grade_supervision_probe_outputs{suffix}'
        output.mkdir(exist_ok=True)
        try:
            log = api.kernels_logs(ref)
            (output / 'execution.log').write_text(log)
        except Exception as error:
            print(json.dumps({'log_error_type': type(error).__name__}), flush=True)
        pattern = r'^grade_probe/.*\.(json|npy|pt)$' if state['status'] == 'COMPLETE' else r'^grade_probe/.*\.json$'
        fetched = subprocess.run([sys.executable, str(ROOT / 'scripts/fetch_kernel_outputs.py'),
                                  ref, '--output', str(output), '--pattern', pattern,
                                  '--attempts', '2'], cwd=ROOT)
        if fetched.returncode:
            print(json.dumps({'status': 'TERMINAL_OUTPUT_FETCH_NEEDS_RETRY', 'job_not_restarted': True}), flush=True)
            return fetched.returncode
        if state['status'] != 'COMPLETE':
            print(json.dumps({'status': 'TERMINAL_INELIGIBLE', 'kernel_state': state['status']}), flush=True)
            return 0
        env = dict(os.environ, PYENV_VERSION='3.11.9')
        audited = subprocess.run(['python', str(ROOT / 'scripts/audit_grade_supervision_probe.py'),
                                  str(output / 'grade_probe'), '--attempt', str(attempt)], cwd=ROOT, env=env)
        return audited.returncode
    print(json.dumps({'status': 'OBSERVATION_WINDOW_ENDED', 'job_not_restarted': True,
                      'next_action': 'reobserve same kernel; deadline is not terminal evidence'}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
