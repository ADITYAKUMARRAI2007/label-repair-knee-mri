"""Observe one existing run, preserve uploads, and audit results; never launch GPUs."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    a = p.parse_args(); folder = ROOT / f'runs/e05_full_fold{a.fold}'
    state_path = folder / 'watcher.json'
    if state_path.exists():
        old = json.loads(state_path.read_text())
        if old['status'] == 'RUNNING':
            try: os.kill(old['pid'], 0)
            except ProcessLookupError: pass
            else: raise RuntimeError('Observe the existing watcher; do not duplicate it')
        elif old['status'] == 'FINISHED':
            raise RuntimeError('This run has already been observed to completion')
    started = time.monotonic()
    record = {'status': 'RUNNING', 'pid': os.getpid(), 'fold': a.fold,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'gpu_launch_enabled': False}
    def persist():
        record['observed_utc'] = datetime.now(timezone.utc).isoformat()
        state_path.write_text(json.dumps(record, indent=2) + '\n')
    persist()
    def run(script, *arguments):
        with (folder / 'watcher_commands.log').open('a') as log:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts' / script),
                                     '--fold', str(a.fold), *arguments],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        return result.returncode == 0
    while time.monotonic() - started < 9000:
        try:
            run('commit_e05_snapshots.py')
            ready = [phase for phase in ['e01','e05','e09','final','terminal']
                     if (folder / f'commit_{phase}.json').exists()
                     and json.loads((folder / f'commit_{phase}.json').read_text())['status'] == 'PRIVATE_READY']
            record['private_ready_phases'] = ready
            for phase in ['e01', 'final', 'terminal']:
                marker = folder / f'verified_{phase}' / ('terminal.json' if phase == 'terminal' else 'integrity.json')
                if phase in ready and not marker.exists(): run('audit_e05_snapshot.py', '--phase', phase)
            terminal = folder / 'verified_terminal/terminal.json'
            if terminal.exists():
                report = json.loads(terminal.read_text())
                record['terminal_status'] = report['status']
                record['job_result'] = report.get('job_result')
                if report['status'] == 'FULL_JOB_FAILED':
                    record['status'] = 'FINISHED'; record['result'] = 'REMOTE_JOB_FAILED'; persist(); break
                integrity = folder / 'verified_final/integrity.json'
                if integrity.exists():
                    manifest = json.loads((folder / f'verified_final/e05_uniform_fold{a.fold}_manifest.json').read_text())
                    if manifest['status'] == 'COMPLETE':
                        with (folder / 'watcher_commands.log').open('a') as log:
                            result = subprocess.run([sys.executable, str(ROOT / 'scripts/audit_e04_fold_training.py'),
                                str(folder / 'verified_final'), '--experiment', 'e05_uniform',
                                '--output', str(folder / 'fold_audit.json')], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT, timeout=120)
                        record['fold_audit_passed'] = result.returncode == 0
                    record['status'] = 'FINISHED'; record['result'] = manifest['status']; persist(); break
            persist()
            print(json.dumps(record), flush=True)
        except Exception as error:
            record['last_observation_error'] = type(error).__name__; persist()
        time.sleep(180)
    else:
        record['status'] = 'OBSERVATION_WINDOW_ENDED'; persist()
    print(json.dumps(record), flush=True)


if __name__ == '__main__': main()
