"""Collect and evaluate an existing paired run; never allocate or submit."""
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys

from kaggle.api.kaggle_api_extended import KaggleApi

from evaluate_coherent_anchor_pair import ROOT, RUN, audit_execution, sha

PATTERN = r'^(submission\.csv|coherent_pair_source/[^/]+|coherent_anchor_pair/(receipt\.json|failure\.json|rejected_submission\.csv|decoded_inputs\.json|decoder\.log|(control|candidate)\.(csv|log)))$'


def read(path):
    return json.loads(path.read_text())


def timeout(*_):
    raise TimeoutError('Recheck the existing handle; do not relaunch')


def remote_status_name(response):
    # The SDK omits the default QUEUED enum value from to_dict().
    status = getattr(response, 'status', None)
    name = getattr(status, 'name', None)
    if isinstance(name, str) and name:
        return name
    value = response.to_dict().get('status')
    if isinstance(value, str) and value:
        return value
    raise ValueError('No explicit remote status available')


def main():
    launch_path = RUN / 'pair_launch.json'
    if not launch_path.exists():
        print(json.dumps({'status': 'WAIT_FOR_ACCEPTED_PAIRED_LAUNCH', 'gpu_launched': False}))
        return
    launch = read(launch_path)
    assert launch['status'] == 'PUSH_ACCEPTED', 'Reconcile the existing attempt before continuing'
    assert isinstance(launch['version'], int) and launch['version'] > 0
    preparation = read(RUN / 'pair_launch_preparation.json')
    for name, expected in preparation['sources'].items():
        assert sha(ROOT / name) == expected, 'Prepared source changed: ' + name
    api = KaggleApi(); api.authenticate()
    record = {'observed_utc': datetime.now(timezone.utc).isoformat(),
              'kernel': launch['kernel'], 'accepted_version': launch['version'],
              'gpu_launched': False, 'competition_submission_enabled': False}
    signal.signal(signal.SIGALRM, timeout)
    try:
        signal.alarm(50)
        status = remote_status_name(api.kernels_status(launch['kernel']))
        signal.alarm(0)
        record['remote_status'] = status
        if status == 'COMPLETE':
            remote = RUN / 'pair_remote_source'
            signal.alarm(50)
            api.kernels_pull(launch['kernel'], str(remote), metadata=True, quiet=True)
            signal.alarm(0)
            meta = read(remote / 'kernel-metadata.json')
            assert sha(remote / meta['code_file']) == launch['code_sha256']
            output = RUN / 'pair_output'
            token, seen, files = None, set(), []
            for _ in range(100):
                signal.alarm(60)
                names, next_token = api.kernels_output(launch['kernel'], str(output), file_pattern=PATTERN,
                                                      force=True, quiet=True, page_token=token, page_size=100)
                signal.alarm(0)
                files.extend(names)
                if not next_token: break
                assert next_token not in seen, 'Repeated output page token'
                seen.add(next_token)
                token = next_token
            else:
                raise AssertionError('Output pagination did not finish')
            record['collected_files'] = len(files)
            audit = audit_execution(output)
            (RUN / 'pair_execution_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
            record['execution_audit_status'] = audit['status']
            with (RUN / 'pair_evaluation.log').open('a') as log:
                subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate_coherent_anchor_pair.py'),
                                '--output', str(output)], check=True, cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, timeout=120)
            evaluation = read(RUN / 'pair_evaluation.json')
            assert evaluation['status'] == 'PAIRED_INPUT_ACCURACY_DIAGNOSTIC_COMPLETE'
            assert evaluation['competition_submission_enabled'] is False
            record['evaluation_status'] = evaluation['status']
            record['decision'] = evaluation['decision']
        elif status in {'ERROR', 'CANCELLED', 'CANCELED', 'CANCEL_ACKNOWLEDGED'}:
            record['result'] = 'INSPECT_FAILED_EXISTING_RUN_NO_RELAUNCH'
    except Exception as error:
        # API exception messages may contain signed URLs; retain only their type.
        record['observation_error_type'] = type(error).__name__
    finally:
        signal.alarm(0)
        (RUN / 'pair_observation.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record))


if __name__ == '__main__':
    main()
