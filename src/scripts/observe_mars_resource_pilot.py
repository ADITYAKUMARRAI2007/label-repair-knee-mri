"""Collect an existing accepted resource pilot; never launch or submit."""
from datetime import datetime, timezone
import json
from pathlib import Path
import signal

from kaggle.api.kaggle_api_extended import KaggleApi

from audit_mars_resource_pilot import ROOT, RUN, audit, sha
from observe_coherent_anchor_pair import remote_status_name

PATTERN = r'^(submission\.csv|mars_resource_(execution\.json|bootstrap_failure\.json|bootstrap\.log|worker\.log)|mars_pilot_source/[^/]+|mars_resource_pilot/(receipt\.json|failure\.json|fixture_parity\.json|study_measurements\.jsonl|features/[0-9.]+\.npz))$'


def expired(*_):
    raise TimeoutError('Observe this same accepted handle; do not relaunch')


def main():
    path = RUN / 'resource_launch.json'
    if not path.exists():
        print(json.dumps({'status': 'WAIT_FOR_ACCEPTED_RESOURCE_PILOT', 'gpu_launched': False})); return
    launch = json.loads(path.read_text())
    assert launch['status'] == 'PUSH_ACCEPTED'
    assert sha(RUN / 'resource_control_preparation.json') == launch['control_preparation_sha256']
    prepared = json.loads((RUN / 'resource_control_preparation.json').read_text())
    for name, expected in prepared['sources_sha256'].items():
        assert sha(ROOT / name) == expected, name
    record = {'observed_utc': datetime.now(timezone.utc).isoformat(),
              'kernel': launch['kernel'], 'accepted_version': launch['version'],
              'gpu_launched': False, 'competition_submission_enabled': False}
    signal.signal(signal.SIGALRM, expired)
    try:
        api = KaggleApi(); api.authenticate()
        signal.alarm(45)
        status = remote_status_name(api.kernels_status(launch['kernel']))
        record['remote_status'] = status
        if status in {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED', 'CANCELED'}:
            signal.alarm(60)
            remote = RUN / 'resource_remote_source'
            api.kernels_pull(launch['kernel'], str(remote), metadata=True, quiet=True)
            meta = json.loads((remote / 'kernel-metadata.json').read_text())
            assert sha(remote / meta['code_file']) == launch['code_sha256']
            output = RUN / 'resource_output'
            token, seen, files = None, set(), []
            for _ in range(100):
                signal.alarm(60)
                names, next_token = api.kernels_output(launch['kernel'], str(output), file_pattern=PATTERN,
                                                      force=True, quiet=True, page_token=token, page_size=100)
                files.extend(names)
                if not next_token: break
                assert next_token not in seen
                seen.add(next_token); token = next_token
            else:
                raise AssertionError('Output pagination did not finish')
            record['collected_files'] = len(files)
            if status == 'COMPLETE':
                reviewed = audit(output)
                (RUN / 'resource_execution_audit.json').write_text(json.dumps(reviewed, indent=2) + '\n')
                record['execution_audit_status'] = reviewed['status']
                record['next_action'] = 'REVIEW_MEASURED_COST_BEFORE_ANY_CORPUS_EXTRACTION_OR_CLASSIFIER_FITTING'
            else:
                record['next_action'] = 'DIAGNOSE_THIS_TERMINAL_RUN_NO_AUTOMATIC_RELAUNCH'
    except Exception as error:
        record['observation_error_type'] = type(error).__name__
    finally:
        signal.alarm(0)
        (RUN / 'resource_observation.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record))


if __name__ == '__main__':
    main()
