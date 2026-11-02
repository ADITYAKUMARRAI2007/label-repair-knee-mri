"""Observe one existing launch; collect and audit completed small artifacts."""
from datetime import datetime, timezone
import json
from pathlib import Path
import signal

from kaggle.api.kaggle_api_extended import KaggleApi

from audit_published_tri_reader import audit, ROOT, RUN, PACKAGE, sha


def timeout(*_):
    raise TimeoutError('Bounded observation expired;inspect the same handle')


def main():
    launch = json.loads((RUN / 'launch.json').read_text())
    assert launch['status'] == 'PUSH_ACCEPTED'
    api = KaggleApi()
    api.authenticate()
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(50)
    record = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'kernel': launch['kernel']}
    try:
        status = api.kernels_status(launch['kernel']).to_dict()['status']
        record['remote_status'] = status
        if status == 'COMPLETE':
            # Source identity is checked before treating executed outputs as eligible.
            remote = RUN / 'remote_source'
            api.kernels_pull(launch['kernel'], str(remote), metadata=True, quiet=True)
            meta = json.loads((remote / 'kernel-metadata.json').read_text())
            assert sha(remote / meta['code_file']) == launch['code_sha256']
            output = RUN / 'output'
            pattern = r'^(submission\.csv|published_tri_reader/.*|fixed_reader_blend/.*|tri_reader_source/.*)$'
            token, seen, files = None, set(), []
            signal.alarm(120)
            while True:
                names, token = api.kernels_output(launch['kernel'], str(output), file_pattern=pattern,
                                                  force=True, quiet=True, page_token=token, page_size=100)
                files.extend(names)
                if not token:
                    break
                assert token not in seen
                seen.add(token)
            record['collected_files'] = len(files)
            report = audit(output)
            (RUN / 'independent_audit.json').write_text(json.dumps(report, indent=2) + '\n')
            record['audit_status'] = report['status']
        elif status in {'ERROR', 'CANCELLED', 'CANCELED', 'CANCEL_ACKNOWLEDGED'}:
            record['result'] = 'INSPECT_FAILED_EXISTING_RUN_NO_RELAUNCH'
    except Exception as e:
        # Network errors may contain signed URLs;persist the type only.
        record['observation_error_type'] = type(e).__name__
    finally:
        signal.alarm(0)
        (RUN / 'observation.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
