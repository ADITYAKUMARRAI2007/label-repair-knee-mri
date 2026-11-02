"""Observe one repair execution and collect its underlying worker failure."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal

from kaggle.api.kaggle_api_extended import KaggleApi
from audit_mars_repair_run import ROOT, RUN, audit, sha
from observe_coherent_anchor_pair import remote_status_name

PATTERN = (r'^(mars_latest_attempt\.json|.*\.log|'
           r'mars_resource_attempt_[^/]+/(mars_resource_(execution|bootstrap_failure)\.json|'
           r'mars_pilot_source/[^/]+|mars_resource_pilot/(receipt\.json|failure\.json|'
           r'fixture_parity\.json|study_measurements\.jsonl|features/[0-9.]+\.npz)))$')
TERMINAL = {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED', 'CANCELED'}


def main():
    launch = json.loads((RUN / 'launch.json').read_text())
    assert launch['status'] == 'PUSH_ACCEPTED'
    review = json.loads((RUN / 'reproduction_review.json').read_text())
    assert sha(RUN / 'reproduction_review.json') == launch['review_sha256']
    for relative, expected in review['source_sha256'].items():
        assert sha(ROOT / relative) == expected, relative
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    record = {'observed_utc': datetime.now(timezone.utc).isoformat(),
              'kernel': launch['kernel'], 'gpu_run_requested': False,
              'competition_submission_enabled': False}
    try:
        api = KaggleApi(); api.authenticate(); signal.alarm(45)
        record['remote_status'] = remote_status_name(api.kernels_status(launch['kernel']))
        if record['remote_status'] in TERMINAL:
            signal.alarm(60)
            dest = RUN / 'remote_source'
            api.kernels_pull(launch['kernel'], str(dest), metadata=True, quiet=True)
            meta = json.loads((dest / 'kernel-metadata.json').read_text())
            book = json.loads((dest / meta['code_file']).read_text())
            cells = [c for c in book['cells'] if c['cell_type'] == 'code']
            assert len(cells) == 1
            code = cells[0]['source']; code = ''.join(code) if isinstance(code, list) else code
            assert hashlib.sha256(code.encode()).hexdigest() == launch['code_cell_sha256']
            log = api.kernels_logs(launch['kernel']) or ''
            (RUN / 'remote_execution.log').write_text(log)
            token, seen, files = None, set(), []
            for _ in range(100):
                signal.alarm(60)
                names, next_token = api.kernels_output(launch['kernel'], str(RUN / 'output'),
                    file_pattern=PATTERN, force=True, quiet=True, page_token=token, page_size=100)
                files.extend(names)
                if not next_token: break
                assert next_token not in seen; seen.add(next_token); token = next_token
            else:
                raise AssertionError('Output pagination did not finish')
            record['collected_files'] = len(files)
            record['source_matches_accepted'] = True
            index = RUN / 'output/mars_latest_attempt.json'
            if index.exists():
                marker = json.loads(index.read_text())
                name = marker['attempt_directory']
                assert Path(name).name == name and name.startswith('mars_resource_attempt_')
                output = RUN / 'output' / name
                record['attempt_directory'] = name
                if record['remote_status'] == 'COMPLETE':
                    reviewed = audit(output)
                    (RUN / 'execution_audit.json').write_text(json.dumps(reviewed, indent=2) + '\n')
                    record['execution_audit_status'] = reviewed['status']
                else:
                    failures = {}
                    for relative in ['mars_resource_bootstrap_failure.json', 'mars_resource_pilot/failure.json']:
                        path = output / relative
                        if path.exists(): failures[relative] = json.loads(path.read_text())
                    record['failure_records'] = failures
            record['next_action'] = 'REVIEW_THIS_EXECUTION_BEFORE_ANY_TRAINING_OR_RETRY'
    except Exception as error:
        record['observation_error_type'] = type(error).__name__
    finally:
        signal.alarm(0)
        (RUN / 'observation.json').write_text(json.dumps(record, indent=2) + '\n')
        # Failure tracebacks are retained on disk; print compact stage/type only.
        display = dict(record)
        if 'failure_records' in display:
            display['failure_records'] = {n: {k: r.get(k) for k in ['stage', 'error_type', 'study_id']}
                                          for n, r in display['failure_records'].items()}
        print(json.dumps(display))


if __name__ == '__main__':
    main()
