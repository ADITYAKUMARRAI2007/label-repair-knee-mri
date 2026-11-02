"""One bounded reproduction of the hidden worker failure using the saved repair."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError
from audit_e04_resource_pilot import duration_seconds
from launch_e05_kaggle_pair import TERMINAL, inspect_missing_session
from observe_coherent_anchor_pair import remote_status_name

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'research/foundation_refresh_20261010/MARS'
RUN = BASE / 'repair_run'
PACKAGE = ROOT / 'public_kernels/mars_resource_restart_notebook'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(exist_ok=True)
    assert not (RUN / 'launch.json').exists(), 'Observe the accepted or uncertain attempt'
    meta = read(PACKAGE / 'kernel-metadata.json')
    saved, build = read(BASE / 'restart_notebook_remote_observation.json'), read(BASE / 'restart_patch_build.json')
    assert saved['source_matches_repair'] if 'source_matches_repair' in saved else saved['code_cell_sha256'] == build['source_sha256']
    assert saved['remote_status'] == 'COMPLETE' and saved['execution_count'] is None
    assert meta['id'] == saved['kernel'] and meta['kernel_type'] == 'notebook'
    assert build['numerical_modules_unchanged'] and read(BASE / 'restart_patch_check.json')['status'].startswith('PASS_')
    controls = ['scripts/launch_mars_repair_reproduction.py', 'scripts/observe_mars_repair_run.py',
                'scripts/audit_mars_repair_run.py', 'scripts/observe_coherent_anchor_pair.py',
                'scripts/launch_e05_kaggle_pair.py', 'scripts/audit_e04_resource_pilot.py',
                'public_kernels/mars_resource_restart_notebook/pilot.ipynb',
                'public_kernels/mars_resource_restart_notebook/kernel-metadata.json']
    review = {
        'decision': 'ONE_BOUNDED_DIAGNOSTIC_REPRODUCTION',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'reason': 'The committed V2 run is terminal with empty logs; interactive worker CalledProcessError has no accessible traceback. Reproduce once using the tested directory repair and visible worker logs to obtain the actual error or resource measurements.',
        'change_from_previous_next_action': 'Original interactive log remains requested; a bounded reproduction supplies the missing evidence without depending on manual log transfer.',
        'maximum_new_runs': 1, 'platform_timeout_seconds': 1260,
        'source_sha256': {name: sha(ROOT / name) for name in controls},
        'code_cell_sha256': build['source_sha256'],
        'protocol_sha256': sha(BASE / 'resource_protocol_v2.json'),
        'source_and_restart_checks_sha256': sha(BASE / 'restart_patch_check.json'),
        'previous_diagnosis_sha256': sha(BASE / 'resource_cancellation_diagnosis.json'),
        'previous_result_review_sha256': sha(ROOT / 'runs/window_integrity_20261010/pair_followup_review.json'),
        'classifier_fitting_enabled': False, 'competition_submission_enabled': False,
        'automatic_retry_or_corpus_continuation': False,
    }
    review_path = RUN / 'reproduction_review.json'
    if review_path.exists():
        old = read(review_path)
        assert old['source_sha256'] == review['source_sha256']
        review = old
    else:
        review_path.write_text(json.dumps(review, indent=2) + '\n')
    if not args.launch:
        print(json.dumps({'status': 'LOCAL_REPRODUCTION_REVIEW_READY', 'gpu_launched': False})); return
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(450)
    api = KaggleApi(); api.authenticate()
    assert remote_status_name(api.kernels_status('adityakumarrai0401/rsna-knee-mars-resource-pilot-v2')) == 'CANCEL_ACKNOWLEDGED'
    assert remote_status_name(api.kernels_status(meta['id'])) == 'COMPLETE'
    history = api.competition_submissions('rsna-knee-abnormality-detection', page_size=100) or []
    assert all(s.status.name in {'COMPLETE', 'ERROR'} for s in history)
    owned = set()
    for page in range(1, 101):
        rows = api.kernels_list(mine=True, page=page, page_size=100) or []
        owned.update(r.ref for r in rows if r and r.ref)
        if len(rows) < 100: break
    else:
        raise AssertionError('Owned listing pagination did not terminate')
    assert meta['id'] in owned
    states = []
    for ref in sorted(owned):
        time.sleep(1.25)
        try:
            status = remote_status_name(api.kernels_status(ref))
            assert status in TERMINAL, f'Active or unknown notebook: {ref}'
            states.append({'ref': ref, 'status': status})
        except HTTPError as error:
            if error.response is None or error.response.status_code != 404: raise
            states.append({'ref': ref, **inspect_missing_session(api, ref)})
        if len(states) % 20 == 0:
            print(json.dumps({'existing_handles_checked': len(states), 'total': len(owned)}), flush=True)
    (RUN / 'owned_status_scan.json').write_text(json.dumps(states, indent=2) + '\n')
    for dataset in meta['dataset_sources']:
        assert api.dataset_status(dataset) == 'ready'
    q = api.quota_view().to_dict()['gpuQuota']; reserved = duration_seconds(q.get('timeReserved', '0s'))
    available = (duration_seconds(q['totalTimeAllowed']) - duration_seconds(q['timeUsed']) - reserved) / 3600
    assert reserved == 0 and available >= 1.7
    for name, expected in review['source_sha256'].items():
        assert sha(ROOT / name) == expected, name
    # CLI push uses SAVE_AND_RUN_ALL for this existing notebook. Keep its21minute cap.
    record = {'status': 'PUSH_ATTEMPT_REQUIRES_OBSERVATION', 'kernel': meta['id'],
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'code_cell_sha256': build['source_sha256'], 'review_sha256': sha(review_path),
              'platform_timeout_seconds': 1260, 'available_gpu_hours': available,
              'historical_status_limitation': 'Old404sessions do not prove inactivity; fresh zeroGPUreservation is also required',
              'classifier_fitting_enabled': False, 'competition_submission_enabled': False}
    receipt = RUN / 'launch.json'
    with receipt.open('x') as handle:
        json.dump(record, handle, indent=2); handle.write('\n')
    signal.alarm(60)
    try:
        response = api.kernels_push(str(PACKAGE), timeout='1260')
        record['push_response'] = response.to_dict()
        assert not response.error
        normalized = response.ref.removeprefix('https://www.kaggle.com/').lstrip('/').removeprefix('code/')
        assert normalized == meta['id']
        record.update(status='PUSH_ACCEPTED', version=response.version_number)
        assert record['version'] > 1
    except BaseException as error:
        record['observation_error_type'] = type(error).__name__
        raise
    finally:
        signal.alarm(0); receipt.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record))


if __name__ == '__main__':
    main()
