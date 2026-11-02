"""Launch one reviewed MARS corpus/classifier stage, preserving uncertain pushes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
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
STAGES = {
    'corpus': ('corpus_resource_review.json', 'corpus_build.json', 'mars_full_corpus',
               'APPROVE_MEASURED_MARS_FULL_FEATURE_EXTRACTION'),
    'classifier': ('classifier_resource_approval.json', 'classifier_build.json',
                   'mars_frozen_classifier', 'APPROVE_COMPLETE_FROZEN_FEATURE_CLASSIFIER'),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def check_dataset_inputs(api, meta):
    """Check owned upload readiness and readable files in public label inputs."""
    owner = meta['id'].split('/')[0]
    expected_files = {}
    frozen = read(ROOT / 'runs/E04_contract/e04_contract.json')
    for receipt in frozen['fold_reconstruction_only_sources'].values():
        path = Path(receipt['path'])
        source_owner, slug = path.parent.name.split('_', 1)
        expected_files.setdefault(source_owner + '/' + slug, set()).add(path.name)
    checks = []
    for dataset in meta['dataset_sources']:
        if dataset.split('/')[0] == owner:
            assert api.dataset_status(dataset) == 'ready'
            checks.append({'ref': dataset, 'check': 'OWNED_UPLOAD_READY'})
            continue
        # GetDatasetStatus returns404 for these readable third-party sources.
        # Verify their required files through the reader endpoint instead.
        required = expected_files[dataset]
        found, tokens, token = {}, set(), None
        for _ in range(100):
            response = api.dataset_list_files(dataset, page_token=token, page_size=100)
            assert not response.error_message, dataset
            for item in response.files:
                name = Path(item.name).name
                if name in required:
                    assert name not in found and item.total_bytes > 0, dataset
                    found[name] = item.total_bytes
            token = response.next_page_token
            if not token:
                break
            assert token not in tokens, 'Dataset file pagination repeated'
            tokens.add(token)
        else:
            raise AssertionError('Dataset file listing did not terminate')
        assert set(found) == required, f'Required source files unavailable: {dataset}'
        checks.append({'ref': dataset, 'check': 'READABLE_EXPECTED_SOURCE_FILES',
                       'files': found, 'byte_hashes_rechecked_natively_before_updates': True})
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=STAGES, required=True)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    approval_name, build_name, package_name, decision = STAGES[args.stage]
    approval_path, build_path = BASE / approval_name, BASE / build_name
    launch_path = BASE / f'{args.stage}_launch.json'
    assert not launch_path.exists(), 'Observe the accepted or uncertain run; do not duplicate it'
    if not approval_path.exists() or not build_path.exists():
        print(json.dumps({'status': 'WAIT_FOR_MEASURED_REVIEW_AND_BUILT_STAGE',
                          'stage': args.stage, 'gpu_launched': False})); return
    approval, build = read(approval_path), read(build_path)
    assert approval['decision'] == decision
    review_key = 'review_sha256' if args.stage == 'corpus' else 'resource_approval_sha256'
    assert sha(approval_path) == build[review_key]
    package = ROOT / 'public_kernels' / package_name
    for name, key in [('pilot.ipynb', 'notebook_sha256'),
                      ('kernel-metadata.json', 'metadata_sha256')]:
        assert sha(package / name) == build[key], name
    meta = read(package / 'kernel-metadata.json')
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert meta['competition_sources'] == ['rsna-knee-abnormality-detection']
    timeout = int(approval['platform_timeout_seconds'])
    assert 0 < approval['maximum_parent_seconds'] < timeout
    allocated_devices = int(approval['allocated_gpu_count'])
    # This execution path uses the already measured two-T4 Kaggle image.
    assert meta['machine_shape'] == 'NvidiaTeslaT4' and allocated_devices == 2
    reserve = float(approval['reserve_gpu_hours'])
    assert math.isfinite(reserve) and reserve >= 0
    required_hours = timeout * allocated_devices / 3600 + reserve
    if not args.launch:
        print(json.dumps({'status': 'REVIEWED_STAGE_PACKAGED_NOT_LAUNCHED', 'stage': args.stage,
                          'required_free_gpu_hours': required_hours, 'gpu_launched': False})); return
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
    signal.alarm(450)
    api = KaggleApi(); api.authenticate()
    # The measured pilot must be terminal and independently audited, not merely saved.
    assert remote_status_name(api.kernels_status(read(BASE / 'cuda_fix_run/launch.json')['kernel'])) == 'COMPLETE'
    assert read(BASE / 'cuda_fix_run/execution_audit.json')['status'] == 'PASS_REAL_DICOM_RESOURCE_OUTPUT_IDENTITY_AND_FEATURE_ARRAYS'
    if args.stage == 'classifier':
        assert read(BASE / 'corpus_execution_audit.json')['status'] == 'PASS_COMPLETE_NATIVE_CORPUS_EXECUTION_AND_COVERAGE'
    for source in meta['kernel_sources']:
        assert remote_status_name(api.kernels_status(source)) == 'COMPLETE'
    dataset_checks = check_dataset_inputs(api, meta)
    (BASE / f'{args.stage}_dataset_preflight.json').write_text(
        json.dumps(dataset_checks, indent=2) + '\n')
    submissions = api.competition_submissions('rsna-knee-abnormality-detection', page_size=100) or []
    assert all(s.status.name in {'COMPLETE', 'ERROR'} for s in submissions)
    owned = set()
    for page in range(1, 101):
        rows = api.kernels_list(mine=True, page=page, page_size=100) or []
        owned.update(r.ref for r in rows if r and r.ref)
        if len(rows) < 100: break
    else:
        raise AssertionError('Owned notebook listing did not terminate')
    assert meta['id'] not in owned, 'Stage slug already exists; investigate its original run'
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
            print(json.dumps({'owned_handles_checked': len(states), 'stage': args.stage}), flush=True)
    (BASE / f'{args.stage}_owned_scan.json').write_text(json.dumps(states, indent=2) + '\n')
    quota = api.quota_view().to_dict()['gpuQuota']
    reserved = duration_seconds(quota.get('timeReserved', '0s'))
    free = (duration_seconds(quota['totalTimeAllowed']) - duration_seconds(quota['timeUsed']) - reserved) / 3600
    assert reserved == 0 and free >= required_hours, 'Measured allocation exceeds current free quota'
    assert sha(approval_path) == build[review_key]
    assert sha(package / 'pilot.ipynb') == build['notebook_sha256']
    assert sha(package / 'kernel-metadata.json') == build['metadata_sha256']
    record = {'status': 'PUSH_ATTEMPT_REQUIRES_OBSERVATION', 'kernel': meta['id'],
              'stage': args.stage, 'observed_utc': datetime.now(timezone.utc).isoformat(),
              'code_cell_sha256': build['code_cell_sha256'], 'build_sha256': sha(build_path),
              'review_sha256': sha(approval_path), 'platform_timeout_seconds': timeout,
              'allocated_gpu_count': allocated_devices, 'available_gpu_hours': free,
              'required_gpu_hours_including_reserve': required_hours,
              'historical_status_limitation': 'Old404does not prove inactivity; zero reservation also required',
              'competition_submission_enabled': False}
    with launch_path.open('x') as handle:
        json.dump(record, handle, indent=2); handle.write('\n')
    signal.alarm(60)
    try:
        response = api.kernels_push(str(package), timeout=str(timeout))
        if response.error:
            raise RuntimeError('Kaggle rejected the reviewed stage push')
        normalized = response.ref.removeprefix('https://www.kaggle.com/').lstrip('/').removeprefix('code/')
        assert normalized == meta['id']
        record.update(status='PUSH_ACCEPTED', version=response.version_number)
    except BaseException as error:
        record['observation_error_type'] = type(error).__name__
        raise
    finally:
        signal.alarm(0)
        launch_path.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record))


if __name__ == '__main__':
    main()
