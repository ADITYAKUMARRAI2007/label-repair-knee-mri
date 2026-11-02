"""Launch the staged fold4 pair once, after completion, identity and quota gates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from requests import HTTPError

from audit_e04_resource_pilot import duration_seconds

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/e05_kaggle_pair4'
TERMINAL = {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED', 'CANCELLED', 'CANCELED'}


def inspect_missing_session(api, ref):
    """Record an unavailable old session without claiming it is terminal."""
    owner, slug = ref.split('/', 1)
    request = ApiGetKernelRequest(); request.user_name = owner; request.kernel_slug = slug
    with api.build_kaggle_client() as client:
        metadata = client.kernels.kernels_api_client.get_kernel(request).metadata.to_dict()
    # Metadata accelerator settings describe the notebook, not a live allocation.
    last = datetime.fromisoformat(metadata['lastRunTime'].replace('Z', '+00:00'))
    assert metadata['ref'] == ref and last.tzinfo is not None
    age = (datetime.now(timezone.utc) - last).total_seconds()
    assert age >= 48 * 3600, 'A recent session is unavailable; inspect it before launching'
    return {'status': 'HISTORICAL_SESSION_UNAVAILABLE', 'http_status': 404,
            'metadata_last_run_time': metadata['lastRunTime'],
            'enable_gpu': metadata.get('enableGpu'), 'enable_tpu': metadata.get('enableTpu'),
            'metadata_age_hours': age / 3600, 'terminal_state_proven': False}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def verify_local():
    assert not (RUN / 'launch.json').exists(), 'Observe the original launch receipt'
    build = read(RUN / 'build.json')
    check = read(RUN / 'local_check.json')
    assert build['status'] == 'STAGED_NOT_LAUNCHED' and build['paired_original_control']
    assert build['fold'] == 4 and not build['competition_submission_enabled']
    assert build['job_budget_hours'] == 7.25 and build['platform_timeout_seconds'] == 27900
    package = ROOT / build['package']
    for name, expected in build['sources'].items():
        assert sha(ROOT / name) == expected, 'Package or upstream source changed: ' + name
    assert sha(ROOT / 'runs/e05_confidence_ablation/protocol.json') == build['protocol_sha256']
    assert check['status'] == 'PASS_PAIRED_WORKER_DISPATCH'
    assert check['package_sha256'] == sha(package / 'train.py')
    assert check['separate_gpu_and_output_directories']
    metadata = read(package / 'kernel-metadata.json')
    assert metadata['id'] == build['kernel'] and metadata['is_private']
    assert metadata['enable_gpu'] and not metadata['enable_internet']
    assert metadata['competition_sources'] == ['rsna-knee-abnormality-detection']
    assert metadata['machine_shape'] == 'NvidiaTeslaT4'
    # A paired run replaces both standalone fold4 jobs, never supplements them.
    duplicates = ['runs/e04_folds_4_launch.json', 'runs/e05_full_fold4/launch.json',
                  'runs/e05_kaggle_fold4/launch.json']
    assert not any((ROOT / p).exists() for p in duplicates), 'A fold4 launch already exists'
    pending = []
    for fold in [0, 1]:
        folder = ROOT / f'runs/e05_full_fold{fold}'
        watcher = read(folder / 'watcher.json')
        if not (watcher['status'] == 'FINISHED' and watcher.get('result') == 'COMPLETE'
                and watcher.get('fold_audit_passed')):
            pending.append(f'Colab fold{fold} completion and independent audit')
            continue
        integrity = read(folder / 'verified_final/integrity.json')
        audit = read(folder / 'fold_audit.json')
        terminal = read(folder / 'verified_terminal/terminal.json')
        assert integrity['status'] == 'PASS_REMOTE_SNAPSHOT_BYTES'
        assert integrity['all_member_bytes_independently_verified']
        assert audit['status'] == 'PASS_PARTIAL_FOLDS_SMALL_OUTPUTS'
        assert len(audit['folds']) == 1 and audit['folds'][0]['fold'] == fold
        assert terminal['job_result']['status'] == 'COMPLETE'
        assert terminal['job_result']['completed_epoch'] == 14
    for folder in sorted((ROOT / 'runs').glob('e05_full_fold[2-4]')):
        if (folder / 'launch.json').exists():
            watcher = folder / 'watcher.json'
            if not watcher.exists() or read(watcher)['status'] != 'FINISHED':
                pending.append(f'Existing {folder.name} run must finish first')
    return build, package, metadata, pending


def quota(api, required):
    value = api.quota_view().to_dict()
    gpu = value['gpuQuota']
    available = (duration_seconds(gpu['totalTimeAllowed']) - duration_seconds(gpu['timeUsed'])
                 - duration_seconds(gpu.get('timeReserved', '0s'))) / 3600
    receipt = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'quota': value,
               'available_gpu_hours': available, 'required_hours_including_reserve': required}
    (RUN / 'quota_preflight.json').write_text(json.dumps(receipt, indent=2) + '\n')
    assert available >= required, 'Insufficient already allocated GPU quota'
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true', help='Push one private paired training notebook')
    args = parser.parse_args()
    build, package, metadata, pending = verify_local()
    preflight = {'status': 'LOCAL_PREFLIGHT', 'observed_utc': datetime.now(timezone.utc).isoformat(),
                 'source_and_dispatch_checks_passed': True, 'pending': pending,
                 'launcher_sha256': sha(Path(__file__)), 'gpu_launched': False,
                 'competition_submission_enabled': False}
    (RUN / 'local_preflight.json').write_text(json.dumps(preflight, indent=2) + '\n')
    if not args.launch:
        print(json.dumps(preflight)); return
    assert not pending, 'Finish the existing run and audits before launching the pair'
    api = KaggleApi(); api.authenticate()
    # Preserve one additional hour beyond the 7.75-hour platform bound.
    required = build['platform_timeout_seconds'] / 3600 + 1.0
    quota(api, required)
    submissions = api.competition_submissions('rsna-knee-abnormality-detection', page_size=100)
    assert all(s.status.name in {'COMPLETE', 'ERROR'} for s in submissions), 'Another submission is active'
    owned, page = set(), 1
    while True:
        rows = api.kernels_list(mine=True, page=page, page_size=100) or []
        owned.update(r.ref for r in rows if r is not None and r.ref)
        if len(rows) < 100: break
        page += 1
        assert page <= 100, 'Owned-kernel pagination did not terminate'
    assert 'adityakumarrai0401/rsna-knee-e04-wide-cache-base' in owned
    assert set(metadata['kernel_sources']) <= owned, 'Owned input cache is missing from the listing'
    assert metadata['id'] not in owned, 'Pair already exists; observe its original run'
    assert 'adityakumarrai0401/rsna-knee-e04-convnext-folds-4' not in owned
    assert 'adityakumarrai0401/rsna-knee-e05-uniform-fold-4' not in owned
    states = []
    for ref in sorted(owned):
        (RUN / 'scan_progress.json').write_text(json.dumps({'ref': ref, 'checked': len(states),
            'total': len(owned), 'observed_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
        missing_session = None
        for attempt in range(4):
            time.sleep(1.25)
            try:
                status = api.kernels_status(ref).to_dict()['status']
                break
            except HTTPError as error:
                if error.response is not None and error.response.status_code == 404:
                    missing_session = inspect_missing_session(api, ref)
                    status = missing_session['status']
                    break
                if error.response is None or error.response.status_code != 429 or attempt == 3:
                    raise
                print(json.dumps({'rate_limited': ref, 'retry_same_handle': True}), flush=True)
                time.sleep(30)
        states.append({'ref': ref, 'status': status, **({'missing_session_session': missing_session} if missing_session else {})})
        assert status in TERMINAL or missing_session is not None, f'An owned notebook is active or unknown: {ref} {status}'
        if ref in metadata['kernel_sources']:
            assert status == 'COMPLETE', 'Mounted cache is not complete'
        if len(states) % 20 == 0:
            print(json.dumps({'checked_terminal': len(states), 'total': len(owned)}), flush=True)
    (RUN / 'owned_status_scan.json').write_text(json.dumps(states, indent=2) + '\n')
    fresh = quota(api, required)
    if any(s.get('missing_session_session', {}).get('enable_gpu') for s in states):
        # A 404 alone cannot establish inactivity. Also require a fresh quota
        # observation with no running-GPU reservation; retain the limitation.
        assert duration_seconds(fresh['quota']['gpuQuota'].get('timeReserved', '0s')) == 0, \
            'GPU quota is reserved while a historical session is unavailable'
    _, _, _, pending = verify_local()
    assert not pending
    receipt = {'status': 'PUSH_ATTEMPT_REQUIRES_OBSERVATION', 'kernel': metadata['id'],
               'observed_utc': datetime.now(timezone.utc).isoformat(),
               'build_sha256': sha(RUN / 'build.json'), 'launcher_sha256': sha(Path(__file__)),
               'code_sha256': sha(package / 'train.py'),
               'available_gpu_hours': fresh['available_gpu_hours'],
               'platform_timeout_seconds': build['platform_timeout_seconds'],
               'historical_session_limit': '404 sessions are unresolved; fresh zero GPU reservation required when any uses GPU',
               'competition_submission_enabled': False}
    # Exclusive creation also prevents two launcher processes from both pushing.
    path = RUN / 'launch.json'
    with path.open('x') as handle:
        json.dump(receipt, handle, indent=2); handle.write('\n')
    response = api.kernels_push(str(package), timeout=str(build['platform_timeout_seconds']))
    error = str(getattr(response, 'error', None) or '')
    receipt.update(status='PUSH_REJECTED' if error else 'PUSH_ACCEPTED',
                   error=error, version=getattr(response, 'version_number', None))
    actual_ref = getattr(response, 'ref', None)
    if actual_ref:
        receipt.update(requested_kernel=metadata['id'], kernel=actual_ref,
                       kernel_id=getattr(response, 'kernel_id', None),
                       notebook_url=getattr(response, 'url', None))
    path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))
    if error: raise RuntimeError('Kaggle rejected the paired training job')


if __name__ == '__main__':
    main()
