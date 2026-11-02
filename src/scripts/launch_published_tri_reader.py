"""Launch the staged public recipe, optionally alongside the authorized pair."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError

from audit_e04_resource_pilot import duration_seconds
from launch_e05_kaggle_pair import ROOT, TERMINAL, inspect_missing_session, sha

RUN = ROOT / 'runs/published_tri_reader'
PACKAGE = ROOT / 'public_kernels/published_tri_reader'
PAIR = ROOT / 'runs/e05_kaggle_pair4'


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--parallel-with-existing-pair', action='store_true',
                        help='User-authorized execution alongside E05fold4 only')
    args = parser.parse_args()
    assert not (RUN / 'launch.json').exists(), 'Observe the existing launch; do not restart'
    build = read(RUN / 'build.json')
    check = read(RUN / 'local_check.json')
    contract = read(RUN / 'contract.json')
    metadata = read(PACKAGE / 'kernel-metadata.json')
    assert sha(PACKAGE / metadata['code_file']) == build['code_sha256'] == check['code_sha256']
    assert sha(PACKAGE / 'kernel-metadata.json') == build['metadata_sha256']
    assert sha(RUN / 'contract.json') == build['contract_sha256'] == check['contract_sha256']
    assert check['status'] == 'PASS_LOCAL_PACKAGE_AND_FUSION_CHECKS'
    assert metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_internet']
    assert metadata['id'] == build['kernel'] and not contract['competition_submission_enabled']
    assert sha(ROOT / 'runs/tri_reader_20261010/result.json') == build['diagnostic_result_sha256']
    watcher = read(PAIR / 'watcher.json')
    pending = []
    if not args.parallel_with_existing_pair and (watcher['status'] != 'FINISHED' or watcher.get('result') != 'BOTH_FOLD4_SMALL_OUTPUT_AUDITS_PASSED'):
        pending.append('Existing E05fold4 pair must finish and pass both independent output audits')
    record = {'status': 'WAIT_FOR_EXISTING_PAIR' if pending else 'LOCAL_GATES_PASS',
              'pending': pending, 'observed_utc': datetime.now(timezone.utc).isoformat(),
              'gpu_launched': False, 'competition_submission_enabled': False}
    record['parallel_with_existing_pair'] = args.parallel_with_existing_pair
    (RUN / 'local_preflight.json').write_text(json.dumps(record, indent=2) + '\n')
    if not args.launch:
        print(json.dumps(record, indent=2))
        return
    assert not pending, 'Evaluate the existing paired output first'
    for arm in ([] if args.parallel_with_existing_pair else ['e04', 'e05_uniform']):
        subprocess.run([sys.executable, str(ROOT / 'scripts/audit_e04_fold_training.py'),
                        str(PAIR / f'output/{arm}_fold4'), '--experiment', arm,
                        '--output', str(RUN / f'prerequisite_{arm}_audit.json')],
                       check=True, stdout=subprocess.DEVNULL, timeout=120)
        audit = read(RUN / f'prerequisite_{arm}_audit.json')
        assert audit['status'] == 'PASS_PARTIAL_FOLDS_SMALL_OUTPUTS'
        assert len(audit['folds']) == 1 and audit['folds'][0]['fold'] == 4
    api = KaggleApi()
    api.authenticate()
    pair_launch = read(PAIR / 'launch.json')
    pair_ref = pair_launch['kernel']
    pair_status = api.kernels_status(pair_ref).to_dict()['status']
    assert pair_status == 'COMPLETE' or (args.parallel_with_existing_pair and pair_status in {'RUNNING', 'QUEUED'}), 'Unexpected existing pair status'
    owned, page = set(), 1
    while True:
        rows = api.kernels_list(mine=True, page=page, page_size=100) or []
        owned.update(row.ref for row in rows if row is not None and row.ref)
        if len(rows) < 100:
            break
        page += 1
        assert page <= 100
    assert metadata['id'] not in owned, 'Notebook already exists; inspect its existing run'
    states = []
    for ref in sorted(owned):
        time.sleep(1.25)
        try:
            status = api.kernels_status(ref).to_dict()['status']
            allowed_pair = args.parallel_with_existing_pair and ref == pair_ref and status in {'RUNNING', 'QUEUED'}
            assert status in TERMINAL or allowed_pair, 'An unexpected existing notebook is active: ' + ref
            states.append({'ref': ref, 'status': status})
        except HTTPError as e:
            if e.response is None or e.response.status_code != 404:
                raise
            states.append({'ref': ref, **inspect_missing_session(api, ref)})
    (RUN / 'owned_status_scan.json').write_text(json.dumps(states, indent=2) + '\n')
    # Ensure required checkpoint files remain listed on attached datasets.
    requirements = {
        metadata['dataset_sources'][0]: ['raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt',
                                         'raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt'],
        metadata['dataset_sources'][1]: ['cnxt_v0_fold0.pt', 'cnxt_v0_fold1.pt', 'cnxt_v0_fold2.pt'],
        metadata['dataset_sources'][2]: ['decoder_identity.json'],
    }
    for ref, expected in requirements.items():
        names, token, seen_tokens = set(), None, set()
        for _ in range(100):
            listing = api.dataset_list_files(ref, page_token=token, page_size=100)
            assert not listing.error_message, 'Dataset listing failed: ' + ref
            names.update(Path(f.name).name for f in listing.dataset_files if f is not None)
            if set(expected) <= names or not listing.next_page_token:
                break
            token = listing.next_page_token
            assert token not in seen_tokens, 'Repeated pagination token'
            seen_tokens.add(token)
        assert set(expected) <= names, 'Required input file missing from ' + ref
    quota = api.quota_view().to_dict()
    gpu = quota['gpuQuota']
    reserved_hours = duration_seconds(gpu.get('timeReserved', '0s')) / 3600
    if not args.parallel_with_existing_pair:
        assert reserved_hours == 0
    # Retain enough allocated time for the existing run even if the API reports
    # zero reservation while that run is active.
    remaining_pair_hours = 0.0
    if pair_status in {'RUNNING', 'QUEUED'}:
        started = datetime.fromisoformat(pair_launch['observed_utc'])
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        remaining_pair_hours = max(0.0, (pair_launch['platform_timeout_seconds'] - elapsed) / 3600)
    committed_hours = max(reserved_hours, remaining_pair_hours)
    available = (duration_seconds(gpu['totalTimeAllowed'])-duration_seconds(gpu['timeUsed']))/3600 - committed_hours
    assert available >= contract['free_gpu_required_hours'], 'Insufficient allocated GPU quota'
    submissions = api.competition_submissions('rsna-knee-abnormality-detection', page_size=100)
    assert all(s.status.name in {'COMPLETE', 'ERROR'} for s in submissions), 'A competition submission is active'
    receipt = {'status': 'PUSH_ATTEMPT_REQUIRES_OBSERVATION', 'kernel': metadata['id'],
               'observed_utc': datetime.now(timezone.utc).isoformat(), 'build_sha256': sha(RUN / 'build.json'),
               'code_sha256': build['code_sha256'], 'launcher_sha256': sha(Path(__file__)),
               'available_gpu_hours': available, 'platform_timeout_seconds': 7800,
               'parallel_with_existing_pair': args.parallel_with_existing_pair,
               'existing_pair_kernel': pair_ref, 'existing_pair_status': pair_status,
               'existing_committed_gpu_hours': committed_hours,
               'user_authorization': 'Run the new approach in parallel' if args.parallel_with_existing_pair else None,
               'historical_session_limit': 'Historical404s remain unresolved;zero reservation alone cannot prove inactivity',
               'competition_submission_enabled': False}
    with (RUN / 'launch.json').open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    response = api.kernels_push(str(PACKAGE), timeout='7800')
    error = bool(getattr(response, 'error', None))
    server_ref = getattr(response, 'ref', None) or metadata['id']
    kernel_ref = server_ref.removeprefix('https://www.kaggle.com/').removeprefix('/').removeprefix('code/')
    assert len(kernel_ref.split('/')) == 2, 'Unexpected accepted kernel reference; inspect receipt before retrying'
    receipt.update(status='PUSH_REJECTED' if error else 'PUSH_ACCEPTED',
                   requested_kernel=metadata['id'], kernel=kernel_ref, server_ref=server_ref,
                   version=getattr(response, 'version_number', None), error_present=error)
    (RUN / 'launch.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
    assert not error, 'Launch rejected; inspect the existing receipt before retrying'


if __name__ == '__main__':
    main()
