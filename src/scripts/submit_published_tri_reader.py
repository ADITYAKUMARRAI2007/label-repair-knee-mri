"""Submit exactly one qualified notebook version; preserve handles on uncertainty."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

from audit_e04_resource_pilot import duration_seconds
from audit_published_tri_reader import ROOT, RUN, PACKAGE, audit, sha

COMPETITION = 'rsna-knee-abnormality-detection'
DESCRIPTION = 'N04 public tri reader crop30 fixed published routing verified pipeline no private priors'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--submit', action='store_true')
    p.add_argument('--parallel-with-existing-pair', action='store_true',
                   help='User-authorized submission while E05fold4 is running')
    args = p.parse_args()
    path = RUN / 'submission.json'
    assert not path.exists(), 'Submission attempt exists;inspect its handle and account history'
    pending = [name for name in ['launch.json', 'independent_audit.json'] if not (RUN / name).exists()]
    if pending:
        assert not args.submit, 'GPU execution and its independent audit are required first'
        print(json.dumps({'status': 'WAIT_FOR_EXECUTED_NOTEBOOK', 'missing': pending,
                          'competition_submission_made': False}))
        return
    launch = read(RUN / 'launch.json')
    assert launch['status'] == 'PUSH_ACCEPTED'
    saved = read(RUN / 'independent_audit.json')
    fresh = audit(RUN / 'output')
    assert fresh == saved, 'Executed artifacts or the independent audit changed'
    assert fresh['status'] == 'PASS_VISIBLE_TRI_READER_EXECUTION_ONLY'
    assert fresh['code_sha256'] == launch['code_sha256']
    if not args.submit:
        print(json.dumps({'status': 'LOCAL_SUBMISSION_GATES_PASS', 'version': launch['version'],
                          'competition_submission_made': False}))
        return
    api = KaggleApi()
    api.authenticate()
    assert api.kernels_status(launch['kernel']).to_dict()['status'] == 'COMPLETE'
    pair = read(ROOT / 'runs/e05_kaggle_pair4/launch.json')
    pair_status = api.kernels_status(pair['kernel']).to_dict()['status']
    if args.parallel_with_existing_pair:
        assert launch.get('parallel_with_existing_pair'), 'Notebook was not authorized for parallel execution'
    assert pair_status == 'COMPLETE' or (args.parallel_with_existing_pair and pair_status in {'RUNNING', 'QUEUED'}), 'Unexpected existing training status'
    # Bind submission to the executed immutable version and actual server source.
    owner, slug = launch['kernel'].split('/')
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = owner, slug
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(request)
    server = response.metadata
    assert server.ref == launch['kernel'] and server.current_version_number == launch['version']
    remote = RUN / 'submission_source_check'
    remote.mkdir(exist_ok=True)
    (remote / 'source.py').write_text(response.blob.source)
    assert sha(remote / 'source.py') == launch['code_sha256']
    expected_meta = read(PACKAGE / 'kernel-metadata.json')
    assert server.enable_gpu and not server.enable_internet
    assert server.docker_image == expected_meta['docker_image']
    assert sorted(server.dataset_data_sources) == sorted(expected_meta['dataset_sources'])
    assert sorted(server.competition_data_sources) == sorted(expected_meta['competition_sources'])
    history = api.competition_submissions(COMPETITION, page_size=100) or []
    assert not any(s.description == DESCRIPTION for s in history), 'Existing matching submission;reconcile it'
    assert all(s.status.name in {'COMPLETE', 'ERROR'} for s in history), 'A competition submission is active'
    limits = api.competition_get_submission_limits(COMPETITION).to_dict(ignore_defaults=False)
    assert limits['numAllowedNow'] > 0, 'No remaining submission slot'
    quota = api.quota_view().to_dict()
    gpu = quota['gpuQuota']
    reserved_hours = duration_seconds(gpu.get('timeReserved', '0s')) / 3600
    remaining_pair_hours = 0.0
    if pair_status in {'RUNNING', 'QUEUED'}:
        elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(pair['observed_utc'])).total_seconds()
        remaining_pair_hours = max(0.0, (pair['platform_timeout_seconds'] - elapsed) / 3600)
    committed_hours = max(reserved_hours, remaining_pair_hours)
    free = (duration_seconds(gpu['totalTimeAllowed'])-duration_seconds(gpu['timeUsed']))/3600 - committed_hours
    contract = read(RUN / 'contract.json')
    assert free >= contract['free_gpu_required_hours'], 'Insufficient allocated GPU quota'
    receipt = {'status': 'ATTEMPT_REQUIRES_OBSERVATION',
               'observed_utc': datetime.now(timezone.utc).isoformat(),
               'kernel': launch['kernel'], 'kernel_version': launch['version'],
               'description': DESCRIPTION, 'file_name': 'submission.csv',
               'independent_audit_sha256': sha(RUN / 'independent_audit.json'),
               'source_check_sha256': sha(remote / 'source.py'),
               'existing_submission_ids': [s.ref for s in history],
               'free_gpu_hours_before': free, 'limits_before': limits,
               'parallel_with_existing_pair': args.parallel_with_existing_pair,
               'existing_pair_status': pair_status, 'existing_committed_gpu_hours': committed_hours,
               'public_score': None, 'score_guarantee': False,
               'private_author0954_method_reproduced': False}
    with path.open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
    # A timeout after this call must be reconciled via history,never blindly retried.
    result = api.competition_submit_code(file_name='submission.csv', message=DESCRIPTION,
                                         competition=COMPETITION, kernel=launch['kernel'],
                                         kernel_version=launch['version'], quiet=True)
    receipt.update(submission_id=result.ref,
                   status='ACCEPTED_PENDING_EVALUATION' if result.ref else 'RESPONSE_REQUIRES_RECONCILIATION')
    write(path, receipt)
    print(json.dumps({k: receipt[k] for k in ['status', 'submission_id', 'kernel', 'kernel_version']}, indent=2))


if __name__ == '__main__':
    main()
