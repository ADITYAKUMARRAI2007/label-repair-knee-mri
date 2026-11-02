"""Submit one audited, explicitly versioned N02 fixed blend; never retry blindly."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inference'))
sys.path.insert(0, str(ROOT / 'scripts'))
from nartaa_reproduction_guard import sha, validate_submission
from audit_e04_resource_pilot import duration_seconds

OUT = ROOT / 'runs/fixed_reader_blend_v3'
DESCRIPTION = 'N02 recovery repair fixed N01 ConvNeXt k16 blend pipeline qualified no weight tuning'
COMPETITION = 'rsna-knee-abnormality-detection'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main():
    receipt_path = OUT / 'submission.json'
    assert not receipt_path.exists(), 'submission attempt exists: reconcile its existing handle instead of retrying'
    launch = json.loads((OUT / 'launch.json').read_text())
    build = json.loads((OUT / 'build.json').read_text())
    audit = json.loads((OUT / 'independent_audit.json').read_text())
    assert launch['status'] == 'PUSH_ACCEPTED' and audit['status'] == 'PASS_VISIBLE_EXECUTION_ONLY'
    pipeline = audit['pipeline_preflight']
    assert pipeline['status'] == 'PASS_PIPELINE_AND_RECOVERY' and pipeline['rows'] == 20
    assert len(pipeline['batches']) == 5 and pipeline['recovery_pixel_array_exact']
    assert not pipeline['labels_used'] and not pipeline['accuracy_measured']
    for relative, expected in {**build['sources'], **build['gates']}.items():
        assert sha(ROOT / relative) == expected, relative
    output = OUT / 'outputs'
    assert audit['receipt_sha256'] == sha(output / 'fixed_reader_blend/receipt.json')
    assert audit['submission'] == validate_submission(output / 'submission.csv', ROOT / 'data/test.csv',
                                                      ROOT / 'data/sample_submission.csv')
    api = KaggleApi()
    api.authenticate()
    assert api.kernels_status(launch['kernel']).to_dict(ignore_defaults=False)['status'] == 'COMPLETE'
    remote = OUT / 'submission_source_check'
    remote.mkdir(exist_ok=True)
    owner, slug = launch['kernel'].split('/')
    request = ApiGetKernelRequest()
    request.user_name = owner
    request.kernel_slug = slug
    with api.build_kaggle_client() as client:
        response = client.kernels.kernels_api_client.get_kernel(request)
    server = response.metadata
    assert server.ref == launch['kernel']
    assert server.current_version_number == launch['version'], 'latest version differs from audited version'
    metadata = dict(code_file='source.py', enable_gpu=server.enable_gpu, enable_internet=server.enable_internet,
                    docker_image=server.docker_image, dataset_sources=server.dataset_data_sources,
                    competition_sources=server.competition_data_sources, verified_version=server.current_version_number)
    (remote / 'source.py').write_text(response.blob.source)
    (remote / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    expected_meta = json.loads((ROOT / 'public_kernels/fixed_reader_blend_v3/kernel-metadata.json').read_text())
    assert sha(remote / metadata['code_file']) == sha(ROOT / 'public_kernels/fixed_reader_blend_v3' / expected_meta['code_file'])
    assert metadata['enable_gpu'] and not metadata['enable_internet']
    assert metadata['docker_image'] == expected_meta['docker_image']
    for key in ['dataset_sources', 'competition_sources']:
        assert sorted(metadata[key]) == sorted(expected_meta[key])
    history = api.competition_submissions(COMPETITION, page_size=100) or []
    assert not any(s.description == DESCRIPTION for s in history), 'matching submission exists: observe it'
    assert not any(str(s.status).split('.')[-1] in {'PENDING', 'RUNNING', 'QUEUED'} for s in history), 'another submission is pending'
    limits = api.competition_get_submission_limits(COMPETITION).to_dict(ignore_defaults=False)
    assert limits['numAllowedNow'] > 0, 'no submissions currently available'
    quota = api.quota_view().to_dict()
    gpu = quota['gpuQuota']
    free = (duration_seconds(gpu['totalTimeAllowed']) - duration_seconds(gpu['timeUsed'])
            - duration_seconds(gpu['timeReserved'])) / 3600
    assert free >= 3.1, 'retain 1h reserve alongside 2.1h bounded execution budget'
    receipt = dict(status='ATTEMPT_REQUIRES_OBSERVATION', observed_utc=datetime.now(timezone.utc).isoformat(),
                   kernel=launch['kernel'], kernel_version=launch['version'], description=DESCRIPTION,
                   file_name='submission.csv', independent_audit_sha256=sha(OUT / 'independent_audit.json'),
                   source_check_sha256=sha(remote / metadata['code_file']), limits_before=limits,
                   quota_before=quota, free_gpu_hours_before=free, existing_submission_ids=[s.ref for s in history],
                   author_reported_public_score='0.950', our_public_score=None)
    write(receipt_path, receipt)
    response = api.competition_submit_code(file_name='submission.csv', message=DESCRIPTION,
                                           competition=COMPETITION, kernel=launch['kernel'],
                                           kernel_version=launch['version'], quiet=True)
    receipt.update(response_message=response.message, submission_id=response.ref,
                   status='ACCEPTED_PENDING_EVALUATION' if response.ref else 'RESPONSE_REQUIRES_RECONCILIATION')
    write(receipt_path, receipt)
    assert response.ref > 0, 'response has no submission id: reconcile account history; do not resubmit'
    print(json.dumps({k: receipt[k] for k in ['status', 'kernel', 'kernel_version', 'submission_id', 'response_message']}, indent=2))


if __name__ == '__main__':
    main()
