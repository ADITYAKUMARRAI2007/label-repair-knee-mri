#!/usr/bin/env python3
"""Launch exactly one source-gated specialist diagnostic on free quota."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError
from audit_e04_resource_pilot import duration_seconds, require

ROOT = Path(__file__).resolve().parents[1]


def main():
    build_path = ROOT / 'runs/orthofoundation_probe_source.json'
    build = json.loads(build_path.read_text())
    require(build['status'] == 'BUILT_NOT_LAUNCHED', 'source gate missing')
    for relative, expected in {**build['sources'], **build['gates']}.items():
        require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected,
                'source/gate changed: ' + relative)
    previous = json.loads((ROOT / 'runs/spatial_pooling_probe_decision_v2.json').read_text())
    require(previous['status'] == 'REJECT_FOR_ADVANCEMENT', 'evaluate previous experiment first')
    directory = ROOT / 'public_kernels/orthofoundation_probe'
    metadata = json.loads((directory / 'kernel-metadata.json').read_text())
    require(metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_internet']
            and metadata['machine_shape'] == 'NvidiaTeslaT4', 'private compute metadata mismatch')
    contract = build['contract']
    require(not contract['competition_submission_enabled'] ,
            'diagnostic only')
    require(contract['trial']['allocated_gpu_hours_cap'] == 1.5, 'budget changed')
    receipt_path = ROOT / 'runs/orthofoundation_probe_launch.json'
    require(not receipt_path.exists(), 'receipt exists: observe the same handle, never duplicate push')
    api = KaggleApi()
    api.authenticate()
    require(api.kernels_status('adityakumarrai0401/rsna-knee-spatial-pooling-probe-v2').to_dict()['status'] == 'COMPLETE',
            'previous diagnostic not terminal')
    assets=json.loads((ROOT/'runs/orthofoundation_asset_upload.json').read_text())
    require(assets['status']=='UPLOAD_ACCEPTED' and assets['private'],'private assets not uploaded')
    dataset_state=api.dataset_status(assets['dataset'])
    require(dataset_state=='ready','asset dataset not ready: '+str(dataset_state))
    for source in metadata['kernel_sources']:
        require(api.kernels_status(source).to_dict()['status'] == 'COMPLETE', 'input notebook not complete: ' + source)
    owned, page = set(), 1
    while True:
        records = api.kernels_list(mine=True, page=page, page_size=100) or []
        owned.update(r.ref for r in records if r and r.ref)
        if len(records) < 100:
            break
        page += 1
        require(page <= 100, 'listing did not terminate')
    require('adityakumarrai0401/rsna-knee-b03-convnext-reader-blend' in owned, 'known own notebook absent')
    require(metadata['id'] not in owned, 'slug already exists; observe original version')
    def observe(ref):
        try:
            state = api.kernels_status(ref).to_dict()['status']
        except HTTPError as error:
            if error.response is None or error.response.status_code != 404:
                raise
            state = 'NO_COMMITTED_SESSION_404'
        return {'kernel': ref, 'status': state}
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(pool.map(observe, sorted(ref for ref in owned if '/rsna-knee-' in ref)))
    require(all(r['status'] in {'COMPLETE', 'ERROR', 'CANCELLED', 'NO_COMMITTED_SESSION_404'} for r in states),
            'another owned knee notebook is active')
    quota = api.quota_view().to_dict()
    gpu = quota['gpuQuota']
    free = (duration_seconds(gpu['totalTimeAllowed']) - duration_seconds(gpu['timeUsed'])
            - duration_seconds(gpu['timeReserved'])) / 3600
    observed = datetime.now(timezone.utc).isoformat()
    observation = {'observed_utc': observed, 'quota': quota, 'free_gpu_hours': free,
                   'owned_knee_states': states, 'new_job_cap_hours': 1.5, 'reserve_hours': 2}
    (ROOT / 'runs/orthofoundation_probe_quota_before.json').write_text(json.dumps(observation, indent=2) + '\n')
    require(free >= 3.5, 'quota must cover diagnostic plus two-hour reserve')
    receipt = {'status': 'PUSH_ATTEMPT_REQUIRES_OBSERVATION', 'kernel': metadata['id'],
               'observed_utc': observed, 'job_budget_hours': 1.5, 'allocated_free_hours': free,
               'competition_submission_enabled': False,
               'build_sha256': hashlib.sha256(build_path.read_bytes()).hexdigest()}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    response = api.kernels_push(str(directory))
    error = str(getattr(response, 'error', None) or '').strip()
    if error:
        receipt.update(status='PUSH_REJECTED', error=error)
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        raise RuntimeError(error)
    receipt.update(status='PUSH_ACCEPTED', version=response.version_number)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
