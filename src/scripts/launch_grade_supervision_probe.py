#!/usr/bin/env python3
"""Push one bounded private supervision comparison, preserving free quota."""
import hashlib
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError
from audit_e04_resource_pilot import duration_seconds, require

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=int, choices=[1,2], default=1)
    attempt = parser.parse_args().attempt
    suffix = '' if attempt == 1 else '_v2'
    build_path = ROOT / f'runs/grade_supervision_probe_source{suffix}.json'
    build = json.loads(build_path.read_text())
    require(build['status'] == 'BUILT_NOT_LAUNCHED', 'build not eligible')
    for relative, expected in {**build['sources'], **build['gates']}.items():
        require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected,
                f'source or gate changed: {relative}')
    directory = ROOT / f'public_kernels/grade_supervision_probe_v{attempt}'
    metadata = json.loads((directory / 'kernel-metadata.json').read_text())
    require(metadata['is_private'] and metadata['enable_gpu']
            and not metadata['enable_internet'] and not metadata['enable_tpu']
            and metadata['machine_shape'] == 'NvidiaTeslaT4', 'compute metadata mismatch')
    require(not build['contract']['competition_submission_enabled'], 'submission prohibited')
    cap = build['contract']['gpu_budget_hours_including_startup_margin']
    require(cap == 1.5, 'budget changed')
    receipt_path = ROOT / f'runs/grade_supervision_probe_launch{suffix}.json'
    require(not receipt_path.exists(), 'receipt exists: observe original job; do not push again')
    api = KaggleApi()
    api.authenticate()
    if attempt == 2:
        original = json.loads((ROOT / 'runs/grade_supervision_probe_launch.json').read_text())
        require(api.kernels_status(original['kernel']).to_dict()['status'] == 'ERROR', 'original must be terminal ERROR')
        failure = json.loads((ROOT / 'runs/grade_supervision_probe_outputs/grade_probe/failure.json').read_text())
        require(failure['message'] == 'host memory guard', 'retry only for documented memory failure')
        old_build = json.loads((ROOT / 'runs/grade_supervision_probe_source.json').read_text())
        require(build['contract'] == old_build['contract'], 'scientific contract changed during repair')
    for source in metadata['kernel_sources']:
        require(api.kernels_status(source).to_dict()['status'] == 'COMPLETE',
                f'input not complete: {source}')
    owned, page = {}, 1
    while True:
        records = api.kernels_list(mine=True, page=page, page_size=100) or []
        owned.update({r.ref: r for r in records if r is not None and r.ref})
        if len(records) < 100:
            break
        page += 1
        require(page <= 100, 'listing did not terminate')
    require(metadata['kernel_sources'][0] in owned, 'known owned cache missing')
    require(metadata['id'] not in owned, 'slug exists: observe original job')
    def observe(ref):
        try:
            state = api.kernels_status(ref).to_dict()['status']
        except HTTPError as error:
            if error.response is None or error.response.status_code != 404:
                raise
            state = 'NO_COMMITTED_SESSION_404'
        return {'kernel': ref, 'status': state}
    refs = sorted(ref for ref in owned if '/rsna-knee-' in ref)
    with ThreadPoolExecutor(max_workers=4) as pool:
        states = list(pool.map(observe, refs))
    # Older uncommitted notebooks may have no status endpoint. Quota below
    # still deducts all account reservations, including unrelated workloads.
    require(all(r['status'] in {'COMPLETE', 'ERROR', 'CANCELLED', 'NO_COMMITTED_SESSION_404'} for r in states),
            'another owned knee job is active or has an unknown state')
    quota = api.quota_view()
    quota = quota.to_dict() if hasattr(quota, 'to_dict') else quota
    gpu = quota['gpuQuota']
    free = (duration_seconds(gpu['totalTimeAllowed']) - duration_seconds(gpu['timeUsed'])
            - duration_seconds(gpu['timeReserved'])) / 3600
    observed = datetime.now(timezone.utc).isoformat()
    observation = {'observed_utc': observed, 'quota': quota, 'allocated_free_hours': free,
                   'owned_job_states': states, 'new_job_cap_hours': cap, 'reserve_hours': 2}
    (ROOT / f'runs/grade_probe_quota_before_launch{suffix}.json').write_text(json.dumps(observation, indent=2)+'\n')
    require(free >= cap + 2, 'free quota cannot cover comparison and two-hour reserve')
    receipt = {'status': 'PUSH_ATTEMPT_REQUIRES_OBSERVATION', 'kernel': metadata['id'],
               'observed_utc': observed, 'job_budget_hours': cap, 'allocated_free_hours': free,
               'competition_submission_enabled': False,
               'build_sha256': hashlib.sha256(build_path.read_bytes()).hexdigest()}
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    response = api.kernels_push(str(directory))
    error = str(getattr(response, 'error', None) or '').strip()
    if error:
        receipt.update(status='PUSH_REJECTED', error=error)
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
        raise RuntimeError(error)
    receipt.update(status='PUSH_ACCEPTED', version=response.version_number)
    receipt_path.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
