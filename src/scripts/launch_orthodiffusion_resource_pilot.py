"""Launch once after source, prior-result, active-job and free-quota checks."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError
from audit_e04_resource_pilot import duration_seconds, require

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'runs/orthodiffusion_resource_v1'

def main():
    build = json.loads((OUT/'build.json').read_text())
    require(build['status']=='BUILT_NOT_LAUNCHED','build missing')
    for relative, expected in build['sources'].items():
        require(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,'source changed: '+relative)
    require(not (OUT/'launch.json').exists(),'receipt exists: observe same handle')
    previous = json.loads((ROOT/'runs/uncertainty_weight_probe_v1/result.json').read_text())
    require(previous['status']=='REJECT_FOR_ADVANCEMENT','evaluate previous experiment first')
    directory = ROOT/'public_kernels/orthodiffusion_resource_pilot'
    metadata = json.loads((directory/'kernel-metadata.json').read_text())
    require(metadata['is_private'] and metadata['enable_gpu'] and not metadata['enable_internet']
            and metadata['machine_shape']=='NvidiaTeslaT4','private free GPU metadata')
    contract = build['contract']
    require(contract['allocated_gpu_hours_cap']==.5 and contract['reserve_gpu_hours']==2
            and not contract['competition_submission_enabled'],'budget/scope drift')
    api=KaggleApi();api.authenticate()
    require(api.kernels_status('adityakumarrai0401/rsna-knee-orthofoundation-probe').to_dict()['status']=='COMPLETE',
            'previous GPU experiment not complete')
    owned=set();page=1
    while True:
        records=api.kernels_list(mine=True,page=page,page_size=100) or []
        owned.update(r.ref for r in records if r and r.ref)
        if len(records)<100:break
        page+=1;require(page<=100,'listing did not terminate')
    require('adityakumarrai0401/rsna-knee-b03-convnext-reader-blend' in owned,'known notebook absent')
    require(metadata['id'] not in owned,'slug exists: inspect original version')
    def observe(ref):
        try:state=api.kernels_status(ref).to_dict()['status']
        except HTTPError as error:
            if error.response is None or error.response.status_code!=404:raise
            state='NO_COMMITTED_SESSION_404'
        return dict(kernel=ref,status=state)
    with ThreadPoolExecutor(max_workers=4) as pool:
        states=list(pool.map(observe,sorted(r for r in owned if '/rsna-knee-' in r)))
    require(all(s['status'] in {'COMPLETE','ERROR','CANCELLED','NO_COMMITTED_SESSION_404'} for s in states),
            'another owned knee notebook active')
    quota=api.quota_view().to_dict();gpu=quota['gpuQuota']
    free=(duration_seconds(gpu['totalTimeAllowed'])-duration_seconds(gpu['timeUsed'])
          -duration_seconds(gpu['timeReserved']))/3600
    now=datetime.now(timezone.utc).isoformat()
    (OUT/'quota_before.json').write_text(json.dumps(dict(observed_utc=now,quota=quota,
        free_gpu_hours=free,owned_knee_states=states),indent=2)+'\n')
    require(free>=2.5,'need .5h pilot allocation and2h reserve')
    receipt=dict(status='PUSH_ATTEMPT_REQUIRES_OBSERVATION',kernel=metadata['id'],
                 observed_utc=now,job_budget_hours=.5,allocated_free_hours=free,
                 competition_submission_enabled=False,
                 build_sha256=hashlib.sha256((OUT/'build.json').read_bytes()).hexdigest())
    path=OUT/'launch.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    response=api.kernels_push(str(directory))
    error=str(getattr(response,'error',None) or '').strip()
    if error:
        receipt.update(status='PUSH_REJECTED',error=error)
        path.write_text(json.dumps(receipt,indent=2)+'\n');raise RuntimeError(error)
    receipt.update(status='PUSH_ACCEPTED',version=response.version_number)
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))

if __name__ == '__main__':main()
