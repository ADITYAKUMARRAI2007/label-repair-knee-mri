"""Launch the frozen comparison exactly once using available free GPU quota."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError
from audit_e04_resource_pilot import duration_seconds,require

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/orthodiffusion_trial_v1'

def main():
    build=json.loads((OUT/'build.json').read_text())
    require(build['status']=='BUILT_NOT_LAUNCHED','missing source gate')
    for relative,expected in {**build['sources'],**build['gates']}.items():
        require(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,'source/gate changed: '+relative)
    require(not (OUT/'launch.json').exists(),'receipt exists: observe original handle')
    directory=ROOT/'public_kernels/orthodiffusion_probe_v1'
    meta=json.loads((directory/'kernel-metadata.json').read_text());contract=build['contract']
    require(meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
            and meta['machine_shape']=='NvidiaTeslaT4','private GPU metadata differs')
    require(contract['allocated_gpu_hours_cap']==2 and contract['reserve_gpu_hours']==2
            and contract['worker_seconds_cap']==6000 and not contract['competition_submission_enabled'],'budget/scope changed')
    api=KaggleApi();api.authenticate()
    require(api.kernels_status('adityakumarrai0401/rsna-knee-orthodiffusion-resource-pilot').to_dict()['status']=='COMPLETE',
            'previous pilot not complete')
    for source in meta['kernel_sources']:
        require(api.kernels_status(source).to_dict()['status']=='COMPLETE','control feature source incomplete')
    owned=set();page=1
    while True:
        records=api.kernels_list(mine=True,page=page,page_size=100) or []
        owned.update(r.ref for r in records if r and r.ref)
        if len(records)<100:break
        page+=1;require(page<=100,'listing did not terminate')
    require('adityakumarrai0401/rsna-knee-b03-convnext-reader-blend' in owned,'known notebook absent')
    require(meta['id'] not in owned,'slug exists: inspect original version')
    def observe(ref):
        try:state=api.kernels_status(ref).to_dict()['status']
        except HTTPError as error:
            if error.response is None or error.response.status_code!=404:raise
            state='NO_COMMITTED_SESSION_404'
        return dict(kernel=ref,status=state)
    with ThreadPoolExecutor(max_workers=4) as pool:
        states=list(pool.map(observe,sorted(r for r in owned if '/rsna-knee-' in r)))
    require(all(s['status'] in {'COMPLETE','ERROR','CANCELLED','NO_COMMITTED_SESSION_404'} for s in states),'another knee notebook active')
    quota=api.quota_view().to_dict();g=quota['gpuQuota']
    free=(duration_seconds(g['totalTimeAllowed'])-duration_seconds(g['timeUsed'])-duration_seconds(g['timeReserved']))/3600
    now=datetime.now(timezone.utc).isoformat()
    (OUT/'quota_before.json').write_text(json.dumps(dict(observed_utc=now,quota=quota,free_gpu_hours=free,
                                                       owned_knee_states=states),indent=2)+'\n')
    require(free>=4,'need2h allocation and2h reserve')
    receipt=dict(status='PUSH_ATTEMPT_REQUIRES_OBSERVATION',kernel=meta['id'],observed_utc=now,
                 job_budget_hours=2,allocated_free_hours=free,competition_submission_enabled=False,
                 build_sha256=hashlib.sha256((OUT/'build.json').read_bytes()).hexdigest())
    p=OUT/'launch.json';p.write_text(json.dumps(receipt,indent=2)+'\n')
    response=api.kernels_push(str(directory));error=str(getattr(response,'error',None) or '').strip()
    if error:
        receipt.update(status='PUSH_REJECTED',error=error);p.write_text(json.dumps(receipt,indent=2)+'\n')
        raise RuntimeError(error)
    receipt.update(status='PUSH_ACCEPTED',version=response.version_number)
    p.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
