"""Launch the exact public-model reproduction only after prior result evaluation."""
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError
from audit_e04_resource_pilot import duration_seconds,require

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/nartaa_reproduction_v1'

def main():
    build=json.loads((OUT/'build.json').read_text())
    require(build['status']=='STAGED_NOT_LAUNCHED','missing source gate')
    for relative,expected in {**build['sources'],**build['gates']}.items():
        require(hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==expected,'source/gate changed: '+relative)
    require(not (OUT/'launch.json').exists(),'receipt exists: observe original handle')
    directory=ROOT/'public_kernels/nartaa_reproduction_v1'
    meta=json.loads((directory/'kernel-metadata.json').read_text());contract=build['contract']
    require(meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
            and meta['machine_shape']=='NvidiaTeslaT4','private GPU metadata differs')
    require(contract['qualification_allocation_hours']==2.1 and contract['reserve_gpu_hours']==2
            and contract['worker_seconds_cap']==7200 and not contract['competition_submission_enabled'],'budget/scope changed')
    prior=ROOT/'runs/orthodiffusion_trial_v2'
    decision=json.loads((prior/'decision.json').read_text())
    require(decision['status'] in {'PASS_SCREENING_ONLY','REJECT_FOR_ADVANCEMENT'},'previous experiment not fully evaluated')
    manifest=prior/'outputs/diffusion_probe_v2/manifest.json'
    require(hashlib.sha256(manifest.read_bytes()).hexdigest()==decision['manifest_sha256'],'prior evaluated artifact changed')
    api=KaggleApi();api.authenticate()
    require(api.kernels_status('adityakumarrai0401/rsna-knee-orthodiffusion-probe-v2').to_dict(ignore_defaults=False)['status']=='COMPLETE',
            'previous experiment is not terminal COMPLETE')
    owned=set();page=1
    while True:
        records=api.kernels_list(mine=True,page=page,page_size=100) or []
        owned.update(r.ref for r in records if r and r.ref)
        if len(records)<100:break
        page+=1;require(page<=100,'listing did not terminate')
    require('adityakumarrai0401/rsna-knee-b03-convnext-reader-blend' in owned,'known notebook absent')
    require(meta['id'] not in owned,'slug exists: inspect original version')
    def observe(ref):
        for attempt in range(4):
            time.sleep(1.25)
            try:
                state=api.kernels_status(ref).to_dict(ignore_defaults=False)['status']
                return dict(kernel=ref,status=state,observed_utc=datetime.now(timezone.utc).isoformat())
            except HTTPError as error:
                response=error.response
                if response is not None and response.status_code==404:
                    return dict(kernel=ref,status='NO_COMMITTED_SESSION_404')
                if response is None or response.status_code!=429 or attempt==3:raise
                try:delay=max(30,float(response.headers.get('Retry-After','30')))
                except ValueError:delay=60
                print(json.dumps(dict(stage='read_status_rate_limit_wait',seconds=delay,upload_attempted=False)),flush=True)
                while delay>0:
                    chunk=min(45,delay);time.sleep(chunk);delay-=chunk
    states=[]
    for ref in sorted(owned):
        states.append(observe(ref))
        if len(states)%20==0:
            (OUT/'owned_status_scan.json').write_text(json.dumps(states,indent=2)+'\n')
            print(json.dumps(dict(stage='active_job_check',checked=len(states),total=len(owned),upload_attempted=False)),flush=True)
    (OUT/'owned_status_scan.json').write_text(json.dumps(states,indent=2)+'\n')
    require(all(s['status'] in {'COMPLETE','ERROR','CANCEL_ACKNOWLEDGED','NO_COMMITTED_SESSION_404'} for s in states),'another owned notebook active')
    quota=api.quota_view().to_dict();g=quota['gpuQuota']
    free=(duration_seconds(g['totalTimeAllowed'])-duration_seconds(g['timeUsed'])-duration_seconds(g['timeReserved']))/3600
    now=datetime.now(timezone.utc).isoformat()
    (OUT/'quota_before.json').write_text(json.dumps(dict(observed_utc=now,quota=quota,free_gpu_hours=free,
                                                       owned_notebook_states=states),indent=2)+'\n')
    require(free>=4.1,'need2.1h allocation and2h reserve')
    receipt=dict(status='PUSH_ATTEMPT_REQUIRES_OBSERVATION',kernel=meta['id'],observed_utc=now,
                 job_budget_hours=2.1,prior_decision_sha256=hashlib.sha256((prior/'decision.json').read_bytes()).hexdigest(),
                 launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),allocated_free_hours=free,competition_submission_enabled=False,
                 build_sha256=hashlib.sha256((OUT/'build.json').read_bytes()).hexdigest())
    p=OUT/'launch.json';p.write_text(json.dumps(receipt,indent=2)+'\n')
    response=api.kernels_push(str(directory));error=str(getattr(response,'error',None) or '').strip()
    if error:
        receipt.update(status='PUSH_REJECTED',error=error);p.write_text(json.dumps(receipt,indent=2)+'\n')
        raise RuntimeError(error)
    receipt.update(status='PUSH_ACCEPTED',version=response.version_number)
    p.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
