"""Launch one private diagnostic only when the free-quota and source gates pass."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi
from requests import HTTPError
from audit_e04_resource_pilot import duration_seconds

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/n03_paired_diagnostic'
PACKAGE = ROOT / 'public_kernels/n03_paired_diagnostic'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--launch',action='store_true',help='Push the qualified private notebook; never make a competition submission')
    args=parser.parse_args()
    assert not (RUN/'launch.json').exists(), 'Observe existing launch; do not restart'
    build=json.loads((RUN/'build.json').read_text())
    check=json.loads((RUN/'local_check.json').read_text())
    protocol=json.loads((RUN/'protocol.json').read_text())
    meta=json.loads((PACKAGE/'kernel-metadata.json').read_text())
    assert sha(PACKAGE/'diagnostic.py')==build['code_sha256']==check['code_sha256']
    assert sha(PACKAGE/'kernel-metadata.json')==build['metadata_sha256']
    assert sha(RUN/'protocol.json')==build['protocol_sha256']
    assert check['status']=='PASS_LOCAL_CHANGED_PATHS'
    assert sha(Path(__file__))==build['launcher_sha256'], 'Launcher changed since source review'
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert not protocol['competition_submission_enabled'] and not protocol['weight_search_allowed']
    api=KaggleApi();api.authenticate()
    def free_quota():
        quota=api.quota_view().to_dict();gpu=quota['gpuQuota']
        free=(duration_seconds(gpu['totalTimeAllowed'])-duration_seconds(gpu['timeUsed'])-duration_seconds(gpu.get('timeReserved','0s')))/3600
        receipt={'observed_utc':datetime.now(timezone.utc).isoformat(),'quota':quota,'free_gpu_hours':free,'required_free_hours':protocol['free_gpu_required_before_launch_hours']}
        (RUN/'quota_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
        assert free>=protocol['free_gpu_required_before_launch_hours'], 'Insufficient free GPU hours; do not use paid capacity'
    free_quota()
    if not args.launch:
        print('Source gates and free quota pass. No notebook pushed.')
        return
    submissions=api.competition_submissions('rsna-knee-abnormality-detection',page_size=100)
    assert all(s.status.name in {'COMPLETE','ERROR'} for s in submissions), 'Another submission is active or has unknown status'
    owned=set();page=1
    while True:
        rows=api.kernels_list(mine=True,page=page,page_size=100) or []
        owned.update(r.ref for r in rows if r.ref)
        if len(rows)<100:break
        page+=1;assert page<=100
    assert meta['id'] not in owned, 'Notebook already exists: observe it'
    states=[]
    for ref in sorted(owned):
        for attempt in range(4):
            time.sleep(1.25)
            try:
                state=api.kernels_status(ref).to_dict().get('status')
                break
            except HTTPError as error:
                if error.response is None or error.response.status_code!=429 or attempt==3:
                    raise
                print(json.dumps({'status_poll_rate_limited':ref,'retry_same_handle':True}),flush=True)
                time.sleep(30)
        states.append({'ref':ref,'status':state})
        assert state in {'COMPLETE','ERROR','CANCEL_ACKNOWLEDGED'}, f'Active or unknown notebook: {ref} {state}'
        if len(states)%20==0:print(json.dumps({'checked_terminal':len(states),'total':len(owned)}),flush=True)
    (RUN/'owned_status_scan.json').write_text(json.dumps(states,indent=2)+'\n')
    free_quota()
    receipt={'status':'PUSH_ATTEMPT_REQUIRES_OBSERVATION','kernel':meta['id'],'code_sha256':build['code_sha256'],'observed_utc':datetime.now(timezone.utc).isoformat(),'competition_submission_enabled':False,'platform_timeout_seconds':1800}
    path=RUN/'launch.json';path.write_text(json.dumps(receipt,indent=2)+'\n')
    response=api.kernels_push(str(PACKAGE),timeout='1800')
    error=str(getattr(response,'error',None) or '').strip()
    receipt.update(status='PUSH_REJECTED' if error else 'PUSH_ACCEPTED',error=error,version=getattr(response,'version_number',None))
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))
    if error:raise RuntimeError(error)


if __name__=='__main__':main()
