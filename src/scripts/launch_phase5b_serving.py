"""Preflight native quota/owned runs and push fixed A--E serving once; never submit."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import time
from requests import HTTPError
from kaggle.api.kaggle_api_extended import KaggleApi
from audit_e04_resource_pilot import duration_seconds
from kaggle_quota import quota_snapshot
from launch_e05_kaggle_pair import TERMINAL,inspect_missing_session
from observe_coherent_anchor_pair import remote_status_name

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'research/phase5b_dual_sota_20261011'
PACKAGE=ROOT/'public_kernels/phase5b_fixed_compact_blend'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())

def main():
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Bounded preflight/push expired; inspect original handle')))
    signal.alarm(480)
    launch=RUN/'serving_launch.json'
    assert not launch.exists(), 'Observe original accepted or uncertain push'
    build=read(RUN/'serving_build.json'); contract=read(RUN/'serving_contract.json')
    assert sha(PACKAGE/'inference.ipynb')==build['notebook_sha256']
    assert sha(PACKAGE/'kernel-metadata.json')==build['metadata_sha256']
    assert sha(RUN/'serving_contract.json')==build['protocol_sha256']
    meta=read(PACKAGE/'kernel-metadata.json')
    assert meta['is_private'] and meta['enable_gpu'] and not meta['enable_internet']
    assert meta['kernel_sources']==[] and meta['competition_sources']==['rsna-knee-abnormality-detection']
    assert not contract['competition_submission_enabled'] and contract['allocated_gpu_count']==2
    api=KaggleApi();api.authenticate()
    owned=set()
    for page in range(1,101):
        rows=api.kernels_list(mine=True,page=page,page_size=100) or []
        owned.update(r.ref for r in rows if r and r.ref)
        if len(rows)<100: break
    else: raise AssertionError('Owned listing did not terminate')
    assert meta['id'] not in owned, 'Slug exists; inspect original run'
    def check(ref):
        try:
            status=remote_status_name(api.kernels_status(ref))
            return {'ref':ref,'status':status}
        except HTTPError as e:
            if e.response is None or e.response.status_code!=404:raise
            return {'ref':ref,**inspect_missing_session(api,ref)}
    states=[]
    for ref in sorted(owned):
        time.sleep(1.5)
        for retry in range(3):
            try:
                item=check(ref); break
            except HTTPError as e:
                if e.response is None or e.response.status_code!=429 or retry==2: raise
                print(json.dumps({'preflight_rate_limited':True,'owned_handles_checked':len(states)}),flush=True)
                time.sleep(20*(retry+1))
        states.append(item)
        (RUN/'serving_owned_scan.json').write_text(json.dumps(states,indent=2)+'\n')
        if len(states)%20==0:print(json.dumps({'owned_handles_checked':len(states)}),flush=True)
    active=[r for r in states if r['status'] not in TERMINAL|{'HISTORICAL_SESSION_UNAVAILABLE'}]
    assert not active, f'Active/unknown owned run: {active}'
    assert api.dataset_status('adityakumarrai0401/rsna-knee-decoder-cp313')=='ready'
    for p in ['dependency_files.json','decoder_files.json','weight_access_probe.json']:
        assert (ROOT/'research/public_refresh_20261011'/p).exists()
    subs=api.competition_submissions('rsna-knee-abnormality-detection',page_size=100) or []
    assert all(s.status.name in {'COMPLETE','ERROR'} for s in subs)
    quota=quota_snapshot(api.quota_view())['gpuQuota']
    reserved=duration_seconds(quota.get('timeReserved','0s'))
    free=(duration_seconds(quota['totalTimeAllowed'])-duration_seconds(quota['timeUsed'])-reserved)/3600
    required=contract['platform_timeout_seconds']*2/3600+contract['reserve_gpu_hours']
    assert reserved==0 and free>=required, f'Insufficient/unavailable quota: {free}h reserved{reserved}s required{required}h'
    assert sha(PACKAGE/'inference.ipynb')==build['notebook_sha256']
    record={'status':'PUSH_ATTEMPT_REQUIRES_OBSERVATION','kernel':meta['id'],
            'observed_utc':datetime.now(timezone.utc).isoformat(),
            'code_cell_sha256':build['code_cell_sha256'],'build_sha256':sha(RUN/'serving_build.json'),
            'protocol_sha256':build['protocol_sha256'],'available_gpu_hours':free,
            'required_gpu_hours':required,'platform_timeout_seconds':contract['platform_timeout_seconds'],
            'competition_submission_enabled':False,'quota_before':quota,'owned_handles_checked':len(states)}
    with launch.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    signal.alarm(60)
    try:
        response=api.kernels_push(str(PACKAGE),timeout=str(contract['platform_timeout_seconds']))
        assert not response.error and response.version_number>0, response
        ref=response.ref.removeprefix('https://www.kaggle.com/').lstrip('/').removeprefix('code/')
        assert ref==meta['id']
        record.update(status='PUSH_ACCEPTED',version=response.version_number,
                      accepted_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as e:
        record['observation_error_type']=type(e).__name__
        raise
    finally:
        signal.alarm(0)
        launch.write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)

if __name__=='__main__':main()
