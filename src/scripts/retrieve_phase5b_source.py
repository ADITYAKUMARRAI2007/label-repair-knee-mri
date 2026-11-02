"""Locate the requested published notebook and preserve exact version replies."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import signal
import time
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
from observe_coherent_anchor_pair import remote_status_name

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/phase5b_dual_sota_20261011'

def main():
    OUT.mkdir(exist_ok=True)
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError()))
    signal.alarm(240)
    api=KaggleApi();api.authenticate()
    states={}
    for ref in ['adityakumarrai0401/rsna-knee-side-phase5-roi-train-20261011',
                'adityakumarrai0401/rsna-knee-helios-compact-qualification']:
        states[ref]=remote_status_name(api.kernels_status(ref));time.sleep(1.5)
    (OUT/'native_status.json').write_text(json.dumps({'observed_utc':datetime.now(timezone.utc).isoformat(),'states':states},indent=2)+'\n')
    rows=api.kernels_list(user='jenillangaliya',search='RSNA Knee',page_size=100,sort_by='dateRun') or []
    results=[r.to_dict() for r in rows]
    (OUT/'author_search.json').write_text(json.dumps(results,indent=2,default=str)+'\n')
    print(json.dumps({'native_states':states,'notebooks':results},default=str),flush=True)
    matches=[r for r in results if r.get('title')=='RSNA Knee [LB 0.954]: Dual SOTA Ensemble']
    assert len(matches)==1, 'Exact title/author did not yield one notebook; inspect saved search'
    ref=matches[0]['ref'];owner,slug=ref.split('/')
    records=[]
    for version in ['3','4']:
        time.sleep(1.5)
        req=ApiGetKernelRequest();req.user_name=owner;req.kernel_slug=slug+'/'+version
        with api.build_kaggle_client() as client:
            response=client.kernels.kernels_api_client.get_kernel(req)
        d=response.to_dict();target=OUT/('v'+version);target.mkdir(exist_ok=True)
        (target/'api_response.json').write_text(json.dumps(d,indent=2,default=str)+'\n')
        records.append({'requested_version':version,'response_fields':list(d),'directory':str(target.relative_to(ROOT))})
    (OUT/'source_identification.json').write_text(json.dumps({'ref':ref,'author':owner,'title':matches[0]['title'],'versions':records},indent=2)+'\n')
    print(json.dumps({'retrieved':records,'ref':ref}),flush=True)

if __name__=='__main__':main()
