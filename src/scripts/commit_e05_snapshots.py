"""Register uploaded immutable snapshots; no Colab connection is required."""
import argparse
import json
from pathlib import Path
import traceback

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.datasets.types.dataset_api_service import (
    ApiCreateDatasetRequest,ApiDatasetNewFile,ApiGetDatasetRequest,
)

ROOT=Path(__file__).resolve().parents[1]


def blob_complete(item):
    with requests.put(item['url'],headers={'Content-Length':'0',
             'Content-Range':f'bytes */{item["bytes"]}'},timeout=(20,30)) as r:
        if r.status_code in (200,201):return True
        if r.status_code==308:return False
        raise RuntimeError('Upload observation HTTP '+str(r.status_code))


def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(5),required=True)
    p.add_argument('--phase',choices=['e01','e05','e09','final','terminal'],action='append',
                   help='Observe selected upload handles without reconnecting a Colab runtime')
    a=p.parse_args();out=ROOT/f'runs/e05_full_fold{a.fold}'
    leases=json.loads((out/'upload_tokens_private.json').read_text())
    api=KaggleApi();api.authenticate()
    for phase,items in leases['snapshots'].items():
        if a.phase and phase not in a.phase:continue
        slug=f'rsna-knee-e05-uniform-f{a.fold}-{phase}-20261009'
        if leases.get('attempt',1)>1:slug+=f'-a{leases["attempt"]}'
        state_path=out/f'commit_{phase}.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else None
        if state and state['status']=='PRIVATE_READY':continue
        if not state:
            if not all(blob_complete(item) for item in items.values()):
                print(json.dumps({'phase':phase,'status':'UPLOAD_PENDING'}),flush=True);continue
            state={'status':'CREATE_ATTEMPT_REQUIRES_OBSERVATION','dataset':'adityakumarrai0401/'+slug,'phase':phase}
            state_path.write_text(json.dumps(state,indent=2)+'\n')
            q=ApiCreateDatasetRequest();q.owner_slug='adityakumarrai0401';q.slug=slug
            q.title=f'E05 Uniform F{a.fold} {phase} Private Snapshot'
            q.license_name='other';q.is_private=True
            q.description='Private owned training artifact. Snapshot status and epoch are recorded inside. No competition submission or score improvement is implied.'
            q.files=[]
            for item in items.values():
                file=ApiDatasetNewFile();file.token=item['token'];q.files.append(file)
            with api.build_kaggle_client() as c:r=c.datasets.dataset_api_client.create_dataset(q)
            state['response']=r.to_dict();state['status']='CREATE_RESPONSE_RECEIVED'
            state_path.write_text(json.dumps(state,indent=2)+'\n')
        # On any ambiguous creation, observe this same slug; never repeat create.
        try:
            with api.build_kaggle_client() as c:
                q=ApiGetDatasetRequest();q.owner_slug='adityakumarrai0401';q.dataset_slug=slug
                metadata=c.datasets.dataset_api_client.get_dataset(q).to_dict()
            assert metadata['isPrivate'] is True
            status=api.dataset_status(state['dataset'])
        except requests.HTTPError as error:
            # Newly created private datasets briefly return404. Keep the saved
            # creation receipt, observe the same slug next pass, and allow other
            # uploaded phases (especially terminal) to be registered this pass.
            if error.response is None or error.response.status_code!=404:
                raise
            print(json.dumps({'phase':phase,'status':'METADATA_PENDING_RETRY_SAME_SLUG',
                              'dataset':state['dataset']}),flush=True)
            continue
        state.update(status='PRIVATE_READY' if status=='ready' else 'PROCESSING',
                     private_verified=True,platform_status=status)
        state_path.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'phase':phase,'status':state['status'],'dataset':state['dataset']}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'OBSERVATION_FAILED_RETRY_SAME_HANDLE','error_type':type(error).__name__,
                          'http_status':getattr(getattr(error,'response',None),'status_code',None),
                          'error_frames':[{'file':Path(f.filename).name,'line':f.lineno,'function':f.name}
                                          for f in traceback.extract_tb(error.__traceback__)]}),flush=True)
        raise SystemExit(1)
