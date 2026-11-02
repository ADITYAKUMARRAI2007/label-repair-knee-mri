"""Preissue bounded artifact upload URLs for one owned Colab training fold."""
import argparse
import json
import os
from pathlib import Path
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.blobs.types.blob_api_service import ApiStartBlobUploadRequest,ApiBlobType

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=range(5),required=True)
    parser.add_argument('--attempt',type=int,default=1)
    args=parser.parse_args();prefix=f'e05_uniform_fold{args.fold}'
    assert args.attempt>=1
    out=ROOT/f'runs/e05_full_fold{args.fold}';out.mkdir(exist_ok=True)
    private=out/'upload_tokens_private.json';remote=out/'upload_leases_private.json'
    assert not private.exists() and not remote.exists(),'Reuse or explicitly retire existing leases'
    api=KaggleApi();api.authenticate();snapshots={};tokens={}
    with api.build_kaggle_client() as c:
        for phase in ['e01','e05','e09','final','terminal']:
            snapshots[phase]={};tokens[phase]={}
            for kind,size,extension in [('bundle',1024**3,'tar'),('receipt',64*1024,'json')]:
                if phase=='terminal' and kind=='bundle':continue
                name=f'{prefix}_{phase}.{extension}'
                q=ApiStartBlobUploadRequest();q.type=ApiBlobType.DATASET;q.name=name
                q.content_length=size;q.last_modified_epoch_seconds=int(time.time())
                blob=c.blobs.blob_api_client.start_blob_upload(q)
                snapshots[phase][kind]={'name':name,'bytes':size,'url':blob.create_url}
                tokens[phase][kind]={'name':name,'bytes':size,'token':blob.token,'url':blob.create_url}
            print(json.dumps({'phase_upload_leases_ready':phase}),flush=True)
    dataset=f'adityakumarrai0401/rsna-knee-e05-uniform-fold{args.fold}-20261009'
    for path,data in [(private,{'prefix':prefix,'dataset':dataset,'attempt':args.attempt,'snapshots':tokens}),
                      (remote,{'prefix':prefix,'dataset':dataset,'attempt':args.attempt,'snapshots':snapshots})]:
        fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
        with os.fdopen(fd,'w') as f:json.dump(data,f)
    print(json.dumps({'status':'SCOPED_UPLOAD_URLS_READY','fold':args.fold,
                      'maximum_uploaded_gib':4+4*64*1024/1024**3,'kaggle_credentials_transferred':False}))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'LEASE_PREPARATION_FAILED','error_type':type(error).__name__}))
        raise SystemExit(1)
