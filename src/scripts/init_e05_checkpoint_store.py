"""Create a private durable artifact store with the resource receipt only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.blobs.types.blob_api_service import ApiStartBlobUploadRequest,ApiBlobType
from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest,ApiDatasetNewFile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/e05_checkpoint_store'
OWNER='adityakumarrai0401'
SLUG='rsna-knee-e05-checkpoints-20261009'


def main():
    OUT.mkdir(exist_ok=True)
    state_path=OUT/'initialization.json'
    assert not state_path.exists(),'An initialization exists; inspect it rather than repeat creation'
    artifact=ROOT/'runs/e05_colab_pilot/resource_audit.json'
    report=json.loads(artifact.read_text());assert report['status']=='PASS_PAIRED_RESOURCE_EPOCHS'
    assert report['accuracy_measured'] is False
    digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
    api=KaggleApi();api.authenticate()
    q=ApiStartBlobUploadRequest();q.type=ApiBlobType.DATASET;q.name=artifact.name
    q.content_length=artifact.stat().st_size;q.last_modified_epoch_seconds=int(time.time())
    q.content_type='application/json'
    with api.build_kaggle_client() as c:blob=c.blobs.blob_api_client.start_blob_upload(q)
    lease={'name':artifact.name,'bytes':artifact.stat().st_size,'sha256':digest,'url':blob.create_url}
    secret=OUT/'initial_upload_private.json'
    fd=os.open(secret,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'w') as f:json.dump({'lease':lease,'token':blob.token},f)
    spec=importlib.util.spec_from_file_location('uploader',ROOT/'training/e05_upload_blob.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.upload(artifact,lease,OUT/'initial_upload_receipt.json')
    request=ApiCreateDatasetRequest();request.owner_slug=OWNER;request.slug=SLUG
    request.title='RSNA Knee E05 Private Checkpoints';request.license_name='other';request.is_private=True
    request.description='Private experiment storage. Initial file is a resource audit only; no trained medal candidate or score improvement is claimed. No raw images, reports, labels or study IDs are included in this initialization.'
    item=ApiDatasetNewFile();item.token=blob.token;request.files=[item]
    state={'status':'CREATE_ATTEMPT_REQUIRES_OBSERVATION','dataset':OWNER+'/'+SLUG,
           'is_private_requested':True,'initial_file':artifact.name,'initial_sha256':digest,
           'competition_submission_enabled':False}
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    with api.build_kaggle_client() as c:response=c.datasets.dataset_api_client.create_dataset(request)
    state.update(status='CREATE_RESPONSE_RECEIVED',response=response.to_dict())
    state_path.write_text(json.dumps(state,indent=2)+'\n')
    secret.unlink()
    print(json.dumps(state,indent=2))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'STORE_INITIALIZATION_FAILED','error_type':type(error).__name__}))
        raise SystemExit(1)
