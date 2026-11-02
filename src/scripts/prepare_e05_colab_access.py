"""Obtain temporary download links locally; never transfer Kaggle credentials."""
import json
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import (
    KaggleApi, ApiListKernelSessionOutputRequest,
    ApiDownloadDatasetRequest, ApiDownloadDataFileRequest,
)

ROOT=Path(__file__).resolve().parents[1]


def main():
    api=KaggleApi();api.authenticate();assets=[]
    contract=json.loads((ROOT/'runs/E04_contract/e04_contract.json').read_text())
    with api.build_kaggle_client() as client:
        for part in ['base','extension']:
            name=f'e04_wide_{part}_manifest.json'
            path=ROOT/f'runs/E04_wide_cache_{part}'/name
            import hashlib
            manifest=json.loads(path.read_text())
            expected={name:{'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},**manifest['artifacts']}
            request=ApiListKernelSessionOutputRequest();request.user_name='adityakumarrai0401'
            request.kernel_slug=f'rsna-knee-e04-wide-cache-{part}';request.page_size=100
            response=client.kernels.kernels_api_client.list_kernel_session_output(request)
            assert not response.next_page_token,'Unexpected output pagination'
            available={f.file_name:f.url for f in response.files}
            for file,receipt in expected.items():
                assert file in available,file
                assets.append({'path':f'wide_{part}/{file}','url':available[file],**receipt})
        dataset_files=[]
        for name,receipt in contract['fold_reconstruction_only_sources'].items():
            if name=='train_targets_weights.csv':continue
            parent=Path(receipt['path']).parent.name
            owner,slug=parent.split('_',1)
            dataset_files.append((owner,slug,name,receipt['sha256']))
        dataset_files.append(('adityakumarrai0401','rsna-knee-e04-convnext-imagenet-weights','model.safetensors',
                              '2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9'))
        for owner,slug,name,digest in dataset_files:
            q=ApiDownloadDatasetRequest();q.owner_slug=owner;q.dataset_slug=slug;q.file_name=name
            with client.datasets.dataset_api_client.download_dataset(q) as response:
                response.raise_for_status()
                assets.append({'path':f'{owner}_{slug}/{name}','url':response.url,'sha256':digest})
        for name in ['train.csv','test.csv']:
            q=ApiDownloadDataFileRequest();q.competition_name='rsna-knee-abnormality-detection';q.file_name=name
            with client.competitions.competition_api_client.download_data_file(q) as response:
                response.raise_for_status()
                assets.append({'path':f'competition/{name}','url':response.url,
                               'sha256':hashlib.sha256((ROOT/'data'/name).read_bytes()).hexdigest()})
    out=ROOT/'runs/e05_colab_pilot/download_access.json'
    fd=os.open(out,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as f:json.dump(assets,f)
    print(json.dumps({'status':'TEMPORARY_ACCESS_READY','files':len(assets),
                      'cache_bytes':sum(a.get('bytes',0) for a in assets),'credentials_transferred':False}))


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'ACCESS_PREPARATION_FAILED','error_type':type(error).__name__}))
        raise SystemExit(1)
