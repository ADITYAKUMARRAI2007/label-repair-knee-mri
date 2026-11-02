"""Stream-hash owned control checkpoints without filling the local disk."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiListKernelSessionOutputRequest

ROOT=Path(__file__).resolve().parents[1]


def main():
    api=KaggleApi();api.authenticate();verified=[]
    out=ROOT/'runs/e05_confidence_ablation/control_integrity.json'
    for pair in ['01','23']:
        with api.build_kaggle_client() as client:
            q=ApiListKernelSessionOutputRequest();q.user_name='adityakumarrai0401'
            q.kernel_slug=f'rsna-knee-e04-convnext-folds-{pair}';q.page_size=100
            response=client.kernels.kernels_api_client.list_kernel_session_output(q)
            assert not response.next_page_token
            files={f.file_name:f.url for f in response.files}
        for fold in map(int,pair):
            prefix=f'e04_fold{fold}'
            manifest_path=ROOT/f'runs/E04_folds_{pair}_outputs'/prefix/f'{prefix}_manifest.json'
            manifest=json.loads(manifest_path.read_text())
            name=f'{prefix}_ema.pt';expected=manifest['artifacts'][name]
            matches=[url for path,url in files.items() if Path(path).name==name]
            assert len(matches)==1
            h=hashlib.sha256();count=0
            with requests.get(matches[0],stream=True,timeout=(30,120)) as r:
                if r.status_code!=200:raise RuntimeError('Checkpoint download HTTP failure')
                for chunk in r.iter_content(8<<20):h.update(chunk);count+=len(chunk)
            assert count==expected['bytes'] and h.hexdigest()==expected['sha256']
            verified.append({'fold':fold,'kernel':q.kernel_slug,'file':name,'bytes':count,
                             'sha256':h.hexdigest(),'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
            record={'status':'PASS_ALL_FOUR' if len(verified)==4 else 'PARTIAL_VERIFICATION',
                    'observed_utc':datetime.now(timezone.utc).isoformat(),'checkpoints':verified,
                    'method':'Streamed actual remote checkpoint bytes through SHA256; not downloaded to local storage',
                    'accuracy_measured':False,'competition_submission_enabled':False}
            out.write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(verified[-1]),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        print(json.dumps({'status':'VERIFICATION_FAILED','error_type':type(error).__name__}),flush=True)
        raise SystemExit(1)
