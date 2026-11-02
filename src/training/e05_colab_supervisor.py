"""Prepare verified assets, then run two bounded full-epoch resource workers."""
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

import requests

SETUP=Path('/content/e05_setup')
INPUT=Path('/content/kaggle/input')
OUTPUT=Path('/content/e05_results')
START=time.monotonic()


def fetch_range(asset, start, end):
    for attempt in range(4):
        try:
            with requests.get(asset['url'],headers={'Range':f'bytes={start}-{end}'},timeout=(30,90)) as response:
                if response.status_code!=206:raise RuntimeError('Server did not honor range request')
                if response.headers.get('Content-Range')!=f'bytes {start}-{end}/{asset["bytes"]}':
                    raise RuntimeError('Incorrect byte range')
                block=response.content
                if len(block)!=end-start+1:raise RuntimeError('Incomplete byte range')
            return block
        except requests.RequestException:
            if attempt==3:raise
            time.sleep(2**attempt)


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def fetch(asset):
    p=INPUT/asset['path']
    assert p.resolve().is_relative_to(INPUT.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and sha(p)==asset['sha256']:
        print(json.dumps({'asset_reused':p.name,'bytes':p.stat().st_size}),flush=True)
        return {'name':p.name,'bytes':p.stat().st_size,'sha256':asset['sha256']}
    part=p.with_suffix(p.suffix+'.part')
    h=hashlib.sha256();count=0
    if asset.get('bytes',0)>512<<20:
        count=part.stat().st_size if part.exists() else 0
        if count>asset['bytes']:raise RuntimeError('Invalid partial file size')
        print(json.dumps({'range_resume':p.name,'offset':count,'total':asset['bytes']}),flush=True)
        with part.open('ab') as f, concurrent.futures.ThreadPoolExecutor(max_workers=4) as ranges:
            while count<asset['bytes']:
                if time.monotonic()-START>900:raise TimeoutError('Asset transfer budget exhausted')
                starts=list(range(count,min(count+(256<<20),asset['bytes']),64<<20))
                pending=[(start,min(start+(64<<20),asset['bytes'])-1) for start in starts]
                futures=[ranges.submit(fetch_range,asset,start,end) for start,end in pending]
                # Commit in order. An interruption leaves a contiguous prefix,
                # and at most four ranges per asset occupy memory at once.
                for (start,end),future in zip(pending,futures):
                    block=future.result();assert start==count
                    f.write(block);f.flush();count=end+1
                    if count%(1<<30)<64<<20 or count==asset['bytes']:
                        print(json.dumps({'range_progress':p.name,'bytes':count,'total':asset['bytes']}),flush=True)
        h=None
    else:
        with requests.get(asset['url'],stream=True,timeout=(30,120)) as response:
            if response.status_code!=200:raise RuntimeError('HTTP_'+str(response.status_code))
            with part.open('wb') as f:
                for block in response.iter_content(8<<20):
                    if time.monotonic()-START>900:raise TimeoutError('Asset transfer budget exhausted')
                    f.write(block);h.update(block);count+=len(block)
    if h is not None and h.hexdigest()!=asset['sha256'] and zipfile.is_zipfile(part):
        with zipfile.ZipFile(part) as z:
            names=[n for n in z.namelist() if Path(n).name==p.name]
            if len(names)!=1:raise RuntimeError('Unexpected downloaded archive')
            with z.open(names[0]) as src,p.open('wb') as dest:
                for block in iter(lambda:src.read(8<<20),b''):dest.write(block)
        part.unlink()
    else:part.replace(p)
    if sha(p)!=asset['sha256']:raise RuntimeError('Asset checksum mismatch: '+p.name)
    if 'bytes' in asset and p.stat().st_size!=asset['bytes']:raise RuntimeError('Asset size mismatch: '+p.name)
    print(json.dumps({'asset_ready':p.name,'bytes':p.stat().st_size}),flush=True)
    return {'name':p.name,'bytes':p.stat().st_size,'sha256':asset['sha256']}


def main():
    if any((OUTPUT/name).exists() for name in ['control','uniform','result.json']):
        raise RuntimeError('Training artifacts already exist; observe instead of restarting')
    OUTPUT.mkdir(exist_ok=True)
    assets=json.loads((SETUP/'download_access.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:receipts=list(pool.map(fetch,assets))
    (SETUP/'download_access.json').unlink()
    (OUTPUT/'assets.json').write_text(json.dumps(receipts,indent=2)+'\n')
    build=json.loads((SETUP/'build.json').read_text())
    assert sha(SETUP/'worker.py')==build['code_sha256']
    os.environ['E04_TRAIN_FOLD']='0'
    spec=importlib.util.spec_from_file_location('e05_worker',SETUP/'worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    verified=worker.verify_wide_files()
    verification=OUTPUT/'cache_verification.json'
    verification.write_text(json.dumps(verified,indent=2)+'\n')
    env=os.environ.copy();env.update(E04_CACHE_VERIFICATION_PATH=str(verification),
        E04_CACHE_VERIFICATION_SHA256=sha(verification),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    start_train=time.monotonic();reports=[]
    for arm in ['control','uniform']:
        remaining=2700-(time.monotonic()-start_train)
        if remaining<60:raise TimeoutError('Insufficient remaining pilot budget')
        print(json.dumps({'stage':'training_epoch','arm':arm,'remaining_seconds':remaining}),flush=True)
        with (OUTPUT/f'{arm}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-u',str(SETUP/'worker.py'),'--arm',arm,
                '--output',str(OUTPUT/arm)],env=env,stdout=log,stderr=subprocess.STDOUT)
            try:code=proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
                raise RuntimeError('Resource epoch exceeded budget') from None
        if code!=0:raise RuntimeError('Worker failed: '+arm+'; inspect its local log')
        reports.append(json.loads((OUTPUT/arm/'receipt.json').read_text()))
    assert reports[0]['training_order_sha256']==reports[1]['training_order_sha256']
    report={'status':'PAIRED_RESOURCE_EPOCHS_COMPLETE','reports':reports,
            'total_seconds':time.monotonic()-START,'accuracy_measured':False,
            'competition_submission_enabled':False}
    (OUTPUT/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as error:
        # Request errors may carry signed URLs: log class only for those.
        message=type(error).__name__ if isinstance(error,requests.RequestException) else str(error)
        print(json.dumps({'status':'PILOT_FAILED','error_type':type(error).__name__,'message':message}),flush=True)
        raise SystemExit(1)
