"""Upload one hash-pinned artifact using a temporary URL, without API credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import requests


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def upload(path, lease, receipt_path):
    path=Path(path);receipt_path=Path(receipt_path)
    assert path.name==lease['name']
    size=path.stat().st_size
    assert size==lease['bytes'] and sha(path)==lease['sha256']
    assert lease['url'].startswith('https://')
    offset=0;completed=False
    for attempt in range(4):
        try:
            if attempt:
                with requests.put(lease['url'],headers={'Content-Length':'0',
                    'Content-Range':f'bytes */{size}'},timeout=(30,60)) as r:
                    if r.status_code in (200,201):completed=True;break
                    if r.status_code!=308:raise RuntimeError('Upload cannot resume; HTTP '+str(r.status_code))
                    received=r.headers.get('Range')
                    if received is None:offset=0
                    else:
                        match=re.fullmatch(r'bytes=0-(\d+)',received)
                        if not match:raise RuntimeError('Invalid upload range')
                        offset=int(match[1])+1
                    if not 0<=offset<size:raise RuntimeError('Invalid upload offset')
            headers={'Content-Length':str(size-offset)}
            if offset:headers['Content-Range']=f'bytes {offset}-{size-1}/{size}'
            with path.open('rb') as f:
                f.seek(offset)
                with requests.put(lease['url'],data=f,headers=headers,timeout=(30,180)) as r:
                    status=r.status_code
            if status in (200,201):completed=True;break
            if status not in (308,429,500,502,503,504):raise RuntimeError('Upload rejected; HTTP '+str(status))
        except requests.RequestException:
            if attempt==3:raise RuntimeError('Upload transport failed after retries') from None
        time.sleep(2**attempt)
    if not completed:raise RuntimeError('Upload remains incomplete')
    result={'status':'UPLOAD_COMPLETE','name':path.name,'bytes':size,'sha256':lease['sha256'],
            'api_credentials_used':False}
    receipt_path.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--file',type=Path,required=True)
    p.add_argument('--lease',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(upload(a.file,json.loads(a.lease.read_text()),a.receipt)))
