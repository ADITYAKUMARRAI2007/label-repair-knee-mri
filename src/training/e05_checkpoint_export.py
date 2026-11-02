"""Publish bounded checkpoint snapshots with preissued, task-specific upload URLs.

Kaggle account credentials stay local. A fixed-size tar permits upload leases
to be allocated before the training run, without predicting tensor values.
"""
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time

from e05_upload_blob import upload

BUNDLE_BYTES=1024**3
RECEIPT_BYTES=64*1024


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def publish_snapshot(directory, prefix, phase, completed_epoch):
    root=Path(directory)
    cfg=json.loads(Path(os.environ['E05_UPLOAD_LEASES']).read_text())
    assert cfg['prefix']==prefix and phase in cfg['snapshots']
    lease=cfg['snapshots'][phase]
    done=root/f'{phase}_durable_upload.json'
    if done.exists():
        previous=json.loads(done.read_text())
        assert previous['status']=='BUNDLE_AND_RECEIPT_UPLOADED' and previous['completed_epoch']==completed_epoch
        return
    allowed={f'{prefix}_{suffix}' for suffix in ['last.pt','ema.pt','history.json','split.json',
             'manifest.json','effective_weights.json','oof_predictions.csv']}
    sources=sorted(p for p in root.iterdir() if p.is_file() and p.name in allowed)
    assert sources and (root/f'{prefix}_last.pt') in sources
    assert sum(p.stat().st_size for p in sources)<BUNDLE_BYTES-2**20
    bundle=root/lease['bundle']['name']
    with tarfile.open(bundle,'w') as tar:
        for p in sources:tar.add(p,arcname=p.name,recursive=False)
    assert bundle.stat().st_size<=BUNDLE_BYTES
    with bundle.open('ab') as f:f.truncate(BUNDLE_BYTES)
    bundle_hash=digest(bundle)
    started=time.time()
    bundle_receipt=root/f'{phase}_bundle_upload.json'
    if bundle_receipt.exists():
        prior=json.loads(bundle_receipt.read_text())
        assert prior['status']=='UPLOAD_COMPLETE' and prior['sha256']==bundle_hash
    else:upload(bundle,{**lease['bundle'],'sha256':bundle_hash},bundle_receipt)
    receipt={'status':'CHECKPOINT_SNAPSHOT_UPLOADED','phase':phase,'prefix':prefix,
             'completed_epoch':completed_epoch,'bundle_sha256':bundle_hash,'bundle_bytes':BUNDLE_BYTES,
             'members':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sources},
             'competition_submission_enabled':False}
    receipt_file=root/lease['receipt']['name']
    data=(json.dumps(receipt,indent=2)+'\n').encode()
    assert len(data)<RECEIPT_BYTES
    receipt_file.write_bytes(data+b' '*(RECEIPT_BYTES-len(data)))
    receipt_hash=digest(receipt_file)
    upload(receipt_file,{**lease['receipt'],'sha256':receipt_hash},root/f'{phase}_receipt_upload.json')
    done.write_text(json.dumps({'status':'BUNDLE_AND_RECEIPT_UPLOADED','phase':phase,
        'completed_epoch':completed_epoch,'bundle_sha256':bundle_hash,'receipt_sha256':receipt_hash,
        'seconds':time.time()-started},indent=2)+'\n')
    bundle.unlink()
    print(json.dumps({'event':'durable_snapshot_uploaded','phase':phase,'epoch':completed_epoch}),flush=True)
