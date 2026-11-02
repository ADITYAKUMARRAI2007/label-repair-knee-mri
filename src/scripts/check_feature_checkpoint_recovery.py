"""Reproduce an interrupted writer and verify safe recovery/corruption rejection."""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'training'))
from feature_checkpoint import save_checkpoint,recover_checkpoint,row_digest

def must_fail(fn,phrase):
    try:fn()
    except AssertionError as error:
        assert phrase in str(error);return
    raise AssertionError('expected rejection: '+phrase)

def main():
    out=ROOT/'runs/orthodiffusion_trial_v2';out.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out,prefix='checkpoint_test_') as name:
        directory=Path(name);path=directory/'features.npy';journal=directory/'feature_checkpoint.npz'
        features=np.lib.format.open_memmap(path,mode='w+',dtype=np.float16,shape=(4,3,256,256))
        ids=np.array(['a','b','c','d']);masks=np.ones((4,3),bool);complete=np.zeros(4,bool)
        digests=np.zeros(4,dtype='S64');key='frozen-feature-key'
        for row in range(2):
            features[row]=row+.25;digests[row]=row_digest(features[row],masks[row]);complete[row]=True
        save_checkpoint(journal,features,ids,masks,complete,digests,key)
        child="import os,sys,numpy as np; x=np.load(sys.argv[1],mmap_mode='r+'); x[2]=17; x.flush(); os._exit(17)"
        result=subprocess.run([sys.executable,'-c',child,str(path)])
        assert result.returncode==17
        recovered_masks,recovered_complete,recovered_digest=recover_checkpoint(journal,features,ids,key)
        assert recovered_complete.tolist()==[True,True,False,False]
        assert np.all(features[2]==17) and not recovered_complete[2]
        for row in np.flatnonzero(~recovered_complete):
            features[row]=row+.25;recovered_digest[row]=row_digest(features[row],recovered_masks[row]);recovered_complete[row]=True
        save_checkpoint(journal,features,ids,recovered_masks,recovered_complete,recovered_digest,key)
        assert recover_checkpoint(journal,features,ids,key)[1].all()
        assert all(np.all(features[row]==row+.25) for row in range(4))
        must_fail(lambda:recover_checkpoint(journal,features,ids,'wrong-key'),'feature contract differs')
        must_fail(lambda:recover_checkpoint(journal,features,ids[::-1],key),'study ordering differs')
        features[0,0,0,0]=99;features.flush()
        must_fail(lambda:recover_checkpoint(journal,features,ids,key),'checksum differs')
    source=ROOT/'training/feature_checkpoint.py'
    record=dict(status='PASS_INTERRUPTED_WRITE_RECOVERY_AND_CORRUPTION_REJECTION',
        interrupted_child_exit=17,committed_rows_before_interruption=2,
        uncommitted_write_ignored=True,all4rows_restored=True,wrong_contract_rejected=True,
        wrong_order_rejected=True,tampered_committed_row_rejected=True,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (out/'checkpoint_recovery_check.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':main()
