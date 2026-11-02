"""Verify the short-volume repair without using clinical target values."""
import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/orthodiffusion_volume_v2'
sys.path[:0]=[str(ROOT/'inference'),str(ROOT/'runs/orthodiffusion_volume_v1/test_dependencies')]
from pydicom.uid import generate_uid
from orthodiffusion_volume import inspect_series,read_volume as old_read,build_study as old_build
from orthodiffusion_volume_v2 import read_volume,build_study
from audit_orthodiffusion_volume import write_slice,must_fail

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    cv2.setNumThreads(1)
    original=json.loads((ROOT/'runs/orthodiffusion_volume_v1/synthetic_audit.json').read_text())
    assert sha(ROOT/'inference/orthodiffusion_volume.py')==original['reader_sha256']
    synthetic=[]
    with tempfile.TemporaryDirectory(dir=OUT,prefix='short_fixture_') as temp:
        for n in [1,11,12,13,14,15,16,17,21,32]:
            study,series=generate_uid(),generate_uid();directory=Path(temp)/study/series
            directory.mkdir(parents=True)
            raw=(np.arange(n)[:,None,None]*31+np.arange(9)[None,:,None]**2
                 +np.arange(13)[None,None,:]*3).astype(np.uint16)
            for i,pixels in enumerate(raw):
                write_slice(directory/f'{n-i:03}.dcm',study,series,generate_uid(),pixels,i*.8,n-i)
            files,meta=inspect_series(directory,study,series)
            volume,receipt=read_volume(files,meta)
            assert volume.shape==(16,256,256) and np.isfinite(volume).all()
            left=max(0,(16-n)//2);right=max(0,16-n-left)
            assert receipt['edge_padding_left']==left and receipt['edge_padding_right']==right
            assert receipt['native_decoded_slices']==min(n,16)
            if n>=16:
                prior,_=old_read(files,meta);assert np.array_equal(prior,volume)
                error=0.
            else:
                must_fail(lambda:old_read(files,meta),'at least16')
                pixels=raw.astype(np.float32)*.75+9
                resized=np.stack([cv2.resize(x,(256,256),interpolation=cv2.INTER_LINEAR) for x in pixels])
                expected=(resized-resized.min())/(resized.max()-resized.min()+1e-8)*2-1
                expected=np.pad(expected,((left,right),(0,0),(0,0)),mode='edge')
                error=float(np.abs(volume-expected).max());assert error==0
            synthetic.append(dict(native_depth=n,padding=[left,right],max_error=error))
    listings=json.loads((OUT/'failed_study_file_listings.json').read_text())
    downloaded=json.loads((OUT/'raw_download_receipt.json').read_text())
    for item in downloaded['files']:
        path=ROOT/item['file'];assert sha(path)==item['sha256'] and path.stat().st_size==item['bytes']
    rows=[r['metadata'] for r in listings];uid=rows[0]['StudyInstanceUID']
    series_root=OUT/'raw_failed_study/train_series'
    must_fail(lambda:old_build(uid,rows,series_root),'no eligible contiguous16 volume')
    volumes,mask,receipt=build_study(uid,rows,series_root)
    assert mask.tolist()==[True,True,True]
    again,mask_again,receipt_again=build_study(uid,rows[::-1],series_root)
    assert np.array_equal(volumes,again) and np.array_equal(mask,mask_again)
    assert receipt==receipt_again
    for r in receipt['planes']:
        assert r['native_depth']==15 and r['edge_padding_left']==0 and r['edge_padding_right']==1
        assert r['decode_failures']==0 and not r['depth_resampling'] and not r['spatial_crop']
    np.savez_compressed(OUT/'repaired_real_volume_fixture.npz',volume=volumes,mask=mask)
    (OUT/'repaired_real_volume_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    result=dict(status='PASS_SHORT_VOLUME_REPAIR_AND_UNCHANGED_LONG_VOLUME_PIXELS',
        reader_sha256=sha(ROOT/'inference/orthodiffusion_volume_v2.py'),
        original_reader_sha256=sha(ROOT/'inference/orthodiffusion_volume.py'),
        synthetic_depth_cases=synthetic,real_original_failure_reproduced=True,
        real_native_depths=[r['native_depth'] for r in receipt['planes']],
        real_planes_present=mask.tolist(),real_replay_max_error=0,
        real_fixture_sha256=sha(OUT/'repaired_real_volume_fixture.npz'),
        targets_read=False,accuracy_measured=False,
        policy='Preserve all original >=16-series selections; otherwise edge-pad the best valid short series for that plane.',
        limitations=['This changes formerly missing short-plane inputs and requires a new fixed accuracy evaluation',
                     'Padding repeats measured edge pixels; no new anatomical information is created',
                     'Other strict geometry and decode guards remain active'])
    (OUT/'repair_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
