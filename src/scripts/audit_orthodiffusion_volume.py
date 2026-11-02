"""Verify actual DICOM decoding and compare with the pinned author NIfTI transform."""
import ast
import hashlib
import json
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/orthodiffusion_volume_v1'
sys.path[:0]=[str(ROOT/'inference'),str(OUT/'test_dependencies')]
import pydicom
from pydicom.dataset import FileDataset,FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian,MRImageStorage,generate_uid
from orthodiffusion_volume import (inspect_series,read_volume,build_study,
                                   ranked_series,series_noise_seed)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def write_slice(path,study,series,sop,pixels,position,instance,photo='MONOCHROME2'):
    meta=FileMetaDataset()
    meta.MediaStorageSOPClassUID=MRImageStorage;meta.MediaStorageSOPInstanceUID=sop
    meta.TransferSyntaxUID=ExplicitVRLittleEndian
    ds=FileDataset(str(path),{},file_meta=meta,preamble=b'\0'*128)
    ds.SOPClassUID=MRImageStorage;ds.SOPInstanceUID=sop
    ds.StudyInstanceUID=study;ds.SeriesInstanceUID=series;ds.Modality='MR'
    ds.Rows,ds.Columns=pixels.shape;ds.SamplesPerPixel=1
    ds.PhotometricInterpretation=photo
    ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0
    ds.ImageOrientationPatient=[1,0,0,0,1,0];ds.ImagePositionPatient=[0,0,position]
    ds.PixelSpacing=[.6,.8];ds.InstanceNumber=instance
    ds.RescaleSlope=.75;ds.RescaleIntercept=9
    ds.PixelData=pixels.astype('<u2').tobytes()
    ds.save_as(path,enforce_file_format=True)

def must_fail(fn,contains):
    try:fn()
    except ValueError as e:
        assert contains in str(e),(contains,str(e))
        return
    raise AssertionError('expected rejection: '+contains)

def main():
    torch.set_num_threads(2);cv2.setNumThreads(1)
    source=ROOT/'research/knee_pretraining/orthodiffusion/source'
    receipt=json.loads((source.parent/'source_receipt.json').read_text())
    for name in ['dataset.py','linear_classifier.py']:
        expected=next(i['sha256'] for i in receipt['files'] if i['path']==name)
        assert sha(source/name)==expected
    node=next(n for n in ast.parse((source/'dataset.py').read_text()).body
              if isinstance(n,ast.ClassDef) and n.name=='NiftiImageLabelDataset')
    env=dict(Dataset=torch.utils.data.Dataset,torch=torch,cv2=cv2,np=np)
    exec(compile(ast.Module(body=[node],type_ignores=[]),'author_dataset.py','exec'),env)
    reference=env['NiftiImageLabelDataset'].__new__(env['NiftiImageLabelDataset'])
    reference.input_size=256;reference.depth_size=16
    reference.samples=[dict(path='synthetic.nii',label=torch.zeros(8),id='synthetic')]
    # Verbatim arithmetic/order from the author's linear_classifier transform.
    def author_transform(array):
        t=torch.tensor(array).float()
        t=(t-t.min())/(t.max()-t.min()+1e-8)
        return (t*2-1).permute(2,0,1).unsqueeze(0)
    reference.transform=author_transform
    raw=(np.arange(21)[:,None,None]*31+np.arange(9)[None,:,None]**2
         +np.arange(13)[None,None,:]*3).astype(np.uint16)
    reference.read_image=lambda _: (raw.astype(np.float64)*.75+9).transpose(1,2,0)
    expected=reference[0][0].numpy()[0]
    checks={};max_error=None
    with tempfile.TemporaryDirectory(dir=OUT,prefix='dicom_fixture_') as temporary:
        root=Path(temporary);study=generate_uid();series=generate_uid()
        directory=root/study/series;directory.mkdir(parents=True)
        sops=[generate_uid() for _ in raw]
        for i,pixels in enumerate(raw):
            # Filenames and InstanceNumber intentionally disagree with spatial order.
            write_slice(directory/f'{20-i:03}.dcm',study,series,sops[i],pixels,
                        i*.8,100-i)
        files,meta=inspect_series(directory,study,series)
        actual,detail=read_volume(files,meta)
        max_error=float(np.max(np.abs(actual-expected)))
        assert max_error<=5e-7,max_error
        assert detail['central_start']==2 and detail['central_stop']==18
        assert detail['selected_files'][0]=='018.dcm' and detail['selected_files'][-1]=='003.dcm'
        checks['spatial_order_and_odd_depth_crop']=True
        checks['author_nifti_resize_and_normalization_parity']=True
        rows=[dict(StudyInstanceUID=study,SeriesInstanceUID=series,
                   Anatomical_Plane='Axial',Fluid_Sensitive='1',Fat_Suppression='1')]
        batch,mask,report=build_study(study,rows,root)
        assert mask.tolist()==[True,False,False]
        assert np.array_equal(batch[0],actual) and not batch[1:].any()
        checks['missing_plane_mask_and_no_prediction_fallback']=not report['prediction_fallback_used']
        altered=directory/'020.dcm'
        original=altered.read_bytes()
        ds=pydicom.dcmread(altered);ds.ImagePositionPatient=[0,0,.8]
        ds.save_as(altered,enforce_file_format=True)
        must_fail(lambda:inspect_series(directory,study,series),'duplicate or indistinguishable')
        altered.write_bytes(original)
        checks['duplicate_position_rejected']=True
        ds=pydicom.dcmread(altered);ds.ImagePositionPatient=[0,0,-1.6]
        ds.save_as(altered,enforce_file_format=True)
        must_fail(lambda:inspect_series(directory,study,series),'nonuniform slice spacing')
        altered.write_bytes(original)
        checks['missing_slice_or_spacing_gap_rejected']=True
        must_fail(lambda:inspect_series(directory,generate_uid(),series),'study identity mismatch')
        checks['wrong_study_rejected']=True
        for path in files:
            ds=pydicom.dcmread(path);ds.PhotometricInterpretation='MONOCHROME1'
            ds.save_as(path,enforce_file_format=True)
        mono_files,mono_meta=inspect_series(directory,study,series)
        inverted,_=read_volume(mono_files,mono_meta)
        assert np.max(np.abs(inverted+actual))<=1.3e-7
        checks['monochrome1_global_inversion']=True
        must_fail(lambda:read_volume(files[:15],meta),'at least16')
        checks['short_series_not_depth_interpolated']=True
        for path in files:
            ds=pydicom.dcmread(path);ds.PixelData=np.full((9,13),12,dtype='<u2').tobytes()
            ds.save_as(path,enforce_file_format=True)
        must_fail(lambda:read_volume(mono_files,mono_meta),'constant MRI volume')
        checks['constant_volume_rejected']=True
    rows=[dict(Anatomical_Plane='Axial',SeriesInstanceUID='z',Fluid_Sensitive='0',Fat_Suppression='0'),
          dict(Anatomical_Plane='Axial',SeriesInstanceUID='b',Fluid_Sensitive='1',Fat_Suppression='1'),
          dict(Anatomical_Plane='Axial',SeriesInstanceUID='a',Fluid_Sensitive='1',Fat_Suppression='1')]
    assert [r['SeriesInstanceUID'] for r in ranked_series(rows[::-1],'Axial')]==['a','b','z']
    checks['series_selection_independent_of_csv_order']=True
    assert series_noise_seed('study','series')==series_noise_seed('study','series')
    assert series_noise_seed('study','series')!=series_noise_seed('study','different')
    checks['noise_seed_keyed_by_study_and_series']=True
    result=dict(status='PASS_SYNTHETIC_DICOM_AND_AUTHOR_ARRAY_TRANSFORM_ONLY',checks=checks,
                author_reference_max_abs_error=max_error,pydicom=pydicom.__version__,cv2=cv2.__version__,
                reader_sha256=sha(ROOT/'inference/orthodiffusion_volume.py'),
                audit_sha256=sha(Path(__file__)),source_commit=receipt['commit'],
                real_competition_dicom_validated=False,accuracy_measured=False,
                limitations=['Uncompressed synthetic DICOM only; compressed real files still require a platform check',
                             'Author upstream DICOM-to-NIfTI axis orientation is unpublished here; acquisition axes are preserved',
                             'Real-series eligibility, geometry rejection rate and decode throughput unmeasured'])
    (OUT/'synthetic_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
