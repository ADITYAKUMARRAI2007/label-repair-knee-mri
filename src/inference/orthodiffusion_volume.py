"""Contiguous DICOM volumes for the original OrthoDiffusion encoder.

This is an RSNA data adaptation, not a claim of reproducing the unavailable
author DICOM-to-NIfTI orientation conversion. Preserve acquisition row/column
directions; sort depth by the shared DICOM slice normal. Never use sparse cache
slices, interpolate depth, crop field of view, or silently replace bad pixels.
"""
import hashlib
from pathlib import Path
import numpy as np

PLANES = ('Axial', 'Coronal', 'Sagittal')
DEPTH = 16
SIZE = 256


def flag(value):
    text = str(value).strip().lower()
    if text in {'1','1.0','true'}: return 1
    if text in {'0','0.0','false'}: return 0
    return None


def ranked_series(records, plane):
    rows = [r for r in records if r['Anatomical_Plane'] == plane]
    return sorted(rows, key=lambda r: (
        flag(r.get('Fluid_Sensitive')) != 1,
        flag(r.get('Fat_Suppression')) != 1,
        r['SeriesInstanceUID']))


def inspect_series(directory, expected_study=None, expected_series=None):
    import pydicom
    files = sorted(Path(directory).glob('*.dcm'))
    if not files: raise ValueError('empty DICOM series')
    records=[];reference=None;positions=[];sop_ids=set()
    for path in files:
        h=pydicom.dcmread(path,stop_before_pixels=True)
        if expected_study is not None and str(h.StudyInstanceUID)!=expected_study:
            raise ValueError('DICOM study identity mismatch')
        if expected_series is not None and str(h.SeriesInstanceUID)!=expected_series:
            raise ValueError('DICOM series identity mismatch')
        uid=str(h.SOPInstanceUID)
        if uid in sop_ids:raise ValueError('duplicate SOP instance')
        sop_ids.add(uid)
        if int(getattr(h,'NumberOfFrames',1))!=1:
            raise ValueError('multi-frame DICOM requires a separate geometry adapter')
        orientation=np.asarray(h.ImageOrientationPatient,dtype=float)
        position=np.asarray(h.ImagePositionPatient,dtype=float)
        spacing=np.asarray(h.PixelSpacing,dtype=float)
        if orientation.shape!=(6,) or position.shape!=(3,) or spacing.shape!=(2,):
            raise ValueError('missing or invalid DICOM geometry')
        if not (np.isfinite(orientation).all() and np.isfinite(position).all()
                and np.isfinite(spacing).all() and (spacing>0).all()):
            raise ValueError('nonfinite/nonpositive DICOM geometry')
        col_dir,row_dir=orientation[:3],orientation[3:]
        if (abs(np.linalg.norm(col_dir)-1)>.001 or abs(np.linalg.norm(row_dir)-1)>.001
                or abs(np.dot(col_dir,row_dir))>.001):
            raise ValueError('invalid DICOM direction cosines')
        dimensions=(int(h.Rows),int(h.Columns))
        photo=str(h.PhotometricInterpretation)
        if photo not in {'MONOCHROME1','MONOCHROME2'}:
            raise ValueError('unsupported photometric interpretation')
        if reference is None:
            normal=np.cross(col_dir,row_dir);normal/=np.linalg.norm(normal)
            reference=(orientation,spacing,dimensions,photo,normal)
        first_iop,first_spacing,first_shape,first_photo,normal=reference
        if (not np.allclose(orientation,first_iop,rtol=0,atol=.001)
                or not np.allclose(spacing,first_spacing,rtol=.001,atol=.00001)
                or dimensions!=first_shape or photo!=first_photo):
            raise ValueError('inconsistent within-series geometry or photometric interpretation')
        coordinate=float(position@normal)
        records.append((coordinate,path))
    records.sort(key=lambda x:x[0])
    positions=np.array([x[0] for x in records])
    gaps=np.diff(positions)
    if len(gaps):
        if np.any(gaps<=.001):raise ValueError('duplicate or indistinguishable slice positions')
        median=float(np.median(gaps))
        if np.max(np.abs(gaps-median))>max(.01,.1*median):
            raise ValueError('nonuniform slice spacing or missing interior slices')
    else:median=None
    receipt=dict(native_slices=len(records),orientation=reference[0].tolist(),
                 spacing_mm=reference[1].tolist(),native_shape=list(reference[2]),
                 photometric=reference[3],slice_normal=reference[4].tolist(),
                 median_slice_spacing_mm=median,position_range_mm=[float(positions[0]),float(positions[-1])],
                 depth_order='ascending dot(ImagePositionPatient, shared slice normal)',
                 in_plane_orientation='preserved acquisition rows and columns',
                 geometry_fallbacks=0)
    return [p for _,p in records],receipt


def read_volume(files, receipt):
    import cv2
    import pydicom
    from pydicom.pixels import apply_modality_lut
    if len(files)<DEPTH:raise ValueError('at least16 contiguous native slices required')
    start=len(files)//2-DEPTH//2
    selected=files[start:start+DEPTH]
    arrays=[]
    for path in selected:
        ds=pydicom.dcmread(path)
        image=apply_modality_lut(ds.pixel_array,ds).astype(np.float32)
        if image.ndim!=2 or image.shape!=tuple(receipt['native_shape']) or not np.isfinite(image).all():
            raise ValueError('invalid decoded DICOM pixels')
        arrays.append(cv2.resize(image,(SIZE,SIZE),interpolation=cv2.INTER_LINEAR)
                      if image.shape!=(SIZE,SIZE) else image)
    volume=np.stack(arrays).astype(np.float32)
    low,high=float(volume.min()),float(volume.max())
    if high<=low:raise ValueError('constant MRI volume')
    # Global, post-resize normalization matches the released NIfTI dataset
    # transform; MONOCHROME1 reversal is required for DICOM display semantics.
    volume=(volume-low)/(high-low+1e-8)
    if receipt['photometric']=='MONOCHROME1':volume=1-volume
    volume=np.ascontiguousarray(volume*2-1,dtype=np.float32)
    if volume.shape!=(16,256,256) or not np.isfinite(volume).all():
        raise ValueError('invalid normalized volume')
    result=dict(receipt,central_start=start,central_stop=start+DEPTH,
                selected_files=[p.name for p in selected],minimum_before_normalization=low,
                maximum_before_normalization=high,volume_sha256=hashlib.sha256(volume.tobytes()).hexdigest(),
                decoded_slices=16,decode_failures=0,depth_resampling=False,spatial_crop=False)
    return volume,result


def build_study(study_id, records, series_root):
    volumes=np.zeros((3,16,256,256),dtype=np.float32)
    mask=np.zeros(3,dtype=bool)
    receipts=[]
    for index,plane in enumerate(PLANES):
        skipped=[];chosen=None
        for row in ranked_series(records,plane):
            series_id=row['SeriesInstanceUID']
            files,meta=inspect_series(Path(series_root)/study_id/series_id,study_id,series_id)
            if len(files)<16:
                skipped.append(dict(series=series_id,slices=len(files),reason='fewer_than16_native_slices'))
                continue
            volume,receipt=read_volume(files,meta)
            volumes[index]=volume;mask[index]=True
            chosen=dict(receipt,plane=plane,SeriesInstanceUID=series_id,
                        Fluid_Sensitive=flag(row.get('Fluid_Sensitive')),
                        Fat_Suppression=flag(row.get('Fat_Suppression')),
                        preferred_contrast=flag(row.get('Fluid_Sensitive'))==1,
                        skipped_short_series=skipped,state='decoded')
            break
        receipts.append(chosen or dict(plane=plane,state='missing_eligible_volume',skipped_short_series=skipped))
    if not mask.any():raise ValueError('study has no eligible contiguous16 volume')
    return volumes,mask,dict(StudyInstanceUID=study_id,planes=receipts,
                            present_planes=int(mask.sum()),prediction_fallback_used=False)


def series_noise_seed(study_id,series_id,base_seed=20261016):
    message=f'{base_seed}\0{study_id}\0{series_id}'.encode()
    return int.from_bytes(hashlib.sha256(message).digest()[:8],'big')%(2**63-1)
