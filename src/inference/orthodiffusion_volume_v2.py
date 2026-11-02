"""Extend the frozen volume reader to valid acquisitions shorter than16 slices.

Keep the original selection and pixels wherever a >=16-slice series exists.
For an otherwise missing plane, use the highest-ranked valid short series and
symmetrically replicate its edge slices to16. No depth interpolation, fabricated
slice positions, target-dependent selection, or silent decode-error recovery.
"""
from pathlib import Path
import numpy as np
from orthodiffusion_volume import (PLANES,DEPTH,SIZE,flag,ranked_series,inspect_series,
                                   read_volume as original_read_volume,series_noise_seed)


def read_volume(files,receipt):
    count=len(files)
    if count<1:raise ValueError('empty volume')
    left=max(0,(DEPTH-count)//2);right=max(0,DEPTH-count-left)
    padded=[files[0]]*left+list(files)+[files[-1]]*right
    volume,result=original_read_volume(padded,receipt)
    native_start=0 if count<DEPTH else result['central_start']
    native_stop=count if count<DEPTH else result['central_stop']
    result.update(native_depth=count,edge_padding_left=left,edge_padding_right=right,
                  edge_padding_used=bool(left or right),native_decoded_slices=min(count,DEPTH),
                  central_native_start=native_start,central_native_stop=native_stop,
                  effective_depth=DEPTH,depth_padding_policy='symmetric edge replication; odd extra goes after final slice')
    return volume,result


def build_study(study_id,records,series_root):
    volumes=np.zeros((3,DEPTH,SIZE,SIZE),np.float32);mask=np.zeros(3,bool);receipts=[]
    for index,plane in enumerate(PLANES):
        short=[];selected=None
        for row in ranked_series(records,plane):
            series_id=row['SeriesInstanceUID']
            files,meta=inspect_series(Path(series_root)/study_id/series_id,study_id,series_id)
            if len(files)<DEPTH:
                short.append((row,files,meta));continue
            selected=(row,files,meta);break
        if selected is None and short:selected=short[0]
        if selected is None:
            receipts.append(dict(plane=plane,state='missing_acquisition'));continue
        row,files,meta=selected
        volume,receipt=read_volume(files,meta);volumes[index]=volume;mask[index]=True
        receipt.update(plane=plane,SeriesInstanceUID=row['SeriesInstanceUID'],
                       Fluid_Sensitive=flag(row.get('Fluid_Sensitive')),
                       Fat_Suppression=flag(row.get('Fat_Suppression')),
                       preferred_contrast=flag(row.get('Fluid_Sensitive'))==1,
                       skipped_short_series=[dict(series=r['SeriesInstanceUID'],slices=len(f),
                         reason='prefer_native16_if_available') for r,f,m in short if r is not row],
                       state='decoded')
        receipts.append(receipt)
    if not mask.any():raise ValueError('study has no readable acquisition')
    return volumes,mask,dict(StudyInstanceUID=study_id,planes=receipts,present_planes=int(mask.sum()),
                            prediction_fallback_used=False)
