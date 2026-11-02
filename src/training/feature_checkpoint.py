"""Atomic completion journal for a persistent, incrementally written feature array."""
import hashlib
import os
from pathlib import Path
import numpy as np


def row_digest(feature,mask):
    return hashlib.sha256(np.asarray(feature).tobytes()+np.asarray(mask,dtype=bool).tobytes()).hexdigest().encode()


def save_checkpoint(path,features,ids,masks,complete,digests,feature_key):
    path=Path(path)
    assert len(features)==len(ids)==len(masks)==len(complete)==len(digests)
    assert masks.dtype==complete.dtype==np.bool_ and digests.dtype==np.dtype('S64')
    assert np.all(digests[complete]!=b'')
    features.flush()
    with Path(features.filename).open('r+b') as handle:os.fsync(handle.fileno())
    temporary=path.with_name(path.name+'.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle,study_ids=ids,mask=masks,complete=complete,
                            row_sha256=digests,feature_key=np.asarray(feature_key),
                            feature_shape=np.asarray(features.shape),feature_dtype=np.asarray(str(features.dtype)))
        handle.flush();os.fsync(handle.fileno())
    os.replace(temporary,path)


def recover_checkpoint(path,features,expected_ids,expected_feature_key):
    with np.load(path,allow_pickle=False) as data:
        assert data['feature_key'].item()==expected_feature_key,'feature contract differs'
        assert np.array_equal(data['study_ids'],expected_ids),'study ordering differs'
        assert tuple(data['feature_shape'])==features.shape,'feature shape differs'
        assert data['feature_dtype'].item()==str(features.dtype),'feature dtype differs'
        masks=data['mask'].copy();complete=data['complete'].copy();digests=data['row_sha256'].copy()
    assert masks.dtype==complete.dtype==np.bool_ and digests.dtype==np.dtype('S64')
    assert len(features)==len(masks)==len(complete)==len(digests)
    for row in np.flatnonzero(complete):
        assert masks[row].any(),'completed row has no valid acquisition'
        assert np.isfinite(features[row]).all(),'nonfinite completed feature'
        assert np.count_nonzero(features[row][~masks[row]])==0,'masked feature is nonzero'
        assert row_digest(features[row],masks[row])==digests[row],f'completed feature checksum differs at row{row}'
    return masks,complete,digests
