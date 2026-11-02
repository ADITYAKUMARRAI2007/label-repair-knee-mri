"""Capture a validated build_cache return without intercepting allocation APIs.

The caller must supply a separately verified preprocessing contract. This module
does not infer preprocessing equivalence from array shape or run inference.
"""
import numpy as np


class CacheCapture:
    def __init__(self, expected_config):
        if not expected_config:
            raise ValueError('verified preprocessing contract required')
        self.expected_config = expected_config
        self.record = None
        self.observations = []

    def observe(self, result, actual_config):
        if not isinstance(result, tuple) or len(result) != 3:
            raise RuntimeError('build_cache return contract differs')
        studies, cache, mask = result
        shape = tuple(getattr(cache, 'shape', ()))
        self.observations.append({'shape': list(shape), 'study_count': len(studies)})
        if len(shape) != 5 or shape[1:] != (6, 12, 336, 336):
            return result
        if actual_config != self.expected_config:
            raise RuntimeError('specialist preprocessing contract differs')
        ids = tuple(str(uid) for uid in studies)
        if not ids or len(set(ids)) != len(ids) or any(not uid for uid in ids):
            raise RuntimeError('invalid cache identities')
        if not isinstance(cache, np.ndarray) or cache.dtype != np.uint8 or shape[0] != len(ids):
            raise RuntimeError('invalid specialist cache')
        if (not isinstance(mask, np.ndarray) or mask.dtype != np.float32
                or mask.shape != (len(ids), 6) or not np.isfinite(mask).all()
                or np.any((mask < 0) | (mask > 1)) or np.any(mask.sum(1) <= 0)):
            raise RuntimeError('invalid specialist mask; empty studies cannot be neutralized')
        if self.record is not None:
            raise RuntimeError('ambiguous repeated specialist cache')
        # Retain the original arrays and exact row order. No copying/resampling.
        self.record = {'studies': ids, 'cache': cache, 'mask': mask,
                       'preprocessing': actual_config}
        return result

    def require_record(self):
        if self.record is None:
            raise RuntimeError('specialist cache absent; fallback prohibited')
        return self.record
