"""Keep selected windows within one populated acquisition;do not resample."""
import operator

import numpy as np


def coherent_centers(centers, mask, slot_sizes):
    sizes = [operator.index(n) for n in slot_sizes]
    if not sizes or any(n <= 0 for n in sizes):
        raise ValueError('Acquisition slot sizes must be positive integers')
    mask = np.asarray(mask)
    if mask.ndim != 1 or len(mask) != sum(sizes) or not np.isin(mask, [0, 1]).all():
        raise ValueError('Slice mask does not match acquisition slots')
    slot = np.repeat(np.arange(len(sizes)), sizes)
    kept = []
    for value in centers:
        center = operator.index(value)
        if center <= 0 or center >= len(mask)-1:
            raise ValueError('Triplet center is outside the valid slice range')
        indices = [center-1, center, center+1]
        if mask[indices].all() and np.all(slot[indices] == slot[center]):
            kept.append(center)
    if not kept:
        raise ValueError('No populated triplet inside an acquisition')
    return kept
