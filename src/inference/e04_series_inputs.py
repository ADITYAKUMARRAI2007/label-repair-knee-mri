"""Pack deterministic evaluation inputs identically to the owned fold trainer."""
import numpy as np


def make_eval_arrays(volume, mask, per_slot=8):
    # select_slot_windows is embedded verbatim from the frozen training source.
    if volume.shape != (44, 336, 336) or volume.dtype != np.uint8:
        raise ValueError("expected the owned 44 x 336 x 336 uint8 pixel contract")
    mask = np.asarray(mask)
    if mask.shape != (44,) or not np.isin(mask, (0, 1)).all() or per_slot != 8:
        raise ValueError("expected a binary 44-slice mask and eight evaluation windows per slot")
    centers, present, positions = select_slot_windows(mask.tolist(), per_slot, False)
    x = np.zeros((5, per_slot, 3, 336, 336), dtype=np.uint8)
    for slot, picks in enumerate(centers):
        for i, center in enumerate(picks):
            if center is not None:
                x[slot, i] = volume[center - 1:center + 2]
    return x, np.asarray(present, dtype=np.bool_), np.asarray(positions, dtype=np.float32)
