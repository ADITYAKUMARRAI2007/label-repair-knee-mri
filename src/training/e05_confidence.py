"""Training-only E04 confidence ablation; held-out rows cannot set weights."""
import numpy as np


def uniform_weak_weights(original, expert, training_rows):
    original = np.asarray(original)
    expert = np.asarray(expert, dtype=bool)
    rows = np.asarray(training_rows, dtype=np.int64)
    if original.ndim != 2 or expert.shape != (len(original),):
        raise ValueError('Invalid weight/expert shapes')
    if rows.ndim != 1 or len(rows) != len(np.unique(rows)):
        raise ValueError('Training indices must be unique')
    if len(rows) == 0 or rows.min() < 0 or rows.max() >= len(original):
        raise ValueError('Invalid training indices')
    if not np.isfinite(original).all() or (original <= 0).any():
        raise ValueError('Weights must be finite and positive')
    weak_rows = rows[~expert[rows]]
    if len(weak_rows) == 0:
        raise ValueError('No weak training rows')
    # Preserve per-target total training weight and expert weighting. Only the
    # relative weighting of weak examples changes, not the optimizer loss scale.
    means = original[weak_rows].mean(axis=0, dtype=np.float64)
    result = original.copy()
    result[weak_rows] = means
    return result, means
