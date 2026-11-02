"""Combine aligned study predictions with fixed weights; never fit weights."""
import numpy as np
import pandas as pd


def rank_ensemble(components, weights, study_ids, targets):
    """Return average-rank scores, preserving requested study and target order.

    Each component is a DataFrame indexed by StudyInstanceUID. All components
    must cover exactly the same cohort; ranks depend on that cohort.
    """
    if set(components) != set(weights) or not components:
        raise ValueError('Every component needs exactly one fixed weight')
    if len(study_ids) != len(set(study_ids)) or not study_ids:
        raise ValueError('Study IDs must be unique and nonempty')
    if len(targets) != len(set(targets)) or not targets:
        raise ValueError('Targets must be unique and nonempty')
    ws = np.asarray(list(weights.values()), dtype=float)
    if not np.isfinite(ws).all() or (ws < 0).any() or not np.isclose(ws.sum(), 1):
        raise ValueError('Weights must be finite, nonnegative and sum to one')
    scores = pd.DataFrame(0., index=pd.Index(study_ids, name='StudyInstanceUID'), columns=targets)
    for name, frame in components.items():
        if not frame.index.is_unique or set(frame.index) != set(study_ids):
            raise ValueError(f'{name}: study identity mismatch')
        if not frame.columns.is_unique or set(frame.columns) != set(targets):
            raise ValueError(f'{name}: target identity mismatch')
        aligned = frame.loc[study_ids, targets].astype(float)
        values = aligned.to_numpy()
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f'{name}: invalid prediction scores')
        scores += weights[name] * aligned.rank(method='average', pct=True)
    return scores.rank(method='average', pct=True)
