"""Series-contained window selection for the pinned 44-slice Raptor corpus.

Standard-library only, shared verbatim by the builder and local checks.
Slots identify the five cached acquisition positions, not scanner identities.
"""
import random

SLOT_RANGES = ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))


def select_slot_windows(mask, per_slot, training=False, rng=None):
    if len(mask) != 44 or per_slot < 1:
        raise ValueError("expected 44 mask entries and a positive window count")
    rng = rng or random
    selected, present, positions = [], [], []
    for start, stop in SLOT_RANGES:
        candidates = [c for c in range(start + 1, stop - 1)
                      if all(mask[j] > 0 for j in (c - 1, c, c + 1))]
        present.append(bool(candidates))
        if not candidates:
            selected.append([None] * per_slot)
            positions.append([0.0] * per_slot)
            continue
        picks = []
        for i in range(per_slot):
            if training:
                lo = i * len(candidates) // per_slot
                hi = max(lo + 1, (i + 1) * len(candidates) // per_slot)
                index = rng.randrange(lo, hi)
            else:
                index = round(i * (len(candidates) - 1) / max(per_slot - 1, 1))
            picks.append(candidates[index])
        selected.append(picks)
        positions.append([(c - start) / (stop - start - 1) for c in picks])
    if not any(present):
        raise ValueError("study has no readable series-contained three-slice window")
    return selected, present, positions
