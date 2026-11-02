#!/usr/bin/env python3
"""Check the E03 series-boundary and missing-acquisition input risks."""
import importlib.util
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e03_windows", ROOT / "training/e03_window_contract.py")
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)
# Geometry is independently pinned by the Raptor 44-slice cache contract.
ranges = [(0, 12), (12, 22), (22, 30), (30, 36), (36, 44)]
rng = random.Random(20261006)
patterns = []
for presence in range(32):
    patterns.append([int(bool(presence & (1 << s)))
                     for s, (a, b) in enumerate(ranges) for _ in range(a, b)])
patterns.extend([[int(i != missing) for i in range(44)] for missing in range(44)])
patterns.extend([[int(rng.random() > 0.25) for _ in range(44)] for _ in range(1024)])
checked, rejected, sampled = 0, 0, 0
for mask in patterns:
    valid_slots = [any(all(mask[j] for j in (c - 1, c, c + 1))
                       for c in range(a + 1, b - 1)) for a, b in ranges]
    for training in (False, True):
        for count in (1, 4, 8):
            checked += 1
            try:
                centers, present, positions = contract.select_slot_windows(
                    mask, count, training, random.Random(42))
            except ValueError:
                assert not any(valid_slots), "readable study was rejected"
                rejected += 1
                continue
            assert present == valid_slots
            assert len(centers) == len(positions) == 5
            for slot, (a, b) in enumerate(ranges):
                assert len(centers[slot]) == len(positions[slot]) == count
                for center, pos in zip(centers[slot], positions[slot]):
                    if not present[slot]:
                        assert center is None and pos == 0
                    else:
                        assert a <= center - 1 < center < center + 1 < b
                        assert all(mask[j] for j in (center - 1, center, center + 1))
                        assert 0 <= pos <= 1
                        sampled += 1
            if not training:
                assert (centers, present, positions) == contract.select_slot_windows(mask, count)
result = {"schema": "e03_window_check_v1", "status": "PASS", "cases": checked,
          "patterns": len(patterns), "unreadable_cases_rejected": rejected,
          "sampled_windows_checked": sampled,
          "scope": "window selection only; GPU forward, resources and scores remain unverified"}
(ROOT / "runs/e03_window_check.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
