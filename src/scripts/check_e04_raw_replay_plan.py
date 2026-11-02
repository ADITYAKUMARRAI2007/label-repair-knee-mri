#!/usr/bin/env python3
"""Check original validation batch preservation and bounded decoder failure paths."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    batches = load("e04_replay_batches", ROOT / "inference/e04_replay_batches.py")
    frozen = json.loads((ROOT / "runs/E04_contract/e04_contract.json").read_text())
    rows = {r["StudyInstanceUID"]: r for r in frozen["rows"]}
    # Use the actual cache ordering used by the training validation DataLoader.
    import numpy as np
    ids = np.concatenate([np.load(ROOT / f"runs/E04_wide_cache_{part}/e04_wide_{part}_ids.npy",
                                   allow_pickle=False) for part in ("base", "extension")]).astype(str).tolist()
    folds, experts = [], set()
    for fold in range(5):
        validation = [u for u in ids if rows[u]["fold"] == fold]
        heldout = [u for u in validation if rows[u]["expert"]]
        selected = batches.expert_replay_batches(validation, heldout)
        replayed = [u for _, selected_ids in selected for u in selected_ids]
        assert len(replayed) == len(set(replayed))
        assert set(heldout).issubset(replayed)
        assert all(start % 2 == 0 and selected_ids == validation[start:start + 2]
                   for start, selected_ids in selected)
        assert not experts.intersection(heldout)
        experts.update(heldout)
        folds.append({"fold": fold, "experts": len(heldout), "replay_rows": len(replayed),
                      "original_batches": len(selected)})
    assert len(experts) == 58
    assert batches.expert_replay_batches(["a", "b", "c"], ["c"]) == [(2, ["c"])]
    for validation, expert in ((["a", "a"], ["a"]), (["a"], ["b"]),
                               (["a"], ["a", "a"]), (["a"], [])):
        try:
            batches.expert_replay_batches(validation, expert)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid replay identities accepted")
    sys.path.insert(0, str(ROOT / "runs/E04_staged_raw_inference"))
    stream = load("e04_decode_stream_check", ROOT / "inference/e04_decode_stream.py")
    pools = []

    class Future:
        def __init__(self, pool, item):
            self.pool, self.item = pool, item

        def result(self):
            self.pool.pending -= 1
            if self.item[0] == self.pool.fail_at:
                raise RuntimeError("synthetic CPU decoding failure")
            return (self.item[0], None, None, {"StudyInstanceUID": self.item[1]})

        def cancel(self):
            return True

    class Pool:
        fail_at = None

        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 4 and kwargs["mp_context"].get_start_method() == "spawn"
            self.pending, self.maximum, self.closed = 0, 0, False
            pools.append(self)

        def submit(self, worker, item):
            self.pending += 1
            self.maximum = max(self.maximum, self.pending)
            return Future(self, item)

        def shutdown(self, **kwargs):
            assert kwargs == {"wait": True, "cancel_futures": True}
            self.closed = True

    stream.ProcessPoolExecutor = Pool
    items = [(i, f"study{i}") for i in range(29)]
    result = list(stream.decoded_studies(items, "unused.csv", "unused"))
    assert [r[0] for r in result] == list(range(29))
    assert pools[-1].maximum == 8 and pools[-1].closed
    Pool.fail_at = 4
    observed = []
    try:
        for item in stream.decoded_studies(items, "unused.csv", "unused"):
            observed.append(item[0])
    except RuntimeError as error:
        assert str(error) == "synthetic CPU decoding failure"
    else:
        raise AssertionError("decoding failure was silently replaced")
    assert observed == [0, 1, 2, 3] and pools[-1].closed and pools[-1].maximum == 8
    check = {"schema": "e04_raw_replay_plan_check_v1", "status": "PASS_LOCAL_PLANNING_ONLY",
             "expert_rows": len(experts), "folds": folds, "max_pending_decodes": 8,
             "decoder_exception_propagated_without_prediction_substitution": True,
             "scope": "real frozen validation roster and synthetic decoder scheduling; no DICOM/GPU replay claim",
             "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (ROOT / "inference/e04_replay_batches.py",
                                   ROOT / "inference/e04_decode_stream.py",
                                   ROOT / "runs/e04_raw_inference_source.json")},
             "competition_submission_enabled": False}
    (ROOT / "runs/E04_contract/raw_replay_plan_check.json").write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
