#!/usr/bin/env python3
"""Safely audit tensor names and shapes in the legacy MedicalNet checkpoint.

The public checkpoint uses PyTorch's legacy pickle plus raw-storage format.  A
restricted unpickler reconstructs tensor metadata only; it never imports torch,
executes arbitrary globals, or reads the raw tensor payload into memory.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pickle
from pathlib import Path


class StorageType:
    def __init__(self, name: str):
        self.name = name


class Storage:
    def __init__(self, persistent_id):
        self.persistent_id = persistent_id


class Tensor:
    def __init__(self, storage, offset, size, stride, *unused):
        self.storage = storage
        self.offset = int(offset)
        self.shape = tuple(int(value) for value in size)
        self.stride = tuple(int(value) for value in stride)


def rebuild_tensor(storage, offset, size, stride, *rest):
    return Tensor(storage, offset, size, stride, *rest)


class RestrictedTorchUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
            return rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return StorageType(name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, persistent_id):
        return Storage(persistent_id)


def batch_norm(prefix: str, channels: int) -> dict[str, tuple[int, ...]]:
    return {
        f"{prefix}.weight": (channels,),
        f"{prefix}.bias": (channels,),
        f"{prefix}.running_mean": (channels,),
        f"{prefix}.running_var": (channels,),
        f"{prefix}.num_batches_tracked": (),
    }


def expected_shapes() -> dict[str, tuple[int, ...]]:
    expected = {"conv1.weight": (64, 1, 7, 7, 7), **batch_norm("bn1", 64)}
    stages = ((1, 64, 64), (2, 64, 128), (3, 128, 256), (4, 256, 512))
    for stage, incoming, outgoing in stages:
        prefix = f"layer{stage}.0"
        expected[f"{prefix}.conv1.weight"] = (outgoing, incoming, 3, 3, 3)
        expected.update(batch_norm(f"{prefix}.bn1", outgoing))
        expected[f"{prefix}.conv2.weight"] = (outgoing, outgoing, 3, 3, 3)
        expected.update(batch_norm(f"{prefix}.bn2", outgoing))
        if incoming != outgoing:
            expected[f"{prefix}.downsample.0.weight"] = (outgoing, incoming, 1, 1, 1)
            expected.update(batch_norm(f"{prefix}.downsample.1", outgoing))
    return expected


def audit(path: Path) -> dict:
    with path.open("rb") as handle:
        magic = pickle.load(handle)
        protocol = pickle.load(handle)
        system_info = pickle.load(handle)
        payload = RestrictedTorchUnpickler(handle).load()
    if not isinstance(payload, dict) or set(payload) != {"state_dict"}:
        raise RuntimeError(f"unexpected checkpoint payload keys {getattr(payload, 'keys', lambda: [])()}")
    state = payload["state_dict"]
    actual = {}
    for name, tensor in state.items():
        if not isinstance(tensor, Tensor):
            raise RuntimeError(f"{name} did not decode as tensor metadata")
        clean = name.removeprefix("module.")
        actual[clean] = tensor.shape
    expected = expected_shapes()
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = {
        name: {"expected": expected[name], "actual": actual[name]}
        for name in sorted(set(expected) & set(actual))
        if expected[name] != actual[name]
    }
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "schema": "medicalnet_checkpoint_audit_v1",
        "status": "PASS" if not (missing or unexpected or mismatched) else "FAIL",
        "path": str(path),
        "sha256": digest,
        "bytes": path.stat().st_size,
        "legacy_magic": magic,
        "legacy_protocol": protocol,
        "system_info": system_info,
        "tensor_count": len(actual),
        "expected_tensor_count": len(expected),
        "missing": missing,
        "unexpected": unexpected,
        "mismatched": mismatched,
        "shapes": {name: list(shape) for name, shape in actual.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.checkpoint)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
