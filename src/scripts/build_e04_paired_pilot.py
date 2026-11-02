#!/usr/bin/env python3
"""Build a bounded paired probe without changing the completed single pilot."""
import ast
import hashlib
import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"source contract not unique: {old}")
    return text.replace(old, new)


def main():
    source_dir = ROOT / "public_kernels/e04_convnext_resource_pilot"
    base_path = source_dir / "rsna-knee-e04-convnext-resource-pilot.py"
    base = base_path.read_text()
    if hashlib.sha256(base.encode()).hexdigest() != "498faaa8c17707d43f8f5fd06ae864cc5950933a18461e358f55e5ea5c97a0c4":
        raise RuntimeError("completed pilot source drift")
    worker = replace_once(base, "SEED = 20261007", "PILOT_FOLD = int(os.environ['E04_PILOT_FOLD'])\nSEED = 20261007 + PILOT_FOLD")
    worker = replace_once(worker, "seed_all(SEED)\n    torch.backends", "seed_all(SEED)\n    torch.set_num_threads(2)\n    torch.backends")
    worker = replace_once(worker, "fold_row_sets(ids, gold, folds, groups, 0)", "fold_row_sets(ids, gold, folds, groups, PILOT_FOLD)")
    worker = replace_once(worker, '"heldout_fold": 0', '"heldout_fold": PILOT_FOLD')
    worker = replace_once(worker, 'out = Path("/kaggle/working")', 'out = Path(os.environ["E04_PILOT_OUTPUT"])\n    out.mkdir(parents=True, exist_ok=True)')
    worker = replace_once(worker,
        'for step, (x, mask, pos, y, row) in enumerate(loader, 1):\n        before = time.time()',
        'iterator = iter(loader)\n    for step in range(1, PILOT_STEPS + 1):\n        before = time.time()\n        x, mask, pos, y, row = next(iterator)')
    worker = replace_once(worker, 'growth < 2 and full_five_hours < 17 and current_rss_gib() < 20',
                          'growth < 2 and current_rss_gib() < 20')
    worker = replace_once(worker, '"PASS" if accepted else "REQUIRES_RESOURCE_REDESIGN"',
                          '"PASS_CONCURRENT_MEMORY_NUMERICS" if accepted else "REQUIRES_RESOURCE_REDESIGN"')
    worker = replace_once(worker, '"schema": "e04_convnext_resource_pilot_v1"',
                          '"schema": "e04_convnext_paired_worker_v1"')
    worker = replace_once(worker, '"gradient_checkpointing": True,',
                          '"gradient_checkpointing": True, "visible_gpu_count": torch.cuda.device_count(),\n              "assigned_physical_gpu": os.environ["CUDA_VISIBLE_DEVICES"],')
    worker = worker.replace('# Compute plus sampled I/O is a conservative estimate, not a full epoch measurement.',
                            '# Includes DataLoader time; startup makes this a conservative short-probe mean.')
    ast.parse(worker)
    directory = ROOT / "public_kernels/e04_paired_resource_pilot"
    directory.mkdir(parents=True, exist_ok=True)
    worker_path = directory / "e04_pair_worker_preview.py"
    worker_path.write_text(worker)
    py_compile.compile(str(worker_path), doraise=True)
    template = ROOT / "training/e04_paired_resource_supervisor.py"
    worker_sha = hashlib.sha256(worker.encode()).hexdigest()
    code = template.read_text().replace("# __WORKER_SOURCE__",
               f"WORKER_SOURCE = {worker!r}\nWORKER_SHA256 = {worker_sha!r}")
    dest = directory / "rsna-knee-e04-paired-resource-pilot.py"
    dest.write_text(code)
    py_compile.compile(str(dest), doraise=True)
    metadata = json.loads((source_dir / "kernel-metadata.json").read_text())
    metadata.update(id="adityakumarrai0401/rsna-knee-e04-paired-resource-pilot",
                    title="RSNA Knee E04 Paired Resource Pilot", code_file=dest.name)
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    receipt = {"schema": "e04_paired_pilot_build_v1", "syntax": "PASS", "submission_enabled": False,
               "accuracy_measured": False, "worker_source_sha256": worker_sha,
               "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (base_path, template, dest)}}
    (ROOT / "runs/e04_paired_pilot_source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
