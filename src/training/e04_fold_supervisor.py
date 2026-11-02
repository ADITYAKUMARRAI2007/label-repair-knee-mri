#!/usr/bin/env python3
"""Hash the immutable owned cache once, then train bounded independent folds."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

# __FROZEN_JOB__
# __WORKER_SOURCE__


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rss_gib(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        if any(line.startswith("State:") and "Z" in line for line in lines):
            return 0.0
        for line in lines:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 2**20
    except FileNotFoundError:
        return 0.0
    raise RuntimeError("cannot inspect worker resident memory")


def last_progress(path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 8192))
        tail = f.read().decode(errors="replace")
    for line in reversed(tail.splitlines()):
        start = line.find("{")
        if start < 0:
            continue
        try:
            item = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if "fold" in item:
            return item
    return None


def main():
    started = time.time()
    out = Path("/kaggle/working")
    visible = torch.cuda.device_count()
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = physical.split(",") if physical else [str(i) for i in range(visible)]
    if visible < len(JOB_FOLDS) or len(set(devices)) < len(JOB_FOLDS):
        raise RuntimeError("independent fold GPU assignment unavailable")
    worker_path = out / "e04_full_fold_worker.py"
    worker_path.write_text(WORKER_SOURCE)
    if sha(worker_path) != WORKER_SHA256:
        raise RuntimeError("full worker source identity changed")
    print(json.dumps({"event": "verify_mounted_wide_cache", "folds": JOB_FOLDS,
                      "worker_sha256": WORKER_SHA256, "allocated_budget_hours": JOB_HOURS,
                      "competition_submission_enabled": False}), flush=True)
    # Loading definitions performs no model construction or training. Only this
    # parent hashes the complete 22-GB mounted volume bytes before workers start.
    os.environ["E04_TRAIN_FOLD"] = str(JOB_FOLDS[0])
    spec = importlib.util.spec_from_file_location("e04_worker_definitions", worker_path)
    definitions = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(definitions)
    verified = definitions.verify_wide_files()
    cache_path = out / "e04_mounted_cache_verification.json"
    cache_path.write_text(json.dumps(verified, indent=2) + "\n")
    cache_sha = sha(cache_path)
    print(json.dumps({"event": "mounted_cache_verified", "seconds": time.time() - started,
                      "receipt_sha256": cache_sha, "files": len(verified["files"])}), flush=True)
    deadline = started + JOB_HOURS * 3600 - 300
    if time.time() + 1800 > deadline:
        raise RuntimeError("cache verification left insufficient bounded training time")
    processes, logs, peak = [], [], rss_gib(os.getpid())
    try:
        for rank, fold in enumerate(JOB_FOLDS):
            directory = out / f"e04_fold{fold}"
            directory.mkdir(parents=True, exist_ok=True)
            handle = (directory / "worker.log").open("w")
            logs.append(handle)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=devices[rank], E04_TRAIN_FOLD=str(fold),
                       E04_TRAIN_OUTPUT=str(directory), E04_PARENT_DEADLINE=str(deadline),
                       E04_CACHE_VERIFICATION_PATH=str(cache_path), E04_CACHE_VERIFICATION_SHA256=cache_sha,
                       OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            processes.append(subprocess.Popen([sys.executable, "-u", str(worker_path)], env=env,
                                              stdout=handle, stderr=subprocess.STDOUT))
        previous, last_note = None, 0.0
        while any(p.poll() is None for p in processes):
            codes = [p.poll() for p in processes]
            if any(c is not None and c != 0 for c in codes):
                raise RuntimeError(f"full fold worker failed: {codes}; inspect its saved log")
            if time.time() > started + JOB_HOURS * 3600:
                raise RuntimeError("bounded full-fold job reached its hard deadline")
            current = rss_gib(os.getpid()) + sum(rss_gib(p.pid) for p in processes)
            peak = max(peak, current)
            if current > 20:
                raise RuntimeError("aggregate full-fold resident memory exceeded 20 GiB")
            states = [last_progress(out / f"e04_fold{f}/worker.log") for f in JOB_FOLDS]
            key = json.dumps(states, sort_keys=True)
            if key != previous or time.time() - last_note >= 60:
                print(json.dumps({"event": "full_fold_progress", "elapsed_seconds": time.time() - started,
                                  "aggregate_rss_gib": current, "workers": states}), flush=True)
                previous, last_note = key, time.time()
            time.sleep(5)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("full fold process exited unsuccessfully")
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
        for handle in logs:
            handle.close()
    results = []
    for fold in JOB_FOLDS:
        path = out / f"e04_fold{fold}/e04_fold{fold}_manifest.json"
        result = json.loads(path.read_text())
        if (result["fold"] != fold or result["status"] not in ("COMPLETE", "INCOMPLETE_RESUMABLE")
                or result["checkpoint_selection_on_expert_labels"] or result["competition_submission_enabled"]):
            raise RuntimeError("full fold output contract failed")
        results.append({"fold": fold, "status": result["status"], "completed_epoch": result["completed_epoch"],
                        "manifest_sha256": sha(path), "contract_sha256": result["contract_sha256"]})
    result = {"schema": "e04_independent_fold_job_v1", "status": "COMPLETE",
              "training_complete": all(r["status"] == "COMPLETE" for r in results),
              "folds": JOB_FOLDS, "results": results, "job_budget_hours": JOB_HOURS,
              "worker_sha256": WORKER_SHA256, "cache_verification_sha256": cache_sha,
              "aggregate_peak_rss_gib": peak, "elapsed_seconds": time.time() - started,
              "competition_submission_enabled": False,
              "remaining_gate": "all five OOF folds and independent artifact/accuracy audit"}
    (out / "e04_fold_job_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
