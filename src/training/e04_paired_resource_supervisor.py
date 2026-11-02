#!/usr/bin/env python3
"""Run two independent resource probes on two T4s; never train a shared trunk."""
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

# __WORKER_SOURCE__


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
    raise RuntimeError("cannot read worker resident memory")


def last_step(path):
    if not path.exists():
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 8192))
        tail = handle.read().decode(errors="replace")
    for line in reversed(tail.splitlines()):
        start = line.find("{")
        if start < 0:
            continue
        try:
            item = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if "step" in item and "rss_gib" in item:
            return item
    return None


def main():
    out = Path("/kaggle/working")
    visible = torch.cuda.device_count()
    if visible < 2:
        raise RuntimeError("paired probe requires two visible GPUs")
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    devices = physical.split(",") if physical else [str(i) for i in range(visible)]
    if len(devices) < 2 or len(set(devices[:2])) != 2:
        raise RuntimeError("ambiguous physical GPU assignment")
    worker = out / "e04_pair_worker.py"
    worker.write_text(WORKER_SOURCE)
    if hashlib.sha256(worker.read_bytes()).hexdigest() != WORKER_SHA256:
        raise RuntimeError("worker source identity drift")
    started, processes, logs, peak_sum = time.time(), [], [], 0.0
    print(json.dumps({"event": "launch_pair", "visible_gpu_count": visible,
                      "assigned_devices": devices[:2], "folds": [0, 1],
                      "worker_source_sha256": WORKER_SHA256,
                      "accuracy_measured": False, "submission_enabled": False}), flush=True)
    try:
        for rank in range(2):
            directory = out / f"e04_pair_worker{rank}"
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / "worker.log").open("w")
            logs.append(log)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=devices[rank], E04_PILOT_FOLD=str(rank),
                       E04_PILOT_OUTPUT=str(directory), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
            processes.append(subprocess.Popen([sys.executable, "-u", str(worker)], env=env,
                                              stdout=log, stderr=subprocess.STDOUT))
        previous = None
        while any(p.poll() is None for p in processes):
            codes = [p.poll() for p in processes]
            if any(c is not None and c != 0 for c in codes):
                raise RuntimeError(f"paired probe worker exited unsuccessfully: {codes}")
            if time.time() - started > 1800:
                raise RuntimeError("paired probe exceeded its fixed 30-minute runtime limit")
            total_rss = rss_gib(os.getpid()) + sum(rss_gib(p.pid) for p in processes)
            peak_sum = max(peak_sum, total_rss)
            if peak_sum > 18:
                raise RuntimeError("paired probe aggregate resident memory exceeded 18 GiB")
            states = [last_step(out / f"e04_pair_worker{r}/worker.log") for r in range(2)]
            steps = tuple(s["step"] if s else 0 for s in states)
            if steps != previous:
                print(json.dumps({"event": "pair_progress", "elapsed_seconds": time.time() - started,
                                  "aggregate_current_rss_gib": total_rss, "workers": states}), flush=True)
                previous = steps
            time.sleep(5)
        if any(p.returncode != 0 for p in processes):
            raise RuntimeError("paired probe ended with an unsuccessful worker")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for log in logs:
            log.close()
    results, rates = [], []
    for rank in range(2):
        directory = out / f"e04_pair_worker{rank}"
        result = json.loads((directory / "e04_pilot_manifest.json").read_text())
        if (result["steps"] != 300 or result["amp_skipped_updates"] != 0
                or result["heldout_fold"] != rank or result["visible_gpu_count"] != 1
                or result["assigned_physical_gpu"] != devices[rank]
                or result["mask_invariance_max_abs_error"] != 0
                or result["checkpoint_replay_max_abs_error"] != 0):
            raise RuntimeError("paired worker contract failed")
        if result["resource_gate"] != "PASS_CONCURRENT_MEMORY_NUMERICS":
            raise RuntimeError("paired worker memory gate failed")
        samples = result["memory_samples"]
        intervals = [(b["seconds"] - a["seconds"]) / (b["step"] - a["step"])
                     for a, b in zip(samples[1:-1], samples[2:])]
        rate = statistics.median(intervals)
        rates.append(rate)
        results.append({"fold": rank, "steady_wall_seconds_per_step_after50": rate,
                        "manifest_sha256": hashlib.sha256((directory / "e04_pilot_manifest.json").read_bytes()).hexdigest(),
                        "artifact": result["artifacts"], "rss_growth_gib": result["rss_growth_after_step50_gib"],
                        "gpu_peak_gib": max(s["gpu_peak_gib"] for s in samples)})
    # Use the slower concurrent worker for a three-job five-fold forecast.
    projection = max(rates) * 1763 * 14 * 3 / 3600
    result = {"schema": "e04_paired_resource_pilot_v1", "status": "COMPLETE",
              "resource_gate": "PASS_CONCURRENT_MEMORY_NUMERICS",
              "visible_gpu_count": visible, "assigned_devices": devices[:2],
              "independent_model_processes": True, "shared_expert_trunk": False,
              "worker_source_sha256": WORKER_SHA256, "workers": results,
              "aggregate_peak_rss_gib": peak_sum, "elapsed_seconds": time.time() - started,
              "projected_three_job_training_hours_before_validation_checkpoint_IO": projection,
              "projection_with_15_percent_margin_hours": projection * 1.15,
              "accuracy_measured": False, "submission_enabled": False,
              "remaining_gates": ["independent small-artifact audit", "observed two-active-GPU quota rate",
                                  "complete wide-cache audit and mounted file hashes",
                                  "full-epoch wall-time budget before subsequent fold jobs"]}
    (out / "e04_pair_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
