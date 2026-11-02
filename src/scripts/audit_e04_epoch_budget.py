#!/usr/bin/env python3
"""Require a complete first epoch per model before budgeting additional folds."""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = ROOT / "runs/E04_contract/first_pair_epoch_budget_audit.json"
    archive = ROOT / "runs/E04_contract/first_pair_epoch_budget.log"
    build_path = ROOT / "runs/e04_folds_01_source.json"
    build = json.loads(build_path.read_text())
    if output.exists():
        frozen = json.loads(output.read_text())
        require(frozen["status"] == "PASS_COMPLETE_EPOCH_TIMING"
                and frozen["archived_log_sha256"] == sha(archive)
                and frozen["worker_sha256"] == build["worker_source_sha256"], "epoch timing receipt changed")
        print(json.dumps(frozen, indent=2))
        return
    log_path = ROOT / "runs/E04_folds_01_live.log"
    records = []
    for line in log_path.read_text().splitlines():
        index = line.find("{")
        if index >= 0:
            try:
                records.append(json.loads(line[index:]))
            except json.JSONDecodeError:
                continue
    require(records and records[0]["worker_sha256"] == build["worker_source_sha256"], "job source identity changed")
    startup = json.loads((ROOT / "runs/E04_contract/first_pair_startup_audit.json").read_text())
    require(startup["worker_source_sha256"] == build["worker_source_sha256"]
            and startup["status"] == "PASS_CACHE_AND_FIRST_UPDATES_ONLY", "first-pair startup gate missing")
    cache = next(r for r in records if r.get("event") == "mounted_cache_verified")
    require(cache["receipt_sha256"] == startup["mounted_cache_runtime_receipt_sha256"], "mounted cache identity changed")
    completed, batches, ready_seconds = {}, {0: {}, 1: {}}, []
    for parent in records:
        if parent.get("event") != "full_fold_progress":
            continue
        require(math.isfinite(parent["aggregate_rss_gib"]) and parent["aggregate_rss_gib"] < 20,
                "aggregate memory limit failed")
        for worker in parent["workers"]:
            if not worker:
                continue
            fold = worker["fold"]
            require(fold in (0, 1), "unexpected fold in original pair")
            if worker.get("event") == "full_fold_ready":
                require(worker["contract_sha256"] == startup["frozen_input_and_checkpoint_loading_ready"][str(fold)]["contract_sha256"],
                        "fold recipe changed")
                ready_seconds.append(parent["elapsed_seconds"])
            if worker.get("epoch") != 1:
                continue
            if worker.get("event") == "epoch_complete":
                require(worker["batches"] == worker["step"] == 1763
                        and all(math.isfinite(worker[k]) for k in ("loss", "epoch_wall_seconds", "checkpoint_seconds"))
                        and worker["epoch_wall_seconds"] > 0 and worker["checkpoint_seconds"] >= 0,
                        "complete epoch record failed")
                completed[fold] = worker
            elif "batch" in worker:
                require(worker["batches"] == 1763 and all(math.isfinite(worker[k]) for k in
                        ("loss", "rss_gib", "gpu_peak_gib", "epoch_seconds")), "invalid batch timing")
                require(worker["rss_gib"] < 18 and worker["gpu_peak_gib"] < 13, "worker memory limit failed")
                batches[fold][worker["batch"]] = worker
    if set(completed) != {0, 1}:
        progress = {"status": "WAIT_COMPLETE_EPOCH", "completed_first_epochs": sorted(completed),
                    "last_observed_batches": {f: max(samples, default=0) for f, samples in batches.items()},
                    "other_full_fold_job_gate_passed": False}
        (ROOT / "runs/E04_contract/first_pair_epoch_progress.json").write_text(json.dumps(progress, indent=2) + "\n")
        print(json.dumps(progress, indent=2))
        return
    workers = []
    for fold in (0, 1):
        samples = [v for k, v in sorted(batches[fold].items()) if k >= 100]
        require(len(samples) >= 5, "insufficient full-epoch steady timing samples")
        rates = [(b["epoch_seconds"] - a["epoch_seconds"]) / (b["batch"] - a["batch"])
                 for a, b in zip(samples, samples[1:])]
        require(all(r > 0 for r in rates), "nonmonotonic batch clock")
        workers.append({"fold": fold, "first_epoch": completed[fold],
                        "steady_seconds_per_batch_after100": statistics.median(rates),
                        "observed_first100_seconds": samples[0]["epoch_seconds"]})
    setup = min(ready_seconds) if ready_seconds else cache["seconds"] + 120
    # Project with the cold first-epoch wall time for all fourteen epochs. Keep
    # checkpoint I/O and fifteen minutes of final validation plus five minutes
    # of parent finalization explicit. This forecasts time, never public AUC.
    measured_epoch = max(w["first_epoch"]["epoch_wall_seconds"] + w["first_epoch"]["checkpoint_seconds"] for w in workers)
    projection = (setup + measured_epoch * 14 + 900 + 300) / 3600
    archive.write_bytes(log_path.read_bytes())
    result = {"schema": "e04_first_complete_epoch_budget_audit_v1", "status": "PASS_COMPLETE_EPOCH_TIMING",
              "worker_sha256": build["worker_source_sha256"], "archived_log_sha256": sha(archive),
              "workers": workers, "setup_seconds": setup,
              "one_job_14_epoch_projection_hours_with_checkpoint_IO_and_validation_reserves": projection,
              "projection_with_15_percent_margin_hours": projection * 1.15,
              "allocation_startup_finalization_margin_hours_per_job": .5,
              "accuracy_measured": False, "competition_submission_enabled": False,
              "remaining_gates": ["fresh allocated quota including all active-job commitments",
                                  "all14epochs per fold", "complete five-fold expert OOF and independent audits"]}
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
