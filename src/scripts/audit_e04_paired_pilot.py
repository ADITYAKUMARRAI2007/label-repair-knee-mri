#!/usr/bin/env python3
"""Audit small paired-probe outputs, source identity and observed quota use."""
import hashlib
import json
import math
import statistics
from pathlib import Path

from audit_e04_resource_pilot import duration_seconds, require

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / "runs/E04_paired_pilot"
    parent_path = out / "e04_pair_manifest.json"
    parent = json.loads(parent_path.read_text())
    build = json.loads((ROOT / "runs/e04_paired_pilot_source.json").read_text())
    frozen = json.loads((ROOT / "runs/E04_contract/e04_contract.json").read_text())
    for relative, expected in build["sources"].items():
        require(sha(ROOT / relative) == expected, "paired source identity changed")
    require(parent["schema"] == "e04_paired_resource_pilot_v1" and parent["status"] == "COMPLETE",
            "paired probe incomplete")
    require(parent["resource_gate"] == "PASS_CONCURRENT_MEMORY_NUMERICS", "paired operational gate failed")
    require(parent["visible_gpu_count"] == 2 and len(set(parent["assigned_devices"])) == 2,
            "independent GPU assignment failed")
    require(parent["independent_model_processes"] and not parent["shared_expert_trunk"]
            and not parent["accuracy_measured"] and not parent["submission_enabled"], "probe scope drift")
    require(parent["worker_source_sha256"] == build["worker_source_sha256"], "worker source drift")
    require(math.isfinite(parent["aggregate_peak_rss_gib"]) and parent["aggregate_peak_rss_gib"] < 18,
            "aggregate memory limit failed")
    rates, results = [], []
    for rank, receipt in enumerate(parent["workers"]):
        path = out / f"e04_pair_worker{rank}/e04_pilot_manifest.json"
        worker = json.loads(path.read_text())
        require(sha(path) == receipt["manifest_sha256"], "worker manifest identity changed")
        require(worker["schema"] == "e04_convnext_paired_worker_v1" and worker["status"] == "COMPLETE",
                "worker incomplete")
        require(worker["heldout_fold"] == rank and receipt["fold"] == rank
                and worker["assigned_physical_gpu"] == parent["assigned_devices"][rank]
                and worker["visible_gpu_count"] == 1, "worker GPU or fold drift")
        require(worker["steps"] == 300 and worker["amp_skipped_updates"] == worker["gold_updates"] == 0,
                "worker updates or expert exposure drift")
        require(not worker["accuracy_measured"] and not worker["competition_submission_enabled"]
                and not worker["pilot_weights_reusable_for_training"], "worker probe scope drift")
        require(worker["fold_audit"] == frozen["split_audit"], "frozen folds changed")
        require(worker["strict_pretrained_sha256"] == "2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9",
                "pretrained identity changed")
        require(worker["mask_invariance_max_abs_error"] == worker["checkpoint_replay_max_abs_error"] == 0,
                "mask invariance or checkpoint replay failed")
        require(worker["resource_gate"] == "PASS_CONCURRENT_MEMORY_NUMERICS", "worker resource gate failed")
        samples = worker["memory_samples"]
        require([s["step"] for s in samples] == list(range(25, 301, 25)), "sample coverage failed")
        require(all(math.isfinite(s[k]) for s in samples for k in ("loss", "seconds", "rss_gib", "gpu_peak_gib")),
                "nonfinite probe samples")
        require(all(b["seconds"] > a["seconds"] for a, b in zip(samples, samples[1:])), "elapsed samples reversed")
        growth = samples[-1]["rss_gib"] - samples[1]["rss_gib"]
        require(abs(growth - worker["rss_growth_after_step50_gib"]) < 1e-12 and growth < 2,
                "resident memory growth failed")
        require(max(s["gpu_peak_gib"] for s in samples) < 13 and max(s["rss_gib"] for s in samples) < 20,
                "worker memory failed")
        require(receipt["artifact"] == worker["artifacts"], "reported checkpoint metadata inconsistent")
        rate = statistics.median((b["seconds"] - a["seconds"]) / (b["step"] - a["step"])
                                 for a, b in zip(samples[1:-1], samples[2:]))
        require(abs(rate - receipt["steady_wall_seconds_per_step_after50"]) < 1e-12, "timing mismatch")
        rates.append(rate)
        results.append({"fold": rank, "steady_seconds_per_step": rate, "manifest_sha256": sha(path),
                        "rss_growth_after50_gib": growth})
    require(len(results) == 2, "two completed workers required")
    projection = max(rates) * 1763 * 14 * 3 / 3600
    require(abs(projection - parent["projected_three_job_training_hours_before_validation_checkpoint_IO"]) < 1e-10,
            "three-job projection mismatch")
    before = json.loads((ROOT / "runs/e04_quota_before_pair.json").read_text())["quota"]["gpuQuota"]
    after = json.loads((ROOT / "runs/e04_quota_after_pair.json").read_text())["quota"]["gpuQuota"]
    used = duration_seconds(after["timeUsed"]) - duration_seconds(before["timeUsed"])
    require(used > 0, "quota observation did not advance")
    remaining = (duration_seconds(after["totalTimeAllowed"]) - duration_seconds(after["timeUsed"])
                 - duration_seconds(after["timeReserved"])) / 3600
    result = {"schema": "e04_independent_paired_resource_audit_v1", "status": "PASS_OPERATIONAL_PROBE_ONLY",
              "parent_manifest_sha256": sha(parent_path), "workers": results,
              "aggregate_peak_rss_gib": parent["aggregate_peak_rss_gib"],
              "observed_quota_delta_seconds": used, "parent_elapsed_seconds": parent["elapsed_seconds"],
              "quota_seconds_per_parent_wall_second": used / parent["elapsed_seconds"],
              "quota_scope": "observed completed two-active-GPU kernel; delta includes kernel startup/finalization",
              "allocated_gpu_hours_remaining": remaining,
              "three_job_training_projection_hours_before_validation_checkpoint_IO": projection,
              "projection_with_15_percent_margin_hours": projection * 1.15,
              "all_five_folds_proven_to_fit_remaining_quota": False,
              "checkpoint_bytes_verified_locally": False, "accuracy_measured": False,
              "competition_submission_enabled": False,
              "remaining_gates": ["complete wide-cache audit and mounted volume hashes",
                                  "bounded resumable first fold pair", "full epoch wall-time budget",
                                  "all five held-out expert prediction folds and independent accuracy audit"]}
    (out / "resource_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
