#!/usr/bin/env python3
"""Independently check the completed operational pilot and its timing limits."""
import hashlib
import json
import math
import statistics
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def duration_seconds(value):
    # This SDK snapshot appends '.0s' to an already fractional seconds field.
    # Handle that observed representation without treating it as valid ISO time.
    if value.endswith(".0s"):
        return float(Decimal(value[:-3]))
    require(value.endswith("s"), "unexpected quota duration representation")
    return float(Decimal(value[:-1]))


def main():
    path = ROOT / "runs/E04_pilot/e04_pilot_manifest.json"
    pilot = json.loads(path.read_text())
    frozen = json.loads((ROOT / "runs/E04_contract/e04_contract.json").read_text())
    build = json.loads((ROOT / "runs/e04_pilot_source_build.json").read_text())
    code = ROOT / "public_kernels/e04_convnext_resource_pilot/rsna-knee-e04-convnext-resource-pilot.py"
    code_sha = hashlib.sha256(code.read_bytes()).hexdigest()
    require(code_sha == build["sources"][str(code.relative_to(ROOT))], "executed source receipt drift")
    require(pilot["schema"] == "e04_convnext_resource_pilot_v1" and pilot["status"] == "COMPLETE",
            "pilot did not complete")
    require(pilot["steps"] == 300 and pilot["amp_skipped_updates"] == 0, "pilot updates drift")
    require(pilot["gold_updates"] == 0 and not pilot["accuracy_measured"]
            and not pilot["competition_submission_enabled"] and not pilot["pilot_weights_reusable_for_training"],
            "resource/accuracy scope drift")
    require(pilot["fold_audit"] == frozen["split_audit"], "fold reproduction drift")
    require(pilot["strict_pretrained_sha256"] == "2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9",
            "pretrained identity drift")
    samples = pilot["memory_samples"]
    require([s["step"] for s in samples] == list(range(25, 301, 25)), "memory sample coverage drift")
    require(all(math.isfinite(s[k]) for s in samples for k in ("loss", "rss_gib", "gpu_peak_gib", "seconds")),
            "nonfinite operational sample")
    require(max(s["rss_gib"] for s in samples) < 20 and max(s["gpu_peak_gib"] for s in samples) < 13,
            "memory gate failed")
    growth = samples[-1]["rss_gib"] - samples[1]["rss_gib"]
    require(abs(growth - pilot["rss_growth_after_step50_gib"]) < 1e-12 and growth < 2,
            "resident-memory growth mismatch")
    require(pilot["mask_invariance_max_abs_error"] == pilot["checkpoint_replay_max_abs_error"] == 0,
            "mask or saved-checkpoint replay failed")
    projected = pilot["estimated_epoch_seconds"] * 14 * 5 / 3600
    require(abs(projected - pilot["estimated_five_fold_training_hours_excluding_validation"]) < 1e-10,
            "reported compute projection is inconsistent")
    require(pilot["resource_gate"] == "REQUIRES_RESOURCE_REDESIGN" and projected > 17,
            "runtime decision mismatch")
    intervals = [(b["seconds"] - a["seconds"]) / (b["step"] - a["step"])
                 for a, b in zip(samples[1:-1], samples[2:])]
    steady_wall = statistics.median(intervals)
    batches = math.ceil((4407 - 882) / 2)
    steady_five_hours = steady_wall * batches * 14 * 5 / 3600
    before = json.loads((ROOT / "runs/e04_accelerator_quota.json").read_text())["gpuQuota"]
    after = json.loads((ROOT / "runs/e04_accelerator_quota_after_pilot.json").read_text())["gpuQuota"]
    used_before, used_after = duration_seconds(before["timeUsed"]), duration_seconds(after["timeUsed"])
    remaining = (duration_seconds(after["totalTimeAllowed"]) - used_after - duration_seconds(after["timeReserved"])) / 3600
    result = {"schema": "e04_independent_resource_audit_v1", "status": "PASS_MEMORY_NUMERICS_REPLAY_ONLY",
              "pilot_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "code_sha256": code_sha,
              "steps": 300, "rss_growth_after_step50_gib": growth,
              "gpu_peak_gib": max(s["gpu_peak_gib"] for s in samples),
              "rss_peak_sampled_gib": max(s["rss_gib"] for s in samples),
              "original_five_fold_compute_projection_hours": projected,
              "steady_end_to_end_seconds_per_step_after_step50": steady_wall,
              "steady_five_fold_projection_hours_before_validation_and_checkpoint_IO": steady_five_hours,
              "hypothetical_three_paired_jobs_hours_without_contention": steady_five_hours * 3 / 5,
              "allocated_gpu_hours_remaining": remaining,
              "observed_pilot_quota_seconds": used_after - used_before,
              "quota_delta_scope": "one active model in a T4 kernel; active-two-model quota rate needs confirmation",
              "full_training_authorized_by_this_audit": False,
              "next_gate": "paired independent-GPU resource pilot, then full epoch wall-time budgeting",
              "checkpoint_bytes_verified_locally": False,
              "scope": "operational manifest, samples and source receipts; no image validation or public score"}
    output = ROOT / "runs/E04_pilot/resource_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
