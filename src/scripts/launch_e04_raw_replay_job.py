#!/usr/bin/env python3
"""Push the gated private Gold58 replay once, without buying quota or submitting."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from audit_e04_resource_pilot import duration_seconds, require
from e04_job_budget import remaining_live_commitment, proposed_job_fits

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "runs/e04_final_raw_replay_source.json"
    build = json.loads(source.read_text())
    require(build["schema"] == "e04_final_raw_replay_job_build_v1" and build["status"] == "BUILT_NOT_PUSHED"
            and not build["competition_submission_enabled"], "reviewed private replay build required")
    for relative, digest in {**build["sources"], **build["gate_receipts"]}.items():
        require(sha(ROOT / relative) == digest, f"built source or gate changed: {relative}")
    directory = ROOT / "public_kernels/e04_final_raw_replay"
    metadata = json.loads((directory / "kernel-metadata.json").read_text())
    require(metadata["id"] == build["kernel"] and metadata["is_private"]
            and not metadata["enable_internet"] and metadata["enable_gpu"]
            and metadata["competition_sources"] == ["rsna-knee-abnormality-detection"],
            "private replay metadata changed")
    receipt_path = ROOT / "runs/e04_final_raw_replay_launch.json"
    require(not receipt_path.exists(), "observe the original replay push; do not push another version")
    api = KaggleApi()
    api.authenticate()
    for origin in metadata["kernel_sources"]:
        require(api.kernels_status(origin).to_dict()["status"] == "COMPLETE", "replay input job is not complete")
    owned, page = set(), 1
    while True:
        records = api.kernels_list(mine=True, page=page, page_size=100) or []
        owned.update(r.ref for r in records if r is not None)
        if len(records) < 100:
            break
        page += 1
        require(page <= 100, "owned kernel listing failed to terminate")
    require("adityakumarrai0401/rsna-knee-e04-wide-cache-base" in owned,
            "authenticated listing omitted the known owned cache")
    require(metadata["id"] not in owned, "replay slug already exists; observe that original job")
    quota = api.quota_view()
    if hasattr(quota, "to_dict"):
        quota = quota.to_dict()
    gpu = quota["gpuQuota"]
    free = (duration_seconds(gpu["totalTimeAllowed"]) - duration_seconds(gpu["timeUsed"])
            - duration_seconds(gpu["timeReserved"])) / 3600
    observed = datetime.now(timezone.utc).isoformat()
    commitments = []
    for path in sorted((ROOT / "runs").glob("e04_folds_*_launch.json")):
        receipt = json.loads(path.read_text())
        if receipt["status"] == "PUSH_REJECTED":
            continue
        state = api.kernels_status(receipt["kernel"]).to_dict()["status"]
        commitments.append({"kernel": receipt["kernel"], "status": state,
                            "remaining_commitment_hours": remaining_live_commitment(receipt, state, observed)})
    require(build["runtime_cap_hours"] == .75 and build["startup_finalization_margin_hours"] == .5,
            "private replay resource budget changed")
    observation = {"observed_utc": observed, "quota": quota, "allocated_free_hours": free,
                   "active_training_commitments": commitments, "proposed_replay_hours_including_margin": 1.25,
                   "competition_submission_enabled": False}
    (ROOT / "runs/e04_quota_before_raw_replay.json").write_text(json.dumps(observation, indent=2) + "\n")
    require(proposed_job_fits(free, [r["remaining_commitment_hours"] for r in commitments], .75),
            "allocated quota cannot cover replay and active jobs; keep this build staged")
    # Persist before mutation. A timeout is resolved by observing this exact slug.
    receipt = {"schema": "e04_final_raw_replay_launch_v1", "status": "PUSH_ATTEMPT_REQUIRES_OBSERVATION",
               "kernel": metadata["id"], "observed_utc": observed, "job_budget_hours": .75,
               "allocated_free_hours": free, "build_sha256": sha(source),
               "competition_submission_enabled": False}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    response = api.kernels_push(str(directory))
    error = getattr(response, "error", None)
    if error:
        receipt.update(status="PUSH_REJECTED", error=str(error))
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        raise RuntimeError("Kaggle explicitly rejected the private replay push; inspect its launch receipt")
    receipt.update(status="PUSH_ACCEPTED", version=response.version_number)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
