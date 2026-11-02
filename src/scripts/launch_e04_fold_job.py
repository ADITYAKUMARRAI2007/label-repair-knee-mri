#!/usr/bin/env python3
"""Launch one built private fold job once, within freshly observed free quota."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from audit_e04_resource_pilot import duration_seconds, require
from e04_job_budget import remaining_live_commitment, proposed_job_fits, STARTUP_FINALIZATION_MARGIN_HOURS

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", choices=("0", "1", "01", "2", "3", "23", "4"), default="01")
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--check-only", action="store_true",
                        help="Inspect sources, existing jobs and quota without pushing a notebook")
    args = parser.parse_args()
    require(args.attempt >= 0, "attempt number must be nonnegative")
    tag = args.folds if not args.attempt else f"{args.folds}_r{args.attempt}"
    directory = ROOT / f"public_kernels/e04_convnext_folds_{tag}"
    build = json.loads((ROOT / f"runs/e04_folds_{tag}_source.json").read_text())
    require(build.get("attempt", 0) == args.attempt, "built job attempt mismatch")
    for relative, expected in {**build["sources"], **build["gate_receipts"]}.items():
        require(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected,
                f"built source or gate changed: {relative}")
    metadata = json.loads((directory / "kernel-metadata.json").read_text())
    require(metadata["is_private"] and not metadata["enable_internet"]
            and metadata["competition_sources"] == ["rsna-knee-abnormality-detection"],
            "private training metadata changed")
    receipt_path = ROOT / f"runs/e04_folds_{tag}_launch.json"
    require(not receipt_path.exists(), "launch receipt already exists; observe that original job instead of repeating the push")
    api = KaggleApi()
    api.authenticate()
    for source in metadata["kernel_sources"]:
        state = api.kernels_status(source)
        state = state.to_dict() if hasattr(state, "to_dict") else json.loads(str(state))
        require(state["status"] == "COMPLETE", f"mounted input kernel is still finalizing: {source}")
    # Kaggle returns 403, rather than 404, for a new nonexistent private slug.
    # Establish absence from a complete authenticated listing of owned kernels.
    owned, page = set(), 1
    while True:
        kernels = api.kernels_list(mine=True, page=page, page_size=100) or []
        owned.update(k.ref for k in kernels if k is not None)
        if len(kernels) < 100:
            break
        page += 1
        require(page <= 100, "owned kernel listing did not terminate")
    require("adityakumarrai0401/rsna-knee-e04-wide-cache-base" in owned,
            "authenticated owned-kernel listing omitted the known private cache")
    require(metadata["id"] not in owned,
            "fold-job slug already exists; observe its original state instead of pushing a new version")
    quota = api.quota_view()
    if hasattr(quota, "to_dict"):
        quota = quota.to_dict()
    gpu = quota["gpuQuota"]
    free = (duration_seconds(gpu["totalTimeAllowed"]) - duration_seconds(gpu["timeUsed"])
            - duration_seconds(gpu["timeReserved"])) / 3600
    observed = datetime.now(timezone.utc).isoformat()
    commitments = []
    for existing_path in sorted((ROOT / "runs").glob("e04_folds_*_launch.json")):
        existing = json.loads(existing_path.read_text())
        if existing["status"] == "PUSH_REJECTED":
            continue
        state = api.kernels_status(existing["kernel"]).to_dict()
        remaining = remaining_live_commitment(existing, state["status"], observed)
        commitments.append({"kernel": existing["kernel"], "status": state["status"],
                            "remaining_budget_hours_including_margin": remaining})
    observation = {"observed_utc": observed, "quota": quota,
                   "other_training_commitments": commitments,
                   "startup_finalization_margin_hours_per_job": STARTUP_FINALIZATION_MARGIN_HOURS}
    (ROOT / f"runs/e04_quota_before_folds_{tag}.json").write_text(json.dumps(observation, indent=2) + "\n")
    fits = proposed_job_fits(free, [r["remaining_budget_hours_including_margin"] for r in commitments],
                            build["job_budget_hours"])
    platform_timeout = int((build["job_budget_hours"] + STARTUP_FINALIZATION_MARGIN_HOURS) * 3600)
    require(0 < platform_timeout <= 9 * 3600, "platform timeout exceeds the job limit")
    if args.check_only:
        print(json.dumps({"status": "PREFLIGHT_ONLY", "kernel": metadata["id"],
                          "allocated_free_hours": free, "quota_covers_job": fits,
                          "platform_timeout_seconds": platform_timeout,
                          "gpu_launched": False, "competition_submission_enabled": False}, indent=2))
        return
    require(fits,
            "allocated free quota cannot cover active-job commitments and this bounded job with margins")
    # Write before the network mutation. An ambiguous timeout must be resolved
    # by observing this exact slug, rather than triggering a duplicate push.
    receipt = {"schema": "e04_fold_job_launch_v1", "status": "PUSH_ATTEMPT_REQUIRES_OBSERVATION",
               "kernel": metadata["id"], "observed_utc": observation["observed_utc"],
               "attempt": args.attempt,
               "platform_timeout_seconds": platform_timeout,
               "job_budget_hours": build["job_budget_hours"], "allocated_free_hours": free,
               "competition_submission_enabled": False, "build_sha256": hashlib.sha256(
                   (ROOT / f"runs/e04_folds_{tag}_source.json").read_bytes()).hexdigest()}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    response = api.kernels_push(str(directory), timeout=str(platform_timeout))
    error = str(response.error or "")
    if error:
        receipt.update(status="PUSH_REJECTED", error=error)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        raise RuntimeError(f"private fold job push rejected: {error}")
    receipt.update(status="PUSH_ACCEPTED", version=response.version_number)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
