#!/usr/bin/env python3
"""Run the gated visible prediction notebook once; never submit or buy quota."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

from audit_e04_resource_pilot import duration_seconds, require
from build_e04_raw_replay_job import sha
from e04_job_budget import remaining_live_commitment, proposed_job_fits

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "runs/e04_final_predict_source.json"
    build = json.loads(source.read_text())
    require(build["schema"] == "e04_final_predict_job_build_v1" and build["status"] == "BUILT_NOT_PUSHED"
            and not build["competition_submission_enabled"], "gated final prediction build required")
    for relative, digest in {**build["sources"], **build["gate_receipts"]}.items():
        require(sha(ROOT / relative) == digest, f"source or acceptance gate changed: {relative}")
    directory = ROOT / "public_kernels/e04_final_predict"
    metadata = json.loads((directory / "kernel-metadata.json").read_text())
    require(metadata["id"] == build["kernel"] and metadata["is_private"] and metadata["enable_gpu"]
            and not metadata["enable_internet"]
            and metadata["competition_sources"] == ["rsna-knee-abnormality-detection"],
            "private offline prediction metadata changed")
    receipt_path = ROOT / "runs/e04_final_predict_launch.json"
    require(not receipt_path.exists(), "observe the existing prediction push instead of pushing another version")
    api = KaggleApi()
    api.authenticate()
    for origin in metadata["kernel_sources"]:
        require(api.kernels_status(origin).to_dict()["status"] == "COMPLETE", "model/replay input is not complete")
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
    require(metadata["id"] not in owned, "prediction slug already exists; inspect its original job")
    # Observe the current visible table from Kaggle, not a potentially stale
    # local count, before committing the short visible-run resource budget.
    visible_directory = ROOT / "runs/E04_visible_input_check"
    visible_directory.mkdir(parents=True, exist_ok=True)
    try:
        api.competition_download_file("rsna-knee-abnormality-detection", "test.csv",
                                      path=str(visible_directory), force=True, quiet=True)
    except Exception as error:
        # SDK download errors can contain signed URLs. Keep the observation
        # failure explicit without printing a temporary download credential.
        raise RuntimeError(f"fresh visible-table download failed ({type(error).__name__}); no job pushed") from None
    table = visible_directory / "test.csv"
    require(table.is_file(), "fresh visible test table was not downloaded under its expected name")
    with table.open(newline="") as handle:
        ids = [r["StudyInstanceUID"] for r in csv.DictReader(handle)]
    require(1 <= len(ids) <= 3 and len(set(ids)) == len(ids),
            "visible cohort changed; re-estimate resources before launching")
    require(build["private_visible_runtime_cap_hours"] == .5
            and build["startup_finalization_margin_hours"] == .5, "visible-run resource cap changed")
    quota = api.quota_view()
    if hasattr(quota, "to_dict"):
        quota = quota.to_dict()
    gpu = quota["gpuQuota"]
    free = (duration_seconds(gpu["totalTimeAllowed"]) - duration_seconds(gpu["timeUsed"])
            - duration_seconds(gpu["timeReserved"])) / 3600
    observed = datetime.now(timezone.utc).isoformat()
    commitments = []
    paths = sorted((ROOT / "runs").glob("e04_folds_*_launch.json"))
    paths += [ROOT / "runs/e04_final_raw_replay_launch.json"]
    for path in paths:
        receipt = json.loads(path.read_text())
        if receipt["status"] == "PUSH_REJECTED":
            continue
        state = api.kernels_status(receipt["kernel"]).to_dict()["status"]
        commitments.append({"kernel": receipt["kernel"], "status": state,
                            "remaining_commitment_hours": remaining_live_commitment(receipt, state, observed)})
    observation = {"observed_utc": observed, "quota": quota, "allocated_free_hours": free,
                   "active_job_commitments": commitments, "fresh_visible_studies": len(ids),
                   "visible_test_csv_sha256": sha(table), "proposed_visible_job_hours_including_margin": 1.0,
                   "competition_submission_enabled": False}
    (ROOT / "runs/e04_quota_before_final_predict.json").write_text(json.dumps(observation, indent=2) + "\n")
    require(proposed_job_fits(free, [r["remaining_commitment_hours"] for r in commitments], .5),
            "allocated quota does not cover the bounded visible run and existing jobs")
    receipt = {"schema": "e04_final_predict_launch_v1", "status": "PUSH_ATTEMPT_REQUIRES_OBSERVATION",
               "kernel": metadata["id"], "observed_utc": observed, "job_budget_hours": .5,
               "allocated_free_hours": free, "fresh_visible_studies": len(ids),
               "build_sha256": sha(source), "competition_submission_enabled": False}
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    response = api.kernels_push(str(directory))
    error = getattr(response, "error", None)
    if error:
        receipt.update(status="PUSH_REJECTED", error=str(error))
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
        raise RuntimeError("Kaggle explicitly rejected the prediction push; inspect its launch receipt")
    receipt.update(status="PUSH_ACCEPTED", version=response.version_number)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
