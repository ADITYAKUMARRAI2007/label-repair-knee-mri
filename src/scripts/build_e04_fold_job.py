#!/usr/bin/env python3
"""Assemble full independent fold workers after cache/resource receipt gates."""
import argparse
import ast
import hashlib
import json
import py_compile
from pathlib import Path

from audit_e04_resource_pilot import duration_seconds, require
from e04_resume_contract import verified_resume_receipts

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", default="0,1", choices=("0", "1", "0,1", "2", "3", "2,3", "4"))
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--resume-kernel", action="append", default=[])
    parser.add_argument("--quota-snapshot", type=Path)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--resume-output", type=Path)
    args = parser.parse_args()
    folds = [int(v) for v in args.folds.split(",")]
    require(args.attempt >= 0, "attempt number must be nonnegative")
    require(bool(args.resume_kernel) == (args.attempt > 0),
            "resumes need a new attempt number and an explicit original completed job input")
    require(bool(args.resume_output) == bool(args.resume_kernel), "resume output metadata is required for a resume")
    require(not args.resume_kernel or len(args.resume_kernel) == 1, "use one original job per resume group")
    require(args.attempt > 0 or args.folds in ("0,1", "2,3", "4"), "single early folds are for completed-epoch resumes")
    suffix = "".join(map(str, folds))
    job_tag = suffix if not args.attempt else f"{suffix}_r{args.attempt}"
    slug_suffix = suffix if not args.attempt else f"{suffix}-r{args.attempt}"
    require(not (ROOT / f"runs/e04_folds_{job_tag}_launch.json").exists(),
            "this job has a launch receipt; preserve its original build and observe it")
    require(1 <= args.hours <= 8, "job budget must leave room within Kaggle's nine-hour limit")
    audit_path = ROOT / "runs/E04_contract/cache_audit.json"
    cache = json.loads(audit_path.read_text())
    require(cache["status"] == "PASS_SMALL_OUTPUTS" and cache["studies"] == 4407
            and cache["independent_wide_reference_studies"] == 59, "complete cache gate failed")
    resource = json.loads((ROOT / "runs/E04_paired_pilot/resource_audit.json").read_text())
    require(resource["status"] == "PASS_OPERATIONAL_PROBE_ONLY", "paired resource gate failed")
    additional_gates, timing = [], None
    if any(f >= 2 for f in folds):
        timing_path = ROOT / "runs/E04_contract/first_pair_epoch_budget_audit.json"
        timing = json.loads(timing_path.read_text())
        require(timing["status"] == "PASS_COMPLETE_EPOCH_TIMING"
                and (args.resume_kernel or timing["projection_with_15_percent_margin_hours"] <= args.hours),
                "complete first-pair epoch timing does not cover the requested bounded job")
        additional_gates.append(timing_path)
    quota_path = args.quota_snapshot or (ROOT / "runs/e04_latest_quota.json")
    if not quota_path.exists():
        quota_path = ROOT / "runs/e04_quota_after_pair.json"
    quota = json.loads(quota_path.read_text())["quota"]["gpuQuota"]
    free = (duration_seconds(quota["totalTimeAllowed"]) - duration_seconds(quota["timeUsed"])
            - duration_seconds(quota["timeReserved"])) / 3600
    require(free >= args.hours + .25, "allocated quota does not cover the bounded job")
    require(all(s.startswith("adityakumarrai0401/rsna-knee-e04-convnext-folds-") for s in args.resume_kernel),
            "resume inputs must be owned full E04 fold jobs")
    expected = {}
    for part in cache["parts"]:
        name = part["part"]
        manifest = ROOT / f"runs/E04_wide_cache_{name}/e04_wide_{name}_manifest.json"
        require(sha(manifest) == part["manifest_sha256"], "audited cache manifest changed")
        expected[name] = {"manifest_sha256": sha(manifest)}
    pilot = ROOT / "public_kernels/e04_convnext_resource_pilot/rsna-knee-e04-convnext-resource-pilot.py"
    require(sha(pilot) == "498faaa8c17707d43f8f5fd06ae864cc5950933a18461e358f55e5ea5c97a0c4",
            "audited model/sampler source changed")
    source = pilot.read_text()
    pieces = []
    skipped = {"main": 0, "open_corpus": 0, "entrypoint": 0}
    for node in ast.parse(source).body:
        name = getattr(node, "name", "")
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and name in ("main", "open_corpus"):
            skipped[name] += 1
            continue
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and "__name__" in ast.unparse(node.test):
            skipped["entrypoint"] += 1
            continue
        pieces.append(ast.get_source_segment(source, node))
    require(skipped == {"main": 1, "open_corpus": 1, "entrypoint": 1}, "pilot extraction drift")
    wide = ROOT / "training/e04_wide_corpus.py"
    trainer = ROOT / "training/e04_convnext_fold_train.py"
    supervisor = ROOT / "training/e04_fold_supervisor.py"
    components = {str(p.relative_to(ROOT)): sha(p) for p in (pilot, wide, trainer)}
    resumes = []
    if args.resume_kernel:
        resumes, resume_gates = verified_resume_receipts(args.resume_output.resolve(), args.resume_kernel[0], folds, components)
        additional_gates.extend(resume_gates)
        remaining_epochs = max(14 - r["completed_epoch"] for r in resumes)
        if timing:
            epoch_seconds = max(w["first_epoch"]["epoch_wall_seconds"] + w["first_epoch"]["checkpoint_seconds"]
                                for w in timing["workers"])
            setup_seconds = timing["setup_seconds"]
        else:
            epoch_seconds = max(w["steady_seconds_per_step"] for w in resource["workers"]) * 1763 + 120
            setup_seconds = 300
        resume_projection = (setup_seconds + epoch_seconds * remaining_epochs + 1200) / 3600 * 1.15
        require(resume_projection <= args.hours, "requested job cannot cover measured remaining epochs and validation reserves")
    worker = "\n\n".join(pieces) + "\n\n"
    worker += wide.read_text().replace("# __EXPECTED_WIDE_CACHE__", f"EXPECTED_WIDE_CACHE = {expected!r}")
    worker += f"\n\nWORKER_COMPONENT_SHA256 = {components!r}\n\n"
    worker += trainer.read_text() + '\n\nif __name__ == "__main__":\n    main()\n'
    ast.parse(worker)
    directory = ROOT / f"public_kernels/e04_convnext_folds_{job_tag}"
    directory.mkdir(parents=True, exist_ok=True)
    preview = directory / "e04_full_fold_worker_preview.py"
    preview.write_text(worker)
    py_compile.compile(str(preview), doraise=True)
    worker_sha = sha(preview)
    code = supervisor.read_text().replace("# __FROZEN_JOB__", f"JOB_FOLDS = {folds!r}\nJOB_HOURS = {args.hours!r}")
    if resumes:
        code = code.replace("JOB_FOLDS =", f"RESUME_CHECKPOINT_RECEIPTS = {resumes!r}\nJOB_FOLDS =", 1)
        anchor = "    verified = definitions.verify_wide_files()"
        require(code.count(anchor) == 1, "resume-verification insertion point changed")
        verification = '''
    for receipt in RESUME_CHECKPOINT_RECEIPTS:
        checkpoint = definitions.find_unique(receipt["filename"], receipt["sha256"])
        if checkpoint.stat().st_size != receipt["bytes"]:
            raise RuntimeError("mounted resume checkpoint size changed")
        if len(definitions.bounded_files(receipt["filename"])) != 1:
            raise RuntimeError("multiple mounted resume checkpoints would make the worker ambiguous")
        print(json.dumps({"event": "resume_checkpoint_verified", **receipt}), flush=True)
'''
        code = code.replace(anchor, anchor + "\n" + verification)
    code = code.replace("# __WORKER_SOURCE__", f"WORKER_SOURCE = {worker!r}\nWORKER_SHA256 = {worker_sha!r}")
    dest = directory / f"rsna-knee-e04-convnext-folds-{slug_suffix}.py"
    dest.write_text(code)
    py_compile.compile(str(dest), doraise=True)
    metadata = json.loads((pilot.parent / "kernel-metadata.json").read_text())
    metadata.update(id=f"adityakumarrai0401/rsna-knee-e04-convnext-folds-{slug_suffix}",
                    title=f"RSNA Knee E04 ConvNeXt Folds {slug_suffix}", code_file=dest.name)
    metadata["dataset_sources"] = [s for s in metadata["dataset_sources"] if "knee-raptor-corpus" not in s]
    metadata["kernel_sources"] = ["adityakumarrai0401/rsna-knee-e04-wide-cache-base",
                                  "adityakumarrai0401/rsna-knee-e04-wide-cache-extension", *args.resume_kernel]
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = {"schema": "e04_full_fold_job_build_v1", "syntax": "PASS", "folds": folds,
              "attempt": args.attempt,
              "job_budget_hours": args.hours, "allocated_hours_at_build": free,
              "fresh_quota_check_required_before_push": True,
              "quota_snapshot": str(quota_path),
              "worker_source_sha256": worker_sha, "frozen_cache": expected,
              "fixed_final_ema_epoch": 14, "expert_checkpoint_selection": False,
              "competition_submission_enabled": False, "resume_kernels": args.resume_kernel,
              "resume_checkpoint_receipts": resumes,
              "sources": {str(p.relative_to(ROOT)): sha(p) for p in (pilot, wide, trainer, supervisor, dest, preview)},
              "gate_receipts": {str(p.relative_to(ROOT)): sha(p) for p in
                                (audit_path, ROOT / "runs/E04_paired_pilot/resource_audit.json", *additional_gates)}}
    (ROOT / f"runs/e04_folds_{job_tag}_source.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
