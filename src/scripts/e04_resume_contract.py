"""Select completed-epoch resumes from checksum-verified owned output metadata."""
import hashlib
import json


def verified_resume_receipts(directory, origin, folds, components):
    download_path = directory / "download_receipt.json"
    downloaded = json.loads(download_path.read_text())
    if downloaded["status"] != "COMPLETE" or downloaded["kernel"] != origin or downloaded["job_mutations"]:
        raise RuntimeError("resume output download does not match the original owned job")
    known = {r["file"]: r for r in downloaded["files"]}
    results, gates = [], [download_path]
    for fold in folds:
        paths = list(directory.rglob(f"e04_fold{fold}_manifest.json"))
        if len(paths) != 1:
            raise RuntimeError("resume requires exactly one original fold manifest")
        path = paths[0]
        relative = str(path.relative_to(directory))
        receipt = known.get(relative)
        if not receipt or path.stat().st_size != receipt["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != receipt["sha256"]:
            raise RuntimeError("resume manifest bytes do not match the download receipt")
        manifest = json.loads(path.read_text())
        contract = manifest["contract"]
        completed = manifest["completed_epoch"]
        if (manifest["schema"] != "rsna_knee_e04_convnext_expert_fold_v1"
                or manifest["status"] != "INCOMPLETE_RESUMABLE" or manifest["fold"] != fold
                or completed not in range(14) or contract["fold"] != fold
                or contract["worker_component_sha256"] != components
                or contract["checkpoint_rule"] != "final_epoch_14_ema_only"
                or contract["expert_checkpoint_selection"]
                or manifest["checkpoint_selection_on_expert_labels"]
                or manifest["competition_submission_enabled"]):
            raise RuntimeError("resume source is complete, a pilot, or a different training recipe")
        if (hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest() != manifest["contract_sha256"]
                or [h["epoch"] for h in manifest["history"]] != list(range(1, completed + 1))
                or manifest["checkpoint_step"] != completed * 1763
                or manifest["amp_skipped_updates"] > 10):
            raise RuntimeError("completed-epoch resume coverage failed")
        name = f"e04_fold{fold}_last.pt"
        state = manifest["artifacts"][name]
        if state["bytes"] <= 0 or len(state["sha256"]) != 64:
            raise RuntimeError("invalid resume checkpoint receipt")
        results.append({"fold": fold, "filename": name, **state, "completed_epoch": completed})
        gates.append(path)
    return results, gates
