"""Stream hidden studies through the fixed five-fold probability mean.

Only a completed, pinned Gold58 raw replay can authorize these checkpoints.
This driver writes a submission file; it never calls the competition API.
"""
import csv
import json
import os
import resource
import time

from e04_decode_stream import decoded_studies


def find_hidden_competition_root():
    roots = [p.parent for p in bounded_files("test.csv")
             if (p.parent / "test_series.csv").is_file()
             and (p.parent / "test_series").is_dir()
             and (p.parent / "sample_submission.csv").is_file()]
    if len(roots) != 1:
        raise RuntimeError("hidden competition input discovery failed")
    return roots[0]


def checkpoint_identity(receipts):
    return [{k: r[k] for k in ("fold", "filename", "sha256", "bytes", "contract_sha256")}
            for r in sorted(receipts, key=lambda r: r["fold"])]


def predict_hidden_studies(fold_receipts, replay_receipt, output_dir):
    started = time.time()
    replay = json.loads(verified_artifact(replay_receipt).read_text())
    if (replay["status"] != "PASS" or replay["expert_rows"] != 58
            or replay["max_abs_probability_error"] > 1e-6
            or not replay["pixel_sha256_matches_owned_training_cache"]
            or replay["expert_prediction_rule"] != "each study uses its heldout fold only"
            or replay["fold_checkpoints"] != checkpoint_identity(fold_receipts)
            or replay["sources"] != INFERENCE_SOURCE_SHA256):
        raise RuntimeError("these exact checkpoints and inference sources lack a complete raw replay")
    root = find_hidden_competition_root()
    with (root / "test.csv").open(newline="") as handle:
        test_ids = [r["StudyInstanceUID"] for r in csv.DictReader(handle)]
    with (root / "sample_submission.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["StudyInstanceUID", *TARGETS]:
            raise RuntimeError("competition submission target order changed")
        ids = [r["StudyInstanceUID"] for r in reader]
    if (not ids or len(set(ids)) != len(ids) or len(set(test_ids)) != len(test_ids)
            or set(ids) != set(test_ids)):
        raise RuntimeError("test and template study identities differ")
    output_dir.mkdir(parents=True, exist_ok=True)
    names = ["submission.csv", "e04_mean_raw.csv", *[f"e04_fold{f}_raw.csv" for f in range(5)],
             "e04_hidden_decode_receipts.jsonl"]
    if any((output_dir / name).exists() for name in names):
        raise RuntimeError("refusing to overwrite a previously completed inference output")
    paths = [output_dir / (name + ".partial") for name in names]
    handles = [path.open("w", newline="") for path in paths]
    writers = [csv.writer(h) for h in handles[:-1]]
    for writer in writers:
        writer.writerow(["StudyInstanceUID", *TARGETS])
    totals = {"studies": 0, "selected_slice_attempts": 0, "decoded_selected_slices": 0,
              "header_failures": 0, "slice_failures": 0, "position_fallbacks": 0,
              "spacing_fallbacks": 0, "missing_acquisitions": 0, "prediction_fallback_studies": 0}
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    predictor = None
    stream = None
    peak_rss, peak_gpu, timing = 0.0, {}, []
    try:
        predictor = EqualFoldPredictor(fold_receipts)
        stream = decoded_studies(enumerate(ids), root / "test_series.csv", root / "test_series")
        completed = 0
        while completed < len(ids):
            batch_started = time.time()
            decoded = [next(stream) for _ in range(min(2, len(ids) - completed))]
            if [r[0] for r in decoded] != list(range(completed, completed + len(decoded))):
                raise RuntimeError("hidden decode order changed")
            mean, raw = predictor.predict(*packed_batch(decoded))
            if (mean.shape != (len(decoded), len(TARGETS)) or raw.shape != (5, len(decoded), len(TARGETS))
                    or not np.isfinite(raw).all() or (raw < 0).any() or (raw > 1).any()):
                raise RuntimeError("invalid hidden fold probabilities")
            for i, (row, _, _, stats) in enumerate(decoded):
                uid = ids[row]
                if uid != stats["StudyInstanceUID"] or stats["prediction_fallback_used"]:
                    raise RuntimeError("hidden study identity or prediction fallback drift")
                for writer in writers[:2]:
                    writer.writerow([uid, *mean[i].tolist()])
                for fold, writer in enumerate(writers[2:]):
                    writer.writerow([uid, *raw[fold, i].tolist()])
                handles[-1].write(json.dumps(stats) + "\n")
                totals["studies"] += 1
                for key in ("selected_slice_attempts", "header_failures", "slice_failures"):
                    totals[key] += stats[key]
                for slot in stats["slots"]:
                    for key in ("decoded_selected_slices", "position_fallbacks", "spacing_fallbacks"):
                        totals[key] += slot[key]
                    totals["missing_acquisitions"] += int(slot["state"] == "missing_acquisition")
            completed += len(decoded)
            timing.append(time.time() - batch_started)
            peak_rss = max(peak_rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2)
            peak_gpu = {str(device): torch.cuda.max_memory_allocated(device) / 1024 ** 3
                        for device in predictor.groups}
            if peak_rss > 16 or max(peak_gpu.values()) > 13:
                raise RuntimeError("hidden inference memory budget exceeded")
            if completed % 50 == 0 or completed == len(ids):
                print(json.dumps({"event": "e04_hidden_progress", "completed": completed,
                                  "total": len(ids), "elapsed_seconds": time.time() - started,
                                  "parent_peak_rss_gib": peak_rss, "gpu_peak_gib": peak_gpu}), flush=True)
            if time.time() - started > 8.5 * 3600:
                raise RuntimeError("hidden inference runtime budget exceeded")
        if (totals["studies"] != len(ids) or totals["slice_failures"] or totals["header_failures"]
                or totals["selected_slice_attempts"] != totals["decoded_selected_slices"]):
            raise RuntimeError("hidden decoding coverage failed")
    finally:
        if stream is not None:
            stream.close()
        if predictor is not None:
            predictor.close()
        for handle in handles:
            handle.close()
    # Final filenames become visible only after complete prediction and coverage.
    for path, name in zip(paths, names):
        os.replace(path, output_dir / name)
    artifacts = {name: {"filename": name, "bytes": (output_dir / name).stat().st_size,
                        "sha256": sha256_file(output_dir / name)} for name in names}
    result = {"schema": "e04_hidden_prediction_v1", "status": "COMPLETE",
              "target_order": TARGETS, "studies": len(ids), "decode_totals": totals,
              "ordered_uid_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
              "fixed_ensemble_rule": "equal_probability_mean_of_all_five_final_epoch14_EMA_models",
              "fold_checkpoints": checkpoint_identity(fold_receipts), "sources": INFERENCE_SOURCE_SHA256,
              "raw_replay_receipt": replay_receipt, "artifacts": artifacts,
              "elapsed_seconds": time.time() - started, "parent_peak_rss_gib": peak_rss,
              "gpu_peak_gib": peak_gpu, "mean_batch_seconds": float(np.mean(timing)),
              "competition_api_submission_performed": False}
    (output_dir / "e04_hidden_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result
