"""Replay Gold58 through each study's held-out final EMA and raw DICOM.

The assembler provides exact executed model definitions, source hashes and
actual final artifact receipts. This file is preparation, not a scored result.
"""
import csv
import json
import time

from e04_replay_batches import expert_replay_batches
from e04_decode_stream import decoded_studies


def verified_artifact(receipt):
    path = find_unique(receipt["filename"], receipt["sha256"])
    if path.stat().st_size != receipt["bytes"]:
        raise RuntimeError("replay artifact size mismatch")
    return path


def load_pixel_references(receipts):
    references = {}
    for receipt in receipts:
        with verified_artifact(receipt).open() as handle:
            for line in handle:
                row = json.loads(line)
                uid = row["StudyInstanceUID"]
                if uid in references or row["span"] != [.06, .94]:
                    raise RuntimeError("duplicate or incompatible wide pixel reference")
                references[uid] = row["volume_sha256"]
    if len(references) != 4407:
        raise RuntimeError("incomplete wide pixel references")
    return references


def packed_batch(decoded):
    arrays = [make_eval_arrays(volume, mask) for _, volume, mask, _ in decoded]
    return tuple(torch.from_numpy(np.stack([row[i] for row in arrays])) for i in range(3))


def replay_raw_experts(fold_receipts, pixel_receipts, output_dir):
    started = time.time()
    if sorted(r["fold"] for r in fold_receipts) != list(range(5)):
        raise RuntimeError("Gold58 replay requires the five completed OOF fold artifacts")
    root = find_competition_root()
    train = root / "train.csv"
    if sha256_file(train) != "8ca2203c0e9d61c080c7a314c7cdb51c1b03a1d9eb4770819f7f34af53ef4e33":
        raise RuntimeError("competition training table changed")
    with train.open(newline="") as handle:
        official = list(csv.DictReader(handle))
    # Labels identify the pre-existing expert set only; no model/weight selection.
    experts = {r["StudyInstanceUID"] for r in official
               if all(r[t] not in ("", "nan", "NaN") for t in TARGETS)}
    if len(experts) != 58:
        raise RuntimeError("official expert set changed")
    references = load_pixel_references(pixel_receipts)
    output_dir.mkdir(parents=True, exist_ok=True)
    records, seen, maximum, batches = [], set(), 0.0, 0
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    for receipt in sorted(fold_receipts, key=lambda r: r["fold"]):
        split = json.loads(verified_artifact(receipt["split"]).read_text())
        if split["heldout_fold"] != receipt["fold"]:
            raise RuntimeError("replay fold identity mismatch")
        validation, heldout = split["validation_ids"], split["expert_validation_ids"]
        if set(heldout) != set(validation).intersection(experts) or seen.intersection(heldout):
            raise RuntimeError("replay experts are not uniquely held out")
        if set(split["training_ids"]).intersection(validation):
            raise RuntimeError("replay training/validation overlap")
        with verified_artifact(receipt["oof"]).open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["StudyInstanceUID", *TARGETS]:
                raise RuntimeError("OOF target order changed")
            rows = list(reader)
        if [r["StudyInstanceUID"] for r in rows] != validation:
            raise RuntimeError("OOF validation ordering changed")
        selected = expert_replay_batches(validation, heldout)
        items = [(start + i, uid) for start, ids in selected for i, uid in enumerate(ids)]
        device = torch.device("cuda:0")
        model = load_verified_fold(receipt, device)
        stream = decoded_studies(items, root / "train_series.csv", root / "train_series")
        try:
            for start, ids in selected:
                decoded = [next(stream) for _ in ids]
                for (row, _, _, stats), uid in zip(decoded, ids):
                    if (row != validation.index(uid) or stats["StudyInstanceUID"] != uid
                            or stats["volume_sha256"] != references[uid]):
                        raise RuntimeError("raw DICOM pixels differ from the owned training cache")
                actual = predict_fold_on_device(model, *packed_batch(decoded), device)
                expected = np.asarray([[float(rows[start + i][t]) for t in TARGETS]
                                       for i in range(len(ids))], dtype=np.float32)
                error = np.abs(actual - expected)
                if not np.isfinite(error).all() or error.max() > 1e-6:
                    raise RuntimeError("raw DICOM final-model predictions differ from saved OOF")
                maximum = max(maximum, float(error.max()))
                batches += 1
                for i, uid in enumerate(ids):
                    records.append({"StudyInstanceUID": uid, "fold": receipt["fold"],
                                    "expert": uid in heldout, "validation_row": start + i,
                                    "max_abs_error": float(error[i].max()),
                                    "probabilities": actual[i].tolist(), "decode": decoded[i][3]})
        finally:
            stream.close()
            del model
            torch.cuda.empty_cache()
        seen.update(heldout)
    if seen != experts:
        raise RuntimeError("Gold58 raw replay is incomplete")
    detail = output_dir / "e04_raw_replay_studies.json"
    detail.write_text(json.dumps(records, indent=2) + "\n")
    result = {"schema": "e04_raw_dicom_expert_replay_v1", "status": "PASS",
              "expert_rows": len(seen), "replayed_studies_including_batch_partners": len(records),
              "original_validation_batches": batches, "max_abs_probability_error": maximum,
              "pixel_sha256_matches_owned_training_cache": True,
              "expert_prediction_rule": "each study uses its heldout fold only",
              "expert_checkpoint_selection": False, "competition_submission_enabled": False,
              "elapsed_seconds": time.time() - started,
              "sources": INFERENCE_SOURCE_SHA256,
              "fold_checkpoints": [{k: r[k] for k in ("fold", "filename", "sha256", "bytes", "contract_sha256")}
                                   for r in sorted(fold_receipts, key=lambda r: r["fold"])],
              "detail": {"filename": detail.name, "bytes": detail.stat().st_size,
                         "sha256": sha256_file(detail)}}
    (output_dir / "e04_raw_replay_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result
