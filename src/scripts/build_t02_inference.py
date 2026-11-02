#!/usr/bin/env python3
"""Build the fail-closed T02 five-fold hidden-test inference kernel."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "public_kernels/t02_goldfold_train/rsna-knee-goldfold-convnext-train.py"
OUT_DIR = ROOT / "public_kernels/t02_goldfold_infer"
OUT_SCRIPT = OUT_DIR / "rsna-knee-goldfold-convnext-infer.py"
MARKER = "# --------------------------- T02 owned driver -------------------------------"


DRIVER = r'''

# ------------------------ T02 inference driver ------------------------------
import hashlib as _hashlib
import json as _json

T02_SCHEMA = "rsna_knee_goldfold_convnext_v1"
T02_INFER_SCHEMA = "rsna_knee_goldfold_convnext_infer_v1"
T02_INIT_SHA256 = "bec04468edeaaa0c75191a221d4e3b80bada5cfd263eeec36cb093548de93147"
CFG.crop_mm = 130.0
CFG.n_group_max = 3
CFG.eval_batch = 16
CFG.time_budget = 11.5 * 3600


def _sha256(path):
    digest = _hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _bounded_files(name, max_depth=5):
    base = Path("/kaggle/input")
    hits = []
    for root, dirs, files in os.walk(base):
        rel = Path(root).relative_to(base)
        depth = len(rel.parts)
        dirs[:] = [d for d in dirs if d not in {
            "train_series", "test_series", "images", "train_images", "test_images"
        } and depth < max_depth]
        if name in files:
            hits.append(Path(root) / name)
    return sorted({path.resolve() for path in hits}, key=str)


def _load_training_artifacts():
    manifest_paths = []
    manifests = []
    for tag in ("f012", "f34"):
        candidates = _bounded_files(f"t02_manifest_{tag}.json")
        if len(candidates) != 1:
            raise RuntimeError(f"expected one T02 {tag} manifest, got {candidates}")
        manifest_paths.append(candidates[0])
        manifests.append(_json.loads(candidates[0].read_text()))
    if [m.get("run_folds") for m in manifests] != [[0, 1, 2], [3, 4]]:
        raise RuntimeError("T02 split fold manifests do not cover folds 0 through 4")
    expected_input = {
        "img": 224, "crop_mm": 130.0,
        "slots": ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
                  "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"],
        "group": 3, "n_group": 3, "window": "0.35,0.65",
    }
    weights = {}
    for manifest in manifests:
        if (manifest.get("schema") != T02_SCHEMA
                or manifest.get("split_schema") != "rsna_knee_goldfold_convnext_split_v1"
                or manifest.get("targets") != TARGETS):
            raise RuntimeError("T02 split manifest schema or targets mismatch")
        if manifest.get("input") != expected_input:
            raise RuntimeError(f"T02 input contract mismatch: {manifest.get('input')}")
        if manifest.get("provenance", {}).get("init_sha256") != T02_INIT_SHA256:
            raise RuntimeError("T02 initialization provenance mismatch")
        if manifest.get("validation", {}).get("fold_counts") != [882, 882, 881, 881, 881]:
            raise RuntimeError("T02 fold completion contract mismatch")
        overlap = set(weights) & set(manifest.get("weights", {}))
        if overlap:
            raise RuntimeError(f"T02 duplicate split checkpoints: {sorted(overlap)}")
        weights.update(manifest.get("weights", {}))
    expected_names = [f"t02_goldfold_convnext_fold{i}.pt" for i in range(5)]
    if set(weights) != set(expected_names):
        raise RuntimeError(f"T02 checkpoint set mismatch: {sorted(weights)}")
    paths = []
    for name in expected_names:
        candidates = _bounded_files(name)
        if len(candidates) != 1:
            raise RuntimeError(f"expected one {name}, got {candidates}")
        path = candidates[0]
        record = weights[name]
        if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise RuntimeError(f"T02 checkpoint identity mismatch for {name}")
        paths.append(path)
    return manifest_paths, manifests, paths


def _build_t02_model(checkpoint_path, dev):
    import timm
    backbone = timm.create_model(
        "convnext_tiny.fb_in22k_ft_in1k", pretrained=False,
        num_classes=0, global_pool="avg",
    )
    model = Model(Encoder(backbone, "timm", int(backbone.num_features)))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (payload.get("schema") != T02_SCHEMA or payload.get("targets") != TARGETS
            or payload.get("fold") not in range(5)
            or payload.get("init_sha256") != T02_INIT_SHA256):
        raise RuntimeError(f"T02 checkpoint metadata mismatch: {checkpoint_path.name}")
    missing, unexpected = model.load_state_dict(payload["state"], strict=True)
    if missing or unexpected:
        raise RuntimeError(f"T02 checkpoint state mismatch: {missing}, {unexpected}")
    model = model.to(dev).eval()
    if torch.cuda.device_count() >= 2:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    return model, int(payload["fold"])


def _build_test_cache(root):
    test_series = pd.read_csv(
        root / "test_series.csv",
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    plane_map = dict(zip(test_series["SeriesInstanceUID"], test_series["Anatomical_Plane"]))
    headers = annotate(walk(root, "test_series"))
    if headers.empty:
        raise RuntimeError("T02 found no hidden-test DICOM headers")
    headers["plane"] = headers["SeriesInstanceUID"].map(plane_map)
    lat_map = {study: resolve_laterality(group)
               for study, group in headers.groupby("StudyInstanceUID")}
    slot_map = pick_slots(headers, plane_map)
    studies, cache, mask = build_cache(slot_map, plane_map, lat_map, "test", 3)
    expected = pd.read_csv(root / "test.csv", dtype={"StudyInstanceUID": str})[
        "StudyInstanceUID"
    ].astype(str).tolist()
    if len(studies) != len(expected) or set(studies) != set(expected):
        raise RuntimeError("T02 decoded study identity differs from hidden test.csv")
    if cache.shape != (len(expected), 6, 9, 224, 224) or mask.shape != (len(expected), 6):
        raise RuntimeError(f"T02 hidden cache shape mismatch: {cache.shape}, {mask.shape}")
    if np.any(mask.sum(axis=1) == 0):
        raise RuntimeError("T02 hidden cache contains a study with no usable MRI slot")
    return studies, cache, mask, expected


def _main_t02_infer():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("T02 inference requires the configured dual T4 accelerator")
    manifest_paths, manifests, checkpoint_paths = _load_training_artifacts()
    root = find_root()
    studies, cache, mask, expected_ids = _build_test_cache(root)
    dev = select_device()
    fold_predictions = []
    fold_ids = []
    checkpoint_receipts = []
    indices = np.arange(len(studies), dtype=np.int64)
    for path in checkpoint_paths:
        model, fold = _build_t02_model(path, dev)
        prediction = predict(model, cache, mask, indices, dev, 3).astype(np.float32)
        if prediction.shape != (len(studies), 12) or not np.isfinite(prediction).all():
            raise RuntimeError(f"T02 fold {fold} produced invalid predictions")
        fold_predictions.append(prediction)
        fold_ids.append(fold)
        checkpoint_receipts.append({"fold": fold, "name": path.name,
                                    "bytes": path.stat().st_size, "sha256": _sha256(path)})
        del model
        gc.collect(); torch.cuda.empty_cache()
        print(f"T02 inferred fold {fold} ({len(fold_predictions)}/5)", flush=True)
        if time.time() - T0 > CFG.time_budget:
            raise RuntimeError("T02 inference exceeded its time budget")
    if sorted(fold_ids) != list(range(5)):
        raise RuntimeError(f"T02 inference fold set mismatch: {fold_ids}")

    mean_probability = np.mean(np.stack(fold_predictions), axis=0)
    ranked = pd.DataFrame(mean_probability, columns=TARGETS).rank(
        method="average", pct=True
    ).to_numpy(np.float64)
    by_uid = pd.DataFrame(ranked, columns=TARGETS)
    by_uid.insert(0, "StudyInstanceUID", studies)
    submission = pd.DataFrame({"StudyInstanceUID": expected_ids}).merge(
        by_uid, on="StudyInstanceUID", how="left", validate="one_to_one"
    )
    if submission.StudyInstanceUID.tolist() != expected_ids:
        raise RuntimeError("T02 final study order mismatch")
    if not np.isfinite(submission[TARGETS].to_numpy(np.float64)).all():
        raise RuntimeError("T02 final submission contains non-finite values")
    output = Path("/kaggle/working/submission.csv")
    submission.to_csv(output, index=False)
    np.savez_compressed(
        "/kaggle/working/t02_fold_predictions.npz",
        study_uids=np.asarray(studies), folds=np.asarray(fold_ids, np.int8),
        values=np.stack(fold_predictions).astype(np.float32),
    )
    receipt = {
        "schema": T02_INFER_SCHEMA, "status": "COMPLETE",
        "studies": len(studies), "folds": fold_ids, "fallback_studies": 0,
        "cache_shape": list(cache.shape), "slot_coverage": mask.mean(axis=0).tolist(),
        "checkpoints": checkpoint_receipts,
        "training_manifests": [str(path) for path in manifest_paths],
        "training_manifest_sha256": [_sha256(path) for path in manifest_paths],
        "partial_gold58_oof_macro_auc": [
            manifest["validation"]["partial_gold_oof_macro_auc"] for manifest in manifests
        ],
        "ensemble": "mean of five fold probabilities then per-target percentile rank",
        "submission_sha256": _sha256(output),
        "elapsed_seconds": time.time() - T0,
    }
    Path("/kaggle/working/t02_inference_receipt.json").write_text(
        _json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(_json.dumps(receipt, indent=2, sort_keys=True), flush=True)


_main_t02_infer()
'''


def main() -> None:
    source = TRAIN_SCRIPT.read_text()
    if source.count(MARKER) != 1:
        raise RuntimeError("T02 training source marker missing or duplicated")
    base = source.split(MARKER, 1)[0].rstrip()
    base = base.replace(
        '"""T02 Gold-fold ConvNeXt trainer.\n\nDerived from the public CC BY 4.0 RSNA pixel-cache pipeline by Steven Lee Hans,\nwith an owned target payload, exact cache contract, MRI-adapted initialization,\ndual-T4 full fine tuning, fold-wise expert supervision, and fail-closed receipts.\n"""',
        '"""T02 five-fold ConvNeXt inference.\n\nUses the public CC BY 4.0 MRI preprocessing pipeline by Steven Lee Hans and\nstrictly audited checkpoints from the owned T02 training kernel.\n"""',
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SCRIPT.write_text(base + DRIVER)
    metadata = {
        "id": "adityakumarrai0401/rsna-knee-goldfold-convnext-infer-v1",
        "title": "RSNA Knee GoldFold ConvNeXt Infer V1",
        "code_file": OUT_SCRIPT.name,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["gpu"], "dataset_sources": [],
        "kernel_sources": [
            "adityakumarrai0401/rsna-knee-goldfold-convnext-train-v1",
            "adityakumarrai0401/rsna-knee-goldfold-convnext-f34",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"script": str(OUT_SCRIPT), "bytes": OUT_SCRIPT.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
