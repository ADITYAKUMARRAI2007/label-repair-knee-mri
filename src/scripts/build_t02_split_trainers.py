#!/usr/bin/env python3
"""Split the audited T02 five-fold trainer across two Kaggle GPU kernels."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "public_kernels/t02_goldfold_train/rsna-knee-goldfold-convnext-train.py"

SPLITS = {
    "f012": {
        "folds": [0, 1, 2],
        "id": "adityakumarrai0401/rsna-knee-goldfold-convnext-train-v1",
        "title": "RSNA Knee GoldFold ConvNeXt Train V1",
    },
    "f34": {
        "folds": [3, 4],
        "id": "adityakumarrai0401/rsna-knee-goldfold-convnext-f34",
        "title": "RSNA Knee GoldFold ConvNeXt F34",
    },
}


MAIN_TEMPLATE = r'''
T02_SPLIT_SCHEMA = "rsna_knee_goldfold_convnext_split_v1"
T02_SPLIT_TAG = __TAG__
T02_RUN_FOLDS = __FOLDS__


def _main_t02():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("T02 requires the configured dual T4 accelerator")
    if not T02_RUN_FOLDS or len(set(T02_RUN_FOLDS)) != len(T02_RUN_FOLDS):
        raise RuntimeError(f"invalid T02 split folds: {T02_RUN_FOLDS}")
    if any(fold not in range(5) for fold in T02_RUN_FOLDS):
        raise RuntimeError(f"out-of-range T02 split fold: {T02_RUN_FOLDS}")
    torch.backends.cudnn.benchmark = True
    root = find_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    uids = train["StudyInstanceUID"].astype(str).tolist()
    global T02_TARGETS_MATRIX, T02_GOLD_MASK
    T02_TARGETS_MATRIX, runtime_weights, runtime_folds, T02_GOLD_MASK, label_provenance = _construct_targets(train)
    if T02_TARGETS_MATRIX.shape != (4407, 12) or T02_GOLD_MASK.sum() != 58:
        raise RuntimeError("T02 runtime target contract mismatch")
    cache, mask, cache_meta_path, cache_meta = _load_cache(uids)
    print("T02 split", T02_SPLIT_TAG, "folds", T02_RUN_FOLDS,
          "cache", cache.shape, "coverage", mask.mean(0), flush=True)

    dev = select_device()
    oof = np.full((len(uids), len(TARGETS)), np.nan, np.float32)
    histories = {}
    paths = []
    fold_estimate = None
    for fold in T02_RUN_FOLDS:
        if fold_estimate is not None and time.time() - T0 + 1.15 * fold_estimate > CFG.time_budget:
            raise RuntimeError(f"insufficient time to begin required fold {fold}")
        train_idx = np.flatnonzero(runtime_folds != fold)
        valid_idx = np.flatnonzero(runtime_folds == fold)
        started = time.time()
        pred, history, path = _train_fold_full(
            cache, mask, T02_TARGETS_MATRIX, runtime_weights,
            train_idx, valid_idx, dev, fold,
        )
        oof[valid_idx] = pred
        histories[str(fold)] = history
        paths.append(path)
        elapsed = time.time() - started
        fold_estimate = elapsed if fold_estimate is None else max(fold_estimate, elapsed)
        print(f"fold={fold} complete seconds={elapsed:.1f}", flush=True)

    completed = np.flatnonzero(np.isfinite(oof).all(axis=1))
    expected = np.flatnonzero(np.isin(runtime_folds, T02_RUN_FOLDS))
    if not np.array_equal(completed, expected):
        raise RuntimeError("T02 split OOF coverage mismatch")
    gold_idx = completed[T02_GOLD_MASK[completed] > 0]
    gold_truth = train.loc[gold_idx, TARGETS].to_numpy(np.int8)
    gold_pred = oof[gold_idx]
    gold_macro = macro_auc(gold_truth, gold_pred)
    target_auc = per_target_auc(gold_truth, gold_pred)
    derived_macro = macro_auc(
        (T02_TARGETS_MATRIX[completed] >= 0.5).astype(np.int8), oof[completed]
    )
    print("T02 split Gold OOF", gold_macro, flush=True)
    print(_json.dumps(target_auc, indent=2), flush=True)

    oof_frame = pd.DataFrame({
        "StudyInstanceUID": np.asarray(uids)[completed],
        "Fold": runtime_folds[completed].astype(np.int8),
    })
    for j, target in enumerate(TARGETS):
        oof_frame[target] = oof[completed, j]
    oof_path = Path("/kaggle/working") / f"t02_partial_oof_{T02_SPLIT_TAG}.csv"
    oof_frame.to_csv(oof_path, index=False)

    weight_info = {}
    for path in paths:
        weight_info[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema": T02_SCHEMA,
        "split_schema": T02_SPLIT_SCHEMA,
        "split_tag": T02_SPLIT_TAG,
        "run_folds": T02_RUN_FOLDS,
        "seed": 20260930,
        "targets": TARGETS,
        "training": {
            "total_folds": 5, "completed_folds": len(T02_RUN_FOLDS),
            "epochs": CFG.epochs, "swa_last": CFG.swa_epochs,
            "gold_weight": 4.0,
            "label_fusion": "0.60 calibrated Rayan + 0.40 median of five public families; Gold override",
            "optimizer": "AdamW OneCycle", "lr_backbone": CFG.lr_backbone,
            "lr_head": CFG.lr_head,
        },
        "input": {"img": 224, "crop_mm": 130.0, "slots": [s[0] for s in SLOTS],
                  "group": 3, "n_group": 3, "window": "0.35,0.65"},
        "validation": {"partial_n": len(completed), "partial_gold_n": len(gold_idx),
                       "partial_gold_oof_macro_auc": gold_macro,
                       "partial_per_target_auc": target_auc,
                       "partial_derived_oof_macro_auc": derived_macro,
                       "fold_counts": np.bincount(runtime_folds, minlength=5).tolist(),
                       "gold_fold_counts": np.bincount(
                           runtime_folds[np.flatnonzero(T02_GOLD_MASK)], minlength=5
                       ).tolist()},
        "provenance": {"public_label_sha256": label_provenance,
                       "init_sha256": T02_INIT_SHA256,
                       "cache_meta": str(cache_meta_path), "cache_contract": cache_meta},
        "histories": histories,
        "weights": weight_info,
        "oof": {"file": oof_path.name, "sha256": _sha256(oof_path)},
        "elapsed_seconds": time.time() - T0,
    }
    manifest_path = Path("/kaggle/working") / f"t02_manifest_{T02_SPLIT_TAG}.json"
    manifest_path.write_text(_json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(_json.dumps({"status": "COMPLETE", "split": T02_SPLIT_TAG,
                       "folds": T02_RUN_FOLDS,
                       "partial_gold_oof_macro_auc": gold_macro,
                       "partial_derived_oof_macro_auc": derived_macro,
                       "elapsed_seconds": time.time() - T0}, indent=2), flush=True)


_main_t02()
'''


def main() -> None:
    source = SOURCE.read_text()
    marker = "def _main_t02():\n"
    start = source.rfind(marker)
    if start < 0 or not source.rstrip().endswith("_main_t02()"):
        raise RuntimeError("could not locate the generated T02 main driver")
    prefix = source[:start]
    base_metadata = json.loads(
        (SOURCE.parent / "kernel-metadata.json").read_text()
    )
    for tag, config in SPLITS.items():
        output_dir = ROOT / f"public_kernels/t02_split_{tag}"
        output_dir.mkdir(parents=True, exist_ok=True)
        code_name = f"rsna-knee-goldfold-convnext-{tag}.py"
        driver = MAIN_TEMPLATE.replace("__TAG__", repr(tag)).replace(
            "__FOLDS__", repr(config["folds"])
        )
        output = output_dir / code_name
        output.write_text(prefix + driver)
        metadata = dict(base_metadata)
        metadata.update({
            "id": config["id"],
            "title": config["title"],
            "code_file": code_name,
        })
        (output_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
        print(json.dumps({"tag": tag, "folds": config["folds"],
                          "script": str(output), "bytes": output.stat().st_size}))


if __name__ == "__main__":
    main()
