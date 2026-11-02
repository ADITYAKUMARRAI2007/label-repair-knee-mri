"""Notebook cell injected before the DINOsaur V5 final publish cell.

It loads the independently trained ConvNeXt arm, decodes test studies in bounded
forked workers, and applies the trainer's conservative per-target rank blend.
"""

import gc as _cx_gc
import hashlib as _cx_hashlib
import json as _cx_json
import math as _cx_math
import os as _cx_os
import subprocess as _cx_subprocess
import sys as _cx_sys
import time as _cx_time
from pathlib import Path as _CxPath

import cv2 as _cx_cv2
import numpy as _cx_np
import pandas as _cx_pd
import pydicom as _cx_pydicom
import timm as _cx_timm
import torch as _cx_torch
import torch.nn as _cx_nn
import torch.nn.functional as _cx_F


_CX_LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]
_CX_FEATURE_DIM = 768
_CX_IMAGE_SIZE = 224
_CX_CROP_MM = 150.0
_CX_MAX_SERIES = 6
_CX_QUANTILES = (0.15, 0.325, 0.50, 0.675, 0.85)
_CX_DECODE_WORKERS = 4
_CX_DECODE_CHUNK = 20
_CX_PLANE_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}
_CX_MEAN = _cx_torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_CX_STD = _cx_torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _cx_find_comp_root():
    for path in (
        _CxPath("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
        _CxPath("/kaggle/input/rsna-knee-abnormality-detection"),
    ):
        if (path / "test.csv").is_file() and (path / "test_series.csv").is_file():
            return path
    raise FileNotFoundError("RSNA Knee test input is not mounted")


def _cx_find_artifact_root():
    matches = []
    blocked = {"train_series", "test_series", "train_images", "test_images"}
    for root, dirs, files in _cx_os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in blocked]
        if "dinov5_convnext_manifest.json" in files:
            matches.append(_CxPath(root))
    matches = sorted(set(matches), key=str)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one ConvNeXt trainer artifact root, found {matches}")
    return matches[0]


def _cx_choose_series(frame):
    f = frame.copy()
    for col in ("Fluid_Sensitive", "Fat_Suppression"):
        if col not in f.columns:
            f[col] = 0
        f[col] = _cx_pd.to_numeric(f[col], errors="coerce").fillna(0).astype(int)
    f["Anatomical_Plane"] = f["Anatomical_Plane"].astype(str)
    chosen, used = [], set()
    for plane in ("Sagittal", "Coronal", "Axial"):
        z = f[f["Anatomical_Plane"].eq(plane)]
        if len(z):
            row = z.sort_values(
                ["Fluid_Sensitive", "Fat_Suppression"], ascending=False
            ).iloc[0]
            chosen.append(row)
            used.add(str(row["SeriesInstanceUID"]))
    for fluid, fat in ((1, 1), (1, 0), (0, 1), (0, 0)):
        z = f[(f["Fluid_Sensitive"] == fluid) & (f["Fat_Suppression"] == fat)]
        for _, row in z.iterrows():
            series_uid = str(row["SeriesInstanceUID"])
            if series_uid not in used:
                chosen.append(row)
                used.add(series_uid)
                break
    for _, row in f.iterrows():
        if len(chosen) >= _CX_MAX_SERIES:
            break
        series_uid = str(row["SeriesInstanceUID"])
        if series_uid not in used:
            chosen.append(row)
            used.add(series_uid)
    return _cx_pd.DataFrame(chosen[:_CX_MAX_SERIES]) if chosen else f.head(0)


def _cx_ordered_files(directory):
    values = []
    for path in _CxPath(directory).glob("*.dcm"):
        try:
            header = _cx_pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            orientation = getattr(header, "ImageOrientationPatient", None)
            position = getattr(header, "ImagePositionPatient", None)
            if orientation is not None and position is not None and len(orientation) == 6:
                normal = _cx_np.cross(
                    _cx_np.asarray(orientation[:3], _cx_np.float64),
                    _cx_np.asarray(orientation[3:6], _cx_np.float64),
                )
                key = float(_cx_np.dot(_cx_np.asarray(position[:3], _cx_np.float64), normal))
            else:
                key = float(getattr(header, "InstanceNumber", 0) or 0)
            values.append((key, str(path)))
            del header
        except Exception:
            pass
    values.sort(key=lambda item: item[0])
    return [path for _, path in values]


def _cx_read_slice(path):
    ds = _cx_pydicom.dcmread(str(path), force=True)
    image = ds.pixel_array.astype(_cx_np.float32, copy=False)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        image = image * slope + intercept
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        image = image.max() - image
    spacing_value = getattr(ds, "PixelSpacing", None)
    spacing = float(spacing_value[0]) if spacing_value is not None else None
    if spacing and spacing > 0:
        side = max(16, min(int(round(_CX_CROP_MM / spacing)), *image.shape[:2]))
        center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
        y0 = max(0, center_y - side // 2)
        x0 = max(0, center_x - side // 2)
        image = image[y0:y0 + side, x0:x0 + side]
    low, high = _cx_np.percentile(image, [1, 99])
    image = _cx_np.clip((image - low) / (high - low + 1e-6), 0, 1)
    image = _cx_cv2.resize(
        image, (_CX_IMAGE_SIZE, _CX_IMAGE_SIZE), interpolation=_cx_cv2.INTER_AREA
    )
    result = _cx_np.clip(image * 255.0 + 0.5, 0, 255).astype(_cx_np.uint8)
    del ds, image
    return result


def _cx_decode_shard(start, stop, test_ids, series_by_study, series_root, output_root):
    """CPU-only child. Its exit is the native-memory reclamation boundary."""
    for index in range(start, stop):
        study_uid = test_ids[index]
        images, metadata = [], []
        frame = series_by_study.get(study_uid)
        if frame is not None:
            for _, row in _cx_choose_series(frame).iterrows():
                series_uid = str(row["SeriesInstanceUID"])
                files = _cx_ordered_files(series_root / study_uid / series_uid)
                if not files:
                    continue
                for quantile in _CX_QUANTILES:
                    center = int(round(quantile * (len(files) - 1)))
                    positions = [max(0, min(len(files) - 1, center + delta)) for delta in (-1, 0, 1)]
                    try:
                        images.append(_cx_np.stack([_cx_read_slice(files[j]) for j in positions]))
                        plane = _cx_np.zeros(3, _cx_np.float32)
                        plane[_CX_PLANE_ID.get(str(row.get("Anatomical_Plane", "")), 2)] = 1.0
                        fluid = _cx_pd.to_numeric(row.get("Fluid_Sensitive", 0), errors="coerce")
                        fat = _cx_pd.to_numeric(row.get("Fat_Suppression", 0), errors="coerce")
                        metadata.append(_cx_np.array([
                            *plane,
                            0.0 if _cx_pd.isna(fluid) else float(fluid),
                            0.0 if _cx_pd.isna(fat) else float(fat),
                            quantile,
                        ], _cx_np.float16))
                    except Exception:
                        continue
        image_array = (
            _cx_np.stack(images)
            if images
            else _cx_np.zeros((0, 3, _CX_IMAGE_SIZE, _CX_IMAGE_SIZE), _cx_np.uint8)
        )
        meta_array = (
            _cx_np.stack(metadata)
            if metadata
            else _cx_np.zeros((0, 6), _cx_np.float16)
        )
        _cx_np.savez(output_root / f"{study_uid}.npz", images=image_array, meta=meta_array)


class _CxLabelAwareHead(_cx_nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.proj = _cx_nn.Sequential(
            _cx_nn.Linear(_CX_FEATURE_DIM, d),
            _cx_nn.LayerNorm(d),
            _cx_nn.GELU(),
            _cx_nn.Dropout(0.15),
        )
        self.meta = _cx_nn.Sequential(
            _cx_nn.Linear(6, 64), _cx_nn.GELU(), _cx_nn.Linear(64, d)
        )
        self.query = _cx_nn.Parameter(_cx_torch.randn(len(_CX_LABELS), d) * 0.02)
        self.key = _cx_nn.Linear(d, d, bias=False)
        self.value = _cx_nn.Linear(d, d, bias=False)
        self.label_emb = _cx_nn.Parameter(_cx_torch.randn(len(_CX_LABELS), d) * 0.02)
        layer = _cx_nn.TransformerEncoderLayer(
            d_model=d,
            nhead=8,
            dim_feedforward=768,
            dropout=0.12,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_label = _cx_nn.TransformerEncoder(layer, num_layers=2)
        self.norm = _cx_nn.LayerNorm(d)
        self.h1 = _cx_nn.Linear(d, 192)
        self.out = _cx_nn.Parameter(_cx_torch.randn(len(_CX_LABELS), 192) * 0.02)
        self.bias = _cx_nn.Parameter(_cx_torch.zeros(len(_CX_LABELS)))

    def forward(self, features, metadata, mask):
        hidden = self.proj(features) + self.meta(metadata)
        keys = self.key(hidden)
        values = self.value(hidden)
        scores = _cx_torch.einsum("ld,bsd->bls", self.query, keys) / _cx_math.sqrt(keys.shape[-1])
        scores = scores.masked_fill(~mask[:, None, :], -1e4)
        attention = _cx_torch.softmax(scores, dim=-1)
        pooled = _cx_torch.einsum("bls,bsd->bld", attention, values)
        pooled = self.cross_label(pooled + self.label_emb[None])
        pooled = _cx_F.gelu(self.h1(self.norm(pooled)))
        return _cx_torch.einsum("blh,lh->bl", pooled, self.out) + self.bias[None]


def _cx_rank01(values):
    values = _cx_np.asarray(values, _cx_np.float64)
    if len(values) == 1:
        return _cx_np.full_like(values, 0.5)
    return (_cx_pd.DataFrame(values).rank(method="average").to_numpy() - 1) / (len(values) - 1)


def _cx_sha256_file(path):
    digest = _cx_hashlib.sha256()
    with _CxPath(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_cx_started = _cx_time.time()
_cx_root = _cx_find_comp_root()
_cx_artifacts = _cx_find_artifact_root()
_cx_manifest_path = _cx_artifacts / "dinov5_convnext_manifest.json"
_cx_manifest = _cx_json.loads(_cx_manifest_path.read_text())
if _cx_manifest.get("schema") != "dinov5_convnext_diversity_v1":
    raise RuntimeError(f"ConvNeXt manifest schema mismatch: {_cx_manifest.get('schema')!r}")
if int(_cx_manifest.get("seed", -1)) != 20260921:
    raise RuntimeError("ConvNeXt manifest seed mismatch")
if _cx_manifest.get("targets") != _CX_LABELS:
    raise RuntimeError("ConvNeXt manifest target order mismatch")
if _cx_manifest.get("backbone_model_name") != "convnext_tiny.fb_in22k_ft_in1k":
    raise RuntimeError("ConvNeXt backbone model contract mismatch")
_cx_expected_input = {
    "crop_mm": _CX_CROP_MM,
    "image_size": _CX_IMAGE_SIZE,
    "max_series": _CX_MAX_SERIES,
    "clip_quantiles": list(_CX_QUANTILES),
    "order": "physical ImagePositionPatient projection",
}
if _cx_manifest.get("input") != _cx_expected_input:
    raise RuntimeError(f"ConvNeXt input contract mismatch: {_cx_manifest.get('input')}")
_cx_validation = _cx_manifest.get("validation", {})
if int(_cx_validation.get("gold_n", -1)) != 58:
    raise RuntimeError("ConvNeXt Gold58 count mismatch")
_cx_fold_metrics = _cx_validation.get("fold_metrics", {})
if set(_cx_fold_metrics) != {"0", "1", "2", "3", "4"}:
    raise RuntimeError("ConvNeXt fold receipt mismatch")
if sum(int(_cx_fold_metrics[str(i)].get("n_gold", -1)) for i in range(5)) != 58:
    raise RuntimeError("ConvNeXt fold validation counts do not cover Gold58")
if set(_cx_validation.get("per_target_auc", {})) != set(_CX_LABELS):
    raise RuntimeError("ConvNeXt per-target validation contract mismatch")
if set(_cx_manifest.get("recommended_main_beta", {})) != set(_CX_LABELS):
    raise RuntimeError("ConvNeXt blend recommendation contract mismatch")
_cx_expected_heads = [f"dinov5_convnext_head_fold{i}.pt" for i in range(5)]
_cx_weights = _cx_manifest.get("weights", {})
if _cx_weights != {
    "backbone": "dinov5_convnext_backbone.pt",
    "heads": _cx_expected_heads,
}:
    raise RuntimeError(f"ConvNeXt weight manifest mismatch: {_cx_weights}")
_cx_weight_paths = [_cx_artifacts / _cx_weights["backbone"]] + [
    _cx_artifacts / name for name in _cx_expected_heads
]
for _cx_weight_path in _cx_weight_paths:
    if not _cx_weight_path.is_file() or _cx_weight_path.stat().st_size < 1_000_000:
        raise RuntimeError(f"ConvNeXt artifact missing or implausibly small: {_cx_weight_path}")
if not _cx_torch.cuda.is_available():
    raise RuntimeError("ConvNeXt inference requires CUDA")

_cx_test = _cx_pd.read_csv(_cx_root / "test.csv", dtype={"StudyInstanceUID": str})
_cx_test_ids = _cx_test["StudyInstanceUID"].astype(str).tolist()
if _cx_test_ids != list(map(str, _RSNA_TEST_IDS)):
    raise RuntimeError("ConvNeXt test study order mismatch")

_cx_device = _cx_torch.device("cuda:0")
_cx_gc.collect()
for _cx_gpu in range(_cx_torch.cuda.device_count()):
    _cx_torch.cuda.empty_cache()

_cx_encoder = _cx_timm.create_model(
    _cx_manifest["backbone_model_name"],
    pretrained=False,
    num_classes=0,
    global_pool="avg",
)
_cx_backbone_state = _cx_torch.load(
    _cx_artifacts / _cx_weights["backbone"],
    map_location="cpu",
    weights_only=True,
)
if not isinstance(_cx_backbone_state, dict) or len(_cx_backbone_state) < 100:
    raise RuntimeError("ConvNeXt backbone state is incomplete")
_cx_encoder.load_state_dict(_cx_backbone_state, strict=True)
_cx_encoder.eval().to(_cx_device)
del _cx_backbone_state

_cx_heads = []
for _cx_fold, _cx_name in enumerate(_cx_expected_heads):
    _cx_checkpoint = _cx_torch.load(_cx_artifacts / _cx_name, map_location="cpu", weights_only=True)
    if not isinstance(_cx_checkpoint, dict):
        raise RuntimeError(f"ConvNeXt fold {_cx_fold} checkpoint is not a mapping")
    if int(_cx_checkpoint.get("fold", -1)) != _cx_fold:
        raise RuntimeError(f"ConvNeXt fold checkpoint identity mismatch: {_cx_name}")
    if _cx_checkpoint.get("targets") != _CX_LABELS:
        raise RuntimeError(f"ConvNeXt fold target contract mismatch: {_cx_name}")
    if not isinstance(_cx_checkpoint.get("state"), dict) or not _cx_checkpoint["state"]:
        raise RuntimeError(f"ConvNeXt fold state is empty: {_cx_name}")
    _cx_head = _CxLabelAwareHead()
    _cx_head.load_state_dict(_cx_checkpoint["state"], strict=True)
    _cx_head.eval().to(_cx_device)
    _cx_heads.append(_cx_head)
    del _cx_checkpoint

_cx_decode_root = _CxPath("/kaggle/temp/dinov5_convnext_test_decode")
try:
    _cx_decode_root.mkdir(parents=True, exist_ok=True)
except OSError:
    _cx_decode_root = _CxPath("/tmp/dinov5_convnext_test_decode")
    _cx_decode_root.mkdir(parents=True, exist_ok=True)

_cx_probabilities = _cx_np.full((len(_cx_test_ids), len(_CX_LABELS)), 0.5, _cx_np.float32)
_cx_token_counts = _cx_np.zeros(len(_cx_test_ids), _cx_np.int16)
_cx_fallbacks = 0
_cx_wave = _CX_DECODE_WORKERS * _CX_DECODE_CHUNK

for _cx_wave_start in range(0, len(_cx_test_ids), _cx_wave):
    _cx_processes, _cx_bounds = [], []
    for _cx_start in range(
        _cx_wave_start,
        min(_cx_wave_start + _cx_wave, len(_cx_test_ids)),
        _CX_DECODE_CHUNK,
    ):
        _cx_stop = min(_cx_start + _CX_DECODE_CHUNK, len(_cx_test_ids))
        _cx_env = dict(_cx_os.environ)
        _cx_env["CUDA_VISIBLE_DEVICES"] = ""
        _cx_env["OMP_NUM_THREADS"] = "1"
        _cx_process = _cx_subprocess.Popen(
            [
                _cx_sys.executable,
                "/kaggle/working/_convnext_decode_worker.py",
                str(_cx_root),
                str(_cx_start),
                str(_cx_stop),
                str(_cx_decode_root),
            ],
            env=_cx_env,
        )
        _cx_processes.append(_cx_process)
        _cx_bounds.append((_cx_start, _cx_stop))
    for _cx_process, (_cx_start, _cx_stop) in zip(_cx_processes, _cx_bounds):
        _cx_code = _cx_process.wait()
        if _cx_code != 0:
            raise RuntimeError(
                f"ConvNeXt decode worker {_cx_start}:{_cx_stop} exited {_cx_code}"
            )

    for _cx_start, _cx_stop in _cx_bounds:
        for _cx_index in range(_cx_start, _cx_stop):
            _cx_uid = _cx_test_ids[_cx_index]
            _cx_raw_path = _cx_decode_root / f"{_cx_uid}.npz"
            with _cx_np.load(_cx_raw_path, allow_pickle=False) as _cx_z:
                _cx_images = _cx_z["images"]
                _cx_meta_array = _cx_z["meta"].astype(_cx_np.float32)
            _cx_raw_path.unlink(missing_ok=True)
            _cx_token_counts[_cx_index] = len(_cx_images)
            if len(_cx_images) == 0:
                _cx_fallbacks += 1
                continue
            _cx_feature_parts = []
            with _cx_torch.inference_mode():
                for _cx_batch_start in range(0, len(_cx_images), 16):
                    _cx_batch = _cx_torch.from_numpy(
                        _cx_images[_cx_batch_start:_cx_batch_start + 16]
                    ).float().div_(255.0)
                    _cx_batch = ((_cx_batch - _CX_MEAN) / _CX_STD).to(_cx_device)
                    with _cx_torch.autocast(device_type="cuda", dtype=_cx_torch.float16):
                        _cx_features = _cx_encoder(_cx_batch).float()
                    _cx_feature_parts.append(_cx_features.cpu())
                    del _cx_batch, _cx_features
                _cx_features = _cx_torch.cat(_cx_feature_parts)[None].to(_cx_device)
                _cx_metadata = _cx_torch.from_numpy(_cx_meta_array)[None].to(_cx_device)
                _cx_mask = _cx_torch.ones(
                    (1, _cx_features.shape[1]), dtype=_cx_torch.bool, device=_cx_device
                )
                _cx_fold_predictions = []
                for _cx_head in _cx_heads:
                    with _cx_torch.autocast(device_type="cuda", dtype=_cx_torch.float16):
                        _cx_logits = _cx_head(_cx_features, _cx_metadata, _cx_mask)
                    _cx_fold_predictions.append(_cx_torch.sigmoid(_cx_logits.float()).cpu().numpy()[0])
                _cx_probabilities[_cx_index] = _cx_np.mean(_cx_fold_predictions, axis=0)
            del _cx_images, _cx_meta_array, _cx_feature_parts, _cx_features, _cx_metadata, _cx_mask

    _cx_completed = min(_cx_wave_start + _cx_wave, len(_cx_test_ids))
    print(
        f"[convnext] {_cx_completed}/{len(_cx_test_ids)} studies; "
        f"fallbacks={_cx_fallbacks}; elapsed={(_cx_time.time()-_cx_started)/60:.1f}m",
        flush=True,
    )
    if "T0" in globals() and _cx_time.time() - T0 > 8.75 * 3600:
        raise TimeoutError("ConvNeXt arm exceeded the 8.75-hour complete-run guard")

if _cx_fallbacks:
    raise RuntimeError(f"ConvNeXt arm had {_cx_fallbacks} empty-study fallbacks")

_cx_arm_rank = _cx_rank01(_cx_probabilities)
_cx_arm_frame = _cx_pd.DataFrame(_cx_arm_rank, columns=_CX_LABELS)
_cx_arm_frame.insert(0, "StudyInstanceUID", _cx_test_ids)
_cx_arm_path = _CxPath("/kaggle/working/_convnext_arm.csv")
_cx_arm_frame.to_csv(_cx_arm_path, index=False)

_cx_standalone = _cx_os.environ.get("RSNA_CONVNEXT_STANDALONE", "0") == "1"
if _cx_standalone:
    _cx_output_path = _CxPath("/kaggle/working/submission.csv")
    _cx_arm_frame.to_csv(_cx_output_path, index=False)
    _cx_beta = _cx_np.zeros(len(_CX_LABELS), _cx_np.float64)
    _cx_schema = "dinov5_convnext_standalone_inference_v1"
else:
    _cx_base_path = _CxPath("/kaggle/working/_pipeline_stage.csv")
    _cx_base = rsna_frame(
        _cx_pd.read_csv(_cx_base_path, dtype={"StudyInstanceUID": str}),
        _cx_test_ids,
        _CX_LABELS,
        "pre-ConvNeXt baseline",
    )
    _cx_base_rank = _cx_rank01(_cx_base[_CX_LABELS].to_numpy())
    _cx_beta = _cx_np.array([
        float(_cx_manifest["recommended_main_beta"].get(label, 0.0))
        for label in _CX_LABELS
    ], _cx_np.float64)
    if (_cx_beta < 0).any() or (_cx_beta > 0.12).any():
        raise RuntimeError(f"Unsafe ConvNeXt blend weights: {_cx_beta.tolist()}")
    _cx_blended = _cx_rank01(
        (1.0 - _cx_beta[None]) * _cx_base_rank + _cx_beta[None] * _cx_arm_rank
    )
    _cx_output = _cx_pd.DataFrame(_cx_blended.astype(_cx_np.float32), columns=_CX_LABELS)
    _cx_output.insert(0, "StudyInstanceUID", _cx_test_ids)
    _cx_output.to_csv(_cx_base_path, index=False)
    _cx_output_path = _cx_base_path
    _cx_schema = "dinov5_convnext_diversity_blend_v1"


rsna_save_predictions("convnext_arm", _cx_test_ids, _cx_arm_rank, _CX_LABELS)
_cx_prediction_path = _CxPath(
    "/kaggle/working/diagnostics/convnext_arm_predictions.npz"
)
if not _cx_prediction_path.is_file():
    raise RuntimeError("ConvNeXt prediction artifact was not persisted")

_cx_receipt = {
    "schema": _cx_schema,
    "status": "COMPLETE",
    "studies": len(_cx_test_ids),
    "fallbacks": int(_cx_fallbacks),
    "fallback_studies": int(_cx_fallbacks),
    "token_count_min": int(_cx_token_counts.min()),
    "token_count_median": float(_cx_np.median(_cx_token_counts)),
    "token_count_max": int(_cx_token_counts.max()),
    "gold58_oof_macro_auc": _cx_manifest["validation"]["oof_macro_auc"],
    "per_target_gold58_auc": _cx_manifest["validation"]["per_target_auc"],
    "per_target_beta": dict(zip(_CX_LABELS, _cx_beta.tolist())),
    "output": str(_cx_output_path),
    "manifest_sha256": _cx_sha256_file(_cx_manifest_path),
    "weight_sha256": {
        path.name: _cx_sha256_file(path) for path in _cx_weight_paths
    },
    "submission_sha256": _cx_sha256_file(_cx_output_path),
    "arm_sha256": _cx_sha256_file(_cx_arm_path),
    "predictions_sha256": _cx_sha256_file(_cx_prediction_path),
    "elapsed_seconds": _cx_time.time() - _cx_started,
}
rsna_json("/kaggle/working/diagnostics/convnext_blend_receipt.json", _cx_receipt)

del _cx_encoder, _cx_heads
_cx_gc.collect()
for _cx_gpu in range(_cx_torch.cuda.device_count()):
    _cx_torch.cuda.empty_cache()
import shutil as _cx_shutil
_cx_shutil.rmtree(_cx_decode_root, ignore_errors=True)
print("[convnext] diversity blend complete", _cx_receipt, flush=True)
