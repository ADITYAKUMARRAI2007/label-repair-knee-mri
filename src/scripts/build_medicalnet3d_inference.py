"""Build the standalone MedicalNet3D Kaggle inference kernel."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "public_kernels/medicalnet3d_infer"
OUT = OUT_DIR / "rsna-knee-medicalnet3d-infer.py"


PROGRAM = r'''#!/usr/bin/env python3
"""Standalone hidden-test inference for the M01 MedicalNet3D arm.

The test DICOM conversion matches the public 44-slice Raptor corpus contract
used in training.  Every study must complete; this program has no neutral or
zero fallback path.  Predictions are ranked per target for macro ROC AUC.
"""
from __future__ import annotations

import gc
import glob
import hashlib
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]
SLOT_SPECS = [("Sagittal", 1, 12), ("Sagittal", 0, 10),
              ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]
SLOT_BOUNDS = ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))
MAX_SLICES = 44
SOURCE_RES = 336
CROP_MM = 140.0
DEPTH = 16
RES = 112
MEDICALNET_SHA256 = "afa8055f3e47f4a18239495d92a7abc587902c69c31c743de2b2784653b72605"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, 1, dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(x + residual)


class MedicalNet10(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 1, 1)
        self.layer2 = self._make_layer(128, 2, 1)
        self.layer3 = self._make_layer(256, 1, 2)
        self.layer4 = self._make_layer(512, 1, 4)

    def _make_layer(self, planes, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm3d(planes))
        layer = BasicBlock(self.inplanes, planes, stride, dilation, downsample)
        self.inplanes = planes
        return nn.Sequential(layer)

    def frozen_stem(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

    def forward(self, x):
        x = self.layer4(self.frozen_stem(x))
        return F.adaptive_avg_pool3d(x, 1).flatten(1)


class StudyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MedicalNet10()
        self.slot_embedding = nn.Parameter(torch.zeros(5, 32))
        self.attention = nn.Sequential(nn.Linear(544, 128), nn.Tanh(), nn.Linear(128, 12))
        self.classifier_weight = nn.Parameter(torch.empty(12, 544))
        self.classifier_bias = nn.Parameter(torch.zeros(12))

    @staticmethod
    def normalize(x):
        flat = x.flatten(2)
        positive = flat > 0
        count = positive.sum(2, keepdim=True).clamp_min(1)
        mean = (flat * positive).sum(2, keepdim=True) / count
        var = (((flat - mean) * positive) ** 2).sum(2, keepdim=True) / count
        flat = (flat - mean) / torch.sqrt(var + 1e-6)
        return flat.view_as(x).clamp(-5, 5)

    def forward(self, x, slot_mask):
        batch, slots = x.shape[:2]
        x = x.float().div_(255.0).reshape(batch * slots, 1, DEPTH, RES, RES)
        features = self.encoder(self.normalize(x)).view(batch, slots, 512)
        slot = self.slot_embedding.unsqueeze(0).expand(batch, -1, -1)
        features = torch.cat([features, slot], dim=-1)
        scores = self.attention(features).masked_fill(~slot_mask.unsqueeze(-1), -1e4)
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("bst,bsd->btd", weights, features)
        return (pooled * self.classifier_weight.unsqueeze(0)).sum(-1) + self.classifier_bias


def find_test_root():
    candidates = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
                  "/kaggle/input/rsna-knee-abnormality-detection"]
    for root in candidates:
        if os.path.isfile(os.path.join(root, "test.csv")):
            return root
    for root, dirs, files in os.walk("/kaggle/input"):
        has_images = "test_series" in dirs or "test_images" in dirs
        dirs[:] = [d for d in dirs if d not in {"train_series", "test_series", "train_images", "test_images"}]
        if "test.csv" in files and has_images:
            return root
    raise RuntimeError("competition test root not found")


def find_checkpoint():
    candidates = []
    direct_roots = [
        "/kaggle/input/notebooks/adityakumarrai0401/rsna-knee-medicalnet3d-train-v1",
        "/kaggle/input/rsna-knee-medicalnet3d-train-v1",
    ]
    for root in direct_roots:
        for name in ("medicalnet3d_best.pt", "medicalnet3d_swa.pt"):
            path = os.path.join(root, name)
            if os.path.isfile(path):
                candidates.append(path)
    for root, dirs, files in os.walk("/kaggle/input"):
        relative = os.path.relpath(root, "/kaggle/input")
        depth = 0 if relative == "." else len(relative.split(os.sep))
        dirs[:] = [d for d in dirs if d not in {
            "competitions", "train_series", "test_series", "train_images", "test_images"
        }]
        if depth >= 6:
            dirs[:] = []
        for name in files:
            if name in ("medicalnet3d_best.pt", "medicalnet3d_swa.pt"):
                candidates.append(os.path.join(root, name))
    candidates = sorted(set(candidates))
    if not candidates:
        raise RuntimeError("no M01 checkpoint found under /kaggle/input")
    requested = os.environ.get("RSNA_MEDICALNET3D_CHECKPOINT", "auto")
    if requested != "auto":
        selected = [path for path in candidates if os.path.basename(path) == requested]
        if len(selected) != 1:
            raise RuntimeError(f"requested checkpoint {requested!r}, got {selected}")
        return selected[0]
    scored = []
    for path in candidates:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        score = float(checkpoint.get("gold_auc", float("nan")))
        if np.isfinite(score):
            scored.append((score, path))
        del checkpoint
    if not scored:
        raise RuntimeError(f"M01 checkpoints have no finite Gold58 score: {candidates}")
    scored.sort(key=lambda item: (item[0], "_swa.pt" in item[1]), reverse=True)
    print("checkpoint candidates", [(round(s, 6), os.path.basename(p)) for s, p in scored],
          flush=True)
    return scored[0][1]


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "medicalnet3d_checkpoint_v1":
        raise RuntimeError(f"unexpected checkpoint schema {checkpoint.get('schema')!r}")
    if (
        checkpoint.get("lab") != LAB
        or tuple(map(tuple, checkpoint.get("slots", ()))) != SLOT_BOUNDS
        or int(checkpoint.get("depth")) != DEPTH
        or int(checkpoint.get("res")) != RES
        or checkpoint.get("medicalnet_sha256") != MEDICALNET_SHA256
    ):
        raise RuntimeError("M01 checkpoint preprocessing contract mismatch")
    score = float(checkpoint.get("gold_auc", float("nan")))
    target_auc = checkpoint.get("gold_target_auc", {})
    state = checkpoint.get("model")
    if (
        not np.isfinite(score) or not 0.0 <= score <= 1.0
        or set(target_auc) != set(LAB)
        or not isinstance(state, dict) or len(state) < 50
    ):
        raise RuntimeError("M01 checkpoint score, targets, or model state is invalid")
    epoch = checkpoint.get("epoch")
    epochs = [int(value) for value in epoch] if isinstance(epoch, (list, tuple)) else [int(epoch)]
    if not epochs or len(epochs) > 3 or any(value not in range(5) for value in epochs):
        raise RuntimeError(f"M01 checkpoint epoch receipt is invalid: {epoch!r}")
    model = StudyClassifier()
    model.load_state_dict(state, strict=True)
    info = {key: checkpoint.get(key) for key in
            ("schema", "gold_auc", "gold_target_auc", "epoch", "medicalnet_sha256")}
    model.eval().to(device)
    del checkpoint
    gc.collect()
    return model, info


def make_reader():
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(series_dir):
        records, spacing = [], []
        for path in glob.glob(os.path.join(series_dir, "*.dcm")):
            try:
                header = pydicom.dcmread(path, stop_before_pixels=True)
                iop = getattr(header, "ImageOrientationPatient", None)
                ipp = getattr(header, "ImagePositionPatient", None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    row = np.asarray(iop[:3], float)
                    column = np.asarray(iop[3:], float)
                    position = float(np.dot(np.asarray(ipp, float), np.cross(row, column)))
                else:
                    position = float(getattr(header, "InstanceNumber", 0) or 0)
                pixel_spacing = getattr(header, "PixelSpacing", None)
                pixel_spacing = float(pixel_spacing[0]) if pixel_spacing is not None else 0.5
                spacing.append(pixel_spacing)
                records.append((position, path, pixel_spacing))
            except Exception:
                records.append((0.0, path, 0.5))
        records.sort(key=lambda item: item[0])
        median_spacing = float(np.median(spacing)) if spacing else 0.5
        return [(path, ps) for _, path, ps in records], median_spacing

    def read_pixels(path):
        dicom = pydicom.dcmread(path)
        image = apply_modality_lut(dicom.pixel_array, dicom).astype(np.float32)
        if str(getattr(dicom, "PhotometricInterpretation", "")) == "MONOCHROME1":
            image = image.max() - image
        return image

    def crop_resize(image, pixel_spacing):
        height, width = image.shape
        pixels = int(round(CROP_MM / max(pixel_spacing, 1e-3)))
        pixels = min(pixels, height, width)
        y0, x0 = (height - pixels) // 2, (width - pixels) // 2
        return cv2.resize(image[y0:y0 + pixels, x0:x0 + pixels],
                          (SOURCE_RES, SOURCE_RES), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_pixels, crop_resize


def pick_series(rows, plane, fluid, used):
    def protocol_flag(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in (
                "", "nan", "none", "<na>", "unknown"):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in ("true", "yes", "y", "t"):
                return 1
            if text in ("false", "no", "n", "f"):
                return 0
            return None
        if not np.isfinite(number) or number not in (0.0, 1.0):
            return None
        return int(number)

    candidates = [row for row in rows if row["Anatomical_Plane"] == plane
                  and row["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        preferred = [row for row in candidates
                     if protocol_flag(row.get("Fluid_Sensitive")) == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def build_raster_stack(study_id, series_records, test_series_dir, reader):
    order_and_meta, read_pixels, crop_resize = reader
    rows = series_records.get(study_id, [])
    volume = np.zeros((MAX_SLICES, SOURCE_RES, SOURCE_RES), np.uint8)
    cursor, used = 0, set()
    for plane, fluid, count in SLOT_SPECS:
        series = pick_series(rows, plane, fluid, used)
        if series is None:
            cursor += count
            continue
        used.add(series["SeriesInstanceUID"])
        files, median_spacing = order_and_meta(
            os.path.join(test_series_dir, study_id, series["SeriesInstanceUID"]))
        if not files:
            cursor += count
            continue
        n = len(files)
        lo, hi = int(n * 0.06), int(n * 0.94) - 1
        hi = max(hi, lo)
        picks = np.linspace(lo, hi, count).round().astype(int) if n > 1 else [0] * count
        images, spacings = [], []
        for pick in picks:
            path, spacing = files[min(int(pick), n - 1)]
            try:
                images.append(read_pixels(path))
            except Exception:
                images.append(None)
            spacings.append(spacing)
        valid = [image for image in images if image is not None]
        if valid:
            pixels = np.concatenate([image.ravel() for image in valid])
            low, high = np.percentile(pixels, [2.0, 98.0])
        else:
            low, high = 0.0, 1.0
        for image, spacing in zip(images, spacings):
            if image is not None:
                image = np.clip((image - low) / (high - low + 1e-6), 0, 1)
                image = crop_resize(image, spacing if spacing > 0 else median_spacing)
                volume[cursor] = (image * 255).astype(np.uint8)
            cursor += 1
    if cursor != MAX_SLICES:
        raise RuntimeError(f"Raptor stack cursor ended at {cursor}, expected {MAX_SLICES}")
    mask = volume.reshape(MAX_SLICES, -1).sum(1) > 0
    return volume, mask


def stack_to_model_input(volume, valid_mask):
    x = np.zeros((1, 5, 1, DEPTH, RES, RES), dtype=np.uint8)
    slot_mask = np.zeros((1, 5), dtype=np.bool_)
    for slot, (lo, hi) in enumerate(SLOT_BOUNDS):
        valid = np.flatnonzero(valid_mask[lo:hi]) + lo
        if len(valid) == 0:
            continue
        slot_mask[0, slot] = True
        small = np.stack([
            cv2.resize(volume[int(slice_index)], (RES, RES), interpolation=cv2.INTER_AREA)
            for slice_index in valid
        ]).astype(np.float32)
        positions = np.linspace(0, len(valid) - 1, DEPTH)
        lower = np.floor(positions).astype(np.int64)
        upper = np.ceil(positions).astype(np.int64)
        alpha = (positions - lower).astype(np.float32)[:, None, None]
        interpolated = small[lower] * (1 - alpha) + small[upper] * alpha
        x[0, slot, 0] = np.clip(interpolated.round(), 0, 255).astype(np.uint8)
    if not slot_mask.any():
        raise RuntimeError("study has no usable sequence slot")
    return torch.from_numpy(x), torch.from_numpy(slot_mask)


def rank_percentile(values):
    if not np.isfinite(values).all():
        raise RuntimeError("prediction matrix contains non-finite values")
    order = values.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, values.shape[0] - 1)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = time.time()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    device = torch.device("cuda")
    test_root = find_test_root()
    series_dir = os.path.join(test_root, "test_series")
    if not os.path.isdir(series_dir):
        series_dir = os.path.join(test_root, "test_images")
    test = pd.read_csv(os.path.join(test_root, "test.csv"),
                       dtype={"StudyInstanceUID": str})
    test_ids = test.StudyInstanceUID.tolist()
    series = pd.read_csv(os.path.join(test_root, "test_series.csv"),
                         dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    grouped = {uid: frame.to_dict("records")
               for uid, frame in series.groupby("StudyInstanceUID", sort=False)}
    sample = pd.read_csv(os.path.join(test_root, "sample_submission.csv"), nrows=1)
    expected_columns = sample.columns.tolist()
    if expected_columns != ["StudyInstanceUID", *LAB]:
        raise RuntimeError(f"unexpected submission columns {expected_columns}")

    checkpoint_path = find_checkpoint()
    model, checkpoint_info = load_model(checkpoint_path, device)
    reader = make_reader()
    raw = np.full((len(test_ids), len(LAB)), np.nan, np.float32)
    with torch.no_grad():
        for index, study_id in enumerate(test_ids):
            try:
                volume, mask = build_raster_stack(study_id, grouped, series_dir, reader)
                x, slot_mask = stack_to_model_input(volume, mask)
                x, slot_mask = x.to(device), slot_mask.to(device)
                with torch.autocast("cuda", dtype=torch.float16):
                    raw[index] = torch.sigmoid(model(x, slot_mask).float())[0].cpu().numpy()
            except Exception as error:
                raise RuntimeError(
                    f"M01 inference failed for study {index} {study_id}: "
                    f"{type(error).__name__}: {error}") from error
            if (index + 1) % 100 == 0 or index + 1 == len(test_ids):
                print(f"M01 {index+1}/{len(test_ids)} elapsed={time.time()-started:.0f}s", flush=True)

    ranked = rank_percentile(raw).astype(np.float32)
    submission = pd.DataFrame(ranked, columns=LAB)
    submission.insert(0, "StudyInstanceUID", test_ids)
    if submission.columns.tolist() != expected_columns:
        raise RuntimeError("submission column order drift")
    if submission.StudyInstanceUID.tolist() != test_ids:
        raise RuntimeError("submission study order drift")
    if not np.isfinite(submission[LAB].to_numpy()).all():
        raise RuntimeError("submission contains non-finite values")

    output = os.environ.get(
        "RSNA_MEDICALNET3D_OUTPUT", "/kaggle/working/submission.csv"
    )
    predictions = os.environ.get(
        "RSNA_MEDICALNET3D_PREDICTIONS",
        "/kaggle/working/medicalnet3d_predictions.npz",
    )
    submission.to_csv(output, index=False)
    np.savez(predictions, study_uids=np.asarray(test_ids), raw_probabilities=raw,
             ranked_predictions=ranked, labels=np.asarray(LAB))
    receipt = {
        "schema": "medicalnet3d_inference_v1", "status": "COMPLETE",
        "studies": len(test_ids), "fallback_studies": 0,
        "input": {"source_resolution": SOURCE_RES, "crop_mm": CROP_MM,
                  "slot_specs": SLOT_SPECS, "slot_bounds": SLOT_BOUNDS,
                  "model_depth": DEPTH, "model_resolution": RES,
                  "span": [0.06, 0.94], "depth_resampling": "linear",
                  "normalization": "nonzero_zscore"},
        "checkpoint": os.path.basename(checkpoint_path),
        "checkpoint_info": checkpoint_info,
        "elapsed_seconds": round(time.time() - started, 2),
        "output": output, "submission_sha256": sha256(output),
        "predictions_sha256": sha256(predictions),
    }
    with open("/kaggle/working/medicalnet3d_receipt.json", "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''


METADATA = {
    "id": "adityakumarrai0401/rsna-knee-medicalnet3d-infer-v1",
    "title": "RSNA Knee MedicalNet3D Infer V1",
    "code_file": OUT.name,
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [],
    "kernel_sources": ["adityakumarrai0401/rsna-knee-medicalnet3d-train-v1"],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": [],
    "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
    "machine_shape": "NvidiaTeslaT4",
}


def build() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(PROGRAM, encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(METADATA, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
