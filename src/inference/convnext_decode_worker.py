"""Bounded CPU-only DICOM decoder used by the ConvNeXt blend notebook."""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom


ROOT = Path(sys.argv[1])
START = int(sys.argv[2])
STOP = int(sys.argv[3])
OUTPUT = Path(sys.argv[4])
OUTPUT.mkdir(parents=True, exist_ok=True)
cv2.setNumThreads(1)

IMAGE_SIZE = 224
CROP_MM = 150.0
MAX_SERIES = 6
QUANTILES = (0.15, 0.325, 0.50, 0.675, 0.85)
PLANE_ID = {"Sagittal": 0, "Coronal": 1, "Axial": 2}

test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
series = pd.read_csv(
    ROOT / "test_series.csv",
    dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
)
series["StudyInstanceUID"] = series["StudyInstanceUID"].astype(str)
series["SeriesInstanceUID"] = series["SeriesInstanceUID"].astype(str)
series_by_study = {key: frame.copy() for key, frame in series.groupby("StudyInstanceUID")}
test_ids = test["StudyInstanceUID"].astype(str).tolist()[START:STOP]


def choose_series(frame):
    frame = frame.copy()
    for col in ("Fluid_Sensitive", "Fat_Suppression"):
        if col not in frame.columns:
            frame[col] = 0
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0).astype(int)
    frame["Anatomical_Plane"] = frame["Anatomical_Plane"].astype(str)
    chosen, used = [], set()
    for plane in ("Sagittal", "Coronal", "Axial"):
        candidates = frame[frame["Anatomical_Plane"].eq(plane)]
        if len(candidates):
            row = candidates.sort_values(
                ["Fluid_Sensitive", "Fat_Suppression"], ascending=False
            ).iloc[0]
            chosen.append(row)
            used.add(str(row["SeriesInstanceUID"]))
    for fluid, fat in ((1, 1), (1, 0), (0, 1), (0, 0)):
        candidates = frame[
            (frame["Fluid_Sensitive"] == fluid)
            & (frame["Fat_Suppression"] == fat)
        ]
        for _, row in candidates.iterrows():
            series_uid = str(row["SeriesInstanceUID"])
            if series_uid not in used:
                chosen.append(row)
                used.add(series_uid)
                break
    for _, row in frame.iterrows():
        if len(chosen) >= MAX_SERIES:
            break
        series_uid = str(row["SeriesInstanceUID"])
        if series_uid not in used:
            chosen.append(row)
            used.add(series_uid)
    return pd.DataFrame(chosen[:MAX_SERIES]) if chosen else frame.head(0)


def ordered_files(directory):
    values = []
    for path in Path(directory).glob("*.dcm"):
        try:
            header = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
            orientation = getattr(header, "ImageOrientationPatient", None)
            position = getattr(header, "ImagePositionPatient", None)
            if orientation is not None and position is not None and len(orientation) == 6:
                normal = np.cross(
                    np.asarray(orientation[:3], np.float64),
                    np.asarray(orientation[3:6], np.float64),
                )
                key = float(np.dot(np.asarray(position[:3], np.float64), normal))
            else:
                key = float(getattr(header, "InstanceNumber", 0) or 0)
            values.append((key, str(path)))
            del header
        except Exception:
            pass
    values.sort(key=lambda item: item[0])
    return [path for _, path in values]


def read_slice(path):
    ds = pydicom.dcmread(str(path), force=True)
    image = ds.pixel_array.astype(np.float32, copy=False)
    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    if slope != 1.0 or intercept != 0.0:
        image = image * slope + intercept
    if str(getattr(ds, "PhotometricInterpretation", "")) == "MONOCHROME1":
        image = image.max() - image
    spacing_value = getattr(ds, "PixelSpacing", None)
    spacing = float(spacing_value[0]) if spacing_value is not None else None
    if spacing and spacing > 0:
        side = max(16, min(int(round(CROP_MM / spacing)), *image.shape[:2]))
        center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
        y0 = max(0, center_y - side // 2)
        x0 = max(0, center_x - side // 2)
        image = image[y0:y0 + side, x0:x0 + side]
    low, high = np.percentile(image, [1, 99])
    image = np.clip((image - low) / (high - low + 1e-6), 0, 1)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    result = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    del ds, image
    return result


for study_uid in test_ids:
    images, metadata = [], []
    frame = series_by_study.get(study_uid)
    if frame is not None:
        for _, row in choose_series(frame).iterrows():
            series_uid = str(row["SeriesInstanceUID"])
            files = ordered_files(ROOT / "test_series" / study_uid / series_uid)
            if not files:
                continue
            for quantile in QUANTILES:
                center = int(round(quantile * (len(files) - 1)))
                positions = [
                    max(0, min(len(files) - 1, center + delta))
                    for delta in (-1, 0, 1)
                ]
                try:
                    images.append(np.stack([read_slice(files[index]) for index in positions]))
                    plane = np.zeros(3, np.float32)
                    plane[PLANE_ID.get(str(row.get("Anatomical_Plane", "")), 2)] = 1.0
                    fluid = pd.to_numeric(row.get("Fluid_Sensitive", 0), errors="coerce")
                    fat = pd.to_numeric(row.get("Fat_Suppression", 0), errors="coerce")
                    metadata.append(np.array([
                        *plane,
                        0.0 if pd.isna(fluid) else float(fluid),
                        0.0 if pd.isna(fat) else float(fat),
                        quantile,
                    ], np.float16))
                except Exception:
                    continue
    image_array = (
        np.stack(images)
        if images
        else np.zeros((0, 3, IMAGE_SIZE, IMAGE_SIZE), np.uint8)
    )
    meta_array = np.stack(metadata) if metadata else np.zeros((0, 6), np.float16)
    np.savez(OUTPUT / f"{study_uid}.npz", images=image_array, meta=meta_array)
