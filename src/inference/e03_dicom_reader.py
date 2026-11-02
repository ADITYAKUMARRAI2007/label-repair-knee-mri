"""Instrumented reconstruction of the published 44-slice training corpus.

The author's corpus metadata specifies a 15--85% depth span. This module is
shared by the pixel-parity job and future E03 inference. It emits acquisition
and slice diagnostics; it never substitutes a study prediction.
"""
import glob
import numpy as np

IMG = 336
CROP_MM = 140.0
SPAN = (0.15, 0.85)
SLOTS = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
         ("Coronal", 0, 6), ("Axial", -1, 8)]
MAXS = 44


def protocol_flag(value):
    if value is None or str(value).strip().lower() in ("", "nan", "none", "<na>", "unknown"):
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
    return int(number) if np.isfinite(number) and number in (0.0, 1.0) else None


def pick_series(rows, plane, fluid, used):
    candidates = [r for r in rows if r["Anatomical_Plane"] == plane
                  and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        preferred = [r for r in candidates if protocol_flag(r.get("Fluid_Sensitive")) == fluid]
        if preferred:
            return preferred[0]
    return candidates[0] if candidates else None


def make_reader():
    import cv2
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(directory, stats):
        files = glob.glob(directory + "/*.dcm")
        records, spacing_values = [], []
        stats["native_files"] = len(files)
        stats["header_failures"] = 0
        stats["position_fallbacks"] = 0
        stats["spacing_fallbacks"] = 0
        for path in files:
            try:
                header = pydicom.dcmread(path, stop_before_pixels=True)
                iop = getattr(header, "ImageOrientationPatient", None)
                ipp = getattr(header, "ImagePositionPatient", None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    normal = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                    pos = float(np.dot(np.array(ipp, float), normal))
                else:
                    pos = float(getattr(header, "InstanceNumber", 0) or 0)
                    stats["position_fallbacks"] += 1
                if not np.isfinite(pos):
                    raise ValueError("nonfinite slice position")
                spacing = getattr(header, "PixelSpacing", None)
                if spacing is None:
                    ps = 0.5
                    stats["spacing_fallbacks"] += 1
                else:
                    ps = float(spacing[0])
                    if not np.isfinite(ps) or ps <= 0:
                        ps = 0.5
                        stats["spacing_fallbacks"] += 1
                spacing_values.append(ps)
                records.append((pos, path, ps))
            except Exception:
                stats["header_failures"] += 1
                records.append((0.0, path, 0.5))
        records.sort(key=lambda r: r[0])
        median_spacing = float(np.median(spacing_values)) if spacing_values else 0.5
        return [(path, ps) for _, path, ps in records], median_spacing

    def read_pixels(path):
        dicom = pydicom.dcmread(path)
        image = apply_modality_lut(dicom.pixel_array, dicom).astype(np.float32)
        if image.ndim != 2 or min(image.shape) < 1 or not np.isfinite(image).all():
            raise ValueError("expected a finite two-dimensional DICOM slice")
        if str(getattr(dicom, "PhotometricInterpretation", "")) == "MONOCHROME1":
            image = image.max() - image
        return image

    def crop_resize(image, spacing):
        height, width = image.shape
        side = min(int(round(CROP_MM / max(spacing, 1e-3))), height, width)
        if side < 1:
            raise ValueError("physical crop is empty")
        y0, x0 = (height - side) // 2, (width - side) // 2
        return cv2.resize(image[y0:y0 + side, x0:x0 + side], (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_pixels, crop_resize


def build_study(study_id, series_records, series_root, reader, span=SPAN):
    order_and_meta, read_pixels, crop_resize = reader
    volume = np.zeros((MAXS, IMG, IMG), dtype=np.uint8)
    used, offset = set(), 0
    receipt = {"StudyInstanceUID": study_id, "span": list(span), "slots": []}
    for slot, (plane, fluid, count) in enumerate(SLOTS):
        stats = {"slot": slot, "plane": plane, "preferred_fluid": fluid,
                 "requested_cached_slices": count, "selected_slice_attempts": 0,
                 "decoded_selected_slices": 0, "slice_failures": 0,
                 "constant_selected_slices": 0, "failure_types": {},
                 "native_files": 0, "header_failures": 0,
                 "position_fallbacks": 0, "spacing_fallbacks": 0}
        row = pick_series(series_records.get(study_id, []), plane, fluid, used)
        if row is None:
            stats["state"] = "missing_acquisition"
            receipt["slots"].append(stats)
            offset += count
            continue
        series_id = row["SeriesInstanceUID"]
        stats["SeriesInstanceUID"] = series_id
        used.add(series_id)
        files, median_spacing = order_and_meta(f"{series_root}/{study_id}/{series_id}", stats)
        if not files:
            stats["state"] = "empty_acquisition"
            receipt["slots"].append(stats)
            offset += count
            continue
        n = len(files)
        lo, hi = int(n * span[0]), max(int(n * span[0]), int(n * span[1]) - 1)
        picks = np.linspace(lo, hi, count).round().astype(int) if n > 1 else np.zeros(count, dtype=int)
        arrays, spacings = [], []
        for index in picks:
            path, ps = files[min(index, n - 1)]
            stats["selected_slice_attempts"] += 1
            try:
                image = read_pixels(path)
                arrays.append(image)
                spacings.append(ps)
                stats["decoded_selected_slices"] += 1
                stats["constant_selected_slices"] += int(image.max() == image.min())
            except Exception as error:
                arrays.append(None)
                spacings.append(median_spacing)
                stats["slice_failures"] += 1
                key = type(error).__name__
                stats["failure_types"][key] = stats["failure_types"].get(key, 0) + 1
        valid = [a for a in arrays if a is not None]
        if valid:
            low, high = np.percentile(np.concatenate([a.ravel() for a in valid]), [2.0, 98.0])
        else:
            low, high = 0.0, 1.0
        for i, (image, spacing) in enumerate(zip(arrays, spacings)):
            if image is None:
                continue
            scaled = np.clip((image - low) / (high - low + 1e-6), 0, 1)
            cropped = crop_resize(scaled, spacing if spacing > 0 else median_spacing)
            volume[offset + i] = (cropped * 255).astype(np.uint8)
        stats["nonzero_cached_slices"] = int(np.count_nonzero(volume[offset:offset + count].reshape(count, -1).sum(1)))
        stats["state"] = "decoded" if valid else "unreadable_acquisition"
        receipt["slots"].append(stats)
        offset += count
    mask = (volume.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    receipt["nonzero_cached_slices"] = int(mask.sum())
    receipt["selected_slice_attempts"] = sum(s["selected_slice_attempts"] for s in receipt["slots"])
    receipt["slice_failures"] = sum(s["slice_failures"] for s in receipt["slots"])
    receipt["header_failures"] = sum(s["header_failures"] for s in receipt["slots"])
    if not mask.any():
        raise RuntimeError(f"study {study_id} has no nonzero decoded slices")
    return volume, mask, receipt
