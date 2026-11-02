#!/usr/bin/env python3
"""Build the staged five-fold T03 hidden-test inference kernel."""

from __future__ import annotations

import json
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECODER_SOURCE = ROOT / "public_kernels/effv2_raptor_infer/rsna-knee-effv2-raptor-infer.py"
OUTPUT_DIR = ROOT / "public_kernels/t03_effb0_infer"
OUTPUT = OUTPUT_DIR / "rsna-knee-t03-effb0-infer.py"


HEADER = r'''#!/usr/bin/env python3
"""T03 five-fold EfficientNet-B0 slot-aware MIL hidden-test inference."""
import gc, glob, hashlib, json, os, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import timm

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
T0 = time.time()
SCHEMA = "rsna_knee_t03_effb0_slotmil_v1"
FOLD_SHA = "62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357"
PRETRAINED_SHA = "827b6e33c2a10291671e2426b902c2fe87a77e88ab6e3ae769cda9c85b9be02e"
IMG = 336
NETWORK_SIZE = 224
CROP_MM = 140.0
SLOTS = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
         ("Coronal", 0, 6), ("Axial", -1, 8)]
SLOT_RANGES = ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))
MAXS = 44
EVAL_PER_SLOT = 5
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class SlotMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b0", pretrained=False,
                                          num_classes=0, global_pool="avg",
                                          drop_rate=0.25, drop_path_rate=0.10)
        dim = int(self.backbone.num_features)
        self.slot_embedding = nn.Embedding(5, dim)
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(0.25),
                                       nn.Linear(256, len(LAB)))
        self.classifier_weight = nn.Parameter(torch.empty(len(LAB), dim))
        self.classifier_bias = nn.Parameter(torch.zeros(len(LAB)))

    def forward(self, images, slots, valid):
        b, k = images.shape[:2]
        features = self.backbone(images.flatten(0, 1)).reshape(b, k, -1)
        features = self.norm(features + self.slot_embedding(slots))
        attention = self.attention(features).masked_fill(~valid.unsqueeze(-1), -1e4)
        attention = torch.softmax(attention, dim=1)
        pooled = torch.einsum("bkt,bkd->btd", attention, features)
        return (pooled * self.classifier_weight.unsqueeze(0)).sum(-1) + self.classifier_bias


def validate_checkpoint(checkpoint, fold, path):
    required = {"schema", "fold", "epoch", "targets", "architecture", "state",
                "input", "fold_audit", "pretrained_sha256"}
    if required - set(checkpoint):
        raise RuntimeError(f"checkpoint missing {sorted(required - set(checkpoint))}: {path}")
    if checkpoint["schema"] != SCHEMA or int(checkpoint["fold"]) != fold:
        raise RuntimeError(f"checkpoint identity drift: {path}")
    if checkpoint["epoch"] != [8, 9, 10] or checkpoint["targets"] != LAB:
        raise RuntimeError(f"checkpoint selection/target drift: {path}")
    if checkpoint["architecture"] != "tf_efficientnet_b0_slotmil":
        raise RuntimeError(f"checkpoint architecture drift: {path}")
    if checkpoint["fold_audit"]["fold_sha256"] != FOLD_SHA:
        raise RuntimeError(f"checkpoint fold drift: {path}")
    if checkpoint["pretrained_sha256"] != PRETRAINED_SHA:
        raise RuntimeError(f"checkpoint initialization drift: {path}")
    if len(checkpoint["state"]) < 350:
        raise RuntimeError(f"checkpoint state too small: {path}")


def find_checkpoints():
    found = {}
    for fold in range(5):
        name = f"t03_effb0_fold{fold}.pt"
        hits = []
        for root, dirs, files in os.walk("/kaggle/input"):
            dirs[:] = [d for d in dirs if d not in {"train_series", "test_series"}]
            if name in files:
                hits.append(Path(root) / name)
        if len(hits) != 1:
            raise RuntimeError(f"expected exactly one {name}, got {hits}")
        found[fold] = hits[0]
    return found


def load_models():
    if not torch.cuda.is_available():
        raise RuntimeError("T03 inference requires GPU")
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    paths = find_checkpoints()
    groups = [[] for _ in devices]
    receipts = []
    for fold, path in paths.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        validate_checkpoint(checkpoint, fold, path)
        model = SlotMIL()
        model.load_state_dict(checkpoint["state"], strict=True)
        device_index = fold % len(devices)
        model.eval().to(devices[device_index])
        groups[device_index].append((fold, model))
        receipts.append({"fold": fold, "file": path.name, "sha256": sha256(path),
                         "device": str(devices[device_index])})
        del checkpoint
    return devices, groups, receipts


def eval_windows(volume, mask):
    tiles, slots, valid = [], [], []
    for slot, (lo, hi) in enumerate(SLOT_RANGES):
        centers = [c for c in range(lo + 1, hi - 1)
                   if mask[c - 1] and mask[c] and mask[c + 1]]
        present = bool(centers)
        if centers:
            take = np.linspace(0, len(centers) - 1, EVAL_PER_SLOT).round().astype(int)
            centers = [centers[i] for i in take]
        else:
            centers = [(lo + hi) // 2] * EVAL_PER_SLOT
        for center in centers:
            tiles.append(torch.from_numpy(np.array(volume[center - 1:center + 2], copy=True)))
            slots.append(slot)
            valid.append(present)
    x = torch.stack(tiles).float().div_(255)
    x = F.interpolate(x, size=(NETWORK_SIZE, NETWORK_SIZE), mode="bilinear",
                      align_corners=False)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).unsqueeze(0), torch.tensor(slots).unsqueeze(0), \
           torch.tensor(valid, dtype=torch.bool).unsqueeze(0)


@torch.no_grad()
def infer_group(device, models, x, slots, valid):
    with torch.cuda.device(device):
        x, slots, valid = x.to(device), slots.to(device), valid.to(device)
        output = []
        for fold, model in models:
            with torch.autocast("cuda", dtype=torch.float16):
                pred = torch.sigmoid(model(x, slots, valid).float())[0]
            output.append((fold, pred.cpu().numpy()))
        return output


def rankpct(values):
    return values.argsort(0).argsort(0).astype(np.float64) / max(len(values) - 1, 1)
'''


MAIN = r'''
def find_test_root():
    candidates = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
                  "/kaggle/input/rsna-knee-abnormality-detection"]
    for root in candidates:
        if os.path.isfile(root + "/test.csv"):
            return root
    raise RuntimeError("competition test root missing")


def main():
    root = find_test_root()
    test = pd.read_csv(root + "/test.csv", dtype={"StudyInstanceUID": str})
    test_ids = test.StudyInstanceUID.astype(str).tolist()
    series = pd.read_csv(root + "/test_series.csv",
                         dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    records = {uid: frame.to_dict("records") for uid, frame in series.groupby("StudyInstanceUID")}
    series_dir = root + "/test_series"
    reader = _make_reader()
    devices, model_groups, checkpoint_receipts = load_models()
    components = np.full((5, len(test_ids), len(LAB)), np.nan, np.float32)
    executor = ThreadPoolExecutor(max_workers=len(devices))
    for row, uid in enumerate(test_ids):
        try:
            volume, mask = build_study(uid, records, series_dir, reader)
            x, slots, valid = eval_windows(volume, mask)
            tasks = [executor.submit(infer_group, device, group, x, slots, valid)
                     for device, group in zip(devices, model_groups) if group]
            for task in tasks:
                for fold, pred in task.result():
                    components[fold, row] = pred
        except Exception as error:
            raise RuntimeError(f"T03 failed on study {row} {uid}: {error}") from error
        if (row + 1) % 100 == 0 or row + 1 == len(test_ids):
            print(f"T03 {row+1}/{len(test_ids)} elapsed={time.time()-T0:.1f}s", flush=True)
    executor.shutdown()
    if not np.isfinite(components).all():
        raise RuntimeError("T03 component predictions incomplete")
    raw = components.mean(0)
    ranked = rankpct(raw).astype(np.float32)
    submission = pd.DataFrame(ranked, columns=LAB)
    submission.insert(0, "StudyInstanceUID", test_ids)
    sample_columns = pd.read_csv(root + "/sample_submission.csv", nrows=1).columns.tolist()
    submission = submission[sample_columns]
    output = Path("/kaggle/working/submission.csv")
    submission.to_csv(output, index=False)
    prediction_path = Path("/kaggle/working/t03_predictions.npz")
    np.savez(prediction_path, study_uids=np.asarray(test_ids), components=components,
             raw_probabilities=raw, ranked_predictions=ranked, labels=np.asarray(LAB))
    receipt = {"schema": "t03_effb0_inference_v1", "status": "COMPLETE",
               "studies": len(test_ids), "fallback_studies": 0,
               "folds": checkpoint_receipts,
               "input": {"image_size": IMG, "network_size": NETWORK_SIZE,
                         "crop_mm": CROP_MM, "slots": SLOTS,
                         "eval_per_slot": EVAL_PER_SLOT},
               "submission_sha256": sha256(output),
               "predictions_sha256": sha256(prediction_path),
               "elapsed_seconds": time.time() - T0}
    Path("/kaggle/working/t03_inference_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
'''


def main():
    source = DECODER_SOURCE.read_text()
    decoder_start = source.index("def _make_reader():")
    decoder_stop = source.index("\n\n# ============================================================================\n# Test-root discovery", decoder_start)
    decoder = source[decoder_start:decoder_stop].rstrip()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(HEADER + "\n\n" + decoder + "\n\n" + MAIN)
    py_compile.compile(str(OUTPUT), doraise=True)
    metadata = {
        "id": "adityakumarrai0401/rsna-knee-t03-effb0-infer",
        "title": "RSNA Knee T03 EffB0 Infer",
        "code_file": OUTPUT.name,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["gpu"], "dataset_sources": [],
        "kernel_sources": [f"adityakumarrai0401/rsna-knee-t03-effb0-fold-{fold}"
                           for fold in range(5)],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    (OUTPUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"source": str(OUTPUT), "bytes": OUTPUT.stat().st_size,
                      "kernel_sources": metadata["kernel_sources"]}, indent=2))


if __name__ == "__main__":
    main()
