#!/usr/bin/env python3
"""Build J01 hidden-test inference using its TorchScript visual encoder."""

from __future__ import annotations

import json
import py_compile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECODER_SOURCE = ROOT / "public_kernels/effv2_raptor_infer/rsna-knee-effv2-raptor-infer.py"
OUT_DIR = ROOT / "public_kernels/j01_biomed_jepa_infer"
OUTPUT = OUT_DIR / "rsna-knee-j01-biomed-jepa-infer.py"

HEADER = r'''#!/usr/bin/env python3
"""J01 BiomedCLIP masked-latent study inference."""
import gc, hashlib, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F

T0 = time.time()
SCHEMA = "rsna_knee_j01_biomed_jepa_v1"
FOLD_SHA = "62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357"
IMG = 336
NETWORK_SIZE = 224
CROP_MM = 140.0
SLOTS = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
         ("Coronal", 0, 6), ("Axial", -1, 8)]
SLOT_RANGES = ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))
MAXS = 44
EXTRACT_PER_SLOT = 3
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one(name):
    hits = []
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in {"train_series", "test_series"}]
        if name in files:
            hits.append(Path(root) / name)
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name}, got {hits}")
    return hits[0]


class SlotHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_embedding = nn.Embedding(len(SLOT_RANGES), 512)
        self.norm = nn.LayerNorm(512)
        self.attention = nn.Sequential(nn.Linear(512, 192), nn.Tanh(), nn.Dropout(0.20),
                                       nn.Linear(192, len(LAB)))
        self.classifier_weight = nn.Parameter(torch.empty(len(LAB), 512))
        self.classifier_bias = nn.Parameter(torch.zeros(len(LAB)))

    def forward(self, feature, valid):
        slots = torch.arange(len(SLOT_RANGES), device=feature.device).repeat_interleave(EXTRACT_PER_SLOT)
        z = self.norm(feature + self.slot_embedding(slots)[None])
        attention = self.attention(z).masked_fill(~valid.unsqueeze(-1), -1e4).softmax(1)
        pooled = torch.einsum("bkt,bkd->btd", attention, z)
        return (pooled * self.classifier_weight[None]).sum(-1) + self.classifier_bias


def load_models(device):
    visual_path, heads_path = find_one("j01_visual.ts"), find_one("j01_heads.pt")
    visual = torch.jit.load(str(visual_path), map_location=device).eval()
    payload = torch.load(heads_path, map_location="cpu", weights_only=False)
    if payload.get("schema") != SCHEMA or payload.get("status") != "COMPLETE":
        raise RuntimeError("J01 checkpoint schema/status drift")
    if payload.get("targets") != LAB or payload.get("fold_sha256") != FOLD_SHA:
        raise RuntimeError("J01 target/fold contract drift")
    if len(payload.get("fold_states", [])) != 5:
        raise RuntimeError("J01 requires exactly five heads")
    heads = []
    for state in payload["fold_states"]:
        model = SlotHead()
        model.load_state_dict(state, strict=True)
        heads.append(model.eval().to(device))
    receipt = {"visual_file": visual_path.name, "visual_sha256": sha256(visual_path),
               "heads_file": heads_path.name, "heads_sha256": sha256(heads_path),
               "gold58_auc": float(payload["gold58_auc"]),
               "per_target_auc": payload["per_target_auc"]}
    return visual, heads, receipt


def valid_centers(mask, slot):
    lo, hi = SLOT_RANGES[slot]
    return [c for c in range(lo + 1, hi - 1) if mask[c - 1] and mask[c] and mask[c + 1]]


@torch.no_grad()
def study_features(visual, volume, mask, device):
    tiles, flags = [], []
    for slot in range(len(SLOT_RANGES)):
        centers = valid_centers(mask, slot)
        present = bool(centers)
        if centers:
            take = np.linspace(0, len(centers) - 1, EXTRACT_PER_SLOT).round().astype(int)
            centers = [centers[i] for i in take]
        else:
            lo, hi = SLOT_RANGES[slot]
            centers = [(lo + hi) // 2] * EXTRACT_PER_SLOT
        for center in centers:
            tiles.append(torch.from_numpy(np.array(volume[center - 1:center + 2], copy=True)))
            flags.append(present)
    x = torch.stack(tiles).to(device).float().div_(255)
    x = F.interpolate(x, size=(NETWORK_SIZE, NETWORK_SIZE), mode="bilinear", align_corners=False)
    mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = x.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    with torch.autocast("cuda", dtype=torch.float16):
        feature = F.normalize(visual((x - mean) / std).float(), dim=-1).unsqueeze(0)
    valid = torch.tensor(flags, dtype=torch.bool, device=device).unsqueeze(0)
    return feature, valid


@torch.no_grad()
def infer_heads(heads, feature, valid):
    output = []
    for model in heads:
        with torch.autocast("cuda", dtype=torch.float16):
            output.append(torch.sigmoid(model(feature, valid).float())[0].cpu().numpy())
    return np.stack(output)


def rankpct(values):
    return pd.DataFrame(values).rank(method="average", pct=True).to_numpy(np.float64)
'''

MAIN = r'''
def find_test_root():
    for root in ("/kaggle/input/competitions/rsna-knee-abnormality-detection",
                 "/kaggle/input/rsna-knee-abnormality-detection"):
        if os.path.isfile(root + "/test.csv"):
            return root
    raise RuntimeError("competition test root missing")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("J01 inference requires GPU")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    root = find_test_root()
    test = pd.read_csv(root + "/test.csv", dtype={"StudyInstanceUID": str})
    test_ids = test.StudyInstanceUID.astype(str).tolist()
    series = pd.read_csv(root + "/test_series.csv",
                         dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    records = {uid: frame.to_dict("records") for uid, frame in series.groupby("StudyInstanceUID")}
    reader = _make_reader()
    visual, heads, checkpoint = load_models(device)
    components = np.full((5, len(test_ids), len(LAB)), np.nan, np.float32)
    for row, uid in enumerate(test_ids):
        try:
            volume, mask = build_study(uid, records, root + "/test_series", reader)
            feature, valid = study_features(visual, volume, mask, device)
            components[:, row] = infer_heads(heads, feature, valid)
            del volume, mask, feature, valid
        except Exception as error:
            raise RuntimeError(f"J01 failed on study {row} {uid}: {error}") from error
        if (row + 1) % 100 == 0 or row + 1 == len(test_ids):
            print(f"J01 {row+1}/{len(test_ids)} elapsed={time.time()-T0:.1f}s", flush=True)
    if not np.isfinite(components).all():
        raise RuntimeError("J01 component predictions incomplete")
    raw = components.mean(0)
    ranked = rankpct(raw).astype(np.float32)
    submission = pd.DataFrame(ranked, columns=LAB)
    submission.insert(0, "StudyInstanceUID", test_ids)
    sample_columns = pd.read_csv(root + "/sample_submission.csv", nrows=1).columns.tolist()
    submission = submission[sample_columns]
    output = Path(os.environ.get("RSNA_J01_OUTPUT", "/kaggle/working/submission.csv"))
    temporary = output.with_suffix(output.suffix + ".tmp")
    submission.to_csv(temporary, index=False)
    os.replace(temporary, output)
    prediction_path = Path("/kaggle/working/j01_predictions.npz")
    np.savez(prediction_path, study_uids=np.asarray(test_ids), components=components,
             raw_probabilities=raw, ranked_predictions=ranked, labels=np.asarray(LAB))
    receipt = {"schema": "j01_inference_v1", "status": "COMPLETE",
               "studies": len(test_ids), "fallback_studies": 0,
               "checkpoint": checkpoint, "submission_sha256": sha256(output),
               "predictions_sha256": sha256(prediction_path),
               "elapsed_seconds": time.time() - T0}
    receipt_path = Path("/kaggle/working/j01_inference_receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
'''


def main():
    source = DECODER_SOURCE.read_text()
    start = source.index("def _make_reader():")
    stop = source.index("\n\n# ============================================================================\n# Test-root discovery", start)
    decoder = source[start:stop].rstrip()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(HEADER + "\n\n" + decoder + "\n\n" + MAIN)
    py_compile.compile(str(OUTPUT), doraise=True)
    metadata = {
        "id": "adityakumarrai0401/rsna-knee-j01-biomed-jepa-infer",
        "title": "RSNA Knee J01 Biomed JEPA Infer", "code_file": OUTPUT.name,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "keywords": ["gpu"], "dataset_sources": [],
        "kernel_sources": ["adityakumarrai0401/rsna-knee-j01-biomed-jepa-train"],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"source": str(OUTPUT), "bytes": OUTPUT.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
