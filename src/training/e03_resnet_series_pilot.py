#!/usr/bin/env python3
"""E03 bounded GPU pilot. This file is assembled with pinned E02 I/O helpers.

It does not score the competition, select checkpoints on Gold58, or produce a
submission. Its only decision is whether full training is operationally viable.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import mmap
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet34

# __E02_IO__
# __WINDOW_CONTRACT__

SEED = 20261006
IMAGE_SIZE = 288
TRAIN_PER_SLOT = 4
EVAL_PER_SLOT = 8
BATCH_STUDIES = 2
PILOT_STEPS = 300
PLANNED_EPOCHS = 14
T0 = time.time()
EXPECTED = {
    "resnet34.pth": "333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef",
    "teacher_soft_labels.csv": "efe27c47782ca90411aa37eff0036f2fbdbb6f449113a0611116fb83ac95a2ae",
    "HYBRID_labels_scores.csv": "40f4d6c4669913a8aedf946ce4645ed16972d554536dcd1c7b1d1fde854dfed6",
}
TEACHER_LOGIT_WEIGHT = 0.75
JEV_LOGIT_WEIGHT = 0.25


class SeriesWindows(Dataset):
    def __init__(self, rows, targets, masks, per_slot, training):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.targets, self.masks = targets, masks
        self.per_slot, self.training = per_slot, training
        self.corpus, self.reads = None, 0

    def __len__(self):
        return len(self.rows)

    def release_pages(self):
        if self.corpus is not None:
            self.corpus.release_pages()

    def __getitem__(self, index):
        if self.corpus is None:
            self.corpus, _, _ = open_corpus()
        row = int(self.rows[index])
        centers, mask, pos = select_slot_windows(
            self.masks[row].tolist(), self.per_slot, self.training)
        x = np.zeros((5, self.per_slot, 3, 336, 336), dtype=np.uint8)
        for slot, picks in enumerate(centers):
            for i, center in enumerate(picks):
                if center is not None:
                    x[slot, i] = self.corpus[row][center - 1:center + 2]
        self.reads += 1
        if self.reads % 16 == 0:
            self.release_pages()
        return (torch.from_numpy(x), torch.tensor(mask, dtype=torch.bool),
                torch.tensor(pos, dtype=torch.float32),
                torch.from_numpy(self.targets[row]), row)


def make_views(x, size, training, per_slot):
    """Spatial and intensity transforms shared by all windows of an acquisition."""
    n, device = len(x), x.device
    groups = n // per_slot
    x = x.float().div_(255)
    if training:
        angle = (torch.rand(groups, device=device) - 0.5) * math.radians(24)
        scale = 1 + (torch.rand(groups, device=device) - 0.5) * 0.20
        tx = (torch.rand(groups, device=device) - 0.5) * 0.10
        ty = (torch.rand(groups, device=device) - 0.5) * 0.10
    else:
        angle = torch.zeros(groups, device=device)
        scale = torch.ones(groups, device=device)
        tx = ty = torch.zeros(groups, device=device)
    c, s = angle.cos() * scale, angle.sin() * scale
    theta = torch.stack([torch.stack([c, -s, tx], 1),
                         torch.stack([s, c, ty], 1)], 1).repeat_interleave(per_slot, 0)
    grid = F.affine_grid(theta, (n, 3, size, size), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    if training:
        gamma = torch.exp((torch.rand(groups, device=device) - 0.5) * 0.4)
        gain = 0.90 + torch.rand(groups, device=device) * 0.20
        x = (x.clamp_min(1e-4).pow(gamma.repeat_interleave(per_slot).view(n, 1, 1, 1))
             * gain.repeat_interleave(per_slot).view(n, 1, 1, 1)).clamp(0, 1)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (x - mean) / std


class SeriesMIL(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.encoder = resnet34(weights=None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.fc = nn.Identity()
        dim = 256
        self.project = nn.Sequential(nn.Dropout(0.2), nn.Linear(512, dim), nn.LayerNorm(dim))
        self.slot_embedding = nn.Embedding(5, dim)
        self.position_embedding = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, 512, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.attention = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(),
                                       nn.Dropout(0.2), nn.Linear(128, len(TARGETS)))
        self.classifier_weight = nn.Parameter(torch.randn(len(TARGETS), dim) * 0.02)
        self.classifier_bias = nn.Parameter(torch.zeros(len(TARGETS)))

    def forward(self, x, mask, pos, training=False):
        b, slots, k = x.shape[:3]
        if not mask.any(1).all():
            raise RuntimeError("cannot encode an entirely missing study")
        keep = mask.reshape(-1)
        windows = x.reshape(b * slots, k, *x.shape[3:])[keep].flatten(0, 1)
        views = make_views(windows, IMAGE_SIZE, training, k)
        encoded = self.encoder(views.contiguous(memory_format=torch.channels_last))
        feat = encoded.new_zeros(b * slots, k, 512)
        feat[keep] = encoded.reshape(-1, k, 512)
        feat = feat.reshape(b, slots, k, 512)
        slot_ids = torch.arange(slots, device=x.device).view(1, slots, 1)
        h = (self.project(feat) + self.slot_embedding(slot_ids)
             + self.position_embedding(pos.unsqueeze(-1).to(feat.dtype)))
        h = h.reshape(b, slots * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, slots, k).reshape(b, slots * k)
        h = self.transformer(h, src_key_padding_mask=pad)
        a = self.attention(h).float().masked_fill(pad.unsqueeze(-1), float("-inf")).softmax(1)
        pooled = torch.einsum("btc,btd->bcd", a.to(h.dtype), h)
        return (pooled * self.classifier_weight).sum(-1) + self.classifier_bias


def current_rss_gib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / (1024 ** 2)
    raise RuntimeError("cannot measure resident memory")


def atomic_save(payload, path):
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("pilot requires a GPU")
    seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    root = find_competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    corpus, masks, ids = open_corpus()
    corpus.release_pages()
    del corpus
    if set(ids) != set(train.StudyInstanceUID.astype(str)):
        raise RuntimeError("corpus/train identity mismatch")
    # Freeze the two label inputs before the inherited fusion helper opens them.
    for name in ("teacher_soft_labels.csv", "HYBRID_labels_scores.csv"):
        find_unique(name, EXPECTED[name])
    targets, labels = build_targets(train, ids)
    gold = set(train.loc[train[TARGETS].notna().all(1), "StudyInstanceUID"].astype(str))
    if len(gold) != 58:
        raise RuntimeError("Gold58 identity drift")
    rows = [i for i, uid in enumerate(ids) if uid not in gold]
    if any(ids[r] in gold for r in rows):
        raise RuntimeError("Gold leakage")
    # Pilot has no validation claims. The full trainer must additionally freeze
    # report groups and an independent weak validation split before launch.
    dataset = SeriesWindows(rows, targets, masks, TRAIN_PER_SLOT, True)
    loader = DataLoader(dataset, batch_size=BATCH_STUDIES, shuffle=True,
                        num_workers=0, pin_memory=False, drop_last=True)
    model = SeriesMIL(find_unique("resnet34.pth", EXPECTED["resnet34.pth"])).cuda()
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    backbone = list(model.encoder.parameters())
    heads = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": backbone, "lr": 1e-4},
                             {"params": heads, "lr": 5e-4}], weight_decay=0.025)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    records, times = [], []
    overflow_steps = 0
    started = time.time()
    for step, (x, mask, pos, y, row) in enumerate(loader, 1):
        before = time.time()
        model.train()
        x, mask, pos, y = x.cuda(), mask.cuda(), pos.cuda(), y.cuda()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x, mask, pos, training=True)
            loss = F.binary_cross_entropy_with_logits(logits.float(), y)
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite pilot forward")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        overflow = not bool(torch.isfinite(grad_norm))
        if overflow:
            overflow_steps += 1
            log(f"AMP overflow at step={step}; scaler will skip this update")
        scaler.step(opt)
        scaler.update()
        if overflow_steps > 10:
            raise RuntimeError("persistent pilot gradient overflow")
        with torch.no_grad():
            decay = min(0.998, (1 + step) / (10 + step))
            if not overflow:
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        torch.cuda.synchronize()
        times.append(time.time() - before)
        if step % 25 == 0:
            rss = current_rss_gib()
            gpu = torch.cuda.max_memory_allocated() / (1024 ** 3)
            item = {"step": step, "loss": float(loss), "rss_gib": rss,
                    "gpu_peak_gib": gpu, "seconds": time.time() - started}
            records.append(item)
            log(json.dumps(item, sort_keys=True))
            if rss > 20 or gpu > 13:
                raise RuntimeError("pilot exceeded its host/GPU memory gate")
        if step >= PILOT_STEPS:
            break
    dataset.release_pages()
    # Exercise evaluation at its real, larger token count and missing-slot mask.
    eval_set = SeriesWindows(rows[:BATCH_STUDIES], targets, masks, EVAL_PER_SLOT, False)
    ex, em, ep, _, _ = next(iter(DataLoader(eval_set, batch_size=BATCH_STUDIES)))
    ex, em, ep = ex.cuda(), em.cuda(), ep.cuda()
    ema.eval()
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16):
            pred = ema(ex, em, ep).float()
        # Simulate a missing acquisition, then alter only that acquisition.
        em[:, -1] = False
        with torch.autocast("cuda", dtype=torch.float16):
            masked_pred = ema(ex, em, ep).float()
            ex[:, -1] = 255 - ex[:, -1]
            perturbed_pred = ema(ex, em, ep).float()
    if not torch.isfinite(pred).all() or not torch.isfinite(masked_pred).all():
        raise RuntimeError("nonfinite evaluation output")
    mask_error = float((masked_pred - perturbed_pred).abs().max())
    if mask_error > 1e-6:
        raise RuntimeError(f"absent acquisition influenced output: {mask_error}")
    out = Path("/kaggle/working")
    checkpoint = out / "e03_pilot_state.pt"
    atomic_save({"model": model.state_dict(), "ema": ema.state_dict(),
                 "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                 "pilot_only": True, "step": step, "targets": TARGETS}, checkpoint)
    # Loading the written EMA proves the checkpoint can restore the same model.
    restored = torch.load(checkpoint, map_location="cuda", weights_only=False)
    ema.load_state_dict(restored["ema"], strict=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        replay = ema(ex, em, ep).float()
    replay_error = float((replay - perturbed_pred).abs().max())
    if replay_error > 1e-6:
        raise RuntimeError(f"checkpoint replay failed: {replay_error}")
    eval_set.release_pages()
    steady = float(np.median(times[25:]))
    # End-to-end rate includes loading, mmap release and optimizer work.
    throughput_seconds = (time.time() - started) / step
    epoch_estimate = throughput_seconds * len(loader)
    growth = records[-1]["rss_gib"] - records[1]["rss_gib"]
    accepted = (growth < 2.0 and epoch_estimate < 1800
                and current_rss_gib() < 20
                and torch.cuda.max_memory_allocated() / (1024 ** 3) < 13)
    result = {"schema": "e03_resource_pilot_v1", "status": "COMPLETE",
              "resource_gate": "PASS" if accepted else "REJECT",
              "steps": step, "amp_skipped_updates": overflow_steps,
              "studies_per_batch": BATCH_STUDIES,
              "train_windows_per_slot": TRAIN_PER_SLOT, "eval_windows_per_slot": EVAL_PER_SLOT,
              "image_size": IMAGE_SIZE, "single_gpu": True,
              "gold_rows_used_for_training": 0, "gold_validation_performed": False,
              "training_rows": len(rows), "labels": labels, "memory_samples": records,
              "rss_growth_after_step_50_gib": growth, "steady_compute_seconds_per_step": steady,
              "end_to_end_seconds_per_step": throughput_seconds,
              "estimated_training_epoch_seconds": epoch_estimate,
              "planned_epochs": PLANNED_EPOCHS,
              "estimated_training_seconds_excluding_validation": epoch_estimate * PLANNED_EPOCHS,
              "mask_invariance_max_abs_error": mask_error,
              "checkpoint_replay_max_abs_error": replay_error,
              "checkpoint_sha256": sha256_file(checkpoint),
              "elapsed_seconds": time.time() - T0,
              "next_action": "freeze grouped validation and full-training contract" if accepted else "redesign resources"}
    (out / "e03_pilot_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    log(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
