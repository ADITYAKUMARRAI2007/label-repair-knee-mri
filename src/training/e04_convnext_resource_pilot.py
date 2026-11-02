#!/usr/bin/env python3
"""Resource-only ConvNeXt pilot on the existing published cache.

The wide cache has identical tensor shapes/dtypes. This pilot neither measures
accuracy nor provides a checkpoint for full training. Full training must use
the owned wide cache and a fresh generic pretrained model per expert fold.
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
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

SEED = 20261007
IMAGE_SIZE = 288
TRAIN_PER_SLOT = 4
EVAL_PER_SLOT = 8
BATCH_STUDIES = 2
PILOT_STEPS = 300
PLANNED_EPOCHS = 14
T0 = time.time()
PRETRAINED_SHA256 = "2457488bec2ffcb2061b9e3fdd63360114549615e3e35ef6dc5c5efe549cbbc9"

# __IO__
# __WINDOW_CONTRACT__
# __DATASET_AND_TRANSFORMS__
# __INPUT_CONTRACT__
# __MODEL__


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("pilot requires a GPU")
    seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    root = find_competition_root()
    corpus, masks, ids = open_corpus()
    corpus.release_pages()
    del corpus
    targets, weights, gold, folds, groups, split = reconstruct_frozen_inputs(root / "train.csv", ids)
    training, _, _ = fold_row_sets(ids, gold, folds, groups, 0)
    rows = training[~gold[training]]
    dataset = SeriesWindows(rows, targets, masks, TRAIN_PER_SLOT, True)
    loader = DataLoader(dataset, batch_size=BATCH_STUDIES, shuffle=True,
                        num_workers=0, pin_memory=False, drop_last=True)
    model = SeriesMIL(find_unique("model.safetensors", PRETRAINED_SHA256)).cuda()
    ema = copy.deepcopy(model).eval()
    ema.requires_grad_(False)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": 3e-5},
        {"params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
         "lr": 3e-4}], weight_decay=0.025)
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    records, times, overflows = [], [], 0
    started = time.time()
    for step, (x, mask, pos, y, row) in enumerate(loader, 1):
        before = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x.cuda(), mask.cuda(), pos.cuda(), training=True)
            confidence = torch.from_numpy(weights[row.numpy()]).cuda()
            loss = (F.binary_cross_entropy_with_logits(logits.float(), y.cuda(), reduction="none")
                    * confidence).mean()
        if not torch.isfinite(loss) or not torch.isfinite(logits).all():
            raise RuntimeError("nonfinite resource-pilot forward")
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        overflow = not bool(torch.isfinite(norm))
        overflows += int(overflow)
        if overflows > 10:
            raise RuntimeError("persistent pilot gradient overflow")
        scaler.step(opt)
        scaler.update()
        if not overflow:
            with torch.no_grad():
                decay = min(.998, (1 + step) / (10 + step))
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - decay)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
        torch.cuda.synchronize()
        times.append(time.time() - before)
        if step % 25 == 0:
            item = {"step": step, "loss": float(loss.detach()), "rss_gib": current_rss_gib(),
                    "gpu_peak_gib": torch.cuda.max_memory_allocated() / 2**30,
                    "seconds": time.time() - started}
            records.append(item)
            log(json.dumps(item))
            if item["rss_gib"] > 20 or item["gpu_peak_gib"] > 13:
                raise RuntimeError("resource-pilot memory gate exceeded")
        if step == PILOT_STEPS:
            break
    if step != PILOT_STEPS:
        raise RuntimeError("resource pilot did not finish all scheduled steps")
    dataset.release_pages()
    eval_set = SeriesWindows(rows[:BATCH_STUDIES], targets, masks, EVAL_PER_SLOT, False)
    ex, em, ep, _, _ = next(iter(DataLoader(eval_set, batch_size=BATCH_STUDIES)))
    ex, em, ep = ex.cuda(), em.cuda(), ep.cuda()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        pred = ema(ex, em, ep).float()
        em[:, -1] = False
        masked = ema(ex, em, ep).float()
        ex[:, -1] = 255 - ex[:, -1]
        perturbed = ema(ex, em, ep).float()
    if not torch.isfinite(pred).all() or not torch.isfinite(masked).all() or not torch.isfinite(perturbed).all():
        raise RuntimeError("nonfinite evaluation forward")
    mask_error = float((masked - perturbed).abs().max())
    if mask_error > 1e-6:
        raise RuntimeError("missing acquisition influenced the prediction")
    out = Path("/kaggle/working")
    checkpoint = out / "e04_pilot_state.pt"
    atomic_save({"model": model.state_dict(), "ema": ema.state_dict(),
                 "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
                 "pilot_only": True, "step": step}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ema.load_state_dict(saved["ema"], strict=True)
    del saved
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        replay = ema(ex, em, ep).float()
    replay_error = float((replay - perturbed).abs().max())
    if not torch.isfinite(replay).all() or replay_error > 1e-6:
        raise RuntimeError("checkpoint replay failed")
    eval_set.release_pages()
    # Compute plus sampled I/O is a conservative estimate, not a full epoch measurement.
    epoch_seconds = float(np.mean(times)) * math.ceil(len(training) / BATCH_STUDIES)
    full_five_hours = (epoch_seconds * PLANNED_EPOCHS * 5) / 3600
    growth = records[-1]["rss_gib"] - records[1]["rss_gib"]
    accepted = growth < 2 and full_five_hours < 17 and current_rss_gib() < 20
    result = {"schema": "e04_convnext_resource_pilot_v1", "status": "COMPLETE",
              "resource_gate": "PASS" if accepted else "REQUIRES_RESOURCE_REDESIGN",
              "accuracy_measured": False, "competition_submission_enabled": False,
              "pilot_weights_reusable_for_training": False,
              "pilot_cache_span": [0.15, 0.85], "full_training_cache_span": [0.06, 0.94],
              "strict_pretrained_sha256": PRETRAINED_SHA256, "timm_version": timm.__version__,
              "steps": step, "amp_skipped_updates": overflows, "memory_samples": records,
              "rss_growth_after_step50_gib": growth, "gradient_checkpointing": True,
              "image_size": IMAGE_SIZE, "batch_studies": BATCH_STUDIES,
              "train_per_slot": TRAIN_PER_SLOT, "eval_per_slot": EVAL_PER_SLOT,
              "estimated_epoch_seconds": epoch_seconds, "planned_epochs": PLANNED_EPOCHS,
              "estimated_five_fold_training_hours_excluding_validation": full_five_hours,
              "gold_updates": 0, "heldout_fold": 0, "fold_audit": split,
              "mask_invariance_max_abs_error": mask_error,
              "checkpoint_replay_max_abs_error": replay_error,
              "artifacts": {checkpoint.name: {"bytes": checkpoint.stat().st_size,
                                               "sha256": sha256_file(checkpoint)}},
              "elapsed_seconds": time.time() - T0}
    (out / "e04_pilot_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    log(json.dumps(result))


if __name__ == "__main__":
    main()
