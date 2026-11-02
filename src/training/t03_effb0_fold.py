#!/usr/bin/env python3
"""T03 one-fold EfficientNet-B0 slot-aware 2.5D MIL trainer."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import re
import resource
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

# __CONTRACT__

RUN_FOLD = __RUN_FOLD__
EXPECTED_FOLD_SHA256 = "62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357"
PRETRAINED_SHA256 = "827b6e33c2a10291671e2426b902c2fe87a77e88ab6e3ae769cda9c85b9be02e"
SCHEMA = "rsna_knee_t03_effb0_slotmil_v1"
SEED = 20260930
SLOT_RANGES = ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))
TRAIN_PER_SLOT = 3
EVAL_PER_SLOT = 5
IMAGE_SIZE = 224
EPOCHS = 10
SWA_EPOCHS = (8, 9, 10)
T0 = time.time()


def log(message):
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bounded_files(name, max_depth=5):
    hits = []
    base = Path("/kaggle/input")
    for root, dirs, files in os.walk(base):
        depth = len(Path(root).relative_to(base).parts)
        dirs[:] = [d for d in dirs if d not in {
            "train_series", "test_series", "train_images", "test_images"
        } and depth < max_depth]
        if name in files:
            hits.append(Path(root) / name)
    return hits


def find_unique(name, expected_sha=None):
    hits = bounded_files(name)
    if expected_sha is not None:
        hits = [path for path in hits if sha256_file(path) == expected_sha]
    unique = sorted({path.resolve() for path in hits}, key=str)
    if len(unique) != 1:
        raise RuntimeError(f"expected exactly one {name}, got {unique}")
    return unique[0]


def find_root():
    hits = []
    for path in bounded_files("train.csv"):
        if (path.parent / "train_series").is_dir() and (path.parent / "test.csv").is_file():
            hits.append(path.parent)
    if len(hits) != 1:
        raise RuntimeError(f"competition root discovery failed: {hits}")
    return hits[0]


class TwoPartCorpus:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = len(a)
        self.shape = (len(a) + len(b), *a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        return self.a[item] if item < self.n_a else self.b[item - self.n_a]


def open_corpus():
    av, bv = find_unique("all_vols.npy"), find_unique("extra_vols.npy")
    am, bm = find_unique("all_masks.npy"), find_unique("extra_masks.npy")
    ai, bi = find_unique("all_ids.npy"), find_unique("extra_ids.npy")
    a = np.load(av, mmap_mode="r")
    b = np.load(bv, mmap_mode="r")
    masks = np.concatenate([np.load(am), np.load(bm)]).astype(np.uint8)
    ids = np.concatenate([np.load(ai, allow_pickle=True), np.load(bi, allow_pickle=True)]).astype(str)
    corpus = TwoPartCorpus(a, b)
    if corpus.shape != (4407, 44, 336, 336) or masks.shape != (4407, 44):
        raise RuntimeError(f"Raptor corpus drift: {corpus.shape}, {masks.shape}")
    if len(set(ids)) != 4407:
        raise RuntimeError("Raptor IDs are not unique")
    return corpus, masks, ids


class SlotWindows(Dataset):
    def __init__(self, uids, id_to_row, targets, weights, train_mode, per_slot):
        self.uids = list(uids)
        self.id_to_row = id_to_row
        self.targets = targets
        self.weights = weights
        self.train_mode = bool(train_mode)
        self.per_slot = int(per_slot)
        self.corpus = self.masks = None

    def __len__(self):
        return len(self.uids)

    def _ensure(self):
        if self.corpus is None:
            self.corpus, self.masks, ids = open_corpus()
            if any(self.id_to_row[uid] != i for i, uid in enumerate(ids)):
                raise RuntimeError("worker corpus identity drift")

    def _centers(self, mask, lo, hi):
        valid = [c for c in range(lo + 1, hi - 1)
                 if mask[c - 1] and mask[c] and mask[c + 1]]
        if not valid:
            return [], False
        if self.train_mode:
            return random.choices(valid, k=self.per_slot), True
        take = np.linspace(0, len(valid) - 1, self.per_slot).round().astype(int)
        return [valid[i] for i in take], True

    def __getitem__(self, item):
        self._ensure()
        uid = self.uids[item]
        row = self.id_to_row[uid]
        volume = self.corpus[row]
        mask = self.masks[row]
        tiles, slots, valid_windows = [], [], []
        for slot, (lo, hi) in enumerate(SLOT_RANGES):
            centers, present = self._centers(mask, lo, hi)
            if not centers:
                centers = [(lo + hi) // 2] * self.per_slot
            for center in centers:
                tri = np.array(volume[center - 1:center + 2], copy=True)
                tiles.append(torch.from_numpy(tri))
                slots.append(slot)
                valid_windows.append(present)
        x = torch.stack(tiles)
        return (x, torch.tensor(slots, dtype=torch.long),
                torch.tensor(valid_windows, dtype=torch.bool),
                torch.from_numpy(self.targets[row]), torch.from_numpy(self.weights[row]), uid)


def collate(batch):
    return (torch.stack([row[0] for row in batch]),
            torch.stack([row[1] for row in batch]),
            torch.stack([row[2] for row in batch]),
            torch.stack([row[3] for row in batch]),
            torch.stack([row[4] for row in batch]),
            [row[5] for row in batch])


def augment_and_normalise(x, train_mode):
    b, k = x.shape[:2]
    x = x.reshape(b * k, 3, 336, 336).float().div_(255)
    x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                      align_corners=False)
    if train_mode:
        angle = (torch.rand(b, device=x.device) - 0.5) * (16 * math.pi / 180)
        scale = 0.94 + torch.rand(b, device=x.device) * 0.12
        tx = (torch.rand(b, device=x.device) - 0.5) * 0.10
        ty = (torch.rand(b, device=x.device) - 0.5) * 0.10
        cosine, sine = torch.cos(angle) / scale, torch.sin(angle) / scale
        theta = torch.stack([torch.stack([cosine, -sine, tx], 1),
                             torch.stack([sine, cosine, ty], 1)], 1)
        theta = theta.repeat_interleave(k, 0)
        grid = F.affine_grid(theta, x.shape, align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros",
                          align_corners=False)
        gain = (0.90 + torch.rand(b, 1, 1, 1, device=x.device) * 0.20).repeat_interleave(k, 0)
        gamma = (0.90 + torch.rand(b, 1, 1, 1, device=x.device) * 0.20).repeat_interleave(k, 0)
        x = (x.clamp(1e-4, 1).pow(gamma) * gain
             + torch.randn_like(x) * 0.012).clamp(0, 1)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).reshape(b, k, 3, IMAGE_SIZE, IMAGE_SIZE)


def pretrained_path():
    hits = []
    for path in bounded_files("tf_efficientnet_b0_aa-827b6e33.pth", max_depth=7):
        if sha256_file(path) == PRETRAINED_SHA256:
            hits.append(path)
    if len(hits) != 1:
        raise RuntimeError(f"official EfficientNet-B0 checkpoint missing or duplicated: {hits}")
    return hits[0]


def build_backbone():
    import timm
    model = timm.create_model("tf_efficientnet_b0", pretrained=False, num_classes=0,
                              global_pool="avg", drop_rate=0.25, drop_path_rate=0.10)
    state = torch.load(pretrained_path(), map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    own = model.state_dict()
    usable = {key.removeprefix("module."): value for key, value in state.items()
              if key.removeprefix("module.") in own
              and own[key.removeprefix("module.")].shape == value.shape}
    missing, unexpected = model.load_state_dict(usable, strict=False)
    if unexpected or len(usable) < 300 or any(not key.startswith("classifier") for key in missing):
        raise RuntimeError(f"pretrained load drift: usable={len(usable)} missing={missing} unexpected={unexpected}")
    log(f"pretrained EfficientNet-B0 tensors={len(usable)} dim={model.num_features}")
    return model


class SlotMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = build_backbone()
        dim = int(self.backbone.num_features)
        self.slot_embedding = nn.Embedding(len(SLOT_RANGES), dim)
        self.norm = nn.LayerNorm(dim)
        self.attention = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(0.25),
                                       nn.Linear(256, len(TARGETS)))
        self.classifier_weight = nn.Parameter(torch.empty(len(TARGETS), dim))
        self.classifier_bias = nn.Parameter(torch.zeros(len(TARGETS)))
        nn.init.trunc_normal_(self.classifier_weight, std=0.02)
        nn.init.trunc_normal_(self.slot_embedding.weight, std=0.02)

    def forward(self, images, slots, valid):
        b, k = images.shape[:2]
        features = self.backbone(images.flatten(0, 1)).reshape(b, k, -1)
        features = self.norm(features + self.slot_embedding(slots))
        attention = self.attention(features).masked_fill(~valid.unsqueeze(-1), -1e4)
        attention = torch.softmax(attention, dim=1)
        pooled = torch.einsum("bkt,bkd->btd", attention, features)
        return (pooled * self.classifier_weight.unsqueeze(0)).sum(-1) + self.classifier_bias


def macro_auc(truth, pred):
    scores = []
    for column in range(truth.shape[1]):
        y = truth[:, column]
        if len(np.unique(y)) == 2:
            scores.append(roc_auc_score(y, pred[:, column]))
    return float(np.mean(scores)) if scores else float("nan")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    predictions, truths, ids = [], [], []
    for x, slots, valid, y, _, batch_ids in loader:
        x = augment_and_normalise(x.to(device, non_blocking=True), False)
        slots = slots.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(x, slots, valid)
        predictions.append(torch.sigmoid(logits.float()).cpu().numpy())
        truths.append(y.numpy())
        ids.extend(batch_ids)
    return np.concatenate(predictions), np.concatenate(truths), ids


def cpu_state(model):
    core = model.module if isinstance(model, nn.DataParallel) else model
    return {key: value.detach().cpu().clone() for key, value in core.state_dict().items()}


def state_average(states):
    averaged = {}
    for key in states[0]:
        if states[0][key].is_floating_point():
            averaged[key] = sum(state[key].double() for state in states).div(len(states)).to(states[0][key].dtype)
        else:
            averaged[key] = states[0][key].clone()
    return averaged


def write_checkpoint(path, state, fold, epoch, fold_audit):
    payload = {
        "schema": SCHEMA, "fold": fold, "epoch": epoch, "targets": TARGETS,
        "architecture": "tf_efficientnet_b0_slotmil", "state": state,
        "input": {"shape": [44, 336, 336], "network_size": IMAGE_SIZE,
                  "slot_ranges": SLOT_RANGES,
                  "train_per_slot": TRAIN_PER_SLOT, "eval_per_slot": EVAL_PER_SLOT},
        "fold_audit": fold_audit, "pretrained_sha256": PRETRAINED_SHA256,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    if RUN_FOLD not in range(5):
        raise RuntimeError(f"invalid fold {RUN_FOLD}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("T03 requires exactly two T4 GPUs")
    seed_all(SEED + RUN_FOLD)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    root = find_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    corpus, masks, corpus_ids = open_corpus()
    del corpus
    train_ids = train["StudyInstanceUID"].astype(str).to_numpy()
    if set(corpus_ids) != set(train_ids):
        raise RuntimeError("Raptor corpus does not exactly cover competition train")
    id_to_row = {uid: row for row, uid in enumerate(corpus_ids)}

    label_paths = {name: find_unique(name, digest)
                   for name, digest in PUBLIC_LABEL_SHA256.items()}
    targets_by_train, weights_by_train, gold_by_train = construct_targets(train, label_paths)
    train_row = {uid: row for row, uid in enumerate(train_ids)}
    order = np.asarray([train_row[uid] for uid in corpus_ids])
    targets = targets_by_train[order]
    weights = weights_by_train[order]
    gold = gold_by_train[order]
    folds_by_train, fold_audit = make_folds(train, targets_by_train, gold_by_train, SEED)
    if fold_audit["fold_sha256"] != EXPECTED_FOLD_SHA256:
        raise RuntimeError(f"fold hash drift: {fold_audit}")
    folds = folds_by_train[order]
    train_uids = corpus_ids[folds != RUN_FOLD]
    valid_uids = corpus_ids[folds == RUN_FOLD]
    log(f"fold={RUN_FOLD} train={len(train_uids)} valid={len(valid_uids)} "
        f"Gold={int(gold[folds == RUN_FOLD].sum())} folds={fold_audit}")

    train_set = SlotWindows(train_uids, id_to_row, targets, weights, True, TRAIN_PER_SLOT)
    valid_set = SlotWindows(valid_uids, id_to_row, targets, weights, False, EVAL_PER_SLOT)
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0,
                              pin_memory=False, drop_last=True,
                              collate_fn=collate,
                              generator=torch.Generator().manual_seed(SEED + RUN_FOLD))
    valid_loader = DataLoader(valid_set, batch_size=4, shuffle=False, num_workers=0,
                              pin_memory=False, collate_fn=collate)

    base = SlotMIL().to(device)
    model = nn.DataParallel(base, device_ids=[0, 1])
    head_names = ("slot_embedding", "norm", "attention", "classifier")
    backbone_parameters, head_parameters = [], []
    for name, parameter in base.named_parameters():
        (head_parameters if name.startswith(head_names) else backbone_parameters).append(parameter)
    optimizer = torch.optim.AdamW([
        {"params": backbone_parameters, "lr": 3e-5},
        {"params": head_parameters, "lr": 6e-4},
    ], weight_decay=0.025)
    total_steps = len(train_loader) * EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[3e-5, 6e-4], total_steps=total_steps,
        pct_start=0.15, div_factor=10, final_div_factor=100,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history, final_states = [], []
    working = Path("/kaggle/working")

    for epoch in range(1, EPOCHS + 1):
        seed_all(SEED + RUN_FOLD * 100 + epoch)
        model.train()
        losses = []
        for step, (x, slots, valid, y, w, _) in enumerate(train_loader, 1):
            x = augment_and_normalise(x.to(device, non_blocking=True), True)
            slots = slots.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            w = w.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x, slots, valid)
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(base.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            del x, slots, valid, y, w, logits, loss
            if step % 50 == 0:
                rss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
                gpu_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
                log(f"fold={RUN_FOLD} epoch={epoch:02d} step={step}/{len(train_loader)} "
                    f"loss={np.mean(losses[-50:]):.4f} rss_max={rss_gib:.2f}GiB "
                    f"gpu_max={gpu_gib:.2f}GiB")
            if step % 100 == 0:
                gc.collect()

        record = {"epoch": epoch, "loss": float(np.mean(losses)),
                  "seconds": time.time() - T0}
        state = cpu_state(model)
        write_checkpoint(working / f"t03_fold{RUN_FOLD}_latest.pt", state,
                         RUN_FOLD, epoch, fold_audit)
        if epoch in SWA_EPOCHS:
            final_states.append(state)
        if epoch in (5, 8, 10):
            pred, truth, ids = predict(model, valid_loader, device)
            hard_truth = (truth >= 0.5).astype(np.int8)
            valid_rows = np.asarray([id_to_row[uid] for uid in ids])
            gold_local = gold[valid_rows] > 0
            record["derived_auc"] = macro_auc(hard_truth, pred)
            record["gold_auc"] = macro_auc(hard_truth[gold_local], pred[gold_local])
        history.append(record)
        log(f"fold={RUN_FOLD} epoch={epoch:02d} {record}")
        del state
        gc.collect()
        torch.cuda.empty_cache()

    if len(final_states) != len(SWA_EPOCHS):
        raise RuntimeError("fixed last-three state capture failed")
    averaged = state_average(final_states)
    base.load_state_dict(averaged, strict=True)
    final_pred, final_truth, final_ids = predict(model, valid_loader, device)
    final_rows = np.asarray([id_to_row[uid] for uid in final_ids])
    final_gold = gold[final_rows] > 0
    derived_auc = macro_auc((final_truth >= 0.5).astype(np.int8), final_pred)
    gold_auc = macro_auc((final_truth[final_gold] >= 0.5).astype(np.int8), final_pred[final_gold])

    checkpoint = working / f"t03_effb0_fold{RUN_FOLD}.pt"
    write_checkpoint(checkpoint, averaged, RUN_FOLD, list(SWA_EPOCHS), fold_audit)
    oof = pd.DataFrame({"StudyInstanceUID": final_ids, "Fold": RUN_FOLD})
    for column, target in enumerate(TARGETS):
        oof[target] = final_pred[:, column]
    oof_path = working / f"t03_oof_fold{RUN_FOLD}.csv"
    oof.to_csv(oof_path, index=False)
    latest = working / f"t03_fold{RUN_FOLD}_latest.pt"
    latest.unlink(missing_ok=True)
    manifest = {
        "schema": SCHEMA, "status": "COMPLETE", "fold": RUN_FOLD,
        "seed": SEED + RUN_FOLD, "fold_audit": fold_audit,
        "training": {"epochs": EPOCHS, "swa_epochs": SWA_EPOCHS,
                     "gold_weight": 2.0, "batch_studies": 4,
                     "optimizer": "AdamW OneCycle", "backbone_lr": 3e-5,
                     "head_lr": 6e-4, "weight_decay": 0.025},
        "validation": {"derived_auc": derived_auc, "gold_auc": gold_auc,
                       "valid_n": len(final_ids), "gold_n": int(final_gold.sum())},
        "history": history,
        "artifacts": {
            checkpoint.name: {"bytes": checkpoint.stat().st_size,
                              "sha256": sha256_file(checkpoint)},
            oof_path.name: {"bytes": oof_path.stat().st_size,
                           "sha256": sha256_file(oof_path)},
        },
        "provenance": {"pretrained_sha256": PRETRAINED_SHA256,
                       "public_label_sha256": PUBLIC_LABEL_SHA256},
        "elapsed_seconds": time.time() - T0,
    }
    manifest_path = working / f"t03_manifest_fold{RUN_FOLD}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log(json.dumps({"status": "COMPLETE", "fold": RUN_FOLD,
                    "derived_auc": derived_auc, "gold_auc": gold_auc,
                    "elapsed_seconds": time.time() - T0}, sort_keys=True))


if __name__ == "__main__":
    main()
