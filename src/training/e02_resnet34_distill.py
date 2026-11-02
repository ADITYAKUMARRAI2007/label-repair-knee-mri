#!/usr/bin/env python3
"""E02: independent ResNet34 student for RSNA knee abnormality detection.

The 58 expert-labelled studies are a strict validation set.  Training uses a
fixed logit blend of the public teacher soft labels and JEV report labels.  The
image model is intentionally unlike the DINO/CoAt ensemble used by B01: a
ResNet34 encoder and simple mean/top-k pooling over 2.5D windows.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import mmap
import os
import random
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet34


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
SCHEMA = "rsna_knee_e02_resnet34_distill_v1"
SEED = 20260930
IMAGE_SIZE = 288
TRAIN_WINDOWS = 16
EVAL_WINDOWS = 32
BATCH_STUDIES = 4
EPOCHS = 18
SWA_EPOCHS = (15, 16, 17, 18)
TEACHER_LOGIT_WEIGHT = 0.75
JEV_LOGIT_WEIGHT = 0.25
EXPECTED = {
    "resnet34.pth": "333f7ec4c6338da2cbed37f1fc0445f9624f1355633fa1d7eab79a91084c6cef",
    "teacher_soft_labels.csv": None,
    "HYBRID_labels_scores.csv": None,
}
T0 = time.time()


def log(message):
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def bounded_files(name, max_depth=6):
    hits = []
    base = Path("/kaggle/input")
    for root, dirs, files in os.walk(base):
        depth = len(Path(root).relative_to(base).parts)
        dirs[:] = [d for d in dirs if d not in {
            "train_series", "test_series", "train_images", "test_images"
        } and depth < max_depth]
        if name in files:
            hits.append(Path(root) / name)
    return sorted({p.resolve() for p in hits}, key=str)


def find_unique(name, expected=None):
    hits = bounded_files(name)
    if expected:
        hits = [p for p in hits if sha256_file(p) == expected]
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name}, got {hits}")
    return hits[0]


def find_competition_root():
    roots = []
    for p in bounded_files("train.csv"):
        if (p.parent / "train_series").is_dir() and (p.parent / "test.csv").is_file():
            roots.append(p.parent)
    if len(roots) != 1:
        raise RuntimeError(f"competition root discovery failed: {roots}")
    return roots[0]


class TwoPartCorpus:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = len(a)
        self.shape = (len(a) + len(b), *a.shape[1:])

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]

    def release_pages(self):
        for array in (self.a, self.b):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and hasattr(mapping, "madvise"):
                mapping.madvise(mmap.MADV_DONTNEED)


def open_corpus():
    a = np.load(find_unique("all_vols.npy"), mmap_mode="r")
    b = np.load(find_unique("extra_vols.npy"), mmap_mode="r")
    masks = np.concatenate([
        np.load(find_unique("all_masks.npy")),
        np.load(find_unique("extra_masks.npy")),
    ]).astype(np.uint8)
    ids = np.concatenate([
        np.load(find_unique("all_ids.npy"), allow_pickle=True),
        np.load(find_unique("extra_ids.npy"), allow_pickle=True),
    ]).astype(str)
    corpus = TwoPartCorpus(a, b)
    if corpus.shape != (4407, 44, 336, 336) or masks.shape != (4407, 44):
        raise RuntimeError(f"Raptor corpus drift: {corpus.shape}, {masks.shape}")
    if len(set(ids)) != 4407:
        raise RuntimeError("Raptor corpus IDs are not unique")
    return corpus, masks, ids


def read_labels(path, uids, require_finite=True):
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if frame["StudyInstanceUID"].duplicated().any():
        raise RuntimeError(f"duplicate labels in {path}")
    values = frame.set_index("StudyInstanceUID").reindex(uids)[TARGETS].to_numpy(np.float32)
    if values.shape != (4407, 12) or (require_finite and not np.isfinite(values).all()):
        raise RuntimeError(f"invalid label matrix from {path}: {values.shape}")
    return values


def build_targets(train, corpus_ids):
    train_uids = train["StudyInstanceUID"].astype(str).tolist()
    teacher_path = find_unique("teacher_soft_labels.csv")
    jev_path = find_unique("HYBRID_labels_scores.csv")
    teacher = read_labels(teacher_path, train_uids)
    jev = read_labels(jev_path, train_uids, require_finite=False)
    jev_missing = ~np.isfinite(jev)
    # JEV currently has a tiny documented coverage defect.  A missing JEV cell
    # contributes zero new evidence, so use the already present teacher value.
    jev = np.where(jev_missing, teacher, jev)
    eps = 1e-4
    teacher_logits = np.log(np.clip(teacher, eps, 1 - eps) / np.clip(1 - teacher, eps, 1 - eps))
    jev_logits = np.log(np.clip(jev, eps, 1 - eps) / np.clip(1 - jev, eps, 1 - eps))
    fused = 1 / (1 + np.exp(-(
        TEACHER_LOGIT_WEIGHT * teacher_logits + JEV_LOGIT_WEIGHT * jev_logits
    )))
    fused = np.clip(fused, 0.01, 0.99).astype(np.float32)
    row = {uid: i for i, uid in enumerate(train_uids)}
    order = np.asarray([row[uid] for uid in corpus_ids])
    audit = {
        "teacher": str(teacher_path), "teacher_sha256": sha256_file(teacher_path),
        "jev": str(jev_path), "jev_sha256": sha256_file(jev_path),
        "jev_imputed_cells_from_teacher": int(jev_missing.sum()),
        "teacher_logit_weight": TEACHER_LOGIT_WEIGHT,
        "jev_logit_weight": JEV_LOGIT_WEIGHT,
        "mean_prevalence": dict(zip(TARGETS, fused.mean(0).astype(float))),
        "mean_teacher_jev_abs_diff": float(np.mean(np.abs(teacher - jev))),
    }
    return fused[order], audit


class StudyWindows(Dataset):
    def __init__(self, rows, targets, masks, count, train_mode):
        self.rows = np.asarray(rows, np.int64)
        self.targets = targets
        self.masks = masks
        self.count = int(count)
        self.train_mode = bool(train_mode)
        self.corpus = None
        self.reads = 0

    def __len__(self):
        return len(self.rows)

    def _ensure(self):
        if self.corpus is None:
            self.corpus, _, _ = open_corpus()

    def release_pages(self):
        if self.corpus is not None:
            self.corpus.release_pages()

    def __getitem__(self, item):
        self._ensure()
        row = int(self.rows[item])
        valid = np.flatnonzero(self.masks[row] > 0)
        valid_set = set(valid.tolist())
        centers = np.asarray([c for c in valid if c - 1 in valid_set and c + 1 in valid_set], np.int64)
        if not len(centers):
            centers = np.arange(1, 43, dtype=np.int64)
        if self.train_mode:
            # Stratified random positions expose the whole study each epoch.
            edges = np.linspace(0, len(centers), self.count + 1).astype(int)
            selected = [centers[random.randrange(edges[i], max(edges[i + 1], edges[i] + 1)) % len(centers)]
                        for i in range(self.count)]
        else:
            selected = centers[np.linspace(0, len(centers) - 1, self.count).round().astype(int)]
        tiles = [np.array(self.corpus[row][c - 1:c + 2], copy=True) for c in selected]
        self.reads += 1
        if self.reads % 32 == 0:
            self.release_pages()
        return torch.from_numpy(np.stack(tiles)), torch.from_numpy(self.targets[row]), row


def augment(x, train_mode):
    b, k = x.shape[:2]
    x = x.flatten(0, 1).float().div_(255)
    x = F.interpolate(x, (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    if train_mode:
        # Apply one intensity transform per study; adjacent slices stay registered.
        gain = (0.88 + 0.24 * torch.rand(b, 1, 1, 1, device=x.device)).repeat_interleave(k, 0)
        gamma = (0.88 + 0.24 * torch.rand(b, 1, 1, 1, device=x.device)).repeat_interleave(k, 0)
        x = (x.clamp(1e-4, 1).pow(gamma) * gain).clamp(0, 1)
        if torch.rand((), device=x.device) < 0.5:
            x = torch.flip(x, dims=(-1,))
        x = (x + 0.008 * torch.randn_like(x)).clamp(0, 1)
    mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return ((x - mean) / std).reshape(b, k, 3, IMAGE_SIZE, IMAGE_SIZE)


class SimpleMIL(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.encoder = resnet34(weights=None)
        # pytorch/resnet34 is the historical torchvision checkpoint in legacy
        # tar serialization.  Its exact SHA-256 is pinned before this load.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, unexpected = self.encoder.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("fc.") for k in missing):
            raise RuntimeError(f"ResNet34 load mismatch: missing={missing} unexpected={unexpected}")
        self.encoder.fc = nn.Identity()
        self.norm = nn.LayerNorm(512)
        self.slice_head = nn.Linear(512, len(TARGETS))
        self.study_head = nn.Linear(1024, len(TARGETS))

    def forward(self, images):
        b, k = images.shape[:2]
        feat = self.norm(self.encoder(images.flatten(0, 1)).reshape(b, k, 512))
        slice_logits = self.slice_head(feat)
        top = max(1, k // 4)
        local = slice_logits.topk(top, dim=1).values.mean(1)
        global_logits = self.study_head(torch.cat([feat.mean(1), feat.amax(1)], dim=-1))
        return 0.5 * local + 0.5 * global_logits


def macro_auc(y, p):
    out = {}
    for j, name in enumerate(TARGETS):
        out[name] = float(roc_auc_score(y[:, j], p[:, j]))
    return float(np.mean(list(out.values()))), out


@torch.no_grad()
def evaluate(model, loader, device, truth):
    model.eval()
    pred, rows = [], []
    for x, _, row in loader:
        x = augment(x.to(device, non_blocking=True), False)
        with torch.autocast("cuda", dtype=torch.float16):
            pred.append(torch.sigmoid(model(x).float()).cpu().numpy())
        rows.extend(row.numpy().tolist())
    pred = np.concatenate(pred)
    y = truth[np.asarray(rows)]
    score, per_target = macro_auc(y, pred)
    loader.dataset.release_pages()
    return score, per_target, pred, np.asarray(rows)


def state_cpu(model):
    core = model.module if isinstance(model, nn.DataParallel) else model
    return {k: v.detach().cpu().clone() for k, v in core.state_dict().items()}


def average_states(states):
    out = {}
    for key in states[0]:
        if states[0][key].is_floating_point():
            out[key] = sum(s[key].double() for s in states).div(len(states)).to(states[0][key].dtype)
        else:
            out[key] = states[0][key].clone()
    return out


def memory_status():
    rss_max = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    current = None
    p = Path("/sys/fs/cgroup/memory.current")
    if p.exists():
        current = int(p.read_text()) / (1024 ** 3)
    return rss_max, current


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("E02 requires exactly two T4 GPUs")
    seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    root = find_competition_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    corpus, masks, ids = open_corpus()
    corpus.release_pages()
    del corpus
    if set(ids) != set(train["StudyInstanceUID"].astype(str)):
        raise RuntimeError("corpus/train identity mismatch")
    targets, label_audit = build_targets(train, ids)
    row_by_uid = {uid: i for i, uid in enumerate(ids)}
    train_by_uid = train.set_index("StudyInstanceUID")
    gold_uids = train.loc[train[TARGETS].notna().all(axis=1), "StudyInstanceUID"].astype(str).tolist()
    if len(gold_uids) != 58:
        raise RuntimeError(f"Gold count drift: {len(gold_uids)}")
    gold_rows = np.asarray([row_by_uid[u] for u in gold_uids])
    train_rows = np.asarray([i for i in range(len(ids)) if i not in set(gold_rows.tolist())])
    gold_truth = np.zeros((len(ids), len(TARGETS)), np.float32)
    for uid, row in zip(gold_uids, gold_rows):
        gold_truth[row] = train_by_uid.loc[uid, TARGETS].to_numpy(np.float32)
    log(f"strict split train={len(train_rows)} Gold={len(gold_rows)}; label_audit={label_audit}")

    train_set = StudyWindows(train_rows, targets, masks, TRAIN_WINDOWS, True)
    gold_set = StudyWindows(gold_rows, targets, masks, EVAL_WINDOWS, False)
    train_loader = DataLoader(train_set, batch_size=BATCH_STUDIES, shuffle=True,
                              num_workers=0, pin_memory=False, drop_last=True)
    gold_loader = DataLoader(gold_set, batch_size=2, shuffle=False,
                             num_workers=0, pin_memory=False)

    checkpoint = find_unique("resnet34.pth", EXPECTED["resnet34.pth"])
    base = SimpleMIL(checkpoint).to(device)
    # Preserve generic low-level filters; train the higher-resolution semantics.
    for name, parameter in base.encoder.named_parameters():
        if name.startswith(("conv1.", "bn1.", "layer1.")):
            parameter.requires_grad = False
    model = nn.DataParallel(base, device_ids=[0, 1])
    backbone = [p for n, p in base.named_parameters()
                if n.startswith("encoder.") and p.requires_grad]
    head = [p for n, p in base.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": 7e-5},
        {"params": head, "lr": 7e-4},
    ], weight_decay=0.025)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[7e-5, 7e-4], total_steps=len(train_loader) * EPOCHS,
        pct_start=0.12, div_factor=10, final_div_factor=100,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history, final_states = [], []
    working = Path("/kaggle/working")

    for epoch in range(1, EPOCHS + 1):
        seed_all(SEED + epoch)
        model.train()
        losses = []
        for step, (x, y, _) in enumerate(train_loader, 1):
            x = augment(x.to(device, non_blocking=True), True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(base.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
            if step % 100 == 0:
                rss_max, current = memory_status()
                log(f"epoch={epoch:02d} step={step}/{len(train_loader)} "
                    f"loss={np.mean(losses[-100:]):.4f} rss_max={rss_max:.2f}GiB "
                    f"memory.current={current if current is not None else -1:.2f}GiB "
                    f"gpu_max={torch.cuda.max_memory_allocated()/(1024**3):.2f}GiB")
        train_set.release_pages()
        score, per_target, pred, rows = evaluate(model, gold_loader, device, gold_truth)
        record = {"epoch": epoch, "loss": float(np.mean(losses)),
                  "gold_auc": score, "per_target_auc": per_target,
                  "seconds": time.time() - T0}
        history.append(record)
        log(f"epoch={epoch:02d} loss={record['loss']:.4f} Gold={score:.6f}")
        if epoch in SWA_EPOCHS:
            final_states.append(state_cpu(model))
        del pred
        gc.collect()
        torch.cuda.empty_cache()

    if len(final_states) != len(SWA_EPOCHS):
        raise RuntimeError("fixed SWA state capture failed")
    averaged = average_states(final_states)
    base.load_state_dict(averaged, strict=True)
    final_score, final_per_target, final_pred, final_rows = evaluate(
        model, gold_loader, device, gold_truth
    )
    checkpoint_out = working / "e02_resnet34_distill_swa.pt"
    payload = {
        "schema": SCHEMA, "architecture": "resnet34_simple_mil",
        "state": averaged, "targets": TARGETS, "seed": SEED,
        "input": {"network_size": IMAGE_SIZE, "train_windows": TRAIN_WINDOWS,
                  "eval_windows": EVAL_WINDOWS, "source_shape": [44, 336, 336]},
        "training": {"epochs": EPOCHS, "swa_epochs": SWA_EPOCHS,
                     "teacher_logit_weight": TEACHER_LOGIT_WEIGHT,
                     "jev_logit_weight": JEV_LOGIT_WEIGHT,
                     "gold_rows_used_for_training": 0},
        "pretrained_sha256": EXPECTED["resnet34.pth"],
    }
    torch.save(payload, checkpoint_out)
    gold_out = pd.DataFrame({"StudyInstanceUID": [ids[r] for r in final_rows]})
    for j, target in enumerate(TARGETS):
        gold_out[target] = final_pred[:, j]
    gold_path = working / "e02_gold_predictions.csv"
    gold_out.to_csv(gold_path, index=False)
    manifest = {
        "schema": SCHEMA, "status": "COMPLETE", "seed": SEED,
        "strict_gold_holdout": True, "train_n": len(train_rows), "gold_n": len(gold_rows),
        "validation": {"gold_auc": final_score, "per_target_auc": final_per_target,
                       "best_epoch_gold_auc_diagnostic": max(r["gold_auc"] for r in history)},
        "history": history, "label_audit": label_audit,
        "artifacts": {
            checkpoint_out.name: {"bytes": checkpoint_out.stat().st_size,
                                  "sha256": sha256_file(checkpoint_out)},
            gold_path.name: {"bytes": gold_path.stat().st_size,
                             "sha256": sha256_file(gold_path)},
        },
        "elapsed_seconds": time.time() - T0,
    }
    (working / "e02_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log(json.dumps({"status": "COMPLETE", "gold_auc": final_score,
                    "elapsed_seconds": time.time() - T0}, sort_keys=True))


if __name__ == "__main__":
    main()
