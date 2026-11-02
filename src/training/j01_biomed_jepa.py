#!/usr/bin/env python3
"""J01: BiomedCLIP masked-latent adaptation and fold-safe MRI MIL heads.

The shared visual encoder is adapted only on the 4,349 non-Gold studies.  It
learns from fused public report labels, fixed BiomedCLIP condition prototypes,
and a masked-view prediction objective against an EMA teacher.  Five small
study-level heads are then trained with grouped folds; every Gold58 prediction
is produced by a head that did not train on that study.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import math
import mmap
import os
import random
import re
import resource
import subprocess
import sys
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

SCHEMA = "rsna_knee_j01_biomed_jepa_v1"
SEED = 20261001
FOLD_SEED = 20260930
EXPECTED_FOLD_SHA256 = "62e8329a1f057a52ca62db4a0f80f2988139630e56bc5792a154a2bae3d36357"
SLOT_RANGES = ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))
IMAGE_SIZE = 224
PRETRAIN_EPOCHS = 2
PRETRAIN_SAMPLES_PER_STUDY = 5
EXTRACT_PER_SLOT = 3
HEAD_EPOCHS = 60
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
    hits, base = [], Path("/kaggle/input")
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
    if expected_sha:
        hits = [p for p in hits if sha256_file(p) == expected_sha]
    hits = sorted({p.resolve() for p in hits}, key=str)
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name}, got {hits}")
    return hits[0]


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
        self.a, self.b, self.n_a = a, b, len(a)
        self.shape = (len(a) + len(b), *a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        return self.a[item] if item < self.n_a else self.b[item - self.n_a]

    def drop_cache(self):
        """Release resident read-only mmap pages; the arrays remain valid."""
        for array in (self.a, self.b):
            mapping = getattr(array, "_mmap", None)
            if mapping is not None and hasattr(mapping, "madvise"):
                mapping.madvise(mmap.MADV_DONTNEED)


def current_rss_gib():
    with open("/proc/self/statm") as handle:
        resident_pages = int(handle.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)


def open_corpus():
    a = np.load(find_unique("all_vols.npy"), mmap_mode="r")
    b = np.load(find_unique("extra_vols.npy"), mmap_mode="r")
    masks = np.concatenate([
        np.load(find_unique("all_masks.npy")), np.load(find_unique("extra_masks.npy"))
    ]).astype(np.uint8)
    ids = np.concatenate([
        np.load(find_unique("all_ids.npy"), allow_pickle=True),
        np.load(find_unique("extra_ids.npy"), allow_pickle=True),
    ]).astype(str)
    corpus = TwoPartCorpus(a, b)
    if corpus.shape != (4407, 44, 336, 336) or masks.shape != (4407, 44):
        raise RuntimeError(f"Raptor corpus drift: {corpus.shape}, {masks.shape}")
    if len(set(ids)) != 4407:
        raise RuntimeError("Raptor IDs are not unique")
    return corpus, masks, ids


def valid_centers(mask, slot):
    lo, hi = SLOT_RANGES[slot]
    return [c for c in range(lo + 1, hi - 1) if mask[c - 1] and mask[c] and mask[c + 1]]


class AdaptSlices(Dataset):
    """One bounded 2.5D sample; opens the mmap lazily in the main process."""

    def __init__(self, rows, targets, weights, masks):
        self.rows = np.asarray(rows, np.int32)
        self.targets = targets
        self.weights = weights
        self.mask_snapshot = masks
        self.corpus = None

    def __len__(self):
        return len(self.rows) * PRETRAIN_SAMPLES_PER_STUDY

    def _ensure(self):
        if self.corpus is None:
            self.corpus, masks, _ = open_corpus()
            if not np.array_equal(masks, self.mask_snapshot):
                raise RuntimeError("corpus mask drift")

    def __getitem__(self, item):
        self._ensure()
        row = int(self.rows[item // PRETRAIN_SAMPLES_PER_STUDY])
        slot = int(item % PRETRAIN_SAMPLES_PER_STUDY)
        centers = valid_centers(self.mask_snapshot[row], slot)
        if not centers:
            lo, hi = SLOT_RANGES[slot]
            center = (lo + hi) // 2
        else:
            center = centers[random.randrange(len(centers))]
        tri = np.array(self.corpus[row][center - 1:center + 2], copy=True)
        return (torch.from_numpy(tri), torch.from_numpy(self.targets[row]),
                torch.from_numpy(self.weights[row]))

    def drop_cache(self):
        if self.corpus is not None:
            self.corpus.drop_cache()


def augment(x):
    x = x.float().div_(255)
    x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
    b = len(x)
    angle = (torch.rand(b, device=x.device) - 0.5) * (12 * math.pi / 180)
    scale = 0.96 + torch.rand(b, device=x.device) * 0.08
    tx = (torch.rand(b, device=x.device) - 0.5) * 0.06
    ty = (torch.rand(b, device=x.device) - 0.5) * 0.06
    cosine, sine = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([torch.stack([cosine, -sine, tx], 1),
                         torch.stack([sine, cosine, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    gain = 0.92 + torch.rand(b, 1, 1, 1, device=x.device) * 0.16
    gamma = 0.92 + torch.rand(b, 1, 1, 1, device=x.device) * 0.16
    return (x.clamp(1e-4, 1).pow(gamma) * gain + torch.randn_like(x) * 0.008).clamp(0, 1)


def normalize(x):
    mean = x.new_tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
    std = x.new_tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
    return (x - mean) / std


def mask_context(x):
    """Mask 5 of the 14x14 ViT patch cells, shared across the three channels."""
    out = x.clone()
    b = len(out)
    for i in range(b):
        for _ in range(5):
            ph = random.choice((2, 3, 4))
            pw = random.choice((2, 3, 4))
            y = random.randrange(0, 14 - ph + 1) * 16
            z = random.randrange(0, 14 - pw + 1) * 16
            out[i, :, y:y + ph * 16, z:z + pw * 16] = 0
    return out


def install_and_load_biomedclip(device):
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "open_clip_torch==2.23.0",
        "transformers==4.35.2",
    ])
    from open_clip import create_model_from_pretrained, get_tokenizer

    name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    model, _ = create_model_from_pretrained(name)
    tokenizer = get_tokenizer(name)
    positive = [
        "knee MRI showing an anterior cruciate ligament tear",
        "knee MRI showing a medial collateral ligament injury",
        "knee MRI showing a medial meniscus tear",
        "knee MRI showing a lateral meniscus tear",
        "knee MRI showing medial compartment osteoarthritis",
        "knee MRI showing lateral compartment osteoarthritis",
        "knee MRI showing patellofemoral osteoarthritis",
        "knee MRI showing joint effusion",
        "knee MRI showing synovitis",
        "knee MRI showing a Baker cyst",
        "knee MRI showing bone contusion",
        "knee MRI showing fracture",
    ]
    negative = [
        "knee MRI showing an intact anterior cruciate ligament",
        "knee MRI showing an intact medial collateral ligament",
        "knee MRI showing an intact medial meniscus",
        "knee MRI showing an intact lateral meniscus",
        "knee MRI without medial compartment osteoarthritis",
        "knee MRI without lateral compartment osteoarthritis",
        "knee MRI without patellofemoral osteoarthritis",
        "knee MRI without joint effusion",
        "knee MRI without synovitis",
        "knee MRI without a Baker cyst",
        "knee MRI without bone contusion",
        "knee MRI without fracture",
    ]
    model.eval().to(device)
    tokens = tokenizer(positive + negative, context_length=256).to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        text = F.normalize(model.encode_text(tokens).float(), dim=-1)
    prototypes = torch.stack([text[:len(TARGETS)], text[len(TARGETS):]], dim=1)
    visual = model.visual.cpu()
    del model, tokens, text
    torch.cuda.empty_cache()
    return visual, prototypes.cpu()


def set_trainable_tail(visual):
    for p in visual.parameters():
        p.requires_grad = False
    selected = []
    for name, p in visual.named_parameters():
        if ("trunk.blocks.10." in name or "trunk.blocks.11." in name or
                "trunk.norm" in name or "head" in name):
            p.requires_grad = True
            selected.append(name)
    count = sum(p.numel() for p in visual.parameters() if p.requires_grad)
    if count < 5_000_000 or count > 30_000_000:
        raise RuntimeError(f"unexpected trainable visual tail: {count} params, names={selected[:20]}")
    log(f"trainable visual tail parameters={count:,} tensors={len(selected)}")


def macro_auc(truth, pred):
    scores = []
    for j in range(truth.shape[1]):
        if len(np.unique(truth[:, j])) == 2:
            scores.append(roc_auc_score(truth[:, j], pred[:, j]))
    return float(np.mean(scores)) if scores else float("nan")


def per_target_auc(truth, pred):
    out = {}
    for j, target in enumerate(TARGETS):
        out[target] = float(roc_auc_score(truth[:, j], pred[:, j]))
    return out


class Adaptation(nn.Module):
    def __init__(self, visual, prototypes):
        super().__init__()
        self.visual = visual
        self.predictor = nn.Sequential(nn.Linear(512, 768), nn.GELU(), nn.Linear(768, 512))
        self.classifier = nn.Linear(512, len(TARGETS))
        direction = F.normalize(prototypes[:, 0] - prototypes[:, 1], dim=-1)
        with torch.no_grad():
            self.classifier.weight.copy_(direction * 3.0)
            self.classifier.bias.zero_()
        self.register_buffer("positive", prototypes[:, 0].clone())
        self.register_buffer("negative", prototypes[:, 1].clone())

    def forward(self, masked):
        feature = F.normalize(self.visual(masked).float(), dim=-1)
        predicted = F.normalize(self.predictor(feature), dim=-1)
        logits = self.classifier(feature)
        prompt_logits = 10.0 * torch.stack([
            (feature * self.positive[j]).sum(-1) - (feature * self.negative[j]).sum(-1)
            for j in range(len(TARGETS))
        ], dim=1)
        return feature, predicted, logits, prompt_logits


@torch.no_grad()
def ema_update(teacher, student, momentum):
    for target, source in zip(teacher.parameters(), student.parameters()):
        target.mul_(momentum).add_(source.detach(), alpha=1.0 - momentum)


def train_adaptation(visual, prototypes, rows, targets, weights, masks, device):
    set_trainable_tail(visual)
    module = Adaptation(visual, prototypes).to(device)
    teacher = copy.deepcopy(module.visual).eval().to(device)
    for p in teacher.parameters():
        p.requires_grad = False
    dataset = AdaptSlices(rows, targets, weights, masks)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0,
                        pin_memory=False, drop_last=True,
                        generator=torch.Generator().manual_seed(SEED))
    visual_params = [p for p in module.visual.parameters() if p.requires_grad]
    other_params = list(module.predictor.parameters()) + list(module.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": visual_params, "lr": 1.5e-5},
        {"params": other_params, "lr": 3e-4},
    ], weight_decay=0.04)
    total_steps = len(loader) * PRETRAIN_EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[1.5e-5, 3e-4], total_steps=total_steps,
        pct_start=0.08, div_factor=10, final_div_factor=100,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history, global_step = [], 0
    for epoch in range(1, PRETRAIN_EPOCHS + 1):
        seed_all(SEED + epoch)
        module.train()
        losses = []
        for step, (image, y, w) in enumerate(loader, 1):
            image = augment(image.to(device))
            clean = normalize(image)
            masked = normalize(mask_context(image))
            y, w = y.to(device), w.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                target_feature = F.normalize(teacher(clean).float(), dim=-1)
            with torch.autocast("cuda", dtype=torch.float16):
                _, predicted, logits, prompt_logits = module(masked)
                label_loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
                prompt_loss = (F.binary_cross_entropy_with_logits(prompt_logits, y, reduction="none") * w).mean()
                latent_loss = (2.0 - 2.0 * (predicted * target_feature).sum(-1)).mean()
                loss = label_loss + 0.15 * prompt_loss + 0.35 * latent_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            momentum = 0.996 + 0.004 * global_step / total_steps
            ema_update(teacher, module.visual, momentum)
            losses.append((float(loss.detach()), float(label_loss.detach()),
                           float(prompt_loss.detach()), float(latent_loss.detach())))
            del image, clean, masked, y, w, target_feature, predicted, logits, prompt_logits, loss
            if step % 100 == 0:
                dataset.drop_cache()
                recent = np.mean(losses[-100:], axis=0)
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
                gpu = torch.cuda.max_memory_allocated() / (1024 ** 3)
                log(f"adapt epoch={epoch} step={step}/{len(loader)} loss={recent.tolist()} "
                    f"rss={current_rss_gib():.2f}GiB rss_max={rss:.2f}GiB gpu_max={gpu:.2f}GiB")
        mean = np.mean(losses, axis=0)
        history.append({"epoch": epoch, "loss": float(mean[0]),
                        "label": float(mean[1]), "prompt": float(mean[2]),
                        "latent": float(mean[3])})
        log(f"adapt epoch={epoch} complete {history[-1]}")
        gc.collect()
        torch.cuda.empty_cache()
    module.visual.eval()
    del teacher, loader, dataset, optimizer, scheduler, scaler
    return module.visual, history


@torch.no_grad()
def extract_features(visual, corpus, masks, device):
    features = np.zeros((len(corpus), len(SLOT_RANGES) * EXTRACT_PER_SLOT, 512), np.float16)
    valid = np.zeros((len(corpus), len(SLOT_RANGES) * EXTRACT_PER_SLOT), np.uint8)
    visual.eval()
    for row in range(len(corpus)):
        tiles, flags = [], []
        volume = corpus[row]
        for slot in range(len(SLOT_RANGES)):
            centers = valid_centers(masks[row], slot)
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
        x = F.interpolate(x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16):
            z = F.normalize(visual(normalize(x)).float(), dim=-1)
        features[row] = z.cpu().numpy().astype(np.float16)
        valid[row] = np.asarray(flags, np.uint8)
        del x, z, volume
        if (row + 1) % 200 == 0 or row + 1 == len(corpus):
            corpus.drop_cache()
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
            log(f"feature extraction {row+1}/{len(corpus)} rss={current_rss_gib():.2f}GiB "
                f"rss_max={rss:.2f}GiB")
    return features, valid


class FeatureDataset(Dataset):
    def __init__(self, rows, features, valid, targets, weights):
        self.rows = np.asarray(rows, np.int32)
        self.features, self.valid = features, valid
        self.targets, self.weights = targets, weights

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        row = int(self.rows[item])
        return (torch.from_numpy(self.features[row].astype(np.float32)),
                torch.from_numpy(self.valid[row].astype(bool)),
                torch.from_numpy(self.targets[row]), torch.from_numpy(self.weights[row]), row)


class SlotHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot_embedding = nn.Embedding(len(SLOT_RANGES), 512)
        self.norm = nn.LayerNorm(512)
        self.attention = nn.Sequential(nn.Linear(512, 192), nn.Tanh(), nn.Dropout(0.20),
                                       nn.Linear(192, len(TARGETS)))
        self.classifier_weight = nn.Parameter(torch.empty(len(TARGETS), 512))
        self.classifier_bias = nn.Parameter(torch.zeros(len(TARGETS)))
        nn.init.trunc_normal_(self.slot_embedding.weight, std=0.02)
        nn.init.trunc_normal_(self.classifier_weight, std=0.02)

    def forward(self, feature, valid):
        slots = torch.arange(len(SLOT_RANGES), device=feature.device).repeat_interleave(EXTRACT_PER_SLOT)
        z = self.norm(feature + self.slot_embedding(slots)[None])
        attention = self.attention(z).masked_fill(~valid.unsqueeze(-1), -1e4).softmax(1)
        pooled = torch.einsum("bkt,bkd->btd", attention, z)
        return (pooled * self.classifier_weight[None]).sum(-1) + self.classifier_bias


@torch.no_grad()
def predict_head(model, loader, device):
    model.eval()
    pred, truth, rows = [], [], []
    for feature, valid, y, _, row in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(feature.to(device), valid.to(device))
        pred.append(torch.sigmoid(logits.float()).cpu().numpy())
        truth.append(y.numpy())
        rows.extend(row.numpy().tolist())
    return np.concatenate(pred), np.concatenate(truth), np.asarray(rows, np.int32)


def train_heads(features, valid, targets, weights, gold, folds, device):
    states, all_gold_pred, all_gold_truth = [], [], []
    fold_metrics = []
    for fold in range(5):
        seed_all(SEED + 100 + fold)
        train_rows = np.flatnonzero(folds != fold)
        valid_rows = np.flatnonzero(folds == fold)
        train_loader = DataLoader(
            FeatureDataset(train_rows, features, valid, targets, weights),
            batch_size=128, shuffle=True, num_workers=0, drop_last=False,
            generator=torch.Generator().manual_seed(SEED + fold),
        )
        valid_loader = DataLoader(
            FeatureDataset(valid_rows, features, valid, targets, weights),
            batch_size=256, shuffle=False, num_workers=0,
        )
        model = SlotHead().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=0.025)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=HEAD_EPOCHS, eta_min=2e-5)
        best_state, best_proxy = None, -np.inf
        for epoch in range(1, HEAD_EPOCHS + 1):
            model.train()
            for feature, mask, y, w, _ in train_loader:
                feature, mask = feature.to(device), mask.to(device)
                y, w = y.to(device), w.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(feature, mask)
                    loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
            scheduler.step()
            if epoch >= 35 and epoch % 5 == 0:
                pred, truth, rows = predict_head(model, valid_loader, device)
                nongold_local = gold[rows] == 0
                proxy = macro_auc((truth[nongold_local] >= 0.5).astype(np.uint8),
                                  pred[nongold_local])
                if proxy > best_proxy:
                    best_proxy = proxy
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is None:
            raise RuntimeError(f"fold {fold} head selection failed")
        model.load_state_dict(best_state)
        pred, truth, rows = predict_head(model, valid_loader, device)
        gold_local = gold[rows] > 0
        if int(gold_local.sum()) not in (11, 12):
            raise RuntimeError(f"fold {fold} Gold OOF count drift: {int(gold_local.sum())}")
        all_gold_pred.append(pred[gold_local])
        all_gold_truth.append(truth[gold_local])
        metrics = {"fold": fold, "derived_auc": macro_auc((truth >= 0.5).astype(np.uint8), pred),
                   "gold_auc": macro_auc((truth[gold_local] >= 0.5).astype(np.uint8), pred[gold_local]),
                   "selected_proxy_auc": best_proxy, "gold_n": int(gold_local.sum())}
        fold_metrics.append(metrics)
        states.append(best_state)
        log(f"head complete {metrics}")
        del model, optimizer, scheduler, train_loader, valid_loader
        gc.collect()
        torch.cuda.empty_cache()
    gold_pred = np.concatenate(all_gold_pred)
    gold_truth = (np.concatenate(all_gold_truth) >= 0.5).astype(np.uint8)
    return states, fold_metrics, gold_truth, gold_pred


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("J01 requires a GPU")
    if torch.cuda.device_count() != 1:
        log(f"J01 will use only cuda:0 of {torch.cuda.device_count()} GPUs")
    seed_all(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device("cuda:0")
    root = find_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    corpus, masks, corpus_ids = open_corpus()
    train_ids = train.StudyInstanceUID.astype(str).to_numpy()
    if set(corpus_ids) != set(train_ids):
        raise RuntimeError("Raptor corpus does not exactly cover competition train")
    paths = {name: find_unique(name, digest) for name, digest in PUBLIC_LABEL_SHA256.items()}
    targets_train, weights_train, gold_train = construct_targets(train, paths)
    folds_train, fold_audit = make_folds(train, targets_train, gold_train, FOLD_SEED)
    if fold_audit["fold_sha256"] != EXPECTED_FOLD_SHA256:
        raise RuntimeError(f"fold hash drift: {fold_audit}")
    train_row = {uid: i for i, uid in enumerate(train_ids)}
    order = np.asarray([train_row[uid] for uid in corpus_ids])
    targets, weights = targets_train[order], weights_train[order]
    gold, folds = gold_train[order], folds_train[order]
    nongold_rows = np.flatnonzero(gold == 0)
    if len(nongold_rows) != 4349:
        raise RuntimeError(f"non-Gold count drift: {len(nongold_rows)}")
    log(f"contract ready train={len(train)} nonGold={len(nongold_rows)} folds={fold_audit}")

    visual, prototypes = install_and_load_biomedclip(device)
    visual, adaptation_history = train_adaptation(
        visual, prototypes, nongold_rows, targets, weights, masks, device
    )
    features, valid = extract_features(visual, corpus, masks, device)
    states, fold_metrics, gold_truth, gold_pred = train_heads(
        features, valid, targets, weights, gold, folds, device
    )
    target_auc = per_target_auc(gold_truth, gold_pred)
    gold_auc = float(np.mean(list(target_auc.values())))

    working = Path("/kaggle/working")
    visual.eval()
    example = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)
    with torch.no_grad():
        traced = torch.jit.trace(visual, example, check_trace=False)
    visual_path = working / "j01_visual.ts"
    torch.jit.save(traced, str(visual_path))
    heads_path = working / "j01_heads.pt"
    payload = {
        "schema": SCHEMA, "status": "COMPLETE", "targets": TARGETS,
        "fold_sha256": EXPECTED_FOLD_SHA256, "fold_states": states,
        "gold58_auc": gold_auc, "per_target_auc": target_auc,
        "fold_metrics": fold_metrics, "input": {"image_size": IMAGE_SIZE,
            "slot_ranges": SLOT_RANGES, "extract_per_slot": EXTRACT_PER_SLOT,
            "normalization": "BiomedCLIP"},
    }
    torch.save(payload, heads_path)
    oof_path = working / "j01_gold58_oof.csv"
    oof = pd.DataFrame(gold_pred, columns=TARGETS)
    for j, target in enumerate(TARGETS):
        oof[f"truth_{target}"] = gold_truth[:, j]
    oof.to_csv(oof_path, index=False)
    manifest = {
        "schema": SCHEMA, "status": "COMPLETE", "seed": SEED,
        "gold_excluded_from_adaptation": True, "non_gold_adaptation_studies": 4349,
        "fold_audit": fold_audit, "adaptation_history": adaptation_history,
        "fold_metrics": fold_metrics, "gold58_auc": gold_auc,
        "per_target_auc": target_auc,
        "artifacts": {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
                      for p in (visual_path, heads_path, oof_path)},
        "elapsed_seconds": time.time() - T0,
    }
    (working / "j01_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    log(json.dumps({"status": "COMPLETE", "gold58_auc": gold_auc,
                    "elapsed_seconds": time.time() - T0}, sort_keys=True))


if __name__ == "__main__":
    main()
