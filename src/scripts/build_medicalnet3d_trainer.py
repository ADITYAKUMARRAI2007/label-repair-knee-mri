"""Build the Kaggle MedicalNet ResNet-10 volumetric diversity trainer.

The generated program uses the public 44-slice Raptor corpus, preserves its
five sequence slots, and encodes each slot as a small 3D volume.  The backbone
is Tencent MedicalNet ResNet-10 pretrained on 23 medical datasets.  All 58
expert-labelled studies are excluded from training and used only as a coarse
checkpoint gate.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "public_kernels/medicalnet3d_train"
OUT = OUT_DIR / "rsna-knee-medicalnet3d-train.py"


PROGRAM = r'''"""Train a MedicalNet ResNet-10 3D knee-MRI diversity arm.

Data contract
-------------
The two public Raptor corpus datasets concatenate to 4,407 studies with a
44 x 336 x 336 uint8 stack and a 44-element validity mask.  The stack contains
five fixed sequence slots of lengths 12, 10, 8, 6, and 8.  Each slot is kept
separate, resampled to 16 slices at 112 x 112, and encoded by a pretrained
MedicalNet ResNet-10.  Target-specific attention pools the five embeddings.

The weak targets use the strongest audited calibrated family as a 60% anchor
and the median of five other public families as a 40% robustness component.
The 58 expert-labelled studies never enter training.  Gold58 is too small for
fine tuning blend weights, so its AUC is used only as a coarse checkpoint and
rejection gate.  This script makes no public-leaderboard claim.

MedicalNet code and weights originate from Tencent/MedicalNet and are licensed
under MIT.  The mounted checkpoint is resnet_10_23dataset.pth with shortcut B.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
       "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
       "Contusion", "Fracture"]
SLOTS = ((0, 12), (12, 22), (22, 30), (30, 36), (36, 44))
DEPTH = 16
RES = 112
SEED = 20260929
OUT_DIR = Path("/kaggle/working")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def find_file(*names, root="/kaggle/input", contains=None):
    hits = []
    wanted = set(names)
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in
                   ("train_series", "test_series", "train_images", "test_images")]
        if contains and contains.lower() not in base.lower():
            continue
        for name in files:
            if name in wanted:
                hits.append(os.path.join(base, name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one of {names} containing {contains!r}, got {hits}")
    return hits[0]


class TwoPartVolumes:
    def __init__(self, a, b):
        self.a, self.b = a, b
        self.n_a = len(a)
        if tuple(a.shape[1:]) != tuple(b.shape[1:]):
            raise RuntimeError(f"corpus part shape mismatch: {a.shape} vs {b.shape}")
        self.shape = (len(a) + len(b),) + tuple(a.shape[1:])

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, row):
        return self.a[row] if row < self.n_a else self.b[row - self.n_a]


def open_corpus():
    av = find_file("all_vols.npy")
    bv = find_file("extra_vols.npy")
    am = find_file("all_masks.npy")
    bm = find_file("extra_masks.npy")
    ai = find_file("all_ids.npy")
    bi = find_file("extra_ids.npy")
    vols = TwoPartVolumes(np.load(av, mmap_mode="r"), np.load(bv, mmap_mode="r"))
    masks = np.concatenate([np.load(am), np.load(bm)], axis=0)
    ids = np.concatenate([np.load(ai, allow_pickle=True).astype(str),
                          np.load(bi, allow_pickle=True).astype(str)])
    if vols.shape != (4407, 44, 336, 336):
        raise RuntimeError(f"unexpected corpus volume shape {vols.shape}")
    if masks.shape != (4407, 44) or ids.shape != (4407,):
        raise RuntimeError(f"unexpected corpus contract masks={masks.shape} ids={ids.shape}")
    if len(set(ids.tolist())) != len(ids):
        raise RuntimeError("corpus IDs are not unique")
    return vols, masks, ids


def target_name(col):
    raw = str(col).strip()
    if raw in LAB:
        return raw
    aliases = {
        "acl": "ACL", "mcl": "MCL",
        "medial meniscus": "Medial Meniscus", "medial_meniscus": "Medial Meniscus",
        "lateral meniscus": "Lateral Meniscus", "lateral_meniscus": "Lateral Meniscus",
        "medial oa": "Medial OA", "medial_oa": "Medial OA",
        "lateral oa": "Lateral OA", "lateral_oa": "Lateral OA",
        "pf oa": "PF OA", "pf_oa": "PF OA", "patellofemoral oa": "PF OA",
        "effusion": "Effusion", "synovitis": "Synovitis",
        "baker's": "Baker's", "bakers": "Baker's", "baker cyst": "Baker's",
        "contusion": "Contusion", "fracture": "Fracture",
    }
    return aliases.get(raw.lower().replace("-", " ").replace("__", "_"))


def dataset_family(path):
    parts = Path(path).parts
    try:
        i = parts.index("datasets")
        return "/".join(parts[i:i + 3])
    except ValueError:
        return str(Path(path).parent)


def source_priority(path):
    name = Path(path).name.lower()
    priorities = (
        ("train_targets_soft", 120), ("v4_blend", 110),
        ("report_labels_v5", 105), ("labels_llm_soft", 100),
        ("labels_llm_gpt56sol", 100), ("report_labels_v2", 95),
        ("llm_labels_v2", 90), ("llm_labels_full", 80),
    )
    return next((score for token, score in priorities if token in name), 50)


def read_label_csv(path):
    try:
        frame = pd.read_csv(path)
    except Exception:
        return None
    uid_col = next((c for c in frame.columns if str(c).strip().lower() in {
        "studyinstanceuid", "study_instance_uid", "studyuid", "study_id", "studyid"
    }), None)
    if uid_col is None:
        return None
    rename = {uid_col: "StudyInstanceUID"}
    for col in frame.columns:
        name = target_name(col)
        if name:
            rename[col] = name
    frame = frame.rename(columns=rename)
    have = [name for name in LAB if name in frame.columns]
    if len(have) < 8:
        return None
    out = frame[["StudyInstanceUID", *have]].copy()
    out["StudyInstanceUID"] = out["StudyInstanceUID"].astype(str)
    for name in have:
        out[name] = pd.to_numeric(out[name], errors="coerce")
        out.loc[~out[name].between(0, 1), name] = np.nan
    return out.drop_duplicates("StudyInstanceUID")


def build_labels(corpus_ids):
    train_path = find_file("train.csv", contains="rsna-knee-abnormality-detection")
    train = pd.read_csv(train_path, dtype={"StudyInstanceUID": str})
    if set(train.StudyInstanceUID) != set(corpus_ids):
        raise RuntimeError("competition train IDs do not exactly match corpus IDs")
    indexed = train.set_index("StudyInstanceUID")
    gold_mask = indexed[LAB].notna().all(axis=1)
    gold = indexed.loc[gold_mask, LAB].astype(np.float32)
    if len(gold) != 58:
        raise RuntimeError(f"expected 58 expert studies, got {len(gold)}")

    uid_index = pd.Index(corpus_ids, name="StudyInstanceUID")
    candidates = {}
    discovered = []
    for base, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in
                   ("train_series", "test_series", "train_images", "test_images", "models")]
        for file in files:
            low = file.lower()
            if not low.endswith(".csv") or not any(
                    token in low for token in ("label", "llm", "pseudo", "report", "target", "soft")):
                continue
            if any(token in low for token in (
                    "target_weights", "targets_weights", "target_hard", "targets_hard",
                    "gold_benchmark", "quality_audit")):
                continue
            path = os.path.join(base, file)
            if os.path.samefile(path, train_path):
                continue
            frame = read_label_csv(path)
            if frame is None or frame.StudyInstanceUID.isin(corpus_ids).mean() < 0.50:
                continue
            candidates.setdefault(dataset_family(path), []).append((path, frame))
            discovered.append(path)

    families = []
    selected = []
    for family, options in sorted(candidates.items()):
        best_priority = max(source_priority(path) for path, _ in options)
        chosen = [(path, frame) for path, frame in options
                  if source_priority(path) == best_priority]
        selected.extend(path for path, _ in chosen)
        family_frame = pd.DataFrame(index=uid_index)
        for name in LAB:
            columns = []
            for _, frame in chosen:
                if name in frame:
                    columns.append(frame.set_index("StudyInstanceUID").reindex(uid_index)[name]
                                   .to_numpy(np.float32))
            if columns:
                family_frame[name] = np.nanmedian(np.stack(columns, axis=1), axis=1)
        families.append((family, family_frame))
        print("label family", family, "selected", [Path(p).name for p, _ in chosen], flush=True)

    if len(families) != 6:
        raise RuntimeError(f"expected exactly six audited label families, got {len(families)}")
    rayan_matches = [
        (family, frame) for family, frame in families
        if "calibrated-soft-targets" in family.lower()
    ]
    if len(rayan_matches) != 1:
        raise RuntimeError(f"expected one calibrated Rayan family, got {[x[0] for x in rayan_matches]}")
    rayan_family, rayan_frame = rayan_matches[0]
    other_families = [(family, frame) for family, frame in families if family != rayan_family]
    if len(other_families) != 5:
        raise RuntimeError(f"expected five non-Rayan families, got {len(other_families)}")

    consensus = pd.DataFrame(index=uid_index)
    for name in LAB:
        rayan_values = rayan_frame[name].to_numpy(np.float32)
        columns = [frame[name].to_numpy(np.float32)
                   for _, frame in other_families if name in frame]
        if len(columns) != 5 or not np.isfinite(rayan_values).all():
            raise RuntimeError(
                f"invalid calibrated-plus-five-family inputs for {name}: {len(columns)}"
            )
        robust_other = np.nanmedian(np.stack(columns, axis=1), axis=1)
        if not np.isfinite(robust_other).all():
            raise RuntimeError(f"non-Rayan family median is incomplete for {name}")
        consensus[name] = 0.60 * rayan_values + 0.40 * robust_other
    if consensus.isna().any().any():
        raise RuntimeError("calibrated label fusion produced missing values")
    consensus.loc[:, LAB] = np.clip(consensus[LAB].to_numpy(np.float32), 1e-4, 1 - 1e-4)

    weight_path = find_file("train_targets_weights.csv", contains="calibrated-soft-targets")
    confidence = pd.read_csv(weight_path, dtype={"StudyInstanceUID": str})
    if not {"StudyInstanceUID", *LAB}.issubset(confidence.columns):
        raise RuntimeError(f"confidence file has unexpected columns: {confidence.columns.tolist()}")
    confidence = confidence.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    confidence = confidence.reindex(uid_index)[LAB].apply(pd.to_numeric, errors="coerce")
    if confidence.isna().any().any():
        raise RuntimeError("confidence weights are incomplete")
    weak_confidence = confidence.loc[~confidence.index.isin(gold.index), LAB].to_numpy(np.float32)
    gold_confidence = confidence.loc[confidence.index.isin(gold.index), LAB].to_numpy(np.float32)
    if weak_confidence.min() < 0 or weak_confidence.max() > 1:
        raise RuntimeError("weak-study confidence weights are outside [0,1]")
    # The calibrated package deliberately marks expert rows with 5.0.  They are
    # held out from training and reset to 1.0 in main() before validation.
    if gold_confidence.shape != (58, len(LAB)) or not np.all(gold_confidence == 5.0):
        raise RuntimeError("expert-study confidence marker contract changed")

    audit = {
        "schema": "medicalnet3d_label_consensus_v1",
        "families": [family for family, _ in families],
        "selected_sources": selected,
        "discovered_sources": discovered,
        "studies": len(consensus),
        "gold_studies": len(gold),
        "confidence_source": weight_path,
        "confidence_sha256": hashlib.sha256(Path(weight_path).read_bytes()).hexdigest(),
        "confidence_contract": {
            "weak_min": float(weak_confidence.min()),
            "weak_max": float(weak_confidence.max()),
            "expert_marker": 5.0,
            "expert_runtime_weight": 1.0,
        },
        "fusion": {
            "calibrated_family": rayan_family,
            "calibrated_weight": 0.60,
            "other_family_count": 5,
            "other_family_median_weight": 0.40,
            "post_fusion_temperature": 1.0,
        },
    }
    (OUT_DIR / "medicalnet3d_label_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return consensus, confidence.astype(np.float32), gold, audit


class VolumeDataset(Dataset):
    def __init__(self, ids, id_to_row, labels, confidence, vols, masks, train):
        self.ids = list(ids)
        self.id_to_row = id_to_row
        self.labels = labels
        self.confidence = confidence
        self.vols = vols
        self.masks = masks
        self.train = bool(train)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        cv2.setNumThreads(0)
        uid = self.ids[index]
        row = self.id_to_row[uid]
        source = self.vols[row]
        valid_mask = self.masks[row] > 0
        x = np.zeros((5, 1, DEPTH, RES, RES), dtype=np.uint8)
        slot_mask = np.zeros(5, dtype=np.bool_)
        for slot, (lo, hi) in enumerate(SLOTS):
            valid = np.flatnonzero(valid_mask[lo:hi]) + lo
            if len(valid) == 0:
                continue
            slot_mask[slot] = True
            small = np.stack([
                cv2.resize(source[int(slice_index)], (RES, RES), interpolation=cv2.INTER_AREA)
                for slice_index in valid
            ]).astype(np.float32)
            positions = np.linspace(0, len(valid) - 1, DEPTH)
            if self.train and len(valid) > 2:
                jitter = np.random.uniform(-0.35, 0.35, size=DEPTH)
                positions = np.clip(positions + jitter, 0, len(valid) - 1)
            lower = np.floor(positions).astype(np.int64)
            upper = np.ceil(positions).astype(np.int64)
            alpha = (positions - lower).astype(np.float32)[:, None, None]
            interpolated = small[lower] * (1 - alpha) + small[upper] * alpha
            x[slot, 0] = np.clip(interpolated.round(), 0, 255).astype(np.uint8)
        if not slot_mask.any():
            raise RuntimeError(f"study {uid} has no valid Raptor slots")
        y = self.labels.loc[uid, LAB].to_numpy(np.float32)
        weight = self.confidence.loc[uid, LAB].to_numpy(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(slot_mask), torch.from_numpy(y),
                torch.from_numpy(weight), uid)


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
        self.layer1 = self._make_layer(64, stride=1, dilation=1)
        self.layer2 = self._make_layer(128, stride=2, dilation=1)
        self.layer3 = self._make_layer(256, stride=1, dilation=2)
        self.layer4 = self._make_layer(512, stride=1, dilation=4)

    def _make_layer(self, planes, stride, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes))
        layer = BasicBlock(self.inplanes, planes, stride=stride,
                           dilation=dilation, downsample=downsample)
        self.inplanes = planes
        return nn.Sequential(layer)

    def frozen_stem(self, x):
        with torch.no_grad():
            x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x

    def forward(self, x):
        x = self.frozen_stem(x)
        x = self.layer4(x)
        return F.adaptive_avg_pool3d(x, 1).flatten(1)


class StudyClassifier(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.slot_embedding = nn.Parameter(torch.randn(5, 32) * 0.02)
        self.attention = nn.Sequential(nn.Linear(544, 128), nn.Tanh(), nn.Linear(128, 12))
        self.classifier_weight = nn.Parameter(torch.empty(12, 544))
        self.classifier_bias = nn.Parameter(torch.zeros(12))
        nn.init.xavier_uniform_(self.classifier_weight)

    @staticmethod
    def normalize(x):
        # MedicalNet pretraining standardized each nonzero medical volume.
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
        x = self.normalize(x)
        features = self.encoder(x).view(batch, slots, 512)
        slot = self.slot_embedding.unsqueeze(0).expand(batch, -1, -1)
        features = torch.cat([features, slot], dim=-1)
        scores = self.attention(features).masked_fill(~slot_mask.unsqueeze(-1), -1e4)
        weights = scores.softmax(dim=1)
        pooled = torch.einsum("bst,bsd->btd", weights, features)
        return (pooled * self.classifier_weight.unsqueeze(0)).sum(-1) + self.classifier_bias


def find_checkpoint():
    return find_file("resnet_10_23dataset.pth", contains="medicalnet-resnet")


def load_pretrained(encoder, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    cleaned = OrderedDict((key.removeprefix("module."), value) for key, value in state.items())
    expected = set(encoder.state_dict())
    usable = {key: value for key, value in cleaned.items() if key in expected}
    missing = sorted(expected - set(usable))
    unexpected = sorted(set(cleaned) - expected)
    if missing:
        raise RuntimeError(f"MedicalNet checkpoint missing encoder keys: {missing[:8]}")
    encoder.load_state_dict(usable, strict=True)
    print(f"loaded MedicalNet encoder {path}; {len(usable)} tensors; ignored={unexpected}", flush=True)
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def safe_auc(y, p):
    aucs = {}
    for index, name in enumerate(LAB):
        if len(np.unique(y[:, index])) == 2:
            aucs[name] = float(roc_auc_score(y[:, index], p[:, index]))
    if len(aucs) != len(LAB):
        raise RuntimeError(f"Gold58 does not contain both classes for: {set(LAB) - set(aucs)}")
    return float(np.mean(list(aucs.values()))), aucs


def cpu_state(model):
    module = model.module if isinstance(model, nn.DataParallel) else model
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def main():
    seed_everything()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required")
    device = torch.device("cuda")
    vols, masks, ids = open_corpus()
    labels, confidence, gold, label_audit = build_labels(ids)
    id_to_row = {uid: index for index, uid in enumerate(ids)}
    gold_ids = [uid for uid in ids if uid in gold.index]
    gold_set = set(gold_ids)
    train_ids = [uid for uid in ids if uid not in gold_set]
    if len(train_ids) != 4349 or len(gold_ids) != 58:
        raise RuntimeError(f"split mismatch train={len(train_ids)} gold={len(gold_ids)}")
    train_labels = labels.copy()
    train_labels.loc[gold_ids, LAB] = gold.loc[gold_ids, LAB]
    confidence.loc[gold_ids, LAB] = 1.0

    train_ds = VolumeDataset(train_ids, id_to_row, train_labels, confidence,
                             vols, masks, train=True)
    gold_ds = VolumeDataset(gold_ids, id_to_row, train_labels, confidence,
                            vols, masks, train=False)
    workers = min(6, max(2, (os.cpu_count() or 4) // 2))
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=workers,
                              pin_memory=True, persistent_workers=True, drop_last=True)
    gold_loader = DataLoader(gold_ds, batch_size=4, shuffle=False, num_workers=workers,
                             pin_memory=True, persistent_workers=True)

    encoder = MedicalNet10()
    checkpoint_path = find_checkpoint()
    checkpoint_sha = load_pretrained(encoder, checkpoint_path)
    for name, parameter in encoder.named_parameters():
        parameter.requires_grad = name.startswith("layer4.")
    model = StudyClassifier(encoder).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    module = model.module if isinstance(model, nn.DataParallel) else model
    layer4_params = [p for p in module.encoder.layer4.parameters() if p.requires_grad]
    head_params = [p for name, p in module.named_parameters()
                   if not name.startswith("encoder.") and p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": layer4_params, "lr": 2e-5},
        {"params": head_params, "lr": 8e-4},
    ], weight_decay=2e-4)
    epochs = 5
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[2e-5, 8e-4], total_steps=len(train_loader) * epochs,
        pct_start=0.15, div_factor=10, final_div_factor=100)
    prevalence = np.clip(labels.loc[train_ids, LAB].to_numpy(np.float32).mean(0), 0.03, 0.7)
    pos_weight = torch.tensor(np.clip((1 - prevalence) / prevalence, 1, 10),
                              device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

    @torch.no_grad()
    def evaluate():
        model.eval()
        predictions, truth, ordered_ids = [], [], []
        for x, slot_mask, y, _, uid in gold_loader:
            x = x.to(device, non_blocking=True)
            slot_mask = slot_mask.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = torch.sigmoid(model(x, slot_mask).float())
            predictions.append(pred.cpu().numpy())
            truth.append(y.numpy())
            ordered_ids.extend(uid)
        predictions = np.concatenate(predictions)
        truth = np.concatenate(truth)
        macro, aucs = safe_auc(truth, predictions)
        return macro, aucs, predictions, truth, np.asarray(ordered_ids)

    best = None
    top = []
    history = []
    started = time.time()
    for epoch in range(epochs):
        model.train()
        # Frozen layers must keep pretrained batch-norm statistics.
        module.encoder.bn1.eval()
        module.encoder.layer1.eval()
        module.encoder.layer2.eval()
        module.encoder.layer3.eval()
        total = 0.0
        for step, (x, slot_mask, y, confidence_weight, _) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            slot_mask = slot_mask.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            confidence_weight = confidence_weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(x, slot_mask)
                element_loss = criterion(logits.float(), y)
                loss = (element_loss * confidence_weight).sum() / confidence_weight.sum().clamp_min(1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            scheduler.step()
            total += float(loss.item())
            if (step + 1) % 250 == 0:
                print(f"epoch={epoch} step={step + 1}/{len(train_loader)} "
                      f"loss={total/(step+1):.4f} elapsed={time.time()-started:.0f}s", flush=True)

        macro, aucs, pred, truth, eval_ids = evaluate()
        record = {"epoch": epoch, "loss": total / len(train_loader),
                  "gold_macro_auc": macro, "gold_target_auc": aucs,
                  "elapsed_seconds": time.time() - started}
        history.append(record)
        candidate = {"state": cpu_state(model), "gold_auc": macro,
                     "gold_target_auc": aucs, "epoch": epoch, "pred": pred}
        top.append(candidate)
        top.sort(key=lambda item: -item["gold_auc"])
        top = top[:3]
        if best is None or macro > best["gold_auc"]:
            best = candidate
            payload = {
                "schema": "medicalnet3d_checkpoint_v1",
                "model": best["state"], "lab": LAB, "slots": SLOTS,
                "depth": DEPTH, "res": RES, "epoch": epoch,
                "gold_auc": macro, "gold_target_auc": aucs,
                "medicalnet_sha256": checkpoint_sha,
            }
            tmp = OUT_DIR / "medicalnet3d_best.pt.tmp"
            torch.save(payload, tmp)
            os.replace(tmp, OUT_DIR / "medicalnet3d_best.pt")
            np.savez(OUT_DIR / "medicalnet3d_gold58.npz", pred=pred, truth=truth,
                     ids=eval_ids, lab=np.asarray(LAB))
        progress = {
            "schema": "medicalnet3d_training_v1", "status": "RUNNING",
            "history": history, "best_gold_auc": best["gold_auc"],
            "best_epoch": best["epoch"], "n_train": len(train_ids),
            "n_gold": len(gold_ids), "medicalnet_sha256": checkpoint_sha,
            "label_families": label_audit["families"],
            "label_fusion": label_audit["fusion"],
            "confidence_contract": label_audit["confidence_contract"],
        }
        (OUT_DIR / "medicalnet3d_metrics.json").write_text(
            json.dumps(progress, indent=2, sort_keys=True) + "\n")
        print(f"epoch={epoch} loss={record['loss']:.4f} GOLD macro-AUC={macro:.4f} "
              f"best={best['gold_auc']:.4f} elapsed={record['elapsed_seconds']:.0f}s", flush=True)

    # Average the best three checkpoints parameter-wise.  This supplies a stable
    # alternative to the single best Gold58 epoch without fitting a blend weight.
    swa_state = {}
    for key in top[0]["state"]:
        values = [item["state"][key] for item in top]
        if values[0].is_floating_point():
            swa_state[key] = torch.stack([v.float() for v in values]).mean(0).to(values[0].dtype)
        else:
            swa_state[key] = values[0]
    module.load_state_dict(swa_state, strict=True)
    swa_auc, swa_target_auc, swa_pred, truth, eval_ids = evaluate()
    torch.save({
        "schema": "medicalnet3d_checkpoint_v1", "model": swa_state,
        "lab": LAB, "slots": SLOTS, "depth": DEPTH, "res": RES,
        "epoch": [item["epoch"] for item in top], "gold_auc": swa_auc,
        "gold_target_auc": swa_target_auc, "medicalnet_sha256": checkpoint_sha,
    }, OUT_DIR / "medicalnet3d_swa.pt")
    np.savez(OUT_DIR / "medicalnet3d_gold58_swa.npz", pred=swa_pred, truth=truth,
             ids=eval_ids, lab=np.asarray(LAB))
    final = {
        "schema": "medicalnet3d_training_v1", "status": "COMPLETE",
        "history": history, "best_gold_auc": best["gold_auc"],
        "best_epoch": best["epoch"], "swa_gold_auc": swa_auc,
        "swa_target_auc": swa_target_auc,
        "swa_epochs": [item["epoch"] for item in top],
        "n_train": len(train_ids), "n_gold": len(gold_ids),
        "medicalnet_sha256": checkpoint_sha,
        "label_families": label_audit["families"],
        "label_fusion": label_audit["fusion"],
        "confidence_contract": label_audit["confidence_contract"],
        "elapsed_seconds": time.time() - started,
    }
    (OUT_DIR / "medicalnet3d_metrics.json").write_text(
        json.dumps(final, indent=2, sort_keys=True) + "\n")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
'''


METADATA = {
    "id": "adityakumarrai0401/rsna-knee-medicalnet3d-train-v1",
    "title": "RSNA Knee MedicalNet3D Train V1",
    "code_file": OUT.name,
    "language": "python",
    "kernel_type": "script",
    "is_private": True,
    "enable_gpu": True,
    "enable_tpu": False,
    "enable_internet": False,
    "keywords": ["gpu"],
    "dataset_sources": [
        "dreaddevelopment/knee-raptor-corpus",
        "dreaddevelopment/knee-raptor-corpus-ext",
        "stevenleehans/rsna-knee-llm-report-labels",
        "lixin73/rsna-knee-llm-report-labels-sol56",
        "pilkwang/rsna-knee-llm-labels",
        "rayanbabur/rsna-knee-2026-calibrated-soft-targets",
        "dreaddevelopment/rsna-knee-labels",
        "yunusgmsoy/rsna-knee-llm-labels-4-source-merged",
    ],
    "kernel_sources": [],
    "competition_sources": ["rsna-knee-abnormality-detection"],
    "model_sources": ["drxc75/medicalnet-resnet/PyTorch/resnet10/1"],
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
