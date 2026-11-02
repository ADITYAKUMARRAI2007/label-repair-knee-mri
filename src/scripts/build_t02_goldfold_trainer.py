#!/usr/bin/env python3
"""Build T02: five-fold full ConvNeXt fine tuning on the public 11 GiB cache."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / "public_kernels/pixel_cache/rsna-knee-500gb-to-11gib-cpu-pixel-cache.ipynb"
OUT_DIR = ROOT / "public_kernels/t02_goldfold_train"
OUT_SCRIPT = OUT_DIR / "rsna-knee-goldfold-convnext-train.py"

TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def normalize_report(value: object) -> str:
    text = re.sub(r"[^\w\s]", " ", str(value).lower())
    text = re.sub(r"\d+", "#", text)
    return re.sub(r"\s+", " ", text).strip()


def read_family(path: str, train_uids: list[str]) -> np.ndarray:
    frame = pd.read_csv(ROOT / path, dtype={"StudyInstanceUID": str})
    frame = frame.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    values = frame.reindex(train_uids)[TARGETS].to_numpy(np.float32)
    return values


def make_folds(train: pd.DataFrame, fused: np.ndarray, seed: int = 20260930) -> np.ndarray:
    """Balance Gold58 labels first, then balance every weak row and fold size."""
    gold_mask = train[TARGETS].notna().all(axis=1).to_numpy()
    gold_idx = np.flatnonzero(gold_mask)
    gold_y = train.loc[gold_mask, TARGETS].to_numpy(np.int8)
    capacities = np.array([12, 12, 12, 11, 11], dtype=np.int64)
    rng = np.random.default_rng(seed)
    best_score = float("inf")
    best_gold_folds = None
    total_pos = gold_y.sum(axis=0)
    for _ in range(50000):
        perm = rng.permutation(len(gold_idx))
        gf = np.empty(len(gold_idx), np.int8)
        start = 0
        for fold, count in enumerate(capacities):
            gf[perm[start:start + count]] = fold
            start += count
        counts = np.stack([gold_y[gf == f].sum(axis=0) for f in range(5)])
        expected = capacities[:, None] * total_pos[None, :] / len(gold_idx)
        denom = np.maximum(expected * (1 - total_pos[None, :] / len(gold_idx)), 0.5)
        score = float(np.sum((counts - expected) ** 2 / denom))
        if np.any(counts == 0) or np.any(counts == capacities[:, None]):
            score += 1000.0
        if score < best_score:
            best_score = score
            best_gold_folds = gf.copy()
    if best_gold_folds is None:
        raise RuntimeError("could not construct Gold58 folds")

    fold = np.full(len(train), -1, np.int8)
    for row, f in zip(gold_idx, best_gold_folds):
        fold[row] = int(f)

    # The image model never receives report text. Balance individual weak studies here;
    # keeping thousands of template-identical reports together creates unusable folds
    # while providing no protection against pixel leakage.
    hard = (fused >= 0.5).astype(np.float32)
    fold_n = np.bincount(fold[fold >= 0], minlength=5).astype(np.float64)
    fold_pos = np.stack([
        hard[fold == f].sum(axis=0) if np.any(fold == f) else np.zeros(len(TARGETS))
        for f in range(5)
    ])
    capacities_all = np.array([882, 882, 881, 881, 881], dtype=np.int64)
    target_pos = hard.sum(axis=0) / 5.0
    remaining = np.flatnonzero(fold < 0)
    rarity = np.sum(hard[remaining] / np.maximum(hard.sum(axis=0), 1), axis=1)
    jitter = rng.random(len(remaining)) * 1e-8
    remaining = remaining[np.argsort(-(rarity + jitter))]
    for idx in remaining:
        pos = hard[idx]
        scores = []
        for f in range(5):
            if fold_n[f] >= capacities_all[f]:
                scores.append(float("inf"))
                continue
            target_n = float(capacities_all[f])
            n_score = ((fold_n[f] + 1 - target_n) / target_n) ** 2
            p_score = np.mean(((fold_pos[f] + pos - target_pos) / np.maximum(target_pos, 1)) ** 2)
            scores.append(float(n_score + 0.35 * p_score))
        chosen = min(range(5), key=lambda f: (scores[f], fold_n[f], f))
        fold[idx] = chosen
        fold_n[chosen] += 1
        fold_pos[chosen] += pos
    if np.any(fold < 0):
        raise RuntimeError("fold assignment incomplete")
    return fold


def build_payload() -> tuple[str, dict]:
    train = pd.read_csv(ROOT / "data/train.csv", dtype={"StudyInstanceUID": str})
    uids = train["StudyInstanceUID"].tolist()
    rayan = read_family("data/labels/rayan/train_targets_soft.csv", uids)
    others = [
        read_family("data/labels/pilkwang/report_labels_v2.csv", uids),
        read_family("data/labels/gpt56/labels_llm_gpt56sol.csv", uids),
        read_family("data/labels/steven/llm_labels_v4_blend.csv", uids),
        read_family("data/labels/dread/labels_llm_soft.csv", uids),
        read_family("data/labels/yunus/report_labels_v5.csv", uids),
    ]
    robust = np.nanmedian(np.stack(others), axis=0)
    fused = np.clip(0.60 * rayan + 0.40 * robust, 0.01, 0.99).astype(np.float32)
    if not np.isfinite(fused).all():
        raise RuntimeError("T02 fused targets contain missing values")

    weight_frame = pd.read_csv(
        ROOT / "data/labels/rayan/train_targets_weights.csv",
        dtype={"StudyInstanceUID": str},
    ).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    weights = weight_frame.reindex(uids)[TARGETS].to_numpy(np.float32)
    weights = np.clip(np.nan_to_num(weights, nan=0.35), 0.25, 1.0)

    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    fused[gold] = train.loc[gold, TARGETS].to_numpy(np.float32)
    weights[gold] = 4.0
    fold = make_folds(train, fused)

    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        uids=np.asarray(uids),
        targets=fused,
        weights=weights,
        folds=fold,
        gold=gold.astype(np.uint8),
    )
    raw = buffer.getvalue()
    info = {
        "rows": len(uids),
        "gold": int(gold.sum()),
        "fold_counts": np.bincount(fold, minlength=5).tolist(),
        "gold_fold_counts": np.bincount(fold[gold], minlength=5).tolist(),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return base64.b64encode(raw).decode("ascii"), info


DRIVER = r'''

# --------------------------- T02 owned driver -------------------------------
import json as _json
import hashlib as _hashlib
from types import SimpleNamespace as _SimpleNamespace

T02_SCHEMA = "rsna_knee_goldfold_convnext_v1"
T02_INIT_SHA256 = "bec04468edeaaa0c75191a221d4e3b80bada5cfd263eeec36cb093548de93147"
T02_PUBLIC_LABEL_SHA256 = {
    "train_targets_soft.csv": "6dac0e2d9ba79065ec794fee743c0286978c2323fcb9625bb74ef2a73d4d1475",
    "train_targets_weights.csv": "3fedf0a0f792d5644f0e9fd537a3add2e0f81aa25e9540db74ec523bdd3a3c18",
    "report_labels_v2.csv": "6f704a7bdb2f894cc49445b19ba7c4378c3f548d3449e00361e10044bee40920",
    "labels_llm_gpt56sol.csv": "a79b05e609336811d51c97f9a48f00ea8004244a1a80fe886504a62bb39867ae",
    "llm_labels_v4_blend.csv": "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4",
    "labels_llm_soft.csv": "8cec477fb555e88bbb4196197fd3f806cbafed8cc2f6b2d48d26a8d49fbd86a1",
    "report_labels_v5.csv": "16cf44a89e027c80ff9b1a69e1ac6713d1f6da1765b3748c01f6ae91827ef167",
}

CFG.epochs = 6
CFG.swa_epochs = 3
CFG.batch_studies = 16
CFG.eval_batch = 32
CFG.lr_backbone = 2e-5
CFG.lr_head = 5e-4
CFG.weight_decay = 0.02
CFG.time_budget = 11.6 * 3600
CFG.crop_mm = 130.0
CFG.n_group_max = 3


def _sha256(path):
    h = _hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _bounded_files(name, max_depth=5):
    base = Path("/kaggle/input")
    hits = []
    for root, dirs, files in os.walk(base):
        rel = Path(root).relative_to(base)
        depth = len(rel.parts)
        dirs[:] = [d for d in dirs if d not in {
            "train_series", "test_series", "images", "train_images", "test_images"
        } and depth < max_depth]
        if name in files:
            hits.append(Path(root) / name)
    return hits


def _find_unique(name, expected_sha=None):
    hits = []
    for path in _bounded_files(name):
        if expected_sha is not None and _sha256(path) != expected_sha:
            continue
        hits.append(path)
    unique = sorted({p.resolve() for p in hits}, key=str)
    if len(unique) != 1:
        raise RuntimeError(f"expected exactly one {name}, got {unique}")
    return unique[0]


def _read_public_family(name, expected_uids):
    path = _find_unique(name, T02_PUBLIC_LABEL_SHA256[name])
    frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    frame = frame.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    values = frame.reindex(expected_uids)[TARGETS].to_numpy(np.float32)
    return values, path


def _make_t02_folds(targets, gold_mask, seed=20260930):
    gold_idx = np.flatnonzero(gold_mask)
    gold_y = targets[gold_idx].astype(np.int8)
    capacities = np.array([12, 12, 12, 11, 11], dtype=np.int64)
    rng = np.random.default_rng(seed)
    best_score, best_gold_folds = float("inf"), None
    total_pos = gold_y.sum(axis=0)
    for _ in range(50000):
        perm = rng.permutation(len(gold_idx))
        gf = np.empty(len(gold_idx), np.int8)
        start = 0
        for fold, count in enumerate(capacities):
            gf[perm[start:start + count]] = fold
            start += count
        counts = np.stack([gold_y[gf == f].sum(axis=0) for f in range(5)])
        expected = capacities[:, None] * total_pos[None, :] / len(gold_idx)
        denom = np.maximum(expected * (1 - total_pos[None, :] / len(gold_idx)), 0.5)
        score = float(np.sum((counts - expected) ** 2 / denom))
        if np.any(counts == 0) or np.any(counts == capacities[:, None]):
            score += 1000.0
        if score < best_score:
            best_score, best_gold_folds = score, gf.copy()
    if best_gold_folds is None:
        raise RuntimeError("could not construct T02 Gold folds")
    fold = np.full(len(targets), -1, np.int8)
    fold[gold_idx] = best_gold_folds
    hard = (targets >= 0.5).astype(np.float32)
    fold_n = np.bincount(fold[fold >= 0], minlength=5).astype(np.float64)
    fold_pos = np.stack([hard[fold == f].sum(axis=0) for f in range(5)])
    capacities_all = np.array([882, 882, 881, 881, 881], dtype=np.int64)
    target_pos = hard.sum(axis=0) / 5.0
    remaining = np.flatnonzero(fold < 0)
    rarity = np.sum(hard[remaining] / np.maximum(hard.sum(axis=0), 1), axis=1)
    remaining = remaining[np.argsort(-(rarity + rng.random(len(remaining)) * 1e-8))]
    for idx in remaining:
        scores = []
        for f in range(5):
            if fold_n[f] >= capacities_all[f]:
                scores.append(float("inf")); continue
            n_score = ((fold_n[f] + 1 - capacities_all[f]) / capacities_all[f]) ** 2
            p_score = np.mean(((fold_pos[f] + hard[idx] - target_pos)
                               / np.maximum(target_pos, 1)) ** 2)
            scores.append(float(n_score + 0.35 * p_score))
        chosen = min(range(5), key=lambda f: (scores[f], fold_n[f], f))
        fold[idx] = chosen; fold_n[chosen] += 1; fold_pos[chosen] += hard[idx]
    if np.bincount(fold, minlength=5).tolist() != capacities_all.tolist():
        raise RuntimeError("T02 fold balance failed")
    return fold


def _construct_targets(train):
    uids = train["StudyInstanceUID"].astype(str).tolist()
    rayan, rayan_path = _read_public_family("train_targets_soft.csv", uids)
    other_names = ["report_labels_v2.csv", "labels_llm_gpt56sol.csv",
                   "llm_labels_v4_blend.csv", "labels_llm_soft.csv",
                   "report_labels_v5.csv"]
    others, paths = [], [rayan_path]
    for name in other_names:
        values, path = _read_public_family(name, uids)
        others.append(values); paths.append(path)
    targets = np.clip(0.60 * rayan + 0.40 * np.nanmedian(np.stack(others), axis=0),
                      0.01, 0.99).astype(np.float32)
    weights, weight_path = _read_public_family("train_targets_weights.csv", uids)
    paths.append(weight_path)
    weights = np.clip(np.nan_to_num(weights, nan=0.35), 0.25, 1.0).astype(np.float32)
    gold = train[TARGETS].notna().all(axis=1).to_numpy()
    targets[gold] = train.loc[gold, TARGETS].to_numpy(np.float32)
    weights[gold] = 4.0
    if not np.isfinite(targets).all() or int(gold.sum()) != 58:
        raise RuntimeError("T02 runtime label construction failed")
    folds = _make_t02_folds(targets, gold)
    provenance = {p.name: _sha256(p) for p in paths}
    return targets, weights, folds, gold.astype(np.uint8), provenance


class _CacheShards:
    def __init__(self, arrays, row_map, shape_tail):
        self.arrays = arrays
        self.row_map = np.asarray(row_map, np.int64)
        self.shape = (len(row_map), *shape_tail)
        self.dtype = arrays[0].dtype

    def __len__(self):
        return len(self.row_map)

    def __getitem__(self, item):
        index = np.arange(len(self))[item]
        scalar = np.ndim(index) == 0
        index = np.atleast_1d(index).astype(np.int64)
        out = np.empty((len(index), *self.shape[1:]), dtype=self.dtype)
        locations = self.row_map[index]
        for shard in np.unique(locations[:, 0]):
            take = np.flatnonzero(locations[:, 0] == shard)
            out[take] = self.arrays[int(shard)][locations[take, 1]]
        return out[0] if scalar else out


def _load_cache(expected_uids):
    metas = []
    for path in _bounded_files("cache_meta.json"):
        try:
            meta = _json.loads(path.read_text())
        except Exception:
            continue
        if (meta.get("img") == 224 and float(meta.get("crop_mm", -1)) == 130.0
                and meta.get("group") == 3 and meta.get("n_group") == 3
                and meta.get("window") == "0.35,0.65"):
            metas.append((path, meta))
    if len(metas) != 1:
        raise RuntimeError(f"expected one exact public pixel cache, got {[str(x[0]) for x in metas]}")
    meta_path, meta = metas[0]
    root = meta_path.parent
    stems = sorted(k for k in meta.get("splits", {}) if k.startswith("train"))
    if len(stems) != 4:
        raise RuntimeError(f"expected four train cache shards, got {stems}")
    arrays, masks, uid_rows = [], [], {}
    for shard, stem in enumerate(stems):
        cache_path = root / f"{stem}_cache.npy"
        mask_path = root / f"{stem}_mask.npy"
        uid_path = root / f"{stem}_studies.csv"
        if not (cache_path.is_file() and mask_path.is_file() and uid_path.is_file()):
            raise RuntimeError(f"cache shard incomplete: {stem}")
        arr = np.load(cache_path, mmap_mode="r")
        msk = np.load(mask_path, mmap_mode="r")
        ids = pd.read_csv(uid_path, dtype={"StudyInstanceUID": str})["StudyInstanceUID"].tolist()
        if arr.shape != (len(ids), 6, 9, 224, 224) or msk.shape != (len(ids), 6):
            raise RuntimeError(f"cache shape drift in {stem}: {arr.shape}, {msk.shape}")
        arrays.append(arr); masks.append(msk)
        for row, uid in enumerate(ids):
            if uid in uid_rows:
                raise RuntimeError(f"duplicate cache UID {uid}")
            uid_rows[uid] = (shard, row)
    if set(uid_rows) != set(expected_uids):
        raise RuntimeError("public cache does not exactly cover competition train IDs")
    row_map = np.asarray([uid_rows[u] for u in expected_uids], np.int64)
    mask = np.stack([masks[s][r] for s, r in row_map]).astype(np.uint8)
    cache = _CacheShards(arrays, row_map, (6, 9, 224, 224))
    return cache, mask, meta_path, meta


def build_encoder():
    import timm
    init_path = _find_unique("dinov5_convnext_backbone.pt", T02_INIT_SHA256)
    backbone = timm.create_model(
        "convnext_tiny.fb_in22k_ft_in1k", pretrained=False,
        num_classes=0, global_pool="avg",
    )
    state = torch.load(init_path, map_location="cpu", weights_only=True)
    missing, unexpected = backbone.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"T02 initialization drift: {missing}, {unexpected}")
    encoder = Encoder(backbone, "timm", int(backbone.num_features))
    for parameter in encoder.parameters():
        parameter.requires_grad = True
    return encoder


def _base_model(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def _train_fold_full(cache, mask, targets, weights, train_idx, valid_idx, dev, fold):
    seed_all(20260930 + fold)
    base = Model(build_encoder()).to(dev)
    model = (nn.DataParallel(base, device_ids=list(range(torch.cuda.device_count())))
             if torch.cuda.device_count() >= 2 else base)
    core = _base_model(model)
    optimizer = torch.optim.AdamW([
        {"params": core.encoder.parameters(), "lr": CFG.lr_backbone},
        {"params": core.head.parameters(), "lr": CFG.lr_head},
    ], weight_decay=CFG.weight_decay)
    steps_per_epoch = max(len(train_idx) // CFG.batch_studies, 1)
    total_steps = CFG.epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[CFG.lr_backbone, CFG.lr_head], total_steps=total_steps,
        pct_start=0.15, div_factor=10.0, final_div_factor=100.0,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    history = []
    swa = None
    swa_n = 0
    y_valid = (targets[valid_idx] >= 0.5).astype(np.int8)

    for epoch in range(CFG.epochs):
        model.train()
        rng = np.random.default_rng(20260930 + fold * 100 + epoch)
        order = rng.permutation(train_idx)
        losses = []
        for start in range(0, len(order) - CFG.batch_studies + 1, CFG.batch_studies):
            sel = order[start:start + CFG.batch_studies]
            rows = torch.from_numpy(cache[sel]).to(dev, non_blocking=True)
            group = int(rng.integers(CFG.n_group_max))
            images = augment(take_group(rows, group))
            present = torch.from_numpy(mask[sel]).to(dev, non_blocking=True)
            y = torch.from_numpy(targets[sel]).to(dev, non_blocking=True)
            w = torch.from_numpy(weights[sel]).to(dev, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(images, present)
                loss = (F.binary_cross_entropy_with_logits(logits, y, reduction="none") * w).mean()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(core.parameters(), 2.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            losses.append(float(loss.detach().cpu()))

        pred = predict(model, cache, mask, valid_idx, dev, CFG.n_group_max)
        derived_auc = macro_auc(y_valid, pred)
        gold_local = np.flatnonzero(T02_GOLD_MASK[valid_idx] > 0)
        gold_auc = float("nan")
        if len(gold_local):
            gold_auc = macro_auc(
                T02_TARGETS_MATRIX[valid_idx[gold_local]].astype(np.int8), pred[gold_local]
            )
        history.append({"epoch": epoch + 1, "loss": float(np.mean(losses)),
                        "derived_auc": derived_auc, "gold_auc": gold_auc})
        print(f"fold={fold} epoch={epoch+1:02d} loss={np.mean(losses):.4f} "
              f"derived={derived_auc:.4f} Gold={gold_auc:.4f}", flush=True)

        if epoch >= CFG.epochs - CFG.swa_epochs:
            state = {k: v.detach().float().cpu().clone()
                     for k, v in core.state_dict().items()}
            if swa is None:
                swa, swa_n = state, 1
            else:
                swa_n += 1
                for key in swa:
                    swa[key] += (state[key] - swa[key]) / swa_n

    if swa is None or swa_n != CFG.swa_epochs:
        raise RuntimeError(f"fold {fold}: incomplete SWA state")
    core.load_state_dict(swa, strict=True)
    final_pred = predict(model, cache, mask, valid_idx, dev, CFG.n_group_max)
    checkpoint = {
        "schema": T02_SCHEMA, "fold": fold, "targets": TARGETS,
        "input": {"img": 224, "crop_mm": 130.0, "slots": [s[0] for s in SLOTS],
                  "group": 3, "n_group": 3, "window": "0.35,0.65"},
        "backbone": "convnext_tiny.fb_in22k_ft_in1k",
        "init_sha256": T02_INIT_SHA256, "swa_epochs": CFG.swa_epochs,
        "state": swa,
    }
    path = Path("/kaggle/working") / f"t02_goldfold_convnext_fold{fold}.pt"
    torch.save(checkpoint, path)
    del model, base, core, optimizer, scheduler, scaler, swa
    gc.collect(); torch.cuda.empty_cache()
    return final_pred, history, path


def _main_t02():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("T02 requires the configured dual T4 accelerator")
    torch.backends.cudnn.benchmark = True
    root = find_root()
    train = pd.read_csv(root / "train.csv", dtype={"StudyInstanceUID": str})
    uids = train["StudyInstanceUID"].astype(str).tolist()
    global T02_TARGETS_MATRIX, T02_GOLD_MASK
    T02_TARGETS_MATRIX, runtime_weights, runtime_folds, T02_GOLD_MASK, label_provenance = _construct_targets(train)
    if T02_TARGETS_MATRIX.shape != (4407, 12) or T02_GOLD_MASK.sum() != 58:
        raise RuntimeError("T02 runtime target contract mismatch")
    cache, mask, cache_meta_path, cache_meta = _load_cache(uids)
    print("T02 cache", cache.shape, "coverage", mask.mean(0), flush=True)

    dev = select_device()
    oof = np.full((len(uids), len(TARGETS)), np.nan, np.float32)
    histories = {}
    paths = []
    fold_started = time.time()
    fold_estimate = None
    for fold in range(5):
        if fold_estimate is not None and time.time() - T0 + 1.15 * fold_estimate > CFG.time_budget:
            raise RuntimeError(f"insufficient time to begin required fold {fold}")
        train_idx = np.flatnonzero(runtime_folds != fold)
        valid_idx = np.flatnonzero(runtime_folds == fold)
        started = time.time()
        pred, history, path = _train_fold_full(
            cache, mask, T02_TARGETS_MATRIX, runtime_weights,
            train_idx, valid_idx, dev, fold,
        )
        oof[valid_idx] = pred
        histories[str(fold)] = history
        paths.append(path)
        elapsed = time.time() - started
        fold_estimate = elapsed if fold_estimate is None else max(fold_estimate, elapsed)
        print(f"fold={fold} complete seconds={elapsed:.1f}", flush=True)

    if not np.isfinite(oof).all():
        raise RuntimeError("T02 OOF is incomplete")
    gold_idx = np.flatnonzero(T02_GOLD_MASK > 0)
    gold_truth = train.loc[gold_idx, TARGETS].to_numpy(np.int8)
    gold_pred = oof[gold_idx]
    gold_macro = macro_auc(gold_truth, gold_pred)
    target_auc = per_target_auc(gold_truth, gold_pred)
    derived_macro = macro_auc((T02_TARGETS_MATRIX >= 0.5).astype(np.int8), oof)
    print("T02 Gold58 OOF", gold_macro, flush=True)
    print(_json.dumps(target_auc, indent=2), flush=True)

    oof_frame = pd.DataFrame({"StudyInstanceUID": np.asarray(uids)[gold_idx]})
    for j, target in enumerate(TARGETS):
        oof_frame[target] = gold_pred[:, j]
    oof_path = Path("/kaggle/working/t02_gold58_oof.csv")
    oof_frame.to_csv(oof_path, index=False)

    weight_info = {}
    for path in paths:
        weight_info[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    manifest = {
        "schema": T02_SCHEMA,
        "seed": 20260930,
        "targets": TARGETS,
        "training": {
            "folds": 5, "epochs": CFG.epochs, "swa_last": CFG.swa_epochs,
            "gold_weight": 4.0,
            "label_fusion": "0.60 calibrated Rayan + 0.40 median of five public families; Gold override",
            "optimizer": "AdamW OneCycle", "lr_backbone": CFG.lr_backbone,
            "lr_head": CFG.lr_head,
        },
        "input": {"img": 224, "crop_mm": 130.0, "slots": [s[0] for s in SLOTS],
                  "group": 3, "n_group": 3, "window": "0.35,0.65"},
        "validation": {"gold_n": 58, "gold_oof_macro_auc": gold_macro,
                       "per_target_auc": target_auc, "derived_oof_macro_auc": derived_macro,
                       "fold_counts": np.bincount(runtime_folds, minlength=5).tolist(),
                       "gold_fold_counts": np.bincount(runtime_folds[gold_idx], minlength=5).tolist()},
        "provenance": {"public_label_sha256": label_provenance,
                       "init_sha256": T02_INIT_SHA256,
                       "cache_meta": str(cache_meta_path), "cache_contract": cache_meta},
        "histories": histories,
        "weights": weight_info,
        "elapsed_seconds": time.time() - T0,
    }
    Path("/kaggle/working/t02_manifest.json").write_text(
        _json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(_json.dumps({"status": "COMPLETE", "gold_oof_macro_auc": gold_macro,
                       "derived_oof_macro_auc": derived_macro,
                       "elapsed_seconds": time.time() - T0}, indent=2), flush=True)


_main_t02()
'''


def main() -> None:
    notebook = json.loads(SOURCE_NOTEBOOK.read_text())
    selected = [3, 5, 6, 8, *range(10, 25)]
    pieces = []
    for index in selected:
        if notebook["cells"][index].get("cell_type") != "code":
            continue
        source = "".join(notebook["cells"][index].get("source", []))
        source = source.replace("from __future__ import annotations\n", "")
        pieces.append(source.rstrip())
    _, info = build_payload()
    driver = DRIVER
    header = '''#!/usr/bin/env python3
"""T02 Gold-fold ConvNeXt trainer.

Derived from the public CC BY 4.0 RSNA pixel-cache pipeline by Steven Lee Hans,
with runtime reconstruction from public label datasets, an exact cache contract,
MRI-adapted initialization, dual-T4 full fine tuning, fold-wise expert supervision,
and fail-closed receipts.
"""
from __future__ import annotations
'''
    text = header + "\n\n" + "\n\n".join(pieces) + driver
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SCRIPT.write_text(text)
    metadata = {
        "id": "adityakumarrai0401/rsna-knee-goldfold-convnext-train-v1",
        "title": "RSNA Knee GoldFold ConvNeXt Train V1",
        "code_file": OUT_SCRIPT.name,
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": [
            "stevenleehans/rsna-knee-llm-report-labels",
            "lixin73/rsna-knee-llm-report-labels-sol56",
            "pilkwang/rsna-knee-llm-labels",
            "rayanbabur/rsna-knee-2026-calibrated-soft-targets",
            "dreaddevelopment/rsna-knee-labels",
            "yunusgmsoy/rsna-knee-llm-labels-4-source-merged",
        ],
        "kernel_sources": [
            "stevenleehans/rsna-knee-500gb-to-11gib-cpu-pixel-cache",
            "adityakumarrai0401/rsna-knee-convnext-diversity-train-v2",
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (OUT_DIR / "payload-info.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps({"script": str(OUT_SCRIPT), "bytes": OUT_SCRIPT.stat().st_size,
                      "payload": info}, indent=2))


if __name__ == "__main__":
    main()
