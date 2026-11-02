"""Build the self-contained, readable Kaggle training notebook.

This file is a local authoring helper; the delivered artifact is the .ipynb.
"""
from __future__ import annotations

import base64
import hashlib
import json
import textwrap
import zlib
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training" / "rsna_knee_end_to_end_train.ipynb"
SOURCE = ROOT / "research" / "competitive_phase45_20261011"
MANIFEST_PATH = ROOT / "research" / "representation_phase3_20261011" / "manifest.json"


def md(s: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())


def code(s: str):
    return nbf.v4.new_code_cell(textwrap.dedent(s).strip())


manifest_bytes = MANIFEST_PATH.read_bytes()
manifest = json.loads(manifest_bytes)
payload = {
    "targets": manifest["targets"],
    "rows": [
        {
            "StudyInstanceUID": r["StudyInstanceUID"],
            "group": r["group"],
            "fold": r["folds"][0],
            "targets": r["targets"],
            "label_confidence": r["label_confidence"],
            "expert_reused": r["expert_reused"],
        }
        for r in manifest["rows"]
    ],
}
payload_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
payload_b64 = base64.b64encode(zlib.compress(payload_bytes, 9)).decode()
payload_sha = hashlib.sha256(payload_bytes).hexdigest()

cells = [
    md("""
    # RSNA Knee: complete raw-DICOM-to-checkpoint training notebook

    **Purpose.** Train a *new* five-fold ConvNeXt-Tiny global + central-region reader for the 12 RSNA knee targets. This notebook contains data loading, label provenance, DICOM decoding, preprocessing, cached three-slice windows, architecture, optimization, grouped validation, and checkpoint export. Run cells in order in a Kaggle GPU notebook with the competition data and the ImageNet weights dataset attached.

    **What it is not.** It does not recreate the historical CoAtNet, original ConvNeXt, or compact checkpoints in the scored 0.954 ensemble. Its 256-pixel cache is a resource-conscious adaptation of the completed Phase 5 regional experiment, whose cache was 336 pixels. No score is claimed for this new variant. No competition submission is made.

    **Data use.** The frozen targets are report-derived soft labels plus 58 expert labels reused for training. They are not independent clinical ground truth; the 58 expert studies must not be used as an independent final test. Five grouped folds preserve the existing study/report-link exclusions. Two contradictory identical-image studies are excluded from supervised loss and weak-reference scoring.
    """),
    md("""
    ## 1. Inputs and resource contract

    Attach the RSNA Knee competition data and private ImageNet-only weights dataset `adityakumarrai0401/rsna-knee-side-phase3-imagenet-20261011`. Use a GPU runtime. The full 4,407-study uint8 cache is about **12.7 GB** before checkpoints and receipts. Preprocessing and five-fold training may take hours and use Kaggle GPU quota. Set `MODE = "smoke"` to test a few studies and training steps; change it to `"full"` to build the full cache and train five folds. A smoke result is never presented as accuracy evidence.

    The DICOM decoder must be able to decode every selected image. If Kaggle lacks the needed JPEG plugins, attach the private `rsna-knee-decoder-cp313` wheel dataset and run the installation cell. Missing or corrupt selected images stop the cache build; they never become default predictions.
    """),
    code("""
    from __future__ import annotations
    import base64, contextlib, copy, csv, hashlib, json, math, os, random, time, zlib
    from pathlib import Path
    import numpy as np
    import pandas as pd

    MODE = "smoke"  # change to "full" for all 4,407 studies and five folds
    assert MODE in {"smoke", "full"}
    SEED = 20261018
    EPOCHS = 8
    FOLDS = tuple(range(5))
    IMAGE_SIZE = 256
    BATCH_SIZE = 4
    OUT = Path("/kaggle/working/rsna_regional_new_reader")
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE = OUT / "cache_256"
    CACHE.mkdir(exist_ok=True)
    EXCLUDED_UIDS = {
        "1.2.826.0.1.3680043.8.498.99725523560254732094432244556623376711",
        "1.2.826.0.1.3680043.8.498.99926624968240681772735102303531613209",
    }
    print({"mode": MODE, "output": str(OUT), "cache": str(CACHE)})
    """),
    code("""
    # Optional offline install. Uncomment only if imports/decoder preflight fail.
    # import subprocess, sys
    # wheel_dir = Path('/kaggle/input/datasets/adityakumarrai0401/rsna-knee-decoder-cp313')
    # wheel_names = ('pylibjpeg-2.1.0-py3-none-any.whl',
    #                'pylibjpeg_libjpeg-2.4.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.whl',
    #                'pylibjpeg_openjpeg-2.6.0-cp313-cp313-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl')
    # subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index',
    #                 *[str(wheel_dir / name) for name in wheel_names]], check=True)
    """),
    code("""
    import cv2, pydicom, torch, timm
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader
    from safetensors.torch import load_file
    from sklearn.metrics import roc_auc_score
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    assert torch.cuda.is_available(), "Select a Kaggle GPU runtime"
    print({"torch": torch.__version__, "timm": timm.__version__,
           "pydicom": pydicom.__version__, "gpu": torch.cuda.get_device_name(0)})
    """),
    md("""
    ## 2. Frozen targets and grouped folds

    The next cell embeds only the minimal training rows from the frozen Phase 3 manifest. Its source SHA256 is `8b7374c984c7b735a3f48ba46b76ab52ee4105d826c929c919e70d57f01ee602`. Embedding avoids an undocumented dependency on a separate label dataset. The row data are compressed for notebook size; the hash and schema checks make the content auditable. Target order is fixed throughout.
    """),
    code(f"""
    FROZEN_TARGETS_B64 = {payload_b64!r}
    FROZEN_TARGETS_SHA256 = {payload_sha!r}
    raw_targets = zlib.decompress(base64.b64decode(FROZEN_TARGETS_B64))
    assert hashlib.sha256(raw_targets).hexdigest() == FROZEN_TARGETS_SHA256
    frozen = json.loads(raw_targets)
    TARGETS = frozen["targets"]
    ROWS = frozen["rows"]
    assert len(TARGETS) == 12 and len(ROWS) == 4407
    assert len({{r["StudyInstanceUID"] for r in ROWS}}) == len(ROWS)
    assert all(len(r["targets"]) == len(r["label_confidence"]) == 12 for r in ROWS)
    assert all(r["fold"] in range(5) for r in ROWS)
    Y = np.asarray([r["targets"] for r in ROWS], dtype=np.float32)
    CONF = np.asarray([r["label_confidence"] for r in ROWS], dtype=np.float32)
    assert np.isfinite(Y).all() and np.isfinite(CONF).all()
    assert ((0 <= Y) & (Y <= 1)).all() and (CONF > 0).all()
    print({{"studies": len(ROWS), "groups": len({{r['group'] for r in ROWS}}),
           "fold_sizes": [sum(r['fold'] == k for r in ROWS) for k in range(5)],
           "targets": TARGETS}})
    """),
    md("""
    ## 3. Competition files and MRI preprocessing

    For each study: choose up to five series by plane and fluid sensitivity; sort DICOM slices by physical position with `InstanceNumber` fallback; select 44 positions across each series' 6–94% central depth span; apply modality LUT and MONOCHROME1 inversion; scale each selected series using pooled 2nd/98th percentiles; take a center crop covering about 140 mm; resize to 256×256; save uint8 slices and a validity mask. Each three-slice window stays inside its own series. The 256-pixel cache differs from the historical 336-pixel Phase 5 cache, so its accuracy is unknown until trained and evaluated.
    """),
    code("""
    roots = [Path('/kaggle/input/competitions/rsna-knee-abnormality-detection'),
             Path('/kaggle/input/rsna-knee-abnormality-detection')]
    DATA_ROOT = next((p for p in roots if (p / 'train.csv').is_file()
                      and (p / 'train_series.csv').is_file()
                      and (p / 'train_series').is_dir()), None)
    assert DATA_ROOT is not None, 'Attach the RSNA Knee competition input'
    train_frame = pd.read_csv(DATA_ROOT / 'train.csv', dtype=str)
    series_frame = pd.read_csv(DATA_ROOT / 'train_series.csv', dtype=str).fillna('')
    assert set(r['StudyInstanceUID'] for r in ROWS).issubset(set(train_frame['StudyInstanceUID']))
    SERIES = {str(uid): group.to_dict('records')
              for uid, group in series_frame.groupby('StudyInstanceUID', sort=False)}
    assert set(r['StudyInstanceUID'] for r in ROWS).issubset(SERIES)
    print({'competition_root': str(DATA_ROOT), 'series_rows': len(series_frame)})
    """),
]

reader = (ROOT / "inference" / "e03_dicom_reader.py").read_text()
reader = reader.replace("IMG = 336", "IMG = IMAGE_SIZE")
reader = reader.replace("SPAN = (0.15, 0.85)", "SPAN = (0.06, 0.94)")
reader = reader.replace("return volume, mask, receipt", "return volume, mask, receipt")
cells.append(code(reader))
window = (SOURCE / "dependencies" / "window_contract.py").read_text()
cells.append(code(window))
cells += [
    code("""
    READER = make_reader()
    # This reads one real study before allocating a 12.7 GB cache.
    probe_uid = ROWS[0]['StudyInstanceUID']
    probe_volume, probe_mask, probe_receipt = build_study(
        probe_uid, SERIES, str(DATA_ROOT / 'train_series'), READER, span=SPAN)
    assert probe_volume.shape == (44, IMAGE_SIZE, IMAGE_SIZE)
    assert probe_mask.shape == (44,)
    assert probe_receipt['slice_failures'] == probe_receipt['header_failures'] == 0, probe_receipt
    select_slot_windows(probe_mask, 3, training=False)
    print({'probe_uid': probe_uid, 'valid_slices': int(probe_mask.sum()),
           'slot_states': [s['state'] for s in probe_receipt['slots']]})
    """),
    md("""
    ### Build or reuse the cache

    A completed study cache is reused only after shape/mask checks. During a fresh build, arrays are first written to temporary files and renamed after decoding succeeds. The smoke mode caches 20 studies and trains only a brief pipeline check; use full mode for all 4,407 studies. Stop if free disk space is insufficient. Kaggle's `/kaggle/working` output capacity varies by runtime, so the capacity check is decisive.
    """),
    code("""
    import shutil
    cache_rows = ROWS[:20] if MODE == 'smoke' else ROWS
    estimated_bytes = len(cache_rows) * 44 * IMAGE_SIZE * IMAGE_SIZE
    free_bytes = shutil.disk_usage(CACHE).free
    assert free_bytes > estimated_bytes * 1.15 + 1_000_000_000, (
        f'Insufficient free space: need about {estimated_bytes/1e9:.1f} GB cache')
    def cache_paths(uid):
        return CACHE / (uid + '.npy'), CACHE / (uid + '.mask.npy')
    receipts = []
    for j, row in enumerate(cache_rows, 1):
        uid = row['StudyInstanceUID']
        vp, mp = cache_paths(uid)
        if vp.exists() and mp.exists():
            v = np.load(vp, mmap_mode='r'); m = np.load(mp)
            assert v.shape == (44, IMAGE_SIZE, IMAGE_SIZE) and v.dtype == np.uint8
            assert m.shape == (44,) and m.dtype == np.uint8
            select_slot_windows(m, 1, training=False)
        else:
            assert not vp.exists() and not mp.exists(), f'Incomplete cache for {uid}'
            v, m, receipt = build_study(uid, SERIES, str(DATA_ROOT / 'train_series'), READER, span=SPAN)
            assert receipt['slice_failures'] == receipt['header_failures'] == 0, receipt
            select_slot_windows(m, 1, training=False)
            # np.save(file handle) avoids an added .npy suffix on temporary paths.
            with (vp.with_suffix('.tmp')).open('wb') as f: np.save(f, v)
            with (mp.with_suffix('.tmp')).open('wb') as f: np.save(f, m)
            os.replace(vp.with_suffix('.tmp'), vp)
            os.replace(mp.with_suffix('.tmp'), mp)
            receipts.append(receipt)
        if j % 100 == 0 or j == len(cache_rows):
            print({'cache_completed': j, 'cache_total': len(cache_rows)}, flush=True)
    (OUT / 'new_cache_receipts.json').write_text(json.dumps(receipts, indent=2))
    print({'cache_studies': len(cache_rows), 'newly_decoded': len(receipts)})
    """),
    md("""
    ## 4. Three-slice windows and data loaders

    Training samples two windows per available slot; validation uses three deterministic windows per slot. Missing slots are masked. Spatial and intensity augmentation is applied coherently to windows from the same slot. No left-right flip is used.
    """),
    code("""
    def prepare_windows(volume, mask, per_slot, training=False, rng=None):
        centers, present, positions = select_slot_windows(mask.tolist(), per_slot, training, rng)
        x = np.zeros((5, per_slot, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        for slot, picks in enumerate(centers):
            for j, center in enumerate(picks):
                if center is not None: x[slot, j] = volume[center-1:center+2]
        return x, np.asarray(present, bool), np.asarray(positions, np.float32)

    class KneeDataset(Dataset):
        def __init__(self, indices, training, fold):
            self.indices = list(map(int, indices)); self.training = training
            self.fold = fold; self.epoch = 0
        def __len__(self): return len(self.indices)
        def __getitem__(self, j):
            i = self.indices[j]; uid = ROWS[i]['StudyInstanceUID']
            vp, mp = cache_paths(uid)
            volume = np.load(vp, mmap_mode='r'); mask = np.load(mp)
            rng = random.Random(SEED + 1000*self.fold + 1_000_003*self.epoch + i)
            x, present, pos = prepare_windows(volume, mask, 2 if self.training else 3,
                                               self.training, rng)
            return (torch.from_numpy(x), torch.from_numpy(present), torch.from_numpy(pos),
                    torch.from_numpy(Y[i]), torch.from_numpy(CONF[i]), i)

    def fold_indices(fold):
        valid = np.asarray([i for i, r in enumerate(ROWS) if r['fold'] == fold
                            and r['StudyInstanceUID'] not in EXCLUDED_UIDS], dtype=int)
        train = np.asarray([i for i, r in enumerate(ROWS) if r['fold'] != fold
                            and r['StudyInstanceUID'] not in EXCLUDED_UIDS], dtype=int)
        assert not {ROWS[i]['group'] for i in train} & {ROWS[i]['group'] for i in valid}
        return train, valid
    for f in FOLDS:
        tr, va = fold_indices(f)
        print({'fold': f, 'train': len(tr), 'valid': len(va), 'group_overlap': 0})
    """),
    md("""
    ## 5. Model architecture

    One ImageNet-initialized `convnext_tiny` encodes two views of each three-slice window: the full 140 mm crop and a central 60%-scale view. A 2-layer, 256-dimensional transformer combines acquisition, relative slice position and view tokens. Twelve separate attention heads pool tokens to 12 logits. The central crop is geometric; it is **not** verified ligament or cartilage localization. The pixel/cache adaptation and all weights trained by this notebook are new.
    """),
]

model = (SOURCE / "model.py").read_text()
model = model.replace("windows=x.reshape(b*s,k,3,336,336)", "windows=x.reshape(b*s,k,3,IMAGE_SIZE,IMAGE_SIZE)")
model = model.replace("from window_contract import select_slot_windows", "# select_slot_windows is defined above")
# The source file's prepare_windows would shadow the notebook's earlier definition;
# retain only the model class and its imports.
model = model.split("\ndef prepare_windows(")[0]
cells.append(code(model))
cells += [
    code("""
    # Generic pretraining only; reject ambiguous or modified weight files.
    EXPECTED_PRETRAINED_SHA256 = '08b9dc9c3a3a29421de7996761e176501896d1ae7fc3085cf56a643772329276'
    weights = list(Path('/kaggle/input').rglob('convnext.safetensors'))
    assert len(weights) == 1, f'Expected one ImageNet ConvNeXt file, found {weights}'
    PRETRAINED = weights[0]
    def sha256_file(path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(8<<20), b''): h.update(block)
        return h.hexdigest()
    assert sha256_file(PRETRAINED) == EXPECTED_PRETRAINED_SHA256
    net = RegionReader(PRETRAINED).cuda()
    sample_i = next(i for i, r in enumerate(ROWS[:len(cache_rows)])
                    if r['StudyInstanceUID'] == probe_uid)
    xx, mm, pp, yy, cc, _ = KneeDataset([sample_i], False, 0)[0]
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        test_logits = net(xx[None].cuda(), mm[None].cuda(), pp[None].cuda())
    assert test_logits.shape == (1, 12) and torch.isfinite(test_logits).all()
    del net
    torch.cuda.empty_cache()
    print({'pretrained': str(PRETRAINED), 'forward_shape': tuple(test_logits.shape)})
    """),
    md("""
    ## 6. Fixed training protocol

    Five grouped folds, eight fixed epochs, weighted soft-label BCE, AdamW (encoder 5e-5; head 4e-4), one-epoch warmup then cosine decay, gradient clip 3, float16 mixed precision and EMA 0.995. Checkpoints are the fixed final-epoch EMA; no validation-driven checkpoint selection. A smoke run uses a tiny subset and two optimizer steps and is deliberately not scored.
    """),
    code("""
    def atomic_torch_save(obj, path):
        path = Path(path); tmp = path.with_suffix(path.suffix + '.tmp')
        torch.save(obj, tmp); os.replace(tmp, path)

    def train_fold(fold):
        train_idx, valid_idx = fold_indices(fold)
        if MODE == 'smoke':
            available = {r['StudyInstanceUID'] for r in cache_rows}
            train_idx = np.asarray([i for i in train_idx if ROWS[i]['StudyInstanceUID'] in available][:8])
            valid_idx = np.asarray([i for i in valid_idx if ROWS[i]['StudyInstanceUID'] in available][:4])
            if len(train_idx) < 2 or len(valid_idx) < 1:
                print({'fold': fold, 'smoke': 'skipped; too few cached rows'})
                return None
        assert len(train_idx) and len(valid_idx)
        torch.manual_seed(SEED + fold*1000)
        model = RegionReader(PRETRAINED).cuda().to(memory_format=torch.channels_last)
        ema = copy.deepcopy(model).eval(); ema.requires_grad_(False)
        ds = KneeDataset(train_idx, True, fold)
        vs = KneeDataset(valid_idx, False, fold)
        loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
                            pin_memory=True, persistent_workers=False)
        val_loader = DataLoader(vs, batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                                pin_memory=True, persistent_workers=False)
        backbone = list(model.encoder.parameters()); ids = {id(p) for p in backbone}
        head = [p for p in model.parameters() if id(p) not in ids]
        optim = torch.optim.AdamW([{'params': backbone, 'lr': 5e-5},
                                   {'params': head, 'lr': 4e-4}], weight_decay=.025)
        epochs = 1 if MODE == 'smoke' else EPOCHS
        total_steps = len(loader)*epochs
        warmup = max(len(loader), 1)
        def factor(step):
            return min((step+1)/warmup, 1.) * (
                .01 + .99*(1+math.cos(math.pi*min(step,total_steps)/total_steps))/2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, factor)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024)
        started = time.monotonic(); steps = 0
        for epoch in range(epochs):
            ds.epoch = epoch; model.train(); losses = []
            for x, mask, pos, target, confidence, _ in loader:
                x=x.cuda(non_blocking=True); mask=mask.cuda(non_blocking=True)
                pos=pos.cuda(non_blocking=True); target=target.cuda(non_blocking=True)
                confidence=confidence.cuda(non_blocking=True)
                optim.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(x, mask, pos, augment=True)
                cell = F.binary_cross_entropy_with_logits(logits.float(), target, reduction='none')
                loss = (cell*confidence).sum()/confidence.sum()
                assert torch.isfinite(loss), 'Nonfinite loss'
                before = scaler.get_scale()
                scaler.scale(loss).backward(); scaler.unscale_(optim)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3.)
                assert torch.isfinite(grad_norm), 'Nonfinite gradients'
                scaler.step(optim); scaler.update()
                if scaler.get_scale() >= before:
                    scheduler.step()
                    with torch.no_grad():
                        for pe, pm in zip(ema.parameters(), model.parameters()):
                            pe.lerp_(pm, .005)
                losses.append(float(loss.detach())); steps += 1
                if MODE == 'smoke' and steps >= 2: break
            print({'fold': fold, 'epoch': epoch+1, 'mean_train_loss': float(np.mean(losses)),
                   'elapsed_seconds': round(time.monotonic()-started, 1)}, flush=True)
            if MODE == 'full':
                atomic_torch_save({'model': {k:v.detach().cpu() for k,v in ema.state_dict().items()},
                    'fold': fold, 'epochs': epoch+1, 'targets': TARGETS, 'cache_image_size': IMAGE_SIZE,
                    'target_sha256': FROZEN_TARGETS_SHA256, 'pretrained_sha256': EXPECTED_PRETRAINED_SHA256,
                    'train_uids': [ROWS[i]['StudyInstanceUID'] for i in train_idx],
                    'valid_uids': [ROWS[i]['StudyInstanceUID'] for i in valid_idx]},
                    OUT / f'fold{fold}_ema.pt')
            if MODE == 'smoke': break
        ema.eval(); predictions=[]; eval_rows=[]
        with torch.inference_mode():
            for x, mask, pos, _, _, ii in val_loader:
                x=x.cuda(non_blocking=True); mask=mask.cuda(non_blocking=True)
                pos=pos.cuda(non_blocking=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    prob=ema(x,mask,pos).float().sigmoid()
                assert torch.isfinite(prob).all(), 'Nonfinite validation prediction'
                predictions.append(prob.cpu().numpy()); eval_rows.extend(ii.tolist())
        assert eval_rows == valid_idx.tolist()
        pred = np.concatenate(predictions)
        if MODE == 'full':
            np.savez_compressed(OUT / f'fold{fold}_validation.npz', indices=valid_idx,
                                predictions=pred)
        del model, ema, optim, scaler, loader, val_loader
        torch.cuda.empty_cache()
        return valid_idx, pred
    """),
    code("""
    # Each fold trains from identical ImageNet initialization and its own held-out group.
    results = []
    for fold in FOLDS if MODE == 'full' else (0, 1, 2, 3, 4):
        result = train_fold(fold)
        if result is not None: results.append(result)
    print({'mode': MODE, 'completed_folds': len(results)})
    """),
    md("""
    ## 7. Grouped out-of-fold evaluation and receipts

    Only a completed full five-fold run produces OOF metrics. These AUCs compare predictions with the frozen report-derived proxy labels, not independent expert clinical truth and not Kaggle public or private leaderboard scores. The two contradictory identical-image IDs are excluded. The known reused 58 expert labels remain a diagnostic limitation. No blend weights are tuned here.
    """),
    code("""
    if MODE == 'full':
        assert len(results) == 5, 'All five folds are required for OOF results'
        indices = np.concatenate([r[0] for r in results])
        preds = np.concatenate([r[1] for r in results])
        expected = {i for i, r in enumerate(ROWS) if r['StudyInstanceUID'] not in EXCLUDED_UIDS}
        assert len(indices) == len(set(indices)) and set(indices) == expected
        oof = pd.DataFrame({'StudyInstanceUID': [ROWS[i]['StudyInstanceUID'] for i in indices],
                            'group': [ROWS[i]['group'] for i in indices],
                            'fold': [ROWS[i]['fold'] for i in indices]})
        for j, target in enumerate(TARGETS): oof[target] = preds[:, j]
        oof.to_csv(OUT / 'oof_predictions.csv', index=False)
        # Soft proxy labels are thresholded only for this diagnostic AUC.
        binary = Y[indices] >= .5
        metrics = []
        for j, target in enumerate(TARGETS):
            label = binary[:, j]
            auc = float(roc_auc_score(label, preds[:, j])) if len(np.unique(label)) == 2 else None
            metrics.append({'target': target, 'positives': int(label.sum()),
                            'negatives': int((~label).sum()), 'weak_proxy_auc': auc})
        macro = float(np.mean([r['weak_proxy_auc'] for r in metrics if r['weak_proxy_auc'] is not None]))
        pd.DataFrame(metrics).to_csv(OUT / 'weak_proxy_per_class_auc.csv', index=False)
        report = {'status': 'COMPLETE_NEW_VARIANT_WEAK_PROXY_OOF', 'macro_auc': macro,
                  'rows': len(indices), 'folds': 5, 'cache_resolution': IMAGE_SIZE,
                  'target_sha256': FROZEN_TARGETS_SHA256,
                  'pretrained_sha256': EXPECTED_PRETRAINED_SHA256,
                  'note': 'Not a Kaggle leaderboard score or independent clinical validation'}
        (OUT / 'training_report.json').write_text(json.dumps(report, indent=2))
        print(report)
    else:
        print('SMOKE ONLY: no OOF AUC and no scored checkpoint')
    """),
    md("""
    ## 8. Checkpoint replay and next decision

    A full run saves `fold0_ema.pt` through `fold4_ema.pt`, five validation arrays, OOF predictions, per-class proxy AUC and a report. Checkpoint replay below verifies one saved model on a known validation study. To decide whether this branch helps the scored 0.954 ensemble, compare its **eligible held-out errors** with the existing readers and then run a separate fixed-blend inference evaluation. This notebook does not submit or publish anything.
    """),
    code("""
    if MODE == 'full':
        ckpt_path = OUT / 'fold0_ema.pt'
        saved = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        assert saved['epochs'] == EPOCHS and saved['targets'] == TARGETS
        assert saved['target_sha256'] == FROZEN_TARGETS_SHA256
        replay = RegionReader().cuda().eval()
        replay.load_state_dict(saved['model'], strict=True)
        row_id = next(i for i, r in enumerate(ROWS)
                      if r['StudyInstanceUID'] == saved['valid_uids'][0])
        x, mask, pos, _, _, _ = KneeDataset([row_id], False, 0)[0]
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            prob = replay(x[None].cuda(), mask[None].cuda(), pos[None].cuda()).float().sigmoid()
        assert prob.shape == (1, 12) and torch.isfinite(prob).all()
        print({'checkpoint': str(ckpt_path), 'sha256': sha256_file(ckpt_path),
               'replay_uid': ROWS[row_id]['StudyInstanceUID'], 'finite': True})
    else:
        print('Run full mode to export and replay final checkpoints.')
    """),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "kaggle": {"accelerator": "gpu", "internet": False, "is_private": True},
})
nbf.validate(nb)
OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(json.dumps({"notebook": str(OUT), "cells": len(cells), "bytes": OUT.stat().st_size,
                  "embedded_target_sha256": payload_sha}))
