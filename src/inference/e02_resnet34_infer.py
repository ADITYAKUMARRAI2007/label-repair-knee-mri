import os as _standalone_os
_standalone_os.environ.setdefault(
    'RSNA_E02_OUTPUT', '/kaggle/working/submission.csv'
)
# %% [code]
#!/usr/bin/env python3
"""E02 ResNet34 distilled student inference.

It recreates the public 44-slice Raptor layout from hidden-test DICOMs, samples
32 deterministic 2.5D windows, and serves the fixed epoch 15--18 checkpoint
average produced by the strict Gold58 holdout trainer. Every study must
complete; there is no neutral-prediction fallback.
"""
import os, sys, glob, time, json, gc, hashlib
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet34
# T4 (Turing) cuDNN v9 has fp16/fp32 conv engines but NOT bf16 for these shapes
# ("GET was unable to find an engine..."); benchmark lets it pick a valid algo for
# the fixed one-study window batch.
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# ---- fixed config (must match training exactly) -----------------------------
IMG = 336
NETWORK_IMG = 288
CROP_MM = 140.0
# Exact public training-corpus layout: 12 + 10 + 8 + 6 + 8 = 44 slices.
SLOTS = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
         ("Coronal", 0, 6), ("Axial", -1, 8)]
MAXS = sum(s[2] for s in SLOTS)
K_EVAL = 32
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# One independently trained E01 checkpoint family. The inference loader
# considers both the saved best epoch and top-three weight average, then selects
# the higher held-out Gold58 score. Public leaderboard performance remains
# unknown until the standalone code submission is graded.
ARMS = [{"file": "e02_resnet34_distill_swa.pt", "res": NETWORK_IMG, "w": 1.0}]


# ============================================================================
# Model -- verbatim from E02 training
# ============================================================================
class SimpleMIL(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet34(weights=None)
        self.encoder.fc = nn.Identity()
        self.norm = nn.LayerNorm(512)
        self.slice_head = nn.Linear(512, len(LAB))
        self.study_head = nn.Linear(1024, len(LAB))

    def forward(self, images):
        b, k = images.shape[:2]
        feat = self.norm(self.encoder(images.flatten(0, 1)).reshape(b, k, 512))
        slice_logits = self.slice_head(feat)
        local = slice_logits.topk(max(1, k // 4), dim=1).values.mean(1)
        global_logits = self.study_head(torch.cat([feat.mean(1), feat.amax(1)], dim=-1))
        return 0.5 * local + 0.5 * global_logits


def validate_checkpoint_contract(ck, pt_path):
    required = {"schema", "architecture", "state", "targets", "input", "training"}
    missing = required - set(ck)
    if missing:
        raise RuntimeError(f"E02 checkpoint {pt_path} is missing {sorted(missing)}")
    if ck["schema"] != "rsna_knee_e02_resnet34_distill_v1":
        raise RuntimeError(f"E02 schema mismatch: {ck['schema']!r}")
    if ck["architecture"] != "resnet34_simple_mil":
        raise RuntimeError(f"E02 architecture mismatch: {ck['architecture']!r}")
    if list(ck["targets"]) != LAB or int(ck["input"]["network_size"]) != NETWORK_IMG:
        raise RuntimeError("E02 target order or input geometry mismatch")
    if list(ck["training"]["swa_epochs"]) != [15, 16, 17, 18]:
        raise RuntimeError("E02 fixed SWA epoch contract mismatch")
    if int(ck["training"]["gold_rows_used_for_training"]) != 0:
        raise RuntimeError("E02 checkpoint violates strict Gold holdout")
    if not isinstance(ck["state"], dict) or len(ck["state"]) < 100:
        raise RuntimeError("E02 checkpoint state is empty or implausibly small")


def load_model(pt_path, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    validate_checkpoint_contract(ck, pt_path)
    ck_res = int(ck["input"]["network_size"])
    info = {"checkpoint": os.path.basename(pt_path), "architecture": ck["architecture"],
            "res": ck_res, "swa_epochs": ck["training"]["swa_epochs"],
            "gold_rows_used_for_training": ck["training"]["gold_rows_used_for_training"]}
    model = SimpleMIL()
    model.load_state_dict(ck["state"], strict=True)
    model.eval().to(device)
    del ck
    gc.collect()
    return model, ck_res, info


# ============================================================================
# Eval windowing -- verbatim from finetune_raptor.py StudyWindows (train=False)
# ============================================================================
def _eval_centers(mask, D, k):
    valid = np.where(mask > 0)[0]
    if len(valid) < 3:
        valid = np.arange(min(3, D))
    lo, hi = int(valid.min()), int(valid.max())
    cs = [c for c in range(lo + 1, hi) if c - 1 >= lo and c + 1 <= hi]
    if not cs:
        cs = [max(1, min((lo + hi) // 2, D - 2))]
    idx = np.linspace(0, len(cs) - 1, k).round().astype(int)
    return [cs[i] for i in idx]


def eval_windows(vol, mask, k, res, norm=NORM):
    D = vol.shape[0]
    cs = _eval_centers(mask, D, k)
    wins = np.empty((len(cs), 3, res, res), np.float32)
    for j, c in enumerate(cs):
        c = max(1, min(c, D - 2))
        tri = np.stack([vol[c - 1], vol[c], vol[c + 1]], 0).astype(np.float32) / 255.0
        t = torch.from_numpy(tri)
        if t.shape[-1] != res:
            t = F.interpolate(t[None], size=(res, res), mode="bilinear",
                              align_corners=False)[0]
        wins[j] = t.numpy()
    x = torch.from_numpy(wins)
    if norm == "imagenet":
        x = (x - _MEAN) / _STD
    return x


@torch.no_grad()
def infer_probs(model, xwins, device):
    x = xwins.unsqueeze(0).to(device)
    use_cuda = device != "cpu" and str(device).startswith("cuda")
    if use_cuda:
        # fp16 conv on T4 is fully cuDNN-supported (bf16 is NOT -> "no engine").
        try:
            with torch.autocast("cuda", dtype=torch.float16):
                o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
        except RuntimeError:
            # fp32 always has a Turing conv engine; slower but never drops a study.
            torch.cuda.empty_cache()
            o = torch.sigmoid(model(x).float())
            return o[0].cpu().numpy()
    o = torch.sigmoid(model(x).float())
    return o[0].cpu().numpy()


def rankpct(x):                                   # per-column percentile rank in [0,1]
    order = x.argsort(0).argsort(0).astype(np.float64)
    return order / max(1, (x.shape[0] - 1))


# ============================================================================
# Preprocessing -- verbatim from kprep2/dino_preprocess.py, retargeted to TEST
# ============================================================================
def _make_reader():
    import pydicom, cv2
    from pydicom.pixel_data_handlers.util import apply_modality_lut

    def order_and_meta(sdir):
        fs = glob.glob(sdir + "/*.dcm"); recs = []; ps_list = []
        for f in fs:
            try:
                h = pydicom.dcmread(f, stop_before_pixels=True)
                iop = getattr(h, 'ImageOrientationPatient', None)
                ipp = getattr(h, 'ImagePositionPatient', None)
                if iop is not None and ipp is not None and len(iop) == 6:
                    r = np.array(iop[:3], float); c = np.array(iop[3:], float)
                    n = np.cross(r, c); pos = float(np.dot(np.array(ipp, float), n))
                else:
                    pos = float(getattr(h, 'InstanceNumber', 0) or 0)
                ps = getattr(h, 'PixelSpacing', None); ps = float(ps[0]) if ps is not None else 0.5
                ps_list.append(ps); recs.append((pos, f, ps))
            except Exception:
                recs.append((0.0, f, 0.5))
        recs.sort(key=lambda x: x[0])
        med_ps = float(np.median(ps_list)) if ps_list else 0.5
        return [(f, ps) for _, f, ps in recs], med_ps

    def read_px(f):
        d = pydicom.dcmread(f)
        a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
        if str(getattr(d, 'PhotometricInterpretation', '')) == 'MONOCHROME1':
            a = a.max() - a
        return a

    def mm_crop_resize(a, ps):
        h, w = a.shape; cpx = int(round(CROP_MM / max(ps, 1e-3)))
        cpx = min(cpx, min(h, w)); y0 = (h - cpx) // 2; x0 = (w - cpx) // 2
        a = a[y0:y0 + cpx, x0:x0 + cpx]
        return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)

    return order_and_meta, read_px, mm_crop_resize


def _pick_series_for_slot(rows, plane, fluid, used):
    def protocol_flag(value):
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() in (
                '', 'nan', 'none', '<na>', 'unknown'):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in ('true', 'yes', 'y', 't'):
                return 1
            if text in ('false', 'no', 'n', 'f'):
                return 0
            return None
        if not np.isfinite(number) or number not in (0.0, 1.0):
            return None
        return int(number)

    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if protocol_flag(r.get('Fluid_Sensitive')) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None


def build_study(sid, ser_records, tsdir, reader):
    order_and_meta, read_px, mm_crop_resize = reader
    rows = ser_records.get(sid, [])
    vol = np.zeros((MAXS, IMG, IMG), np.uint8); idx = 0; used = set()
    for plane, fluid, k in SLOTS:
        r = _pick_series_for_slot(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r['SeriesInstanceUID'])
        files, med_ps = order_and_meta(f"{tsdir}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        # wide span: the collateral ligaments and lateral meniscus live in the
        # peripheral slices the old 0.15-0.85 crop threw away. Must match the corpus
        # the weights were trained on (knee_corpus_v2.py, SPAN_LO/SPAN_HI).
        n = len(files); lo, hi = int(n * 0.06), int(n * 0.94) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []; pss = []
        for p in picks:
            fp, ps = files[min(p, n - 1)]
            try:
                arrs.append(read_px(fp)); pss.append(ps)
            except Exception:
                arrs.append(None); pss.append(med_ps)
        valid = [a for a in arrs if a is not None]
        if valid:
            allpx = np.concatenate([a.ravel() for a in valid])
            loq, hiq = np.percentile(allpx, [2.0, 98.0])
        else:
            loq, hiq = 0.0, 1.0
        for a, ps in zip(arrs, pss):
            if idx >= MAXS: break
            if a is None: idx += 1; continue
            aw = np.clip((a - loq) / (hiq - loq + 1e-6), 0, 1)
            aw = mm_crop_resize(aw, ps if ps > 0 else med_ps)
            vol[idx] = (aw * 255).astype(np.uint8); idx += 1
        if idx >= MAXS: break
    mask = (vol.reshape(MAXS, -1).sum(1) > 0).astype(np.uint8)
    return vol, mask


# ============================================================================
# Test-root discovery + weights + main
# ============================================================================
def find_test_root():
    cands = ["/kaggle/input/competitions/rsna-knee-abnormality-detection",
             "/kaggle/input/rsna-knee-abnormality-detection"]
    for b in cands:
        if os.path.exists(b + "/test.csv"):
            return b
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f and (os.path.isdir(d + "/test_series") or os.path.isdir(d + "/test_images")):
            return d
    for d, _, f in os.walk("/kaggle/input"):
        if "test.csv" in f:
            return d
    raise RuntimeError("no test root under /kaggle/input")


def find_weight_file(fname):
    candidates = []
    pattern = fname
    for root in sorted(glob.glob("/kaggle/input/*/")):
        if "competition" in root.lower():
            continue
        candidates.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    candidates = sorted(set(candidates))
    if not candidates:
        raise RuntimeError(f"no E02 checkpoint matching {pattern} under /kaggle/input")
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one E02 checkpoint, got {candidates}")
    return candidates[0]


def main():
    import pandas as pd
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ngpu = torch.cuda.device_count()
    print(f"device {dev} | gpus {ngpu} | torch {torch.__version__}", flush=True)

    ROOT = find_test_root()
    tsdir = ROOT + "/test_series"
    if not os.path.isdir(tsdir):
        tsdir = ROOT + "/test_images"
    print("test root:", ROOT, "| series dir:", tsdir, flush=True)

    test = pd.read_csv(ROOT + "/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    test_ids = test["StudyInstanceUID"].tolist()
    tser = pd.read_csv(ROOT + "/test_series.csv")
    tser["StudyInstanceUID"] = tser["StudyInstanceUID"].astype(str)
    tser["SeriesInstanceUID"] = tser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in tser.groupby("StudyInstanceUID")}
    print(f"test studies {len(test_ids)} | test series {len(tser)}", flush=True)

    sub_cols = ["StudyInstanceUID"] + LAB
    ssub = os.path.join(ROOT, "sample_submission.csv")
    if os.path.exists(ssub):
        sub_cols = list(pd.read_csv(ssub, nrows=1).columns)

    reader = _make_reader()
    N = len(test_ids); A = len(ARMS)
    arm_probs = [np.full((N, len(LAB)), np.nan, np.float32) for _ in range(A)]
    checkpoint_info = []

    # Keep one checkpoint resident at a time.
    for a, arm in enumerate(ARMS):
        wp = find_weight_file(arm["file"])
        model, res, ck_info = load_model(wp, arm["res"], dev)
        checkpoint_info.append(ck_info)
        print(f"[arm {a}] loaded {arm['file']} | res {res} | {time.time()-t0:.0f}s", flush=True)
        for i, sid in enumerate(test_ids):
            try:
                vol, mask = build_study(sid, SER, tsdir, reader)
                xw = eval_windows(vol, mask, k=K_EVAL, res=res, norm=NORM)
                arm_probs[a][i] = infer_probs(model, xw, dev)
                del vol, mask, xw
            except Exception as e:
                raise RuntimeError(
                    f"E02 inference failed for study {i} {sid}: {type(e).__name__}: {e}"
                ) from e
            if (i + 1) % 100 == 0 or i + 1 == N:
                print(f"  [arm {a}] {i+1}/{N} | {time.time()-t0:.0f}s", flush=True)
        del model
        gc.collect()
        if str(dev).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[arm {a}] done + freed | {time.time()-t0:.0f}s", flush=True)

    # WEIGHTED rank-mean blend across the test set, per finding (the offline recipe).
    # Weights come from ARMS[*]["w"] and are normalised here, so dropping/adding an arm can
    # never silently change the scale. Falls back to equal weights if none are declared.
    _w = np.array([float(a.get("w", 1.0)) for a in ARMS], dtype=np.float64)
    _w = _w / _w.sum()
    print(f"[blend] weighted rank-mean w={dict(zip([a['file'] for a in ARMS], _w.round(4)))}", flush=True)
    ranks = np.tensordot(_w, np.stack([rankpct(np.clip(p, 0, 1)) for p in arm_probs]),
                         axes=(0, 0))                                          # (N,12) in [0,1]
    if not np.isfinite(ranks).all():
        raise RuntimeError("E02 ranked prediction matrix contains non-finite values")

    sub = pd.DataFrame(ranks.astype(np.float32), columns=LAB)
    sub.insert(0, "StudyInstanceUID", test_ids)
    sub = sub[sub_cols]
    assert list(sub.columns) == sub_cols, "column order drift"
    assert sub["StudyInstanceUID"].tolist() == test_ids, "row identity drift"
    assert np.isfinite(sub[LAB].values).all()
    out = os.environ.get("RSNA_E02_OUTPUT", "/kaggle/working/submission.csv")
    sub.to_csv(out, index=False)
    raw = np.asarray(arm_probs[0], dtype=np.float32)
    if not np.isfinite(raw).all():
        raise RuntimeError("E02 raw prediction matrix contains non-finite values")
    prediction_path = "/kaggle/working/e02_resnet34_predictions.npz"
    np.savez(prediction_path,
             study_uids=np.asarray(test_ids), raw_probabilities=raw,
             ranked_predictions=ranks.astype(np.float32), labels=np.asarray(LAB))
    def _artifact_sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as artifact:
            for block in iter(lambda: artifact.read(1 << 20), b""):
                digest.update(block)
        return digest.hexdigest()
    receipt = {
        "schema": "e02_resnet34_inference_v1",
        "status": "COMPLETE",
        "studies": N,
        "fallback_studies": 0,
        "input": {"cache_image_size": IMG, "network_image_size": NETWORK_IMG,
                  "crop_mm": CROP_MM,
                  "slots": SLOTS, "max_slices": MAXS, "eval_windows": K_EVAL,
                  "span": [0.06, 0.94], "normalization": "imagenet"},
        "checkpoint": checkpoint_info[0],
        "elapsed_seconds": round(time.time() - t0, 2),
        "output": out,
        "submission_sha256": _artifact_sha256(out),
        "predictions_sha256": _artifact_sha256(prediction_path),
    }
    with open("/kaggle/working/e02_resnet34_receipt.json", "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print("wrote", out, "|", len(sub), "rows x", len(sub.columns), "cols", flush=True)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
