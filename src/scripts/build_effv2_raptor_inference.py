"""Build the standalone hidden-test inference arm for the E01 trainer."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "public_kernels/dread_infer/knee-mri-twelve-findings-from-a-single-model.py"
OUTPUT = ROOT / "inference/effv2_raptor_arm.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"{label}: expected one source match, found {text.count(old)}")
    return text.replace(old, new, 1)


def build() -> None:
    text = SOURCE.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '"""Knee MRI: twelve findings from a single model',
        '"""Knee MRI: EfficientNetV2-S Raptor-corpus diversity arm',
        "title",
    )
    doc_start = text.index('"""')
    doc_end = text.index('"""', doc_start + 3) + 3
    honest_doc = '''"""Knee MRI: EfficientNetV2-S Raptor-corpus diversity arm.

This inference program serves the checkpoint produced by the E01 trainer. It
uses the audited 44-slice public Raptor layout, all 42 adjacent-slice windows,
336-pixel physical crops, and target-specific attention pooling. The best
available best-epoch or SWA checkpoint is selected from its held-out Gold58
metadata. Every test study must complete; an exception aborts the run instead
of publishing neutral fallback predictions.

E01 has no claimed public leaderboard score until its own standalone Kaggle
submission is graded. Scores and validation numbers from the upstream Raptor
source do not describe this EfficientNetV2 checkpoint.
"""'''
    text = text[:doc_start] + honest_doc + text[doc_end:]
    old_config = '''IMG = 336
CROP_MM = 140.0
# 64 slices per study instead of 44, same proportions. Must match the corpus the weights
# were trained on (knee_corpus_v4.py).
SLOTS = [("Sagittal", 1, 18), ("Sagittal", 0, 14), ("Coronal", 1, 12),
         ("Coronal", 0, 8), ("Axial", -1, 12)]
MAXS = sum(s[2] for s in SLOTS)                     # 64
K_EVAL = 42   # every window position the volume holds, not an evenly spaced subset
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
'''
    new_config = '''IMG = 336
CROP_MM = 140.0
# Exact public training-corpus layout: 12 + 10 + 8 + 6 + 8 = 44 slices.
SLOTS = [("Sagittal", 1, 12), ("Sagittal", 0, 10), ("Coronal", 1, 8),
         ("Coronal", 0, 6), ("Axial", -1, 8)]
MAXS = sum(s[2] for s in SLOTS)
K_EVAL = 42
NORM = "imagenet"
LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
       "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
# tf_efficientnetv2_s.in21k uses the timm TensorFlow-style [-1, 1] input transform.
_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
_STD = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
'''
    text = replace_once(text, old_config, new_config, "input contract")

    arms_comment_start = text.index("# Three arms:")
    arms_start = text.index("ARMS = [")
    arms_comment = '''# One independently trained E01 checkpoint family. The inference loader
# considers both the saved best epoch and top-three weight average, then selects
# the higher held-out Gold58 score. Public leaderboard performance remains
# unknown until the standalone code submission is graded.
'''
    text = text[:arms_comment_start] + arms_comment + text[arms_start:]
    arms_start = text.index("ARMS = [")
    arms_end = text.index("\n]\n", arms_start) + 3
    text = text[:arms_start] + '''ARMS = [
    {"file": "AUTO_EFFV2_RAPTOR", "arch": "tf_efficientnetv2_s.in21k",
     "res": 336, "w": 1.0},
]\n''' + text[arms_end:]

    old_load = '''def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    arch = ck.get("arch", arch_default)
    ck_res = int(ck.get("res", res_default))
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    # NOTE: DataParallel removed on purpose. On the full hidden test it drove a system-RAM OOM
    # (per-forward module replication over many studies); a single T4 handles K_EVAL=24 windows
    # fine. Arms are also run SEQUENTIALLY (see main) so peak RAM == one model, not two.
    del ck
    gc.collect()
    return model, ck_res
'''
    new_load = '''def validate_checkpoint_contract(ck, pt_path):
    required = {"model", "gold_auc", "aucs", "src", "res", "arch", "lab", "epoch"}
    missing = required - set(ck)
    if missing:
        raise RuntimeError(f"E01 checkpoint {pt_path} is missing {sorted(missing)}")
    if ck["arch"] != "tf_efficientnetv2_s.in21k" or int(ck["res"]) != 336:
        raise RuntimeError(
            f"E01 checkpoint contract mismatch: arch={ck['arch']!r} res={ck['res']!r}"
        )
    if list(ck["lab"]) != LAB or set(ck["aucs"]) != set(LAB):
        raise RuntimeError("E01 checkpoint target order or per-target metrics are invalid")
    score = float(ck["gold_auc"])
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError(f"E01 checkpoint has invalid Gold58 score {score!r}")
    if not isinstance(ck["model"], dict) or len(ck["model"]) < 100:
        raise RuntimeError("E01 checkpoint model state is empty or implausibly small")
    if pt_path.endswith("_swa.pt"):
        epochs = [int(value) for value in ck.get("swa_over", [])]
        if epochs != [int(value) for value in ck["epoch"]] or not 2 <= len(epochs) <= 3:
            raise RuntimeError(f"E01 SWA checkpoint has invalid epoch receipt {epochs}")
    elif not 0 <= int(ck["epoch"]) < 12:
        raise RuntimeError(f"E01 checkpoint has invalid epoch {ck['epoch']!r}")
    return score


def load_model(pt_path, arch_default, res_default, device, ngpu=1):
    ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    validate_checkpoint_contract(ck, pt_path)
    arch = ck.get("arch", arch_default)
    ck_res = int(ck.get("res", res_default))
    info = {"checkpoint": os.path.basename(pt_path), "arch": arch, "res": ck_res,
            "gold_auc": ck.get("gold_auc"), "per_target_auc": ck.get("aucs", {}),
            "epoch": ck.get("epoch")}
    bb = build_backbone(arch, pretrained=False)
    model = RaptorClassifier(bb, F_dim=bb.num_features)
    model.load_state_dict(ck["model"], strict=True)
    model.eval().to(device)
    del ck
    gc.collect()
    return model, ck_res, info
'''
    text = replace_once(text, old_load, new_load, "checkpoint loader")

    finder_start = text.index("def find_weight_file(fname):")
    finder_end = text.index("\n\ndef main():", finder_start)
    finder = '''def find_weight_file(fname):
    # Select the best available released checkpoint using its held-out Gold58 score.
    candidates = []
    pattern = "raptor_ft_effv2s_336_k10_consensus_v1*.pt"
    for root in sorted(glob.glob("/kaggle/input/*/")):
        if "competition" in root.lower():
            continue
        candidates.extend(glob.glob(os.path.join(root, "**", pattern), recursive=True))
    candidates = sorted(set(candidates))
    if not candidates:
        raise RuntimeError(f"no E01 checkpoint matching {pattern} under /kaggle/input")
    preference = os.environ.get("RSNA_EFFV2_CHECKPOINT", "auto").strip()
    if preference != "auto":
        selected = [p for p in candidates if os.path.basename(p) == preference]
        if len(selected) != 1:
            raise RuntimeError(f"requested E01 checkpoint {preference!r}, got {selected}")
        return selected[0]
    scored = []
    for path in candidates:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        score = validate_checkpoint_contract(ck, path)
        del ck
        gc.collect()
        scored.append((score, path))
    if not scored:
        raise RuntimeError(f"E01 checkpoints have no finite Gold58 score: {candidates}")
    scored.sort(key=lambda item: (item[0], "_swa.pt" in item[1]), reverse=True)
    print("E01 checkpoint candidates:", [(round(s, 6), os.path.basename(p)) for s, p in scored],
          flush=True)
    return scored[0][1]
'''
    text = text[:finder_start] + finder + text[finder_end:]

    text = replace_once(
        text,
        '    arm_probs = [np.full((N, len(LAB)), 0.5, np.float32) for _ in range(A)]\n',
        '    arm_probs = [np.full((N, len(LAB)), np.nan, np.float32) for _ in range(A)]\n'
        '    checkpoint_info = []\n',
        "prediction initialization",
    )
    text = replace_once(
        text,
        '        model, res = load_model(wp, arm["arch"], arm["res"], dev)\n',
        '        model, res, ck_info = load_model(wp, arm["arch"], arm["res"], dev)\n'
        '        checkpoint_info.append(ck_info)\n',
        "model invocation",
    )
    old_except = '''            except Exception as e:
                print(f"  [arm {a}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
'''
    new_except = '''            except Exception as e:
                raise RuntimeError(
                    f"E01 inference failed for study {i} {sid}: {type(e).__name__}: {e}"
                ) from e
'''
    text = replace_once(text, old_except, new_except, "fallback policy")

    old_output = '''    out = "/kaggle/working/submission.csv"
    sub.to_csv(out, index=False)
    print("wrote", out, "|", len(sub), "rows x", len(sub.columns), "cols", flush=True)
    print(sub.head().to_string(index=False), flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)
'''
    new_output = '''    out = os.environ.get("RSNA_EFFV2_OUTPUT", "/kaggle/working/_effv2_arm.csv")
    sub.to_csv(out, index=False)
    raw = np.asarray(arm_probs[0], dtype=np.float32)
    if not np.isfinite(raw).all():
        raise RuntimeError("E01 raw prediction matrix contains non-finite values")
    prediction_path = "/kaggle/working/effv2_raptor_predictions.npz"
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
        "schema": "effv2_raptor_inference_v1",
        "status": "COMPLETE",
        "studies": N,
        "fallback_studies": 0,
        "input": {"image_size": IMG, "crop_mm": CROP_MM,
                  "slots": SLOTS, "max_slices": MAXS, "eval_windows": K_EVAL,
                  "span": [0.06, 0.94], "normalization": "mean_std_0.5"},
        "checkpoint": checkpoint_info[0],
        "elapsed_seconds": round(time.time() - t0, 2),
        "output": out,
        "submission_sha256": _artifact_sha256(out),
        "predictions_sha256": _artifact_sha256(prediction_path),
    }
    with open("/kaggle/working/effv2_raptor_receipt.json", "w") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
    print("wrote", out, "|", len(sub), "rows x", len(sub.columns), "cols", flush=True)
    print(json.dumps(receipt, indent=2), flush=True)
'''
    text = replace_once(text, old_output, new_output, "artifact output")
    text = replace_once(
        text,
        '''    if not np.isfinite(ranks).all():
        ranks[~np.isfinite(ranks)] = 0.5
''',
        '''    if not np.isfinite(ranks).all():
        raise RuntimeError("E01 ranked prediction matrix contains non-finite values")
''',
        "non-finite rank policy",
    )
    text = replace_once(
        text,
        "import os, sys, glob, time, json, gc\n",
        "import os, sys, glob, time, json, gc, hashlib\n",
        "hash import",
    )

    old_series_picker = '''def _pick_series_for_slot(rows, plane, fluid, used):
    cands = [r for r in rows if r['Anatomical_Plane'] == plane and r['SeriesInstanceUID'] not in used]
    if fluid in (0, 1):
        pref = [r for r in cands if int(r.get('Fluid_Sensitive', 0) or 0) == fluid]
        if pref:
            return pref[0]
    return cands[0] if cands else None
'''
    robust_series_picker = '''def _pick_series_for_slot(rows, plane, fluid, used):
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
'''
    text = replace_once(text, old_series_picker, robust_series_picker,
                        "tri-state protocol flag parser")

    text = text.replace(
        "# the fixed (1,24,3,res,res) input.",
        "# the fixed one-study window batch.",
    )
    text = text.replace(
        '''    # SEQUENTIAL ARMS (the OOM fix): only ONE model is resident at a time, so peak system RAM ==
    # one model == the single-arm champion's footprint (which graded fine at 0.879). Holding both
    # arms simultaneously OOM'd system RAM on the full hidden test. Each study is re-preprocessed
    # per arm (build_study is cheap vs inference) and every per-study buffer is freed. Same models,
    # same windowing, same rank-mean blend -> identical 0.8893 result, just serialized.
''',
        '''    # Keep one checkpoint resident at a time. E01 currently has one arm, and this
    # structure preserves a bounded memory contract if a validated arm is added later.
''',
    )

    required = (
        "tf_efficientnetv2_s.in21k",
        "MAXS = sum(s[2] for s in SLOTS)",
        "K_EVAL = 42",
        "mean_std_0.5",
        "fallback_studies\": 0",
        "effv2_raptor_predictions.npz",
        "submission_sha256",
        "def protocol_flag(value):",
    )
    for token in required:
        if token not in text:
            raise RuntimeError(f"generated E01 inference is missing {token}")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    build()
