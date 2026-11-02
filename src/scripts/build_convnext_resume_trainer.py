#!/usr/bin/env python3
"""Build a fail-closed T01 resume kernel from the completed V2 adaptation state.

The original V2 run finished preprocessing and all three adaptation epochs, then
hit a Python variable-name collision after writing the first feature wave.  This
builder removes the already-completed adaptation block, strictly loads the
audited backbone, restores the exact first 80 feature files, and continues with
feature extraction plus five-fold head training.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "public_kernels/dinosaur_v5_train/rsna-knee-dinosaur-v5-train.py"
OUT_DIR = ROOT / "public_kernels/convnext_diversity_resume"
OUTPUT = OUT_DIR / "rsna-knee-convnext-diversity-resume.py"

START = "# ----------------------- balanced pseudo-label source audit -------------------"
END = "# ----------------------- stage B: cached study features -----------------------"
BACKBONE_SHA = "bec04468edeaaa0c75191a221d4e3b80bada5cfd263eeec36cb093548de93147"
FEATURE_SHA = "01470d97ff6f8f0e2321d8cd9e9489ba40eeb151f097af1d0c02858640be6227"


RECOVERY = rf'''
# ---------------------- audited V2 recovery checkpoint ------------------------
import hashlib as _recovery_hashlib
import shutil as _recovery_shutil

RECOVERY_BACKBONE_SHA256 = "{BACKBONE_SHA}"
RECOVERY_FEATURES_SHA256 = "{FEATURE_SHA}"

def _recovery_sha256(path):
    digest=_recovery_hashlib.sha256()
    with open(path,"rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):
            digest.update(block)
    return digest.hexdigest()

backbone_hits=sorted(
    Path(p) for p in glob.glob(
        "/kaggle/input/**/dinov5_convnext_backbone.pt",recursive=True
    ) if Path(p).is_file()
)
if len(backbone_hits)!=1:
    raise RuntimeError(
        f"expected exactly one recovery backbone, found {{backbone_hits}}"
    )
RECOVERY_ROOT=backbone_hits[0].parent
RECOVERY_PROVENANCE={{
    "schema":"rsna_knee_convnext_recovery_v1",
    "source_kernel":"adityakumarrai0401/rsna-knee-convnext-diversity-train-v2",
    "source_status":"ERROR_AFTER_FIRST_FEATURE_WAVE",
    "failure":"Popen variable shadowed psutil Process handle at RSS check",
    "adaptation_epochs_completed":3,
    "adaptation_losses":[0.5605,0.5348,0.5141],
    "adaptation_fallback_triplets":0,
    "feature_prefix":{{
        "studies":80,
        "expected_prefix_studies":80,
        "exact_first_80_ids":True,
    }},
}}
backbone_hash=_recovery_sha256(backbone_hits[0])
if backbone_hash!=RECOVERY_BACKBONE_SHA256:
    raise RuntimeError(
        f"recovery backbone hash mismatch: {{backbone_hash}}"
    )
recovery_state=torch.load(
    backbone_hits[0],map_location="cpu",weights_only=True
)
if not isinstance(recovery_state,dict) or len(recovery_state)<100:
    raise RuntimeError("recovery backbone state is empty or implausibly small")
encoder.load_state_dict(recovery_state,strict=True)
del recovery_state
torch.save(encoder.state_dict(),WORK/"dinov5_convnext_backbone.pt")

# Restore only a contiguous, receipt-backed prefix.  Starting mid-wave would
# make the worker schedule ambiguous, so the prefix must be exactly 80 studies.
seed_dir=RECOVERY_ROOT/"dinov5_convnext_features"
if not seed_dir.is_dir():
    seed_dir=RECOVERY_ROOT
seed_files=sorted(seed_dir.glob("*.npz"))
expected_seed_ids=UIDS[:80]
if {{p.stem for p in seed_files}}!=set(expected_seed_ids):
    raise RuntimeError(
        f"recovery feature IDs are not the exact first 80 studies: {{len(seed_files)}}"
    )
feature_digest=_recovery_hashlib.sha256()
for path in seed_files:
    feature_digest.update(path.name.encode())
    feature_digest.update(bytes.fromhex(_recovery_sha256(path)))
if feature_digest.hexdigest()!=RECOVERY_FEATURES_SHA256:
    raise RuntimeError("recovery feature aggregate hash mismatch")
for uid in expected_seed_ids:
    source=seed_dir/f"{{uid}}.npz"
    with np.load(source,allow_pickle=False) as payload:
        feat=np.asarray(payload["feat"])
        meta=np.asarray(payload["meta"])
        if (
            feat.ndim!=2 or feat.shape[1]!=FEATURE_DIM
            or meta.shape!=(len(feat),6)
            or not np.isfinite(feat).all() or not np.isfinite(meta).all()
        ):
            raise RuntimeError(f"invalid recovery feature file {{source}}")
    _recovery_shutil.copy2(source,FEATURE_CACHE/source.name)

FEATURE_START=80
RECOVERY_PROVENANCE["backbone_sha256_verified"]=backbone_hash
RECOVERY_PROVENANCE["feature_prefix_sha256_verified"]=feature_digest.hexdigest()
RECOVERY_PROVENANCE["resume_feature_start"]=FEATURE_START
(WORK/"dinov5_convnext_recovery_receipt.json").write_text(
    json.dumps(RECOVERY_PROVENANCE,indent=2,sort_keys=True)+"\n"
)
print(
    "recovery checkpoint loaded:",backbone_hits[0],
    "seeded_features=",FEATURE_START,flush=True
)

try:
    import psutil as _recovery_psutil
    rss_process=_recovery_psutil.Process(os.getpid())
    def rss_gb():
        return rss_process.memory_info().rss/(1024**3)
except Exception:
    def rss_gb():
        return float("nan")

'''


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    if source.count(START) != 1 or source.count(END) != 1:
        raise RuntimeError("ConvNeXt trainer stage markers drifted")
    prefix, tail = source.split(START, 1)
    _, stage_b = tail.split(END, 1)
    resumed = prefix + RECOVERY + END + stage_b

    old_loop = "for wave_start in range(0,len(UIDS),wave):"
    new_loop = "for wave_start in range(FEATURE_START,len(UIDS),wave):"
    if resumed.count(old_loop) != 1:
        raise RuntimeError("feature-wave loop contract drifted")
    resumed = resumed.replace(old_loop, new_loop, 1)

    old_manifest = '    "seed":SEED,\n'
    new_manifest = old_manifest + '    "recovery":RECOVERY_PROVENANCE,\n'
    if resumed.count(old_manifest) != 1:
        raise RuntimeError("manifest insertion point drifted")
    resumed = resumed.replace(old_manifest, new_manifest, 1)

    # The V2 collision must be absent from the generated recovery source.
    if "proc=psutil.Process" in resumed or "processes.append(proc)" in resumed:
        raise RuntimeError("subprocess/RSS variable collision survived recovery build")
    for token in (
        "RECOVERY_BACKBONE_SHA256",
        "encoder.load_state_dict(recovery_state,strict=True)",
        "FEATURE_START=80",
        "range(FEATURE_START,len(UIDS),wave)",
        '"recovery":RECOVERY_PROVENANCE',
        "dinov5_convnext_gold58_oof.csv",
    ):
        if token not in resumed:
            raise RuntimeError(f"resume source is missing contract token {token!r}")
    ast.parse(resumed)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(resumed, encoding="utf-8")
    metadata = {
        "id": "adityakumarrai0401/rsna-knee-convnext-diversity-resume-v3",
        "title": "RSNA Knee ConvNeXt Diversity Resume V3",
        "code_file": OUTPUT.name,
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
            "adityakumarrai0401/rsna-knee-convnext-diversity-train-v2"
        ],
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": [
            "ambrosm/convnext-tiny-fb-in22k-ft-in1k/PyTorch/default/1"
        ],
        "docker_image": (
            "gcr.io/kaggle-private-byod/python@sha256:"
            "37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
        ),
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
