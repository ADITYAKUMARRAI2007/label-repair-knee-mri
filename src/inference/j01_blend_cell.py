"""Rank-blend the independent J01 arm into the staged 0.943 B01 prediction."""

import hashlib as _j1_hashlib
import json as _j1_json
import os as _j1_os
import subprocess as _j1_subprocess
import sys as _j1_sys
import time as _j1_time
from pathlib import Path as _J1Path

import numpy as _j1_np
import pandas as _j1_pd

_J1_LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
_J1_BETA = __J01_BETA__
_j1_started = _j1_time.time()
_j1_program = _J1Path("/kaggle/working/_j01_arm.py")
_j1_arm_path = _J1Path("/kaggle/working/_j01_arm.csv")
_j1_log_path = _J1Path("/kaggle/working/j01_blend_runtime.log")
_j1_receipt_path = _J1Path("/kaggle/working/j01_inference_receipt.json")
_j1_stage_path = _J1Path("/kaggle/working/_pipeline_stage.csv")

if not (0 < _J1_BETA <= 0.20):
    raise RuntimeError(f"unsafe J01 blend weight {_J1_BETA}")
if not _j1_program.is_file() or not _j1_stage_path.is_file():
    raise FileNotFoundError("J01 program or B01 staged prediction is missing")

_j1_env = dict(_j1_os.environ)
_j1_env["RSNA_J01_OUTPUT"] = str(_j1_arm_path)
_j1_env["CUDA_VISIBLE_DEVICES"] = "0"
_j1_remaining = 8.75 * 3600
if "T0" in globals():
    _j1_remaining -= _j1_time.time() - T0
if _j1_remaining <= 60:
    raise TimeoutError("no runtime budget remains for J01")
with _j1_log_path.open("w", encoding="utf-8") as _j1_log:
    _j1_result = _j1_subprocess.run(
        [_j1_sys.executable, str(_j1_program)], env=_j1_env,
        stdout=_j1_log, stderr=_j1_subprocess.STDOUT,
        timeout=_j1_remaining, check=False,
    )
if _j1_result.returncode != 0:
    tail = _j1_log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    raise RuntimeError(f"J01 subprocess failed with code {_j1_result.returncode}:\n{tail}")
print(_j1_log_path.read_text(encoding="utf-8", errors="replace")[-3000:], flush=True)

if not _j1_receipt_path.is_file() or not _j1_arm_path.is_file():
    raise RuntimeError("J01 did not publish its arm and receipt")
_j1_receipt = _j1_json.loads(_j1_receipt_path.read_text())
if _j1_receipt.get("status") != "COMPLETE" or int(_j1_receipt.get("fallback_studies", -1)) != 0:
    raise RuntimeError(f"J01 incomplete receipt: {_j1_receipt}")
_j1_gold_auc = float(_j1_receipt.get("checkpoint", {}).get("gold58_auc", float("nan")))
if not _j1_np.isfinite(_j1_gold_auc) or _j1_gold_auc < 0.88:
    raise RuntimeError(f"J01 failed preregistered Gold58 gate: {_j1_gold_auc}")

_j1_competition = None
for _j1_candidate in (
    _J1Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    _J1Path("/kaggle/input/rsna-knee-abnormality-detection"),
):
    if (_j1_candidate / "test.csv").is_file():
        _j1_competition = _j1_candidate
        break
if _j1_competition is None:
    raise FileNotFoundError("RSNA Knee test input is not mounted")
_j1_test_ids = _j1_pd.read_csv(
    _j1_competition / "test.csv", dtype={"StudyInstanceUID": str}
).StudyInstanceUID.astype(str).tolist()
_j1_base = rsna_frame(
    _j1_pd.read_csv(_j1_stage_path, dtype={"StudyInstanceUID": str}),
    _j1_test_ids, _J1_LABELS, "pre-J01 B01 stage",
)
_j1_arm = rsna_frame(
    _j1_pd.read_csv(_j1_arm_path, dtype={"StudyInstanceUID": str}),
    _j1_test_ids, _J1_LABELS, "J01 standalone arm",
)
if int(_j1_receipt.get("studies", -1)) != len(_j1_test_ids):
    raise RuntimeError("J01 study count differs from hidden test")


def _j1_rank01(values):
    return _j1_pd.DataFrame(values).rank(method="average", pct=True).to_numpy(_j1_np.float64)


_j1_base_rank = _j1_rank01(_j1_base[_J1_LABELS].to_numpy(_j1_np.float64))
_j1_arm_rank = _j1_rank01(_j1_arm[_J1_LABELS].to_numpy(_j1_np.float64))
_j1_blended = _j1_rank01((1.0 - _J1_BETA) * _j1_base_rank + _J1_BETA * _j1_arm_rank)
if not _j1_np.isfinite(_j1_blended).all():
    raise RuntimeError("B01+J01 blend contains non-finite values")
_j1_output = _j1_pd.DataFrame(_j1_blended.astype(_j1_np.float32), columns=_J1_LABELS)
_j1_output.insert(0, "StudyInstanceUID", _j1_test_ids)
_j1_tmp = _j1_stage_path.with_suffix(".csv.j01.tmp")
_j1_output.to_csv(_j1_tmp, index=False)
_j1_os.replace(_j1_tmp, _j1_stage_path)


def _j1_sha256(path):
    digest = _j1_hashlib.sha256()
    with _J1Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


_j1_blend_receipt = {
    "schema": "dinov5_j01_rank_blend_v1", "status": "COMPLETE",
    "studies": len(_j1_test_ids), "fallback_studies": 0,
    "base_public_score": 0.943, "global_beta": _J1_BETA,
    "gold58_auc": _j1_gold_auc,
    "arm_sha256": _j1_sha256(_j1_arm_path),
    "output_sha256": _j1_sha256(_j1_stage_path),
    "upstream_receipt_sha256": _j1_sha256(_j1_receipt_path),
    "elapsed_seconds": round(_j1_time.time() - _j1_started, 2),
}
rsna_json("/kaggle/working/diagnostics/j01_blend_receipt.json", _j1_blend_receipt)
rsna_save_predictions("j01_arm", _j1_test_ids, _j1_arm_rank, _J1_LABELS)
print("[j01] B01 rank blend complete", _j1_blend_receipt, flush=True)
