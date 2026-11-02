"""Notebook cell injected before the B01 final publish step.

The M01 program runs in a fresh subprocess so its model definitions and global
configuration cannot alter the large public ensemble. Its ranked predictions
are then blended into the staged B01 output using a conservative Gold58 gate.
This notebook is staged only; leaderboard evidence decides whether it is run.
"""

import hashlib as _m3_hashlib
import json as _m3_json
import os as _m3_os
import subprocess as _m3_subprocess
import sys as _m3_sys
import time as _m3_time
from pathlib import Path as _M3Path

import numpy as _m3_np
import pandas as _m3_pd


_M3_LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
_m3_started = _m3_time.time()
_m3_program = _M3Path("/kaggle/working/_medicalnet3d_arm.py")
_m3_arm_path = _M3Path("/kaggle/working/_medicalnet3d_arm.csv")
_m3_log_path = _M3Path("/kaggle/working/medicalnet3d_blend_runtime.log")
_m3_receipt_path = _M3Path("/kaggle/working/medicalnet3d_receipt.json")
_m3_stage_path = _M3Path("/kaggle/working/_pipeline_stage.csv")

if not _m3_program.is_file():
    raise FileNotFoundError(f"M01 inference program was not written: {_m3_program}")
if not _m3_stage_path.is_file():
    raise FileNotFoundError(f"B01 staged output is missing: {_m3_stage_path}")

_m3_environment = dict(_m3_os.environ)
_m3_environment["RSNA_MEDICALNET3D_OUTPUT"] = str(_m3_arm_path)
_m3_environment["CUDA_VISIBLE_DEVICES"] = "0"
_m3_remaining = 8.75 * 3600
if "T0" in globals():
    _m3_remaining -= _m3_time.time() - T0
if _m3_remaining <= 60:
    raise TimeoutError("No complete-run budget remains for the required M01 arm")

with _m3_log_path.open("w", encoding="utf-8") as _m3_log:
    _m3_result = _m3_subprocess.run(
        [_m3_sys.executable, str(_m3_program)],
        env=_m3_environment,
        stdout=_m3_log,
        stderr=_m3_subprocess.STDOUT,
        timeout=_m3_remaining,
        check=False,
    )
if _m3_result.returncode != 0:
    _m3_tail = _m3_log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    raise RuntimeError(
        f"Required M01 subprocess failed with code {_m3_result.returncode}:\n{_m3_tail}"
    )
print(_m3_log_path.read_text(encoding="utf-8", errors="replace")[-3000:], flush=True)

if not _m3_receipt_path.is_file() or not _m3_arm_path.is_file():
    raise RuntimeError("M01 subprocess did not publish its arm and receipt")
_m3_receipt = _m3_json.loads(_m3_receipt_path.read_text(encoding="utf-8"))
if (
    _m3_receipt.get("schema") != "medicalnet3d_inference_v1"
    or _m3_receipt.get("status") != "COMPLETE"
    or int(_m3_receipt.get("fallback_studies", -1)) != 0
):
    raise RuntimeError(f"M01 receipt is not a complete zero-fallback run: {_m3_receipt}")

_m3_competition = None
for _m3_candidate in (
    _M3Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    _M3Path("/kaggle/input/rsna-knee-abnormality-detection"),
):
    if (_m3_candidate / "test.csv").is_file():
        _m3_competition = _m3_candidate
        break
if _m3_competition is None:
    raise FileNotFoundError("RSNA Knee test input is not mounted")
_m3_test_ids = _m3_pd.read_csv(
    _m3_competition / "test.csv", dtype={"StudyInstanceUID": str}
).StudyInstanceUID.astype(str).tolist()
if len(_m3_test_ids) == 0 or len(_m3_test_ids) != len(set(_m3_test_ids)):
    raise RuntimeError("Invalid hidden-test identity list")

_m3_base = rsna_frame(
    _m3_pd.read_csv(_m3_stage_path, dtype={"StudyInstanceUID": str}),
    _m3_test_ids,
    _M3_LABELS,
    "pre-M01 B01 stage",
)
_m3_arm = rsna_frame(
    _m3_pd.read_csv(_m3_arm_path, dtype={"StudyInstanceUID": str}),
    _m3_test_ids,
    _M3_LABELS,
    "M01 standalone arm",
)
if int(_m3_receipt.get("studies", -1)) != len(_m3_test_ids):
    raise RuntimeError("M01 receipt study count does not match hidden test")


def _m3_rank01(values):
    return _m3_pd.DataFrame(values).rank(method="average", pct=True).to_numpy(_m3_np.float64)


_m3_base_rank = _m3_rank01(_m3_base[_M3_LABELS].to_numpy(_m3_np.float64))
_m3_arm_rank = _m3_rank01(_m3_arm[_M3_LABELS].to_numpy(_m3_np.float64))
_m3_target_auc = _m3_receipt.get("checkpoint_info", {}).get("gold_target_auc", {})
if set(_m3_target_auc) != set(_M3_LABELS):
    raise RuntimeError("M01 checkpoint receipt lacks the exact 12 Gold58 target metrics")

# Gold58 is too small for fine weight fitting. It is used only as a coarse
# rejection gate, with a maximum M01 contribution of eight percent.
_m3_beta = []
for _m3_label in _M3_LABELS:
    _m3_auc = float(_m3_target_auc[_m3_label])
    if not _m3_np.isfinite(_m3_auc) or _m3_auc < 0.70:
        _m3_weight = 0.0
    else:
        _m3_weight = float(_m3_np.clip(0.03 + 0.20 * (_m3_auc - 0.70), 0.03, 0.08))
    _m3_beta.append(_m3_weight)
_m3_beta = _m3_np.asarray(_m3_beta, _m3_np.float64)
if (_m3_beta < 0).any() or (_m3_beta > 0.08).any() or not (_m3_beta > 0).any():
    raise RuntimeError(f"M01 blend gate rejected all targets or emitted unsafe weights: {_m3_beta}")

_m3_blended = _m3_rank01(
    (1.0 - _m3_beta[None, :]) * _m3_base_rank
    + _m3_beta[None, :] * _m3_arm_rank
)
if not _m3_np.isfinite(_m3_blended).all():
    raise RuntimeError("B01+M01 blend contains non-finite values")
_m3_output = _m3_pd.DataFrame(_m3_blended.astype(_m3_np.float32), columns=_M3_LABELS)
_m3_output.insert(0, "StudyInstanceUID", _m3_test_ids)
_m3_tmp = _m3_stage_path.with_suffix(".csv.m01.tmp")
_m3_output.to_csv(_m3_tmp, index=False)
_m3_os.replace(_m3_tmp, _m3_stage_path)


def _m3_sha256(path):
    _m3_digest = _m3_hashlib.sha256()
    with _M3Path(path).open("rb") as _m3_handle:
        for _m3_block in iter(lambda: _m3_handle.read(1 << 20), b""):
            _m3_digest.update(_m3_block)
    return _m3_digest.hexdigest()


_m3_blend_receipt = {
    "schema": "dinov5_medicalnet3d_diversity_blend_v1",
    "status": "COMPLETE",
    "studies": len(_m3_test_ids),
    "fallback_studies": 0,
    "checkpoint": _m3_receipt["checkpoint"],
    "per_target_gold58_auc": _m3_target_auc,
    "per_target_beta": dict(zip(_M3_LABELS, _m3_beta.tolist())),
    "beta_cap": 0.08,
    "arm_sha256": _m3_sha256(_m3_arm_path),
    "output_sha256": _m3_sha256(_m3_stage_path),
    "upstream_receipt_sha256": _m3_sha256(_m3_receipt_path),
    "elapsed_seconds": round(_m3_time.time() - _m3_started, 2),
}
rsna_json("/kaggle/working/diagnostics/medicalnet3d_blend_receipt.json", _m3_blend_receipt)
rsna_save_predictions("medicalnet3d_arm", _m3_test_ids, _m3_arm_rank, _M3_LABELS)
print("[medicalnet3d] conservative diversity blend complete", _m3_blend_receipt, flush=True)
