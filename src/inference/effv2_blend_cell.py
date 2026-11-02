"""Notebook cell injected before the B01 final publish step.

The E01 program runs in a fresh subprocess so its model definitions and global
configuration cannot alter the large public ensemble. Its ranked predictions
are then blended into the staged B01 output using a conservative Gold58 gate.
This notebook is staged only; leaderboard evidence decides whether it is run.
"""

import hashlib as _e2_hashlib
import json as _e2_json
import os as _e2_os
import subprocess as _e2_subprocess
import sys as _e2_sys
import time as _e2_time
from pathlib import Path as _E2Path

import numpy as _e2_np
import pandas as _e2_pd


_E2_LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]
_e2_started = _e2_time.time()
_e2_program = _E2Path("/kaggle/working/_effv2_raptor_arm.py")
_e2_arm_path = _E2Path("/kaggle/working/_effv2_arm.csv")
_e2_log_path = _E2Path("/kaggle/working/effv2_blend_runtime.log")
_e2_receipt_path = _E2Path("/kaggle/working/effv2_raptor_receipt.json")
_e2_stage_path = _E2Path("/kaggle/working/_pipeline_stage.csv")

if not _e2_program.is_file():
    raise FileNotFoundError(f"E01 inference program was not written: {_e2_program}")
if not _e2_stage_path.is_file():
    raise FileNotFoundError(f"B01 staged output is missing: {_e2_stage_path}")

_e2_environment = dict(_e2_os.environ)
_e2_environment["RSNA_EFFV2_OUTPUT"] = str(_e2_arm_path)
_e2_environment["CUDA_VISIBLE_DEVICES"] = "0"
_e2_remaining = 8.75 * 3600
if "T0" in globals():
    _e2_remaining -= _e2_time.time() - T0
if _e2_remaining <= 60:
    raise TimeoutError("No complete-run budget remains for the required E01 arm")

with _e2_log_path.open("w", encoding="utf-8") as _e2_log:
    _e2_result = _e2_subprocess.run(
        [_e2_sys.executable, str(_e2_program)],
        env=_e2_environment,
        stdout=_e2_log,
        stderr=_e2_subprocess.STDOUT,
        timeout=_e2_remaining,
        check=False,
    )
if _e2_result.returncode != 0:
    _e2_tail = _e2_log_path.read_text(encoding="utf-8", errors="replace")[-6000:]
    raise RuntimeError(
        f"Required E01 subprocess failed with code {_e2_result.returncode}:\n{_e2_tail}"
    )
print(_e2_log_path.read_text(encoding="utf-8", errors="replace")[-3000:], flush=True)

if not _e2_receipt_path.is_file() or not _e2_arm_path.is_file():
    raise RuntimeError("E01 subprocess did not publish its arm and receipt")
_e2_receipt = _e2_json.loads(_e2_receipt_path.read_text(encoding="utf-8"))
if _e2_receipt.get("status") != "COMPLETE" or int(_e2_receipt.get("fallback_studies", -1)) != 0:
    raise RuntimeError(f"E01 receipt is not a complete zero-fallback run: {_e2_receipt}")

_e2_competition = None
for _e2_candidate in (
    _E2Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    _E2Path("/kaggle/input/rsna-knee-abnormality-detection"),
):
    if (_e2_candidate / "test.csv").is_file():
        _e2_competition = _e2_candidate
        break
if _e2_competition is None:
    raise FileNotFoundError("RSNA Knee test input is not mounted")
_e2_test_ids = _e2_pd.read_csv(
    _e2_competition / "test.csv", dtype={"StudyInstanceUID": str}
).StudyInstanceUID.astype(str).tolist()
if len(_e2_test_ids) == 0 or len(_e2_test_ids) != len(set(_e2_test_ids)):
    raise RuntimeError("Invalid hidden-test identity list")

_e2_base = rsna_frame(
    _e2_pd.read_csv(_e2_stage_path, dtype={"StudyInstanceUID": str}),
    _e2_test_ids,
    _E2_LABELS,
    "pre-E01 B01 stage",
)
_e2_arm = rsna_frame(
    _e2_pd.read_csv(_e2_arm_path, dtype={"StudyInstanceUID": str}),
    _e2_test_ids,
    _E2_LABELS,
    "E01 standalone arm",
)
if int(_e2_receipt.get("studies", -1)) != len(_e2_test_ids):
    raise RuntimeError("E01 receipt study count does not match hidden test")


def _e2_rank01(values):
    return _e2_pd.DataFrame(values).rank(method="average", pct=True).to_numpy(_e2_np.float64)


_e2_base_rank = _e2_rank01(_e2_base[_E2_LABELS].to_numpy(_e2_np.float64))
_e2_arm_rank = _e2_rank01(_e2_arm[_E2_LABELS].to_numpy(_e2_np.float64))
_e2_target_auc = _e2_receipt.get("checkpoint", {}).get("per_target_auc", {})
if set(_e2_target_auc) != set(_E2_LABELS):
    raise RuntimeError("E01 checkpoint receipt lacks the exact 12 Gold58 target metrics")

# Gold58 is too small for fine weight fitting. It is used only as a coarse
# rejection gate, with a maximum E01 contribution of eight percent.
_e2_beta = []
for _e2_label in _E2_LABELS:
    _e2_auc = float(_e2_target_auc[_e2_label])
    if not _e2_np.isfinite(_e2_auc) or _e2_auc < 0.70:
        _e2_weight = 0.0
    else:
        _e2_weight = float(_e2_np.clip(0.03 + 0.20 * (_e2_auc - 0.70), 0.03, 0.08))
    _e2_beta.append(_e2_weight)
_e2_beta = _e2_np.asarray(_e2_beta, _e2_np.float64)
if (_e2_beta < 0).any() or (_e2_beta > 0.08).any() or not (_e2_beta > 0).any():
    raise RuntimeError(f"E01 blend gate rejected all targets or emitted unsafe weights: {_e2_beta}")

_e2_blended = _e2_rank01(
    (1.0 - _e2_beta[None, :]) * _e2_base_rank
    + _e2_beta[None, :] * _e2_arm_rank
)
if not _e2_np.isfinite(_e2_blended).all():
    raise RuntimeError("B01+E01 blend contains non-finite values")
_e2_output = _e2_pd.DataFrame(_e2_blended.astype(_e2_np.float32), columns=_E2_LABELS)
_e2_output.insert(0, "StudyInstanceUID", _e2_test_ids)
_e2_tmp = _e2_stage_path.with_suffix(".csv.e01.tmp")
_e2_output.to_csv(_e2_tmp, index=False)
_e2_os.replace(_e2_tmp, _e2_stage_path)


def _e2_sha256(path):
    _e2_digest = _e2_hashlib.sha256()
    with _E2Path(path).open("rb") as _e2_handle:
        for _e2_block in iter(lambda: _e2_handle.read(1 << 20), b""):
            _e2_digest.update(_e2_block)
    return _e2_digest.hexdigest()


_e2_blend_receipt = {
    "schema": "dinov5_effv2_diversity_blend_v1",
    "status": "COMPLETE",
    "studies": len(_e2_test_ids),
    "fallback_studies": 0,
    "checkpoint": _e2_receipt["checkpoint"],
    "per_target_gold58_auc": _e2_target_auc,
    "per_target_beta": dict(zip(_E2_LABELS, _e2_beta.tolist())),
    "beta_cap": 0.08,
    "arm_sha256": _e2_sha256(_e2_arm_path),
    "output_sha256": _e2_sha256(_e2_stage_path),
    "upstream_receipt_sha256": _e2_sha256(_e2_receipt_path),
    "elapsed_seconds": round(_e2_time.time() - _e2_started, 2),
}
rsna_json("/kaggle/working/diagnostics/effv2_blend_receipt.json", _e2_blend_receipt)
rsna_save_predictions("effv2_arm", _e2_test_ids, _e2_arm_rank, _E2_LABELS)
print("[effv2] conservative diversity blend complete", _e2_blend_receipt, flush=True)

