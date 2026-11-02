#!/usr/bin/env python3
"""Synthetic end-to-end smoke test for the staged B01+M01 blend cell."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "inference/medicalnet3d_blend_cell.py"
TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="m01_blend_smoke_") as directory:
        root = Path(directory)
        working = root / "working"
        competition = root / "competition"
        diagnostics = working / "diagnostics"
        working.mkdir()
        competition.mkdir()
        diagnostics.mkdir()

        ids = [f"study-{index:02d}" for index in range(8)]
        pd.DataFrame({"StudyInstanceUID": ids}).to_csv(
            competition / "test.csv", index=False
        )
        base = pd.DataFrame({"StudyInstanceUID": ids})
        for index, target in enumerate(TARGETS):
            base[target] = np.roll(np.linspace(0.05, 0.95, len(ids)), index % len(ids))
        base.to_csv(working / "_pipeline_stage.csv", index=False)

        fake_program = f'''import json, os
import numpy as np
import pandas as pd
targets={TARGETS!r}
ids={ids!r}
out=os.environ["RSNA_MEDICALNET3D_OUTPUT"]
frame=pd.DataFrame({{"StudyInstanceUID":ids}})
for j,target in enumerate(targets):
    frame[target]=np.roll(np.linspace(0.97,0.03,len(ids)),(j*3)%len(ids))
frame.to_csv(out,index=False)
receipt={{
    "schema":"medicalnet3d_inference_v1",
    "status":"COMPLETE",
    "studies":len(ids),
    "fallback_studies":0,
    "checkpoint":"medicalnet3d_best.pt",
    "checkpoint_info":{{
        "gold_auc":0.80,
        "gold_target_auc":{{target:0.80 for target in targets}},
    }},
}}
with open({str(working / 'medicalnet3d_receipt.json')!r},"w") as handle:
    json.dump(receipt,handle)
'''
        (working / "_medicalnet3d_arm.py").write_text(fake_program)

        source = CELL.read_text()
        source = source.replace(
            "/kaggle/input/competitions/rsna-knee-abnormality-detection",
            str(competition),
        ).replace(
            "/kaggle/input/rsna-knee-abnormality-detection", str(competition)
        ).replace("/kaggle/working", str(working))

        saved = {}

        def rsna_frame(frame, expected_ids, labels, name):
            if frame["StudyInstanceUID"].astype(str).tolist() != expected_ids:
                raise RuntimeError(f"{name} identity mismatch")
            if frame.columns.tolist() != ["StudyInstanceUID", *labels]:
                raise RuntimeError(f"{name} schema mismatch")
            return frame

        def rsna_json(path, payload):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

        def rsna_save_predictions(name, study_ids, values, labels):
            saved[name] = (list(study_ids), np.asarray(values), list(labels))

        namespace = {
            "rsna_frame": rsna_frame,
            "rsna_json": rsna_json,
            "rsna_save_predictions": rsna_save_predictions,
        }
        exec(compile(source, str(CELL), "exec"), namespace)

        result = pd.read_csv(working / "_pipeline_stage.csv")
        if result.shape != (len(ids), len(TARGETS) + 1):
            raise RuntimeError(f"unexpected blend shape {result.shape}")
        if not np.isfinite(result[TARGETS].to_numpy()).all():
            raise RuntimeError("blend contains non-finite values")
        receipt = json.loads(
            (diagnostics / "medicalnet3d_blend_receipt.json").read_text()
        )
        if receipt.get("status") != "COMPLETE" or receipt.get("fallback_studies") != 0:
            raise RuntimeError(f"invalid blend receipt {receipt}")
        betas = np.asarray(list(receipt["per_target_beta"].values()), np.float64)
        if not np.allclose(betas, 0.05) or (betas > 0.08).any():
            raise RuntimeError(f"unexpected synthetic blend weights {betas}")
        if "medicalnet3d_arm" not in saved:
            raise RuntimeError("component prediction persistence was not invoked")
        print("medicalnet3d blend smoke PASS", receipt["output_sha256"])


if __name__ == "__main__":
    main()
