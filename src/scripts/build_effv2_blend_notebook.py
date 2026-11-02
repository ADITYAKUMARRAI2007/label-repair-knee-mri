#!/usr/bin/env python3
"""Build the staged B01+E01 code-submission notebook."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "public_kernels/dinosaur_v5/rsna-knee-dinosaur-v5.ipynb"
ARM = ROOT / "inference/effv2_raptor_arm.py"
BLEND = ROOT / "inference/effv2_blend_cell.py"
OUT_DIR = ROOT / "public_kernels/dinosaur_v5_effv2"
OUTPUT = OUT_DIR / "rsna-knee-dinosaur-v5-effv2.ipynb"
FINAL_NEEDLE = "_release_path=Path('/kaggle/working/_pipeline_stage.csv')"


def code_cell(source: str, tag: str) -> dict:
    ast.parse(source)
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": [tag]},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def main() -> None:
    notebook = json.loads(BASE.read_text(encoding="utf-8"))
    matches = [
        index for index, cell in enumerate(notebook["cells"])
        if FINAL_NEEDLE in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one B01 final publish cell, found {matches}")

    arm_source = ARM.read_text(encoding="utf-8")
    ast.parse(arm_source)
    writer = (
        "from pathlib import Path as _E2ProgramPath\n"
        f"_E2_PROGRAM_SOURCE = {arm_source!r}\n"
        "_E2ProgramPath('/kaggle/working/_effv2_raptor_arm.py').write_text("
        "_E2_PROGRAM_SOURCE, encoding='utf-8')\n"
        "print('[effv2] wrote isolated inference program')\n"
    )
    blend_source = BLEND.read_text(encoding="utf-8")
    insertion = matches[0]
    notebook["cells"][insertion:insertion] = [
        code_cell(writer, "effv2-program-writer"),
        code_cell(blend_source, "effv2-diversity-blend"),
    ]
    if FINAL_NEEDLE not in "".join(notebook["cells"][insertion + 2].get("source", [])):
        raise RuntimeError("B01 final publish no longer follows the E01 blend")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, separators=(",", ":")) + "\n", encoding="utf-8")

    base_meta = json.loads(
        (ROOT / "public_kernels/dinosaur_v5/kernel-metadata.json").read_text(encoding="utf-8")
    )
    base_meta.update({
        "id": "adityakumarrai0401/rsna-knee-medal-effv2-blend-v1",
        "title": "RSNA Knee Medal EfficientNetV2 Blend V1",
        "code_file": OUTPUT.name,
    })
    sources = list(base_meta.get("kernel_sources", []))
    trainer = "adityakumarrai0401/rsna-knee-efficientnetv2-raptor-train-v1"
    if trainer not in sources:
        sources.append(trainer)
    base_meta["kernel_sources"] = sources
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(base_meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
