#!/usr/bin/env python3
"""Rebuild the ConvNeXt blend notebook from the scored baseline and source cells."""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "public_kernels/dinosaur_v5/rsna-knee-dinosaur-v5.ipynb"
OUTPUT = ROOT / "public_kernels/dinosaur_v5_convnext/rsna-knee-dinosaur-v5-convnext.ipynb"
MAIN = ROOT / "inference/convnext_diversity_cell.py"
WORKER = ROOT / "inference/convnext_decode_worker.py"
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
        index
        for index, cell in enumerate(notebook["cells"])
        if FINAL_NEEDLE in "".join(cell.get("source", []))
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one final publish cell, found {matches}")

    worker_source = WORKER.read_text(encoding="utf-8")
    writer_source = (
        "from pathlib import Path as _CxWorkerPath\n"
        f"_CX_DECODE_WORKER_SOURCE = {worker_source!r}\n"
        "_CxWorkerPath('/kaggle/working/_convnext_decode_worker.py').write_text("
        "_CX_DECODE_WORKER_SOURCE, encoding='utf-8')\n"
        "print('[convnext] wrote isolated CPU decode worker')\n"
    )
    main_source = MAIN.read_text(encoding="utf-8")
    insertion = matches[0]
    notebook["cells"][insertion:insertion] = [
        code_cell(writer_source, "convnext-decode-worker"),
        code_cell(main_source, "convnext-diversity-blend"),
    ]

    tagged = {
        tuple(cell.get("metadata", {}).get("tags", [])): index
        for index, cell in enumerate(notebook["cells"])
    }
    if tagged.get(("convnext-decode-worker",)) != insertion:
        raise RuntimeError("Decoder writer insertion order drift")
    if tagged.get(("convnext-diversity-blend",)) != insertion + 1:
        raise RuntimeError("Blend insertion order drift")
    if FINAL_NEEDLE not in "".join(notebook["cells"][insertion + 2].get("source", [])):
        raise RuntimeError("Final publish cell no longer follows the blend")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(notebook, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT} with {len(notebook['cells'])} cells")


if __name__ == "__main__":
    main()
