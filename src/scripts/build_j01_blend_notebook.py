#!/usr/bin/env python3
"""Build the B01 0.943 plus J01 rank-blend submission notebook."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "public_kernels/dinosaur_v5/rsna-knee-dinosaur-v5.ipynb"
ARM = ROOT / "public_kernels/j01_biomed_jepa_infer/rsna-knee-j01-biomed-jepa-infer.py"
BLEND = ROOT / "inference/j01_blend_cell.py"
FINAL_NEEDLE = "_release_path=Path('/kaggle/working/_pipeline_stage.csv')"


def code_cell(source, tag):
    ast.parse(source)
    return {"cell_type": "code", "execution_count": None,
            "metadata": {"tags": [tag]}, "outputs": [],
            "source": source.splitlines(keepends=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.08)
    args = parser.parse_args()
    if not (0 < args.beta <= 0.20):
        raise SystemExit("beta must be in (0, 0.20]")
    notebook = json.loads(BASE.read_text())
    matches = [i for i, cell in enumerate(notebook["cells"])
               if FINAL_NEEDLE in "".join(cell.get("source", []))]
    if len(matches) != 1:
        raise RuntimeError(f"expected one B01 final publish cell, found {matches}")
    arm_source = ARM.read_text()
    ast.parse(arm_source)
    writer = ("from pathlib import Path as _J1ProgramPath\n"
              f"_J1_PROGRAM_SOURCE = {arm_source!r}\n"
              "_J1ProgramPath('/kaggle/working/_j01_arm.py').write_text("
              "_J1_PROGRAM_SOURCE, encoding='utf-8')\n"
              "print('[j01] wrote isolated inference program')\n")
    blend_source = BLEND.read_text().replace("__J01_BETA__", repr(float(args.beta)))
    insertion = matches[0]
    notebook["cells"][insertion:insertion] = [
        code_cell(writer, "j01-program-writer"),
        code_cell(blend_source, "j01-rank-blend"),
    ]
    out_dir = ROOT / "public_kernels/dinosaur_v5_j01"
    output = out_dir / "rsna-knee-dinosaur-v5-j01.ipynb"
    out_dir.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, separators=(",", ":")) + "\n")
    metadata = json.loads((ROOT / "public_kernels/dinosaur_v5/kernel-metadata.json").read_text())
    metadata.update({"id": "adityakumarrai0401/rsna-knee-medal-j01-blend-v1",
                     "title": "RSNA Knee Medal J01 Blend V1", "code_file": output.name})
    sources = list(metadata.get("kernel_sources", []))
    trainer = "adityakumarrai0401/rsna-knee-j01-biomed-jepa-train"
    if trainer not in sources:
        sources.append(trainer)
    metadata["kernel_sources"] = sources
    (out_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), "beta": args.beta,
                      "cells": len(notebook["cells"])}, indent=2))


if __name__ == "__main__":
    main()
