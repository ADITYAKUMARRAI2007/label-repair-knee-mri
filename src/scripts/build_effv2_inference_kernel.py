#!/usr/bin/env python3
"""Package the reproducible E01 inference source as a Kaggle script kernel."""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "inference/effv2_raptor_arm.py"
OUTPUT = ROOT / "public_kernels/effv2_raptor_infer/rsna-knee-effv2-raptor-infer.py"


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    ast.parse(source)
    required = (
        "tf_efficientnetv2_s.in21k",
        "K_EVAL = 42",
        "fallback_studies\": 0",
        "/kaggle/working/_effv2_arm.csv",
    )
    for token in required:
        if token not in source:
            raise RuntimeError(f"missing E01 inference contract token: {token}")
    source = (
        "import os as _standalone_os\n"
        "_standalone_os.environ.setdefault(\n"
        "    'RSNA_EFFV2_OUTPUT', '/kaggle/working/submission.csv'\n"
        ")\n"
        + source
    )
    ast.parse(source)
    if "/kaggle/working/submission.csv" not in source:
        raise RuntimeError("standalone E01 kernel does not publish submission.csv")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(source, encoding="utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
