#!/usr/bin/env python3
"""Fail-closed audit for a locally downloaded Kaggle submission artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion", "Synovitis",
    "Baker's", "Contusion", "Fracture",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--test", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    test = pd.read_csv(args.test, dtype={"StudyInstanceUID": str})
    sub = pd.read_csv(args.submission, dtype={"StudyInstanceUID": str})
    expected_columns = ["StudyInstanceUID", *TARGETS]
    errors = []
    if sub.columns.tolist() != expected_columns:
        errors.append(f"columns {sub.columns.tolist()} != {expected_columns}")
    expected_ids = test["StudyInstanceUID"].astype(str).tolist()
    actual_ids = sub.get("StudyInstanceUID", pd.Series(dtype=str)).astype(str).tolist()
    if actual_ids != expected_ids:
        errors.append("submission StudyInstanceUID order differs from test.csv")
    if len(actual_ids) != len(set(actual_ids)):
        errors.append("submission contains duplicate StudyInstanceUID values")

    target_stats = {}
    if all(t in sub for t in TARGETS):
        values = sub[TARGETS].to_numpy(np.float64)
        if not np.isfinite(values).all():
            errors.append("submission contains non-finite predictions")
        if ((values < 0) | (values > 1)).any():
            errors.append("submission predictions are outside [0, 1]")
        for j, target in enumerate(TARGETS):
            column = values[:, j]
            unique = int(np.unique(column).size)
            if unique < min(10, len(column)):
                errors.append(f"{target} has only {unique} unique predictions")
            target_stats[target] = {
                "min": float(np.min(column)),
                "median": float(np.median(column)),
                "max": float(np.max(column)),
                "unique": unique,
            }

    receipts = {}
    submission_digest = sha256(args.submission)
    for path in args.receipt:
        payload = json.loads(path.read_text())
        status = payload.get("status")
        fallback = payload.get("fallback_studies", payload.get("fallbacks", 0))
        status_ok = status is None or status == "COMPLETE" or str(status).startswith("VALID_")
        if not status_ok:
            errors.append(f"receipt {path} status is {status!r}")
        if int(fallback or 0) != 0:
            errors.append(f"receipt {path} reports {fallback} fallbacks")
        recorded_submission_digest = payload.get("submission_sha256")
        if recorded_submission_digest is not None and recorded_submission_digest != submission_digest:
            errors.append(f"receipt {path} submission SHA-256 does not match the downloaded CSV")
        receipts[str(path)] = payload

    report = {
        "schema": "rsna_knee_submission_audit_v1",
        "status": "PASS" if not errors else "FAIL",
        "submission": str(args.submission),
        "sha256": submission_digest,
        "rows": len(sub),
        "errors": errors,
        "target_stats": target_stats,
        "receipts": receipts,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
