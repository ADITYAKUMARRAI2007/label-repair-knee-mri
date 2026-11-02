#!/usr/bin/env python3
"""Retry one audited Kaggle kernel push until a batch GPU slot is available."""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def log(message: str) -> None:
    print(f"[{dt.datetime.now(dt.UTC).isoformat()}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_dir", type=Path)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--max-hours", type=float, default=12.0)
    args = parser.parse_args()
    if not (args.kernel_dir / "kernel-metadata.json").is_file():
        raise RuntimeError(f"missing kernel metadata in {args.kernel_dir}")
    api = KaggleApi()
    api.authenticate()
    deadline = time.monotonic() + args.max_hours * 3600
    attempt = 0
    while time.monotonic() < deadline:
        attempt += 1
        response = api.kernels_push(str(args.kernel_dir))
        error = str(response.error or "")
        if not error:
            log(f"push succeeded on attempt {attempt}: version={response.version_number} url={response.url}")
            return
        if "Maximum batch GPU session count" not in error:
            raise RuntimeError(f"kernel push failed: {error}")
        log(f"GPU slots occupied on attempt {attempt}; retrying in {args.interval}s")
        time.sleep(args.interval)
    raise TimeoutError("GPU slot did not become available before retry deadline")


if __name__ == "__main__":
    main()
