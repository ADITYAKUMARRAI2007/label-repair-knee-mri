#!/usr/bin/env python3
"""Safely fill Kaggle's two GPU slots with the remaining audited T03 folds."""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


ROOT = Path(__file__).resolve().parents[1]
INITIAL = [
    "adityakumarrai0401/rsna-knee-t03-effb0-fold-0",
    "adityakumarrai0401/rsna-knee-t03-effb0-fold-1",
]
PENDING = [ROOT / f"public_kernels/t03_effb0_f{fold}" for fold in (2, 3, 4)]


def log(message):
    print(f"[{dt.datetime.now(dt.UTC).isoformat()}] {message}", flush=True)


def main():
    api = KaggleApi()
    api.authenticate()
    known = list(INITIAL)
    deadline = time.monotonic() + 18 * 3600
    for directory in PENDING:
        metadata = json.loads((directory / "kernel-metadata.json").read_text())
        reference = metadata["id"]
        while time.monotonic() < deadline:
            states = {}
            for prior in known:
                reply = api.kernels_status(prior)
                state = str(reply.status).split(".")[-1]
                states[prior] = state
                if state == "ERROR":
                    raise RuntimeError(f"refusing to replicate after {prior} failed: "
                                       f"{reply.failure_message}")
            response = api.kernels_push(str(directory))
            error = str(response.error or "")
            if not error:
                known.append(reference)
                log(f"launched {reference}; version={response.version_number}; prior={states}")
                break
            if "Maximum batch GPU session count" not in error:
                raise RuntimeError(f"push of {reference} failed: {error}")
            log(f"slots occupied; pending={reference}; states={states}")
            time.sleep(300)
        else:
            raise TimeoutError(f"deadline reached before launching {reference}")
    log("all remaining T03 folds launched")


if __name__ == "__main__":
    main()
