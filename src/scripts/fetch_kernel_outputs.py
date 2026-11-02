#!/usr/bin/env python3
"""Stream existing Kaggle output files atomically; never restart or submit jobs.

Download errors are reported by exception type without printing signed URLs.
When a run manifest supplies checksums, completed local files are verified and
reused. A failed partial download is restarted, never appended blindly.
"""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default=r"^e03_.*\.(json|csv|pt)$")
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    owner, slug = args.kernel.split("/", 1)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    pattern = re.compile(args.pattern)
    files, token, log = [], None, None
    with api.build_kaggle_client() as client:
        while True:
            req = ApiListKernelSessionOutputRequest()
            req.user_name, req.kernel_slug = owner, slug
            req.page_size = 100
            if token:
                req.page_token = token
            response = client.kernels.kernels_api_client.list_kernel_session_output(req)
            files.extend(response.files or [])
            log = response.log or log
            token = response.next_page_token
            if not token:
                break
    if log:
        (directory / f"{slug}.log").write_text(log)
    selected = [f for f in files if pattern.search(f.file_name)]
    # Obtain checksums and validation CSVs before the large state blobs.
    selected.sort(key=lambda f: (0 if f.file_name.endswith("manifest.json") else
                                 2 if f.file_name.endswith(".pt") else 1, f.file_name))
    print(json.dumps({"kernel": args.kernel, "files": [f.file_name for f in selected]}), flush=True)
    if not selected:
        raise RuntimeError("requested existing outputs were absent")
    expected, receipts = {}, []
    for item in selected:
        path = (directory / item.file_name).resolve()
        if directory not in path.parents:
            raise RuntimeError("output filename escapes destination")
        path.parent.mkdir(parents=True, exist_ok=True)
        recorded = expected.get(item.file_name)
        valid_existing = (recorded and path.exists() and path.stat().st_size == recorded["bytes"]
                          and sha256(path) == recorded["sha256"])
        if valid_existing:
            print(json.dumps({"file": item.file_name, "state": "VERIFIED_EXISTING"}), flush=True)
        else:
            partial = path.with_suffix(path.suffix + ".part")
            completed = False
            for attempt in range(1, args.attempts + 1):
                try:
                    h, total, last_report = hashlib.sha256(), 0, time.monotonic()
                    with requests.get(item.url, stream=True, timeout=(30, 60)) as response:
                        response.raise_for_status()
                        with partial.open("wb") as out:
                            for chunk in response.iter_content(chunk_size=1 << 20):
                                if not chunk:
                                    continue
                                out.write(chunk)
                                h.update(chunk)
                                total += len(chunk)
                                if time.monotonic() - last_report >= 30:
                                    print(json.dumps({"file": item.file_name, "state": "DOWNLOADING", "bytes": total}), flush=True)
                                    last_report = time.monotonic()
                    if recorded and (total != recorded["bytes"] or h.hexdigest() != recorded["sha256"]):
                        raise RuntimeError("download checksum/size mismatch")
                    partial.replace(path)
                    completed = True
                    print(json.dumps({"file": item.file_name, "state": "DOWNLOADED", "bytes": total,
                                      "sha256": h.hexdigest()}), flush=True)
                    break
                except (requests.RequestException, OSError, RuntimeError) as error:
                    print(json.dumps({"file": item.file_name, "attempt": attempt,
                                      "download_error_type": type(error).__name__,
                                      "job_is_not_restarted": True}), flush=True)
                    if attempt < args.attempts:
                        time.sleep(5)
            if not completed:
                return 2
        receipts.append({"file": item.file_name, "bytes": path.stat().st_size, "sha256": sha256(path)})
        if path.name.endswith("manifest.json"):
            manifest = json.loads(path.read_text())
            expected = manifest.get("artifacts", {})
    (directory / "download_receipt.json").write_text(json.dumps({"schema": "kaggle_existing_output_download_v1",
        "kernel": args.kernel, "status": "COMPLETE", "job_mutations": False, "files": receipts}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
