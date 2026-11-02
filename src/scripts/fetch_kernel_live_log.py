#!/usr/bin/env python3
"""Fetch the currently available Kaggle kernel log stream, then disconnect."""

from __future__ import annotations

import argparse
import json
import signal

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import (
    ApiGetKernelSessionLogsStreamRequest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel")
    parser.add_argument("--seconds", type=int, default=8)
    args = parser.parse_args()
    owner, slug = args.kernel.split("/", 1)

    api = KaggleApi()
    api.authenticate()
    with api.build_kaggle_client() as client:
        request = ApiGetKernelSessionLogsStreamRequest()
        request.user_name = owner
        request.kernel_slug = slug
        request.wait_for_logs_url_seconds = min(max(args.seconds, 1), 60)
        response = client.kernels.kernels_api_client.get_kernel_session_logs_stream(request)

        def stop(*_):
            raise TimeoutError

        signal.signal(signal.SIGALRM, stop)
        signal.alarm(args.seconds)
        try:
            for line in response.iter_lines(decode_unicode=True, chunk_size=1):
                if line:
                    if line.startswith("data: "):
                        raw = line[6:].strip()
                        if not raw:
                            continue
                        if raw == "[DONE]":
                            break
                        payload = json.loads(raw)
                        print(f"[{float(payload.get('time', 0)):9.1f}s] "
                              f"{payload.get('data', '').rstrip()}")
                    else:
                        print(line)
        except (TimeoutError, requests.exceptions.ConnectionError):
            pass
        finally:
            signal.alarm(0)
            response.close()


if __name__ == "__main__":
    main()
