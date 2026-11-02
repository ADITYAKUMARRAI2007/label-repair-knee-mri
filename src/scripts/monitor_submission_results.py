#!/usr/bin/env python3
"""Read existing Kaggle submissions until one finishes; never submit or restart jobs."""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.submission_status import SubmissionStatus


def save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", default="rsna-knee-abnormality-detection")
    parser.add_argument("--refs", type=int, nargs="+", required=True)
    parser.add_argument("--interval", type=int, default=300)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--output", type=Path, default=Path("runs/submission_monitor"))
    args = parser.parse_args()
    if args.interval < 60 or args.max_hours <= 0:
        parser.error("interval must be at least 60 seconds and max-hours positive")
    refs = set(args.refs)
    args.output.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    started = time.monotonic()
    errors = 0
    last_observation = None
    while time.monotonic() - started < args.max_hours * 3600:
        try:
            api.authenticate()
            found = {
                s.ref: s for s in api.competition_submissions(
                    args.competition, page_size=100
                ) if s.ref in refs
            }
            if set(found) != refs:
                raise RuntimeError("requested submission absent from API response")
            now = datetime.now(timezone.utc)
            rows = []
            for ref, submission in sorted(found.items()):
                date = submission.date
                if date.tzinfo is None:
                    date = date.replace(tzinfo=timezone.utc)
                rows.append({
                    "submission_id": ref,
                    "description": submission.description,
                    "submitted_utc": date.astimezone(timezone.utc).isoformat(),
                    "submitted_ist": date.astimezone(ZoneInfo("Asia/Kolkata")).isoformat(),
                    "elapsed_minutes": round((now - date).total_seconds() / 60, 1),
                    "status": submission.status.name,
                    "public_score": submission.public_score or None,
                    "error": submission.error_description or None,
                })
            terminal = [s.ref for s in found.values() if s.status in (
                SubmissionStatus.COMPLETE, SubmissionStatus.ERROR
            )]
            report = {
                "schema": "rsna_submission_observation_v1",
                "source": "Kaggle CompetitionApiService ListSubmissions",
                "observed_utc": now.isoformat(),
                "observed_ist": now.astimezone(ZoneInfo("Asia/Kolkata")).isoformat(),
                "monitor_pid": os.getpid(),
                "monitor_state": "RESULT_AVAILABLE" if terminal else "WAITING",
                "terminal_submission_ids": sorted(terminal),
                "automatic_submission_enabled": False,
                "submissions": rows,
            }
            save(args.output / "latest.json", report)
            with (args.output / "observations.jsonl").open("a") as handle:
                handle.write(json.dumps(report) + "\n")
            print(json.dumps(report), flush=True)
            last_observation = report
            if terminal:
                save(args.output / "result.json", report)
                return 0
            errors = 0
        except KeyboardInterrupt:
            print("Monitor stopped; existing Kaggle submissions are unchanged.", flush=True)
            return 130
        except Exception as exc:
            errors += 1
            # Observation failures never label or restart the remote submissions.
            print(json.dumps({"observation_error_type": type(exc).__name__,
                              "consecutive_errors": errors,
                              "retry_same_submission_ids": sorted(refs)}), flush=True)
        remaining = args.max_hours * 3600 - (time.monotonic() - started)
        if remaining > 0:
            time.sleep(min(args.interval, remaining))
    save(args.output / "monitor_timeout.json", {
        "monitor_state": "OBSERVATION_TIMEOUT",
        "remote_submissions_are_not_declared_terminal": True,
        "last_observation": last_observation,
    })
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
