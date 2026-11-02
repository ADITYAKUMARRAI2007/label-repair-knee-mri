"""Watch the accepted E06 job and evaluate small outputs; never launch or submit."""
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/e06_spatial"
EXPERIMENT = ROOT / "experiments/kaggle_e06_spatial"
OUTPUT_PATTERN = (r"^(e06_(result|failure)\.json|"
                  r"(control|spatial)/(effective_weights\.json|model_preflight\.json|worker\.log|"
                  r"e06_(control|spatial)_fold2_(manifest|split|history)\.json|"
                  r"e06_(control|spatial)_fold2_oof_predictions\.csv))$")


def now():
    return datetime.now(timezone.utc).isoformat()


def timeout(*_):
    raise TimeoutError("bounded observation expired")


def main():
    RUN.mkdir(exist_ok=True)
    lock = (RUN / "watcher.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = RUN / "watcher.json"
    if state_path.exists() and json.loads(state_path.read_text()).get("status") == "FINISHED":
        raise RuntimeError("E06 already observed to completion; inspect existing results")
    launch = json.loads((EXPERIMENT / "launch.json").read_text())
    if launch["status"] != "PUSH_ACCEPTED":
        raise RuntimeError("no accepted E06 launch")
    api = KaggleApi()
    api.authenticate()
    ref = launch["kernel"]
    record = {"status": "RUNNING", "pid": os.getpid(), "started_utc": now(),
              "kernel": ref, "gpu_launch_enabled": False, "competition_submission_enabled": False,
              "poll_interval_seconds": 300, "maximum_observation_seconds": 21600}
    signal.signal(signal.SIGALRM, timeout)

    def persist():
        record["updated_utc"] = now()
        temp = state_path.with_suffix(".tmp")
        temp.write_text(json.dumps(record, indent=2) + "\n")
        temp.replace(state_path)

    start = time.monotonic()
    persist()
    while time.monotonic() - start < 21600:
        try:
            signal.alarm(60)
            status = api.kernels_status(ref).to_dict()["status"]
            signal.alarm(0)
            record.update(remote_status=status, remote_observed_utc=now())
            record.pop("last_observation_error", None)
            persist()
            print(json.dumps({"event": "observed", "status": status, "utc": now()}), flush=True)
            if status in {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED", "CANCELED", "CANCELLED"}:
                signal.alarm(60)
                source_dir = RUN / "remote_source"
                source = Path(api.kernels_pull(ref, str(source_dir), metadata=False, quiet=True))
                signal.alarm(0)
                if hashlib.sha256(source.read_bytes()).hexdigest() != launch["code_sha256"]:
                    record.update(status="FINISHED", result="REMOTE_SOURCE_CHANGED_REVIEW_REQUIRED")
                    persist()
                    break
                signal.alarm(60)
                output = RUN / "output"
                files, _ = api.kernels_output(ref, str(output), file_pattern=OUTPUT_PATTERN,
                                               force=True, quiet=True)
                signal.alarm(0)
                record["downloaded_small_files"] = len(files)
                if status != "COMPLETE":
                    record.update(status="FINISHED", result="REMOTE_JOB_FAILED_OR_CANCELLED")
                    persist()
                    break
                result_path = output / "e06_result.json"
                if not result_path.exists():
                    record["output_state"] = "WAITING_FOR_RESULT_FILE"
                else:
                    result = json.loads(result_path.read_text())
                    if result["status"] != "COMPLETE":
                        record.update(status="FINISHED", result=result["status"],
                                      remote_result=result)
                        persist()
                        break
                    with (RUN / "watcher_evaluation.log").open("a") as log:
                        evaluation = subprocess.run(
                            [sys.executable, str(ROOT / "scripts/screen_e06_complement.py"),
                             "--job-output", str(output)], cwd=ROOT, stdout=log,
                            stderr=subprocess.STDOUT, timeout=120)
                    record.update(status="FINISHED", result="EVALUATED" if evaluation.returncode == 0
                                  else "EVALUATION_FAILED_REVIEW_REQUIRED",
                                  evaluation_returncode=evaluation.returncode,
                                  checkpoint_bytes_downloaded=False)
                    persist()
                    break
        except Exception as error:
            # Never log exception messages: download errors can contain signed URLs.
            signal.alarm(0)
            record["last_observation_error"] = type(error).__name__
            record["last_observation_error_utc"] = now()
            persist()
            print(json.dumps({"event": "observation_retry", "error_type": type(error).__name__}), flush=True)
        signal.alarm(0)
        persist()
        time.sleep(300)
    else:
        record.update(status="OBSERVATION_WINDOW_ENDED", result="REMOTE_STATE_REQUIRES_RECHECK")
        persist()
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
