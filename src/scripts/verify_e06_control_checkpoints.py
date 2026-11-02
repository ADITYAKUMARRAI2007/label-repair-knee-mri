"""Hash the completed E06 control's actual checkpoint bytes without loading tensors."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal

import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/e06_spatial"


def main():
    launch = json.loads((ROOT / "experiments/kaggle_e06_spatial/launch.json").read_text())
    manifest_path = RUN / "output/control/e06_control_fold2_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "COMPLETE" and manifest["completed_epoch"] == 14
    assert manifest["contract"]["arm"] == "control"
    wanted = {"control/" + name: receipt for name, receipt in manifest["artifacts"].items()
              if name in {"e06_control_fold2_last.pt", "e06_control_fold2_ema.pt"}}
    assert len(wanted) == 2
    api = KaggleApi(); api.authenticate()
    assert api.kernels_status(launch["kernel"]).to_dict()["status"] == "COMPLETE"
    source = Path(api.kernels_pull(launch["kernel"], str(RUN / "remote_source"), quiet=True))
    assert hashlib.sha256(source.read_bytes()).hexdigest() == launch["code_sha256"]
    owner, slug = launch["kernel"].split("/", 1)
    request = ApiListKernelSessionOutputRequest()
    request.user_name, request.kernel_slug, request.page_size = owner, slug, 100
    found = {}
    with api.build_kaggle_client() as client:
        while True:
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            for item in response.files or []:
                if item.file_name in wanted:
                    assert item.file_name not in found
                    found[item.file_name] = item.url
            if not response.next_page_token:
                break
            request.page_token = response.next_page_token
    assert set(found) == set(wanted), "completed control checkpoints missing from output listing"
    report = {"status": "VERIFYING", "kernel": launch["kernel"],
              "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "source_sha256": launch["code_sha256"], "checkpoints": {},
              "tensor_deserialization_performed": False, "competition_submission_enabled": False}
    receipt_path = RUN / "control_checkpoint_integrity.json"
    for name in sorted(wanted):
        expected = wanted[name]
        retain = name.endswith("_ema.pt")
        target = RUN / "output" / name
        temp = target.with_suffix(".download")
        print(json.dumps({"event": "hashing_checkpoint", "name": name, "bytes": expected["bytes"]}), flush=True)
        digest, size = hashlib.sha256(), 0
        handle = temp.open("wb") if retain else None
        try:
            with requests.get(found[name], stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                for block in response.iter_content(8 << 20):
                    if not block:
                        continue
                    size += len(block)
                    assert size <= expected["bytes"], "checkpoint exceeded expected size"
                    digest.update(block)
                    if handle is not None:
                        handle.write(block)
            assert size == expected["bytes"] and digest.hexdigest() == expected["sha256"], "checkpoint bytes differ"
        finally:
            if handle is not None:
                handle.close()
        if retain:
            temp.replace(target)
        report["checkpoints"][name] = {"bytes": size, "sha256": digest.hexdigest(),
                                         "actual_remote_bytes_verified": True, "retained_locally": retain}
        report["updated_utc"] = datetime.now(timezone.utc).isoformat()
        receipt_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"event": "checkpoint_verified", "name": name, "bytes": size}), flush=True)
    report["status"] = "PASS_BOTH_CONTROL_CHECKPOINT_BYTE_HASHES"
    receipt_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    def expired(*_):
        raise TimeoutError("checkpoint observation time limit")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(900)
    try:
        main()
    except Exception as error:
        # Signed download URLs must not appear in error output.
        print(json.dumps({"status": "CHECKPOINT_VERIFICATION_FAILED", "error_type": type(error).__name__}), flush=True)
        raise SystemExit(1)
