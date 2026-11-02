"""Run the published inference bytes and reject reported failures or bad output."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import selectors
import signal
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_csv(path):
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return reader.fieldnames, rows


def validate_submission(path, test_path, sample_path):
    fields, rows = read_csv(path)
    expected_fields, _ = read_csv(sample_path)
    _, test = read_csv(test_path)
    assert fields == expected_fields and len(fields) == 13, 'submission columns differ'
    expected_ids = [row['StudyInstanceUID'] for row in test]
    ids = [row['StudyInstanceUID'] for row in rows]
    assert ids == expected_ids and len(set(ids)) == len(ids) > 0, 'submission identities differ'
    for row in rows:
        assert set(row) == set(fields), 'malformed submission row'
        for name in fields[1:]:
            value = float(row[name])
            assert math.isfinite(value) and 0 <= value <= 1, 'nonfinite or invalid prediction'
    return dict(rows=len(rows), columns=len(fields), sha256=sha(path))


def reported_failure(line):
    return bool(re.search(r'\[pre\].*\bFAILED\b|\[gpu[^\]]*\].*\bFALLBACK\b', line))


def run_source(source, log_path, seconds_cap, cwd):
    """A separate process preserves the original script's __main__ execution."""
    start = time.monotonic()
    process = subprocess.Popen([sys.executable, '-u', str(source)], cwd=cwd,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               start_new_session=True)
    failures = []
    pending = b''
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        with Path(log_path).open('wb') as log:
            while selector.get_map():
                if time.monotonic() - start > seconds_cap:
                    raise TimeoutError('published inference exceeded fixed runtime cap')
                for key, _ in selector.select(timeout=1):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    log.write(chunk)
                    log.flush()
                    pending += chunk
                    while b'\n' in pending:
                        raw, pending = pending.split(b'\n', 1)
                        line = raw.decode('utf-8', errors='replace')
                        print(line, flush=True)
                        if reported_failure(line):
                            failures.append(line)
            if pending:
                line = pending.decode('utf-8', errors='replace')
                print(line, flush=True)
                if reported_failure(line):
                    failures.append(line)
        code = process.wait(timeout=max(1, seconds_cap - (time.monotonic() - start)))
        assert code == 0, f'published inference exited {code}'
        assert not failures, f'{len(failures)} reported preparation/inference failures; see inference.log'
        return dict(elapsed_seconds=time.monotonic() - start, reported_failure_count=0)
    finally:
        selector.close()
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        process.stdout.close()


def main():
    root = Path(__file__).resolve().parent
    contract = json.loads((root / 'contract.json').read_text())
    out = Path('/kaggle/working/nartaa_reproduction')
    out.mkdir(exist_ok=False)
    submission = Path('/kaggle/working/submission.csv')
    assert not submission.exists(), 'stale submission exists before inference'
    source = root / 'published_inference.py'
    try:
        assert sha(source) == contract['source_sha256']
        import torch
        import timm
        assert torch.cuda.device_count() == 2, 'requires original two-GPU environment'
        assert str(torch.__version__) == contract['torch_version']
        assert timm.__version__ == contract['timm_version']
        paths = [Path('/kaggle/input') / suffix / contract['checkpoint_filename'] for suffix in
                 ['rsna-knee-publication-swa-weights-20261007',
                  'datasets/nartaa/rsna-knee-publication-swa-weights-20261007']]
        checkpoint = next(p for p in paths if p.is_file())
        assert sha(checkpoint) == contract['checkpoint_sha256']
        data = next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
                               Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
                    if (p / 'test.csv').is_file())
        result = run_source(source, out / 'inference.log', contract['worker_seconds_cap'], '/kaggle/working')
        output = validate_submission(submission, data / 'test.csv', data / 'sample_submission.csv')
        result.update(status='PASS_EXECUTION_ONLY', contract=contract, submission=output,
                      checkpoint_sha256=sha(checkpoint), source_sha256=sha(source),
                      accuracy_measured=False, competition_submission_enabled=False,
                      limitations=['Checks reported study failures; original per-slice reader recovery is unchanged',
                                   'Visible test execution is not an accuracy measurement'])
        (out / 'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    except BaseException as error:
        if submission.exists():
            submission.replace(out / 'rejected_submission.csv')
        (out / 'failure.json').write_text(json.dumps(dict(status='REJECTED_NO_ELIGIBLE_SUBMISSION',
             error_type=type(error).__name__, message=str(error)), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
