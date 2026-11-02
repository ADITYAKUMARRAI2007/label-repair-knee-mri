"""Package a single bounded full fold; no remote execution from this script."""
import base64
import argparse
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold',type=int,choices=range(5),default=0)
    args = parser.parse_args()
    out = ROOT / f'runs/e05_full_fold{args.fold}'
    assert not (out / 'launch.json').exists(), 'Observe existing launch before rebuilding'
    read = lambda p: json.loads((ROOT / p).read_text())
    if args.fold > 0:
        first = read('runs/e05_full_fold0/watcher.json')
        assert first['status'] == 'FINISHED' and first['result'] == 'COMPLETE'
        assert first['fold_audit_passed'], 'Finish the first fold artifact audit before scheduling another'
        assert read('runs/e05_full_fold0/verified_final/integrity.json')['status'] == 'PASS_REMOTE_SNAPSHOT_BYTES'
    assert read('runs/e05_colab_pilot/resource_audit.json')['status'] == 'PASS_PAIRED_RESOURCE_EPOCHS'
    assert read('runs/e05_confidence_ablation/control_integrity.json')['status'] == 'PASS_ALL_FOUR'
    amendment = read('runs/e05_confidence_ablation/execution_amendment.json')
    assert amendment['first_full_job']['max_training_seconds'] == 7200
    sources = {
        'worker.py': 'runs/e05_confidence_ablation/full_worker.py',
        'supervisor.py': 'training/e05_full_colab_supervisor.py',
        'e05_fetch_assets.py': 'training/e05_colab_supervisor.py',
        'e05_upload_blob.py': 'training/e05_upload_blob.py',
        'e05_checkpoint_export.py': 'training/e05_checkpoint_export.py',
        'protocol.json': 'runs/e05_confidence_ablation/protocol.json',
        'execution_amendment.json': 'runs/e05_confidence_ablation/execution_amendment.json',
    }
    files = {name: (ROOT / path).read_text() for name, path in sources.items()}
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in files.items()}
    build = read('runs/e05_confidence_ablation/full_worker_build.json')
    assert hashes['worker.py'] == build['code_sha256']
    assert hashes['e05_checkpoint_export.py'] == build['export_helper_sha256']
    for name, value in files.items():
        if name.endswith('.py'): compile(value, name, 'exec')
    job = {'fold': args.fold, 'arm': 'uniform', 'training_hours_cap': 2.0,
           'competition_submission_enabled': False, 'files_sha256': hashes}
    files['job.json'] = json.dumps(job)
    files['download_access.json'] = (ROOT / 'runs/e05_colab_pilot/download_access.json').read_text()
    files['upload_leases_private.json'] = (out / 'upload_leases_private.json').read_text()
    assert len(json.loads(files['download_access.json'])) == 19
    leases = json.loads(files['upload_leases_private.json'])
    assert leases['prefix'] == f'e05_uniform_fold{args.fold}'
    for phase in leases['snapshots'].values():
        for item in phase.values(): assert 'token' not in item
    payload = base64.b64encode(zlib.compress(json.dumps(files).encode(), 9)).decode()
    code = '''# E05 single-fold bootstrap; private transfer capabilities redacted after launch.
import base64, json, zlib, os, sys, subprocess, threading
from pathlib import Path
from google.colab import runtime
setup = Path('/content/e05_full_setup')
assert not setup.exists(), 'Existing setup: inspect its process before restarting'
setup.mkdir()
def release_runtime():
    try: runtime.unassign()
    except Exception: pass
e05_release_timer = threading.Timer(8700, release_runtime)
e05_release_timer.daemon = True
e05_release_timer.start()
files = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
for name, value in files.items():
    assert Path(name).name == name
    path = setup / name
    path.write_text(value)
    path.chmod(0o600)
del files
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm==1.0.29'], check=True, capture_output=True, timeout=120)
    import torch, timm, numpy
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert 'A100' in torch.cuda.get_device_name(0), 'Resource estimate requires A100'
    assert torch.__version__ == '2.11.0+cu130' and timm.__version__ == '1.0.29'
    assert numpy.__version__ == '2.1.3'
    environment = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'timm': timm.__version__, 'numpy': numpy.__version__}
    (setup / 'environment.json').write_text(json.dumps(environment))
    e05_full_log = (setup / 'supervisor.log').open('w')
    e05_full_process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=15', '8400', sys.executable, '-u', str(setup / 'supervisor.py')], stdout=e05_full_log, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'status': 'FULL_FOLD_SUPERVISOR_STARTED', 'pid': e05_full_process.pid, 'environment': environment, 'submission_enabled': False}))
except Exception:
    release_runtime()
    raise
'''.replace('PAYLOAD', repr(payload))
    compile(code, 'bootstrap.py', 'exec')
    request = {'action': 'call', 'name': 'add_code_cell', 'arguments': {'cellIndex': 17, 'language': 'python', 'code': code}}
    private = out / 'stage_request.json'
    private.write_text(json.dumps(request)); private.chmod(0o600)
    (out / 'job.json').write_text(json.dumps(job, indent=2)+'\n')
    (out / 'package_build.json').write_text(json.dumps({'status': 'STAGED_NOT_LAUNCHED', 'job': job,
        'bootstrap_sha256': hashlib.sha256(code.encode()).hexdigest(), 'total_runtime_cap_seconds': 9000,
        'private_request_bytes': private.stat().st_size}, indent=2)+'\n')
    print(json.dumps({'status': 'STAGED_NOT_LAUNCHED', 'bytes': private.stat().st_size, 'source_files': len(hashes)}))


if __name__ == '__main__': main()
