"""Restart-safe offline resource-pilot driver with visible subprocess failures."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_attempt(work):
    """Hold one process lock and preserve every earlier run's files."""
    work = Path(work)
    lock = (work / 'mars_resource_pilot.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('A MARS pilot is already active; inspect that run') from None
    attempt = Path(tempfile.mkdtemp(prefix='mars_resource_attempt_', dir=work))
    root = attempt / 'mars_pilot_source'
    root.mkdir()
    return lock, attempt, root


def report_failure(attempt, stage, error):
    record = {
        'status': 'PILOT_EXECUTION_REJECTED', 'stage': stage,
        'error_type': type(error).__name__, 'error_message': str(error),
        'return_code': getattr(error, 'returncode', None),
        'traceback': traceback.format_exc(),
        'classification_fitting_enabled': False,
        'competition_submission_enabled': False,
    }
    (attempt / 'mars_resource_bootstrap_failure.json').write_text(
        json.dumps(record, indent=2) + '\n')
    print(json.dumps({k: v for k, v in record.items() if k != 'traceback'}),
          file=sys.stderr, flush=True)
    for relative in ['mars_resource_pilot/failure.json',
                     'mars_resource_worker.log', 'mars_resource_bootstrap.log']:
        path = attempt / relative
        if path.is_file():
            print(f'--- {path} (last 60 lines) ---', file=sys.stderr, flush=True)
            print('\n'.join(path.read_text(errors='replace').splitlines()[-60:]),
                  file=sys.stderr, flush=True)


def run(modules, work_directory='/kaggle/working'):
    work = Path(work_directory)
    lock, attempt, root = prepare_attempt(work)
    started = time.monotonic()
    stage = 'write_sources'
    try:
        # All source, dependency, feature and log paths are fresh per attempt.
        # An interactive session restart never overwrites previous evidence.
        for name, source in modules.items():
            if Path(name).name != name:
                raise ValueError('Unexpected packaged source filename')
            (root / name).write_text(source)
        contract = json.loads(modules['pilot_contract.json'])
        marker = {'schema': 'mars_latest_resource_attempt_v1',
                  'attempt_directory': attempt.name,
                  'contract_sha256': sha(root / 'pilot_contract.json')}
        temp_marker = attempt / 'latest_attempt.json'
        temp_marker.write_text(json.dumps(marker, indent=2) + '\n')
        os.replace(temp_marker, work / 'mars_latest_attempt.json')
        print(json.dumps({'stage': 'pilot_attempt', 'directory': str(attempt)}), flush=True)

        def remaining():
            value = contract['bootstrap_and_worker_seconds_cap'] - (time.monotonic() - started)
            if value <= 0:
                raise TimeoutError('Fixed resource pilot budget exhausted')
            return value

        def attached(slug, filename):
            paths = [Path('/kaggle/input') / slug,
                     Path('/kaggle/input/datasets/adityakumarrai0401') / slug]
            return next(p for p in paths if (p / filename).is_file())

        stage = 'runtime_and_assets'
        assert sys.version_info[:2] == (3, 13), 'Pinned Kaggle Python runtime changed'
        import torch
        assert str(torch.__version__) == contract['torch_version'], 'Pinned Torch runtime changed'
        assert torch.cuda.device_count() == 2, 'Pilot requires its allocated T4 pair'
        for name, expected in contract['module_sha256'].items():
            assert sha(root / name) == expected, name
        assets = attached('rsna-knee-mars-model-assets-v2', 'MARS.pt')
        for name, expected in contract['asset_sha256'].items():
            assert sha(assets / name) == expected, name
        decoder = attached('rsna-knee-decoder-cp313', 'decoder_identity.json')
        wheels = []
        for name, expected in contract['decoder_wheels'].items():
            assert sha(decoder / name) == expected, name
            wheels.append(str(decoder / name))
        target = attempt / 'mars_pilot_dependencies'
        stage = 'offline_dependencies_and_decoder'
        with (attempt / 'mars_resource_bootstrap.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                            '--target', str(target), str(assets / contract['monai_wheel']), *wheels],
                           check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=min(180, remaining()))
            env = dict(os.environ, MARS_PILOT_DEPENDENCIES=str(target),
                       PYTHONPATH=os.pathsep.join(
                           [str(target), str(root), os.environ.get('PYTHONPATH', '')]))
            subprocess.run([sys.executable, str(root / 'check_decoders.py')], env=env,
                           check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=min(60, remaining()))
        assert 'DECODER_ABI_PASS cp313' in (attempt / 'mars_resource_bootstrap.log').read_text()
        data = next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
                               Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
                    if (p / 'train_series.csv').is_file())
        stage = 'resource_worker'
        with (attempt / 'mars_resource_worker.log').open('w') as log:
            subprocess.run([sys.executable, str(root / 'mars_resource_pilot.py'),
                            '--input', str(data), '--assets', str(assets),
                            '--out', str(attempt / 'mars_resource_pilot'),
                            '--protocol', str(root / 'resource_protocol.json'), '--device', 'cuda:0'],
                           env=env, check=True, pass_fds=(lock.fileno(),),
                           stdout=log, stderr=subprocess.STDOUT,
                           timeout=remaining())
        result = json.loads((attempt / 'mars_resource_pilot/receipt.json').read_text())
        assert result['status'] == 'REAL_DICOM_CUDA_RESOURCE_PILOT_COMPLETE_NO_ACCURACY'
        assert not (attempt / 'submission.csv').exists()
        (attempt / 'mars_resource_execution.json').write_text(json.dumps({
            'status': 'BOUNDED_PILOT_EXECUTION_COMPLETE',
            'contract_sha256': sha(root / 'pilot_contract.json'),
            'pilot_receipt_sha256': sha(attempt / 'mars_resource_pilot/receipt.json'),
            'elapsed_seconds': time.monotonic() - started,
            'classification_fitting_enabled': False, 'competition_submission_enabled': False,
        }, indent=2) + '\n')
        print(json.dumps({'status': result['status'],
                          'completed_studies': result['completed_studies']}), flush=True)
    except BaseException as error:
        report_failure(attempt, stage, error)
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(MODULES)
