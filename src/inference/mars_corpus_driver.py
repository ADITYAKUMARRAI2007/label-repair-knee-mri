"""Bounded two-GPU frozen extraction; stops both owned workers on failure."""
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


def run(modules):
    work = Path('/kaggle/working')
    lock = (work / 'mars_corpus.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = Path(tempfile.mkdtemp(prefix='mars_corpus_attempt_', dir=work))
    (work / 'mars_latest_corpus_attempt.json').write_text(
        json.dumps({'attempt_directory': attempt.name}) + '\n')
    source = attempt / 'source'; source.mkdir()
    processes, logs, progress_streams = [], [], []
    started = time.monotonic()
    try:
        for name, code in modules.items():
            assert Path(name).name == name
            (source / name).write_text(code)
        contract = json.loads(modules['corpus_contract.json'])
        review = json.loads(modules['corpus_review.json'])
        maximum = contract['maximum_parent_seconds']

        def remaining():
            value = maximum - (time.monotonic() - started)
            if value <= 0: raise TimeoutError('Reviewed full extraction budget exhausted')
            return value

        def attached(slug, name):
            return next(p for p in [Path('/kaggle/input') / slug,
                                    Path('/kaggle/input/datasets/adityakumarrai0401') / slug]
                        if (p / name).is_file())

        assert sys.version_info[:2] == (3, 13)
        import torch
        assert str(torch.__version__) == contract['torch_version']
        assert torch.cuda.device_count() == 2
        for name, expected in contract['module_sha256'].items():
            assert sha(source / name) == expected, name
        assets = attached('rsna-knee-mars-model-assets-v2', 'MARS.pt')
        for name, expected in contract['asset_sha256'].items():
            assert sha(assets / name) == expected, name
        decoder = attached('rsna-knee-decoder-cp313', 'decoder_identity.json')
        wheels = []
        for name, expected in contract['decoder_wheels'].items():
            assert sha(decoder / name) == expected, name
            wheels.append(str(decoder / name))
        target = attempt / 'dependencies'
        env = dict(os.environ, MARS_PILOT_DEPENDENCIES=str(target), PYTHONPATH=os.pathsep.join(
            [str(target), str(source), os.environ.get('PYTHONPATH', '')]))
        with (attempt / 'bootstrap.log').open('w') as log:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                            '--target', str(target), str(assets / contract['monai_wheel']), *wheels],
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=min(180, remaining()))
            subprocess.run([sys.executable, str(source / 'check_decoders.py')], env=env,
                           check=True, stdout=log, stderr=subprocess.STDOUT, timeout=min(60, remaining()))
        assert 'DECODER_ABI_PASS cp313' in (attempt / 'bootstrap.log').read_text()
        data = next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
                               Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
                    if (p / 'train_series.csv').is_file())
        output = attempt / 'corpus'; output.mkdir()
        for shard in range(2):
            log = (attempt / f'worker{shard}.log').open('w'); logs.append(log)
            processes.append(subprocess.Popen([
                sys.executable, str(source / 'mars_corpus_extract.py'), '--input', str(data),
                '--assets', str(assets), '--out', str(output), '--review', str(source / 'corpus_review.json'),
                '--shard', str(shard)], env=env, stdout=log, stderr=subprocess.STDOUT,
                pass_fds=(lock.fileno(),)))
            progress_streams.append((attempt / f'worker{shard}.log').open())
        while any(p.poll() is None for p in processes):
            remaining()
            for progress in progress_streams:
                text = progress.read()
                if text:
                    print(text, end='', flush=True)
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError('A corpus worker failed; preserve partial outputs and stop its sibling')
            time.sleep(2)
        for progress in progress_streams:
            text = progress.read()
            if text:
                print(text, end='', flush=True)
        assert all(p.returncode == 0 for p in processes)
        receipts = [json.loads((output / f'shard{s}_receipt.json').read_text()) for s in range(2)]
        rows = []
        for s, receipt in enumerate(receipts):
            assert receipt['status'] == 'COMPLETE_MARS_FEATURE_SHARD' and receipt['shard'] == s
            assert receipt['review_sha256'] == sha(source / 'corpus_review.json')
            assert receipt['extraction_contract_sha256'] == review['extraction_contract_sha256']
            rows.extend(receipt['records'])
        rows.sort(key=lambda r: r['study_id']); ids = [r['study_id'] for r in rows]
        assert len(ids) == len(set(ids)) == 4407
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == review['ordered_study_ids_sha256']
        assert {p.name for p in (output / 'features').iterdir()} == {u + '.npz' for u in ids}
        for row in rows:
            assert sha(output / 'features' / (row['study_id'] + '.npz')) == row['cache_sha256']
        receipt = {'status': 'COMPLETE_4407_STUDY_FROZEN_MARS_FEATURE_CORPUS',
                   'studies': len(ids), 'contract_sha256': sha(source / 'corpus_contract.json'),
                   'review_sha256': sha(source / 'corpus_review.json'),
                   'extraction_contract_sha256': review['extraction_contract_sha256'],
                   'elapsed_seconds': time.monotonic() - started, 'shard_receipts_sha256': [
                       sha(output / f'shard{s}_receipt.json') for s in range(2)],
                   'label_values_used': False, 'classifier_training_started': False,
                   'competition_submission_enabled': False}
        (output / 'corpus_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        failure = {'status': 'FULL_CORPUS_EXECUTION_INCOMPLETE', 'error_type': type(error).__name__,
                   'error_message': str(error), 'traceback': traceback.format_exc()}
        (attempt / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        print(json.dumps({k: v for k, v in failure.items() if k != 'traceback'}), flush=True)
        raise
    finally:
        for process in processes:
            if process.poll() is None: process.terminate()
        for process in processes:
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)
        for log in logs: log.close()
        for progress in progress_streams: progress.close()
        lock.close()


if __name__ == '__main__':
    run(MODULES)
