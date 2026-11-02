"""Train the fixed classifier from a verified native Kaggle feature corpus."""
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
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def locate_corpus(inputs, expected_receipt):
    roots = [inputs / 'rsna-knee-mars-full-frozen-corpus',
             inputs / 'notebooks/adityakumarrai0401/rsna-knee-mars-full-frozen-corpus']
    matches = []
    for root in roots:
        if not root.is_dir():
            continue
        candidates = [root / 'corpus_receipt.json', *root.glob(
            'mars_corpus_attempt_*/corpus/corpus_receipt.json')]
        matches.extend(p for p in candidates if p.is_file() and sha(p) == expected_receipt)
    if len(matches) != 1:
        raise ValueError('Expected exactly one source-matched complete native feature corpus')
    return matches[0].parent


def run(modules):
    work = Path('/kaggle/working')
    lock = (work / 'mars_classifier.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = Path(tempfile.mkdtemp(prefix='mars_classifier_attempt_', dir=work))
    (work / 'mars_latest_classifier_attempt.json').write_text(json.dumps({
        'attempt_directory': attempt.name}) + '\n')
    # Public label tables and reports are reconstructed in temporary storage;
    # only training outputs and execution logs are persisted by this notebook.
    workspace = Path(tempfile.mkdtemp(prefix='mars_classifier_workspace_', dir='/tmp'))
    process = None
    started = time.monotonic()
    try:
        for relative, source in modules.items():
            name = Path(relative)
            if name.is_absolute() or '..' in name.parts:
                raise ValueError('Unsafe packaged source path')
            destination = workspace / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(source)
        contract = json.loads(modules['classifier_contract.json'])
        approval = json.loads(modules['classifier_resource_approval.json'])
        maximum = contract['maximum_parent_seconds']

        def remaining():
            value = maximum - (time.monotonic() - started)
            if value <= 0:
                raise TimeoutError('Reviewed classifier execution budget exhausted')
            return value

        assert sys.version_info[:2] == (3, 13)
        import torch
        assert str(torch.__version__) == contract['torch_version']
        assert torch.cuda.is_available()
        for name, expected in contract['module_sha256'].items():
            assert sha(workspace / name) == expected, name
        assert sha(workspace / 'classifier_protocol.json') == approval['classifier_protocol_sha256']
        inputs = Path('/kaggle/input')
        corpus = locate_corpus(inputs, approval['corpus_receipt_sha256'])
        receipt = json.loads((corpus / 'corpus_receipt.json').read_text())
        assert receipt['status'] == 'COMPLETE_4407_STUDY_FROZEN_MARS_FEATURE_CORPUS'
        assert receipt['studies'] == approval['complete_study_coverage'] == 4407
        assert receipt['extraction_contract_sha256'] == approval['extraction_contract_sha256']
        records = []
        for shard in range(2):
            path = corpus / f'shard{shard}_receipt.json'
            assert sha(path) == receipt['shard_receipts_sha256'][shard]
            item = json.loads(path.read_text())
            assert item['status'] == 'COMPLETE_MARS_FEATURE_SHARD' and item['shard'] == shard
            assert item['extraction_contract_sha256'] == receipt['extraction_contract_sha256']
            records.extend(item['records'])
        ids = sorted(r['study_id'] for r in records)
        assert len(ids) == len(set(ids)) == 4407
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == approval['ordered_study_ids_sha256']
        assert {p.name for p in (corpus / 'features').iterdir()} == {uid + '.npz' for uid in ids}
        for row in records:
            assert sha(corpus / 'features' / (row['study_id'] + '.npz')) == row['cache_sha256']
        print(json.dumps({'stage': 'native_complete_corpus_verified', 'studies': len(ids)}), flush=True)
        train = next(p for p in [inputs / 'rsna-knee-abnormality-detection/train.csv',
                                inputs / 'competitions/rsna-knee-abnormality-detection/train.csv']
                     if p.is_file())
        (workspace / 'data').mkdir()
        (workspace / 'data/train.csv').symlink_to(train)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([
            str(workspace / 'training'), str(workspace / 'inference'),
            os.environ.get('PYTHONPATH', '')]))
        with (attempt / 'input_reconstruction.log').open('w') as log:
            subprocess.run([sys.executable, str(workspace / 'training/mars_runtime_inputs.py'),
                            '--workspace', str(workspace), '--train', str(train), '--inputs', str(inputs)],
                           env=env, check=True, stdout=log, stderr=subprocess.STDOUT,
                           timeout=min(600, remaining()))
        print(json.dumps({'stage': 'fixed_five_fold_training_start', 'epochs_per_fold': 30}), flush=True)
        output = attempt / 'training'
        with (attempt / 'training.log').open('w') as log:
            process = subprocess.Popen([
                sys.executable, str(workspace / 'training/mars_classifier_train.py'),
                '--workspace', str(workspace), '--features', str(corpus / 'features'),
                '--protocol', str(workspace / 'classifier_protocol.json'),
                '--resource-approval', str(workspace / 'classifier_resource_approval.json'),
                '--out', str(output), '--device', 'cuda:0'], env=env,
                stdout=log, stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),))
            with (attempt / 'training.log').open() as progress:
                while process.poll() is None:
                    remaining()
                    text = progress.read()
                    if text:
                        print(text, end='', flush=True)
                    time.sleep(2)
                text = progress.read()
                if text:
                    print(text, end='', flush=True)
            returncode = process.returncode
            if returncode:
                raise subprocess.CalledProcessError(returncode, process.args)
        trained = json.loads((output / 'training_receipt.json').read_text())
        assert trained['status'] == 'FIVE_FROZEN_MARS_HEADS_AND_COMPLETE_OOF_TRAINED'
        assert trained['protocol_sha256'] == approval['classifier_protocol_sha256']
        assert trained['studies'] == 4407 and not trained['competition_submission_enabled']
        with (attempt / 'heldout_replay.log').open('w') as log:
            subprocess.run([sys.executable, str(workspace / 'scripts/replay_mars_heldout_heads.py'),
                            '--workspace', str(workspace), '--features', str(corpus / 'features'),
                            '--training-output', str(output),
                            '--output', str(output / 'heldout_replay.json')],
                           env=env, check=True, stdout=log, stderr=subprocess.STDOUT,
                           pass_fds=(lock.fileno(),), timeout=min(600, remaining()))
        replay = json.loads((output / 'heldout_replay.json').read_text())
        assert replay['status'] == 'PASS_ALL_4407_RELOADED_HELDOUT_HEAD_PREDICTIONS'
        print(json.dumps({'stage': 'heldout_head_replay_complete',
                          'studies': replay['studies'],
                          'maximum_absolute_error': replay['maximum_absolute_error']}), flush=True)
        result = {'status': 'BOUNDED_COMPLETE_FIVE_FOLD_CLASSIFIER_EXECUTION',
                  'contract_sha256': sha(workspace / 'classifier_contract.json'),
                  'resource_approval_sha256': sha(workspace / 'classifier_resource_approval.json'),
                  'corpus_receipt_sha256': sha(corpus / 'corpus_receipt.json'),
                  'training_receipt_sha256': sha(output / 'training_receipt.json'),
                  'heldout_replay_sha256': sha(output / 'heldout_replay.json'),
                  'elapsed_seconds': time.monotonic() - started,
                  'competition_submission_enabled': False, 'automatic_blend_search': False}
        (attempt / 'execution_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
    except BaseException as error:
        failure = {'status': 'CLASSIFIER_EXECUTION_INCOMPLETE', 'error_type': type(error).__name__,
                   'error_message': str(error), 'traceback': traceback.format_exc()}
        (attempt / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        for name in ['input_reconstruction.log', 'training.log', 'heldout_replay.log']:
            path = attempt / name
            if path.exists():
                print('\n'.join(path.read_text(errors='replace').splitlines()[-60:]), flush=True)
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait(timeout=10)
        lock.close()


if __name__ == '__main__':
    run(MODULES)
