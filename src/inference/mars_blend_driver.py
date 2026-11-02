"""Execute the unchanged scored baseline and fixed MARS reader before fusion."""
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
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def attached(slug, filename):
    matches = [root for root in [Path('/kaggle/input') / slug,
        Path('/kaggle/input/datasets/adityakumarrai0401') / slug] if (root / filename).is_file()]
    if len(matches) != 1:
        raise ValueError('Expected exactly one attached asset: ' + slug)
    return matches[0]


def locate_heads(inputs, expected):
    slug = 'rsna-knee-mars-frozen-spatial-classifier'
    matches = []
    for root in [inputs / slug, inputs / 'notebooks/adityakumarrai0401' / slug]:
        if root.is_dir():
            for path in [root / 'training/training_receipt.json',
                         *root.glob('mars_classifier_attempt_*/training/training_receipt.json')]:
                if path.is_file() and sha(path) == expected:
                    matches.append(path.parent)
    if len(matches) != 1:
        raise ValueError('Expected exactly one audited native trained-head output')
    return matches[0]


def run(modules):
    work = Path('/kaggle/working')
    lock = (work / 'mars_blend.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempt = Path(tempfile.mkdtemp(prefix='mars_blend_attempt_', dir=work))
    (work / 'mars_latest_blend_attempt.json').write_text(json.dumps({'attempt_directory': attempt.name}) + '\n')
    source = attempt / 'source'; source.mkdir()
    submission = work / 'submission.csv'
    started = time.monotonic()
    owns_submission = False
    try:
        assert not submission.exists(), 'Existing submission requires review before another run'
        for name, value in modules.items():
            relative = Path(name)
            assert not relative.is_absolute() and '..' not in relative.parts
            destination = source / relative; destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(value)
        contract = json.loads(modules['blend_execution_contract.json'])
        for name, expected in contract['module_sha256'].items():
            assert sha(source / name) == expected, name
        assert sys.version_info[:2] == (3, 13)
        import torch
        assert str(torch.__version__) == contract['torch_version'] and torch.cuda.device_count() == 2
        inputs = Path('/kaggle/input')
        data = next(root for root in [inputs / 'rsna-knee-abnormality-detection',
            inputs / 'competitions/rsna-knee-abnormality-detection'] if (root / 'test.csv').is_file())
        trained = locate_heads(inputs, contract['training_receipt_sha256'])
        replay = trained / 'heldout_replay.json'
        assert sha(replay) == contract['heldout_replay_sha256']
        assert json.loads(replay.read_text())['status'] == 'PASS_ALL_4407_RELOADED_HELDOUT_HEAD_PREDICTIONS'
        assert sha(trained.parent / 'execution_receipt.json') == contract['classifier_execution_receipt_sha256']
        for name, expected in contract['head_checkpoint_sha256'].items():
            assert sha(trained / name) == expected, name
        sys.path.insert(0, str(source / 'fusion'))
        from nartaa_reproduction_guard import read_csv, run_source, validate_submission
        from fixed_reader_blend import blend
        _, tests = read_csv(data / 'test.csv')
        cap = contract['visible_seconds_cap'] if len(tests) <= 3 else contract['hidden_seconds_cap']

        def remaining():
            seconds = cap - (time.monotonic() - started)
            if seconds <= 0:
                raise TimeoutError('Reviewed combined inference budget exhausted')
            return seconds

        baseline_cap = contract['baseline_visible_seconds_cap'] if len(tests) <= 3 else contract['baseline_hidden_seconds_cap']
        # The scored baseline owns nested worker groups. Give its own bounded
        # supervisor time to clean them up before the outer timeout can fire.
        assert remaining() > baseline_cap + 120
        owns_submission = True
        baseline = run_source(source / 'baseline.py', attempt / 'baseline.log', baseline_cap + 60, work)
        baseline_details = validate_submission(submission, data / 'test.csv', data / 'sample_submission.csv')
        submission.replace(attempt / 'baseline.csv')
        assets = attached('rsna-knee-mars-model-assets-v2', 'MARS.pt')
        decoder = attached('rsna-knee-decoder-cp313', 'decoder_identity.json')
        for name, expected in contract['asset_sha256'].items():
            assert sha(assets / name) == expected, name
        wheels = []
        for name, expected in contract['decoder_wheels'].items():
            assert sha(decoder / name) == expected, name
            wheels.append(str(decoder / name))
        with tempfile.TemporaryDirectory(prefix='mars_serving_dependencies_', dir='/tmp') as temporary:
            dependencies = Path(temporary)
            with (attempt / 'bootstrap.log').open('w') as log:
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                    '--target', str(dependencies), str(assets / contract['monai_wheel']), *wheels],
                    check=True, stdout=log, stderr=subprocess.STDOUT, timeout=min(180, remaining()))
                environment = dict(os.environ, MARS_PILOT_DEPENDENCIES=str(dependencies),
                    PYTHONPATH=os.pathsep.join([str(dependencies), str(source / 'mars')]))
                subprocess.run([sys.executable, str(source / 'mars/check_decoders.py')], env=environment,
                    check=True, stdout=log, stderr=subprocess.STDOUT, timeout=min(60, remaining()))
            assert 'DECODER_ABI_PASS cp313' in (attempt / 'bootstrap.log').read_text()
            runtime = {'data': str(data), 'assets': str(assets), 'trained': str(trained),
                       'out': str(attempt / 'mars')}
            (source / 'mars/test_runtime.json').write_text(json.dumps(runtime))
            runner = source / 'run_mars.py'
            runner.write_text('import runpy,sys\n' +
                f'sys.path[:0]=[{str(dependencies)!r},{str(source / "mars")!r}]\n' +
                f'runpy.run_path({str(source / "mars/mars_test_inference.py")!r},run_name="__main__")\n')
            candidate = run_source(runner, attempt / 'mars.log', remaining(), work)
        inference_path = attempt / 'mars/inference_receipt.json'
        inference = json.loads(inference_path.read_text())
        assert inference['status'] == 'COMPLETE_NATIVE_UNSEEN_TEST_FIVE_HEAD_MARS_READER'
        assert inference['training_receipt_sha256'] == contract['training_receipt_sha256']
        assert inference['extraction_contract_sha256'] == contract['extraction_contract_sha256']
        assert inference['candidate_sha256'] == sha(attempt / 'mars/candidate.csv')
        assert inference['studies'] == len(tests) and not inference['fallback_predictions_used']
        pending = attempt / 'blend.pending.csv'
        weights = {target: .1 for target in json.loads(modules['mars/classifier_protocol.json'])['target_order']}
        details = blend(attempt / 'baseline.csv', attempt / 'mars/candidate.csv', pending,
                        weights, data / 'test.csv', data / 'sample_submission.csv')
        assert remaining() > 0
        pending.replace(submission)
        receipt = {'status': 'COMPLETE_FIXED_N02_MARS_BLEND_EXECUTION_ONLY',
            'execution_contract_sha256': sha(source / 'blend_execution_contract.json'),
            'baseline': baseline, 'baseline_output': baseline_details,
            'candidate': candidate, 'candidate_receipt_sha256': sha(inference_path),
            'output': details, 'elapsed_seconds': time.monotonic() - started, 'runtime_cap': cap,
            'fixed_rank_blend': {'baseline': .9, 'candidate': .1},
            'weight_search_performed': False, 'clinical_accuracy_measured': False,
            'competition_submission_enabled': False}
        (attempt / 'execution_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
    except BaseException as error:
        if owns_submission and submission.exists():
            submission.replace(attempt / 'rejected_submission.csv')
        (attempt / 'failure.json').write_text(json.dumps({'status': 'COMBINED_INFERENCE_INCOMPLETE',
            'error_type': type(error).__name__, 'error_message': str(error),
            'traceback': traceback.format_exc()}, indent=2) + '\n')
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    run(MODULES)
