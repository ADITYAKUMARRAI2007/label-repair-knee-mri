"""Run a fixed, paired input ablation; emit diagnostics, never a submission."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from nartaa_reproduction_guard import run_source, sha, validate_submission


def prepare_cohort(data, cohort, studies):
    """Expose selected training images through the public inference interface."""
    assert len(studies) == len(set(studies)) == 78
    with (data / 'train.csv').open() as handle:
        available = {row['StudyInstanceUID'] for row in csv.DictReader(handle)}
    assert set(studies) <= available
    cohort.mkdir(exist_ok=False)
    with (cohort / 'test.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['StudyInstanceUID'])
        writer.writerows([[uid] for uid in studies])
    with (data / 'sample_submission.csv').open() as handle:
        columns = next(csv.reader(handle))
    assert len(columns) == 13 and columns[0] == 'StudyInstanceUID'
    with (cohort / 'sample_submission.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows([[uid] + [0.5] * 12 for uid in studies])
    with (data / 'train_series.csv').open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = [row for row in reader if row['StudyInstanceUID'] in set(studies)]
    assert set(row['StudyInstanceUID'] for row in rows) == set(studies)
    with (cohort / 'test_series.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    series = next(path for path in [data / 'train_series', data / 'train_images'] if path.is_dir())
    (cohort / 'test_series').symlink_to(series, target_is_directory=True)


def read_input_audit(path, studies):
    prefix = 'PAIRED_INPUT_AUDIT '
    rows = [json.loads(line[len(prefix):]) for line in path.read_text().splitlines()
            if line.startswith(prefix)]
    assert len(rows) == len(studies)
    assert len({row['study'] for row in rows}) == len(studies)
    assert {row['study'] for row in rows} == set(studies)
    return {row['study']: row for row in rows}


def main():
    root = Path(__file__).resolve().parent
    contract = json.loads((root / 'pair_contract.json').read_text())
    work = Path('/kaggle/working')
    out = work / 'coherent_anchor_pair'
    out.mkdir(exist_ok=False)
    submission = work / 'submission.csv'
    assert not submission.exists()
    start = time.monotonic()

    def remaining():
        seconds = contract['runtime_cap_seconds'] - (time.monotonic() - start)
        if seconds <= 0:
            raise TimeoutError('Paired anchor budget exhausted')
        return seconds

    try:
        for name, expected in contract['module_sha256'].items():
            assert sha(root / name) == expected, name
        import torch
        import timm
        assert torch.cuda.device_count() == 2
        assert str(torch.__version__) == contract['torch_version']
        assert timm.__version__ == contract['timm_version']
        assert f'cp{sys.version_info.major}{sys.version_info.minor}' == 'cp313'
        weights = next(Path('/kaggle/input') / suffix / contract['checkpoint_filename']
                       for suffix in ['rsna-knee-publication-swa-weights-20261007',
                                      'datasets/nartaa/rsna-knee-publication-swa-weights-20261007']
                       if (Path('/kaggle/input') / suffix / contract['checkpoint_filename']).is_file())
        assert sha(weights) == contract['checkpoint_sha256']
        data = next(path for path in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
                                     Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
                    if (path / 'train.csv').is_file())
        cohort = Path(contract['cohort_path'])
        prepare_cohort(data, cohort, contract['studies'])
        decoder_assets = next(path for path in [Path('/kaggle/input/rsna-knee-decoder-cp313'),
                                                Path('/kaggle/input/datasets/adityakumarrai0401/rsna-knee-decoder-cp313')]
                              if (path / 'decoder_identity.json').is_file())
        wheels = []
        for name, expected in contract['decoder_wheels'].items():
            path = decoder_assets / name
            assert sha(path) == expected, name
            wheels.append(str(path))
        target = work / 'reader_decoder_env'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps',
                        '--target', str(target), *wheels], check=True, timeout=min(180, remaining()))
        # Both processes inherit exactly the same decoder environment.
        os.environ['PYTHONPATH'] = os.pathsep.join([str(target), str(root), os.environ.get('PYTHONPATH', '')])
        decoder = run_source(root / 'check_decoders.py', out / 'decoder.log', min(60, remaining()), work)
        assert 'DECODER_ABI_PASS cp313' in (out / 'decoder.log').read_text()
        arms = {}
        inputs = {}
        for arm in ['control', 'candidate']:
            assert not submission.exists()
            result = run_source(root / f'{arm}.py', out / f'{arm}.log', remaining(), work)
            record = validate_submission(submission, cohort / 'test.csv', cohort / 'sample_submission.csv')
            inputs[arm] = read_input_audit(out / f'{arm}.log', contract['studies'])
            submission.replace(out / f'{arm}.csv')
            arms[arm] = {'execution': result, 'output': record,
                         'source_sha256': sha(root / f'{arm}.py')}
        for uid in contract['studies']:
            a, b = inputs['control'][uid], inputs['candidate'][uid]
            for key in ['volume_sha256', 'mask', 'shape']:
                assert a[key] == b[key], f'Paired decoded input differs: {key}'
            assert b['windows'] == b['coherent_windows'] > 0
            assert a['windows'] >= b['windows']
            assert a['coherent_windows'] == b['coherent_windows']
        (out / 'decoded_inputs.json').write_text(json.dumps(inputs, indent=2) + '\n')
        assert remaining() > 0 and not submission.exists()
        receipt = {'status': 'PAIRED_ANCHOR_PREDICTIONS_READY_NOT_SCORED', 'arms': arms,
                   'contract': contract, 'decoded_inputs_sha256': sha(out / 'decoded_inputs.json'),
                   'decoder': decoder, 'elapsed_seconds': time.monotonic() - start,
                   'accuracy_measured': False, 'competition_submission_enabled': False}
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({'status': receipt['status'], 'rows': len(contract['studies'])}), flush=True)
    except BaseException as error:
        (out / 'failure.json').write_text(json.dumps({'status': 'REJECTED', 'error_type': type(error).__name__}) + '\n')
        raise
    finally:
        if submission.exists():
            submission.replace(out / 'rejected_submission.csv')


if __name__ == '__main__':
    main()
