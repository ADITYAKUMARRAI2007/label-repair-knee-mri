"""Run the frozen public tri-reader recipe with verified components and outputs."""
import json
from pathlib import Path
import time

from nartaa_reproduction_guard import sha, read_csv, run_source, validate_submission
from fixed_reader_blend import blend


def main():
    root = Path(__file__).resolve().parent
    contract = json.loads((root / 'tri_reader_contract.json').read_text())
    work = Path('/kaggle/working')
    out = work / 'published_tri_reader'
    out.mkdir(exist_ok=False)
    submission = work / 'submission.csv'
    assert not submission.exists(), 'Stale submission exists before inference'
    start = time.monotonic()
    try:
        for name, expected in contract['module_sha256'].items():
            assert sha(root / name) == expected, name
        data = next(p for p in [Path('/kaggle/input/rsna-knee-abnormality-detection'),
                               Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')]
                    if (p / 'test.csv').is_file())
        _, studies = read_csv(data / 'test.csv')
        cap = contract['visible_seconds_cap'] if len(studies) <= 3 else contract['worker_seconds_cap']

        def remaining():
            seconds = cap - (time.monotonic() - start)
            if seconds <= 0:
                raise TimeoutError('Tri-reader runtime budget exhausted')
            return seconds

        # Direct call preserves the baseline's own subprocess timeouts and cleanup.
        # Wrapping it in another run_source would create detached grandchildren.
        import fixed_reader_blend
        fixed_reader_blend.main()
        validate_submission(submission, data / 'test.csv', data / 'sample_submission.csv')
        baseline_receipt = json.loads((work / 'fixed_reader_blend/receipt.json').read_text())
        assert baseline_receipt['status'] == 'PASS_EXECUTION_ONLY'
        submission.replace(out / 'baseline.csv')
        crop = run_source(root / 'crop224.py', out / 'crop224.log', remaining(), work)
        crop_record = validate_submission(submission, data / 'test.csv', data / 'sample_submission.csv')
        assert crop['reported_failure_count'] == 0
        submission.replace(out / 'crop224.csv')
        anchor = work / 'fixed_reader_blend/anchor.csv'
        reader = work / 'fixed_reader_blend/reader.csv'
        crop_weights = {label: contract['crop_weight'] for label in contract['reader_weights']}
        blend(anchor, out / 'crop224.csv', out / 'multiscale_anchor.csv', crop_weights,
              data / 'test.csv', data / 'sample_submission.csv')
        pending = out / 'candidate.pending.csv'
        result = blend(out / 'multiscale_anchor.csv', reader, pending, contract['reader_weights'],
                       data / 'test.csv', data / 'sample_submission.csv')
        assert remaining() > 0
        pending.replace(submission)
        receipt = {
            'status': 'PASS_EXECUTION_ONLY', 'contract': contract,
            'baseline_receipt_sha256': sha(work / 'fixed_reader_blend/receipt.json'),
            'components': {name: sha(path) for name, path in {
                'anchor': anchor, 'reader': reader, 'crop224': out / 'crop224.csv',
                'baseline': out / 'baseline.csv', 'multiscale_anchor': out / 'multiscale_anchor.csv'}.items()},
            'crop_execution': crop, 'crop_output': crop_record,
            'submission': result, 'elapsed_seconds': time.monotonic() - start,
            'runtime_cap': cap, 'accuracy_measured': False,
            'competition_submission_enabled': False,
        }
        (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print('TRI_READER_COMPLETE ' + json.dumps(result), flush=True)
    except BaseException as error:
        if submission.exists():
            submission.replace(out / 'rejected_submission.csv')
        (out / 'failure.json').write_text(json.dumps({
            'status': 'REJECTED_NO_ELIGIBLE_SUBMISSION', 'error_type': type(error).__name__}, indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
