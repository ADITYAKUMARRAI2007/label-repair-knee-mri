"""Exercise guards against successful-looking fallback and invalid predictions."""
import csv
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inference'))
from nartaa_reproduction_guard import run_source, sha, validate_submission


def rejected(call):
    try:
        call()
    except (AssertionError, TimeoutError, ValueError):
        return
    raise AssertionError('invalid execution/output was accepted')


def main():
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / 'source.py'
        log = root / 'inference.log'
        source.write_text("print('XLA unavailable: ModuleNotFoundError | device cuda')\nprint('DONE 1s')\n")
        run_source(source, log, 5, root)
        checks.append('benign diagnostic accepted')
        for message in ['[pre] example FAILED (ValueError: bad pixels)',
                        '[gpu cuda:0] study 1 example FALLBACK (ValueError: bad logits)']:
            source.write_text('print(' + repr(message) + ')\n')
            rejected(lambda: run_source(source, log, 5, root))
        checks.append('both reported failure types rejected despite zero process exit')
        source.write_text('raise SystemExit(7)\n')
        rejected(lambda: run_source(source, log, 5, root))
        checks.append('nonzero exit rejected')
        source.write_text('import time\ntime.sleep(30)\n')
        rejected(lambda: run_source(source, log, .2, root))
        checks.append('stalled process terminated at deadline')
        fields = ['StudyInstanceUID'] + ['finding' + str(i) for i in range(12)]
        rows = [['one'] + [.1] * 12, ['two'] + [.7] * 12]

        def write(name, header, values):
            with (root / name).open('w', newline='') as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(values)

        write('test.csv', fields[:1], [['one'], ['two']])
        write('sample.csv', fields, rows)
        write('submission.csv', fields, rows)
        validate = lambda: validate_submission(root / 'submission.csv', root / 'test.csv', root / 'sample.csv')
        assert validate()['rows'] == 2
        for header, values in [(fields, rows[::-1]), (fields, [rows[0], rows[0]]),
                               (fields[:-1], [r[:-1] for r in rows]),
                               (fields, [['one'] + [float('nan')] * 12, rows[1]]),
                               (fields, [['one'] + [float('inf')] * 12, rows[1]]),
                               (fields, [['one'] + [1.1] * 12, rows[1]]),
                               (fields, [rows[0] + [.2], rows[1]])]:
            write('submission.csv', header, values)
            rejected(validate)
        checks.append('wrong order, duplicate IDs, columns, NaN, infinity, range and extra cells rejected')
    out = ROOT / 'runs/nartaa_reproduction_v1'
    out.mkdir(exist_ok=True)
    report = dict(status='PASS_EXECUTION_AND_OUTPUT_GUARDS', checks=checks,
                  source_sha256=sha(ROOT / 'inference/nartaa_reproduction_guard.py'),
                  model_inference_tested=False)
    (out / 'guard_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
