"""Audit the visible-test reproduction; never infer accuracy from three rows."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inference'))
from nartaa_reproduction_guard import sha, reported_failure, validate_submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    local = ROOT / 'runs/nartaa_reproduction_v1'
    build = json.loads((local / 'build.json').read_text())
    for name, expected in {**build['sources'], **build['gates']}.items():
        assert sha(ROOT / name) == expected, name
    out = args.directory
    folder = out / 'nartaa_reproduction'
    assert not (folder / 'failure.json').exists()
    receipt = json.loads((folder / 'receipt.json').read_text())
    contract = build['contract']
    assert receipt['status'] == 'PASS_EXECUTION_ONLY' and receipt['contract'] == contract
    assert receipt['source_sha256'] == contract['source_sha256']
    assert receipt['checkpoint_sha256'] == contract['checkpoint_sha256']
    assert not receipt['accuracy_measured'] and not receipt['competition_submission_enabled']
    assert receipt['reported_failure_count'] == 0 and receipt['elapsed_seconds'] < contract['worker_seconds_cap']
    assert sha(out / 'nartaa_source/published_inference.py') == contract['source_sha256']
    assert sha(out / 'nartaa_source/nartaa_reproduction_guard.py') == sha(ROOT / 'inference/nartaa_reproduction_guard.py')
    submission = validate_submission(out / 'submission.csv', ROOT / 'data/test.csv', ROOT / 'data/sample_submission.csv')
    assert submission == receipt['submission']
    log = (folder / 'inference.log').read_text()
    assert not any(reported_failure(line) for line in log.splitlines())
    assert '[gpu cuda:0] loaded 1 arm(s)' in log and '[gpu cuda:1] loaded 1 arm(s)' in log
    assert f"[gpu] done: {submission['rows']}/{submission['rows']} studies on 2 device(s)" in log
    assert re.search(r'^DONE \d+s$', log, flags=re.M)
    report = dict(status='PASS_VISIBLE_EXECUTION_ONLY', source_sha256=contract['source_sha256'],
                  checkpoint_sha256=contract['checkpoint_sha256'], submission=submission,
                  execution_seconds=receipt['elapsed_seconds'], author_reported_public_score='0.949',
                  our_public_score=None, accuracy_measured=False, new_competition_submissions=0,
                  receipt_sha256=sha(folder / 'receipt.json'), auditor_sha256=sha(Path(__file__)),
                  limitations=contract['limitations'] + ['Visible test rows verify execution, not leaderboard accuracy'])
    (local / 'independent_audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
