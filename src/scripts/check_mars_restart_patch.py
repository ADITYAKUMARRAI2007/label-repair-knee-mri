"""Reproduce preserved-directory restart and verify failure visibility locally."""
import ast
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inference'))
from mars_pilot_driver import prepare_attempt, report_failure, run


def main():
    code = ROOT / 'public_kernels/mars_resource_pilot_v3/pilot.py'
    modules = ast.literal_eval(ast.parse(code.read_text()).body[0].value)
    assert sys.version_info[:2] != (3, 13), 'Run this rejection check on the local runtime'
    with tempfile.TemporaryDirectory(prefix='mars_restart_check_') as directory:
        work = Path(directory)
        prior = work / 'mars_pilot_source'; prior.mkdir()
        (prior / 'old_source.py').write_text('preserved evidence\n')
        out = work / 'mars_resource_pilot'; out.mkdir()
        (out / 'worker.log').write_text('prior worker failure\n')
        lock, attempt, _ = prepare_attempt(work)
        before = len(list(work.glob('mars_resource_attempt_*')))
        try:
            prepare_attempt(work)
        except RuntimeError as error:
            assert 'already active' in str(error)
        else:
            raise AssertionError('Concurrent run was not rejected')
        assert len(list(work.glob('mars_resource_attempt_*'))) == before
        lock.close()
        messages = io.StringIO()
        for _ in range(2):
            with contextlib.redirect_stdout(messages), contextlib.redirect_stderr(messages):
                try:
                    run(modules, work)
                except AssertionError as error:
                    assert str(error) == 'Pinned Kaggle Python runtime changed'
                else:
                    raise AssertionError('Local runtime incorrectly accepted')
        assert 'FileExistsError' not in messages.getvalue()
        failures = list(work.glob('mars_resource_attempt_*/mars_resource_bootstrap_failure.json'))
        assert len(failures) == 2
        assert all(json.loads(p.read_text())['stage'] == 'runtime_and_assets' for p in failures)
        assert (prior / 'old_source.py').read_text() == 'preserved evidence\n'
        assert (out / 'worker.log').read_text() == 'prior worker failure\n'
        # Test the real diagnostic renderer, using an actual child traceback.
        worker_log = attempt / 'mars_resource_worker.log'
        with worker_log.open('w') as log:
            try:
                subprocess.run([sys.executable, '-c',
                                "raise ValueError('underlying worker error is visible')"],
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            except subprocess.CalledProcessError as error:
                captured = io.StringIO()
                with contextlib.redirect_stderr(captured):
                    report_failure(attempt, 'resource_worker', error)
            else:
                raise AssertionError('Fixture child unexpectedly succeeded')
        assert 'ValueError: underlying worker error is visible' in captured.getvalue()
        assert 'CalledProcessError' in captured.getvalue()
    result = {'status': 'PASS_RESTART_AND_WORKER_DIAGNOSTIC_REPRODUCTION',
              'old_sources_and_outputs_preserved': True,
              'two_repeated_attempts_have_distinct_directories': True,
              'concurrent_attempt_rejected_before_directory_creation': True,
              'underlying_child_traceback_visible': True,
              'cuda_execution_verified': False, 'accuracy_measured': False}
    output = ROOT / 'research/foundation_refresh_20261010/MARS/restart_patch_check.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
